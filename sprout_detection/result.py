"""
sprout_detection/result.py
==========================
Shared data model for sprout detection results.

Every detector layer (HSV, CLIP, Gemini) returns a SproutResult object.
This guarantees a consistent interface throughout the cascade pipeline and
makes downstream consumers (loggers, visualisers, tests) layer-agnostic.

Usage
-----
    from sprout_detection.result import SproutResult

    result = SproutResult(
        sprout_detected=True,
        confidence=0.87,
        method="hsv_masking",
        reasoning="Green ratio 0.045 > threshold 0.01.",
        image_path="photo.jpg",
    )
    print(result)            # 🌱 SproutResult | detected=True | confidence=0.87 | ...
    print(result.to_dict())  # JSON-serialisable dict
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class SproutResult:
    """
    Unified result object returned by every sprout detection layer.

    Attributes
    ----------
    sprout_detected : bool
        True if the detector concluded a sprout is present.
    confidence : float
        Score in [0.0, 1.0].  Values >= config threshold cause the cascade
        to stop and return this result immediately.
    method : str
        Identifier for the detector that produced this result.
        One of: 'hsv_masking', 'clip_zero_shot', 'gemini_flash_api'.
    reasoning : str
        Human-readable explanation of the decision, useful for debugging
        and for the Gemini layer which returns a natural-language sentence.
    image_path : str
        Absolute or relative path to the analysed image.
    timestamp : str
        ISO-8601 datetime string set automatically at creation time.
    escalated : bool
        True when a previous layer triggered escalation to this layer
        (i.e. this is not the first layer that ran on this image).
    green_ratio : float or None
        Layer 1 specific.  Fraction of image pixels classified as green.
    clip_scores : dict or None
        Layer 2 specific.  Maps each CLIP text prompt to its softmax score.
    hsv_mask : any or None
        Layer 1 specific.  Raw OpenCV binary mask array (not serialised).
        Stored here so callers can access it for visualisation without a
        second call to the detector.
    """

    sprout_detected: bool
    confidence: float
    method: str
    reasoning: str
    image_path: str

    # Auto-populated fields
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    # Cascade metadata
    escalated: bool = False

    # Layer-specific extras (optional, serialised where JSON-compatible)
    green_ratio: Optional[float] = None      # Layer 1
    clip_scores: Optional[dict] = None       # Layer 2
    hsv_mask: Optional[object] = field(      # Layer 1 — excluded from serialisation
        default=None, repr=False
    )

    # ------------------------------------------------------------------ #
    # Serialisation
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dictionary of all result fields.

        Note: hsv_mask is intentionally excluded — it is a numpy array
        and is not needed for logging or API responses.
        """
        return {
            "sprout_detected": self.sprout_detected,
            "confidence": round(self.confidence, 4),
            "method": self.method,
            "reasoning": self.reasoning,
            "image_path": self.image_path,
            "timestamp": self.timestamp,
            "escalated": self.escalated,
            "green_ratio": (
                round(self.green_ratio, 6) if self.green_ratio is not None else None
            ),
            "clip_scores": (
                {k: round(v, 4) for k, v in self.clip_scores.items()}
                if self.clip_scores is not None
                else None
            ),
        }

    def to_json(self, indent: int = 2) -> str:
        """Return the result serialised as a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    # ------------------------------------------------------------------ #
    # Display helpers
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        icon = "🌱" if self.sprout_detected else "🪨"
        esc = " [escalated]" if self.escalated else ""
        return (
            f"{icon} SproutResult | detected={self.sprout_detected} | "
            f"confidence={self.confidence:.2f} | method={self.method}{esc}"
        )

    def __str__(self) -> str:
        return self.__repr__()

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #

    @property
    def passed_threshold(self) -> bool:
        """True if confidence reached the default cascade threshold (0.60)."""
        return self.confidence >= 0.60

    @property
    def summary_line(self) -> str:
        """Single-line summary suitable for log output."""
        icon = "🌱" if self.sprout_detected else "🪨"
        return (
            f"[{self.timestamp}] {icon} {self.method:<20} "
            f"detected={str(self.sprout_detected):<5}  "
            f"confidence={self.confidence:.3f}  "
            f"escalated={self.escalated}"
        )
