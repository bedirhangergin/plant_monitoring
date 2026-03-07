"""
capabilities/base.py
====================
Shared base class and image validation used by every capability module.

All capability classes inherit from BaseCapability, which provides:
  - validate_image()  — consistent pre-flight check before any processing
  - _SUPPORTED_EXT    — canonical set of accepted image extensions

Result base class
-----------------
All capability-specific result dataclasses inherit from CapabilityResult,
giving every result a consistent set of fields:
  - image_path   : str       — path to the analysed image
  - model_name   : str       — identifier of the model that produced the result
  - timestamp    : str       — ISO-8601 creation time
  - duration_ms  : float     — inference wall-clock time in milliseconds
  - metadata     : dict      — any extra model-specific data

This means every result can be logged, compared, and serialised in the same
way regardless of which capability produced it.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional
import json


# ---------------------------------------------------------------------------
# Supported image extensions (shared across all capabilities)
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Base result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CapabilityResult:
    """
    Base dataclass inherited by every capability-specific result.

    Subclasses add their own domain-specific fields on top of these.

    Attributes
    ----------
    image_path : str
        Path to the image that was analysed.
    model_name : str
        Short identifier of the model/method used (e.g. 'clip_vit_b32').
    timestamp : str
        ISO-8601 string set automatically at result creation time.
    duration_ms : float
        Wall-clock inference time in milliseconds.  Set by the capability
        class after timing the inference call.
    metadata : dict
        Optional bag of any extra model-specific data that doesn't fit
        into a typed field.  Not part of the primary API but useful for
        debugging.
    """

    image_path: str
    model_name: str
    duration_ms: float = 0.0
    timestamp: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """
        Return a JSON-serialisable dictionary of all result fields.
        Subclasses should override and call super().to_dict() to merge fields.
        """
        return {
            "image_path": self.image_path,
            "model_name": self.model_name,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 2),
            "metadata": self.metadata,
        }

    def to_json(self, indent: int = 2) -> str:
        """Return result as a pretty-printed JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"model={self.model_name} "
            f"image={os.path.basename(self.image_path)} "
            f"({self.duration_ms:.0f}ms)>"
        )


# ---------------------------------------------------------------------------
# Base capability class
# ---------------------------------------------------------------------------

class BaseCapability(ABC):
    """
    Abstract base class for all plant vision capability classes.

    Every capability subclass must implement:
        - model_name  (property) — short identifier string
        - run()                  — primary analysis method

    Every capability subclass inherits:
        - validate_image()       — pre-flight file + extension check
        - _time_call()           — context manager for timing inference

    Design rules for subclasses
    ---------------------------
    1. Models are loaded lazily — only when run() is first called.
    2. run() always returns a typed CapabilityResult subclass instance.
    3. run() never raises on a valid image — errors become result fields.
    4. Each capability is independently testable with no other loaded.
    5. No capability imports from sprout_detection or other capabilities.
    """

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def model_name(self) -> str:
        """
        Short, lowercase identifier for this capability's model.
        Examples: 'clip_vit_b32', 'grounding_dino_base', 'sam2_base',
                  'depth_anything_v2_small', 'colour_hsv'
        """

    @abstractmethod
    def run(self, image_path: str, **kwargs) -> CapabilityResult:
        """
        Run this capability on a single image.

        Parameters
        ----------
        image_path : str
            Path to a JPEG, PNG, WebP, or BMP image file.
        **kwargs
            Capability-specific parameters (prompts, thresholds, etc.)

        Returns
        -------
        CapabilityResult (or subclass)
            Always returns a result — never raises on a valid image path.
        """

    # ------------------------------------------------------------------ #
    # Shared validation
    # ------------------------------------------------------------------ #

    def validate_image(self, image_path: str) -> None:
        """
        Raise a clear error if the image file is invalid before any
        expensive model loading or inference occurs.

        Parameters
        ----------
        image_path : str
            Path to validate.

        Raises
        ------
        FileNotFoundError
            File does not exist.
        ValueError
            File extension is not a supported image format.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"[{self.model_name}] Image not found: '{image_path}'"
            )
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"[{self.model_name}] Unsupported format '{ext}'. "
                f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
            )

    # ------------------------------------------------------------------ #
    # Timing helper
    # ------------------------------------------------------------------ #

    class _Timer:
        """Simple context manager that records elapsed wall-clock time."""
        def __enter__(self):
            self._start = time.perf_counter()
            return self
        def __exit__(self, *_):
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000

    def _timer(self) -> "_Timer":
        """Return a timing context manager. Use as: with self._timer() as t:"""
        return self._Timer()

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} model='{self.model_name}'>"
