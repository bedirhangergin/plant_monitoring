"""
capabilities/detection/base_detection_result.py
================================================
Shared result types for all object detection capabilities.

All three detectors (Grounding DINO, OWL-ViT, YOLO) return the same
DetectionResult type, making downstream code detector-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from capabilities.base import CapabilityResult


# Type alias: bounding box as [x1, y1, x2, y2] in pixel coordinates
BBox = Tuple[float, float, float, float]


@dataclass
class Detection:
    """
    A single detected object.

    Attributes
    ----------
    label : str
        Text label / class name of the detected object.
    confidence : float
        Detection confidence score (0.0–1.0).
    bbox : tuple
        Bounding box as (x1, y1, x2, y2) in pixel coordinates.
        x1, y1 = top-left corner.  x2, y2 = bottom-right corner.
    bbox_normalised : tuple or None
        Same bounding box normalised to [0.0, 1.0] relative to image size.
    area_px : float
        Area of the bounding box in pixels.
    area_ratio : float
        Fraction of total image area occupied by this detection.
    """

    label: str
    confidence: float
    bbox: BBox                               # (x1, y1, x2, y2) pixels
    bbox_normalised: Optional[BBox] = None   # (x1, y1, x2, y2) 0–1
    area_px: float = 0.0
    area_ratio: float = 0.0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "bbox": list(self.bbox),
            "bbox_normalised": list(self.bbox_normalised) if self.bbox_normalised else None,
            "area_px": round(self.area_px, 1),
            "area_ratio": round(self.area_ratio, 4),
        }

    def __repr__(self) -> str:
        x1, y1, x2, y2 = [round(v, 1) for v in self.bbox]
        return (
            f"<Detection label='{self.label}' conf={self.confidence:.2f} "
            f"bbox=({x1},{y1},{x2},{y2})>"
        )

    @property
    def centre(self) -> Tuple[float, float]:
        """Centre point of the bounding box as (cx, cy)."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        return self.bbox[3] - self.bbox[1]


@dataclass
class DetectionResult(CapabilityResult):
    """
    Result from any object detection capability.

    Attributes
    ----------
    detections : list of Detection
        All detected objects, sorted by confidence (highest first).
    prompts : list of str
        Text prompts or class names that were searched for.
    image_width : int
        Image width in pixels.
    image_height : int
        Image height in pixels.
    """

    detections: List[Detection] = field(default_factory=list)
    prompts: List[str] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    @property
    def count(self) -> int:
        """Number of detected objects."""
        return len(self.detections)

    @property
    def best(self) -> Optional[Detection]:
        """The detection with the highest confidence, or None if empty."""
        return self.detections[0] if self.detections else None

    @property
    def found(self) -> bool:
        """True if at least one object was detected."""
        return len(self.detections) > 0

    def filter_by_label(self, label: str) -> List[Detection]:
        """Return all detections matching a specific label."""
        return [d for d in self.detections if d.label == label]

    def filter_by_confidence(self, min_confidence: float) -> List[Detection]:
        """Return detections above a minimum confidence threshold."""
        return [d for d in self.detections if d.confidence >= min_confidence]

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "count": self.count,
            "found": self.found,
            "prompts": self.prompts,
            "image_size": [self.image_width, self.image_height],
            "detections": [d.to_dict() for d in self.detections],
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<DetectionResult count={self.count} found={self.found} "
            f"model={self.model_name}>"
        )
