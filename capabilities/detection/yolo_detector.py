"""
capabilities/detection/yolo_detector.py
========================================
YOLOv8 — Ultra-Fast Object Detector (Closed Vocabulary)

YOLOv8 by Ultralytics is the fastest local detector available — <50ms
on CPU. It detects 80 COCO classes including: person, potted plant,
bottle, scissors, and many more.

Because YOLO is trained on COCO classes, it cannot detect "sprout" or
"diseased leaf" directly. However, it is extremely useful for:

  - Detecting pots / containers (class: "potted plant", "vase")
  - Counting plants in overview shots (class: "potted plant")
  - Detecting people in the frame (class: "person")
  - Detecting tools/objects near the plant (class: "scissors", "bottle")
  - Scene framing and image normalisation

For plant-specific detection, use GroundingDINODetector or OWLViTDetector.

Model variants (nano → xlarge, speed vs accuracy tradeoff)
-----------------------------------------------------------
    "yolov8n.pt"  6 MB    <5ms GPU  / <50ms CPU   (default — fastest)
    "yolov8s.pt"  22 MB   <10ms GPU / ~100ms CPU
    "yolov8m.pt"  52 MB   <20ms GPU / ~200ms CPU
    "yolov8l.pt"  87 MB   <30ms GPU / ~500ms CPU
    "yolov8x.pt"  130 MB  <50ms GPU / ~1s CPU     (most accurate)

Requirements
------------
    pip install ultralytics

Usage
-----
    from capabilities.detection.yolo_detector import YOLODetector

    detector = YOLODetector()

    # Detect all objects above 0.5 confidence
    result = detector.detect("greenhouse.jpg")
    print(result.count)               # 5
    print(result.best.label)          # "potted plant"
    print(result.best.confidence)     # 0.91

    # Detect only specific COCO classes
    result = detector.detect("greenhouse.jpg", filter_classes=["potted plant", "person"])

    # Check if a pot/container is in frame (useful for framing/normalisation)
    has_pot = any(d.label == "potted plant" for d in result.detections)
"""

from __future__ import annotations

from typing import List, Optional

from capabilities.base import BaseCapability
from capabilities.detection.base_detection_result import (
    Detection, DetectionResult
)


class YOLODetector(BaseCapability):
    """
    Ultra-fast object detector using YOLOv8 (Ultralytics).

    Parameters
    ----------
    model_variant : str
        YOLOv8 model file. Default: 'yolov8n.pt' (nano — fastest).
        Downloads automatically on first use (~6 MB for nano).
    confidence_threshold : float
        Minimum detection confidence. Default: 0.50.
    iou_threshold : float
        NMS IoU threshold. Default: 0.45.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        model_variant: str = "yolov8n.pt",
        confidence_threshold: float = 0.50,
        iou_threshold: float = 0.45,
        device: Optional[str] = None,
    ) -> None:
        self._variant = model_variant
        self._conf = confidence_threshold
        self._iou = iou_threshold
        self._device_override = device
        self._model = None  # Lazy-loaded

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return "yolo_" + self._variant.replace(".pt", "").replace("yolov8", "v8")

    def run(self, image_path: str, **kwargs) -> DetectionResult:
        """Alias for detect(). Pass filter_classes= as keyword argument."""
        filter_classes = kwargs.get("filter_classes", None)
        return self.detect(image_path, filter_classes=filter_classes)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        image_path: str,
        filter_classes: Optional[List[str]] = None,
        confidence_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Run YOLO detection on an image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        filter_classes : list of str, optional
            If provided, only return detections for these COCO class names.
            Example: ["potted plant", "person"]
        confidence_threshold : float, optional
            Override instance-level threshold for this call.

        Returns
        -------
        DetectionResult
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        conf = confidence_threshold if confidence_threshold is not None else self._conf

        with self._timer() as t:
            results = self._model.predict(
                source=image_path,
                conf=conf,
                iou=self._iou,
                verbose=False,
            )

        result = results[0]
        img_h, img_w = result.orig_shape
        img_area = img_w * img_h

        detections = []
        for box in result.boxes:
            label = self._model.names[int(box.cls)]

            # Apply class filter if specified
            if filter_classes and label not in filter_classes:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
            area_px = (x2 - x1) * (y2 - y1)

            detections.append(Detection(
                label=label,
                confidence=float(box.conf),
                bbox=(x1, y1, x2, y2),
                bbox_normalised=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                area_px=area_px,
                area_ratio=area_px / img_area if img_area > 0 else 0.0,
            ))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        prompts = filter_classes or list(self._model.names.values())

        return DetectionResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            detections=detections,
            prompts=prompts,
            image_width=img_w,
            image_height=img_h,
        )

    @property
    def coco_classes(self) -> List[str]:
        """List of all 80 COCO class names this model can detect."""
        self._ensure_loaded()
        return list(self._model.names.values())

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Load YOLOv8 model. Downloads automatically on first use."""
        if self._model is not None:
            return

        from ultralytics import YOLO  # type: ignore

        print(f"[YOLODetector] Loading '{self._variant}' ...")
        self._model = YOLO(self._variant)
        print("[YOLODetector] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
