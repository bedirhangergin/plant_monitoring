"""
capabilities/detection/grounding_dino.py
=========================================
Grounding DINO — Open-Vocabulary Object Detector

Detects objects described by free-form text prompts. No fixed class list,
no retraining. Give it a text description, get bounding boxes back.

This is the most capable free local detector in the toolkit. It combines
a DINO vision backbone with BERT text encoding to achieve open-vocabulary
detection at near-state-of-the-art accuracy.

Typical use cases in plant monitoring
--------------------------------------
    detector.detect("plant.jpg", prompts=["green sprout"])
    detector.detect("plant.jpg", prompts=["yellow leaf", "brown spot"])
    detector.detect("plant.jpg", prompts=["flower bud", "open flower"])
    detector.detect("plant.jpg", prompts=["aphid", "caterpillar", "pest"])
    detector.detect("plant.jpg", prompts=["fruit", "unripe tomato"])

Hardware
--------
    CPU: 5–8s per image
    GPU (2 GB VRAM): ~500ms per image

Model variants
--------------
    "IDEA-Research/grounding-dino-base"   700 MB   Best accuracy  (default)
    "IDEA-Research/grounding-dino-tiny"   175 MB   Faster, smaller

Requirements
------------
    pip install transformers torch torchvision

Usage
-----
    from capabilities.detection.grounding_dino import GroundingDINODetector

    detector = GroundingDINODetector()

    result = detector.detect("plant.jpg", prompts=["green sprout", "bare soil"])
    print(result.found)           # True
    print(result.count)           # 2
    print(result.best.label)      # "green sprout"
    print(result.best.bbox)       # (120, 45, 310, 280)
    print(result.best.confidence) # 0.73

    # Filter by label
    sprouts = result.filter_by_label("green sprout")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from capabilities.base import BaseCapability
from capabilities.detection.base_detection_result import (
    BBox, Detection, DetectionResult
)


class GroundingDINODetector(BaseCapability):
    """
    Open-vocabulary object detector using Grounding DINO.

    Detects objects by text description. The model is loaded lazily on
    the first call to detect() and cached for reuse.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Default: 'IDEA-Research/grounding-dino-base'.
    score_threshold : float
        Minimum detection confidence to include in results (0.0–1.0).
        Lower = more detections but more false positives. Default: 0.35.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    _DEFAULT_MODEL = "IDEA-Research/grounding-dino-base"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        score_threshold: float = 0.35,
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._score_threshold = score_threshold
        self._device_override = device
        # Lazy-loaded
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        # "IDEA-Research/grounding-dino-base" → "grounding_dino_base"
        return self._model_id.split("/")[-1].replace("-", "_")

    def run(self, image_path: str, **kwargs) -> DetectionResult:
        """Alias for detect(). Pass prompts= as a keyword argument."""
        prompts = kwargs.get("prompts", ["plant"])
        return self.detect(image_path, prompts=prompts)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def detect(
        self,
        image_path: str,
        prompts: List[str],
        score_threshold: Optional[float] = None,
    ) -> DetectionResult:
        """
        Detect objects matching the text prompts in an image.

        Grounding DINO takes a text string where multiple objects are
        separated by ' . ' (period-space). We handle this automatically.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompts : list of str
            Text descriptions of objects to find.
            Examples: ["green sprout"], ["yellow leaf", "brown spot"]
        score_threshold : float, optional
            Override the instance-level score threshold for this call.

        Returns
        -------
        DetectionResult
            All detections above threshold, sorted by confidence.
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        threshold = score_threshold if score_threshold is not None else self._score_threshold

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # Grounding DINO expects prompts joined with " . "
        text_prompt = " . ".join(p.lower().strip(".") for p in prompts) + " ."

        with self._timer() as t:
            inputs = self._processor(
                images=image,
                text=text_prompt,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process: convert to absolute bounding boxes
            target_sizes = torch.tensor([(img_h, img_w)])
            results = self._processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=target_sizes,
            )[0]

        detections = self._build_detections(
            results, img_w, img_h, prompts
        )

        return DetectionResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            detections=detections,
            prompts=prompts,
            image_width=img_w,
            image_height=img_h,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _build_detections(
        self,
        raw_results: dict,
        img_w: int,
        img_h: int,
        prompts: List[str],
    ) -> List[Detection]:
        """Convert raw model output to sorted list of Detection objects."""
        detections = []
        img_area = img_w * img_h

        scores = raw_results["scores"].tolist()
        labels = raw_results["labels"]
        boxes  = raw_results["boxes"].tolist()

        for score, label, box in zip(scores, labels, boxes):
            x1, y1, x2, y2 = box
            # Clamp to image bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img_w, x2), min(img_h, y2)
            area_px = (x2 - x1) * (y2 - y1)

            detections.append(Detection(
                label=label,
                confidence=float(score),
                bbox=(x1, y1, x2, y2),
                bbox_normalised=(x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h),
                area_px=area_px,
                area_ratio=area_px / img_area if img_area > 0 else 0.0,
            ))

        # Sort by confidence descending
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _ensure_loaded(self) -> None:
        """Load model from HuggingFace if not already loaded."""
        if self._model is not None:
            return

        import torch
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[GroundingDINODetector] Loading '{self._model_id}' on {self._device} ...")
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForZeroShotObjectDetection.from_pretrained(
            self._model_id
        ).to(self._device)
        self._model.eval()
        print("[GroundingDINODetector] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        """Release model from memory."""
        self._model = None
        self._processor = None
        self._device = None
