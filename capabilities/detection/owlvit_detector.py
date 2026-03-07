"""
capabilities/detection/owlvit_detector.py
==========================================
OWL-ViT — Open-Vocabulary Object Detector (CPU-Friendly)

OWL-ViT (Owlvit) by Google is an open-vocabulary detector that is
significantly lighter and faster than Grounding DINO, making it the
preferred choice for CPU-only deployments.

Model variants
--------------
    "google/owlvit-base-patch32"  200 MB  Fastest  (default)
    "google/owlvit-base-patch16"  200 MB  Better accuracy, slower
    "google/owlvit-large-patch14" 800 MB  Best accuracy, GPU recommended

Speed comparison vs Grounding DINO (CPU)
-----------------------------------------
    OWL-ViT base-patch32:  ~2–3s/image
    Grounding DINO base:   ~5–8s/image

OWL-ViT is the right choice when:
  - GPU is not available
  - Speed is more important than maximum accuracy
  - You need detection on modest hardware (Raspberry Pi, laptop)

Requirements
------------
    pip install transformers torch torchvision

Usage
-----
    from capabilities.detection.owlvit_detector import OWLViTDetector

    detector = OWLViTDetector()

    result = detector.detect(
        "plant.jpg",
        prompts=["a green seedling", "a yellow leaf", "a wilting plant"]
    )
    for d in result.detections:
        print(f"  {d.label}: conf={d.confidence:.2f}  bbox={d.bbox}")
"""

from __future__ import annotations

from typing import List, Optional

from capabilities.base import BaseCapability
from capabilities.detection.base_detection_result import (
    Detection, DetectionResult
)


class OWLViTDetector(BaseCapability):
    """
    Open-vocabulary object detector using OWL-ViT (Google).

    Lighter and faster than Grounding DINO. Returns bounding boxes for
    any text-described objects. Good CPU performance.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Default: 'google/owlvit-base-patch32'.
    score_threshold : float
        Minimum detection score. Default: 0.10 (OWL-ViT scores are lower
        than Grounding DINO scores — calibrated differently).
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    _DEFAULT_MODEL = "google/owlvit-base-patch32"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        score_threshold: float = 0.10,
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
        return self._model_id.split("/")[-1].replace("-", "_")

    def run(self, image_path: str, **kwargs) -> DetectionResult:
        """Alias for detect(). Pass prompts= as keyword argument."""
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

        OWL-ViT prompts work best as short noun phrases prefixed with "a":
          ✅ "a green sprout"
          ✅ "a diseased leaf"
          ✅ "a yellow flower"
          ⚠️  "diseased" (too short, less reliable)

        Parameters
        ----------
        image_path : str
            Path to the image file.
        prompts : list of str
            Text descriptions of objects to detect.
        score_threshold : float, optional
            Override instance-level threshold for this call.

        Returns
        -------
        DetectionResult
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        threshold = score_threshold if score_threshold is not None else self._score_threshold

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        # OWL-ViT expects a list-of-lists: [[prompt1, prompt2, ...]]
        text_queries = [prompts]

        with self._timer() as t:
            inputs = self._processor(
                text=text_queries,
                images=image,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            # Post-process to absolute pixel coordinates
            target_sizes = torch.tensor([(img_h, img_w)])
            results = self._processor.post_process_object_detection(
                outputs=outputs,
                threshold=threshold,
                target_sizes=target_sizes,
            )[0]

        detections = self._build_detections(results, img_w, img_h, prompts)

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
        """Convert raw model outputs to sorted Detection objects."""
        detections = []
        img_area = img_w * img_h

        scores = raw_results["scores"].tolist()
        labels_indices = raw_results["labels"].tolist()
        boxes = raw_results["boxes"].tolist()

        for score, label_idx, box in zip(scores, labels_indices, boxes):
            # label_idx is an index into prompts
            label = prompts[label_idx] if label_idx < len(prompts) else f"class_{label_idx}"
            x1, y1, x2, y2 = box
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

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections

    def _ensure_loaded(self) -> None:
        """Load model from HuggingFace if not already loaded."""
        if self._model is not None:
            return

        import torch
        from transformers import OwlViTProcessor, OwlViTForObjectDetection

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[OWLViTDetector] Loading '{self._model_id}' on {self._device} ...")
        self._processor = OwlViTProcessor.from_pretrained(self._model_id)
        self._model = OwlViTForObjectDetection.from_pretrained(
            self._model_id
        ).to(self._device)
        self._model.eval()
        print("[OWLViTDetector] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
