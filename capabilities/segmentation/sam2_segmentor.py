"""
capabilities/segmentation/sam2_segmentor.py
============================================
SAM 2 — Segment Anything Model (Meta, 2024)

Generates precise pixel-level masks of any object, prompted by:
  - A point (click on the object)
  - A bounding box (from a detector like Grounding DINO or OWL-ViT)
  - Automatic everything-mode (segments all objects in the image)

SAM 2 does NOT classify — it only segments. You tell it WHERE to look
(with a point or box), and it draws the mask. Pair it with a detector
(Grounding DINO → finds WHERE, SAM 2 → draws exact shape).

This combination is the backbone of Use Case B (height estimation):
  1. Grounding DINO finds the sprout → bounding box
  2. SAM 2 draws the exact plant mask → precise pixel outline
  3. Depth Anything maps depth per pixel → real-world height

Hardware
--------
    CPU:       3–5s/image (SAM2-base)
    GPU 2GB:   ~200ms/image

Model variants
--------------
    "facebook/sam2-hiera-tiny"   38 MB  Fastest (default)
    "facebook/sam2-hiera-small" 185 MB  Good balance
    "facebook/sam2-hiera-base"  350 MB  Best accuracy

Requirements
------------
    pip install transformers torch torchvision

Usage
-----
    from capabilities.segmentation.sam2_segmentor import SAM2Segmentor

    seg = SAM2Segmentor()

    # Segment using a bounding box (e.g. from a detector)
    result = seg.segment_from_box(
        "plant.jpg",
        box=(120, 45, 310, 280)   # (x1, y1, x2, y2) pixels
    )
    print(result.mask.shape)        # (H, W) boolean numpy array
    print(result.area_px)           # 18400
    print(result.area_ratio)        # 0.115

    # Segment using a point click
    result = seg.segment_from_point(
        "plant.jpg",
        point=(215, 162)            # (x, y) pixel coordinate
    )

    # Auto-segment everything in the image
    result = seg.segment_all("plant.jpg")
    print(len(result.all_masks))    # 7
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SegmentationResult(CapabilityResult):
    """
    Result from SAM2Segmentor.

    Attributes
    ----------
    mask : np.ndarray or None
        Primary boolean mask (H × W). True = object pixel.
        Set by segment_from_box() and segment_from_point().
    all_masks : list of np.ndarray
        All masks when using segment_all() (automatic mode).
    area_px : float
        Number of pixels in the primary mask.
    area_ratio : float
        Fraction of image area covered by primary mask.
    image_width, image_height : int
        Dimensions of the source image.
    prompt_type : str
        How the segmentation was prompted: 'box', 'point', or 'auto'.
    prompt_value : any
        The actual prompt value (box tuple, point tuple, or None).
    """

    mask: Optional[np.ndarray] = field(default=None, repr=False)
    all_masks: List[np.ndarray] = field(default_factory=list, repr=False)
    area_px: float = 0.0
    area_ratio: float = 0.0
    image_width: int = 0
    image_height: int = 0
    prompt_type: str = ""
    prompt_value: Any = None

    @property
    def has_mask(self) -> bool:
        return self.mask is not None and self.mask.any()

    def get_bounding_box(self) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute tight bounding box around the primary mask.

        Returns (x1, y1, x2, y2) or None if no mask.
        """
        if self.mask is None or not self.mask.any():
            return None
        rows = np.any(self.mask, axis=1)
        cols = np.any(self.mask, axis=0)
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]
        return (int(x1), int(y1), int(x2), int(y2))

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "has_mask": self.has_mask,
            "area_px": round(self.area_px, 1),
            "area_ratio": round(self.area_ratio, 4),
            "image_size": [self.image_width, self.image_height],
            "prompt_type": self.prompt_type,
            "prompt_value": str(self.prompt_value) if self.prompt_value else None,
            "num_auto_masks": len(self.all_masks),
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<SegmentationResult has_mask={self.has_mask} "
            f"area_ratio={self.area_ratio:.3f} "
            f"prompt={self.prompt_type} model={self.model_name}>"
        )


# ---------------------------------------------------------------------------
# SAM2 Segmentor
# ---------------------------------------------------------------------------

class SAM2Segmentor(BaseCapability):
    """
    Pixel-level segmentation using Meta's SAM 2.

    Supports three prompting modes:
      - segment_from_box()   : segment the object inside a bounding box
      - segment_from_point() : segment the object at a clicked point
      - segment_all()        : automatically segment everything in the image

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Default: 'facebook/sam2-hiera-tiny'.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    _DEFAULT_MODEL = "facebook/sam2-hiera-tiny"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
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

    def run(self, image_path: str, **kwargs) -> SegmentationResult:
        """Alias for segment_from_box() if box= given, else segment_all()."""
        if "box" in kwargs:
            return self.segment_from_box(image_path, box=kwargs["box"])
        if "point" in kwargs:
            return self.segment_from_point(image_path, point=kwargs["point"])
        return self.segment_all(image_path)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def segment_from_box(
        self,
        image_path: str,
        box: Tuple[float, float, float, float],
    ) -> SegmentationResult:
        """
        Segment the object inside the given bounding box.

        Typical workflow: pair with Grounding DINO or OWL-ViT.
          1. detector.detect() → result.best.bbox
          2. segmentor.segment_from_box(image_path, box=result.best.bbox)

        Parameters
        ----------
        image_path : str
            Path to the image file.
        box : tuple (x1, y1, x2, y2)
            Bounding box in pixel coordinates.

        Returns
        -------
        SegmentationResult
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with self._timer() as t:
            inputs = self._processor(
                images=image,
                input_boxes=[[list(box)]],  # SAM2 expects nested list
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                    )

        # Take the highest-score mask
        mask = self._best_mask(masks[0])
        return self._build_result(
            image_path, mask, img_w, img_h, t.elapsed_ms, "box", box
        )

    def segment_from_point(
        self,
        image_path: str,
        point: Tuple[int, int],
        label: int = 1,
    ) -> SegmentationResult:
        """
        Segment the object at a clicked point.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        point : (x, y)
            Pixel coordinate of a point on the object to segment.
        label : int
            1 = foreground point (segment this), 0 = background point.

        Returns
        -------
        SegmentationResult
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with self._timer() as t:
            inputs = self._processor(
                images=image,
                input_points=[[[list(point)]]],
                input_labels=[[[label]]],
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
            )

        mask = self._best_mask(masks[0])
        return self._build_result(
            image_path, mask, img_w, img_h, t.elapsed_ms, "point", point
        )

    def segment_all(self, image_path: str) -> SegmentationResult:
        """
        Automatically segment all objects in the image.

        Returns all masks in result.all_masks, sorted by area (largest first).
        The primary result.mask is the largest detected region.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        SegmentationResult
            result.all_masks contains all detected segments.
            result.mask is the largest segment.
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        from PIL import Image
        import torch

        image = Image.open(image_path).convert("RGB")
        img_w, img_h = image.size

        with self._timer() as t:
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)

            masks = self._processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        inputs["original_sizes"].cpu(),
                    )

        # Flatten all masks into a list, sorted by area descending
        all_masks = []
        for m_batch in masks:
            for m in m_batch:
                all_masks.append(m.numpy().astype(bool))

        all_masks.sort(key=lambda m: m.sum(), reverse=True)
        primary = all_masks[0] if all_masks else None

        result = self._build_result(
            image_path, primary, img_w, img_h, t.elapsed_ms, "auto", None
        )
        result.all_masks = all_masks
        return result

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _reshaped_sizes(inputs):
        """
        Compatibility shim for Sam2Processor API changes across transformers versions.

        'reshaped_input_sizes' was renamed to 'input_sizes' in newer releases.
        Falls back to 'original_sizes' if neither key is present.
        """
        for key in ("reshaped_input_sizes", "input_sizes"):
            if key in inputs:
                return inputs[key].cpu()
        return inputs["original_sizes"].cpu()

    @staticmethod
    def _best_mask(masks_tensor) -> Optional[np.ndarray]:
        """Unwrap the mask tensor returned by post_process_masks."""
        if masks_tensor is None or masks_tensor.numel() == 0:
            return None
        # masks_tensor shape: [1, H, W] or [H, W]
        m = masks_tensor[0] if masks_tensor.dim() == 3 else masks_tensor
        return m.numpy().astype(bool)

    @staticmethod
    def _build_result(
        image_path, mask, img_w, img_h, elapsed_ms, prompt_type, prompt_value
    ) -> SegmentationResult:
        """Construct a SegmentationResult from computed mask."""
        img_area = img_w * img_h
        area_px = float(mask.sum()) if mask is not None else 0.0
        area_ratio = area_px / img_area if img_area > 0 else 0.0

        return SegmentationResult(
            image_path=image_path,
            model_name="sam2",
            duration_ms=elapsed_ms,
            mask=mask,
            area_px=area_px,
            area_ratio=area_ratio,
            image_width=img_w,
            image_height=img_h,
            prompt_type=prompt_type,
            prompt_value=prompt_value,
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import Sam2Processor, Sam2Model  # type: ignore

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[SAM2Segmentor] Loading '{self._model_id}' on {self._device} ...")
        self._processor = Sam2Processor.from_pretrained(self._model_id)
        self._model = Sam2Model.from_pretrained(self._model_id).to(self._device)
        self._model.eval()
        print("[SAM2Segmentor] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._device = None