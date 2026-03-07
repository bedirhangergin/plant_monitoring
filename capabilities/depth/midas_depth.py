"""
capabilities/depth/midas_depth.py
===================================
MiDaS — Robust Monocular Depth Estimator (Alternative to Depth Anything)

MiDaS by Intel ISL is a battle-tested depth estimation model known for
strong generalisation across different environments and lighting conditions.
It is a reliable fallback when Depth Anything V2 produces poor results on
unusual plant images (e.g. close-up macro shots, overhead views).

When to use MiDaS vs Depth Anything V2
----------------------------------------
  Depth Anything V2:  Better on general scenes, newer architecture, default choice.
  MiDaS:             Better generalisation on unusual viewpoints and lighting.
                     Good fallback when Depth Anything fails.

Model variants
--------------
    "Intel/dpt-hybrid-midas"   400 MB  Best quality  (default)
    "Intel/dpt-large"          800 MB  Larger, more accurate
    "Intel/dpt-beit-large-512" 900 MB  State-of-the-art (GPU recommended)

Requirements
------------
    pip install transformers torch torchvision

Usage
-----
    from capabilities.depth.midas_depth import MiDaSDepth

    midas = MiDaSDepth()
    result = midas.estimate("plant.jpg")

    print(result.depth_map.shape)     # (H, W) float32
    print(result.depth_mean)          # average depth across image

    # The result is identical in interface to DepthEstimator result,
    # so you can swap them without changing downstream code
    tip_depth  = result.sample_point(x=215, y=40)
    base_depth = result.sample_point(x=215, y=390)
    height     = result.estimate_height((215, 40), (215, 390))
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from capabilities.base import BaseCapability
from capabilities.depth.depth_estimator import DepthResult   # Reuse same result type


class MiDaSDepth(BaseCapability):
    """
    Monocular depth estimator using MiDaS (Intel ISL).

    Returns the same DepthResult type as DepthEstimator, so both can be
    used interchangeably in any pipeline that consumes depth maps.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Default: 'Intel/dpt-hybrid-midas'.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    _DEFAULT_MODEL = "Intel/dpt-hybrid-midas"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._device_override = device
        self._model = None
        self._processor = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return self._model_id.split("/")[-1].replace("-", "_").lower()

    def run(self, image_path: str, **kwargs) -> DepthResult:
        """Alias for estimate()."""
        return self.estimate(image_path)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def estimate(self, image_path: str) -> DepthResult:
        """
        Generate a normalised depth map using MiDaS.

        Returns the same DepthResult type as DepthEstimator for
        full interface compatibility.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        DepthResult
            Identical interface to DepthEstimator.estimate().
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
                predicted_depth = outputs.predicted_depth

            # Interpolate to original image size
            import torch.nn.functional as F
            prediction = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(img_h, img_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        depth_np = prediction.cpu().numpy().astype(np.float32)

        # MiDaS outputs inverse depth (larger = closer), normalise to [0,1]
        # Invert so that closer objects have smaller values (consistent with Depth Anything)
        d_min, d_max = float(depth_np.min()), float(depth_np.max())
        if d_max > d_min:
            normalised = (depth_np - d_min) / (d_max - d_min)
            # Invert: MiDaS larger value = closer, we want smaller = closer
            normalised = 1.0 - normalised
        else:
            normalised = np.zeros_like(depth_np)

        return DepthResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            depth_map=normalised,
            depth_mode="relative",
            image_width=img_w,
            image_height=img_h,
            depth_min=float(normalised.min()),
            depth_max=float(normalised.max()),
            depth_mean=float(normalised.mean()),
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import DPTImageProcessor, DPTForDepthEstimation

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[MiDaSDepth] Loading '{self._model_id}' on {self._device} ...")
        self._processor = DPTImageProcessor.from_pretrained(self._model_id)
        self._model = DPTForDepthEstimation.from_pretrained(
            self._model_id
        ).to(self._device)
        self._model.eval()
        print("[MiDaSDepth] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
