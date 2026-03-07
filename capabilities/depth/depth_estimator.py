"""
capabilities/depth/depth_estimator.py
======================================
Depth Anything V2 — Monocular Depth Estimator

Estimates a per-pixel depth map from a single image with no stereo
cameras, no LiDAR, and no reference objects required.

Output modes
------------
    "relative"  — Depth values in [0, 1]. Closer objects have lower
                  values. Good for comparing regions within one image.
                  Use this when you don't know the real-world scale.

    "metric"    — Absolute depth in metres (requires metric model variant).
                  Use when you need real-world measurements.

Plant monitoring use cases
--------------------------
  - Use Case B (height estimation): depth of tip vs depth of soil base
    gives height = (depth_base - depth_tip) × scale_factor
  - Volume estimation: integrate depth over the segmented plant mask
  - Canopy height map: relative depth across the whole scene
  - Growth tracking: compare depth maps over time

Hardware
--------
    CPU: 5–30s/image depending on variant
    GPU 2GB: ~200ms/image

Model variants (via HuggingFace)
---------------------------------
    "depth-anything/Depth-Anything-V2-Small-hf"   100 MB  CPU-friendly  (default)
    "depth-anything/Depth-Anything-V2-Base-hf"    400 MB  Balanced
    "depth-anything/Depth-Anything-V2-Large-hf"   800 MB  Best accuracy, GPU rec.

Requirements
------------
    pip install transformers torch torchvision

Usage
-----
    from capabilities.depth.depth_estimator import DepthEstimator
    import numpy as np

    depth = DepthEstimator()

    result = depth.estimate("plant.jpg")

    print(result.depth_map.shape)       # (H, W) float32 array
    print(result.depth_map.min())       # closest point
    print(result.depth_map.max())       # furthest point

    # Height estimation: compare tip vs soil
    tip_depth  = result.sample_point(x=215, y=40)    # top of plant
    soil_depth = result.sample_point(x=215, y=390)   # soil level
    height_rel = soil_depth - tip_depth               # relative height

    # Depth of a region (e.g. from a segmentation mask)
    import numpy as np
    mask = np.ones((H, W), dtype=bool)  # your plant mask
    region_stats = result.region_stats(mask)
    print(region_stats)   # {mean, median, min, max, std}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class DepthResult(CapabilityResult):
    """
    Result from DepthEstimator.estimate().

    Attributes
    ----------
    depth_map : np.ndarray
        Float32 array (H × W) of depth values.
        Relative mode: values in [0, 1], lower = closer.
        Metric mode: values in metres.
    depth_mode : str
        'relative' or 'metric'.
    image_width, image_height : int
        Source image dimensions.
    depth_min, depth_max : float
        Min and max values across the entire depth map.
    depth_mean : float
        Mean depth across the entire image.
    """

    depth_map: Optional[np.ndarray] = field(default=None, repr=False)
    depth_mode: str = "relative"
    image_width: int = 0
    image_height: int = 0
    depth_min: float = 0.0
    depth_max: float = 0.0
    depth_mean: float = 0.0

    # ------------------------------------------------------------------ #
    # Query helpers
    # ------------------------------------------------------------------ #

    def sample_point(self, x: int, y: int) -> float:
        """
        Return the depth value at pixel coordinate (x, y).

        Parameters
        ----------
        x : int   Horizontal pixel position (column).
        y : int   Vertical pixel position (row).

        Returns
        -------
        float   Depth value at that pixel.
        """
        if self.depth_map is None:
            raise ValueError("No depth map available.")
        y = max(0, min(y, self.image_height - 1))
        x = max(0, min(x, self.image_width - 1))
        return float(self.depth_map[y, x])

    def region_stats(self, mask: np.ndarray) -> Dict[str, float]:
        """
        Compute depth statistics within a boolean mask region.

        Parameters
        ----------
        mask : np.ndarray (H × W, bool)
            Boolean mask — True pixels are included in statistics.
            Typically a segmentation mask from HSVSegmentor or SAM2.

        Returns
        -------
        dict with keys: mean, median, min, max, std, pixel_count
        """
        if self.depth_map is None:
            raise ValueError("No depth map available.")
        if mask.shape != self.depth_map.shape:
            raise ValueError(
                f"Mask shape {mask.shape} does not match "
                f"depth map shape {self.depth_map.shape}."
            )
        region = self.depth_map[mask]
        if region.size == 0:
            return {"mean": 0, "median": 0, "min": 0, "max": 0, "std": 0, "pixel_count": 0}
        return {
            "mean":        round(float(region.mean()), 4),
            "median":      round(float(np.median(region)), 4),
            "min":         round(float(region.min()), 4),
            "max":         round(float(region.max()), 4),
            "std":         round(float(region.std()), 4),
            "pixel_count": int(region.size),
        }

    def estimate_height(
        self,
        tip_point: Tuple[int, int],
        base_point: Tuple[int, int],
    ) -> Dict[str, float]:
        """
        Estimate the height of a plant from tip to base using depth values.

        This is the core computation for Use Case B. The depth difference
        between the soil (base_point) and the plant tip (tip_point) is
        proportional to the real-world height.

        In relative mode the result is in normalised depth units [0–1].
        In metric mode it is in metres.

        Parameters
        ----------
        tip_point : (x, y)
            Pixel coordinate of the topmost point of the plant.
        base_point : (x, y)
            Pixel coordinate of the soil/base of the plant.

        Returns
        -------
        dict with keys:
            tip_depth, base_depth, depth_difference, mode
        """
        tip_depth  = self.sample_point(*tip_point)
        base_depth = self.sample_point(*base_point)
        # In depth maps, closer = smaller value, so base is deeper (larger)
        difference = base_depth - tip_depth

        return {
            "tip_depth":        round(tip_depth, 4),
            "base_depth":       round(base_depth, 4),
            "depth_difference": round(difference, 4),
            "mode":             self.depth_mode,
        }

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "depth_mode": self.depth_mode,
            "image_size": [self.image_width, self.image_height],
            "depth_min":  round(self.depth_min, 4),
            "depth_max":  round(self.depth_max, 4),
            "depth_mean": round(self.depth_mean, 4),
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<DepthResult mode={self.depth_mode} "
            f"range=[{self.depth_min:.3f}, {self.depth_max:.3f}] "
            f"model={self.model_name}>"
        )


# ---------------------------------------------------------------------------
# Depth Estimator
# ---------------------------------------------------------------------------

class DepthEstimator(BaseCapability):
    """
    Monocular depth estimator using Depth Anything V2.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID.
        Default: 'depth-anything/Depth-Anything-V2-Small-hf' (100 MB).
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    _DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"

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
        return self._model_id.split("/")[-1].replace("-", "_").lower()

    def run(self, image_path: str, **kwargs) -> DepthResult:
        """Alias for estimate()."""
        return self.estimate(image_path)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def estimate(self, image_path: str) -> DepthResult:
        """
        Generate a per-pixel depth map for the image.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        DepthResult
            result.depth_map is a float32 (H × W) array.
            Use result.sample_point(), result.region_stats(), and
            result.estimate_height() for downstream analysis.
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
            # Post-process to original image size
            depth = self._processor.post_process_depth_estimation(
                outputs,
                target_sizes=[(img_h, img_w)],
            )[0]["predicted_depth"]

        depth_np = depth.squeeze().cpu().numpy().astype(np.float32)

        # Normalise to [0, 1] for relative mode
        d_min, d_max = float(depth_np.min()), float(depth_np.max())
        if d_max > d_min:
            depth_normalised = (depth_np - d_min) / (d_max - d_min)
        else:
            depth_normalised = np.zeros_like(depth_np)

        return DepthResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            depth_map=depth_normalised,
            depth_mode="relative",
            image_width=img_w,
            image_height=img_h,
            depth_min=float(depth_normalised.min()),
            depth_max=float(depth_normalised.max()),
            depth_mean=float(depth_normalised.mean()),
        )

    def estimate_with_mask(
        self,
        image_path: str,
        mask: np.ndarray,
    ) -> Tuple[DepthResult, Dict[str, float]]:
        """
        Estimate depth and immediately compute statistics over a mask region.

        Convenience method combining estimate() + result.region_stats()
        in a single call.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        mask : np.ndarray (H × W, bool)
            Boolean mask of the region of interest (e.g. plant pixels).

        Returns
        -------
        (DepthResult, dict)
            The full depth result and region statistics dict.
        """
        result = self.estimate(image_path)
        stats = result.region_stats(mask)
        return result, stats

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[DepthEstimator] Loading '{self._model_id}' on {self._device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(
            self._model_id
        ).to(self._device)
        self._model.eval()
        print("[DepthEstimator] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
