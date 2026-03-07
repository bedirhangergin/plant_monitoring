"""
capabilities/segmentation/hsv_segmentor.py
===========================================
Classical HSV Colour Segmentation

Segments plant regions from images using HSV colour space masking.
Free, zero model download, <10ms per image.

Unlike the HSVDetector in sprout_detection (which only gives a yes/no
answer), this segmentor returns a full pixel mask that can be used for:
  - Precise plant area measurement
  - Feeding into depth estimation (mask the plant region only)
  - Colour analysis on segmented plant vs background separately
  - Leaf count via connected components
  - Coverage tracking over time

Colour profiles
---------------
The segmentor ships with built-in profiles for common use cases, and
supports fully custom HSV ranges:

    "green_plant"    — General green plant material (default)
    "yellow_stress"  — Yellowing / chlorosis detection
    "brown_disease"  — Browning, necrosis, dead tissue
    "red_fruit"      — Red/orange fruits (tomato, pepper, apple)
    "purple_flower"  — Purple/violet flowers
    "custom"         — Pass your own hsv_lower / hsv_upper

Usage
-----
    from capabilities.segmentation.hsv_segmentor import HSVSegmentor

    seg = HSVSegmentor()

    # Segment green plant material
    result = seg.segment("plant.jpg", profile="green_plant")
    print(result.coverage_ratio)       # 0.34  (34% of image is plant)
    print(result.mask.shape)           # (480, 640)
    print(result.component_count)      # 3  (three separate plant regions)

    # Segment yellowing areas to detect stress
    result = seg.segment("leaf.jpg", profile="yellow_stress")
    print(result.coverage_ratio)       # 0.08  (8% of leaf is yellowing)

    # Custom HSV range
    result = seg.segment(
        "flower.jpg",
        profile="custom",
        hsv_lower=(140, 50, 50),    # Purple lower bound
        hsv_upper=(170, 255, 255),  # Purple upper bound
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Built-in HSV colour profiles
# ---------------------------------------------------------------------------

# Each profile is (hsv_lower, hsv_upper) tuples
# OpenCV HSV: H 0–179, S 0–255, V 0–255

HSV_PROFILES: Dict[str, Tuple[Tuple, Tuple]] = {
    # General green plant material
    "green_plant": (
        (35, 40, 40),
        (85, 255, 255),
    ),
    # Yellowing / chlorosis — leaves turning yellow
    "yellow_stress": (
        (20, 60, 80),
        (35, 255, 255),
    ),
    # Browning / necrosis / dead tissue
    "brown_disease": (
        (8, 40, 30),
        (20, 200, 180),
    ),
    # Red/orange fruits (tomato, pepper, strawberry)
    "red_fruit": (
        (0, 80, 80),
        (15, 255, 255),
    ),
    # Purple / violet flowers
    "purple_flower": (
        (130, 50, 50),
        (170, 255, 255),
    ),
    # White / pale features (flower petals, mold)
    "white_pale": (
        (0, 0, 180),
        (179, 40, 255),
    ),
    # Dark spots / lesions on leaves
    "dark_lesion": (
        (0, 0, 0),
        (179, 255, 60),
    ),
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class HSVSegmentationResult(CapabilityResult):
    """
    Result from HSVSegmentor.segment().

    Attributes
    ----------
    mask : np.ndarray
        Binary mask (H × W, uint8). 255 = target colour, 0 = background.
    profile : str
        Name of the colour profile used.
    coverage_ratio : float
        Fraction of image pixels matching the target colour (0.0–1.0).
    coverage_pct : float
        Coverage as a percentage (0.0–100.0).
    component_count : int
        Number of separate connected regions found in the mask.
        e.g. 3 separate plant clusters = 3.
    components : list of dict
        Per-component stats: area, centroid, bounding box.
    image_width, image_height : int
        Source image dimensions.
    """

    mask: Optional[np.ndarray] = field(default=None, repr=False)
    profile: str = ""
    coverage_ratio: float = 0.0
    coverage_pct: float = 0.0
    component_count: int = 0
    components: List[Dict] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "profile": self.profile,
            "coverage_ratio": round(self.coverage_ratio, 4),
            "coverage_pct": round(self.coverage_pct, 2),
            "component_count": self.component_count,
            "image_size": [self.image_width, self.image_height],
            "components": self.components,
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<HSVSegmentationResult profile='{self.profile}' "
            f"coverage={self.coverage_pct:.1f}% "
            f"components={self.component_count}>"
        )


# ---------------------------------------------------------------------------
# Segmentor class
# ---------------------------------------------------------------------------

class HSVSegmentor(BaseCapability):
    """
    Classical HSV colour segmentation for plant images.

    No model download, no GPU, <10ms per image.

    Parameters
    ----------
    morph_kernel_size : int
        Morphological kernel size for noise removal. Default: 5.
    min_component_area_px : int
        Minimum pixel area for a connected component to be reported.
        Smaller regions are treated as noise. Default: 50.
    """

    def __init__(
        self,
        morph_kernel_size: int = 5,
        min_component_area_px: int = 50,
    ) -> None:
        self._kernel_size = morph_kernel_size
        self._min_area = min_component_area_px
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return "hsv_segmentor"

    def run(self, image_path: str, **kwargs) -> HSVSegmentationResult:
        """Alias for segment(). Pass profile= as keyword argument."""
        profile = kwargs.get("profile", "green_plant")
        return self.segment(image_path, profile=profile)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def segment(
        self,
        image_path: str,
        profile: str = "green_plant",
        hsv_lower: Optional[Tuple[int, int, int]] = None,
        hsv_upper: Optional[Tuple[int, int, int]] = None,
    ) -> HSVSegmentationResult:
        """
        Segment pixels matching a colour profile.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        profile : str
            Colour profile name. One of: green_plant, yellow_stress,
            brown_disease, red_fruit, purple_flower, white_pale,
            dark_lesion, custom.
        hsv_lower : tuple, optional
            Custom HSV lower bound (H, S, V). Required when profile='custom'.
        hsv_upper : tuple, optional
            Custom HSV upper bound (H, S, V). Required when profile='custom'.

        Returns
        -------
        HSVSegmentationResult
        """
        self.validate_image(image_path)

        # Resolve HSV bounds
        if profile == "custom":
            if hsv_lower is None or hsv_upper is None:
                raise ValueError(
                    "profile='custom' requires hsv_lower and hsv_upper to be provided."
                )
            lower = np.array(hsv_lower, dtype=np.uint8)
            upper = np.array(hsv_upper, dtype=np.uint8)
        elif profile in HSV_PROFILES:
            lo, hi = HSV_PROFILES[profile]
            lower = np.array(lo, dtype=np.uint8)
            upper = np.array(hi, dtype=np.uint8)
        else:
            available = list(HSV_PROFILES.keys()) + ["custom"]
            raise ValueError(
                f"Unknown profile '{profile}'. Available: {available}"
            )

        with self._timer() as t:
            img_bgr = cv2.imread(image_path)
            img_h, img_w = img_bgr.shape[:2]
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # Build binary mask
            mask = cv2.inRange(img_hsv, lower, upper)

            # Morphological cleanup
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

            # Coverage statistics
            total_px = img_h * img_w
            covered_px = int(np.count_nonzero(mask))
            coverage_ratio = covered_px / total_px if total_px > 0 else 0.0

            # Connected component analysis
            components = self._analyse_components(mask, img_w, img_h)

        return HSVSegmentationResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            mask=mask,
            profile=profile,
            coverage_ratio=coverage_ratio,
            coverage_pct=coverage_ratio * 100.0,
            component_count=len(components),
            components=components,
            image_width=img_w,
            image_height=img_h,
        )

    def segment_multi(
        self,
        image_path: str,
        profiles: List[str],
    ) -> Dict[str, HSVSegmentationResult]:
        """
        Run multiple colour profiles on one image in a single pass.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        profiles : list of str
            Profile names to run simultaneously.

        Returns
        -------
        dict mapping profile name → HSVSegmentationResult
        """
        return {profile: self.segment(image_path, profile=profile) for profile in profiles}

    @property
    def available_profiles(self) -> List[str]:
        """List of built-in colour profiles."""
        return list(HSV_PROFILES.keys()) + ["custom"]

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _analyse_components(
        self, mask: np.ndarray, img_w: int, img_h: int
    ) -> List[Dict]:
        """
        Find connected regions in the mask and compute per-component stats.

        Returns a list of dicts, each with:
          area_px, area_ratio, centroid_x, centroid_y,
          bbox (x1, y1, x2, y2)
        Sorted by area descending. Regions smaller than min_component_area_px
        are excluded.
        """
        img_area = img_w * img_h
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        components = []
        # Label 0 is background — skip it
        for i in range(1, num_labels):
            area_px = int(stats[i, cv2.CC_STAT_AREA])
            if area_px < self._min_area:
                continue

            x1 = int(stats[i, cv2.CC_STAT_LEFT])
            y1 = int(stats[i, cv2.CC_STAT_TOP])
            w  = int(stats[i, cv2.CC_STAT_WIDTH])
            h  = int(stats[i, cv2.CC_STAT_HEIGHT])
            cx, cy = centroids[i]

            components.append({
                "area_px": area_px,
                "area_ratio": round(area_px / img_area, 4),
                "centroid_x": round(float(cx), 1),
                "centroid_y": round(float(cy), 1),
                "bbox": [x1, y1, x1 + w, y1 + h],
            })

        components.sort(key=lambda c: c["area_px"], reverse=True)
        return components
