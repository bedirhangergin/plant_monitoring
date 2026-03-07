"""
capabilities/analysis/coverage_estimator.py
============================================
Canopy Coverage Estimator

Tracks the fraction of the image occupied by plant material over time.
Pure OpenCV, no model download, <10ms per image.

Coverage is one of the most informative single numbers in plant monitoring:
  - Rising coverage = healthy vegetative growth
  - Plateauing coverage = canopy closure / maturity
  - Falling coverage = senescence, disease defoliation, or harvest
  - Sudden drop = wilting or physical damage

The estimator tracks any configurable set of colour profiles (green, yellow,
etc.) so you can monitor overall coverage AND health ratio simultaneously.

Usage
-----
    from capabilities.analysis.coverage_estimator import CoverageEstimator

    est = CoverageEstimator()

    # Single image
    result = est.estimate("plant_day7.jpg")
    print(result.green_coverage_pct)    # 23.4%
    print(result.total_coverage_pct)    # 26.1%  (green + yellow)
    print(result.health_ratio)          # 0.897   (green / total)

    # Track a time series
    image_paths = ["day1.jpg", "day7.jpg", "day14.jpg", "day21.jpg"]
    series = est.estimate_series(image_paths)
    for r in series:
        print(f"{r.image_path}: {r.green_coverage_pct:.1f}% green")

    # Plot the time series
    est.plot_series(series, save_path="growth_curve.png")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CoverageResult(CapabilityResult):
    """
    Result from CoverageEstimator.estimate().

    Attributes
    ----------
    green_coverage_ratio : float
        Fraction of image that is healthy green.
    yellow_coverage_ratio : float
        Fraction of image that is yellowing.
    brown_coverage_ratio : float
        Fraction of image that is brown/necrotic.
    total_coverage_ratio : float
        Fraction of image containing any plant material (all bands combined).
    health_ratio : float
        green / total_plant.  High = most plant material is healthy green.
    green_coverage_pct : float
        Green coverage as a percentage.
    total_coverage_pct : float
        Total plant coverage as a percentage.
    image_width, image_height : int
        Source image dimensions.
    """

    green_coverage_ratio: float = 0.0
    yellow_coverage_ratio: float = 0.0
    brown_coverage_ratio: float = 0.0
    total_coverage_ratio: float = 0.0
    health_ratio: float = 0.0
    green_coverage_pct: float = 0.0
    total_coverage_pct: float = 0.0
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "green_coverage_ratio":  round(self.green_coverage_ratio, 4),
            "yellow_coverage_ratio": round(self.yellow_coverage_ratio, 4),
            "brown_coverage_ratio":  round(self.brown_coverage_ratio, 4),
            "total_coverage_ratio":  round(self.total_coverage_ratio, 4),
            "health_ratio":          round(self.health_ratio, 4),
            "green_coverage_pct":    round(self.green_coverage_pct, 2),
            "total_coverage_pct":    round(self.total_coverage_pct, 2),
            "image_size":            [self.image_width, self.image_height],
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<CoverageResult green={self.green_coverage_pct:.1f}% "
            f"total={self.total_coverage_pct:.1f}% "
            f"health_ratio={self.health_ratio:.2f}>"
        )


# ---------------------------------------------------------------------------
# Coverage Estimator
# ---------------------------------------------------------------------------

class CoverageEstimator(BaseCapability):
    """
    Plant canopy coverage estimator using HSV colour analysis.

    No model download. Pure OpenCV. <10ms per image.

    Parameters
    ----------
    morph_kernel_size : int
        Morphological kernel for noise removal. Default: 5.
    """

    # HSV range definitions for each tracked colour
    _GREEN  = ((35, 40, 40),  (85, 255, 255))
    _YELLOW = ((20, 60, 80),  (35, 255, 255))
    _BROWN  = (( 8, 40, 30),  (20, 200, 180))

    def __init__(self, morph_kernel_size: int = 5) -> None:
        k = morph_kernel_size
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    @property
    def model_name(self) -> str:
        return "coverage_estimator_hsv"

    def run(self, image_path: str, **kwargs) -> CoverageResult:
        """Alias for estimate()."""
        return self.estimate(image_path)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def estimate(self, image_path: str) -> CoverageResult:
        """
        Estimate plant coverage for a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        CoverageResult
        """
        self.validate_image(image_path)

        with self._timer() as t:
            img_bgr = cv2.imread(image_path)
            img_h, img_w = img_bgr.shape[:2]
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
            total_px = img_h * img_w

            green_ratio  = self._band_ratio(img_hsv, self._GREEN,  total_px)
            yellow_ratio = self._band_ratio(img_hsv, self._YELLOW, total_px)
            brown_ratio  = self._band_ratio(img_hsv, self._BROWN,  total_px)

            total_ratio = min(green_ratio + yellow_ratio + brown_ratio, 1.0)
            health_ratio = green_ratio / (total_ratio + 1e-9)

        return CoverageResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            green_coverage_ratio=green_ratio,
            yellow_coverage_ratio=yellow_ratio,
            brown_coverage_ratio=brown_ratio,
            total_coverage_ratio=total_ratio,
            health_ratio=float(health_ratio),
            green_coverage_pct=green_ratio * 100.0,
            total_coverage_pct=total_ratio * 100.0,
            image_width=img_w,
            image_height=img_h,
        )

    def estimate_series(
        self,
        image_paths: List[str],
        stop_on_error: bool = False,
    ) -> List[CoverageResult]:
        """
        Estimate coverage for a time-ordered list of images.

        Parameters
        ----------
        image_paths : list of str
            Paths in chronological order.
        stop_on_error : bool
            If False (default), skip failed images. If True, re-raise.

        Returns
        -------
        list of CoverageResult
        """
        results = []
        for path in image_paths:
            try:
                results.append(self.estimate(path))
            except Exception as e:
                if stop_on_error:
                    raise
                print(f"[CoverageEstimator] Skipped '{path}': {e}")
        return results

    def plot_series(
        self,
        results: List[CoverageResult],
        save_path: Optional[str] = None,
        title: str = "Plant Coverage Over Time",
    ):
        """
        Plot green, yellow, and brown coverage over a time series.

        Parameters
        ----------
        results : list of CoverageResult
            Ordered list of coverage results.
        save_path : str, optional
            If provided, save figure to this path instead of showing it.
        title : str
            Plot title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        import os

        labels = [os.path.basename(r.image_path) for r in results]
        x = range(len(results))

        green  = [r.green_coverage_pct  for r in results]
        yellow = [r.yellow_coverage_pct for r in results]
        brown  = [r.brown_coverage_pct  for r in results]
        total  = [r.total_coverage_pct  for r in results]

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        fig.suptitle(title, fontsize=13, fontweight="bold")

        # Coverage stacked area
        axes[0].stackplot(
            x, green, yellow, brown,
            labels=["Green (healthy)", "Yellow (stressed)", "Brown (necrotic)"],
            colors=["#2ecc71", "#f1c40f", "#a04000"],
            alpha=0.8,
        )
        axes[0].plot(x, total, "k--", linewidth=1.5, label="Total coverage")
        axes[0].set_ylabel("Coverage (%)")
        axes[0].legend(loc="upper left", fontsize=8)
        axes[0].set_ylim(0, 100)
        axes[0].grid(alpha=0.3)

        # Health ratio
        health = [r.health_ratio for r in results]
        axes[1].plot(x, health, color="#2980b9", linewidth=2, marker="o", markersize=4)
        axes[1].axhline(0.80, color="#27ae60", linestyle="--", linewidth=1, alpha=0.5, label="Healthy threshold (0.80)")
        axes[1].set_ylabel("Health Ratio (green/total)")
        axes[1].set_ylim(0, 1.05)
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _band_ratio(self, img_hsv: np.ndarray, bounds: tuple, total_px: int) -> float:
        """Compute the pixel fraction matching an HSV colour band."""
        lower = np.array(bounds[0], dtype=np.uint8)
        upper = np.array(bounds[1], dtype=np.uint8)
        mask = cv2.inRange(img_hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return float(np.count_nonzero(mask)) / total_px
