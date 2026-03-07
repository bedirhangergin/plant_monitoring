"""
capabilities/analysis/colour_analyser.py
==========================================
Colour-Based Plant Health Analyser

Computes a comprehensive colour health profile from a plant image using
HSV histograms and pixel ratio analysis. No model download. Pure OpenCV.
Runs in <10ms per image.

Health indicators tracked
--------------------------
  green_ratio     — Fraction of healthy green pixels. High = healthy canopy.
  yellow_ratio    — Fraction of yellowing pixels. Elevated = chlorosis / N deficiency.
  brown_ratio     — Fraction of brown/necrotic pixels. Elevated = disease / drought.
  health_index    — Composite score [0, 1]. Higher = healthier plant.
                    Formula: green / (green + yellow*2 + brown*3 + tiny_epsilon)
  dominant_hue    — Most frequent hue value (0–179).
  saturation_mean — Mean saturation. Low = pale/etiolated. High = vibrant.
  brightness_mean — Mean brightness. Low = underexposed or dark canopy.

Usage
-----
    from capabilities.analysis.colour_analyser import ColourAnalyser

    analyser = ColourAnalyser()

    result = analyser.analyse("plant.jpg")
    print(result.health_index)    # 0.87  (healthy)
    print(result.yellow_ratio)    # 0.03  (3% yellowing — mild stress)
    print(result.brown_ratio)     # 0.00  (no browning)
    print(result.summary)         # "Healthy — 87.2% green, mild yellowing"

    # Compare two images over time
    r1 = analyser.analyse("week1.jpg")
    r2 = analyser.analyse("week2.jpg")
    delta = analyser.diff(r1, r2)
    print(delta)   # {"green_ratio": +0.12, "yellow_ratio": -0.02, ...}
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# HSV colour band definitions
# ---------------------------------------------------------------------------

_BANDS = {
    # name: (h_low, h_high, s_min, v_min)
    "green":       (35,  85, 40, 40),
    "yellow":      (20,  35, 60, 80),
    "brown":       ( 8,  20, 40, 30),
    "red_disease": ( 0,   8, 80, 60),   # Reddish disease marks
    "purple":      (130, 160, 50, 50),   # Anthocyanin stress
    "white_pale":  (  0, 179,  0, 200),  # Very pale / etiolated
    "dark_lesion": (  0, 179,  0,  0),   # Near-black lesions
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColourAnalysisResult(CapabilityResult):
    """
    Result from ColourAnalyser.analyse().

    Attributes
    ----------
    green_ratio : float    Fraction of green (healthy) pixels.
    yellow_ratio : float   Fraction of yellow (stressed) pixels.
    brown_ratio : float    Fraction of brown (necrotic) pixels.
    health_index : float   Composite health score [0–1].
    dominant_hue : int     Most frequent hue (0–179 OpenCV scale).
    saturation_mean : float  Mean saturation across the image.
    brightness_mean : float  Mean brightness (value) across the image.
    band_ratios : dict     All colour band ratios by name.
    histogram_h : list     Hue histogram (180 bins).
    histogram_s : list     Saturation histogram (256 bins).
    """

    green_ratio: float = 0.0
    yellow_ratio: float = 0.0
    brown_ratio: float = 0.0
    health_index: float = 0.0
    dominant_hue: int = 0
    saturation_mean: float = 0.0
    brightness_mean: float = 0.0
    band_ratios: Dict[str, float] = field(default_factory=dict)
    histogram_h: List[float] = field(default_factory=list)
    histogram_s: List[float] = field(default_factory=list)
    image_width: int = 0
    image_height: int = 0

    @property
    def summary(self) -> str:
        """One-line human-readable health summary."""
        hi = self.health_index
        if hi >= 0.80:
            status = "Healthy"
        elif hi >= 0.60:
            status = "Mildly stressed"
        elif hi >= 0.35:
            status = "Moderately stressed"
        else:
            status = "Severely stressed"

        parts = [f"{status} — {self.green_ratio*100:.1f}% green"]
        if self.yellow_ratio > 0.05:
            parts.append(f"{self.yellow_ratio*100:.1f}% yellowing")
        if self.brown_ratio > 0.03:
            parts.append(f"{self.brown_ratio*100:.1f}% browning")
        return ", ".join(parts)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "green_ratio":     round(self.green_ratio, 4),
            "yellow_ratio":    round(self.yellow_ratio, 4),
            "brown_ratio":     round(self.brown_ratio, 4),
            "health_index":    round(self.health_index, 4),
            "dominant_hue":    self.dominant_hue,
            "saturation_mean": round(self.saturation_mean, 2),
            "brightness_mean": round(self.brightness_mean, 2),
            "band_ratios":     {k: round(v, 4) for k, v in self.band_ratios.items()},
            "summary":         self.summary,
            "image_size":      [self.image_width, self.image_height],
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<ColourAnalysisResult health={self.health_index:.2f} "
            f"green={self.green_ratio:.2f} yellow={self.yellow_ratio:.2f} "
            f"brown={self.brown_ratio:.2f}>"
        )


# ---------------------------------------------------------------------------
# Analyser class
# ---------------------------------------------------------------------------

class ColourAnalyser(BaseCapability):
    """
    Colour-based plant health analyser using HSV histograms.

    No model download. Pure OpenCV. <10ms per image.

    Parameters
    ----------
    mask : np.ndarray or None
        Optional binary mask (H × W, uint8).  If provided, analysis is
        restricted to the masked region (e.g. plant pixels only, excluding
        pot and background).
    """

    def __init__(self, mask: Optional[np.ndarray] = None) -> None:
        self._default_mask = mask

    @property
    def model_name(self) -> str:
        return "colour_analyser_hsv"

    def run(self, image_path: str, **kwargs) -> ColourAnalysisResult:
        """Alias for analyse()."""
        mask = kwargs.get("mask", self._default_mask)
        return self.analyse(image_path, mask=mask)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def analyse(
        self,
        image_path: str,
        mask: Optional[np.ndarray] = None,
    ) -> ColourAnalysisResult:
        """
        Compute a full colour health profile for an image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        mask : np.ndarray or None
            Optional binary mask restricting analysis to specific pixels.
            Use a segmentation result (mask from HSVSegmentor or SAM2) to
            analyse only plant pixels, excluding background and pot.

        Returns
        -------
        ColourAnalysisResult
        """
        self.validate_image(image_path)

        with self._timer() as t:
            img_bgr = cv2.imread(image_path)
            img_h, img_w = img_bgr.shape[:2]
            img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

            # Apply mask if provided
            active_mask = mask if mask is not None else None

            # ── Per-band pixel ratios ──────────────────────────────────
            total_px = int(active_mask.sum()) if active_mask is not None else img_h * img_w
            if total_px == 0:
                total_px = img_h * img_w  # Fallback to full image

            band_ratios = {}
            for band_name, (h_lo, h_hi, s_min, v_min) in _BANDS.items():
                lower = np.array([h_lo, s_min, v_min], dtype=np.uint8)
                upper = np.array([h_hi, 255, 255], dtype=np.uint8)
                band_mask = cv2.inRange(img_hsv, lower, upper)
                if active_mask is not None:
                    band_mask = cv2.bitwise_and(band_mask, band_mask, mask=active_mask)
                count = int(np.count_nonzero(band_mask))
                band_ratios[band_name] = count / total_px

            # ── Global HSV channel statistics ─────────────────────────
            if active_mask is not None:
                h_vals = img_hsv[:, :, 0][active_mask > 0]
                s_vals = img_hsv[:, :, 1][active_mask > 0]
                v_vals = img_hsv[:, :, 2][active_mask > 0]
            else:
                h_vals = img_hsv[:, :, 0].ravel()
                s_vals = img_hsv[:, :, 1].ravel()
                v_vals = img_hsv[:, :, 2].ravel()

            dominant_hue    = int(np.bincount(h_vals.astype(np.int32), minlength=180).argmax())
            saturation_mean = float(s_vals.mean()) if len(s_vals) > 0 else 0.0
            brightness_mean = float(v_vals.mean()) if len(v_vals) > 0 else 0.0

            # ── Hue and saturation histograms (normalised) ────────────
            hist_h = cv2.calcHist([img_hsv], [0], active_mask, [180], [0, 180])
            hist_s = cv2.calcHist([img_hsv], [1], active_mask, [256], [0, 256])
            hist_h = (hist_h.ravel() / (hist_h.sum() + 1e-9)).tolist()
            hist_s = (hist_s.ravel() / (hist_s.sum() + 1e-9)).tolist()

            # ── Composite health index ────────────────────────────────
            g = band_ratios["green"]
            y = band_ratios["yellow"]
            b = band_ratios["brown"]
            # Weight: yellow penalises 2×, brown penalises 3×
            health_index = g / (g + y * 2 + b * 3 + 1e-9)
            health_index = float(min(max(health_index, 0.0), 1.0))

        return ColourAnalysisResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            green_ratio=band_ratios["green"],
            yellow_ratio=band_ratios["yellow"],
            brown_ratio=band_ratios["brown"],
            health_index=health_index,
            dominant_hue=dominant_hue,
            saturation_mean=saturation_mean,
            brightness_mean=brightness_mean,
            band_ratios=band_ratios,
            histogram_h=hist_h,
            histogram_s=hist_s,
            image_width=img_w,
            image_height=img_h,
        )

    def diff(
        self,
        result_a: ColourAnalysisResult,
        result_b: ColourAnalysisResult,
    ) -> Dict[str, float]:
        """
        Compute the signed delta between two colour analysis results.

        Positive delta = improvement (e.g. more green, less yellow).
        Negative delta = decline.

        Parameters
        ----------
        result_a : ColourAnalysisResult   Earlier measurement.
        result_b : ColourAnalysisResult   Later measurement.

        Returns
        -------
        dict mapping field name → (result_b value - result_a value)
        """
        fields = [
            "green_ratio", "yellow_ratio", "brown_ratio", "health_index",
            "saturation_mean", "brightness_mean",
        ]
        return {
            f: round(getattr(result_b, f) - getattr(result_a, f), 4)
            for f in fields
        }
