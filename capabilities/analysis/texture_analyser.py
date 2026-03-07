"""
capabilities/analysis/texture_analyser.py
==========================================
Texture Analyser for Disease & Surface Pattern Detection

Analyses image texture using classical signal processing filters to detect
surface anomalies like fungal spots, powdery mildew, rust, and lesions —
without any model download.

Two complementary methods
--------------------------
  LBP (Local Binary Patterns)
    Fast, rotation-invariant texture descriptor. Encodes each pixel by
    comparing it to its circular neighbourhood. Produces a histogram
    signature that distinguishes smooth (healthy) from rough/spotted
    (diseased) surfaces.
    Speed: <5ms per image.

  Gabor Filters
    Bank of oriented frequency filters (like the human visual cortex).
    Captures directional texture at multiple scales — excellent for
    detecting fine patterns like veins, mildew, and regular spot patterns.
    Speed: ~20–50ms per image.

Use cases in plant monitoring
------------------------------
  - Powdery mildew:     Uniform fine white texture → high LBP uniformity
  - Rust / orange pustules: Spotted texture at specific Gabor frequencies
  - Healthy smooth leaf: Low LBP variance, clean Gabor response
  - Senescence / aging:  Increasing LBP entropy over time

Usage
-----
    from capabilities.analysis.texture_analyser import TextureAnalyser

    analyser = TextureAnalyser()

    result = analyser.analyse("leaf.jpg")
    print(result.lbp_entropy)        # Higher = more complex/spotted texture
    print(result.lbp_uniformity)     # Higher = more regular/uniform texture
    print(result.gabor_energy_mean)  # Overall texture energy
    print(result.anomaly_score)      # 0–1, higher = more unusual texture
    print(result.texture_class)      # "smooth", "moderate", or "rough/spotted"
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
class TextureAnalysisResult(CapabilityResult):
    """
    Result from TextureAnalyser.analyse().

    Attributes
    ----------
    lbp_histogram : list of float
        Normalised LBP histogram (59 bins for uniform LBP).
    lbp_entropy : float
        Shannon entropy of the LBP histogram.
        Low = uniform/regular texture. High = complex/random texture.
    lbp_uniformity : float
        Fraction of LBP codes that are "uniform" (at most 2 bit transitions).
        High uniformity = structured patterns like spots or regular mildew.
    gabor_energy_mean : float
        Mean energy response across all Gabor filter orientations and scales.
    gabor_energy_std : float
        Standard deviation of Gabor energy. High = directional texture (veins).
    gabor_responses : dict
        Per-orientation, per-frequency Gabor energy values.
    anomaly_score : float
        Composite anomaly score [0–1].
        0 = perfectly smooth healthy leaf. 1 = highly abnormal texture.
    texture_class : str
        Coarse classification: "smooth", "moderate", or "rough/spotted".
    """

    lbp_histogram: List[float] = field(default_factory=list)
    lbp_entropy: float = 0.0
    lbp_uniformity: float = 0.0
    gabor_energy_mean: float = 0.0
    gabor_energy_std: float = 0.0
    gabor_responses: Dict[str, float] = field(default_factory=dict)
    anomaly_score: float = 0.0
    texture_class: str = ""
    image_width: int = 0
    image_height: int = 0

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "lbp_entropy":       round(self.lbp_entropy, 4),
            "lbp_uniformity":    round(self.lbp_uniformity, 4),
            "gabor_energy_mean": round(self.gabor_energy_mean, 4),
            "gabor_energy_std":  round(self.gabor_energy_std, 4),
            "anomaly_score":     round(self.anomaly_score, 4),
            "texture_class":     self.texture_class,
            "image_size":        [self.image_width, self.image_height],
            "gabor_responses":   {k: round(v, 4) for k, v in self.gabor_responses.items()},
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<TextureAnalysisResult class='{self.texture_class}' "
            f"anomaly={self.anomaly_score:.2f} "
            f"lbp_entropy={self.lbp_entropy:.2f}>"
        )


# ---------------------------------------------------------------------------
# Texture Analyser
# ---------------------------------------------------------------------------

class TextureAnalyser(BaseCapability):
    """
    LBP + Gabor texture analyser for plant surface analysis.

    Parameters
    ----------
    lbp_radius : int
        LBP neighbourhood radius. Default: 3. Larger captures broader patterns.
    lbp_n_points : int
        Number of neighbours for LBP. Convention: 8 × radius. Default: 24.
    gabor_frequencies : list of float
        Spatial frequencies for Gabor filter bank. Default: [0.1, 0.25, 0.4].
    gabor_orientations : int
        Number of evenly-spaced filter orientations (0–π). Default: 4.
    """

    def __init__(
        self,
        lbp_radius: int = 3,
        lbp_n_points: int = 24,
        gabor_frequencies: Optional[List[float]] = None,
        gabor_orientations: int = 4,
    ) -> None:
        self._lbp_radius = lbp_radius
        self._lbp_n_points = lbp_n_points
        self._gabor_frequencies = gabor_frequencies or [0.1, 0.25, 0.4]
        self._gabor_orientations = gabor_orientations
        # Pre-build Gabor filter bank for efficiency
        self._gabor_kernels = self._build_gabor_bank()

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return "texture_lbp_gabor"

    def run(self, image_path: str, **kwargs) -> TextureAnalysisResult:
        """Alias for analyse()."""
        mask = kwargs.get("mask", None)
        return self.analyse(image_path, mask=mask)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def analyse(
        self,
        image_path: str,
        mask: Optional[np.ndarray] = None,
    ) -> TextureAnalysisResult:
        """
        Analyse texture in an image using LBP and Gabor filters.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        mask : np.ndarray or None
            Optional binary mask restricting analysis to plant pixels only.
            Use a segmentation result to exclude background.

        Returns
        -------
        TextureAnalysisResult
        """
        self.validate_image(image_path)

        with self._timer() as t:
            img_bgr = cv2.imread(image_path)
            img_h, img_w = img_bgr.shape[:2]
            # Convert to greyscale — texture is luminance-based
            grey = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

            # ── LBP analysis ───────────────────────────────────────────
            lbp_map = self._compute_lbp(grey)
            lbp_hist, lbp_entropy, lbp_uniformity = self._lbp_stats(lbp_map, mask)

            # ── Gabor analysis ─────────────────────────────────────────
            gabor_responses, gabor_energies = self._apply_gabor_bank(grey, mask)
            gabor_mean = float(np.mean(gabor_energies))
            gabor_std  = float(np.std(gabor_energies))

            # ── Composite anomaly score ────────────────────────────────
            # Entropy [0, log2(59)] → normalise to [0, 1]
            max_entropy = np.log2(self._lbp_n_points + 2 + 1e-9)
            entropy_norm = lbp_entropy / (max_entropy + 1e-9)

            # Gabor energy [0, ~inf] — clamp at 1.0 as rough upper bound
            gabor_norm = min(gabor_mean / 0.1, 1.0)

            anomaly_score = float(0.6 * entropy_norm + 0.4 * gabor_norm)
            anomaly_score = min(anomaly_score, 1.0)

            # ── Coarse texture class ───────────────────────────────────
            if anomaly_score < 0.25:
                texture_class = "smooth"
            elif anomaly_score < 0.55:
                texture_class = "moderate"
            else:
                texture_class = "rough/spotted"

        return TextureAnalysisResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            lbp_histogram=lbp_hist,
            lbp_entropy=lbp_entropy,
            lbp_uniformity=lbp_uniformity,
            gabor_energy_mean=gabor_mean,
            gabor_energy_std=gabor_std,
            gabor_responses=gabor_responses,
            anomaly_score=anomaly_score,
            texture_class=texture_class,
            image_width=img_w,
            image_height=img_h,
        )

    # ------------------------------------------------------------------ #
    # Private — LBP
    # ------------------------------------------------------------------ #

    def _compute_lbp(self, grey: np.ndarray) -> np.ndarray:
        """
        Compute uniform LBP map using circular neighbourhood sampling.

        Each pixel gets a code based on whether its neighbours are
        brighter or darker than the centre pixel.
        """
        h, w = grey.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        r = self._lbp_radius
        n = self._lbp_n_points

        for i in range(n):
            angle = 2 * np.pi * i / n
            dx = r * np.cos(angle)
            dy = -r * np.sin(angle)

            # Bilinear interpolation of neighbour pixel
            x0, y0 = int(np.floor(dx)), int(np.floor(dy))
            wx, wy = dx - x0, dy - y0
            neighbour = np.zeros_like(grey)

            for oy in range(2):
                for ox in range(2):
                    w_coef = ((1 - wx) if ox == 0 else wx) * ((1 - wy) if oy == 0 else wy)
                    shifted = np.roll(np.roll(grey, y0 + oy, axis=0), x0 + ox, axis=1)
                    neighbour += w_coef * shifted

            lbp |= ((neighbour >= grey).astype(np.uint8) << i)

        return lbp

    def _lbp_stats(
        self, lbp_map: np.ndarray, mask: Optional[np.ndarray]
    ):
        """Compute normalised histogram, entropy, and uniformity from LBP map."""
        n_bins = self._lbp_n_points + 2
        flat = lbp_map.ravel() if mask is None else lbp_map[mask > 0]
        hist, _ = np.histogram(flat, bins=n_bins, range=(0, n_bins))
        hist_norm = hist / (hist.sum() + 1e-9)

        # Shannon entropy
        nonzero = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(nonzero * np.log2(nonzero)))

        # Uniformity: proportion of codes with ≤ 2 bit transitions
        # (These correspond to smooth edges and spots)
        uniform_count = sum(1 for v in flat if bin(int(v)).count('1') <= 2)
        uniformity = float(uniform_count / (len(flat) + 1e-9))

        return hist_norm.tolist(), entropy, uniformity

    # ------------------------------------------------------------------ #
    # Private — Gabor
    # ------------------------------------------------------------------ #

    def _build_gabor_bank(self):
        """Pre-build a bank of Gabor kernels at all orientations × frequencies."""
        kernels = {}
        for freq in self._gabor_frequencies:
            for i in range(self._gabor_orientations):
                theta = np.pi * i / self._gabor_orientations
                key = f"f{freq:.2f}_theta{i}"
                kernels[key] = cv2.getGaborKernel(
                    ksize=(21, 21),
                    sigma=4.0,
                    theta=theta,
                    lambd=1.0 / (freq + 1e-9),
                    gamma=0.5,
                    psi=0,
                    ktype=cv2.CV_32F,
                )
        return kernels

    def _apply_gabor_bank(
        self, grey: np.ndarray, mask: Optional[np.ndarray]
    ):
        """Apply all Gabor kernels and return per-kernel energy stats."""
        responses = {}
        energies = []

        grey_uint8 = (grey * 255).astype(np.uint8)

        for key, kernel in self._gabor_kernels.items():
            filtered = cv2.filter2D(grey_uint8, cv2.CV_32F, kernel)
            if mask is not None:
                region = filtered[mask > 0]
            else:
                region = filtered.ravel()

            energy = float(np.mean(np.abs(region))) if len(region) > 0 else 0.0
            responses[key] = energy
            energies.append(energy)

        return responses, energies
