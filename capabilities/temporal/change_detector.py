"""
capabilities/temporal/change_detector.py
==========================================
Frame Change Detector — Motion and Visual Difference Analysis

Detects and quantifies visual changes between two consecutive images.
Pure OpenCV. No model download. <20ms per pair.

Use cases in plant monitoring
------------------------------
  - Wilting detection: large structural change overnight
  - Growth confirmation: appearance of new green regions
  - Pest activity: local change in unexpected regions
  - Camera movement: global uniform change (normalise it out)
  - Leaf drop: sudden disappearance of covered area

Three change signals computed
------------------------------
  pixel_diff_ratio    — Fraction of pixels that changed beyond threshold.
                        Simple and fast. Good for detecting gross changes.

  structural_diff     — SSIM-based structural difference.
                        Captures perceptual change better than raw pixels.

  colour_shift        — Change in HSV colour distribution.
                        Detects yellowing/browning even without movement.

Usage
-----
    from capabilities.temporal.change_detector import ChangeDetector

    detector = ChangeDetector()

    result = detector.compare("day1.jpg", "day2.jpg")
    print(result.pixel_diff_ratio)    # 0.08   (8% of pixels changed)
    print(result.structural_diff)     # 0.12   (12% structural change)
    print(result.colour_shift)        # 0.05   (5% colour distribution shift)
    print(result.change_magnitude)    # 0.09   (composite score)
    print(result.change_type)         # "growth"  or "wilting"  or "minor"
    print(result.diff_map.shape)      # (H, W)  spatial change heatmap
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ChangeResult(CapabilityResult):
    """
    Result from ChangeDetector.compare().

    Attributes
    ----------
    image_path_a : str   Earlier image path.
    image_path_b : str   Later image path.
    pixel_diff_ratio : float
        Fraction of pixels that changed (0.0–1.0).
    structural_diff : float
        SSIM-based structural change (0.0 = identical, 1.0 = completely different).
    colour_shift : float
        Change in HSV colour distribution between images (0.0–1.0).
    change_magnitude : float
        Composite change score [0–1].
    change_type : str
        Coarse label: "none", "minor", "growth", "wilting", or "major".
    green_delta : float
        Change in green pixel ratio. Positive = more green (growth).
    diff_map : np.ndarray or None
        Spatial (H × W) heatmap of change intensity.
    """

    image_path_a: str = ""
    image_path_b: str = ""
    pixel_diff_ratio: float = 0.0
    structural_diff: float = 0.0
    colour_shift: float = 0.0
    change_magnitude: float = 0.0
    change_type: str = ""
    green_delta: float = 0.0
    diff_map: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "image_path_a":     self.image_path_a,
            "image_path_b":     self.image_path_b,
            "pixel_diff_ratio": round(self.pixel_diff_ratio, 4),
            "structural_diff":  round(self.structural_diff, 4),
            "colour_shift":     round(self.colour_shift, 4),
            "change_magnitude": round(self.change_magnitude, 4),
            "change_type":      self.change_type,
            "green_delta":      round(self.green_delta, 4),
        })
        return base

    def __repr__(self) -> str:
        return (
            f"<ChangeResult type='{self.change_type}' "
            f"magnitude={self.change_magnitude:.3f} "
            f"green_delta={self.green_delta:+.3f}>"
        )


# ---------------------------------------------------------------------------
# Change Detector
# ---------------------------------------------------------------------------

class ChangeDetector(BaseCapability):
    """
    Visual change detector between two consecutive plant images.

    Pure OpenCV, no model download, <20ms per pair.

    Parameters
    ----------
    pixel_diff_threshold : int
        Pixel-level change threshold (0–255 greyscale). Default: 25.
        Pixels that change by more than this are counted as "changed".
    resize_to : tuple or None
        Resize both images to this size before comparison.
        None = use original size (both must match). Default: (512, 512).
    """

    def __init__(
        self,
        pixel_diff_threshold: int = 25,
        resize_to: Optional[tuple] = (512, 512),
    ) -> None:
        self._pixel_threshold = pixel_diff_threshold
        self._resize_to = resize_to

    @property
    def model_name(self) -> str:
        return "change_detector_cv"

    def run(self, image_path: str, **kwargs) -> ChangeResult:
        """Alias for compare(). Pass image_path_b= as keyword argument."""
        b = kwargs.get("image_path_b")
        if b is None:
            raise ValueError("Pass image_path_b= when calling run().")
        return self.compare(image_path, b)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def compare(
        self,
        image_path_a: str,
        image_path_b: str,
    ) -> ChangeResult:
        """
        Detect and quantify changes between two images.

        Parameters
        ----------
        image_path_a : str   Earlier image.
        image_path_b : str   Later image (must be same crop/framing).

        Returns
        -------
        ChangeResult
        """
        self.validate_image(image_path_a)
        self.validate_image(image_path_b)

        with self._timer() as t:
            img_a = self._load(image_path_a)
            img_b = self._load(image_path_b)

            # ── Pixel-level difference ─────────────────────────────────
            grey_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY).astype(np.float32)
            grey_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY).astype(np.float32)
            abs_diff = np.abs(grey_b - grey_a)
            diff_map = abs_diff.astype(np.uint8)

            pixel_diff_ratio = float(
                np.count_nonzero(abs_diff > self._pixel_threshold) / abs_diff.size
            )

            # ── Structural similarity (SSIM) ───────────────────────────
            structural_diff = self._compute_ssim_diff(grey_a, grey_b)

            # ── Colour distribution shift ─────────────────────────────
            colour_shift = self._colour_histogram_distance(img_a, img_b)

            # ── Green delta ───────────────────────────────────────────
            green_a = self._green_ratio(img_a)
            green_b = self._green_ratio(img_b)
            green_delta = green_b - green_a

            # ── Composite score ───────────────────────────────────────
            change_magnitude = float(
                0.40 * pixel_diff_ratio +
                0.35 * structural_diff +
                0.25 * colour_shift
            )

            # ── Change type classification ────────────────────────────
            change_type = self._classify_change(change_magnitude, green_delta)

        return ChangeResult(
            image_path=image_path_a,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            image_path_a=image_path_a,
            image_path_b=image_path_b,
            pixel_diff_ratio=pixel_diff_ratio,
            structural_diff=structural_diff,
            colour_shift=colour_shift,
            change_magnitude=change_magnitude,
            change_type=change_type,
            green_delta=green_delta,
            diff_map=diff_map,
        )

    def compare_series(
        self,
        image_paths: list,
    ) -> list:
        """
        Compare consecutive pairs in a time-ordered list.

        Returns
        -------
        list of ChangeResult
            Length = len(image_paths) - 1.
        """
        results = []
        for i in range(len(image_paths) - 1):
            try:
                results.append(self.compare(image_paths[i], image_paths[i+1]))
            except Exception as e:
                print(f"[ChangeDetector] Skipped pair {i}→{i+1}: {e}")
        return results

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _load(self, path: str) -> np.ndarray:
        """Load and optionally resize an image."""
        img = cv2.imread(path)
        if self._resize_to:
            img = cv2.resize(img, self._resize_to, interpolation=cv2.INTER_AREA)
        return img

    @staticmethod
    def _compute_ssim_diff(a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute SSIM-based structural difference.
        Returns 0.0 (identical) to 1.0 (completely different).
        """
        # Manual SSIM without scikit-image dependency
        C1, C2 = 6.5025, 58.5225  # (0.01*255)^2, (0.03*255)^2
        mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
        mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
        mu_a2 = mu_a * mu_a
        mu_b2 = mu_b * mu_b
        mu_ab = mu_a * mu_b
        sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
        sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
        sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab

        ssim_map = (
            (2 * mu_ab + C1) * (2 * sigma_ab + C2) /
            ((mu_a2 + mu_b2 + C1) * (sigma_a2 + sigma_b2 + C2))
        )
        ssim_score = float(ssim_map.mean())
        # SSIM ∈ [-1, 1], convert to difference [0, 1]
        return float((1.0 - ssim_score) / 2.0)

    @staticmethod
    def _colour_histogram_distance(img_a: np.ndarray, img_b: np.ndarray) -> float:
        """Bhattacharyya distance between HSV colour histograms."""
        hsv_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2HSV)
        hsv_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2HSV)
        total = 0.0
        for ch in range(3):
            h_a = cv2.calcHist([hsv_a], [ch], None, [64], [0, 256])
            h_b = cv2.calcHist([hsv_b], [ch], None, [64], [0, 256])
            cv2.normalize(h_a, h_a)
            cv2.normalize(h_b, h_b)
            dist = cv2.compareHist(h_a, h_b, cv2.HISTCMP_BHATTACHARYYA)
            total += dist
        return float(min(total / 3.0, 1.0))

    @staticmethod
    def _green_ratio(img_bgr: np.ndarray) -> float:
        """Fraction of green pixels in an image."""
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([35, 40, 40]), np.array([85, 255, 255]))
        return float(np.count_nonzero(mask)) / mask.size

    @staticmethod
    def _classify_change(magnitude: float, green_delta: float) -> str:
        """Assign a coarse change type label."""
        if magnitude < 0.03:
            return "none"
        if magnitude < 0.10:
            if green_delta > 0.02:
                return "growth"
            if green_delta < -0.02:
                return "wilting"
            return "minor"
        if green_delta > 0.05:
            return "growth"
        if green_delta < -0.05:
            return "wilting"
        return "major"
