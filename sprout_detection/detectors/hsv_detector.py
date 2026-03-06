"""
sprout_detection/detectors/hsv_detector.py
==========================================
Layer 1 — Classical HSV Colour Masking

Algorithm overview
------------------
1. Load the image with OpenCV (BGR colour space).
2. Convert to HSV (Hue-Saturation-Value) colour space.
   HSV separates colour (hue) from brightness, making green detection
   far more robust than working in raw BGR.
3. Create a binary mask: pixels inside the configured green HSV range → 255,
   everything else → 0.
4. Apply morphological OPEN then CLOSE operations to remove salt-and-pepper
   noise and fill small holes inside detected green regions.
5. Compute green_ratio = green_pixels / total_pixels.
6. Compare against the configured threshold:
     green_ratio >= threshold  →  sprout_detected = True
7. Derive confidence from how far the ratio is from the threshold boundary.
   A ratio far above or far below the threshold means high certainty.
   A ratio close to the boundary means low certainty → cascade escalation.

Confidence formula
------------------
    distance   = |green_ratio - threshold|
    normalised = min(distance / (threshold × 5), 1.0)
    confidence = 0.50 + normalised × 0.45   ∈ [0.50, 0.95]

The floor of 0.50 ensures this layer always signals that it has *some*
opinion (it never returns 0.0 as if it didn't run).

Best for
--------
- Controlled indoor or greenhouse environments with stable lighting.
- High-volume batch processing where speed is critical.
- First-pass filter before invoking heavier models.

Limitations
-----------
- Sensitive to changing or outdoor lighting conditions.
- HSV thresholds may need per-environment calibration.
- Cannot distinguish a green painted surface from a real sprout.

Usage
-----
    from sprout_detection.detectors.hsv_detector import HSVDetector
    from config import CONFIG

    detector = HSVDetector(config=CONFIG)
    result = detector.detect("photo.jpg")

    print(result)
    # 🌱 SproutResult | detected=True | confidence=0.87 | method=hsv_masking

    # Access the raw binary mask for visualisation:
    mask = result.hsv_mask
"""

from __future__ import annotations

import cv2
import numpy as np

from config import CONFIG
from sprout_detection.detectors.base_detector import BaseDetector
from sprout_detection.result import SproutResult


class HSVDetector(BaseDetector):
    """
    Sprout detector using classical HSV colour masking (Layer 1).

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  Defaults to the global CONFIG.
        Relevant keys:
          - hsv_lower              : tuple (H, S, V) lower bound
          - hsv_upper              : tuple (H, S, V) upper bound
          - hsv_green_ratio_threshold : float, min green fraction
          - hsv_morph_kernel_size  : int, morphological kernel side length
          - confidence_threshold   : float, cascade exit threshold
    """

    def __init__(self, config: dict = None) -> None:
        self._config = config or CONFIG

        # Cache numpy arrays to avoid repeated allocation on every call
        self._lower = np.array(self._config["hsv_lower"], dtype=np.uint8)
        self._upper = np.array(self._config["hsv_upper"], dtype=np.uint8)

        # Elliptical kernel gives smoother morphological results than a square
        k = self._config.get("hsv_morph_kernel_size", 5)
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))

    # ------------------------------------------------------------------ #
    # BaseDetector interface
    # ------------------------------------------------------------------ #

    @property
    def layer_name(self) -> str:
        return "hsv_masking"

    def detect(self, image_path: str, escalated: bool = False) -> SproutResult:
        """
        Run HSV masking detection on a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        escalated : bool
            True if a previous cascade layer already ran.

        Returns
        -------
        SproutResult
            Includes .hsv_mask (numpy array) for optional visualisation.
        """
        self.validate_image(image_path)

        # ── Step 1: Load image ──────────────────────────────────────────
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            raise IOError(
                f"[{self.layer_name}] OpenCV failed to decode image: '{image_path}'"
            )

        # ── Step 2: BGR → HSV ───────────────────────────────────────────
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        # ── Step 3: Green pixel mask ────────────────────────────────────
        # cv2.inRange sets pixel to 255 if all channels are within range
        mask = cv2.inRange(img_hsv, self._lower, self._upper)

        # ── Step 4: Morphological noise removal ─────────────────────────
        # OPEN  = erode then dilate → removes isolated bright specks
        # CLOSE = dilate then erode → fills small dark holes inside blobs
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        # ── Step 5: Green ratio ─────────────────────────────────────────
        total_pixels = mask.size                      # height × width
        green_pixels = int(np.count_nonzero(mask))    # pixels == 255
        green_ratio = green_pixels / total_pixels

        # ── Step 6: Detection decision ──────────────────────────────────
        threshold = self._config["hsv_green_ratio_threshold"]
        sprout_detected = green_ratio >= threshold

        # ── Step 7: Confidence from boundary distance ───────────────────
        confidence = self._compute_confidence(green_ratio, threshold)

        # ── Build result ────────────────────────────────────────────────
        reasoning = (
            f"Green pixel ratio = {green_ratio:.4f} "
            f"({'above' if sprout_detected else 'below'} threshold {threshold}). "
            f"{'Sprout detected.' if sprout_detected else 'No sprout.'}"
        )

        return SproutResult(
            sprout_detected=sprout_detected,
            confidence=confidence,
            method=self.layer_name,
            reasoning=reasoning,
            image_path=image_path,
            escalated=escalated,
            green_ratio=green_ratio,
            hsv_mask=mask,  # Stored for visualisation; excluded from serialisation
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _compute_confidence(green_ratio: float, threshold: float) -> float:
        """
        Map the distance between green_ratio and threshold to a confidence
        score in the range [0.50, 0.95].

        The formula is linear:
          - At the threshold (distance = 0)     → confidence = 0.50
          - At 5× threshold distance            → confidence = 0.95
          - Beyond 5× threshold distance        → capped at 0.95

        This means the cascade will always escalate when the green ratio is
        very close to the decision boundary, regardless of which side it's on.

        Parameters
        ----------
        green_ratio : float
            Observed fraction of green pixels.
        threshold : float
            Decision boundary value from config.

        Returns
        -------
        float
            Confidence in [0.50, 0.95].
        """
        # Guard against zero threshold (would cause division by zero)
        if threshold <= 0:
            return 0.95 if green_ratio > 0 else 0.50

        distance = abs(green_ratio - threshold)
        # Normalise: 5× the threshold is treated as "maximum certainty"
        normalised = min(distance / (threshold * 5.0), 1.0)
        return 0.50 + normalised * 0.45

    # ------------------------------------------------------------------ #
    # Standalone utility: get mask without a full SproutResult
    # ------------------------------------------------------------------ #

    def get_mask(self, image_path: str) -> np.ndarray:
        """
        Return only the binary green mask for a given image.

        Useful for debugging or displaying masks without running the full
        cascade.  Does NOT return a SproutResult.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        np.ndarray
            Binary mask, same height × width as the input image.
            Pixel value 255 = green (potential sprout), 0 = background.
        """
        result = self.detect(image_path)
        return result.hsv_mask
