"""
sprout_detection/detectors/base_detector.py
============================================
Abstract base class that every sprout detector must implement.

Why an ABC?
-----------
Enforces a consistent interface across all three layers so that:
  1. The cascade pipeline can call any detector identically.
  2. New detectors (e.g. Grounding DINO) can be added by subclassing
     without touching the cascade logic.
  3. Tests can use the ABC as a contract to verify any implementation.

Usage
-----
    from sprout_detection.detectors.base_detector import BaseDetector

    class MyDetector(BaseDetector):
        @property
        def layer_name(self) -> str:
            return "my_detector"

        def detect(self, image_path: str, escalated: bool = False) -> SproutResult:
            ...
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from sprout_detection.result import SproutResult


class BaseDetector(ABC):
    """
    Abstract base class for all sprout detectors.

    Subclasses must implement:
        - layer_name  (property) — short identifier string
        - detect()               — main detection method

    Subclasses may optionally override:
        - validate_image()       — pre-flight image checks
    """

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def layer_name(self) -> str:
        """
        Short identifier for this detector layer.
        Used in SproutResult.method and log output.
        Examples: 'hsv_masking', 'clip_zero_shot', 'gemini_flash_api'
        """

    @abstractmethod
    def detect(self, image_path: str, escalated: bool = False) -> SproutResult:
        """
        Run detection on a single image file.

        Parameters
        ----------
        image_path : str
            Path to a JPEG or PNG image file.
        escalated : bool
            Pass True if a previous cascade layer already ran and returned
            low confidence.  Stored in SproutResult.escalated.

        Returns
        -------
        SproutResult
            Always returns a fully populated SproutResult — never raises
            on a valid image path.  Use validate_image() to catch bad paths
            before calling detect().
        """

    # ------------------------------------------------------------------ #
    # Shared helpers
    # ------------------------------------------------------------------ #

    def validate_image(self, image_path: str) -> None:
        """
        Raise a clear error if the image file cannot be found or has an
        unsupported extension.  Call this at the top of detect().

        Parameters
        ----------
        image_path : str
            Path to validate.

        Raises
        ------
        FileNotFoundError
            If the file does not exist on disk.
        ValueError
            If the file extension is not a supported image format.
        """
        if not os.path.isfile(image_path):
            raise FileNotFoundError(
                f"[{self.layer_name}] Image not found: '{image_path}'"
            )

        supported = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in supported:
            raise ValueError(
                f"[{self.layer_name}] Unsupported image format '{ext}'. "
                f"Supported: {sorted(supported)}"
            )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} layer='{self.layer_name}'>"
