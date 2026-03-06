"""
sprout_detection/detectors/gemini_detector.py
==============================================
Layer 3 — Gemini Flash Vision API Fallback

This is the last resort in the cascade.  It is only invoked when both
Layer 1 (HSV) and Layer 2 (CLIP) return confidence below the threshold.

Algorithm overview
------------------
1. Load and base64-encode the image.
2. Send it to the Gemini Flash API with a structured prompt that requests
   a JSON response containing:
     - sprout_detected : bool
     - confidence      : float (0.0–1.0)
     - reasoning       : str  (one-sentence explanation)
3. Parse the JSON response into a SproutResult.

Why Gemini Flash?
-----------------
- Handles any lighting, angle, or environmental condition.
- No GPU required; runs on any machine with internet.
- Cost-effective: ~$0.001 per image (as of 2025).
- Returns a human-readable reasoning sentence for auditability.

Cost management
---------------
In typical controlled environments, 90 %+ of images resolve at Layer 1
or Layer 2 with zero API cost.  Gemini is only called for genuinely
ambiguous images.

API key setup
-------------
Set the environment variable before running:

    export GEMINI_API_KEY=your-key-here

Or at runtime:

    from config import CONFIG
    CONFIG["gemini_api_key"] = "your-key-here"

If GEMINI_API_KEY is not set, the detector raises a clear ConfigurationError
rather than silently failing.

Requirements
------------
    pip install google-generativeai>=0.4

Usage
-----
    from sprout_detection.detectors.gemini_detector import GeminiDetector
    from config import CONFIG

    detector = GeminiDetector(config=CONFIG)
    result = detector.detect("ambiguous_photo.jpg")
    print(result)
    # 🌱 SproutResult | detected=True | confidence=0.95 | method=gemini_flash_api
    print(result.reasoning)
    # "A small green cotyledon is clearly emerging from the soil surface."
"""

from __future__ import annotations

import json
import os

from config import CONFIG
from sprout_detection.detectors.base_detector import BaseDetector
from sprout_detection.result import SproutResult


# Structured prompt sent with every image.
# Requesting JSON-only output prevents preamble text that breaks parsing.
_DETECTION_PROMPT = """You are a plant growth monitoring system.

Analyse the image and determine whether a plant sprout is visibly emerging
from the soil.

Respond ONLY with valid JSON — no markdown fences, no preamble, no explanation
outside the JSON object:

{
  "sprout_detected": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "One sentence explaining your decision based on visual evidence."
}

Guidelines:
- sprout_detected = true  if any green plant tissue is breaking through the soil.
- sprout_detected = false if only bare soil, mulch, or substrate is visible.
- confidence should reflect how clearly you can see the sprout (or lack thereof).
  Use 0.95+ only when the evidence is unambiguous."""

# Mapping of file extensions to MIME types accepted by the Gemini API
_MIME_MAP: dict = {
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png":  "image/png",
    ".webp": "image/webp",
    ".bmp":  "image/bmp",
}


class GeminiDetector(BaseDetector):
    """
    Sprout detector using the Gemini Flash Vision API (Layer 3 fallback).

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  Defaults to the global CONFIG.
        Relevant keys:
          - gemini_api_key : str  (required — can also be set via env var)
          - gemini_model   : str  (default: 'gemini-2.0-flash')
    """

    def __init__(self, config: dict = None) -> None:
        self._config = config or CONFIG

    # ------------------------------------------------------------------ #
    # BaseDetector interface
    # ------------------------------------------------------------------ #

    @property
    def layer_name(self) -> str:
        return "gemini_flash_api"

    def detect(self, image_path: str, escalated: bool = True) -> SproutResult:
        """
        Run Gemini Flash API detection on a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        escalated : bool
            Defaults to True because this layer is almost always reached
            via escalation from Layer 1 and Layer 2.

        Returns
        -------
        SproutResult
            Populated from the JSON response returned by Gemini.

        Raises
        ------
        ConfigurationError
            If GEMINI_API_KEY is not configured.
        APIError
            If the Gemini API returns an unexpected response.
        """
        self.validate_image(image_path)
        self._validate_api_key()

        import google.generativeai as genai  # type: ignore

        # ── Configure Gemini client ─────────────────────────────────────
        genai.configure(api_key=self._config["gemini_api_key"])
        model = genai.GenerativeModel(
            self._config.get("gemini_model", "gemini-2.0-flash")
        )

        # ── Load image ──────────────────────────────────────────────────
        image_bytes, mime_type = self._load_image_bytes(image_path)

        # ── Call API ────────────────────────────────────────────────────
        response = model.generate_content([
            {"mime_type": mime_type, "data": image_bytes},
            _DETECTION_PROMPT,
        ])

        # ── Parse response ──────────────────────────────────────────────
        parsed = self._parse_response(response.text, image_path)

        return SproutResult(
            sprout_detected=bool(parsed["sprout_detected"]),
            confidence=float(parsed["confidence"]),
            method=self.layer_name,
            reasoning=parsed.get("reasoning", "No reasoning provided."),
            image_path=image_path,
            escalated=escalated,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _validate_api_key(self) -> None:
        """
        Raise a clear error if the Gemini API key is missing.

        Checks both the config dict and the GEMINI_API_KEY environment
        variable as a fallback.
        """
        key = self._config.get("gemini_api_key", "") or os.environ.get(
            "GEMINI_API_KEY", ""
        )
        if not key:
            raise EnvironmentError(
                "[GeminiDetector] GEMINI_API_KEY is not set.\n"
                "  Option A: export GEMINI_API_KEY=your-key-here\n"
                "  Option B: from config import CONFIG; "
                "CONFIG['gemini_api_key'] = 'your-key-here'"
            )
        # Write back in case it was found only in the environment
        self._config["gemini_api_key"] = key

    @staticmethod
    def _load_image_bytes(image_path: str) -> tuple[bytes, str]:
        """
        Load image file as raw bytes and determine its MIME type.

        Parameters
        ----------
        image_path : str
            Path to the image.

        Returns
        -------
        (bytes, str)
            Raw file bytes and MIME type string.
        """
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = _MIME_MAP.get(ext, "image/jpeg")
        with open(image_path, "rb") as f:
            return f.read(), mime_type

    @staticmethod
    def _parse_response(raw_text: str, image_path: str) -> dict:
        """
        Parse the Gemini API text response into a Python dict.

        Strips any accidental markdown code fences that the model might
        include despite the explicit instruction not to.

        Parameters
        ----------
        raw_text : str
            Raw text from response.text.
        image_path : str
            Used only in error messages.

        Returns
        -------
        dict
            Parsed JSON with keys: sprout_detected, confidence, reasoning.

        Raises
        ------
        ValueError
            If the response cannot be parsed as valid JSON.
        """
        # Strip markdown code fences defensively
        cleaned = (
            raw_text.strip()
            .replace("```json", "")
            .replace("```", "")
            .strip()
        )
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"[GeminiDetector] Could not parse API response for '{image_path}'.\n"
                f"  Raw response: {raw_text[:300]}\n"
                f"  Error: {exc}"
            ) from exc

        # Validate required fields
        for key in ("sprout_detected", "confidence"):
            if key not in data:
                raise ValueError(
                    f"[GeminiDetector] API response missing required field '{key}'. "
                    f"Full response: {data}"
                )

        # Clamp confidence to valid range
        data["confidence"] = max(0.0, min(1.0, float(data["confidence"])))

        return data
