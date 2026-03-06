"""
sprout_detection/cascade.py
============================
Cascade Orchestrator — Use Case A: Sprout Detection

This module is the heart of the sprout detection pipeline.  It wires
together the three detector layers and implements the escalation logic:

    Layer 1 (HSV)   → free, <10ms
        ↓ if confidence < threshold
    Layer 2 (CLIP)  → free, 1–3s CPU
        ↓ if confidence < threshold
    Layer 3 (Gemini API) → ~$0.001/call, 1–2s

The pipeline exits as early as possible — the moment a layer returns
confidence >= threshold, that result is returned without running further layers.

Design principles
-----------------
- The SproutCascade.analyze() method has a single, stable signature.
  Whether the input is a manually taken photo or a frame extracted from
  a video feed, the same code path runs unchanged.
- Detectors are injected at construction time, making the cascade fully
  testable with mock detectors.
- All results are logged to a JSONL file automatically.

Usage
-----
    from sprout_detection.cascade import SproutCascade
    from config import CONFIG

    # Create with default detectors
    cascade = SproutCascade(config=CONFIG)

    # Analyse a single image
    result = cascade.analyze("photo.jpg")
    print(result)

    # Analyse a batch
    results = cascade.analyze_batch(["frame_001.jpg", "frame_002.jpg"])

    # Print log summary
    cascade.print_log_summary()

Video workflow (identical pipeline, zero changes)
-----------------
    from sprout_detection.utils.video import extract_frames

    frames = extract_frames("timelapse.mp4", every_n_seconds=30)
    for frame_path in frames:
        result = cascade.analyze(frame_path)   # Same call as for photos
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import List, Optional

from config import CONFIG
from sprout_detection.detectors.base_detector import BaseDetector
from sprout_detection.detectors.clip_detector import CLIPDetector
from sprout_detection.detectors.gemini_detector import GeminiDetector
from sprout_detection.detectors.hsv_detector import HSVDetector
from sprout_detection.result import SproutResult


class SproutCascade:
    """
    Three-layer confidence cascade for sprout detection.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  Defaults to the global CONFIG.
    layer1 : BaseDetector, optional
        Override the Layer 1 detector.  Defaults to HSVDetector.
    layer2 : BaseDetector, optional
        Override the Layer 2 detector.  Defaults to CLIPDetector.
    layer3 : BaseDetector, optional
        Override the Layer 3 detector.  Defaults to GeminiDetector.
    log_path : str, optional
        Path to the JSONL log file.  Defaults to CONFIG["log_file"].
    verbose : bool
        Print progress messages to stdout.  Default True.
    """

    def __init__(
        self,
        config: dict = None,
        layer1: Optional[BaseDetector] = None,
        layer2: Optional[BaseDetector] = None,
        layer3: Optional[BaseDetector] = None,
        log_path: Optional[str] = None,
        verbose: bool = True,
    ) -> None:
        self._config = config or CONFIG
        self._threshold = self._config["confidence_threshold"]
        self._verbose = verbose

        # Instantiate default detectors if not injected
        self._layer1: BaseDetector = layer1 or HSVDetector(config=self._config)
        self._layer2: BaseDetector = layer2 or CLIPDetector(config=self._config)
        self._layer3: BaseDetector = layer3 or GeminiDetector(config=self._config)

        # Set up logging
        self._log_path = log_path or self._config.get("log_file", "sprout_log.jsonl")
        os.makedirs(os.path.dirname(self._log_path) or ".", exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def analyze(self, image_path: str) -> SproutResult:
        """
        Run the cascade pipeline on a single image.

        This method works identically for manually taken photos and for
        video frames extracted by extract_frames() — no code changes needed
        when switching from photo to video input.

        Parameters
        ----------
        image_path : str
            Path to a JPEG or PNG image file.

        Returns
        -------
        SproutResult
            The result from whichever layer resolved the detection.
            result.method tells you which layer was used.
            result.escalated is True if Layer 1 ran but was insufficient.
        """
        self._log_message(f"\n🔍 Analysing: {os.path.basename(image_path)}")
        self._log_message(f"   Cascade threshold: {self._threshold}")

        # ── Layer 1: HSV Masking ────────────────────────────────────────
        self._log_message("   Layer 1 → HSV Masking...")
        result = self._layer1.detect(image_path, escalated=False)
        self._log_message(
            f"             confidence={result.confidence:.3f}  "
            f"green_ratio={result.green_ratio:.4f}"
        )

        if result.confidence >= self._threshold:
            self._log_message(
                f"   ✅ Resolved at Layer 1 (confidence={result.confidence:.3f})"
            )
            self._write_log(result)
            return result

        # ── Layer 2: CLIP Zero-Shot ─────────────────────────────────────
        self._log_message(
            f"   ⚠️  Layer 1 confidence {result.confidence:.3f} < {self._threshold}"
            f" → escalating to Layer 2 (CLIP)"
        )
        result = self._layer2.detect(image_path, escalated=True)
        self._log_message(
            f"   Layer 2 → CLIP  confidence={result.confidence:.3f}"
        )

        if result.confidence >= self._threshold:
            self._log_message(
                f"   ✅ Resolved at Layer 2 (confidence={result.confidence:.3f})"
            )
            self._write_log(result)
            return result

        # ── Layer 3: Gemini API ─────────────────────────────────────────
        api_key = self._config.get("gemini_api_key", "")
        if not api_key:
            self._log_message(
                f"   ⚠️  Layer 2 confidence {result.confidence:.3f} < {self._threshold}"
                f" → Layer 3 unavailable (GEMINI_API_KEY not set). "
                f"Returning Layer 2 result."
            )
            self._write_log(result)
            return result

        self._log_message(
            f"   ⚠️  Layer 2 confidence {result.confidence:.3f} < {self._threshold}"
            f" → escalating to Layer 3 (Gemini API, ~$0.001)"
        )
        result = self._layer3.detect(image_path, escalated=True)
        self._log_message(
            f"   ✅ Resolved at Layer 3 (confidence={result.confidence:.3f})"
        )

        self._write_log(result)
        return result

    def analyze_batch(
        self,
        image_paths: List[str],
        stop_on_error: bool = False,
    ) -> List[SproutResult]:
        """
        Run the cascade on a list of image paths.

        Parameters
        ----------
        image_paths : list of str
            Paths to image files.  Typically the output of extract_frames().
        stop_on_error : bool
            If True, raise exceptions immediately.
            If False (default), log the error and continue to the next image.

        Returns
        -------
        list of SproutResult
            One result per successfully processed image.
            Failed images are skipped (error printed to stdout).
        """
        results = []
        total = len(image_paths)

        for idx, path in enumerate(image_paths, start=1):
            self._log_message(f"\n[{idx}/{total}] Processing: {path}")
            try:
                result = self.analyze(path)
                results.append(result)
            except Exception as exc:  # pylint: disable=broad-except
                if stop_on_error:
                    raise
                print(f"   ❌ Error on '{path}': {exc}")

        return results

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def _write_log(self, result: SproutResult) -> None:
        """
        Append a result to the JSONL log file.

        Each line in the log file is a complete, self-contained JSON object.
        This makes it easy to stream, parse, or process with tools like jq.
        """
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result.to_dict()) + "\n")
        except OSError as exc:
            print(f"[SproutCascade] Warning: could not write to log: {exc}")

    def print_log_summary(self) -> None:
        """
        Print a formatted table of all logged results.

        Reads the JSONL log file and displays each entry as a table row.
        """
        if not os.path.isfile(self._log_path):
            print("No log entries found.")
            return

        records = []
        with open(self._log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))

        if not records:
            print("Log file exists but contains no entries.")
            return

        # Table header
        print(
            f"\n{'Timestamp':<22} {'Method':<22} {'Detected':<10} "
            f"{'Confidence':<12} {'Esc':<5} Image"
        )
        print("─" * 95)

        for r in records:
            icon = "🌱" if r["sprout_detected"] else "🪨"
            ts = r["timestamp"][:19]
            esc = "✓" if r.get("escalated") else " "
            fname = os.path.basename(r.get("image_path", ""))
            print(
                f"{ts:<22} {r['method']:<22} {icon} {str(r['sprout_detected']):<8} "
                f"{r['confidence']:<12.3f} {esc:<5} {fname}"
            )

        print(f"\nTotal entries: {len(records)}")

    def clear_log(self) -> None:
        """Delete all log entries by truncating the log file."""
        if os.path.isfile(self._log_path):
            open(self._log_path, "w").close()
            print(f"[SproutCascade] Log cleared: {self._log_path}")

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _log_message(self, msg: str) -> None:
        """Print a message only when verbose mode is enabled."""
        if self._verbose:
            print(msg)
