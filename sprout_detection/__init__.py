"""
sprout_detection
================
Package for Use Case A — Sprout Detection.

Public API
----------
Import the cascade pipeline directly:

    from sprout_detection import SproutCascade

Or import individual detectors for fine-grained control / testing:

    from sprout_detection.detectors.hsv_detector   import HSVDetector
    from sprout_detection.detectors.clip_detector  import CLIPDetector
    from sprout_detection.detectors.gemini_detector import GeminiDetector
"""

from sprout_detection.cascade import SproutCascade
from sprout_detection.result import SproutResult

__all__ = ["SproutCascade", "SproutResult"]
