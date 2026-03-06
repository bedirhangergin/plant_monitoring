"""
sprout_detection/detectors
==========================
Individual detector modules for each cascade layer.

Each detector is a self-contained class that:
  - Accepts a single image path
  - Returns a SproutResult
  - Has no knowledge of other detectors or cascade logic

Imports
-------
    from sprout_detection.detectors.hsv_detector    import HSVDetector
    from sprout_detection.detectors.clip_detector   import CLIPDetector
    from sprout_detection.detectors.gemini_detector import GeminiDetector
"""
