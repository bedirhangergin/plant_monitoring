"""
capabilities/
=============
Reusable, model-backed capability classes for plant vision analysis.

Every class in this package is completely independent — it has no knowledge
of sprout detection, the cascade, or any other use case.  Use cases import
whichever capabilities they need and compose them freely.

Package layout
--------------
capabilities/
  classification/   — zero-shot label prediction (CLIP, VLM)
  detection/        — object localisation with bounding boxes (DINO, OWL-ViT, YOLO)
  segmentation/     — pixel-level masks (SAM2, FastSAM, HSV)
  depth/            — monocular depth & height estimation (Depth Anything, MiDaS)
  analysis/         — colour health, texture, coverage, anomaly (OpenCV + DINOv2)
  temporal/         — growth tracking and change detection across frames

Shared contracts
----------------
Every capability class:
  - Accepts image_path: str as its primary input
  - Returns a typed *Result dataclass (e.g. ClassificationResult)
  - Validates the image path before doing any work
  - Loads heavy models lazily on first call
  - Is independently testable with no other capability loaded

Quick usage example
-------------------
    from capabilities.classification.clip_classifier import CLIPClassifier
    from capabilities.detection.grounding_dino import GroundingDINODetector
    from capabilities.segmentation.sam2_segmentor import SAM2Segmentor
    from capabilities.analysis.colour_analyser import ColourAnalyser
    from capabilities.depth.depth_anything import DepthEstimator

    # Each class is completely independent
    classifier = CLIPClassifier()
    result = classifier.classify("leaf.jpg", labels=["healthy", "diseased", "wilting"])
    print(result.top_label, result.confidence)
"""
