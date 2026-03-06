"""
config.py
=========
Central configuration for the Plant Growth Monitor project.

All tunable parameters live here so you never need to hunt through source
files to change a threshold or swap a model name.  Import this module
anywhere in the project:

    from config import CONFIG

Environment variables take priority over the defaults below.  Set them in
your shell or in a .env file before launching the application:

    export GEMINI_API_KEY=your-key-here
    export CONFIDENCE_THRESHOLD=0.70
"""

import os

# ---------------------------------------------------------------------------
# CASCADE CONFIGURATION
# ---------------------------------------------------------------------------
# The pipeline runs Layer 1 → Layer 2 → Layer 3 in order.
# It exits as soon as a layer returns confidence >= CONFIDENCE_THRESHOLD.
# Lowering this value means the pipeline trusts cheap layers more easily.
# Raising it forces escalation to stronger (and slower/costlier) layers.

CONFIDENCE_THRESHOLD: float = float(
    os.environ.get("CONFIDENCE_THRESHOLD", "0.60")
)

# ---------------------------------------------------------------------------
# LAYER 1 — Classical HSV Masking
# ---------------------------------------------------------------------------
# OpenCV represents Hue on a 0–179 scale (not 0–360).
# Green occupies roughly H 35–85 in this scale.
# S and V floor values of 40 filter out near-black and near-white pixels
# that could otherwise be misclassified as green.

HSV_LOWER: tuple = (35, 40, 40)   # (H_min, S_min, V_min)
HSV_UPPER: tuple = (85, 255, 255)  # (H_max, S_max, V_max)

# Fraction of total image pixels that must be green to declare a sprout.
# 0.01 = 1 %.  Increase for stricter detection, decrease for sensitivity.
HSV_GREEN_RATIO_THRESHOLD: float = float(
    os.environ.get("HSV_GREEN_RATIO_THRESHOLD", "0.01")
)

# Morphological kernel size used to denoise the green mask.
# Larger kernel removes bigger noise blobs but can also erase tiny sprouts.
HSV_MORPH_KERNEL_SIZE: int = 5

# ---------------------------------------------------------------------------
# LAYER 2 — CLIP Zero-Shot Classification
# ---------------------------------------------------------------------------
# Text prompts scored against the image via cosine similarity.
# Add or remove prompts to tune sensitivity for your growing environment.

CLIP_MODEL_NAME: str = "ViT-B/32"    # 350 MB download on first use

CLIP_PROMPTS: list = [          
    "green sprout emerging from soil",    
    #"seedling breaking through soil surface",  
    "seed tray with plants",  
    "bare soil with no plant", 
    "empty seed tray",      
]

# Indices in CLIP_PROMPTS that represent "sprout present".
# Probability mass over these indices is summed for the sprout score.
CLIP_SPROUT_PROMPT_INDICES: list = [0, 1]

# ---------------------------------------------------------------------------
# LAYER 3 — Gemini Flash API Fallback
# ---------------------------------------------------------------------------
# Only invoked when both Layer 1 and Layer 2 return confidence < threshold.
# Costs ~$0.001 per call with Gemini 2.0 Flash (as of 2025).
# Leave GEMINI_API_KEY empty to disable Layer 3 gracefully.

GEMINI_MODEL: str = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_API_KEY: str = os.environ.get("GEMINI_API_KEY", "")

# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
# All analysis results are appended as JSON Lines to this file.
# Each line is a complete, self-contained JSON object.

LOG_DIR: str = os.path.join(os.path.dirname(__file__), "logs")
LOG_FILE: str = os.path.join(LOG_DIR, "sprout_detections.jsonl")

# ---------------------------------------------------------------------------
# CONVENIENCE DICT (for passing to functions that expect a plain dict)
# ---------------------------------------------------------------------------
# This mirrors the CONFIG dict style used in the notebook so you can pass
# config=CONFIG to any detector function unchanged.

CONFIG: dict = {
    # Cascade
    "confidence_threshold": CONFIDENCE_THRESHOLD,

    # Layer 1
    "hsv_lower": HSV_LOWER,
    "hsv_upper": HSV_UPPER,
    "hsv_green_ratio_threshold": HSV_GREEN_RATIO_THRESHOLD,
    "hsv_morph_kernel_size": HSV_MORPH_KERNEL_SIZE,

    # Layer 2
    "clip_model_name": CLIP_MODEL_NAME,
    "clip_prompts": CLIP_PROMPTS,
    "clip_sprout_prompt_indices": CLIP_SPROUT_PROMPT_INDICES,

    # Layer 3
    "gemini_model": GEMINI_MODEL,
    "gemini_api_key": GEMINI_API_KEY,

    # Logging
    "log_file": LOG_FILE,
}
