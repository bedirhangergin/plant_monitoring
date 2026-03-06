# 🌱 Plant Monitoring — Use Case A: Sprout Detection

A production-grade, three-layer computer vision pipeline for detecting when a plant first emerges from soil. Built entirely from free, pre-trained models with a Gemini API fallback — **no custom model training required**.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Running the Demo Notebook](#running-the-demo-notebook)
7. [CLI Usage](#cli-usage)
8. [Configuration](#configuration)
9. [Running Tests](#running-tests)
10. [Adding Your Own Images](#adding-your-own-images)
11. [Hardware Requirements](#hardware-requirements)
12. [Cost Summary](#cost-summary)
13. [Extending the Pipeline](#extending-the-pipeline)

---

## Overview

The sprout detection pipeline accepts **a single image at a time** — whether it is a manually taken photo or a frame extracted from a timelapse video. The same code path runs for both.

### Two Detection Use Cases

| Use Case | Input | Output |
|----------|-------|--------|
| **A — Sprout Detection** (this project) | Photo or video frame | `sprout_detected: bool` + `confidence: float` |
| **B — Height Estimation** *(future)* | Confirmed sprout photo | `height_value: float` (relative or metric) |

---

## Architecture

The pipeline uses a **three-layer confidence cascade**. It exits as early as possible — only escalating when the current layer returns `confidence < threshold` (default: 0.60).

```
Input Image (photo or extracted video frame)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  Layer 1 — Classical HSV Masking                    │
│  Cost: Free  │  Speed: <10ms  │  GPU: Not required  │
│                                                     │
│  Green pixel ratio >= threshold → sprout_detected   │
│  Confidence scales with distance from boundary      │
└───────────────────────┬─────────────────────────────┘
         confidence >= 0.60  → return result
         confidence <  0.60  ↓
┌─────────────────────────────────────────────────────┐
│  Layer 2 — CLIP Zero-Shot Classification            │
│  Cost: Free  │  Speed: 1–3s CPU / ~100ms GPU        │
│  Model: ViT-B/32  (~350 MB, downloaded once)        │
│                                                     │
│  Scores image against text prompts via cosine sim.  │
│  Confidence = winning probability mass              │
└───────────────────────┬─────────────────────────────┘
         confidence >= 0.60  → return result
         confidence <  0.60  ↓
┌─────────────────────────────────────────────────────┐
│  Layer 3 — Gemini Flash Vision API (Fallback)       │
│  Cost: ~$0.001/call  │  Speed: 1–2s  │  No GPU      │
│                                                     │
│  Sends image + structured prompt to Gemini Flash.   │
│  Returns: sprout_detected, confidence, reasoning    │
└─────────────────────────────────────────────────────┘
                  └→ return result
```

**Key design principle:** In a well-calibrated environment, **90%+ of images resolve at Layer 1** — fast, free, and offline.

---

## Project Structure

```
plant_monitoring_project/
│
├── config.py                        # Central configuration (all tunable parameters)
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── sprout_detection/                # Use Case A package
│   ├── __init__.py                  # Public API: SproutCascade, SproutResult
│   ├── cascade.py                   # Cascade orchestrator — the main pipeline
│   ├── result.py                    # SproutResult dataclass (shared by all layers)
│   │
│   ├── detectors/                   # One module per detection layer
│   │   ├── __init__.py
│   │   ├── base_detector.py         # Abstract base class (contract for all detectors)
│   │   ├── hsv_detector.py          # Layer 1 — Classical HSV masking
│   │   ├── clip_detector.py         # Layer 2 — CLIP zero-shot classification
│   │   └── gemini_detector.py       # Layer 3 — Gemini Flash API fallback
│   │
│   └── utils/                       # Shared utilities
│       ├── __init__.py
│       ├── video.py                 # extract_frames() — video → image frames
│       ├── visualiser.py            # Matplotlib visualisations per layer + batch
│       └── image_gen.py             # Synthetic test image generator
│
├── tests/
│   ├── conftest.py                  # Shared pytest fixtures (images, configs)
│   ├── unit/
│   │   ├── test_result.py           # SproutResult dataclass tests
│   │   ├── test_hsv_detector.py     # Layer 1 tests (full — no mocking needed)
│   │   ├── test_clip_detector.py    # Layer 2 tests (mocked CLIP model)
│   │   └── test_gemini_detector.py  # Layer 3 tests (mocked Gemini API)
│   └── integration/
│       └── test_cascade.py          # Full pipeline tests with stub detectors
│
├── notebooks/
│   └── 01_sprout_detection_demo.ipynb   # Guided demonstration notebook
│
├── assets/
│   └── sample_images/               # Generated test images (auto-created)
│
└── logs/                            # JSONL detection logs (auto-created)
```

### Why this structure?

- **One file per detector** — `hsv_detector.py`, `clip_detector.py`, `gemini_detector.py` are completely independent. You can debug, test, or replace any layer without touching others.
- **`cascade.py` only orchestrates** — it never contains detection logic, making escalation flow trivially readable.
- **`result.py` is the contract** — every layer returns the same `SproutResult`, so downstream code (logger, visualiser, tests) is layer-agnostic.
- **`config.py` is the single source of truth** — no magic numbers buried in source files.

---

## Installation

### Prerequisites

- Python 3.9 or higher
- pip

### Step 1 — Clone and enter the project

```bash
git clone https://github.com/your-org/plant-monitoring-project.git
cd plant-monitor
```

### Step 2 — Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate           # Windows
```

### Step 3 — Install dependencies

**CPU-only (recommended for most users):**
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install git+https://github.com/openai/CLIP.git
```

**CUDA GPU (faster CLIP inference):**
```bash
# Replace cu121 with your CUDA version (cu118, cu121, cu124, etc.)
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**Minimal install (Layer 1 only — no CLIP, no model download):**
```bash
pip install opencv-python numpy Pillow matplotlib google-generativeai
```

### Step 4 — (Optional) Set Gemini API key for Layer 3

```bash
export GEMINI_API_KEY=your-key-here
```

---

## Quick Start

```python
from config import CONFIG
from sprout_detection import SproutCascade

# Create the cascade
cascade = SproutCascade(config=CONFIG)

# Analyse a single image
result = cascade.analyze("path/to/your/photo.jpg")

print(result)
# 🌱 SproutResult | detected=True | confidence=0.87 | method=hsv_masking

print(result.to_json())
# {
#   "sprout_detected": true,
#   "confidence": 0.87,
#   "method": "hsv_masking",
#   "reasoning": "Green pixel ratio = 0.0450 (above threshold 0.01). Sprout detected.",
#   ...
# }
```

### Analyse a batch of images

```python
results = cascade.analyze_batch(["img1.jpg", "img2.jpg", "img3.jpg"])
for result in results:
    print(result.summary_line)
```

### Analyse a video

```python
from sprout_detection.utils.video import extract_frames

# Extract one frame every 60 seconds
frame_paths = extract_frames("timelapse.mp4", every_n_seconds=60)

# analyze() is identical for video frames and photos
results = cascade.analyze_batch(frame_paths)
```

### Use a specific layer directly

```python
from sprout_detection.detectors.hsv_detector import HSVDetector

detector = HSVDetector(config=CONFIG)
result = detector.detect("photo.jpg")

# Access the binary mask for custom visualisation
mask = result.hsv_mask
```

---

## Running the Demo Notebook

The notebook at `notebooks/01_sprout_detection_demo.ipynb` provides a fully narrated, visual walkthrough of every layer and the cascade.

```bash
jupyter notebook notebooks/01_sprout_detection_demo.ipynb
```

The notebook covers:
- Synthetic test image generation
- Layer 1 (HSV) with confidence formula explanation
- Layer 2 (CLIP) with prompt score bar charts
- Layer 3 (Gemini) with reasoning card
- Full cascade with escalation
- Batch analysis with grid summary
- Video frame workflow

---

## CLI Usage

Run detection from the command line without writing any Python:

```bash
# Analyse a single image
python main.py --image path/to/photo.jpg

# Analyse multiple images
python main.py --image img1.jpg img2.jpg img3.jpg

# Analyse a video (extract frames every 30 seconds)
python main.py --video timelapse.mp4 --interval 30

# Run a quick demo with synthetic images
python main.py --demo

# Show the detection log
python main.py --log

# Clear the detection log
python main.py --clear-log

# Override confidence threshold
python main.py --image photo.jpg --threshold 0.75

# Disable Layer 3 (Gemini API) even if key is set
python main.py --image photo.jpg --no-gemini

# Suppress per-step output
python main.py --image photo.jpg --quiet
```

---

## Configuration

All parameters are in `config.py`. You can override any value at runtime:

```python
from config import CONFIG

# Override at runtime
CONFIG["confidence_threshold"] = 0.70   # More conservative cascade exit
CONFIG["hsv_green_ratio_threshold"] = 0.005  # More sensitive to small sprouts
CONFIG["gemini_api_key"] = "your-key"   # Enable Layer 3

cascade = SproutCascade(config=CONFIG)
```

Or via environment variables before launching:

```bash
export CONFIDENCE_THRESHOLD=0.70
export HSV_GREEN_RATIO_THRESHOLD=0.005
export GEMINI_API_KEY=your-key
python main.py --image photo.jpg
```

### Key configuration parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `confidence_threshold` | `0.60` | Cascade exit threshold. Lower = faster (accepts less confident results). |
| `hsv_green_ratio_threshold` | `0.01` | Fraction of image that must be green to declare a sprout (Layer 1). |
| `hsv_lower` | `(35, 40, 40)` | HSV lower bound for green. Adjust for different plant species or lighting. |
| `hsv_upper` | `(85, 255, 255)` | HSV upper bound for green. |
| `clip_model_name` | `"ViT-B/32"` | CLIP model variant. `ViT-L/14` is more accurate but larger. |
| `clip_prompts` | *(see config.py)* | Text prompts scored against the image. Add prompts for your specific environment. |
| `gemini_model` | `"gemini-2.0-flash"` | Gemini model variant for Layer 3. |
| `gemini_api_key` | `""` | Set to enable Layer 3. Can also use `GEMINI_API_KEY` env var. |

### Tuning for your environment

**Indoor greenhouse with consistent lighting → increase HSV confidence:**
```python
CONFIG["hsv_green_ratio_threshold"] = 0.008   # Slightly more sensitive
CONFIG["confidence_threshold"] = 0.55         # Exit earlier
```

**Outdoor / variable lighting → lean on CLIP more:**
```python
CONFIG["confidence_threshold"] = 0.70         # Require CLIP to confirm HSV
```

**High-volume, cost-conscious deployment:**
```python
CONFIG["gemini_api_key"] = ""                 # Disable API entirely
CONFIG["confidence_threshold"] = 0.55         # Accept Layer 1/2 results faster
```

## Adding Your Own Images

1. Place your images anywhere on disk (JPEG or PNG).
2. Pass the path to `cascade.analyze()`:

```python
result = cascade.analyze("/path/to/my_plant_photo.jpg")
```

3. For best results with Layer 1 (HSV):
   - Ensure consistent framing (same camera angle and distance)
   - Avoid direct sunlight causing overexposed green areas
   - If detection is unreliable, adjust `hsv_lower`/`hsv_upper` in `config.py` for your specific plant colour

---

## Hardware Requirements

| Configuration | CPU | RAM | GPU VRAM | Notes |
|--------------|-----|-----|----------|-------|
| Minimal (Layer 1 only) | Any (2015+) | 1 GB | None | Classical CV only, <10ms/image |
| Balanced (Layers 1+2) | Modern (2020+) | 4 GB | None | CLIP on CPU, ~1–3s/image |
| Fast (Layers 1+2) | Any | 4 GB | 2 GB+ | CLIP on GPU, ~100ms/image |
| Cloud (Layer 3 only) | Any | 1 GB | None | API calls, internet required |

---

## Cost Summary

In a typical controlled environment (stable lighting, consistent angle), over 90% of images resolve at Layer 1 with zero API cost.

| Daily Volume | API Calls (10% escalation) | Daily Cost | Monthly Cost |
|-------------|--------------------------|------------|--------------|
| 100 images  | ~10 calls | ~$0.01 | ~$0.30 |
| 1,000 images | ~100 calls | ~$0.10 | ~$3.00 |
| 10,000 images | ~1,000 calls | ~$1.00 | ~$30.00 |

If your environment is well-calibrated for HSV masking, actual escalation rates will be **well below 10%**, reducing costs further.

---

## Extending the Pipeline

### Add a new detector layer

Create a new file in `sprout_detection/detectors/`:

```python
# sprout_detection/detectors/grounding_dino_detector.py

from sprout_detection.detectors.base_detector import BaseDetector
from sprout_detection.result import SproutResult

class GroundingDINODetector(BaseDetector):

    @property
    def layer_name(self) -> str:
        return "grounding_dino"

    def detect(self, image_path: str, escalated: bool = False) -> SproutResult:
        # Your implementation here
        ...
```

Inject it into the cascade:

```python
from sprout_detection.detectors.grounding_dino_detector import GroundingDINODetector

cascade = SproutCascade(
    layer2=GroundingDINODetector(config=CONFIG)  # Replace or add as new layer
)
```

### Customise CLIP prompts

Add domain-specific prompts to `config.py` for your exact growing environment:

```python
CLIP_PROMPTS = [
    "bare brown soil with no vegetation",
    "tiny green seedling sprouting",
    "first leaves emerging from potting mix",
    "empty black seed tray",
    "cotyledon leaves visible above soil",  # ← Add specific botanical terms
]
CLIP_SPROUT_PROMPT_INDICES = [1, 2, 4]  # ← Mark which are "sprout" prompts
```

---


