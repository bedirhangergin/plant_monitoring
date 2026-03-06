"""
sprout_detection/detectors/clip_detector.py
============================================
Layer 2 — CLIP Zero-Shot Classification

Algorithm overview
------------------
CLIP (Contrastive Language-Image Pretraining) by OpenAI learns a shared
embedding space for images and text.  We exploit this to classify images
without any task-specific training:

1. Encode the input image into a 512-dim embedding vector.
2. Encode each text prompt (e.g. "green sprout emerging from soil") into
   the same embedding space.
3. Compute cosine similarity between the image and every prompt.
4. Apply softmax to convert similarities to probabilities.
5. Sum probabilities over the "sprout" prompts and "no sprout" prompts.
6. The larger sum determines the prediction.
   Confidence = the winning sum (already in [0.0, 1.0] after softmax).

Lazy model loading
------------------
The CLIP model (~350 MB) is only downloaded and loaded into memory the
first time detect() is called.  Subsequent calls reuse the cached model.
This avoids paying the load cost on import.

Best for
--------
- Variable lighting or outdoor conditions where HSV masking is unreliable.
- CPU-only deployments where 1–3 s per image is acceptable.
- Cases where zero threshold calibration is desired.

Limitations
-----------
- 350 MB model download on first run (cached by PyTorch automatically).
- ~1–3 s per image on CPU; ~100 ms on GPU.
- Too slow for real-time video processing on CPU.

Requirements
------------
    pip install git+https://github.com/openai/CLIP.git
    pip install torch torchvision

Usage
-----
    from sprout_detection.detectors.clip_detector import CLIPDetector
    from config import CONFIG

    detector = CLIPDetector(config=CONFIG)
    result = detector.detect("photo.jpg")
    print(result)
    # 🌱 SproutResult | detected=True | confidence=0.73 | method=clip_zero_shot
    print(result.clip_scores)
    # {'bare soil with no plant': 0.12, 'green sprout emerging from soil': 0.61, ...}
"""

from __future__ import annotations

from typing import Optional

from config import CONFIG
from sprout_detection.detectors.base_detector import BaseDetector
from sprout_detection.result import SproutResult


class CLIPDetector(BaseDetector):
    """
    Sprout detector using CLIP zero-shot image-text matching (Layer 2).

    The CLIP model is loaded lazily on the first call to detect() and
    cached as an instance attribute for the lifetime of this object.

    Parameters
    ----------
    config : dict, optional
        Configuration dictionary.  Defaults to the global CONFIG.
        Relevant keys:
          - clip_model_name            : str,  e.g. 'ViT-B/32'
          - clip_prompts               : list, text prompts to score
          - clip_sprout_prompt_indices : list, indices of "sprout" prompts
          - confidence_threshold       : float
    """

    def __init__(self, config: dict = None) -> None:
        self._config = config or CONFIG
        self._model = None          # Loaded lazily
        self._preprocess = None     # Torchvision transform pipeline
        self._device: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseDetector interface
    # ------------------------------------------------------------------ #

    @property
    def layer_name(self) -> str:
        return "clip_zero_shot"

    def detect(self, image_path: str, escalated: bool = False) -> SproutResult:
        """
        Run CLIP zero-shot classification on a single image.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        escalated : bool
            True if Layer 1 already ran and returned low confidence.

        Returns
        -------
        SproutResult
            Includes .clip_scores dict mapping each prompt to its probability.
        """
        self.validate_image(image_path)
        self._ensure_model_loaded()

        # Local imports kept inside method so the module can be imported
        # even if CLIP / torch are not installed (graceful degradation).
        import torch
        import clip
        from PIL import Image

        prompts = self._config["clip_prompts"]
        sprout_indices = set(self._config["clip_sprout_prompt_indices"])

        # ── Encode image ────────────────────────────────────────────────
        # preprocess applies resize, centre-crop, normalisation
        image_tensor = (
            self._preprocess(Image.open(image_path))
            .unsqueeze(0)          # Add batch dimension: [1, C, H, W]
            .to(self._device)
        )

        # ── Encode text prompts ─────────────────────────────────────────
        # clip.tokenize handles padding / truncation to 77 tokens
        text_tokens = clip.tokenize(prompts).to(self._device)

        with torch.no_grad():
            # logits_per_image shape: [1, num_prompts]
            logits_per_image, _ = self._model(image_tensor, text_tokens)
            # Softmax converts raw logits to a probability distribution
            # that sums to 1.0 across all prompts
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        # ── Build score dict ────────────────────────────────────────────
        scores: dict = {
            prompt: float(prob) for prompt, prob in zip(prompts, probs)
        }

        # ── Aggregate into sprout vs no-sprout probability mass ─────────
        sprout_score = sum(
            probs[i] for i in range(len(prompts)) if i in sprout_indices
        )
        no_sprout_score = sum(
            probs[i] for i in range(len(prompts)) if i not in sprout_indices
        )

        sprout_detected = sprout_score > no_sprout_score
        # Confidence = the winning probability mass (already 0–1)
        confidence = float(max(sprout_score, no_sprout_score))

        # ── Build reasoning string ──────────────────────────────────────
        best_prompt = max(scores, key=scores.get)
        reasoning = (
            f"Best matching prompt: '{best_prompt}' "
            f"(score={scores[best_prompt]:.3f}). "
            f"Sprout mass={sprout_score:.3f}, "
            f"No-sprout mass={no_sprout_score:.3f}."
        )

        return SproutResult(
            sprout_detected=str(sprout_detected),
            confidence=confidence,
            method=self.layer_name,
            reasoning=reasoning,
            image_path=image_path,
            escalated=escalated,
            clip_scores=scores,
        )

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_model_loaded(self) -> None:
        """
        Load the CLIP model if it has not been loaded yet.

        This is called automatically by detect() on the first invocation.
        The model is cached as self._model so subsequent calls are instant.

        The model is placed on CUDA if available, otherwise CPU.
        """
        if self._model is not None:
            return  # Already loaded

        import torch
        import clip  # type: ignore

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = self._config.get("clip_model_name", "ViT-B/32")

        print(
            f"[CLIPDetector] Loading '{model_name}' on {self._device}. "
            f"First load downloads ~350 MB — subsequent loads are instant."
        )
        self._model, self._preprocess = clip.load(model_name, device=self._device)
        print(f"[CLIPDetector] Model loaded successfully.")

    def is_loaded(self) -> bool:
        """Return True if the CLIP model has been loaded into memory."""
        return self._model is not None

    def unload(self) -> None:
        """
        Release the model from memory.

        Useful in memory-constrained environments when switching between
        detector layers.  The model will be reloaded on the next detect() call.
        """
        self._model = None
        self._preprocess = None
        self._device = None
