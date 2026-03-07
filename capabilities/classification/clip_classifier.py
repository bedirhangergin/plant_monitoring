"""
capabilities/classification/clip_classifier.py
===============================================
CLIP Zero-Shot Image Classifier

Scores any image against any set of text labels using OpenAI CLIP's
shared image-text embedding space. No training, no fixed class list —
you define the labels at call time.

This is the most versatile classification capability in the toolkit.
It can answer any question that can be phrased as competing text prompts:

    "healthy leaf" vs "diseased leaf"
    "flower open" vs "flower closed"
    "wilting plant" vs "upright plant"
    "early stage" vs "mature stage"
    "sprout" vs "bare soil"

Supported models (set via model_variant param)
----------------------------------------------
    "ViT-B/32"   350 MB   Balanced speed/accuracy  (default)
    "ViT-B/16"   335 MB   Slightly better accuracy
    "ViT-L/14"   890 MB   Best accuracy, slower CPU inference

Hardware
--------
    CPU: 1–3s per image (ViT-B/32)
    GPU: ~100ms per image

Requirements
------------
    pip install git+https://github.com/openai/CLIP.git
    pip install torch torchvision

Usage
-----
    from capabilities.classification.clip_classifier import CLIPClassifier

    clf = CLIPClassifier()

    # Basic classification
    result = clf.classify(
        "leaf.jpg",
        labels=["healthy leaf", "yellowing leaf", "diseased leaf", "dead leaf"]
    )
    print(result.top_label)       # "yellowing leaf"
    print(result.confidence)      # 0.61
    print(result.all_scores)      # {"healthy leaf": 0.12, "yellowing leaf": 0.61, ...}

    # Group labels into positive/negative for binary decisions
    result = clf.classify(
        "plant.jpg",
        labels=["healthy plant", "stressed plant"],
        positive_indices=[0],     # index 0 = "healthy"
    )
    print(result.binary_decision)  # True  (healthy wins)
    print(result.positive_score)   # 0.73  (summed probability of positive labels)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult(CapabilityResult):
    """
    Result from CLIPClassifier.classify().

    Attributes
    ----------
    labels : list of str
        The text labels that were scored.
    all_scores : dict
        Maps each label to its softmax probability (sums to 1.0).
    top_label : str
        The label with the highest softmax probability.
    top_index : int
        Index of top_label in the labels list.
    confidence : float
        Softmax probability of the top label.
    positive_score : float or None
        Sum of probabilities over positive_indices (if provided).
    negative_score : float or None
        Sum of probabilities over negative_indices (if provided).
    binary_decision : bool or None
        True if positive_score > negative_score (if indices provided).
    """

    labels: List[str] = field(default_factory=list)
    all_scores: Dict[str, float] = field(default_factory=dict)
    top_label: str = ""
    top_index: int = 0
    confidence: float = 0.0
    positive_score: Optional[float] = None
    negative_score: Optional[float] = None
    binary_decision: Optional[bool] = None

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "top_label": self.top_label,
            "top_index": self.top_index,
            "confidence": round(self.confidence, 4),
            "all_scores": {k: round(v, 4) for k, v in self.all_scores.items()},
            "positive_score": round(self.positive_score, 4) if self.positive_score is not None else None,
            "negative_score": round(self.negative_score, 4) if self.negative_score is not None else None,
            "binary_decision": self.binary_decision,
        })
        return base

    def __repr__(self) -> str:
        decision = f" | binary={self.binary_decision}" if self.binary_decision is not None else ""
        return (
            f"<ClassificationResult top='{self.top_label}' "
            f"conf={self.confidence:.2f}{decision} "
            f"model={self.model_name}>"
        )


# ---------------------------------------------------------------------------
# Classifier class
# ---------------------------------------------------------------------------

class CLIPClassifier(BaseCapability):
    """
    Zero-shot image classifier using OpenAI CLIP.

    The model is loaded lazily on the first call to classify() and
    cached for the lifetime of this object.  Models are ~350 MB and
    download automatically from OpenAI's servers on first use.

    Parameters
    ----------
    model_variant : str
        CLIP model variant. Options: 'ViT-B/32' (default), 'ViT-B/16', 'ViT-L/14'.
        Larger variants are more accurate but slower and heavier.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        model_variant: str = "ViT-B/32",
        device: Optional[str] = None,
    ) -> None:
        self._variant = model_variant
        self._device_override = device
        # Lazy-loaded fields
        self._model = None
        self._preprocess = None
        self._device: Optional[str] = None

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        # e.g. "clip_ViT-B/32" → normalise to "clip_vit_b32"
        return "clip_" + self._variant.lower().replace("/", "_").replace("-", "_")

    def run(self, image_path: str, **kwargs) -> ClassificationResult:
        """
        Alias for classify() — satisfies the BaseCapability interface.
        Pass labels= as a keyword argument.
        """
        labels = kwargs.get("labels", ["plant", "no plant"])
        positive_indices = kwargs.get("positive_indices", None)
        return self.classify(image_path, labels=labels, positive_indices=positive_indices)

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def classify(
        self,
        image_path: str,
        labels: List[str],
        positive_indices: Optional[List[int]] = None,
    ) -> ClassificationResult:
        """
        Score an image against a list of text labels.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        labels : list of str
            Text labels to score the image against.
            Define these as competing hypotheses, e.g.:
            ["healthy plant", "stressed plant", "diseased plant"]
        positive_indices : list of int, optional
            Indices in `labels` considered the "positive" outcome.
            When provided, enables binary_decision and positive/negative scores.
            Example: positive_indices=[0] if labels[0] is the "good" outcome.

        Returns
        -------
        ClassificationResult
        """
        self.validate_image(image_path)
        if not labels:
            raise ValueError("labels list must not be empty.")

        self._ensure_loaded()

        import torch
        import clip
        from PIL import Image

        with self._timer() as t:
            # Encode image
            image_tensor = (
                self._preprocess(Image.open(image_path))
                .unsqueeze(0)
                .to(self._device)
            )

            # Encode all text labels
            text_tokens = clip.tokenize(labels).to(self._device)

            with torch.no_grad():
                logits, _ = self._model(image_tensor, text_tokens)
                # Softmax across all labels → probability distribution
                probs = logits.softmax(dim=-1).cpu().numpy()[0]

        # Build score dict
        scores = {label: float(prob) for label, prob in zip(labels, probs)}
        top_idx = int(probs.argmax())
        top_label = labels[top_idx]
        confidence = float(probs[top_idx])

        # Binary decision logic (optional)
        positive_score = None
        negative_score = None
        binary_decision = None

        if positive_indices is not None:
            pos_set = set(positive_indices)
            positive_score = float(sum(probs[i] for i in pos_set))
            negative_score = float(sum(probs[i] for i in range(len(labels)) if i not in pos_set))
            binary_decision = positive_score > negative_score

        return ClassificationResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            labels=labels,
            all_scores=scores,
            top_label=top_label,
            top_index=top_idx,
            confidence=confidence,
            positive_score=positive_score,
            negative_score=negative_score,
            binary_decision=binary_decision,
        )

    def classify_multi_question(
        self,
        image_path: str,
        questions: Dict[str, List[str]],
    ) -> Dict[str, ClassificationResult]:
        """
        Run multiple independent classification questions on one image.

        Each question is a separate softmax over its own label set.
        More efficient than calling classify() multiple times because the
        image is encoded only once and text tokens are batched.

        Parameters
        ----------
        image_path : str
            Path to the image file.
        questions : dict
            Maps question name → list of labels.
            Example:
                {
                    "health":   ["healthy", "diseased", "stressed"],
                    "stage":    ["seedling", "vegetative", "flowering"],
                    "coverage": ["sparse", "moderate", "dense"],
                }

        Returns
        -------
        dict mapping question name → ClassificationResult
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        import torch
        import clip
        from PIL import Image

        # Encode image once
        image_tensor = (
            self._preprocess(Image.open(image_path))
            .unsqueeze(0)
            .to(self._device)
        )

        results = {}
        for question_name, labels in questions.items():
            with self._timer() as t:
                text_tokens = clip.tokenize(labels).to(self._device)
                with torch.no_grad():
                    logits, _ = self._model(image_tensor, text_tokens)
                    probs = logits.softmax(dim=-1).cpu().numpy()[0]

            scores = {label: float(p) for label, p in zip(labels, probs)}
            top_idx = int(probs.argmax())

            results[question_name] = ClassificationResult(
                image_path=image_path,
                model_name=self.model_name,
                duration_ms=t.elapsed_ms,
                labels=labels,
                all_scores=scores,
                top_label=labels[top_idx],
                top_index=top_idx,
                confidence=float(probs[top_idx]),
            )

        return results

    # ------------------------------------------------------------------ #
    # Model lifecycle
    # ------------------------------------------------------------------ #

    def _ensure_loaded(self) -> None:
        """Load CLIP model if not yet loaded. Called automatically by classify()."""
        if self._model is not None:
            return
        import torch
        import clip  # type: ignore

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[CLIPClassifier] Loading '{self._variant}' on {self._device} ...")
        self._model, self._preprocess = clip.load(self._variant, device=self._device)
        print(f"[CLIPClassifier] Ready.")

    def is_loaded(self) -> bool:
        """True if the model is currently in memory."""
        return self._model is not None

    def unload(self) -> None:
        """Release model from memory. Will reload on next classify() call."""
        self._model = None
        self._preprocess = None
        self._device = None
