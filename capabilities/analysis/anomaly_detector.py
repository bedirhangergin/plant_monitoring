"""
capabilities/analysis/anomaly_detector.py
==========================================
DINOv2 Feature-Based Anomaly Detector

Detects unusual regions in plant images WITHOUT any labelled training data.
The approach:

  1. Extract deep visual features from the image using DINOv2 (a powerful
     self-supervised vision transformer).
  2. Compare the feature distribution to a reference set of "normal" images.
  3. Pixels/regions far from the normal feature distribution are flagged
     as anomalous.

This means you can detect disease, stress, or physical damage purely from
the model's learned representation — no disease labels, no annotated dataset,
no fine-tuning required.

Use cases
---------
  - First-time disease detection (you don't know what disease looks like yet)
  - Detecting unusual leaf colour, texture, or shape
  - Flagging images that need human review
  - Building a reference library of "healthy" appearances

Two operating modes
-------------------
  "reference"  — Fit a reference distribution from healthy images first,
                 then score new images against it. Most accurate.
  "self"       — No reference. Detect internal patch inconsistencies.
                 Useful when you don't have reference images yet.

Hardware
--------
    CPU: 2–5s per image (DINOv2-small)
    GPU: ~200ms per image

Model variants
--------------
    "facebook/dinov2-small"   90 MB   Fastest  (default)
    "facebook/dinov2-base"   330 MB   Balanced
    "facebook/dinov2-large"  1.1 GB  Best accuracy

Requirements
------------
    pip install transformers torch torchvision scikit-learn

Usage
-----
    from capabilities.analysis.anomaly_detector import AnomalyDetector

    detector = AnomalyDetector()

    # Mode 1: Self-supervised (no reference needed)
    result = detector.detect("leaf.jpg")
    print(result.anomaly_score)       # 0.0–1.0, higher = more abnormal
    print(result.is_anomalous)        # True if score > threshold
    print(result.anomaly_map.shape)   # (H, W) spatial anomaly heatmap

    # Mode 2: Reference-based (more accurate)
    detector.fit_reference(["healthy1.jpg", "healthy2.jpg", "healthy3.jpg"])
    result = detector.detect("suspicious_leaf.jpg")
    print(result.anomaly_score)       # How far from normal
    print(result.reference_fitted)    # True
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from capabilities.base import BaseCapability, CapabilityResult


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AnomalyResult(CapabilityResult):
    """
    Result from AnomalyDetector.detect().

    Attributes
    ----------
    anomaly_score : float
        Global anomaly score [0–1]. Higher = more unusual.
    is_anomalous : bool
        True if anomaly_score exceeds the configured threshold.
    anomaly_map : np.ndarray or None
        Spatial (H × W) heatmap of anomaly intensity.
        Brighter regions = more anomalous patches.
    reference_fitted : bool
        True if the detector was fitted on reference images before scoring.
    patch_scores : list of float
        Per-patch anomaly scores (before upsampling to image resolution).
    """

    anomaly_score: float = 0.0
    is_anomalous: bool = False
    anomaly_map: Optional[np.ndarray] = field(default=None, repr=False)
    reference_fitted: bool = False
    patch_scores: List[float] = field(default_factory=list)
    threshold: float = 0.50

    def to_dict(self) -> dict:
        base = super().to_dict()
        base.update({
            "anomaly_score":    round(self.anomaly_score, 4),
            "is_anomalous":     self.is_anomalous,
            "reference_fitted": self.reference_fitted,
            "threshold":        self.threshold,
            "patch_count":      len(self.patch_scores),
        })
        return base

    def __repr__(self) -> str:
        status = "⚠️ ANOMALOUS" if self.is_anomalous else "✅ NORMAL"
        return (
            f"<AnomalyResult {status} score={self.anomaly_score:.3f} "
            f"ref_fitted={self.reference_fitted} model={self.model_name}>"
        )


# ---------------------------------------------------------------------------
# Anomaly Detector
# ---------------------------------------------------------------------------

class AnomalyDetector(BaseCapability):
    """
    Unsupervised anomaly detector using DINOv2 visual features.

    No labelled data required. Works by comparing feature distributions
    against either a reference set of normal images, or internally
    across patches within the same image.

    Parameters
    ----------
    model_id : str
        HuggingFace DINOv2 model ID. Default: 'facebook/dinov2-small'.
    anomaly_threshold : float
        Score above which an image is flagged as anomalous. Default: 0.50.
    device : str or None
        'cuda', 'cpu', or None (auto-detect).
    """

    _DEFAULT_MODEL = "facebook/dinov2-small"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        anomaly_threshold: float = 0.50,
        device: Optional[str] = None,
    ) -> None:
        self._model_id = model_id
        self._threshold = anomaly_threshold
        self._device_override = device
        # Lazy-loaded
        self._model = None
        self._processor = None
        self._device: Optional[str] = None
        # Reference distribution (fitted from healthy images)
        self._reference_features: Optional[np.ndarray] = None  # shape (N_images, N_patches, D)
        self._reference_mean: Optional[np.ndarray] = None       # shape (D,)
        self._reference_cov_inv: Optional[np.ndarray] = None    # Mahalanobis

    # ------------------------------------------------------------------ #
    # BaseCapability interface
    # ------------------------------------------------------------------ #

    @property
    def model_name(self) -> str:
        return self._model_id.split("/")[-1].replace("-", "_")

    def run(self, image_path: str, **kwargs) -> AnomalyResult:
        """Alias for detect()."""
        return self.detect(image_path)

    # ------------------------------------------------------------------ #
    # Reference fitting
    # ------------------------------------------------------------------ #

    def fit_reference(self, image_paths: List[str]) -> None:
        """
        Fit the reference distribution from a set of healthy images.

        Call this once with your "normal" images before running detect().
        More reference images = better anomaly detection.
        Recommended: 5–20 healthy images from your specific environment.

        Parameters
        ----------
        image_paths : list of str
            Paths to healthy/normal images.
        """
        self._ensure_loaded()
        print(f"[AnomalyDetector] Fitting reference from {len(image_paths)} images ...")

        all_features = []
        for path in image_paths:
            try:
                feats = self._extract_features(path)  # (N_patches, D)
                all_features.append(feats.mean(axis=0))  # Mean patch feature
            except Exception as e:
                print(f"  Skipped '{path}': {e}")

        if not all_features:
            raise ValueError("No valid reference images could be processed.")

        ref_matrix = np.stack(all_features)  # (N_images, D)
        self._reference_mean = ref_matrix.mean(axis=0)
        # Covariance for Mahalanobis distance
        cov = np.cov(ref_matrix.T) + np.eye(ref_matrix.shape[1]) * 1e-5
        self._reference_cov_inv = np.linalg.inv(cov)
        self._reference_features = ref_matrix
        print(f"[AnomalyDetector] Reference fitted. ({len(all_features)} images)")

    @property
    def reference_fitted(self) -> bool:
        """True if fit_reference() has been called."""
        return self._reference_mean is not None

    # ------------------------------------------------------------------ #
    # Primary API
    # ------------------------------------------------------------------ #

    def detect(self, image_path: str) -> AnomalyResult:
        """
        Detect anomalies in an image.

        If reference has been fitted (fit_reference() called), uses
        Mahalanobis distance from the reference distribution.
        Otherwise uses intra-image patch variance as the anomaly signal.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        AnomalyResult
        """
        self.validate_image(image_path)
        self._ensure_loaded()

        with self._timer() as t:
            features = self._extract_features(image_path)  # (N_patches, D)

            if self.reference_fitted:
                patch_scores = self._mahalanobis_scores(features)
            else:
                patch_scores = self._intra_image_scores(features)

            # Global score: mean of top 10% most anomalous patches
            sorted_scores = np.sort(patch_scores)[::-1]
            top_n = max(1, len(sorted_scores) // 10)
            anomaly_score = float(sorted_scores[:top_n].mean())

            # Normalise to [0, 1] — clamp at 5 for practical purposes
            anomaly_score = min(anomaly_score / 5.0, 1.0)

            # Build spatial anomaly map (upsampled to image resolution)
            anomaly_map = self._build_anomaly_map(image_path, patch_scores)

        return AnomalyResult(
            image_path=image_path,
            model_name=self.model_name,
            duration_ms=t.elapsed_ms,
            anomaly_score=anomaly_score,
            is_anomalous=anomaly_score >= self._threshold,
            anomaly_map=anomaly_map,
            reference_fitted=self.reference_fitted,
            patch_scores=patch_scores.tolist(),
            threshold=self._threshold,
        )

    # ------------------------------------------------------------------ #
    # Private helpers
    # ------------------------------------------------------------------ #

    def _extract_features(self, image_path: str) -> np.ndarray:
        """Extract DINOv2 patch-level features. Returns (N_patches, D)."""
        import torch
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)

        with torch.no_grad():
            outputs = self._model(**inputs, output_hidden_states=True)
            # Last hidden state: (1, N_patches+1, D) — skip CLS token [0]
            patch_features = outputs.last_hidden_state[0, 1:, :]

        return patch_features.cpu().numpy()  # (N_patches, D)

    def _mahalanobis_scores(self, features: np.ndarray) -> np.ndarray:
        """Mahalanobis distance of each patch from reference mean."""
        diff = features - self._reference_mean  # (N, D)
        # Vectorised: diag(diff @ cov_inv @ diff.T)
        scores = np.einsum("nd,dd,nd->n", diff, self._reference_cov_inv, diff)
        return np.sqrt(np.abs(scores))

    @staticmethod
    def _intra_image_scores(features: np.ndarray) -> np.ndarray:
        """
        Score each patch by its distance from the image's own mean feature.
        Used when no reference set is available.
        """
        mean = features.mean(axis=0, keepdims=True)
        diffs = features - mean
        scores = np.linalg.norm(diffs, axis=1)
        return scores

    def _build_anomaly_map(
        self, image_path: str, patch_scores: np.ndarray
    ) -> Optional[np.ndarray]:
        """
        Reshape patch scores into a spatial grid and upsample to image size.
        """
        try:
            from PIL import Image
            import cv2

            image = Image.open(image_path)
            img_w, img_h = image.size

            # DINOv2 patches are 14×14 pixels
            grid_h = img_h // 14
            grid_w = img_w // 14
            n_patches = grid_h * grid_w

            if len(patch_scores) < n_patches:
                # Pad if needed
                scores = np.pad(patch_scores, (0, n_patches - len(patch_scores)))
            else:
                scores = patch_scores[:n_patches]

            grid = scores.reshape(grid_h, grid_w).astype(np.float32)

            # Normalise to [0, 255] for visualisation
            g_min, g_max = grid.min(), grid.max()
            if g_max > g_min:
                grid = (grid - g_min) / (g_max - g_min) * 255

            # Upsample to original image resolution
            anomaly_map = cv2.resize(
                grid.astype(np.uint8), (img_w, img_h), interpolation=cv2.INTER_LINEAR
            )
            return anomaly_map
        except Exception:
            return None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if self._device_override:
            self._device = self._device_override
        else:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[AnomalyDetector] Loading '{self._model_id}' on {self._device} ...")
        self._processor = AutoImageProcessor.from_pretrained(self._model_id)
        self._model = AutoModel.from_pretrained(self._model_id).to(self._device)
        self._model.eval()
        print("[AnomalyDetector] Ready.")

    def is_loaded(self) -> bool:
        return self._model is not None

    def unload(self) -> None:
        self._model = None
        self._processor = None
        self._device = None
        # Keep reference distribution — it's cheap (just arrays)
