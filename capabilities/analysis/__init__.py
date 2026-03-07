"""
capabilities/analysis
======================
Image analysis capabilities for plant health and growth monitoring.
All classes here are pure Python / OpenCV — no model downloads, no GPU.

Classes
-------
ColourAnalyser      — HSV histograms, health index, yellowing/browning scores
TextureAnalyser     — LBP & Gabor filters for disease surface texture
CoverageEstimator   — Green canopy coverage % over time
AnomalyDetector     — DINOv2 feature-based anomaly detection (no labels needed)
"""
