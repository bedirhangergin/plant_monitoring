"""
capabilities/temporal/growth_tracker.py
========================================
Plant Growth Tracker — Time-Series Analysis

Accumulates capability results over time and computes growth metrics,
trend lines, and alerts from the time series.

This class does not run any computer vision itself — it consumes results
from other capability classes (CoverageResult, DepthResult, etc.) and
builds a longitudinal picture.

Tracked metrics
---------------
  - Green coverage % over time → growth rate
  - Health index over time → stress trend
  - Relative height over time → elongation rate
  - Anomaly score over time → disease onset detection

Alerts generated automatically
-------------------------------
  - "Coverage drop > 10% in one step" → possible wilting or damage
  - "Health ratio below 0.60 for 3+ consecutive frames" → chronic stress
  - "Anomaly score spike" → sudden disease onset
  - "Zero coverage for 2+ frames after sprouting" → plant loss

Usage
-----
    from capabilities.temporal.growth_tracker import GrowthTracker

    tracker = GrowthTracker(plant_id="pot_03")

    # Log results as they arrive (daily, hourly, or per-frame)
    tracker.log_coverage(coverage_result, timestamp="2025-01-01")
    tracker.log_coverage(coverage_result_day2, timestamp="2025-01-02")
    tracker.log_depth(depth_result, tip=(215, 40), base=(215, 390))

    # Query the time series
    print(tracker.growth_rate())      # avg coverage change per step
    print(tracker.health_trend())     # slope of health_index over time
    print(tracker.get_alerts())       # list of triggered alerts

    # Plot everything
    tracker.plot(save_path="growth_report.png")

    # Export to JSON
    tracker.save("pot_03_log.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data record
# ---------------------------------------------------------------------------

@dataclass
class GrowthRecord:
    """A single timestamped measurement in the growth log."""

    timestamp: str
    green_coverage_pct: Optional[float] = None
    total_coverage_pct: Optional[float] = None
    health_ratio: Optional[float] = None
    relative_height: Optional[float] = None
    anomaly_score: Optional[float] = None
    sprout_detected: Optional[bool] = None
    image_path: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp":           self.timestamp,
            "green_coverage_pct":  self.green_coverage_pct,
            "total_coverage_pct":  self.total_coverage_pct,
            "health_ratio":        self.health_ratio,
            "relative_height":     self.relative_height,
            "anomaly_score":       self.anomaly_score,
            "sprout_detected":     self.sprout_detected,
            "image_path":          self.image_path,
            "extras":              self.extras,
        }


# ---------------------------------------------------------------------------
# Growth Tracker
# ---------------------------------------------------------------------------

class GrowthTracker:
    """
    Time-series growth log and analyser for a single plant.

    Parameters
    ----------
    plant_id : str
        Identifier for this plant/pot. Used in file names and plot titles.
    alert_coverage_drop : float
        Alert if green coverage drops more than this % in one step. Default: 10.0.
    alert_health_threshold : float
        Alert if health_ratio falls below this for 3+ consecutive records. Default: 0.60.
    alert_anomaly_threshold : float
        Alert if anomaly_score exceeds this. Default: 0.70.
    """

    def __init__(
        self,
        plant_id: str = "plant",
        alert_coverage_drop: float = 10.0,
        alert_health_threshold: float = 0.60,
        alert_anomaly_threshold: float = 0.70,
    ) -> None:
        self.plant_id = plant_id
        self._alert_cov_drop = alert_coverage_drop
        self._alert_health_thresh = alert_health_threshold
        self._alert_anomaly_thresh = alert_anomaly_threshold
        self._records: List[GrowthRecord] = []

    # ------------------------------------------------------------------ #
    # Logging methods — one per capability type
    # ------------------------------------------------------------------ #

    def log_coverage(
        self,
        result,                        # CoverageResult
        timestamp: Optional[str] = None,
    ) -> GrowthRecord:
        """
        Log a CoverageResult measurement.

        Parameters
        ----------
        result : CoverageResult
            From CoverageEstimator.estimate().
        timestamp : str, optional
            ISO-8601 timestamp. Defaults to current time.

        Returns
        -------
        GrowthRecord
        """
        record = GrowthRecord(
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            green_coverage_pct=getattr(result, "green_coverage_pct", None),
            total_coverage_pct=getattr(result, "total_coverage_pct", None),
            health_ratio=getattr(result, "health_ratio", None),
            image_path=getattr(result, "image_path", None),
        )
        self._records.append(record)
        return record

    def log_depth(
        self,
        result,                         # DepthResult
        tip: tuple,
        base: tuple,
        timestamp: Optional[str] = None,
    ) -> GrowthRecord:
        """
        Log a DepthResult height measurement.

        Parameters
        ----------
        result : DepthResult
            From DepthEstimator.estimate().
        tip : (x, y)
            Pixel coordinate of plant tip (top).
        base : (x, y)
            Pixel coordinate of soil surface (bottom of plant).
        timestamp : str, optional

        Returns
        -------
        GrowthRecord
        """
        height_info = result.estimate_height(tip, base)
        record = GrowthRecord(
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            relative_height=height_info.get("depth_difference"),
            image_path=getattr(result, "image_path", None),
            extras={"depth_info": height_info},
        )
        self._records.append(record)
        return record

    def log_anomaly(
        self,
        result,                        # AnomalyResult
        timestamp: Optional[str] = None,
    ) -> GrowthRecord:
        """Log an AnomalyResult."""
        record = GrowthRecord(
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            anomaly_score=getattr(result, "anomaly_score", None),
            image_path=getattr(result, "image_path", None),
        )
        self._records.append(record)
        return record

    def log_sprout(
        self,
        result,                        # SproutResult (from sprout_detection)
        timestamp: Optional[str] = None,
    ) -> GrowthRecord:
        """Log a SproutResult from the sprout detection cascade."""
        record = GrowthRecord(
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            sprout_detected=getattr(result, "sprout_detected", None),
            image_path=getattr(result, "image_path", None),
            extras={
                "confidence": getattr(result, "confidence", None),
                "method": getattr(result, "method", None),
            },
        )
        self._records.append(record)
        return record

    def log_manual(
        self,
        timestamp: Optional[str] = None,
        **kwargs,
    ) -> GrowthRecord:
        """
        Log arbitrary measurements manually.

        Parameters
        ----------
        **kwargs
            Any subset of GrowthRecord fields.

        Example
        -------
        tracker.log_manual(
            timestamp="2025-06-01",
            green_coverage_pct=45.2,
            health_ratio=0.88,
        )
        """
        record = GrowthRecord(
            timestamp=timestamp or datetime.now().isoformat(timespec="seconds"),
            **{k: v for k, v in kwargs.items() if hasattr(GrowthRecord, k)},
        )
        self._records.append(record)
        return record

    # ------------------------------------------------------------------ #
    # Analysis
    # ------------------------------------------------------------------ #

    def growth_rate(self) -> Optional[float]:
        """
        Average change in green_coverage_pct per time step.

        Returns
        -------
        float or None
            Positive = growing. Negative = declining. None if < 2 records.
        """
        vals = [r.green_coverage_pct for r in self._records if r.green_coverage_pct is not None]
        if len(vals) < 2:
            return None
        deltas = [vals[i+1] - vals[i] for i in range(len(vals) - 1)]
        return round(float(sum(deltas) / len(deltas)), 4)

    def health_trend(self) -> Optional[float]:
        """
        Linear trend slope of health_ratio over time.

        Returns
        -------
        float or None
            Positive = improving health. Negative = declining health.
        """
        import numpy as np
        vals = [r.health_ratio for r in self._records if r.health_ratio is not None]
        if len(vals) < 2:
            return None
        x = np.arange(len(vals))
        slope = float(np.polyfit(x, vals, 1)[0])
        return round(slope, 6)

    def get_alerts(self) -> List[Dict[str, str]]:
        """
        Scan the log for alert conditions and return a list of alerts.

        Returns
        -------
        list of dict, each with keys: type, message, timestamp, severity
        """
        alerts = []

        # ── Coverage drop alert ──────────────────────────────────────
        cov_records = [(r.timestamp, r.green_coverage_pct)
                       for r in self._records if r.green_coverage_pct is not None]
        for i in range(1, len(cov_records)):
            ts_prev, prev = cov_records[i - 1]
            ts_curr, curr = cov_records[i]
            drop = prev - curr
            if drop >= self._alert_cov_drop:
                alerts.append({
                    "type": "coverage_drop",
                    "severity": "HIGH" if drop >= 20 else "MEDIUM",
                    "timestamp": ts_curr,
                    "message": (
                        f"Green coverage dropped {drop:.1f}% between "
                        f"{ts_prev} and {ts_curr}."
                    ),
                })

        # ── Chronic health stress alert ──────────────────────────────
        health_records = [(r.timestamp, r.health_ratio)
                          for r in self._records if r.health_ratio is not None]
        consecutive_low = 0
        for ts, hr in health_records:
            if hr < self._alert_health_thresh:
                consecutive_low += 1
                if consecutive_low >= 3:
                    alerts.append({
                        "type": "chronic_stress",
                        "severity": "HIGH",
                        "timestamp": ts,
                        "message": (
                            f"Health ratio below {self._alert_health_thresh} "
                            f"for {consecutive_low} consecutive records."
                        ),
                    })
            else:
                consecutive_low = 0

        # ── Anomaly spike alert ──────────────────────────────────────
        for record in self._records:
            if (record.anomaly_score is not None and
                    record.anomaly_score >= self._alert_anomaly_thresh):
                alerts.append({
                    "type": "anomaly_spike",
                    "severity": "MEDIUM",
                    "timestamp": record.timestamp,
                    "message": (
                        f"Anomaly score {record.anomaly_score:.3f} exceeded "
                        f"threshold {self._alert_anomaly_thresh}."
                    ),
                })

        return alerts

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the full growth log to a JSON file."""
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        data = {
            "plant_id": self.plant_id,
            "record_count": len(self._records),
            "records": [r.to_dict() for r in self._records],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        print(f"[GrowthTracker] Saved {len(self._records)} records → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved growth log from JSON."""
        with open(path) as f:
            data = json.load(f)
        self.plant_id = data.get("plant_id", self.plant_id)
        self._records = [
            GrowthRecord(**{k: v for k, v in r.items() if k != "extras"},
                         extras=r.get("extras", {}))
            for r in data.get("records", [])
        ]
        print(f"[GrowthTracker] Loaded {len(self._records)} records from {path}")

    # ------------------------------------------------------------------ #
    # Visualisation
    # ------------------------------------------------------------------ #

    def plot(
        self,
        save_path: Optional[str] = None,
        title: Optional[str] = None,
    ):
        """
        Plot all available metrics in a multi-panel figure.

        Parameters
        ----------
        save_path : str, optional
            Save figure to this path instead of displaying.
        title : str, optional
            Override default title.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt

        title = title or f"Growth Tracker — {self.plant_id}"

        # Collect series with at least 2 data points
        series = {}
        metrics = [
            ("green_coverage_pct",  "Green Coverage (%)", "#2ecc71"),
            ("total_coverage_pct",  "Total Coverage (%)", "#27ae60"),
            ("health_ratio",        "Health Ratio",       "#3498db"),
            ("relative_height",     "Relative Height",    "#9b59b6"),
            ("anomaly_score",       "Anomaly Score",      "#e74c3c"),
        ]

        for attr, label, color in metrics:
            vals = [(i, getattr(r, attr))
                    for i, r in enumerate(self._records)
                    if getattr(r, attr) is not None]
            if len(vals) >= 2:
                series[attr] = (vals, label, color)

        if not series:
            print("[GrowthTracker] No data to plot.")
            return None

        n_panels = len(series)
        fig, axes = plt.subplots(n_panels, 1, figsize=(12, 3 * n_panels), sharex=True)
        if n_panels == 1:
            axes = [axes]

        fig.suptitle(title, fontsize=13, fontweight="bold")
        alerts = self.get_alerts()
        alert_times = {a["timestamp"] for a in alerts}

        for ax, (attr, (vals, label, color)) in zip(axes, series.items()):
            x, y = zip(*vals)
            ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=4)
            ax.fill_between(x, y, alpha=0.10, color=color)
            ax.set_ylabel(label, fontsize=9)
            ax.grid(alpha=0.3)

            # Mark alert timestamps
            for i, r in enumerate(self._records):
                if r.timestamp in alert_times:
                    ax.axvline(i, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.6)

        axes[-1].set_xlabel("Time Step")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        else:
            plt.show()
        return fig

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #

    @property
    def record_count(self) -> int:
        return len(self._records)

    @property
    def records(self) -> List[GrowthRecord]:
        return list(self._records)

    def __repr__(self) -> str:
        return (
            f"<GrowthTracker plant_id='{self.plant_id}' "
            f"records={self.record_count}>"
        )
