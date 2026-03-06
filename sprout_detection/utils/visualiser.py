"""
sprout_detection/utils/visualiser.py
=====================================
Visualisation Helpers for Sprout Detection Results

Provides one visualisation function per detector layer, plus a unified
function that automatically picks the right visualisation based on
SproutResult.method.

All functions follow the same signature:
    visualise_*(image_path, result, **kwargs) → matplotlib Figure

Functions
---------
visualise_hsv(image_path, result)
    Three-panel: original | green mask | detection overlay

visualise_clip(image_path, result)
    Two-panel: original | horizontal bar chart of prompt scores

visualise_gemini(image_path, result)
    Two-panel: original | reasoning text card

visualise_result(image_path, result)
    Auto-dispatch to the correct function above.

visualise_cascade_summary(results)
    Multi-row summary grid for batch / video analysis.

Usage
-----
    from sprout_detection.utils.visualiser import visualise_result
    from sprout_detection.cascade import SproutCascade

    cascade = SproutCascade()
    result = cascade.analyze("photo.jpg")
    visualise_result("photo.jpg", result)
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sprout_detection.result import SproutResult

# Consistent colour palette used across all visualisations
_GREEN = "#2ecc71"
_RED = "#e74c3c"
_GREY = "#95a5a6"
_DARK = "#2c3e50"
_LIGHT_BG = "#f8f9fa"


def visualise_result(image_path: str, result: SproutResult, **kwargs) -> plt.Figure:
    """
    Auto-dispatch visualisation based on result.method.

    Picks the correct layer-specific function automatically so callers
    don't need to know which layer produced the result.

    Parameters
    ----------
    image_path : str
        Path to the analysed image.
    result : SproutResult
        The detection result to visualise.

    Returns
    -------
    matplotlib.figure.Figure
    """
    dispatch = {
        "hsv_masking":      visualise_hsv,
        "clip_zero_shot":   visualise_clip,
        "gemini_flash_api": visualise_gemini,
    }
    func = dispatch.get(result.method, visualise_generic)
    return func(image_path, result, **kwargs)


def visualise_hsv(
    image_path: str,
    result: SproutResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Layer 1 — Three-panel visualisation.

    Panels: [Original Image] | [Green Binary Mask] | [Detection Overlay]

    Parameters
    ----------
    image_path : str
        Path to the original image.
    result : SproutResult
        Must be a result from HSVDetector (method='hsv_masking').
        result.hsv_mask is used for the mask panel.
    save_path : str, optional
        If provided, save the figure to this path instead of showing it.

    Returns
    -------
    matplotlib.figure.Figure
    """
    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = result.hsv_mask  # numpy binary mask

    # Build green overlay blended with original
    overlay = img_rgb.copy()
    if mask is not None:
        overlay[mask > 0] = [0, 220, 80]
    blended = cv2.addWeighted(img_rgb, 0.65, overlay, 0.35, 0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor(_LIGHT_BG)

    _add_result_suptitle(fig, result, layer_label="Layer 1 — HSV Masking")

    # Panel 1: original
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image", fontsize=11, color=_DARK)
    axes[0].axis("off")

    # Panel 2: green mask
    axes[1].imshow(mask if mask is not None else np.zeros_like(img_rgb[:, :, 0]),
                   cmap="Greens", vmin=0, vmax=255)
    ratio_str = f"{result.green_ratio:.4f}" if result.green_ratio is not None else "N/A"
    axes[1].set_title(f"Green Mask  (ratio = {ratio_str})", fontsize=11, color=_DARK)
    axes[1].axis("off")

    # Panel 3: overlay
    axes[2].imshow(blended)
    axes[2].set_title("Detection Overlay", fontsize=11, color=_DARK)
    axes[2].axis("off")

    # Reasoning text below the panels
    fig.text(
        0.5, 0.01,
        f"Reasoning: {result.reasoning}",
        ha="center", va="bottom", fontsize=9, color=_DARK,
        style="italic", wrap=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()




def visualise_clip(
    image_path: str,
    result: SproutResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Layer 2 — Two-panel visualisation.

    Panels: [Original Image] | [CLIP Prompt Score Bar Chart]

    Green bars = "sprout" prompts.  Grey bars = "no sprout" prompts.

    Parameters
    ----------
    image_path : str
        Path to the original image.
    result : SproutResult
        Must be a result from CLIPDetector (method='clip_zero_shot').
        result.clip_scores is used for the bar chart.
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    scores = result.clip_scores or {}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.patch.set_facecolor(_LIGHT_BG)

    _add_result_suptitle(fig, result, layer_label="Layer 2 — CLIP Zero-Shot")

    # Panel 1: original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=11, color=_DARK)
    axes[0].axis("off")

    # Panel 2: bar chart of CLIP prompt scores
    if scores:
        # Determine which prompts are "sprout" prompts for colour coding
        from config import CONFIG
        sprout_indices = set(CONFIG.get("clip_sprout_prompt_indices", []))
        prompts_list = list(scores.keys())
        values = [scores[p] for p in prompts_list]
        colors = [
            _GREEN if i in sprout_indices else _GREY
            for i in range(len(prompts_list))
        ]

        bars = axes[1].barh(
            [p[:45] for p in prompts_list],  # Truncate long labels
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        axes[1].set_xlim(0, max(values) * 1.25)
        axes[1].set_xlabel("Softmax Probability", fontsize=10)
        axes[1].set_title(
            "Prompt Similarity Scores\n( = sprout prompts)",
            fontsize=11, color=_DARK,
        )
        axes[1].bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
        axes[1].tick_params(labelsize=9)

        # Legend
        legend_handles = [
            mpatches.Patch(color=_GREEN, label="Sprout prompt"),
            mpatches.Patch(color=_GREY,  label="No-sprout prompt"),
        ]
        axes[1].legend(handles=legend_handles, loc="lower right", fontsize=9)
    else:
        axes[1].text(0.5, 0.5, "No CLIP scores available",
                     ha="center", va="center", transform=axes[1].transAxes)
        axes[1].axis("off")

    fig.text(
        0.5, 0.01,
        f"Reasoning: {result.reasoning}",
        ha="center", va="bottom", fontsize=9, color=_DARK,
        style="italic", wrap=True,
    )

    plt.tight_layout(rect=[0, 0.05, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()




def visualise_gemini(
    image_path: str,
    result: SproutResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Layer 3 — Two-panel visualisation.

    Panels: [Original Image] | [API Result Card with reasoning text]

    Parameters
    ----------
    image_path : str
        Path to the original image.
    result : SproutResult
        Must be a result from GeminiDetector (method='gemini_flash_api').
    save_path : str, optional
        If provided, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
    """
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(_LIGHT_BG)

    _add_result_suptitle(fig, result, layer_label="Layer 3 — Gemini Flash API")

    # Panel 1: original image
    axes[0].imshow(img_rgb)
    axes[0].set_title("Input Image", fontsize=11, color=_DARK)
    axes[0].axis("off")

    # Panel 2: result card
    axes[1].axis("off")
    card_color = _GREEN if result.sprout_detected else _RED
    card_bg = "#eafaf1" if result.sprout_detected else "#fdedec"

    axes[1].set_facecolor(card_bg)
    icon = "🌱" if result.sprout_detected else "🪨"
    detected_str = "SPROUT DETECTED" if result.sprout_detected else "NO SPROUT"

    card_text = (
        f"{icon} conf={result.confidence:.2f}\n"
        f"Confidence : {result.confidence:.2f}\n"
        f"Method     : {result.method}\n"
        f"Escalated  : {result.escalated}\n\n"
        f"Reasoning:\n{result.reasoning}"
    )

    axes[1].text(
        0.5, 0.5, card_text,
        ha="center", va="center",
        transform=axes[1].transAxes,
        fontsize=11, color=_DARK,
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor=card_bg,
                  edgecolor=card_color, linewidth=2),
    )
    axes[1].set_title("Gemini API Result", fontsize=11, color=_DARK)

    plt.tight_layout(rect=[0, 0.03, 1, 0.93])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()




def visualise_generic(
    image_path: str,
    result: SproutResult,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Fallback visualisation for unrecognised method names.
    Shows the image and a text summary of the result.
    """
    img_rgb = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    _add_result_suptitle(fig, result, layer_label=f"Method: {result.method}")
    axes[0].imshow(img_rgb)
    axes[0].axis("off")
    axes[1].axis("off")
    axes[1].text(0.5, 0.5, result.to_json(), ha="center", va="center",
                 transform=axes[1].transAxes, fontsize=8, fontfamily="monospace")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    return fig


def visualise_cascade_summary(
    image_paths: List[str],
    results: List[SproutResult],
    max_cols: int = 4,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Multi-image grid summary, one thumbnail per analysed frame/image.

    Useful for visualising the output of a batch or video analysis.
    Each thumbnail is annotated with the detection result and confidence.

    Parameters
    ----------
    image_paths : list of str
        Paths to the original images.
    results : list of SproutResult
        Corresponding detection results.
    max_cols : int
        Maximum number of columns in the grid.  Default 4.
    save_path : str, optional
        If provided, save the figure.

    Returns
    -------
    matplotlib.figure.Figure
    """
    n = len(results)
    if n == 0:
        raise ValueError("No results to visualise.")

    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.5))
    fig.patch.set_facecolor(_LIGHT_BG)
    fig.suptitle(
        "Sprout Detection — Cascade Summary",
        fontsize=14, fontweight="bold", color=_DARK, y=1.01,
    )

    # Normalise axes to always be a flat list
    if rows == 1 and cols == 1:
        axes = [axes]
    elif rows == 1 or cols == 1:
        axes = list(axes.flat)
    else:
        axes = [ax for row in axes for ax in row]

    for i, (path, result) in enumerate(zip(image_paths, results)):
        ax = axes[i]
        img_bgr = cv2.imread(path)
        if img_bgr is not None:
            ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        else:
            ax.set_facecolor("#eeeeee")

        ax.axis("off")
        color = _GREEN if result.sprout_detected else _RED
        icon = "🌱" if result.sprout_detected else "🪨"
        label = (
            f"{icon} conf={result.confidence:.2f}\n"
            f"{result.method[:18]}"
        )
        ax.set_title(label, fontsize=8, color=color, fontweight="bold", pad=3)

        # Coloured border on each thumbnail
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
            spine.set_visible(True)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    return fig


# ── Private helper ──────────────────────────────────────────────────────────

def _add_result_suptitle(fig: plt.Figure, result: SproutResult, layer_label: str) -> None:
    """Add a standardised suptitle showing detection outcome and confidence."""
    icon = "SPROUT DETECTED" if result.sprout_detected else "NO SPROUT"
    color = _GREEN if result.sprout_detected else _RED
    fig.suptitle(
        f"{layer_label}   |   {icon}   |   Confidence: {result.confidence:.2f}",
        fontsize=13,
        fontweight="bold",
        color=color,
    )
