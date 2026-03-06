"""
sprout_detection/utils/image_gen.py
====================================
Synthetic Test Image Generator

Generates programmatic test images so the full pipeline can be validated
without needing real plant photos.  Used by both the test suite and the
demonstration notebooks.

All functions write JPEG files to disk and return the file path so they
can be passed directly to any detector or to SproutCascade.analyze().

Available generators
--------------------
make_sprout_image(path)
    Clear green sprout on brown soil background.  HSV Layer 1 should
    detect this with high confidence.

make_bare_soil_image(path)
    Brown/grey soil with no green.  Should return sprout_detected=False.

make_ambiguous_image(path)
    A tiny speck of green on a noisy background — designed to sit near
    the HSV decision boundary to trigger cascade escalation.

make_batch(output_dir, n_sprout, n_bare, n_ambiguous)
    Generate a mixed batch suitable for batch/video testing.

Usage
-----
    from sprout_detection.utils.image_gen import (
        make_sprout_image, make_bare_soil_image, make_ambiguous_image
    )

    sprout_path = make_sprout_image("assets/sample_images/sprout.jpg")
    bare_path   = make_bare_soil_image("assets/sample_images/bare.jpg")
"""

from __future__ import annotations

import os
from typing import List, Tuple

import cv2
import numpy as np


# ── Public generators ───────────────────────────────────────────────────────

def make_sprout_image(
    path: str,
    size: Tuple[int, int] = (400, 400),
    seed: int = 42,
) -> str:
    """
    Create an image of a clear green sprout on brown soil.

    The sprout is drawn as:
      - A dark green stem (vertical rectangle)
      - Two leaf ellipses (bright green, slightly rotated)
      - A textured soil background

    This image is designed to produce a confident detection at Layer 1
    (HSV masking) because the green content is unambiguous.

    Parameters
    ----------
    path : str
        File path to save the image (JPEG).
    size : (width, height)
        Image dimensions in pixels.
    seed : int
        Random seed for reproducible noise texture.

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    np.random.seed(seed)
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # ── Soil background (dark brown with subtle texture) ─────────────────
    img[:] = (40, 55, 80)   # BGR: dark brown
    noise = np.random.randint(0, 25, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Add horizontal soil striations for realism
    for y in range(0, h, 12):
        thickness = np.random.randint(1, 3)
        brightness = np.random.randint(-15, 15)
        color = tuple(int(c + brightness) for c in (40, 55, 80))
        cv2.line(img, (0, y), (w, y), color, thickness)

    cx = w // 2  # Centre x

    # ── Stem (dark green vertical bar) ───────────────────────────────────
    stem_top = h // 3
    stem_bottom = int(h * 0.72)
    cv2.rectangle(img, (cx - 8, stem_top), (cx + 8, stem_bottom), (20, 130, 25), -1)

    # ── Left leaf (angled ellipse) ────────────────────────────────────────
    cv2.ellipse(
        img,
        center=(cx - 55, stem_top + 10),
        axes=(55, 22),
        angle=-25,
        startAngle=0, endAngle=360,
        color=(35, 175, 40), thickness=-1,
    )

    # ── Right leaf ────────────────────────────────────────────────────────
    cv2.ellipse(
        img,
        center=(cx + 55, stem_top - 5),
        axes=(55, 22),
        angle=25,
        startAngle=0, endAngle=360,
        color=(45, 195, 50), thickness=-1,
    )

    # ── Leaf veins (thin darker lines for detail) ─────────────────────────
    cv2.line(img, (cx, stem_top + 10), (cx - 90, stem_top + 25), (20, 120, 20), 1)
    cv2.line(img, (cx, stem_top - 5), (cx + 90, stem_top + 10), (20, 120, 20), 1)

    _save(img, path)
    return os.path.abspath(path)


def make_bare_soil_image(
    path: str,
    size: Tuple[int, int] = (400, 400),
    seed: int = 99,
) -> str:
    """
    Create an image of bare soil with no plant material.

    The image contains only brown and grey tones — no green pixels in the
    HSV range used by HSVDetector.  Should return sprout_detected=False
    with high confidence at Layer 1.

    Parameters
    ----------
    path : str
        File path to save the image (JPEG).
    size : (width, height)
        Image dimensions in pixels.
    seed : int
        Random seed for reproducible noise texture.

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    np.random.seed(seed)
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Base soil colour
    img[:] = (45, 62, 90)   # BGR: medium brown

    # Add noise and texture variation
    noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Pebble-like darker blobs
    for _ in range(30):
        cx = np.random.randint(20, w - 20)
        cy = np.random.randint(20, h - 20)
        rx = np.random.randint(5, 18)
        ry = np.random.randint(4, 12)
        shade = np.random.randint(20, 55)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (shade, shade + 10, shade + 20), -1)

    _save(img, path)
    return os.path.abspath(path)


def make_ambiguous_image(
    path: str,
    size: Tuple[int, int] = (400, 400),
    green_ratio: float = 0.012,
    seed: int = 7,
) -> str:
    """
    Create an image designed to sit right at the HSV detection boundary.

    A tiny patch of green (just above the default 1 % threshold) is placed
    on a noisy background.  Layer 1 will detect it but with low confidence,
    triggering cascade escalation to Layer 2.

    This is the most useful image type for testing cascade escalation logic.

    Parameters
    ----------
    path : str
        File path to save the image (JPEG).
    size : (width, height)
        Image dimensions in pixels.
    green_ratio : float
        Target fraction of green pixels.  Default 0.012 (just above 0.01 threshold).
    seed : int
        Random seed.

    Returns
    -------
    str
        Absolute path to the saved image.
    """
    np.random.seed(seed)
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Mixed soil/leaf-litter background (some brownish-green noise)
    img[:] = (50, 65, 75)
    noise = np.random.randint(0, 50, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)

    # Calculate how many green pixels we need
    target_pixels = int(w * h * green_ratio)
    # Place a small green patch at a random location
    patch_side = int(np.sqrt(target_pixels))
    px = np.random.randint(10, w - patch_side - 10)
    py = np.random.randint(10, h - patch_side - 10)

    # Draw the tiny green patch with some colour variation
    for dy in range(patch_side):
        for dx in range(patch_side):
            g_val = np.random.randint(130, 180)
            img[py + dy, px + dx] = (20, g_val, 30)   # BGR green

    _save(img, path)
    return os.path.abspath(path)


def make_batch(
    output_dir: str,
    n_sprout: int = 3,
    n_bare: int = 3,
    n_ambiguous: int = 2,
) -> List[str]:
    """
    Generate a mixed batch of test images for batch/video analysis testing.

    Parameters
    ----------
    output_dir : str
        Directory to save all generated images.
    n_sprout : int
        Number of sprout images to generate.
    n_bare : int
        Number of bare soil images to generate.
    n_ambiguous : int
        Number of ambiguous/borderline images to generate.

    Returns
    -------
    list of str
        Sorted list of all generated image paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    paths = []

    for i in range(n_sprout):
        p = os.path.join(output_dir, f"sprout_{i+1:02d}.jpg")
        paths.append(make_sprout_image(p, seed=i * 10))

    for i in range(n_bare):
        p = os.path.join(output_dir, f"bare_{i+1:02d}.jpg")
        paths.append(make_bare_soil_image(p, seed=i * 10 + 100))

    for i in range(n_ambiguous):
        p = os.path.join(output_dir, f"ambiguous_{i+1:02d}.jpg")
        # Vary the green ratio slightly around the threshold
        ratio = 0.009 + i * 0.004
        paths.append(make_ambiguous_image(p, green_ratio=ratio, seed=i * 10 + 200))

    print(
        f"✅ Generated {len(paths)} test images → '{output_dir}/'\n"
        f"   {n_sprout} sprout  |  {n_bare} bare  |  {n_ambiguous} ambiguous"
    )
    return sorted(paths)


# ── Private helper ──────────────────────────────────────────────────────────

def _save(img: np.ndarray, path: str) -> None:
    """Create parent directories and save image as JPEG."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
