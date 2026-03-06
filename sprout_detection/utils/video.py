"""
sprout_detection/utils/video.py
================================
Video Frame Extraction Utility

This is the ONLY addition required to add video support to the pipeline.
The SproutCascade.analyze() method itself requires zero changes — it simply
receives a frame path instead of a photo path and processes it identically.

Workflow
--------
    video.mp4
        ↓  extract_frames(every_n_seconds=30)
    frames/frame_0000030.0s.jpg
    frames/frame_0000060.0s.jpg
    ...
        ↓  cascade.analyze(frame_path)   ← identical to photo analysis
    SproutResult

Usage
-----
    from sprout_detection.utils.video import extract_frames
    from sprout_detection.cascade import SproutCascade

    cascade = SproutCascade()
    frame_paths = extract_frames("timelapse.mp4", every_n_seconds=60)

    for frame_path in frame_paths:
        result = cascade.analyze(frame_path)
        print(result)
"""

from __future__ import annotations

import os
from typing import List, Optional

import cv2


def extract_frames(
    video_path: str,
    every_n_seconds: float = 60.0,
    output_dir: str = "frames",
    prefix: str = "frame",
    max_frames: Optional[int] = None,
) -> List[str]:
    """
    Extract frames from a video file at a fixed time interval.

    Saves each frame as a JPEG file named by its timestamp so that
    filenames sort chronologically and are meaningful at a glance:
        frame_0000030.0s.jpg  → frame at 30 seconds
        frame_0000060.0s.jpg  → frame at 60 seconds

    Parameters
    ----------
    video_path : str
        Path to the video file (any OpenCV-supported format).
    every_n_seconds : float
        Interval between extracted frames in seconds.  Default 60.
        E.g. 30 → extract one frame every 30 seconds.
    output_dir : str
        Directory where extracted JPEG frames will be saved.
        Created automatically if it does not exist.
    prefix : str
        Filename prefix for saved frames.  Default 'frame'.
    max_frames : int, optional
        Maximum number of frames to extract.  None = no limit.

    Returns
    -------
    list of str
        Sorted list of absolute paths to the saved frame files.
        Feed each path directly to SproutCascade.analyze().

    Raises
    ------
    FileNotFoundError
        If video_path does not exist.
    IOError
        If OpenCV cannot open the video file.

    Examples
    --------
    >>> paths = extract_frames("timelapse.mp4", every_n_seconds=30)
    >>> print(paths[0])
    'frames/frame_0000000.0s.jpg'
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    # ── Open video ──────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / fps if fps > 0 else 0

    if fps <= 0:
        cap.release()
        raise IOError(
            f"Could not determine FPS for video: '{video_path}'. "
            f"The file may be corrupt or in an unsupported codec."
        )

    # Number of frames to skip between extractions
    frame_interval = max(1, int(fps * every_n_seconds))

    print(f"📹 Video   : {video_path}")
    print(f"   FPS     : {fps:.1f}")
    print(f"   Duration: {duration_s:.1f} s  ({total_frames} total frames)")
    print(f"   Interval: every {every_n_seconds} s  (~{frame_interval} frames)")
    estimated = min(
        int(total_frames / frame_interval) + 1,
        max_frames or 999999,
    )
    print(f"   Expected: ~{estimated} frames → {output_dir}/")

    os.makedirs(output_dir, exist_ok=True)

    # ── Extract frames ──────────────────────────────────────────────────
    saved_paths: List[str] = []
    frame_idx = 0

    while True:
        # Seek to the desired frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            # End of video or read error
            break

        timestamp_s = frame_idx / fps
        filename = f"{prefix}_{timestamp_s:010.1f}s.jpg"
        out_path = os.path.join(output_dir, filename)

        cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(os.path.abspath(out_path))

        # Enforce max_frames limit
        if max_frames and len(saved_paths) >= max_frames:
            break

        frame_idx += frame_interval

    cap.release()

    print(f"   ✅ Extracted {len(saved_paths)} frames → '{output_dir}/'")
    return sorted(saved_paths)


def get_video_info(video_path: str) -> dict:
    """
    Return basic metadata about a video file without extracting any frames.

    Parameters
    ----------
    video_path : str
        Path to the video file.

    Returns
    -------
    dict with keys:
        path, fps, total_frames, duration_seconds, width, height, codec
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: '{video_path}'")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: '{video_path}'")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    cap.release()

    return {
        "path": video_path,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps if fps > 0 else 0,
        "width": width,
        "height": height,
        "codec": codec.strip(),
    }
