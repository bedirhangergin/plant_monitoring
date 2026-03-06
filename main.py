"""
main.py
=======
Command-Line Entry Point — Plant Growth Monitor

Provides a simple CLI for running sprout detection on one or more images,
or on a video file, without writing any Python code.

Usage
-----
Analyse a single image:
    python main.py --image path/to/photo.jpg

Analyse a batch of images:
    python main.py --image img1.jpg img2.jpg img3.jpg

Analyse frames from a video:
    python main.py --video timelapse.mp4 --interval 30

Show the result log:
    python main.py --log

Clear the result log:
    python main.py --clear-log

Generate sample test images and run the pipeline on them:
    python main.py --demo

Options
-------
--image / -i      One or more image file paths to analyse.
--video / -v      Path to a video file.
--interval        Frame extraction interval in seconds (default: 60).
--threshold       Override the cascade confidence threshold (default: 0.60).
--no-gemini       Disable Layer 3 (Gemini API) even if API key is set.
--log             Print the detection log summary.
--clear-log       Erase all log entries.
--demo            Run a quick demo using generated synthetic images.
--verbose / -V    Show detailed cascade progress (default: True).
--quiet / -q      Suppress per-step cascade messages.
"""

import argparse
import sys
import os

# ── Add project root to path so this works when run from any directory ──────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import CONFIG
from sprout_detection.cascade import SproutCascade
from sprout_detection.utils.image_gen import make_batch
from sprout_detection.utils.video import extract_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="🌱 Plant Growth Monitor — Sprout Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--image", "-i",
        nargs="+",
        metavar="PATH",
        help="One or more image file paths to analyse.",
    )
    input_group.add_argument(
        "--video", "-v",
        metavar="PATH",
        help="Path to a video file for frame extraction + analysis.",
    )
    input_group.add_argument(
        "--demo",
        action="store_true",
        help="Run a demo using synthetically generated test images.",
    )

    # Video options
    parser.add_argument(
        "--interval",
        type=float,
        default=60.0,
        metavar="SECONDS",
        help="Frame extraction interval for --video mode (default: 60).",
    )
    parser.add_argument(
        "--frames-dir",
        default="frames",
        metavar="DIR",
        help="Output directory for extracted video frames (default: frames/).",
    )

    # Cascade options
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        metavar="FLOAT",
        help="Override cascade confidence threshold (default: 0.60).",
    )
    parser.add_argument(
        "--no-gemini",
        action="store_true",
        help="Disable Layer 3 (Gemini API) regardless of API key.",
    )

    # Log options
    parser.add_argument(
        "--log",
        action="store_true",
        help="Print a summary table of all logged detections.",
    )
    parser.add_argument(
        "--clear-log",
        action="store_true",
        help="Delete all entries from the detection log.",
    )

    # Verbosity
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "--verbose", "-V",
        action="store_true",
        default=True,
        help="Show detailed cascade progress (default: on).",
    )
    verbosity.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress per-step cascade messages.",
    )

    return parser.parse_args()


def build_cascade(args: argparse.Namespace) -> SproutCascade:
    """Construct a SproutCascade with the options from CLI args."""
    config = dict(CONFIG)  # Shallow copy so we don't mutate the global

    if args.threshold is not None:
        config["confidence_threshold"] = args.threshold
        print(f"⚙️  Confidence threshold set to {args.threshold}")

    if args.no_gemini:
        config["gemini_api_key"] = ""
        print("⚙️  Layer 3 (Gemini API) disabled via --no-gemini")

    verbose = not args.quiet
    return SproutCascade(config=config, verbose=verbose)


def run_on_images(cascade: SproutCascade, image_paths: list) -> None:
    """Analyse a list of images and print a final summary."""
    print(f"\n📸 Analysing {len(image_paths)} image(s)...\n")
    results = cascade.analyze_batch(image_paths)

    # ── Summary ─────────────────────────────────────────────────────────
    detected = sum(1 for r in results if r.sprout_detected)
    print("\n" + "═" * 55)
    print(f"  SUMMARY: {detected}/{len(results)} images show a sprout")
    print("═" * 55)
    for r in results:
        print(f"  {r.summary_line}")
    print()


def run_demo(cascade: SproutCascade) -> None:
    """Generate synthetic images and run the full cascade on them."""
    print("\n🎬 Demo mode — generating synthetic test images...\n")
    demo_dir = "assets/sample_images/demo"
    image_paths = make_batch(demo_dir, n_sprout=2, n_bare=2, n_ambiguous=2)
    run_on_images(cascade, image_paths)


def main() -> None:
    args = parse_args()
    cascade = build_cascade(args)

    # ── Handle log-only commands first ──────────────────────────────────
    if args.clear_log:
        cascade.clear_log()
        return

    if args.log:
        cascade.print_log_summary()
        return

    # ── Determine image paths ────────────────────────────────────────────
    if args.demo:
        run_demo(cascade)
        return

    if args.video:
        print(f"\n📹 Extracting frames from: {args.video}")
        image_paths = extract_frames(
            video_path=args.video,
            every_n_seconds=args.interval,
            output_dir=args.frames_dir,
        )
        run_on_images(cascade, image_paths)
        return

    if args.image:
        run_on_images(cascade, args.image)
        return

    # No input provided — show help
    print("No input specified.  Use --help for usage information.")
    print("Quick start:  python main.py --demo")


if __name__ == "__main__":
    main()
