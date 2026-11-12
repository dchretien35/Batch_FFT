#!/usr/bin/env python3
"""Batch FFT image processing helper.

This lightweight utility processes a folder of scientific images, reduces their
size in the frequency domain, and boosts contrast so details stand out more
clearly during review.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    np = None

try:
    from PIL import Image
except Exception:  # pragma: no cover - optional runtime dependency
    Image = None


def _require_dependencies() -> None:
    if Image is None or np is None:
        raise RuntimeError(
            "batch_fft.py requires pillow and numpy. Install them with: "
            "pip install pillow numpy"
        )


def load_image(path: str | Path):
    """Open an image and return a floating-point array for FFT processing."""
    _require_dependencies()
    image = Image.open(path).convert("L")
    return np.asarray(image, dtype=np.float32)


def process_array(image_array, target_size=None, contrast=1.0):
    """Apply a lightweight FFT-based enhancement on a 2D image array."""
    _require_dependencies()
    array = np.asarray(image_array, dtype=np.float32)
    if array.ndim == 3:
        array = np.mean(array, axis=2)

    if target_size is not None:
        target_height, target_width = target_size
        height, width = array.shape
        max_height = min(height, target_height)
        max_width = min(width, target_width)
        array = array[:max_height, :max_width]

    fft = np.fft.fftshift(np.fft.fft2(array))
    magnitude = np.log1p(np.abs(fft))
    magnitude = np.clip(magnitude * contrast, 0, 255)
    return np.asarray(magnitude, dtype=np.uint8)


def process_image(path: str | Path, output_path: str | Path, target_size=None, contrast=1.0):
    """Process a single image and write the FFT-enhanced output to disk."""
    source = Path(path)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    image_data = load_image(source)
    processed = process_array(image_data, target_size=target_size, contrast=contrast)
    Image.fromarray(processed, mode="L").save(target)


def process_image_batch(input_dir, output_dir, target_size=None, contrast=1.0, recursive=False):
    """Batch-process images found in a directory."""
    source_dir = Path(input_dir)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp"]
    files = []
    for pattern in patterns:
        files.extend(source_dir.glob(pattern) if not recursive else source_dir.rglob(pattern))

    unique_files = sorted({path.resolve(): path for path in files}.values())
    processed = []
    for file_path in unique_files:
        output_file = destination_dir / file_path.name
        process_image(file_path, output_file, target_size=target_size, contrast=contrast)
        processed.append(output_file)
    return processed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Batch FFT preprocessing for microscopy or scientific images."
    )
    parser.add_argument(
        "--input-dir",
        default=".",
        help="Folder containing source images (default: current directory)",
    )
    parser.add_argument(
        "--output-dir",
        default="./batch_fft_output",
        help="Folder that will receive processed images",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=1024,
        help="Maximum output width to keep when reducing image size",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=1024,
        help="Maximum output height to keep when reducing image size",
    )
    parser.add_argument(
        "--contrast",
        type=float,
        default=1.5,
        help="Contrast boost multiplier applied after the FFT",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Visit subdirectories when collecting images",
    )
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        processed_files = process_image_batch(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            target_size=(args.target_height, args.target_width),
            contrast=args.contrast,
            recursive=args.recursive,
        )
    except RuntimeError as exc:  # pragma: no cover - runtime guard
        parser.exit(1, f"{exc}\n")

    print(f"Processed {len(processed_files)} image(s). Output: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
