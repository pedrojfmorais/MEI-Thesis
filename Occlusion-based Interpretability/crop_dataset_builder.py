#!/usr/bin/env python3
"""
Face Cropper with MTCNN (Multiprocessing)
=========================================

Crops faces from images in a directory (with infinite subdirectories) using MTCNN.
Crops are saved preserving folder structure, logs are created, and processing time is tracked.
Supports multiprocessing for faster processing.
"""

import argparse
import os
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
from mtcnn import MTCNN
import tensorflow as tf
from PIL import Image

import warnings

warnings.warn(
    """
⚠️ WARNING: Do NOT use Google Drive as the direct output path for large datasets.

Writing thousands of files directly to Drive in Colab can:
  - Be extremely slow due to network-backed file system
  - Cause incomplete writes or missing files even after waiting
  - Be unreliable with multiprocessing

✅ Recommended workflow for Colab + Drive:
  1. Save all files locally in /content (Colab ephemeral storage)
  2. Zip the entire dataset into a single archive
  3. Copy or move the .zip file to Google Drive
  4. Unzip locally when needed, not inside Drive

This ensures faster, more reliable writes and avoids sync issues.
""",
    UserWarning
)

# -----------------------------
# Redirect TensorFlow logs to logging
# -----------------------------
class TFLogger(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        logging.log(record.levelno, log_entry)

# -----------------------------
# Utility: IO & filesystem
# -----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def iter_image_files(root: Path):
    """Yield (absolute_path, relative_path) for all images recursively."""
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield p, p.relative_to(root)

def ensure_parent_dir(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Face cropping
# -----------------------------

def crop_face(abs_path: str):
    """Detect the first face and return cropped image bytes or None."""
    try:
        detector = MTCNN()
        img = Image.open(abs_path)
        img_array = np.array(img)
        detections = detector.detect_faces(img_array)
        if not detections:
            logging.warning(f"No face detected: {abs_path}")
            return None
        x, y, w, h = detections[0]['box']
        cropped = img.crop((x, y, x + w, y + h))
        return cropped
    except Exception as e:
        logging.error(f"Failed processing {abs_path}: {e}")
        return None

def process_one_image(args):
    abs_path, rel_path, output_root = args
    cropped = crop_face(str(abs_path))
    if cropped is None:
        return False
    try:
        out_path = output_root / rel_path
        ensure_parent_dir(out_path)
        cropped.save(out_path, "PNG")
        return True
    except Exception as e:
        logging.error(f"Failed saving {rel_path}: {e}")
        return False

# -----------------------------
# Processing pipeline
# -----------------------------

def run(input_path: Path, output_path: Path, multiprocessing: bool, workers: int):
    images = list(iter_image_files(input_path))
    total_in = len(images)
    logging.info(f"Found {total_in} images under {input_path}")

    saved_total = 0
    args_list = [(abs_path, rel_path, output_path) for abs_path, rel_path in images]

    if multiprocessing:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_one_image, arg): arg for arg in args_list}
            for i, fut in enumerate(as_completed(futures), 1):
                if fut.result():
                    saved_total += 1
                if i % 50 == 0:
                    logging.info(f"Processed {i}/{total_in} images...")
    else:
        # Single-process fallback
        for i, arg in enumerate(args_list, 1):
            if process_one_image(arg):
                saved_total += 1
            if i % 50 == 0:
                logging.info(f"Processed {i}/{total_in} images...")

    logging.info(f"Done. Saved {saved_total} cropped images total.")
    return saved_total

# -----------------------------
# CLI
# -----------------------------

def build_argparser():
    p = argparse.ArgumentParser(description="Crop faces from images using MTCNN")
    p.add_argument("--input_path", required=True, type=Path, help="Root folder with images")
    p.add_argument("--output_path", required=True, type=Path, help="Root folder to save cropped faces")
    p.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing.")
    p.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker processes")
    return p

def main():
    args = build_argparser().parse_args()
    input_path: Path = args.input_path
    output_path: Path = args.output_path
    multiprocessing: bool = args.multiprocessing
    workers: int = args.workers

    if not input_path.exists() or not input_path.is_dir():
        raise SystemExit(f"Input path does not exist or is not a directory: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Logging setup
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    log_file = output_path / f"face_crop_log_{timestamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    # Redirect TensorFlow logging to Python logging
    tf_logger = tf.get_logger()
    tf_logger.handlers = []  # remove default handlers
    tf_logger.addHandler(logging.FileHandler(log_file))  # send TF Python logs to file
    tf_logger.setLevel(logging.INFO)

    start_time = time.time()
    logging.info("=== Face cropping started ===")
    saved_total = run(input_path, output_path, multiprocessing, workers)
    end_time = time.time()

    elapsed = end_time - start_time
    elapsed_readable = timedelta(seconds=elapsed)
    avg_time_per_image = elapsed / max(saved_total, 1)  # avoid divide by zero

    logging.info("=== Face cropping finished ===")
    logging.info(f"Start time: {datetime.fromtimestamp(start_time)}")
    logging.info(f"End time:   {datetime.fromtimestamp(end_time)}")
    logging.info(f"Time elapsed: {elapsed_readable} ({elapsed:.6f} seconds)")
    logging.info(f"Average time per cropped image: {avg_time_per_image:.10f} seconds")

if __name__ == "__main__":
    main()
