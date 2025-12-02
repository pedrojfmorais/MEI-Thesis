#!/usr/bin/env python3
"""
Occlusion Dataset Builder
=========================

Builds occlusion-augmented datasets from a root folder of images (with any depth of subfolders).
Supports three modes:
  - grid:     occlude one grid cell at a time -> r{row}_c{col}/.../image.jpg
  - landmark: occlude semantic facial regions using dlib 68 landmarks -> left_eye/, right_eye/, etc.
  - both:     run both of the above in one go.

Output hierarchy is preserved under each occlusion-label folder.

Dependencies:
  - Python 3.8+
  - numpy, opencv-python
  - (for landmark mode) dlib + a 68-face-landmarks model file (shape_predictor_68_face_landmarks.dat)

Example:
  python occlusion_dataset_builder.py \
      --mode both \
      --input_path /path/to/dataset_root \
      --output_path /path/to/output_root \
      --grid_rows_count 4 \
      --grid_columns_count 4 \
      --landmark_scale_factor 1.5 \
      --landmark_precision bbox \
      --multiprocessing --workers 8

Notes:
  - Each input image yields N outputs per occlusion label (one per cell for grid; one per region for landmarks).
  - Images are saved with the same filename into the appropriate occlusion label folder while preserving subfolders.
  - On errors, the file is skipped and a warning is printed.
"""

import argparse
import concurrent.futures as futures
import functools
import os
from pathlib import Path
from typing import Iterable, List, Tuple, Dict, Optional
import urllib.request
import bz2

import cv2
import numpy as np

import time
import logging
from datetime import datetime, timedelta

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

# Optional imports for landmark mode (imported lazily per process)
_dlib = None
_predictor = None

# -----------------------------
# Auto-download shape predictor
# -----------------------------
PREDICTOR_URL = "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
PREDICTOR_BZ2 = Path("temp/shape_predictor_68_face_landmarks.dat.bz2")
PREDICTOR_DAT = Path("temp/shape_predictor_68_face_landmarks.dat")

def ensure_shape_predictor() -> str:
    """Ensure the dlib shape predictor file exists, downloading if necessary."""
    if not PREDICTOR_DAT.exists():
        PREDICTOR_DAT.parent.mkdir(parents=True, exist_ok=True)
        print("Downloading Dlib 68-point shape predictor...")
        urllib.request.urlretrieve(PREDICTOR_URL, PREDICTOR_BZ2)
        print("Extracting predictor...")
        with bz2.open(PREDICTOR_BZ2, "rb") as f_in, open(PREDICTOR_DAT, "wb") as f_out:
            f_out.write(f_in.read())
        PREDICTOR_BZ2.unlink()  # remove bz2 after extraction
        print("✅ Model downloaded and extracted successfully!")
    return str(PREDICTOR_DAT)

# -----------------------------
# Utility: IO & filesystem
# -----------------------------

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def iter_image_files(root: Path) -> Iterable[Tuple[Path, Path]]:
    """Yield (abs_path, relative_path) for every image under root (recursively)."""
    root = root.resolve()
    for p in root.rglob("*"):
        if p.is_file() and is_image_file(p):
            yield (p, p.relative_to(root))

def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Grid occlusion
# -----------------------------

def occlude_grid_cell(img: np.ndarray, rows: int, cols: int, r: int, c: int) -> np.ndarray:
    """Return a copy of img where the cell (r, c) is occluded (set to 0)."""
    h, w = img.shape[:2]
    y0 = int(round(r * h / rows))
    y1 = int(round((r + 1) * h / rows))
    x0 = int(round(c * w / cols))
    x1 = int(round((c + 1) * w / cols))
    occluded = img.copy()
    occluded[y0:y1, x0:x1] = 0
    return occluded

def grid_labels(rows: int, cols: int) -> List[str]:
    return [f"r{r}_c{c}" for r in range(rows) for c in range(cols)]

def generate_grid_occlusions(img: np.ndarray, rows: int, cols: int) -> List[Tuple[str, np.ndarray]]:
    """Produce one occluded image per grid cell, labelled by r{r}_c{c}."""
    outs = []
    for r in range(rows):
        for c in range(cols):
            label = f"r{r}_c{c}"
            out = occlude_grid_cell(img, rows, cols, r, c)
            outs.append((label, out))
    return outs


# -----------------------------
# Landmark occlusion (dlib 68)
# -----------------------------

# 68-point landmark indices by region
DLIB_REGION_IDXS: Dict[str, List[int]] = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose": list(range(27, 36)),           # full nose (bridge + base)
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
    "mouth": list(range(48, 68)),          # combined
}

# Default set used for occlusions (kept focused to common regions)
DEFAULT_LANDMARK_REGIONS = [
    "left_eye",
    "right_eye",
    "nose",
    "mouth",
]

def _lazy_load_dlib(shape_predictor_path: str):
    global _dlib, _predictor
    if _dlib is None:
        import dlib as _d
        _dlib = _d
    if _predictor is None:
        if not os.path.exists(shape_predictor_path):
            raise FileNotFoundError(f"Shape predictor not found: {shape_predictor_path}")
        _predictor = _dlib.shape_predictor(shape_predictor_path)
    return _dlib, _predictor

def detect_landmarks_dlib(img_bgr: np.ndarray, shape_predictor_path: str) -> Optional[np.ndarray]:
    """Return (68,2) ndarray of landmark points in image coordinate space, or None."""
    dlib, predictor = _lazy_load_dlib(shape_predictor_path)
    detector = dlib.get_frontal_face_detector()
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    if len(rects) == 0:
        return None
    # take the largest face (by area)
    rect = max(rects, key=lambda r: r.width() * r.height())
    shape = predictor(gray, rect)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.float32)
    return pts

def scale_polygon(points: np.ndarray, scale: float) -> np.ndarray:
    """Scale polygon around its centroid by 'scale' factor."""
    if len(points) == 0:
        return points
    centroid = points.mean(axis=0, keepdims=True)
    return (points - centroid) * scale + centroid

def polygon_mask(img_shape: Tuple[int, int, int], polygon: np.ndarray) -> np.ndarray:
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    if polygon.shape[0] >= 3:
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 255)
    return mask

def occlude_bbox(img: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Occlude the bounding box of the given points."""
    if points.size == 0:
        return img.copy()
    x0, y0 = points.min(axis=0)
    x1, y1 = points.max(axis=0)
    x0, y0 = int(x0), int(y0)
    x1, y1 = int(x1), int(y1)
    occluded = img.copy()
    occluded[y0:y1, x0:x1] = 0
    return occluded

def occlude_region(img: np.ndarray, region_poly: np.ndarray) -> np.ndarray:
    """Return a copy of img with region (polygon) set to 0."""
    occluded = img.copy()
    mask = polygon_mask(img.shape, region_poly)
    occluded[mask > 0] = 0
    return occluded

def generate_landmark_occlusions(
    img: np.ndarray,
    landmark_precision: str,
    shape_predictor_path: str,
    landmark_scale_factor: float,
    use_regions: Optional[List[str]] = None,
) -> List[Tuple[str, np.ndarray]]:
    """Produce occluded images for each requested facial region."""
    if use_regions is None:
        use_regions = DEFAULT_LANDMARK_REGIONS

    pts = detect_landmarks_dlib(img, shape_predictor_path)
    if pts is None:
        return []  # no face; skip

    outs: List[Tuple[str, np.ndarray]] = []
    for region in use_regions:
        idxs = DLIB_REGION_IDXS.get(region)
        if not idxs:
            continue
        region_pts = pts[idxs, :]
        region_pts = scale_polygon(region_pts, landmark_scale_factor)
        region_pts = np.clip(region_pts, [0, 0], [img.shape[1]-1, img.shape[0]-1])

        if landmark_precision == "pixel":
            occluded = occlude_region(img, region_pts)
        elif landmark_precision == "bbox":
            occluded = occlude_bbox(img, region_pts)
        else:
            raise ValueError(f"Unknown landmark_precision: {landmark_precision}")

        outs.append((region, occluded))
    return outs



# -----------------------------
# Processing pipeline
# -----------------------------

def process_one_image(
    abs_path: Path,
    rel_path: Path,
    mode: str,
    output_root: Path,
    grid_rows: int,
    grid_cols: int,
    lmk_scale: float,
    shape_predictor: Optional[str],
    landmark_precision: Optional[str],
) -> int:
    """Process a single image, save outputs, return count of saved images."""
    img = cv2.imread(str(abs_path), cv2.IMREAD_COLOR)
    if img is None:
        logging.warning(f"{rel_path} | N/A | Could not read image")
        return 0
    saved = 0

    if mode in ("grid", "both"):
        root = output_root / "grid-occlusion"
        try:
            for label, occluded in generate_grid_occlusions(img, grid_rows, grid_cols):
                out_path = root / label / rel_path
                ensure_parent_dir(out_path)
                success = cv2.imwrite(str(out_path), occluded)
                if not success:
                    logging.warning(f"Failed to save {out_path}")
                saved += 1
        except Exception as e:
            logging.error(f"{rel_path} | grid | {str(e)}")

    if mode in ("landmark", "both"):
        root = output_root / "landmark-occlusion"
        if not shape_predictor:
            logging.warning(f"{rel_path} | landmark | shape_predictor not provided, skipping")
        else:
            try:
                occluded_list = generate_landmark_occlusions(
                    img=img,
                    landmark_precision=landmark_precision,
                    shape_predictor_path=shape_predictor,
                    landmark_scale_factor=lmk_scale
                )
                if len(occluded_list) == 0:
                    logging.warning(f"{rel_path} | landmark | No face/landmarks found")
                for label, occluded in occluded_list:
                    out_path = root / label / rel_path
                    ensure_parent_dir(out_path)
                    success = cv2.imwrite(str(out_path), occluded)
                    if not success:
                        logging.warning(f"Failed to save {out_path}")
                    saved += 1
            except Exception as e:
                logging.error(f"{rel_path} | landmark | {str(e)}")

    return saved

def _process_wrapper(process, args):
    return process(*args)

def run(
    input_path: Path,
    output_path: Path,
    mode: str,
    grid_rows_count: int,
    grid_columns_count: int,
    landmark_scale_factor: float,
    shape_predictor: Optional[str],
    multiprocessing: bool,
    workers: int,
    landmark_precision: str,
) -> None:
    images = list(iter_image_files(input_path))
    total_in = len(images)
    logging.info(f"Found {total_in} images under {input_path}")

    process = functools.partial(
        process_one_image,
        mode=mode,
        output_root=output_path,
        grid_rows=grid_rows_count,
        grid_cols=grid_columns_count,
        lmk_scale=landmark_scale_factor,
        shape_predictor=shape_predictor,
        landmark_precision=landmark_precision,
    )

    saved_total = 0
    if multiprocessing and total_in > 1:
        logging.info(f"Using multiprocessing with {workers} workers.")
        with futures.ProcessPoolExecutor(max_workers=workers) as ex:
            # map with progress-ish prints
            for i, saved in enumerate(ex.map(functools.partial(_process_wrapper, process), images), 1):
                saved_total += saved
                if i % 50 == 0:
                    logging.info(f"Processed {i}/{total_in} images...")
    else:
        for i, (abs_path, rel_path) in enumerate(images, 1):
            saved_total += process(abs_path, rel_path)
            if i % 200 == 0:
                logging.info(f"Processed {i}/{total_in} images...")

    logging.info(f"Done. Saved {saved_total} occluded images total.")
    return saved_total  # add this


# -----------------------------
# CLI
# -----------------------------

def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"Expected positive integer, got {value}")
    return ivalue

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Create occlusion datasets (grid and/or landmark) preserving folder hierarchy."
    )
    p.add_argument("--mode", choices=["grid", "landmark", "both"], required=True,
                   help="Which occlusion(s) to generate.")
    p.add_argument("--input_path", required=True, type=Path,
                   help="Root folder with images (nested subfolders supported).")
    p.add_argument("--output_path", required=True, type=Path,
                   help="Root folder to save outputs. Will create 'grid-occlusion' and/or 'landmark-occlusion'.")
    p.add_argument("--grid_rows_count", type=positive_int, default=4,
                   help="Grid rows for grid occlusion (default: 4).")
    p.add_argument("--grid_columns_count", type=positive_int, default=4,
                   help="Grid columns for grid occlusion (default: 4).")
    p.add_argument("--landmark_precision", choices=["pixel", "bbox"], default="pixel",
                   help="Landmark occlusion mode: pixel (polygon) or bbox (bounding box). (default: pixel).")
    p.add_argument("--landmark_scale_factor", type=float, default=1.5,
                   help="Scale factor to expand/shrink landmark occlusion polygons (default: 1.5).")
    p.add_argument("--multiprocessing", action="store_true",
                   help="Enable multiprocessing.")
    p.add_argument("--workers", type=positive_int, default=os.cpu_count() or 4,
                   help="Number of worker processes when --multiprocessing is set.")
    return p

def main():
    args = build_argparser().parse_args()
    input_path: Path = args.input_path
    output_path: Path = args.output_path
    mode: str = args.mode
    grid_rows: int = args.grid_rows_count
    grid_cols: int = args.grid_columns_count
    lmk_scale: float = args.landmark_scale_factor
    mp: bool = args.multiprocessing
    workers: int = args.workers
    landmark_precision: str = args.landmark_precision

    if not input_path.exists() or not input_path.is_dir():
        raise SystemExit(f"Input path does not exist or is not a directory: {input_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    # Always ensure predictor is available
    shape_predictor = ensure_shape_predictor()


    # Create log file in output path
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]  # up to milliseconds
    log_file = output_path / f"logs-{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # also print to console
        ]
    )

    start_time = time.time()
    logging.info("=== Processing started ===")

    saved_total = run(
        input_path=input_path,
        output_path=output_path,
        mode=mode,
        grid_rows_count=grid_rows,
        grid_columns_count=grid_cols,
        landmark_scale_factor=lmk_scale,
        shape_predictor=shape_predictor,
        multiprocessing=mp,
        workers=workers,
        landmark_precision=landmark_precision
    )

    end_time = time.time()
    elapsed = end_time - start_time
    elapsed_readable = timedelta(seconds=elapsed)
    avg_time_per_image = elapsed / max(saved_total, 1)  # avoid divide by zero

    logging.info("=== Processing finished ===")
    logging.info(f"Start time: {datetime.fromtimestamp(start_time)}")
    logging.info(f"End time:   {datetime.fromtimestamp(end_time)}")
    logging.info(f"Time elapsed: {elapsed_readable} ({elapsed:.6f} seconds)")
    logging.info(f"Average time per occluded image: {avg_time_per_image:.10f} seconds")

if __name__ == "__main__":
    main()
