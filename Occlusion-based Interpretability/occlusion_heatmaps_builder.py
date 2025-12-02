
"""
Occlusion Heatmaps Builder
--------------------------

This script builds occlusion heatmaps (grid- and landmark-based) and overlays them
on the original image. It consolidates logic previously prototyped in a Jupyter
notebook into a single Python module/CLI.

Key features:
- Parse "regular", "grid", and "landmark" result files (CSV-like: image_path,score,label?)
- Robust extraction of grid cell indices and landmark region names from file paths
- Build per-image heatmaps and save overlays with matplotlib
- Optional dlib-based landmark detection to map landmark regions to grid cells
- Multiprocessing support

Usage (examples):
    python occlusion_heatmaps_builder.py \
        --regular /path/to/regular_scores.txt \
        --grid /path/to/grid_scores.txt \
        --landmark /path/to/landmark_scores.txt \
        --output /path/to/output

The parser is flexible but you can customize regex patterns via CLI flags.
"""

import os
import argparse
import logging
import time
import re
from datetime import datetime, timedelta
from collections import defaultdict
from multiprocessing import Pool
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any
import json
import urllib.request
import bz2

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import csv
from tqdm import tqdm

try:
    import dlib  # optional
    DLIB_AVAILABLE = True
except Exception:
    DLIB_AVAILABLE = False


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

# --------------------------- Logging -----------------------------------------

def setup_logging(output_path: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3]
    log_file = output_path / f"heatmap_builder_log_{timestamp}.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

# Create a global buffer (or pass it around explicitly)
MATCH_LOGS: list[str] = []

def flush_match_logs():
    logging.warning("\n".join(MATCH_LOGS))
    MATCH_LOGS.clear()


# --------------------------- Parsing utilities -------------------------------

def _build_suffix_lookup(regular_map: Dict[str, float]) -> Dict[str, str]:
    """
    Build a reverse lookup {suffix -> canonical_path}.
    This avoids scanning all keys per row.
    """
    lookup = {}
    for k in regular_map.keys():
        parts = k.split("/")
        for i in range(len(parts)):
            suffix = "/".join(parts[i:])
            if suffix not in lookup:  # keep first occurrence
                lookup[suffix] = k
    return lookup


def match_to_regular(suffix_lookup: Dict[str, str], raw_path: str, mode: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Fast version: extract suffix directly and lookup.
    Returns a tuple (canonical_regular_path, relative_suffix) or (None, None).
    """
    parts = raw_path.replace("\\", "/").split("/")

    for i, p in enumerate(parts[:-1]):  # exclude filename
        valid = False
        if mode == "grid":
            valid = _extract_grid_rc(p) is not None
        elif mode == "landmark":
            valid = p.lower() in LANDMARK_REGION_NAMES
        if not valid:
            continue

        # suffix is the path *after* the variant folder that identifies the actual
        # image location relative to the canonical entry in the regular map.
        suffix = "/".join(parts[i + 1 :])
        candidate = suffix_lookup.get(suffix)
        if candidate:
            # Return both canonical regular path and the suffix we used to match it.
            return candidate, suffix

        MATCH_LOGS.append(f"[{mode}] No match found for '{raw_path}' (suffix='{suffix}')")
        return None, None

    MATCH_LOGS.append(f"[{mode}] No folder matching pattern found in {raw_path}")
    return None, None


def parse_regular_scores(path: Path) -> dict[str, float]:
    """
    Ultra-fast parsing of regular MAD scores CSV with a progress bar:
        image_path,score[,label]
    Returns: {absolute_image_path: score}
    """
    out: dict[str, float] = defaultdict(dict)

    # Count total lines for tqdm (optional, can be omitted for unknown length)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        total_lines = sum(1 for _ in f)

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        append = out.__setitem__  # local reference is faster
        for row in tqdm(reader, total=total_lines, desc="Parsing regular scores"):
            if len(row) < 2:
                continue
            img_path, score_str = row[0].strip(), row[1].strip()
            try:
                score = float(score_str)
            except ValueError:
                continue
            append(img_path, score)

    return out


# Common region names we look for in landmark occlusion file paths
LANDMARK_REGION_NAMES = [
    "left_eye", "right_eye", "eyes",
    "left_eyebrow", "right_eyebrow", "eyebrows",
    "nose", "nose_bridge", "nose_tip",
    "mouth", "inner_mouth", "outer_mouth", "lips",
    "jaw", "chin", "cheeks", "left_cheek", "right_cheek",
    "forehead"
]

# Combined regex patterns to match common formats
GRID_INDEX_PATTERNS = [
    # r0_c0, row0_col0, r-0_c-0, row-0-col-0
    re.compile(r"(?:r|row[_-]?)(?P<row>\d+)[_-]?(?:c|col[_-]?)(?P<col>\d+)", re.IGNORECASE),
    # cell_3_5, cell-3-5
    re.compile(r"cell[_-]?(?P<row>\d+)[_-](?P<col>\d+)", re.IGNORECASE),
    # just numbers separated by _ or -
    re.compile(r"(?P<row>\d+)[_-](?P<col>\d+)")
]

def _extract_grid_rc(path: str) -> Optional[Tuple[int, int]]:
    """
    Extract (row, col) from a string path or filename.
    Returns a tuple (row, col) if found, else None.
    """
    for rx in GRID_INDEX_PATTERNS:
        m = rx.search(path)
        if m:
            try:
                r = int(m.group("row"))
                c = int(m.group("col"))
                return r, c
            except Exception:
                continue
    return None


def parse_grid_scores(path: Path, regular_map: Dict[str, float]) -> Tuple[Dict[str, Dict[Tuple[int, int], float]], Dict[str, str]]:
    """
    Optimized parsing of grid occlusion scores CSV.
    Returns:
        out: {regular_image_path: {(row, col): score, ...}}
        rel_map: {regular_image_path: matched_relative_suffix}
    """
    out: Dict[str, Dict[Tuple[int, int], float]] = defaultdict(dict)
    rel_map: Dict[str, str] = {}

    # Precompute suffix lookup to avoid O(N*M) scans
    suffix_lookup = _build_suffix_lookup(regular_map)

    # Count total lines for tqdm 
    with path.open("r", encoding="utf-8", errors="ignore") as f: 
        total_lines = sum(1 for _ in f)

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        pbar = tqdm(reader, total=total_lines,desc="Parsing grid scores", unit="rows")

        for row in pbar:
            if len(row) < 2:
                continue
            variant_path_raw, score_str = row[0].strip(), row[1].strip()

            try:
                score_f = float(score_str)
            except ValueError:
                continue

            rc = _extract_grid_rc(variant_path_raw)
            if not rc:
                continue
            r, c = rc

            regular_path, suffix = match_to_regular(suffix_lookup, variant_path_raw, mode="grid")
            if not regular_path:
                continue

            # store mapping and score
            out[regular_path][(r, c)] = score_f
            # keep first matched relative suffix for this regular path
            if regular_path not in rel_map:
                rel_map[regular_path] = suffix

    flush_match_logs()
    return out, rel_map


def parse_landmark_scores(path: Path, regular_map: Dict[str, float]) -> Tuple[Dict[str, Dict[str, float]], Dict[str, str]]:
    """
    Optimized parsing of landmark occlusion scores CSV.
    Returns:
        out: {regular_image_path: {region_name: score, ...}}
        rel_map: {regular_image_path: matched_relative_suffix}
    """
    out: Dict[str, Dict[str, float]] = defaultdict(dict)
    rel_map: Dict[str, str] = {}

    # Pre-compile landmark regex (once per call, not per row)
    name_union = "|".join(map(re.escape, sorted(LANDMARK_REGION_NAMES, key=len, reverse=True)))
    name_rx = re.compile(rf"[\\/](?P<name>{name_union})(?:[\\/]|$)", re.IGNORECASE)

    # Precompute suffix lookup for fast regular_map mapping
    suffix_lookup = _build_suffix_lookup(regular_map)

    # Count total lines for tqdm 
    with path.open("r", encoding="utf-8", errors="ignore") as f: 
        total_lines = sum(1 for _ in f)

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.reader(f)
        pbar = tqdm(reader, total=total_lines, desc="Parsing landmark scores", unit="rows")

        for row in pbar:
            if len(row) < 2:
                continue
            variant_path_raw, score_str = row[0].strip(), row[1].strip()

            try:
                score_f = float(score_str)
            except ValueError:
                continue

            # Match landmark region (fast regex)
            m = name_rx.search(variant_path_raw)
            if not m:
                continue
            region = m.group("name").lower()

            # Fast lookup instead of scanning all regular_map
            regular_path, suffix = match_to_regular(suffix_lookup, variant_path_raw, mode="landmark")
            if not regular_path:
                continue

            out[regular_path][region] = score_f
            if regular_path not in rel_map:
                rel_map[regular_path] = suffix

    flush_match_logs()
    return out, rel_map


# --------------------------- Landmark geometry helpers -----------------------

# 68-point landmark indices (iBUG) region groupings
IBUG_GROUPS = {
    "jaw": list(range(0, 17)),
    "right_eyebrow": list(range(17, 22)),
    "left_eyebrow": list(range(22, 27)),
    "nose_bridge": list(range(27, 31)),
    "nose_tip": list(range(31, 36)),
    "right_eye": list(range(36, 42)),
    "left_eye": list(range(42, 48)),
    "outer_mouth": list(range(48, 60)),
    "inner_mouth": list(range(60, 68)),
}

# Map our region names to iBUG groups
REGION_TO_GROUPS = {
    "left_eye": ["left_eye"],
    "right_eye": ["right_eye"],
    "eyes": ["left_eye", "right_eye"],
    "left_eyebrow": ["left_eyebrow"],
    "right_eyebrow": ["right_eyebrow"],
    "eyebrows": ["left_eyebrow", "right_eyebrow"],
    "nose": ["nose_bridge", "nose_tip"],
    "nose_bridge": ["nose_bridge"],
    "nose_tip": ["nose_tip"],
    "mouth": ["outer_mouth", "inner_mouth"],
    "outer_mouth": ["outer_mouth"],
    "inner_mouth": ["inner_mouth"],
    "lips": ["outer_mouth", "inner_mouth"],
    "jaw": ["jaw"],
    "chin": ["jaw"],
    "left_cheek": ["left_eyebrow", "left_eye"],  # approximations
    "right_cheek": ["right_eyebrow", "right_eye"],
    "cheeks": ["left_eyebrow", "left_eye", "right_eyebrow", "right_eye"],
    "forehead": ["left_eyebrow", "right_eyebrow"],
}


def _ensure_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_landmarks_68(img_bgr: np.ndarray, sp_path: Optional[Path]) -> Optional[np.ndarray]:
    """Return array of shape (68,2) with (x,y) or None if not found or dlib unavailable."""
    if not DLIB_AVAILABLE:
        return None
    if sp_path is None or not Path(sp_path).exists():
        return None

    gray = _ensure_grayscale(img_bgr)
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(str(sp_path))
    rects = detector(gray, 1)
    if not rects:
        return None
    # Take the most confident
    rect = rects[0]
    shape = sp(gray, rect)
    pts = np.array([(shape.part(i).x, shape.part(i).y) for i in range(68)], dtype=np.int32)
    return pts


def region_bbox_from_landmarks(landmarks68: np.ndarray, region: str) -> Optional[Tuple[int, int, int, int]]:
    """Return (x_min, y_min, x_max, y_max) bbox for a region name using groups."""
    region = region.lower()
    groups = REGION_TO_GROUPS.get(region)
    if not groups:
        return None
    idxs = []
    for g in groups:
        idxs += IBUG_GROUPS.get(g, [])
    if not idxs:
        return None
    pts = landmarks68[idxs, :]
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    return int(x0), int(y0), int(x1), int(y1)


def bbox_to_grid_cells(bbox: Tuple[int, int, int, int], img_shape: Tuple[int, int], grid_shape: Tuple[int, int]) -> Tuple[slice, slice]:
    """Convert pixel bbox to grid (row slice, col slice)."""
    h, w = img_shape[:2]
    rows, cols = grid_shape
    y0, x0, y1, x1 = bbox[1], bbox[0], bbox[3], bbox[2]

    cell_h = h / rows
    cell_w = w / cols
    r0 = max(0, int(y0 // cell_h))
    r1 = min(rows, int(np.ceil(y1 / cell_h)))
    c0 = max(0, int(x0 // cell_w))
    c1 = min(cols, int(np.ceil(x1 / cell_w)))
    return slice(r0, r1), slice(c0, c1)


# --------------------------- Heatmap building --------------------------------

def refine_grid_matrix(grid_M: np.ndarray, refinement: int = 2) -> np.ndarray:
    """
    Refine a grid matrix by subdividing each cell into `refinement x refinement` subcells,
    duplicating the original cell's value in all subcells.
    
    Args:
        grid_M: (rows, cols) original grid matrix
        refinement: number of subcells per axis (2 -> 2x2, 3 -> 3x3)
    
    Returns:
        refined_M: (rows*refinement, cols*refinement) refined matrix
    """
    if refinement <= 1:
        return grid_M
    return np.kron(grid_M, np.ones((refinement, refinement), dtype=grid_M.dtype))


def build_grid_matrix(grid_scores: Dict[Tuple[int, int], float], grid_shape: Tuple[int, int]) -> np.ndarray:
    """Build a (rows, cols) matrix from {(r,c): score} mapping. Missing cells -> 0."""
    rows, cols = grid_shape
    M = np.zeros((rows, cols), dtype=np.float32)
    for (r, c), s in grid_scores.items():
        if 0 <= r < rows and 0 <= c < cols:
            M[r, c] = float(s)
    return M


def adjust_matrix_to_heatmap_mode(
        M: np.ndarray,
    regular_score: float,
    heatmap_mode: str,
    mask: np.ndarray | None = None
) -> np.ndarray:
    """
    Adjusts the heatmap matrix based on the selected mode and optionally
    applies a binary mask (e.g., to restrict output to landmark regions).

    Args:
        M: Input matrix of scores.
        regular_score: The regular (baseline) score of the image.
        heatmap_mode: 'occluded' to use raw scores,
                      'diff' to show absolute difference from regular_score.
        mask: Optional binary mask (same shape as M). Areas with 0 will be hidden.

    Returns:
        np.ndarray with the same shape as M.
    """
    if heatmap_mode == "occluded":
        adjusted = M
    elif heatmap_mode == "diff":
        if regular_score is None:
            logging.warning("regular_score is None")
            adjusted = M
        else:
            adjusted = np.abs(M - regular_score)
    else:
        raise ValueError(f"Invalid heatmap_mode: {heatmap_mode}")

    # Apply mask if provided
    if mask is not None:
        if mask.shape != adjusted.shape:
            raise ValueError("Mask shape must match the input matrix shape.")
        adjusted = adjusted * mask

    return adjusted


def combine_grid_and_landmarks(
    grid_M: np.ndarray,
    landmark_M: np.ndarray,
    grid_weight: float = 0.5,
    landmark_weight: float = 0.5
) -> np.ndarray:
    """
    Combine a regular grid heatmap and a landmark heatmap by weighted averaging,
    but only where landmark_M has non-zero values.
    """
    if grid_M.shape != landmark_M.shape:
        raise ValueError(
            f"grid_M shape {grid_M.shape} does not match landmark_M shape {landmark_M.shape}"
        )

    grid_f = grid_M.astype(np.float32)
    land_f = landmark_M.astype(np.float32)

    # mask of cells that actually contain landmark scores
    mask = land_f > 0

    # start with the original grid
    combined = grid_f.copy()

    # update only the masked cells
    combined[mask] = (
        grid_weight * grid_f[mask] +
        landmark_weight * land_f[mask]
    )

    return combined


# --- helper: ensure rsl/csl are slices and clamped to grid shape ---
def _to_slice(obj, max_len: int) -> slice:
    """
    Convert obj to a Python slice suitable for indexing an axis with length max_len.
    Accepts slice, int, (start, stop), or (start, stop, step).
    Clamps to [0, max_len].
    """
    if isinstance(obj, slice):
        start = 0 if obj.start is None else int(obj.start)
        stop = max_len if obj.stop is None else int(obj.stop)
    elif isinstance(obj, (list, tuple, np.ndarray)):
        if len(obj) >= 2:
            start, stop = int(obj[0]), int(obj[1])
        else:
            start, stop = int(obj[0]), int(obj[0]) + 1
    else:  # int-like
        start, stop = int(obj), int(obj) + 1

    # clamp
    start = max(0, min(max_len, start))
    stop = max(0, min(max_len, stop))
    if stop <= start:
        stop = min(max_len, start + 1)
    return slice(start, stop)


# Helper to build output path while preserving relative path structure
def _create_output_path(
    output_dir: Path,
    subfolder: str,
    rel_path_str: str
) -> Path:
    """
    Build the output file path while preserving the relative folder structure.

    Args:
        output_dir: Root output directory as a Path object.
        subfolder: Name of the subfolder ("grid", "landmarks", "combined").
        rel_path_str: Relative suffix that was matched (e.g. 'subdir/img.png')

    Returns:
        A Path object pointing to the final output file.
    """
    out_p = output_dir / subfolder / rel_path_str

    out_p.parent.mkdir(parents=True, exist_ok=True)
    return out_p


def calculate_grid_font_size(rows: int, cols: int, scale: float = 1.0) -> float:
    """
    Calculate a suitable font size for text overlay in a grid heatmap.

    Args:
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        scale (float): Optional scaling factor to adjust the font size.

    Returns:
        float: Calculated font size.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive integers")
    
    font_size = (75.0 / max(rows, cols)) * scale
    return font_size


def compute_landmark_grid(
    img_bgr: np.ndarray,
    landmark_scores: Dict[str, float],
    shape_predictor: Path,
    grid_size: tuple = (8, 8),
) -> tuple[np.ndarray, Dict[str, tuple]]:
    """
    Compute a grid of averaged landmark scores and return landmark bounding boxes.

    Returns:
        grid_M: np.ndarray of shape `grid_size` with averaged scores
        region_bboxes: dict mapping region name -> (x0, y0, x1, y1)
    """
    H, W = img_bgr.shape[:2]
    rows, cols = grid_size

    # accumulators
    score_sum = np.zeros((rows, cols), dtype=np.float32)
    score_count = np.zeros((rows, cols), dtype=np.int32)

    # detect landmarks & region bboxes
    landmarks68 = detect_landmarks_68(img_bgr, shape_predictor)
    region_bboxes = {}

    # helper approximate fallback regions if landmarks not detected
    approx = {
        "left_eye": (slice(rows // 4, rows // 2), slice(cols // 6, cols // 3)),
        "right_eye": (slice(rows // 4, rows // 2), slice(cols - cols // 3, cols - cols // 6)),
        "nose": (slice(rows // 3, rows - rows // 3), slice(cols // 3, cols - cols // 3)),
        "mouth": (slice(rows - rows // 3, rows - rows // 6), slice(cols // 4, cols - cols // 4)),
    }

    for region, score in landmark_scores.items():
        if landmarks68 is not None:
            bbox = region_bbox_from_landmarks(landmarks68, region)  # expects x0,y0,x1,y1
            if bbox:
                region_bboxes[region] = bbox
                rsl_csl = bbox_to_grid_cells(bbox, (H, W), (rows, cols))
                # ensure we support different return shapes from bbox_to_grid_cells
                if isinstance(rsl_csl, (list, tuple, np.ndarray)) and len(rsl_csl) == 2:
                    rsl, csl = rsl_csl
                else:
                    # fallback to whole grid if unexpected
                    rsl, csl = slice(0, rows), slice(0, cols)
            else:
                rsl, csl = slice(0, rows), slice(0, cols)
        else:
            rsl, csl = approx.get(region, (slice(0, rows), slice(0, cols)))

        # convert to valid slices (handle ints/tuples)
        rsl = _to_slice(rsl, rows)
        csl = _to_slice(csl, cols)

        # add to accumulators
        score_sum[rsl, csl] += float(score)
        score_count[rsl, csl] += 1

    # compute averages (only where count>0)
    grid_M = np.zeros_like(score_sum, dtype=np.float32)
    mask = score_count > 0
    grid_M[mask] = score_sum[mask] / score_count[mask]

    return grid_M, region_bboxes


# --- renderer returning fig, ax (draws ONLY non-empty cells if mask provided) ---
def render_heatmap_on_image(
    img_bgr: np.ndarray,
    score_M: np.ndarray,
    heatmap_mode: str,
    alpha: float = 0.6,
    colormap: str = "hot",
    draw_grid: bool = True,
    title: Optional[str] = None,
    show_values: bool = True,
    draw_mask: bool = True,
    mask: Optional[np.ndarray] = None, 
) -> Tuple[Any, Any]:
    """
    Render a heatmap on top of image and return (fig, ax). Does NOT save/show.
    Only draws cells where draw_mask is True (or where score_M > 0 if draw_mask is None).
    """
    rows, cols = score_M.shape
    fontsize = calculate_grid_font_size(rows, cols)
    H, W = img_bgr.shape[:2]

    # determine mask: only draw cells with values (or explicit mask)
    if draw_mask:
        if mask is None:
            mask = (score_M != 0)
    else:
        mask = np.ones_like(score_M, dtype=bool)

    # if no cell to draw, still create figure (but colorbar will be empty)
    any_cells = mask.any()
    # compute max over drawn cells
    if any_cells:
        max_val = float(score_M[mask].max())
    else:
        max_val = 0.0

    # prepare normalized map for coloring (avoid divide-by-zero)
    norm_scores = np.zeros_like(score_M, dtype=float)
    if max_val > 0:
        norm_scores[mask] = score_M[mask] / max_val

    cmap = plt.colormaps.get_cmap(colormap)
    cell_h, cell_w = H / rows, W / cols

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

    # draw only masked cells
    indices = np.argwhere(mask)
    for (i, j) in indices:
        val = float(score_M[i, j])
        norm_val = float(norm_scores[i, j])
        x, y = j * cell_w, i * cell_h
        color = cmap(norm_val) if max_val > 0 else cmap(0.0)

        rect = patches.Rectangle(
            (x, y), cell_w, cell_h,
            linewidth=0.5,
            edgecolor='white' if draw_grid else None,
            facecolor=color,
            alpha=alpha
        )
        ax.add_patch(rect)

        if show_values:
            score_text = f"{val:.2e}" if val != 0 else "0"
            ax.text(
                x + cell_w / 2, y + cell_h / 2, score_text,
                ha="center", va="center", fontsize=fontsize,
                color="white", weight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6)
            )

    # colorbar (use vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)

    if heatmap_mode == "occluded":
        cbar.set_label("Prediction score when occluded (0 -> morph, 1 -> bona fide)")
    elif heatmap_mode == "diff":
        cbar.set_label("Difference between regular and occluded prediction scores (0 -> less relevant, 1 -> more relevant)")
    else:
        raise ValueError(f"Invalid heatmap_mode: {heatmap_mode}")

    if title:
        ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    return fig, ax


def finalize_figure(fig, out_path: Optional[Path] = None) -> Optional[Path]:
    """
    Save or show a fig returned by render_heatmap_on_image (caller must close fig via this).
    """
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return out_path
    else:
        plt.show()
        plt.close(fig)
        return None
    

# --- wrapper kept for backward compatibility that uses the renderer ---
def overlay_heatmap_on_image(
    img_bgr: np.ndarray,
    score_M: np.ndarray,
    heatmap_mode: str,
    alpha: float = 0.6,
    colormap: str = "hot",
    draw_grid: bool = True,
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_values: bool = True,
    draw_mask: bool = True,
) -> Optional[Path]:
    fig, ax = render_heatmap_on_image(
        img_bgr=img_bgr,
        score_M=score_M,
        heatmap_mode=heatmap_mode,
        alpha=alpha,
        colormap=colormap,
        draw_grid=draw_grid,
        title=title,
        show_values=show_values,
        draw_mask=draw_mask
    )
    return finalize_figure(fig, out_path)


def overlay_landmark_heatmap_on_image(
    img_bgr: np.ndarray,
    score_M: np.ndarray,
    region_bboxes: Dict[str, Tuple[int, int, int, int]],
    heatmap_mode: str,
    alpha: float = 0.6,
    colormap: str = "hot",
    out_path: Optional[Path] = None,
    title: Optional[str] = None,
    show_values: bool = True,
    draw_grid: bool = True,
    draw_mask: bool = True,
    show_landmark_boxes: bool = True,
) -> Optional[Path]:
    """
    Render a heatmap from a precomputed landmark grid (score_M) and overlay landmark bounding boxes.

    Args:
        img_bgr: Input image (BGR).
        score_M: Grid of scores (landmark heatmap).
        region_bboxes: Dict mapping landmark names -> (x0, y0, x1, y1).
        alpha: Heatmap transparency.
        colormap: Colormap name.
        out_path: Optional path to save output.
        title: Optional figure title.
        show_values: Whether to show cell values.
        draw_grid: Whether to draw grid lines.
        show_landmark_boxes: Whether to overlay landmark bounding boxes.
    """

    # Render heatmap
    fig, ax = render_heatmap_on_image(
        img_bgr=img_bgr,
        score_M=score_M,
        heatmap_mode=heatmap_mode,
        alpha=alpha,
        colormap=colormap,
        draw_grid=draw_grid,
        title=title,
        show_values=show_values,
        draw_mask=draw_mask,
    )

    # Draw landmark boxes and labels
    if show_landmark_boxes and region_bboxes:
        for region, bbox in region_bboxes.items():
            x0, y0, x1, y1 = [int(v) for v in bbox]
            ax.add_patch(patches.Rectangle(
                (x0, y0), x1 - x0, y1 - y0,
                linewidth=2, edgecolor="cyan",
                facecolor="none", linestyle="--", alpha=0.9
            ))
            ax.text(
                x0 + (x1 - x0) / 2, y0 - 10,
                region.replace("_", " ").title(),
                ha="center", va="center", fontsize=12,
                color="cyan", weight="bold",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7)
            )

    return finalize_figure(fig, out_path)


# --------------------------- Processing pipeline -----------------------------

def _process_image_wrapper(args_tuple):
    """Top-level wrapper for multiprocessing."""
    return process_one_image(*args_tuple)


def _determine_grid_shape(grid_map: Dict[Tuple[int, int], float]) -> Tuple[int, int]:
    max_r = max((rc[0] for rc in grid_map.keys()), default=-1)
    max_c = max((rc[1] for rc in grid_map.keys()), default=-1)
    # If indices are 0-based in filenames, this is fine; if 1-based, we still get >= proper size
    return max_r + 1, max_c + 1


def process_one_image(
    base_image_path: Path,
    regular_score: Optional[float],
    grid_scores_map: Optional[Dict[Tuple[int, int], float]],
    landmark_scores_map: Optional[Dict[str, float]],
    relative_path: str,
    heatmap_mode: str,
    output_dir: Path,
    shape_predictor: Optional[str],
    alpha: float,
    colormap: str,
) -> Dict[str, Any]:
    img_bgr = cv2.imread(str(base_image_path))
    if img_bgr is None:
        return {"image": str(base_image_path), "ok": False, "error": "failed to read image"}

    grid_M = None
    combined_M = None
    grid_out = None
    combined_out = None

    refinement = 2

    if grid_scores_map:
        rows, cols = _determine_grid_shape(grid_scores_map)
        grid_M_occluded = build_grid_matrix(grid_scores_map, (rows, cols))

        # 🔹 Refine the grid (e.g., 2x2 subcells per cell)
        grid_M_occluded = refine_grid_matrix(grid_M_occluded, refinement=refinement)
        
        grid_M = adjust_matrix_to_heatmap_mode(grid_M_occluded, regular_score, heatmap_mode)
        
        grid_out = overlay_heatmap_on_image(
            img_bgr, grid_M, heatmap_mode,
            alpha=alpha, colormap=colormap, draw_grid=True, draw_mask=False,
            out_path= _create_output_path(output_dir, "grid", relative_path),
            title="Grid Occlusion Heatmap",
        )


    if landmark_scores_map:
        # If we don't have a grid, assume 8x8 for baseline overlay of landmarks
        if grid_M is None:
            rows, cols = 4, 4
            grid_M = np.zeros((rows, cols), dtype=np.float32)

        landmark_M_occluded, region_bboxes = compute_landmark_grid(
            img_bgr, landmark_scores_map, shape_predictor, grid_size=(rows*refinement, cols*refinement)
        )
        landmark_M = adjust_matrix_to_heatmap_mode(
            landmark_M_occluded, 
            regular_score, 
            heatmap_mode, 
            mask=(landmark_M_occluded != 0)
        )

        landmark_out = overlay_landmark_heatmap_on_image(
            img_bgr, landmark_M, region_bboxes, heatmap_mode,
            alpha=alpha, colormap=colormap, draw_grid=False, draw_mask=True,
            out_path= _create_output_path(output_dir, "landmarks", relative_path),
            title="Landmark Occlusion Heatmap",
        )

        combined_M = combine_grid_and_landmarks(grid_M_occluded, landmark_M_occluded)
        combined_M = adjust_matrix_to_heatmap_mode(combined_M, regular_score, heatmap_mode)

        combined_out = overlay_landmark_heatmap_on_image(
            img_bgr, combined_M, region_bboxes, heatmap_mode,
            alpha=alpha, colormap=colormap, draw_grid=True, draw_mask=False,
            out_path= _create_output_path(output_dir, "combined", relative_path),
            title="Composite Heatmap (Grid + Landmark)",
        )

    return {
        "image": str(base_image_path),
        "ok": True,
        "grid_out": str(grid_out) if grid_out else None,
        "landmark_out": str(landmark_out) if landmark_out else None,
        "combined_out": str(combined_out) if combined_out else None,
    }


def build_mappings(
    regular_path: Path,
    grid_path: Optional[Path],
    landmark_path: Optional[Path],
) -> Tuple[Dict[str, float], Dict[str, Dict[Tuple[int, int], float]], Dict[str, Dict[str, float]], Dict[str, str]]:
    regular_map = parse_regular_scores(regular_path)
    grid_map, grid_rel = parse_grid_scores(grid_path, regular_map) if grid_path else ({}, {})
    landmark_map, landmark_rel = parse_landmark_scores(landmark_path, regular_map) if landmark_path else ({}, {})

    # merge rel maps
    rel_map = {}
    rel_map.update(landmark_rel)
    rel_map.update(grid_rel)
    return regular_map, grid_map, landmark_map, rel_map



# --------------------------- CLI --------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build occlusion heatmap overlays from score files.")
    parser.add_argument("--regular", required=True, help="Path to regular scores file (CSV-like: img,score[,label])")
    parser.add_argument("--grid", help="Path to grid occlusion scores file")
    parser.add_argument("--landmark", help="Path to landmark occlusion scores file")
    parser.add_argument("--heatmap-mode", choices=["occluded", "diff"], default="occluded", help=(
            "How to fill each cell:\n"
            "  occluded → show the score when occluded\n"
            "  diff     → show the difference abs(regular - occluded)"
        ),
    )
    parser.add_argument("--output", required=True, help="Output folder for overlays")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha")
    parser.add_argument("--colormap", default="hot", help="Matplotlib colormap name")
    parser.add_argument("--multiprocessing", action="store_true", help="Enable multiprocessing")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes if multiprocessing is enabled")
    parser.add_argument("--log-level", default="INFO", help="Logging level")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Always ensure predictor is available
    shape_predictor = ensure_shape_predictor()

    setup_logging(output_dir, args.log_level)
    start_time = time.time()

    # Build mappings
    logging.info("Parsing score files...")
    regular_map, grid_map, landmark_map, rel_map = build_mappings(
        Path(args.regular),
        Path(args.grid) if args.grid else None,
        Path(args.landmark) if args.landmark else None
    )

    # Only keep images that are present in all three maps
    logging.info("Filtering images present in all maps...")
    keys = set(regular_map.keys()) & set(grid_map.keys()) & set(landmark_map.keys())
    base_images = [Path(k) for k in keys]
    logging.info(f"Found {len(base_images)} images present in all maps.")

    output_dir.mkdir(parents=True, exist_ok=True)

    def _args_for_image(p: Path):
        # Use the absolute path string as key directly
        key = str(p.resolve())
        return (
            p,
            regular_map.get(key, None),
            grid_map.get(key, {}),
            landmark_map.get(key, {}),
            rel_map.get(key),
            args.heatmap_mode,
            output_dir,
            Path(shape_predictor),
            float(args.alpha),
            str(args.colormap)
        )

    results: List[Dict[str, Any]] = []
    if args.multiprocessing and len(base_images) > 1:
        logging.info(f"Multiprocessing with {args.workers} workers...")
        args_list = list(map(_args_for_image, base_images))

        with Pool(processes=args.workers) as pool:
            for res in tqdm(
                pool.imap_unordered(_process_image_wrapper, args_list),
                total=len(args_list),
                desc="Processing images",
            ):
                results.append(res)
    else:
        for p in tqdm(base_images, desc="Processing images"):
            results.append(process_one_image(*_args_for_image(p)))

    # Save an index JSON
    index_path = output_dir / "overlays_index.json"
    index_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    end_time = time.time()
    elapsed = timedelta(seconds=end_time - start_time)
    logging.info("=== Done ===")
    logging.info(f"Created overlays for {sum(1 for r in results if r.get('ok'))}/{len(results)} images")
    logging.info(f"Index JSON: {index_path}")
    logging.info(f"Elapsed: {elapsed}")

if __name__ == "__main__":
    main()
