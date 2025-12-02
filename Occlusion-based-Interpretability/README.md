# 🧠 Occlusion-Based Interpretability Method – Project Overview

This folder contains the **original contribution** of this thesis: a complete pipeline for occlusion-based interpretability analysis of Morphing Attack Detection (MAD) models. The methodology provides tools to understand which facial regions deep learning models rely on when making detection decisions, enhancing model transparency and trustworthiness in security-critical applications.

The approach systematically occludes different facial regions (both grid-based and landmark-based) and measures the impact on model predictions, generating visual heatmaps that highlight the most influential areas for morphing detection.

---

## 📁 Script Summaries

### 1. **`crop_dataset_builder.py`**
#### 🔍 Objective:
Prepares a clean, standardized dataset by cropping and resizing facial images before any occlusion experiments, ensuring that every image has a consistent face region and size.

#### 🧬 Code Description:
- **Face Detection & Cropping**:
  - Uses MTCNN to detect the face and crop the image to the bounding box.
  - Ensures consistent facial framing across all datasets.
- **Folder Structure Preservation**:
  - Cropped faces are saved in the same relative directory tree under the output folder.
  - Maintains dataset organization for easy tracking.
- **Multiprocessing Support**:
  - Optional parallel processing for large datasets.
  - Configurable number of worker processes.
- **Detailed Logging**:
  - Records progress, warnings, and processing times to both console and log file.
  - Tracks successful and failed processing attempts.

#### 📋 Arguments:
| Argument            | Type    | Required | Default             | Description                                                    |
|---------------------|---------|----------|---------------------|----------------------------------------------------------------|
| `--input_path`      | Path    | Yes      | —                   | Root folder containing images to process.                      |
| `--output_path`     | Path    | Yes      | —                   | Root folder where cropped faces will be saved.                 |
| `--multiprocessing` | Flag    | No       | Disabled            | Enable multiprocessing for faster processing.                  |
| `--workers`         | Integer | No       | Number of CPU cores | Number of worker processes when multiprocessing is enabled.    |

#### 💻 Example Command:
```bash
python crop_dataset_builder.py \
    --input /path/to/raw_dataset \
    --output /path/to/cropped_dataset \
    --multiprocessing --workers 12
```

---

### 2. **`occlusion_dataset_builder.py`**
#### 🔍 Objective:
Generates occluded variants of each cropped image to simulate different types of facial occlusion, used to test the robustness of MAD models and understand which facial regions are most critical for detection.

#### 🧬 Code Description:
- **Grid-Based Occlusion**:
  - Divides the face into an R×C grid (default 4×4).
  - Covers one cell at a time with black rectangles to hide parts of the face.
  - Generates R×C occluded variants per image.
- **Landmark-Based Occlusion**:
  - Uses Dlib 68-point facial landmarks to identify semantic facial regions.
  - Occludes specific regions: left eye, right eye, nose, mouth.
  - Supports pixel-precise polygon masking or bounding box masking.
- **Flexible Modes**:
  - `grid` – generates only grid-based occlusions.
  - `landmark` – generates only landmark-based occlusions.
  - `both` – generates both types in a single run.
- **Folder Structure Preservation**:
  - Output images maintain the relative folder paths of the original dataset.
  - Organized into `grid-occlusion/` and `landmark-occlusion/` subfolders.
- **Multiprocessing Support**:
  - Speeds up dataset generation on large datasets.
  - Configurable number of worker processes.
- **Automatic Dlib Model Download**:
  - Downloads and extracts the 68-point shape predictor if missing.
- **Detailed Logging**:
  - Tracks processing progress, warnings, and errors in timestamped log files.

#### 📋 Arguments:
| Argument                  | Type / Choices                     | Required | Default             | Description                                                                                           |
|---------------------------|------------------------------------|----------|---------------------|-------------------------------------------------------------------------------------------------------|
| `--mode`                  | `grid` \| `landmark` \| `both`     | Yes      | —                   | Which occlusion(s) to generate.                                                                       |
| `--input_path`            | Path                               | Yes      | —                   | Root folder containing input images (nested subfolders supported).                                    |
| `--output_path`           | Path                               | Yes      | —                   | Root folder where occluded images will be saved.                                                      |
| `--grid_rows_count`       | Integer                            | No       | 4                   | Number of rows to divide the image grid for grid occlusion.                                           |
| `--grid_columns_count`    | Integer                            | No       | 4                   | Number of columns to divide the image grid for grid occlusion.                                        |
| `--landmark_precision`    | `pixel` \| `bbox`                  | No       | `pixel`             | Landmark occlusion method: pixel-precise polygon or bounding box.                                     |
| `--landmark_scale_factor` | Float                              | No       | 1.5                 | Scale factor to expand or shrink the landmark polygons or bounding boxes.                             |
| `--multiprocessing`       | Flag                               | No       | Disabled            | Enable multiprocessing for faster dataset creation.                                                   |
| `--workers`               | Integer                            | No       | Number of CPU cores | Number of worker processes when multiprocessing is enabled.                                           |

#### 💻 Example Command:
```bash
python occlusion_dataset_builder.py \
    --mode both \
    --input /path/to/cropped_dataset \
    --output /path/to/occluded_dataset \
    --grid_rows_count 4 \
    --grid_columns_count 4 \
    --landmark_precision bbox \
    --landmark_scale_factor 1.5 \
    --multiprocessing --workers 12
```

---

### 3. **`occlusion_heatmaps_builder.py`**
#### 🔍 Objective:
Creates visual heatmaps that highlight which facial regions influence the model's predictions the most, combining grid and landmark occlusion scores into composite visualizations overlaid on original images.

#### 📄 CSV File Structure:
The script expects three types of CSV files, each with a different structure:

1. **Regular Scores CSV** (`--regular`):
   - Format: `image_path,score[,label]`
   - Example:
     ```
     /path/to/dataset/subfolder/image1.png,0.85,1
     /path/to/dataset/another/path/image2.png,0.23,0
     ```
   - Contains baseline predictions on original (non-occluded) images
   - Label: `0` = morph, `1` = bona fide

2. **Grid Scores CSV** (`--grid`):
   - Format: `occluded_image_path,score[,label]`
   - Example:
     ```
     /path/to/dataset/r0_c0/subfolder/image1.png,0.72,1
     /path/to/dataset/row1_col2/subfolder/image1.png,0.81,1
     /path/to/dataset/cell_3_5/another/path/image2.png,0.15,0
     ```
   - Path structure: `/base/path/{grid_cell}/common_path/image.png`
   - The `common_path/image.png` after the grid cell folder must match the corresponding path in the regular scores CSV
   - Grid cell indices can be in formats like: `r0_c0`, `row1_col2`, `cell_3_5`, `0_1`
   - Each row represents a prediction when a specific grid cell is occluded
   - Label: `0` = morph, `1` = bona fide

3. **Landmark Scores CSV** (`--landmark`):
   - Format: `occluded_image_path,score[,label]`
   - Example:
     ```
     /path/to/dataset/left_eye/subfolder/image1.png,0.65,1
     /path/to/dataset/nose/subfolder/image1.png,0.78,1
     /path/to/dataset/mouth/another/path/image2.png,0.12,0
     ```
   - Path structure: `/base/path/{landmark_region}/common_path/image.png`
   - The `common_path/image.png` after the landmark region folder must match the corresponding path in the regular scores CSV
   - Landmark region names: `left_eye`, `right_eye`, `nose`, `mouth`, `jaw`, `eyebrows`, `chin`, `forehead`, etc.
   - Each row represents a prediction when a specific facial landmark region is occluded
   - Label: `0` = morph, `1` = bona fide

**Path Matching Example:**
```
Regular:  /dataset/bonafide/subject001/image.png,0.85,1
Grid:     /dataset/r2_c3/bonafide/subject001/image.png,0.72,1
Landmark: /dataset/left_eye/bonafide/subject001/image.png,0.65,1
```
The script matches `bonafide/subject001/image.png` across all three files to associate the occluded predictions with the baseline.

#### 🧬 Code Description:
- **Flexible Score Parsing**:
  - Supports "regular", "grid", and "landmark" score files in CSV-like format.
  - Matches each occluded image back to its corresponding original image using flexible path matching and regex patterns.
- **Grid-Based Heatmaps**:
  - Builds per-image heatmaps from grid occlusion scores.
  - Divides face into grid cells and visualizes prediction sensitivity per cell.
- **Landmark-Based Heatmaps**:
  - Builds per-image heatmaps using facial landmark occlusion scores.
  - Uses Dlib 68-point shape predictor to map landmark regions to image coordinates.
- **Composite Heatmaps**:
  - Optionally combines grid and landmark heatmaps into unified visualization.
  - Provides comprehensive view of both spatial and semantic importance.
- **Heatmap Visualization Modes**:
  - `occluded` – Shows raw model prediction when a region is occluded.
  - `diff` – Shows absolute difference from the regular (baseline) score, highlighting impact of occlusion.
- **Visual Overlay**:
  - Overlays heatmaps on original images with grid lines, bounding boxes, and cell values.
  - Uses color gradients to indicate prediction sensitivity (lighter colors = high impact, darker colors = low impact).
  - Generates PNG heatmaps for grid occlusion, landmark occlusion, and composite visualizations.
- **Multiprocessing Support**:
  - Efficient processing of large datasets.
  - Configurable number of worker processes.
- **Automatic Dlib Model Download**:
  - Downloads the 68-point shape predictor if missing.
- **Detailed Logging**:
  - Tracks progress, warnings, and errors in timestamped log files.

#### 📋 Arguments:
| Argument                  | Type / Choices                                | Required | Default         | Description                                                                                                         |
|---------------------------|-----------------------------------------------|---------|----------------|---------------------------------------------------------------------------------------------------------------------|
| `--regular`               | Path                                          | Yes     | —              | Path to regular (baseline) score file.                                                                              |
| `--grid`                  | Path                                          | No      | —              | Path to grid occlusion score file.                                                                                  |
| `--landmark`              | Path                                          | No      | —              | Path to landmark occlusion score file.                                                                              |
| `--output`                | Path                                          | Yes     | —              | Output folder where overlay images will be saved.                                                                  |
| `--heatmap-mode`          | Choice: `occluded` \| `diff`                 | No      | `occluded`     | How to fill each cell:<br>• `occluded`: show raw occlusion score<br>• `diff`: show absolute difference from baseline |
| `--alpha`                 | Float                                         | No      | 0.6            | Transparency of the heatmap overlay.                                                                               |
| `--colormap`              | String                                        | No      | `hot`          | Matplotlib colormap name.                                                                                           |
| `--multiprocessing`       | Flag                                          | No      | Disabled       | Enable multiprocessing for faster processing.                                                                      |
| `--workers`               | Integer                                       | No      | 8              | Number of worker processes when multiprocessing is enabled.                                                        |
| `--log-level`             | String                                        | No      | `INFO`         | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`).                                                              |

#### 💻 Example Command:
```bash
python occlusion_heatmaps_builder.py \
    --regular /path/to/regular_scores.csv \
    --grid /path/to/grid_scores.csv \
    --landmark /path/to/landmark_scores.csv \
    --output /path/to/output_heatmaps \
    --heatmap-mode diff \
    --multiprocessing --workers 12
```
---

## 🧪 Suggested Workflow

Follow this end-to-end pipeline to perform occlusion-based interpretability analysis:

1. **Crop Dataset** → Run `crop_dataset_builder.py` to obtain clean, standardized face crops from your raw dataset.

2. **Generate Occlusions** → Use `occlusion_dataset_builder.py` to generate grid-based and landmark-based occluded variants of the cropped images.

3. **Model Prediction** → Pass all images (regular + occluded) through your MAD model to produce prediction score files (CSV format with image paths and scores).

4. **Visualize Heatmaps** → Run `occlusion_heatmaps_builder.py` with the regular/grid/landmark score files to create visual heatmaps showing which facial regions influence model decisions.

This pipeline helps evaluate which facial regions most influence the model's ability to detect morphing attacks and ensures a reproducible, automated analysis.

---

## 🧩 Additional Notes

- **Performance**: Use multiprocessing (`--multiprocessing --workers 8`) for large datasets to significantly speed up processing.
- **Storage**: Avoid writing directly to Google Drive during processing. Save locally, zip, then upload if needed to prevent I/O bottlenecks.
- **Dependencies**: All scripts automatically download required models (MTCNN, Dlib shape predictor) on first run.
- **Output Formats**: Heatmaps are saved as PNG images with the same folder structure as the input dataset for easy comparison.
