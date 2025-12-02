# 📓 Example Notebooks – Methodology Tools Usage

This folder contains **Google Colab notebooks** demonstrating the practical usage of the three core methodology scripts developed for this thesis. These notebooks provide end-to-end examples of dataset preparation, occlusion generation, and heatmap visualization for morphing attack detection research.

---

## 📁 Notebook Summaries

### 1. **`Crop-Dataset-Builder.ipynb`**
#### 🔍 Objective:
Demonstrates how to use `crop_dataset_builder.py` to detect faces and crop them to a standardized size using MTCNN face detection in a Google Colab environment.

#### 🧬 Code Description:
- **Environment Setup**:
  - Mounts Google Drive to access datasets and scripts.
  - Installs MTCNN library for face detection.
- **Face Cropping Execution**:
  - Runs `crop_dataset_builder.py` with multiprocessing enabled (12 workers).
  - Processes 7,265 images from the FRLL-Morphs dataset.
  - Crops detected faces to standardized dimensions.
  - Saves output to local Colab storage (not directly to Drive for performance).
- **Dataset Packaging**:
  - Zips the entire cropped dataset with progress tracking.
  - Copies the zip file to Google Drive for persistent storage.
  - Demonstrates best practices for handling large datasets in Colab.
- **Cleanup**:
  - Unmounts Google Drive and releases Colab resources.

#### 📊 Example Results:
- **Input**: 7,265 raw facial images
- **Output**: 7,265 cropped face images (943 MB compressed)
- **Processing Time**: ~11 minutes (0.092 seconds per image)
- **Configuration**: 12 parallel workers on Google Colab GPU

---

### 2. **`Occlusion-Dataset-Builder.ipynb`**
#### 🔍 Objective:
Demonstrates how to use `occlusion_dataset_builder.py` to generate occluded versions of facial images using both grid-based and landmark-based occlusion methods.

#### 🧬 Code Description:
- **Environment Setup**:
  - Mounts Google Drive to access datasets and scripts.
  - Automatically downloads Dlib 68-point shape predictor model.
- **Occlusion Generation**:
  - Runs `occlusion_dataset_builder.py` in `both` mode (grid + landmark).
  - Applies grid-based occlusion with custom 6×5 grid configuration.
  - Applies landmark-based occlusion with 1.5× scale factor and bounding box precision.
  - Uses multiprocessing with 12 workers for efficient batch processing.
  - Processes 7,265 images, generating 247,010 occluded variants.
- **Dataset Packaging**:
  - Zips all occluded images with progress tracking.
  - Copies the 13.6 GB zip file to Google Drive.
  - Demonstrates handling of very large output datasets.
- **Cleanup**:
  - Unmounts Google Drive and releases Colab resources.

#### 📊 Example Results:
- **Input**: 7,265 facial images
- **Output**: 247,010 occluded images (13.6 GB compressed)
- **Processing Time**: ~1 hour 52 minutes (0.027 seconds per occluded image)
- **Configuration**: 6×5 grid, landmark scale 1.5×, bbox precision, 12 workers

---

### 3. **`Occlusion-Heatmaps-Builder.ipynb`**
#### 🔍 Objective:
Demonstrates how to use `occlusion_heatmaps_builder.py` to generate interpretability heatmaps from model prediction scores on regular, grid-occluded, and landmark-occluded images.

#### 🧬 Code Description:
- **Environment Setup**:
  - Mounts Google Drive to access prediction score files and datasets.
- **Heatmap Generation**:
  - Runs `occlusion_heatmaps_builder.py` with three input CSV files:
    - Regular scores (baseline predictions)
    - Grid occlusion scores
    - Landmark occlusion scores
  - Generates three types of heatmaps:
    - **Grid heatmaps**: Shows sensitivity per grid cell
    - **Landmark heatmaps**: Shows sensitivity per facial region
    - **Composite heatmaps**: Combines grid and landmark information
  - Uses `diff` mode to visualize absolute difference from baseline scores.
  - Enables multiprocessing for faster processing.
- **Output**:
  - Saves heatmap overlays as PNG images organized by type.
  - Preserves original dataset folder structure.
  - Provides visual interpretability of model decisions.

#### 📋 Input CSV Requirements:
All three CSV files must follow the format: `image_path,score[,label]`

1. **Regular Scores**: Baseline predictions on non-occluded images
2. **Grid Scores**: Predictions with grid cell occlusions (path contains grid indices like `r0_c0`)
3. **Landmark Scores**: Predictions with landmark occlusions (path contains region names like `left_eye`)

See the main [Methodology-Contribution README](../README.md#-csv-file-structure) for detailed CSV format specifications.

---

## 🧪 Suggested Workflow

Follow this sequence to reproduce the complete occlusion-based interpretability pipeline:

1. **Crop Dataset**:
   - Run `Crop-Dataset-Builder.ipynb` to standardize face images.
   - Download and extract the cropped dataset zip from Google Drive.

2. **Generate Occlusions**:
   - Run `Occlusion-Dataset-Builder.ipynb` on the cropped dataset.
   - Download and extract the occluded dataset zip from Google Drive.

3. **Run Model Predictions**:
   - Use your MAD model to generate predictions on:
     - Regular (cropped) images
     - Grid-occluded images
     - Landmark-occluded images
   - Save predictions as CSV files following the required format.

4. **Generate Heatmaps**:
   - Run `Occlusion-Heatmaps-Builder.ipynb` with the three prediction CSV files.
   - Analyze the generated heatmaps to understand model behavior.

---

## 🧩 Additional Notes

### Google Colab Best Practices:
- **Never write directly to Google Drive**: Always save to `/content` first, then zip and copy to Drive.
- **Use multiprocessing**: Speeds up processing.
- **Monitor resource usage**: Large datasets may require Colab Pro for extended runtime.
- **Zip before transferring**: Reduces sync issues and speeds up Drive uploads.

### Performance Tips:
- **Crop Dataset Builder**: ~0.09 sec/image with 12 workers
- **Occlusion Dataset Builder**: ~0.03 sec/occluded image with 12 workers
- **Heatmaps Builder**: Processing time depends on number of images and heatmap types

### Dataset Sizes:
- **Cropped Dataset**: ~943 MB for 7,265 images
- **Occluded Dataset**: ~13.6 GB for 247,010 images (6×5 grid + landmarks)
- **Heatmaps**: Size varies based on output image resolution

### Requirements:
- **Google Colab** with GPU runtime (recommended)
- **Google Drive** with sufficient storage space
- **Python packages**: Automatically installed in notebooks (MTCNN, Dlib, OpenCV, etc.)

---

## 📚 Related Documentation

- **[Methodology-Contribution README](../README.md)**: Detailed documentation of all three Python scripts
- **[Main Thesis README](../../README.md)**: Overview of the entire thesis repository
- **[Experiments README](../../Experiments/README.md)**: Related experimental implementations

