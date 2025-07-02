# 🧠 Zhang2024 Occlusion Analysis – Project Overview

This repository supports the analysis and evaluation of occlusion effects on facial morphing attack detection systems, based on the architecture proposed by *Zhang et al., 2024* (DOI: [10.1109/CVPRW63382.2024.00158](https://doi.org/10.1109/CVPRW63382.2024.00158)). It uses a **Vision Transformer (ViT)** to extract deep visual features, followed by a **Support Vector Machine (SVM)** classifier to distinguish between bona fide and morphed facial images.

---

## 📁 Notebook Summaries

### 1. **`Zhang2024.ipynb`**
#### 🔍 Objective:
Implements the morphing detection pipeline where a **Vision Transformer (ViT)** extracts facial representations and a **SVM** classifier determines whether the input is genuine or morphed.

#### 🧬 Code Description:
- **Environment Setup**:
  - Imports, GPU selection, and dependencies.
- **Dataset Loading**:
  - Reads bona fide and morphed face images from directories.
  - Applies preprocessing transforms compatible with ViT.
- **Feature Extraction**:
  - Loads a pre-trained ViT model (e.g., `vit_b_16`) with the classification head removed.
  - Extracts embedding vectors for each image.
- **SVM Classification**:
  - Trains a SVM on the extracted embeddings.
  - Evaluates using EER, accuracy, AUC, and confusion matrix.
- **Export**:
  - Saves evaluation metrics and trained model.

---

### 2. **`Zhang2024_Analysis.ipynb`**
#### 🔍 Objective:
Analyzes the impact of occlusions on the classifier's decisions. Provides insight into which facial regions most affect classification.

#### 🧬 Code Description:
- **Loads occlusion maps and classification scores**.
- **Aggregates occlusion effects**:
  - Computes average degradation in classifier confidence per region.
- **Visualization**:
  - Generates heatmaps showing model sensitivity to occluded areas.
  - Highlights critical facial zones based on performance drop.

---

### 3. **`OcclusionMappingForDatasets.ipynb`**
#### 🔍 Objective:
Generates occluded versions of datasets to evaluate the robustness of morphing detection systems under controlled occlusion scenarios.

#### 🧬 Code Description:
- **Configuration**:
  - Chooses occlusion type: grid-based (e.g., 3x3) or landmark-based.
  - Defines mask shape, size, and opacity.
- **Image Processing**:
  - Detects landmarks using Dlib.
  - Applies black masks to selected grid cells or facial landmarks.
- **Output**:
  - Saves modified images preserving original dataset structure.
  - Supports batch processing.

---

## 🧪 Suggested Workflow

1. **Generate Occlusions**:
   - Use `OcclusionMappingForDatasets.ipynb` to apply occlusions.
2. **Train Baseline and Evaluate**:
   - Run `Zhang2024.ipynb` on regular and occluded data.
4. **Analyze Occlusion Sensitivity**:
   - Run `Zhang2024_Analysis.ipynb` to identify vulnerable regions.

