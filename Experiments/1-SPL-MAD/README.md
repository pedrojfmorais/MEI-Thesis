# 🧠 SPL-MAD – Project Overview

This experiment supports the experiments and evaluation of the **Self-Paced Learning Morphing Attack Detection (SPL-MAD)** system, proposed by *Fang et al., 2022* (DOI: [10.1109/IJCB54206.2022.10008003](https://doi.org/10.1109/IJCB54206.2022.10008003)). The approach focuses on **unsupervised anomaly detection** for morphing attack detection, where the model learns to distinguish bona fide facial representations from morphs through a **self-paced learning mechanism**, gradually including more challenging samples during training to improve generalization.

---

## 📁 Notebook Summaries

### 1. **`SPL-MAD.ipynb`**
#### 🔍 Objective:
Implements the dataset preparation and experimental setup for SPL-MAD, covering data organization and preprocessing across multiple datasets, including **FRLL-Morphs** and **SMDD**.

#### 🧬 Code Description:
- **Environment Setup**:
  - Mounts Google Drive and configures all dataset and script paths.
  - Downloads or synchronizes required datasets (e.g., SMDD, FRLL-Morphs).
- **Dataset Preparation**:
  - Structures image data for use in the SPL-MAD pipeline.
  - Executes helper scripts (e.g., `frll_morphs_create_csv.py`) to generate CSV metadata files for different FRLL-Morph subsets.
- **Training Pipeline Support**:
  - Prepares and organizes all necessary resources for subsequent SPL-MAD training and testing stages.
  - Although model training is performed using the papers code, made available publicly by the authors, this notebook ensures datasets are properly formatted and accessible.

---

### 2. **`SPL-MAD_ERR-Analysis.ipynb`**
#### 🔍 Objective:
Performs error rate analysis on experimental results from the SPL-MAD pipeline, focusing on **Equal Error Rate (EER)** and other performance indicators.

#### 🧬 Code Description:
- **Setup**:
  - Mounts Google Drive and locates experiment log files.
- **Log Analysis**:
  - Executes the script `analyze_logs.py` to parse EER results and summarize performance across datasets.
  - Processes multiple datasets and morphing methods, including **AMSL**, **FaceMorpher**, **OpenCV**, **StyleGAN**, and **WebMorph**.
- **Evaluation**:
  - Aggregates and compares error metrics from tests on **SMDD** and **FRLL-Morphs** datasets.
  - Facilitates visual and quantitative assessment of SPL-MAD’s robustness under diverse morphing techniques.

---

## 🧪 Suggested Workflow

1. **Dataset Preparation**:
   - Run `SPL-MAD.ipynb` to mount the environment, organize datasets, and generate required CSV metadata.
2. **Model Training & Testing**:
   - Train and evaluate the SPL-MAD system using the prepared data.
3. **Error Analysis**:
   - Use `SPL-MAD_ERR-Analysis.ipynb` to analyze and visualize EER metrics from experiment logs.

---

## 🧩 Additional Notes
- Designed for execution in **Google Colab** environments.
- Requires project-specific scripts such as:
  - `/MyDrive/mad/code/SPL-MAD/EER Analysis/analyze_logs.py`
  - `/MyDrive/mad/datasets/scripts/frll_morphs_create_csv.py`
- Ensure that dataset paths and filenames are correctly configured before running.



