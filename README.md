# 🎓 Face Morphing Detection in Identification Documents

**Master's Thesis in Informatics Engineering**  
**Specialization in Software Engineering**  
Instituto Superior de Engenharia de Coimbra (ISEC)

**Author:** Pedro Jorge Fernandes Morais  
**Supervisor:** Inês Campos Monteiro Sabino Domingues  
**Co-Supervisor:** Teresa Raquel Corga Teixeira da Rocha  
**Date:** October 2025

---

## 📋 Abstract

The falsification of identification documents presents a remarkable threat to public security, personal data protection, and fraud prevention. A growing concern within this domain is the use of **morphed facial images**, a deceptive technique that has become more prevalent with the rapid advancement of artificial intelligence and image processing technologies.

**Morphed images** are created by blending the facial features of two or more individuals to create a composite image that resembles all contributors. This manipulation enables multiple individuals to use a single identification document, for example, to gain unauthorized access across international borders.

Detecting these morphed images poses a considerable challenge. Humans are often unable to detect the manipulation, and even advanced facial recognition systems struggle to identify these sophisticated attacks. This thesis investigates **deep learning techniques for Morphing Attack Detection (MAD)**, with a particular focus on:

- Evaluating state-of-the-art MAD approaches across diverse datasets
- Exploring **Vision Transformer (ViT)** architectures for morphing detection
- Developing **occlusion-based interpretability methods** to understand model decision-making
- Analyzing the robustness of detection methods under various morphing techniques
- Identifying critical facial regions for accurate morphing detection

---

## 🎯 Research Objectives

This thesis addresses the following key research questions:

1. **How can deep learning techniques be used to improve Morphing Attack Detection?**
2. **What are the challenges in detecting sophisticated morphing attacks?**
3. **How can we improve model interpretability and explainability in MAD systems?**
4. **Which facial regions are most critical for accurate morphing detection?**

The research focuses on:
- Replicating and evaluating existing MAD methods from recent literature
- Investigating Vision Transformers as an underexplored architecture for MAD
- Developing novel occlusion-based interpretability techniques
- Benchmarking performance across multiple datasets and morphing techniques
- Analyzing generalization capabilities across different attack types

---

## 📁 Repository Structure

This repository contains two main folders:

### 1. **Experiments/**
Contains all experimental implementations and evaluations based on published research papers. Each subfolder represents a distinct experiment including dataset preparation, model implementation, evaluation, and analysis.

**Experiments included:**
- **1-SPL-MAD**: Self-Paced Learning approach for unsupervised MAD (Fang et al., 2022)
- **2-MorDeephy-Benchmark**: Benchmark framework for evaluating MAD methods (Medvedev et al., 2023)
- **3-Attack-Agnostic-Features**: Attack-agnostic feature extraction (Colbois et al., 2024) - partial implementation
- **5-ViT_SVM-OcclusionMapping**: Vision Transformer + SVM with occlusion analysis (Zhang et al., 2024)

Each experiment folder contains detailed README files with specific setup instructions, workflow steps, and results.

[📖 View Experiments README](Experiments/README.md)

---

### 2. **Occlusion-based-Interpretability/**
Contains the **original contribution** of this thesis: a complete pipeline for occlusion-based interpretability analysis of MAD models.

**Core Components:**

#### **crop_dataset_builder.py**
Prepares clean, standardized datasets by cropping and resizing facial images using MTCNN face detection. Ensures consistent face regions across all images before occlusion experiments.

**Key Features:**
- Automatic face detection and cropping
- Preserves folder structure
- Multiprocessing support for large datasets
- Detailed logging

#### **occlusion_dataset_builder.py**
Generates occluded variants of facial images to test MAD model robustness under controlled occlusion scenarios.

**Occlusion Types:**
- **Grid-Based**: Divides face into R×C grid and occludes one cell at a time
- **Landmark-Based**: Uses Dlib 68-point facial landmarks to occlude semantic regions (eyes, nose, mouth)

**Key Features:**
- Flexible occlusion modes (grid, landmark, or both)
- Automatic Dlib model download
- Multiprocessing support
- Preserves dataset structure

#### **occlusion_heatmaps_builder.py**
Creates visual heatmaps that highlight which facial regions influence the model's predictions most.

**Key Features:**
- Combines grid and landmark occlusion scores
- Generates composite heatmaps overlaid on original images
- Supports multiple visualization modes (raw scores or difference from baseline)
- Identifies vulnerable facial regions for MAD systems

**End-to-End Workflow:**
1. **Crop** → Use `crop_dataset_builder.py` to obtain standardized face crops
2. **Occlude** → Use `occlusion_dataset_builder.py` to generate occluded variants
3. **Predict** → Pass images through MAD model to produce prediction scores
4. **Visualize** → Use `occlusion_heatmaps_builder.py` to create interpretability heatmaps

[📖 View Occlusion-based Interpretability README](Occlusion-based-Interpretability/README.md)

---

## 🔬 Key Findings

### Deep Learning Techniques for MAD

1. **Feature Extraction**: CNNs and Vision Transformers effectively automate feature extraction, identifying subtle manipulation traces that traditional methods miss

2. **Hybrid Approaches**: Combining multiple feature types (high-frequency, RGB, wavelet domain) improves detection accuracy across different morphing techniques

3. **Vision Transformers**: ViT-based architectures show competitive performance and strong potential for MAD applications, though remain underexplored in the literature

4. **Generalization Challenges**: Models often struggle with cross-dataset and cross-technique generalization, particularly with unseen morphing methods

5. **Image Type Variations**: Digital images yield better detection results than Print & Scan (P&S) images, which are common in real-world passport verification scenarios

### Model Interpretability

The occlusion-based interpretability method developed in this thesis reveals:

- **Critical Facial Regions**: Mouth and central nose areas are typically most important for model predictions
- **Spatial Dependencies**: Grid-based occlusion provides fine-grained understanding of spatial importance
- **Semantic Regions**: Landmark-based occlusion identifies which facial features (eyes, nose, mouth) contribute most to detection
- **Model Transparency**: Heatmap visualizations enhance trust and auditability in security-critical applications

---

## 📊 Datasets Used

- **FRLL-Morphs**: Face Research London Lab morphing dataset
- **SMDD**: Synthetic Morphing Detection Dataset
- **FFHQ-Morphs**: High-quality facial morphs based on Flickr-Faces-HQ
- **FRGC-Morphs**: Face Recognition Grand Challenge morphing dataset

**Morphing Techniques Evaluated:**
- AMSL
- FaceMorpher
- OpenCV
- StyleGAN
- WebMorph
- Landmark-based morphing
- Diffusion-based morphing

---

## 🛠️ Technologies & Tools

- **Deep Learning Frameworks**: TensorFlow, PyTorch
- **Computer Vision**: OpenCV, Dlib, MTCNN
- **Architectures**: CNNs (VGG, ResNet, AlexNet), Vision Transformers (ViT), SVMs
- **Interpretability**: Custom occlusion mapping, heatmap visualization
- **Evaluation**: ROC curves, DET curves, EER, APCER, BPCER
- **Environment**: Google Colab with GPU support

---

## 📈 Evaluation Metrics

The thesis employs comprehensive evaluation metrics:

- **D-EER (Equal Error Rate)**: Point where APCER equals BPCER
- **APCER @ BPCER**: Attack presentation error at fixed bona fide error rates (1%, 5%, 10%)
- **Accuracy**: Overall classification accuracy
- **ROC Curves**: True positive rate vs. false positive rate
- **DET Curves**: Detection error trade-off visualization
- **Occlusion Heatmaps**: Visual interpretability of model decisions

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Colab account (for running experiments)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/pedrojfmorais/MEI-Thesis.git
   cd MEI-Thesis
   ```

2. **Explore Experiments**
   - Navigate to `Experiments/` folder
   - Each subfolder contains Jupyter notebooks and detailed READMEs
   - Most experiments are designed for Google Colab execution

3. **Use Methodology Tools**
   - Navigate to `Occlusion-based-Interpretability/` folder
   - Install dependencies:
     ```bash
     pip install numpy opencv-python dlib matplotlib tqdm tensorflow mtcnn pillow
     ```
   - Follow the workflow: crop → occlude → predict → visualize

---

## 📞 Contact

**Pedro Jorge Fernandes Morais**  
a21280686@isec.pt  
Instituto Superior de Engenharia de Coimbra (ISEC)  
Master's in Informatics Engineering

For questions or collaboration opportunities, please send an email.

---

## 📖 Citation

If you use this work in your research, please cite:

```
@masterthesis{moraispedrojorgefernandes2026,
	author	=	"Morais, Pedro Jorge Fernandes",
	title	=	"Face morphing detection in identification documents",
	year	=	"2026"
}
```

---

## 🙏 Acknowledgments

Special thanks to:
- Supervisor: Inês Campos Monteiro Sabino Domingues
- Co-Supervisor: Teresa Raquel Corga Teixeira da Rocha
- Authors who shared their implementations: Fang et al. (2022), Medvedev et al. (2023), Colbois et al. (2024), and Zhang et al. (2024)
- Instituto Superior de Engenharia de Coimbra (ISEC)

---

## 📜 License

This repository is provided for academic and research purposes. Please refer to individual experiment folders for specific licensing information related to third-party code.
