This experiment explores the MorDeephy framework from Medvedev et al. (2023), a publicly available benchmark designed to evaluate Morphing Attack Detection (MAD) methods. The repository does not offer a complete, ready-to-use implementation of the model described in the publication (DOI: [10.5220/0011606100003411](https://doi.org/10.5220/0011606100003411)); instead, it provides a structured protocol where researchers can integrate and test their own models. A custom Jupyter notebook was developed to verify this demonstration pipeline. The workflow confirms that the framework is built for benchmark integration, following key steps from data acquisition and alignment to the final calculation of performance metrics based on the prediction scores obtained.

---

### Notebook Cell Explanations

1.  **Clone Repository**
    * This cell clones the `MorDeephy` project from its GitHub repository to the local environment.

2.  **Download Datasets**
    * Executes the `download_data.py` script to fetch the required facial image datasets for the benchmark.
    * The script successfully downloads several datasets, but the log indicates a failure for the AR Face Database.

3.  **Extract Data**
    * This cell installs the `unlzw` and `patool` packages, which are required for file decompression.
    * It then runs the `data_extract.py` script to extract the downloaded archives into the `data_extracted/` directory.

4.  **Align Images**
    * This cell runs the `align_protocol_insf.py` script to preprocess and align the faces in the images according to the benchmark's protocol.
    * The script fails during execution with an `OpenCV error` when it attempts to process a file from a dataset that was not successfully extracted in the previous step.

5.  **Prediction Simulation**
    * This step runs the `sd_demo_extracting_predictions.py` script. This script's function within the demonstration pipeline is to simulate a model's output by generating illustrative prediction scores.
    * This serves to illustrate the data structure required by the benchmark, showing how prediction scores and ground-truth labels should be formatted for evaluation. It does not perform actual inference with a deep learning model.

6.  **Benchmark Execution**
    * The final cell executes `sd_benchmark_model.py`, which processes the simulated predictions generated in the prior step.
    * It computes and reports evaluation metrics, including APCER, BPCER, and an illustrative AUC ROC of 0.949. This output demonstrates the calculation functionality of the benchmark framework, showing the results format for when a real model's predictions are supplied.
