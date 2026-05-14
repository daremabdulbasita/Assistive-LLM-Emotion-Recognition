# Emotion Recognition in Disabled Individuals Through an Improved Assistive Communication System Based on Large Language Models and Ensemble Deep Learning

This repository contains the official implementation of the paper:
**"Emotion Recognition in Disabled Individuals Through an Improved Assistive Communication System Based on Large Language Models and Ensemble Deep Learning"**
Published in: **The Visual Computer**

## Overview
TERLLMV-DLN is a hybrid deep learning architecture designed for high-precision emotion classification from textual data. It integrates Large Language Model (LLM) embeddings with a multi-branch neural network consisting of:
- **Temporal Convolutional Network (TCN)** for capturing long-range dependencies.
- **Stacked Denoising Autoencoder (SDAE)** for robust feature extraction.
- **Ensemble Neural Network (ENN)** for sequential pattern recognition.

## Proposed Methodology
The architecture follows a three-stage pipeline:
1. **Preprocessing:** Advanced text cleaning, lemmatization, and class balancing.
2. **Vectorization:** Leveraging `all-MiniLM-L6-v2` Sentence Transformers for semantic feature representation.
3. **Classification:** A late-fusion ensemble of TCN, SDAE, and ENN branches followed by dense layers and softmax activation.

## Repository Structure
- `code.ipynb`: The main Jupyter notebook containing the full implementation, from data preprocessing to model evaluation.
- `terllmv_dln_implementation.py`: A clean, production-ready Python script version of the model.
- `requirements.txt`: List of Python dependencies required to run the project.

## Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. You can install the dependencies using:
```bash
pip install -r requirements.txt
```

### Usage
1. Open the `code.ipynb` notebook in Jupyter Lab or Google Colab.
2. Follow the instructions to download the datasets (links are provided in the notebook comments).
3. Run the cells sequentially to preprocess data, train the model, and generate evaluation plots.

## Datasets
The implementation utilizes two benchmark emotion datasets:
1. [Emotions Dataset 1 (Kaggle)](https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset)
2. [Emotion Detection from Text (Kaggle)](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)

## Citation
If you use this code or methodology in your research, please cite our work:
```text
[Your Name/Author List], "Emotion Recognition in Disabled Individuals Through an Improved Assistive Communication System Based on Large Language Models and Ensemble Deep Learning", The Visual Computer, [Year].
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

---
**DOI:** [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.[Insert_Your_DOI_Here].svg)](https://doi.org/10.5281/zenodo.[Insert_Your_DOI_Here])
