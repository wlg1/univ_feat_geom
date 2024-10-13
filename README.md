# SPARSE AUTOENCODERS REVEAL UNIVERSAL FEATURE SPACES ACROSS LARGE LANGUAGE MODELS


## Overview
This repository provides the code and resources for our research paper, **Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models**. In this study, we investigate how different large language models (LLMs) represent concepts in their latent spaces and identify shared features across models, addressing challenges like polysemanticity. By employing sparse autoencoders (SAEs), we transform model activations into interpretable spaces, revealing universal features that enhance understanding of model interpretability, AI safety, and cross-model generalization.

## Features
- **Sparse Autoencoder (SAE) Implementation**: Code for training SAEs to decompose LLM activations into interpretable feature spaces.
- **Feature Matching Across Models**: Tools for aligning and comparing features across different LLMs using activation correlations.
- **Similarity Analysis**: Methods for analyzing representational similarity with metrics such as Singular Value Canonical Correlation Analysis (SVCCA).
- **Visualization**: Scripts for generating heatmaps and visual aids to illustrate feature universality.

## Repository Structure
- `notebooks/`: Jupyter notebooks with code for experiments and analyses.
- `modal_scripts/`: Scripts for running experiments on various computational platforms.
- `__nbs_as_py/`: Python scripts generated from Jupyter notebooks.
- `README.md`: Project documentation (you are here).

## Getting Started
### Prerequisites
- Python 3.8 or higher
- Dependencies listed in `requirements.txt`

### Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/univ_feat_geom.git
cd univ_feat_geom
pip install -r requirements.txt


### Usage
Train Sparse Autoencoders: Use the scripts in modal_scripts/ or the notebooks in notebooks/ to train SAEs on LLM activations.
Feature Matching and Analysis: Follow the steps in the notebooks to align features across models and perform similarity analysis.
Visualization: Generate heatmaps and other visualizations to illustrate the results.
Citations

If you use this code or our findings in your research, please cite our paper:

@article{barez2024sparse,
  title={Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models},
  author={Lan, Michael and Torr, Philip and Meek, Austin and Khakzar, Ashkan and Krueger, David and Barez, Fazl},
  journal={Journal/Conference Name},
  year={2024}
}

