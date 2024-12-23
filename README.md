# SPARSE AUTOENCODERS REVEAL UNIVERSAL FEATURE SPACES ACROSS LARGE LANGUAGE MODELS


## Overview
This repository provides the code and resources for our research paper, **Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models**. In this study, we investigate how different large language models (LLMs) represent concepts in their latent spaces and identify shared features across models, addressing challenges like polysemanticity. By employing sparse autoencoders (SAEs), we transform model activations into interpretable spaces, revealing universal features that enhance understanding of model interpretability, AI safety, and cross-model generalization.

## Features
- **Feature Matching Across Models**: Tools for aligning and comparing features across different LLMs using activation correlations.
- **Similarity Analysis**: Methods for analyzing representational similarity with metrics such as Singular Value Canonical Correlation Analysis (SVCCA).
- **Visualization**: Scripts for generating heatmaps and visual aids to illustrate feature universality.

## Repository Structure
- `main_results_nbs/`: Jupyter notebooks with code for experiments and analyses.
- `modal_scripts/`: Scripts for running experiments on various computational platforms.
- `README.md`: Project documentation (you are here).

## Getting Started
### Prerequisites
- Python 3.8 or higher

### Installation
Clone the repository and install dependencies:

```bash```
git clone https://github.com/yourusername/SAE_feature_univ.git \
cd SAE_feature_univ \
pip install -r requirements.txt


### Usage
Feature Matching and Analysis: Follow the steps in the notebooks to align features across models and perform similarity analysis.
Visualization: Generate heatmaps and other visualizations to illustrate the results.
Citations

If you use this code or our findings in your research, please cite our paper:

```
@article{lan2024sparse,
  title={Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models},
  author={Lan, Michael and Torr, Philip and Meek, Austin and Khakzar, Ashkan and Krueger, David and Barez, Fazl},
  journal={Journal/Conference Name},
  year={2024}
}

