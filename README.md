# SPARSE AUTOENCODERS REVEAL UNIVERSAL FEATURE SPACES ACROSS LARGE LANGUAGE MODELS

## Overview
This repository provides the code and resources for our research paper, **Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models**. In this study, we investigate how different large language models (LLMs) represent concepts in their latent spaces and identify shared features across models, addressing challenges like polysemanticity. By employing sparse autoencoders (SAEs), we transform model activations into interpretable spaces, revealing weakly shared features that enhance understanding of model interpretability, AI safety, and cross-model generalization.

- **Feature Matching Across Models**: Tools for aligning and comparing features across different LLMs using activation correlations.
- **Similarity Analysis**: Methods for analyzing representational similarity with metrics such as Singular Value Canonical Correlation Analysis (SVCCA).
- **Visualization**: Scripts for generating heatmaps

## Usage

In `run_pipeline/`, run:

`chmod +x run_pythia.sh`

`./run_pythia.sh --batch_size 300 --max_length 300 --num_rand_runs 1 --oneToOne_bool --model_A_endLayer 6 --model_B_endLayer 12 --layer_step_size 2`

(TBD- update .sh to do this) to eval separate model pairs in one run:

## Repository Structure
- `run_pipeline/`: Run the main results
- `main_results_nbs/`: Jupyter notebooks with code for experiments and analyses.
- `modal_scripts/`: Scripts for running experiments on Modal platforms.
- `README.md`: Project documentation

## Getting Started
### Prerequisites
- Python 3.8 or higher

### Installation
Clone the repository and install dependencies:

`pip install -r requirements.txt`

Install these SAE libraries:

pip install git+https://github.com/sae-lens/sae-lens.git

pip install git+https://github.com/EleutherAI/sparisfy.git

## Citations
If you use this code or our findings in your research, please cite our paper:

```
@misc{lan2025sparseautoencodersrevealuniversal,
      title={Sparse Autoencoders Reveal Universal Feature Spaces Across Large Language Models}, 
      author={Michael Lan and Philip Torr and Austin Meek and Ashkan Khakzar and David Krueger and Fazl Barez},
      year={2025},
      eprint={2410.06981},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.06981}, 
}
```

## TO DO
This repo is currently being restructured