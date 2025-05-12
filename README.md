# Dynamic Sensor Selection

This repository implements the **Dynamic Sensor Selection (DSS)** method, as proposed in the paper: [Dynamic Sensor Selection for Biomarker Discovery](https://arxiv.org/abs/2405.09809). DSS is an observability-guided sensor selection method designed to identify biomarkers that change throughout time.

## Structure
This repository contains two directories:

1. `src/`: Contains Python modules implementing DSS and related methods including [Dynamic Mode Decomposition (DMD)](https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-030121-015835), [Data Guided Control (DGC)](https://www.pnas.org/doi/10.1073/pnas.1712350114), and estimation proceedures.
2. `notebooks/`: Includes Jupyter notebooks to recreate the figures and analyses presented in the paper

## Cite As:
```
@article{pickard2024dynamic,
  title={Dynamic Sensor Selection for Biomarker Discovery},
  author={Pickard, Joshua and Stansbury, Cooper and Surana, Amit and Muir, Lindsey and Bloch, Anthony and Rajapakse, Indika},
  journal={arXiv preprint arXiv:2405.09809},
  year={2024}
}
```
