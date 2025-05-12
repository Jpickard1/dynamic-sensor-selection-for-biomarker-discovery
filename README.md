# Dynamic Sensor Selection

This repository implements the **Dynamic Sensor Selection (DSS)** method, as proposed in the paper: [Dynamic Sensor Selection for Biomarker Discovery](https://arxiv.org/abs/2405.09809). DSS is an observability-guided sensor selection method designed to identify biomarkers that change throughout time.

## Structure
This repository contains two directories:

1. `src/`: Contains Python modules implementing DSS and related methods including [Dynamic Mode Decomposition (DMD)](https://www.annualreviews.org/content/journals/10.1146/annurev-fluid-030121-015835), [Data Guided Control (DGC)](https://www.pnas.org/doi/10.1073/pnas.1712350114), and estimation proceedures.
2. `notebooks/`: Includes Jupyter notebooks to recreate the figures and analyses presented in the paper

## Data

The notebooks are designed to run as is after downloading the corresponding data to the appropriate paths. Data analyzed in the paper can be accessed from the original publications that reported it. The datasets can be accessed here:
1. Cell Proliferation: https://www.pnas.org/doi/10.1073/pnas.1505822112
2. Cell Reprogramming: https://www.cell.com/iscience/fulltext/S2589-0042(18)30114-7
3. Pestecide treatment: https://www.nature.com/articles/s41467-023-37897-9
1. EEG Data: https://physionet.org/
2. Neural Imaging Data: https://www.pnas.org/doi/10.1073/pnas.2011140118

## Cite As:
```
@article{pickard2024dynamic,
  title={Dynamic Sensor Selection for Biomarker Discovery},
  author={Pickard, Joshua and Stansbury, Cooper and Surana, Amit and Muir, Lindsey and Bloch, Anthony and Rajapakse, Indika},
  journal={arXiv preprint arXiv:2405.09809},
  year={2024}
}
```
