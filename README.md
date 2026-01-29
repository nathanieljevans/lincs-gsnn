# LINCS-GSNN

**Graph-Structured Neural Networks for Modeling Drug-Induced Gene Expression Dynamics**

LINCS-GSNN is a computational framework that combines graph neural networks with ordinary differential equations (ODEs) to model and explain drug-induced transcriptional responses. By embedding biological network structure directly into the neural architecture, LINCS-GSNN enables mechanistic interpretation of drug effects through biologically meaningful pathways.

## Overview

The framework integrates multiple data sources to construct a heterogeneous biological network:
- **Drug-target interactions** from Targetome Extended
- **Protein-protein interactions** and **transcription factor networks** from OmniPath
- **Gene regulatory relationships** from DOROTHEA

This network serves as the scaffold for a Graph Structured Neural Network (GSNN) that learns to predict gene expression derivatives (dX/dt) from initial cellular states and drug perturbations. The model can then be integrated over time using neural ODEs to simulate full transcriptional trajectories.

## Key Features

- **Biologically-structured architecture**: Network topology derived from curated biological databases ensures predictions flow through interpretable pathways
- **Trajectory prediction**: ODE integration enables prediction of gene expression dynamics over time
- **Contrastive explanations**: Identify network edges and nodes that explain differential drug responses between cell lines
- **Multi-modal integration**: Combines drug targets, protein interactions, and transcriptional regulation in a unified framework

## Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/your-username/lincs-gsnn.git
cd lincs-gsnn
```

2. Create the conda environment:
```bash
conda env create -f environment.yaml
conda activate lincs-gsnn
```

3. Install the package:
```bash
pip install -e .
```

4. Install GSNN packages: 

```bash 
pip install git+https://github.com/nathanieljevans/GSNN
```

5. Download relevant data 

See `data_availability.md` for data accessibility. Place the data in a `/data/` folder and update the relevant workflow paths. 

Note that this project is dependent on data generated from the `lincs-traj` workflow, which can be found [here](https://github.com/nathanieljevans/DeepTraj). 

6. Run workflows 

```bash
cd workflow/myworkflow/
snakemake -j 1
```

### Dependencies

The project requires the following main packages (see `environment.yaml` for full list):
- PyTorch
- PyTorch Geometric
- torchdiffeq
- pypath-omnipath
- snakemake
- scikit-learn
- pandas, numpy, matplotlib, seaborn

Additionally, this project depends on:
- [`gsnn`](https://github.com/nathanieljevans/gsnn) - Core GSNN model implementation
- [`DeepTraj`](https://github.com/nathanieljevans/DeepTraj) - Data generation for input into this model. 