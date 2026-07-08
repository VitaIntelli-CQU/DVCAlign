# DVCAlign

DVCAlign is a Python package for alignment and integration of spatial transcriptomics data across slices, conditions, platforms, and developmental stages.

## Repository

- GitHub: https://github.com/VitaIntelli-CQU/DVCAlign
- Author: Cheng Wei
- Contact: 2804775192@qq.com

## Contents

This repository currently includes:

- the core package code in `DVCAlign/`
- dependency files for Linux and macOS
- packaging metadata in `setup.py`

Large datasets are not bundled in the repository.

## Installation

Create a clean Python environment first:

```bash
conda create -n env_DVCAlign python=3.8
conda activate env_DVCAlign
```

Install dependencies:

```bash
pip install -r requirement.txt
```

For macOS:

```bash
pip install -r requirement_for_macOS.txt
```

Then install the package itself:

```bash
pip install -e .
```

## Notes

- `mclust_R` requires both `rpy2` in Python and the R package `mclust`.
- `torch-geometric`, `torch-scatter`, `torch-sparse`, and `torch-cluster` must match your local PyTorch environment.

## Minimal Usage

```python
import DVCAlign

DVCAlign.Cal_Spatial_Net(adata, rad_cutoff=150)
adata = DVCAlign.train_DVCAlign(adata, verbose=True)
adata = DVCAlign.mclust_R(adata, num_cluster=7, used_obsm="DVCAlign")
```

## Citation

If this code supports your research, please cite the DVCAlign method paper:

Zhou, X., Dong, K. and Zhang, S. Integrating spatial transcriptomics data across different conditions, technologies and developmental stages. Nat Comput Sci 3, 894-906 (2023). https://doi.org/10.1038/s43588-023-00528-w
