# GSVGD

[![Test](https://github.com/harrisonzhu508/M-SVGD/actions/workflows/tests.yml/badge.svg)](https://github.com/harrisonzhu508/M-SVGD/actions/workflows/tests.yml)


<!-- [![Test](./thumbnail/summary.pdf)](./thumbnail/summary.pdf) -->
[![Test](./thumbnail/summary.png)](./thumbnail/summary.png)

## Data
`Covertype` data downloaded from https://archive.ics.uci.edu/ml/datasets/covertype

## Other Dependencies
- Code for Sliced-SVGD is adapted from [Wenbo Gong's repo](https://github.com/WenboGong/Sliced_Kernelized_Stein_Discrepancy)
- Code for optimization on Grassmann manifold is adapted from [Pymanopt](https://www.pymanopt.org/)

## Run experiments
The code below runs the numerical experiments in the paper. 

1. The `.sh` scripts assume 8 GPUs are available. You can also CPUs by changing the arguments in these scripts to `--gpu=-1`.
2. These experiments can take hours to finish.
```
# install GSVGD module
pip install .

# e.g.1 run multivariate gaussian experiment and generate plots
sh scripts/run_gaussian.sh

# e.g.2 run conditioned diffusion and generate plots
sh scripts/run_diffusion.sh
```

## Run tests

```python
python -m pytest
```
