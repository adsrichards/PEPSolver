# PEPSolver

Code for calculating infinite projected entangle-pair state (iPEPS) quantum tensor-network states based on the corner-transfer-matrix renormalization group (CTMRG) algorithm using LibTorch and automatic differentiation.

Based on the Python implementation: [github.com/wangleiphy/tensorgrad](https://github.com/wangleiphy/tensorgrad).

## Installation

Compilation of PEPSolver requires a distribution of LibTorch. The latest version of LibTorch can be obtained with

```bash
wget https://download.pytorch.org/libtorch/nightly/cpu/libtorch-shared-with-deps-latest.zip
unzip libtorch-shared-with-deps-latest.zip
```

Then PEPSolver can be built and installed with

```bash
mkdir build && cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
```