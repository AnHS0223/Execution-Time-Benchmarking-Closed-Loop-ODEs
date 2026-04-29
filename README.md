# Execution-Time-Benchmarking-Closed-Loop-ODEs
Compare the Execution time &amp; Training time between TPUs and GPUs for ML by solving Closed-Loop ODEs.

## Installation

First, ensure you have the common dependencies installed for data visualization and progress tracking:

```bash
pip install tqdm matplotlib
```

JAX installation depends on your hardware. Please follow the instructions below based on your environment:

for using gpu
```bash
pip install -U "jax[cuda12_pip]"
```

for using tpu
```bash
pip install -U "jax[tpu]"
```
