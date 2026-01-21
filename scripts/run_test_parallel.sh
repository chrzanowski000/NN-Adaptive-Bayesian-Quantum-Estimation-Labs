#!/usr/bin/env bash

# -------- common thread limits --------
export OMP_NUM_THREADS=5
export MKL_NUM_THREADS=5
export OPENBLAS_NUM_THREADS=5
export NUMEXPR_NUM_THREADS=5

# -------- run A on cores 0–4 --------
taskset -c 0-4 \
python -m pipeline.test_CEM_N