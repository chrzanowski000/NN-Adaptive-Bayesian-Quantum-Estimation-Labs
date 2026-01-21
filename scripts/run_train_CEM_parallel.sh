#!/usr/bin/env bash

# -------- common thread limits --------
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

# -------- run A on cores 0–4 --------
taskset -c 4-7 \
python -m pipeline.train_CEM