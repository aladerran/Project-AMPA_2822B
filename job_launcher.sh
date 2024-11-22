#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request n CPU core
#SBATCH -n 1
#SBATCH -t 03:00:00

# Load a CUDA module
module load cuda

# Set up
./setup.sh
export PYTHONPATH=./python
export NEEDLE_BACKEND=nd

# Run program
nsys profile -o train_profile python ./train.py > train.log 2>&1
