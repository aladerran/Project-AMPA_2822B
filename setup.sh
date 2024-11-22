#!/bin/bash
set -e
# setup.sh

cd /users/slin153/workspace/CMU10-714/Project-AMPA_2822B

# Load CUDA module
module load cuda

# Install Python dependencies
pip install -r ./requirements.txt

# Build backend
make

# Create necessary directories
mkdir -p ./data/ptb
mkdir -p ./data

# Download Penn Treebank dataset
ptb_data="https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb."
for file in train.txt test.txt valid.txt; do
    if [ ! -f "./data/ptb/$file" ]; then
        wget "${ptb_data}${file}" -O "./data/ptb/$file"
    fi
done

# Download CIFAR-10 dataset
if [ ! -d "./data/cifar-10-batches-py" ]; then
    wget "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz" -O "./data/cifar-10-python.tar.gz"
    tar -xvzf "./data/cifar-10-python.tar.gz" -C "./data"
    rm "./data/cifar-10-python.tar.gz"
fi
