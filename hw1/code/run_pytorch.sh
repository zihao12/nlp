#!/bin/bash

module load cuda/9.0
module load Anaconda3/5.0.1
source activate pytorch_cuda_9.0
module unload gcc
which python

python lstm_neg.py
