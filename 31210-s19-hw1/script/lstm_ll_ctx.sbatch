#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --mem=16G
#SBATCH --time=08:00:00

module load cuda/9.0
module load Anaconda3/5.0.1
source activate pytorch_cuda_9.0
module unload gcc


echo 'python -u ../code/lstm_ll_ctx.py > ../output/lstm_ll_ctx.pyout'
python -u ../code/lstm_ll_ctx.py > ../output/lstm_ll_ctx.pyout
