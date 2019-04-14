#!/bin/bash

#SBATCH --partition=broadwl
#SBATCH --mem=16G
#SBATCH --time=08:00:00

module load cuda/9.0
module load Anaconda3/5.0.1
source activate pytorch_cuda_9.0
module unload gcc


echo 'python -u ../code/lstm_neg.py 500 0.5 > ../output/lstm_neg_r500_f0.5.pyout'
python -u ../code/lstm_neg.py 500 0.5 > ../output/lstm_neg_r500_f0.5.pyout