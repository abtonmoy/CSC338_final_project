#!/bin/bash
#SBATCH --job-name=vessel_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=vessel_train_%j.out

eval "$(micromamba shell hook --shell=bash)"
micromamba activate tfenv

export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cublas/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/home/jhub/jhub-venv/lib/python3.10/site-packages/nvidia/cusolver/lib:$LD_LIBRARY_PATH

jupyter execute new_main.ipynb