#!/bin/bash
#SBATCH --job-name=gan_demo
#SBATCH --partition=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gan_demo.log

export PATH=$HOME/.local/bin:$PATH
python3 train_gan.py

