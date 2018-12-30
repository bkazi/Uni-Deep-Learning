#!/bin/bash
module load languages/anaconda2/5.0.1
module load libs/cudnn
pip install --user librosa tensorflow-gpu
srun -p gpu_veryshort --cpus-per-task=10 --gres=gpu:1 -A comsm0018 --mem=100G --pty bash
