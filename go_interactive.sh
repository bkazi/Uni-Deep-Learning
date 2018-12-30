#!/bin/bash
module load languages/anaconda2/5.0.1
module load libs/cudnn
pip install --user librosa tensorflow-gpu
srun -p gpu --cpus-per-task=10 --gres=gpu:1 -A comsm0018 -t 0-02:00 --mem=100G --pty bash
