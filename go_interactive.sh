#!/bin/bash
module load libs/cudnn
pip install --user librosa tensorflow-gpu
srun -p gpu --gres=gpu:1 -A comsm0018 -t 0-02:00 --mem=6G --pty bash
