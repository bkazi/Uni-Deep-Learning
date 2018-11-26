#!/bin/bash
module add libs/tensorflow/1.2
pip install --user librosa
srun -p gpu --gres=gpu:1 -A comsm0018 --reservation=comsm0018-lab7  -t 0-02:00 --mem=4G --pty bash
