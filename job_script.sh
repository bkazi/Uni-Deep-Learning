#!/bin/bash
#SBATCH -t 0-01:00 # Runtime in D-HH:MM
#SBATCH -p gpu # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --account=comsm0018       # use the course account
#SBATCH -J adl    # name
#SBATCH -o hostname_%j.out # File to which STDOUT will be written
#SBATCH -e hostname_%j.err # File to which STDERR will be written

module load languages/anaconda2/5.0.1
module load libs/cudnn
pip install --user librosa tensorflow-gpu
rm -r logs/

srun python main.py --epochs 100 --network 0
# srun python main.py --epochs 200 --network 0
srun python main.py --epochs 100 --network 1
# srun python main.py --epochs 200 --network 1
