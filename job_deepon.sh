#!/bin/bash
#SBATCH --partition=gpu_irmb
#SBATCH --nodes=1
#SBATCH --time=23:00:00
#SBATCH --job-name=pi_deepon
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:ampere
#SBATCH --output=/home/hessafar/codes/pi-power/zlogs/output_%j.log  # Standard output

# Run your singularity command
singularity exec --nv /home/hessafar/images/pideep.sif python -u main.py