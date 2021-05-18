#!/bin/bash

#SBATCH --partition=gpu_titanrtx_shared_course
#SBATCH --gres=gpu:1
#SBATCH --job-name=InferSent
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --time=04:00:00
#SBATCH --mem=32000M
#SBATCH --output=slurm_output_%A.out

module purge
module load 2020
module load Python/3.8.2-GCCcore-9.3.0
module load CUDA/11.0.2-GCC-9.3.0
module load cuDNN
module load Anaconda3/2020.02

# Your job starts in the directory where you call sbatch
cd $HOME/infersent
# # Activate your environment
source activate infersent
# # Run your code
srun python train.py --encoder_type='BiLSTM'