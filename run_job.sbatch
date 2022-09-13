#!/bin/sh
#SBATCH --job-name="baseline"
#SBATCH --partition=compute
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-type=END

module load 2022r2

python3 -m venv venv/
source venv/bin/activate
pip install gym
pip install stable-baselines3[extra]

python evaluating_baselines3.py