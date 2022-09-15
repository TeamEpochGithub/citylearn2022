#!/bin/sh
#SBATCH --job-name="baseline"
#SBATCH --partition=compute
#SBATCH --time=20:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-type=END

module load 2022r2
module load python/3.8.12
module load py-pip
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"
conda info --envs

# Comment out next line after first run
conda create -n mypython
conda info --envs

conda activate mypython

# Comment out next few lines after first run
conda install -c conda-forge gym
conda update gym
conda install -c conda-forge stable-baselines3
conda install -c conda-forge pip
conda install -c conda-forge git
pip install git+https://github.com/intelligent-environments-lab/CityLearn.git@citylearn_2022

conda list
python evaluating_baselines3.py
conda deactivate