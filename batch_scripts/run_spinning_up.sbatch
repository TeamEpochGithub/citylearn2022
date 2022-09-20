#!/bin/sh
#SBATCH --job-name="baseline"
#SBATCH --partition=compute
#SBATCH --time=20:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --mail-type=END
#SBATCH --account=Education-DDREAM-Epoch

module load 2022r2
module load python/3.8.12
module load miniconda3

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

### Comment out next lines after first run
#conda info --envs
#conda create -n mypython
###

conda activate mypython

### Comment out next few lines after first run
#conda install -c conda-forge gym
#conda install -c conda-forge stable-baselines3
#conda install -c conda-forge pip
#conda install -c conda-forge git
#conda install -c anaconda scipy
#conda install -c anaconda joblib
#conda install -c conda-forge tensorflow
#conda install -c conda-forge mpi4py
#conda install -c conda-forge psutil
#conda install -c conda-forge tqdm
#pip install git+https://github.com/intelligent-environments-lab/CityLearn.git@v1.3.6
#pip install requirements.txt
#conda list
###

python ../training/spinningup/ppo/ppo.py
conda deactivate