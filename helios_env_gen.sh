#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=envgen_res/job-%j.out
#SBATCH --error=envgen_res/job-%j.err

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

cd $SCRATCH

# create and activate the virtual environment 
python -m venv  .venv/
source .venv/bin/activate

# install one of torch versions available at Helios wheel repo
pip install --no-cache-dir torch==2.5.1+cu124.post3

# install the rest of requirements, for example via requirements file
pip install -e ~/CRL_experiments
pip install wandb

