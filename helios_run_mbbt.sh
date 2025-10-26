#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --account=plgcrlreason-gpu-gh200
#SBATCH --partition=plgrid-gpu-gh200
#SBATCH --output=res/job-%j.out
#SBATCH --error=res/job-%j.err

# IMPORTANT: load the modules for machine learning tasks and libraries
ml ML-bundle/24.06a

cd $SCRATCH

# activate the virtual environment 
source .venv/bin/activate

export WANDB_API_KEY=$(cat ~/.wandb_key)

# run the program
python ~/CRL_experiments/runner.py \
       	--config_file ~/CRL_experiments/configs/train/obbt/sokoban.gin \
       	--experiment_name sokoban_mbbt_2traj \
	--gin_bindings \
	"run.seed=$SLURM_JOB_ID" \
	"run.wandb=True" \
	"TrainJobOBBT.mbbt=True" \
	"TrainJobOBBT.max_traj=2" \
	"TrainJobOBBT.batch_size=512" \
	"TrainJobOBBT.test_interval=20000" \
	"TrainJobOBBT.train_steps=200000"

