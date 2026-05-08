#!/bin/bash
#SBATCH --job-name=fedavg_olmoe
#SBATCH --partition=a01
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/fedavg_%j.log

source ~/WORK/miniconda3/etc/profile.d/conda.sh
conda activate gfq11

cd /home/fit/qianxueh/WORK/alchem/gfq/SparseMoEFed

export SPARSEFEDMOE_MODEL=/home/fit/qianxueh/WORK/alchem/gfq/SparseFedMoE_baseline/model/OLMoE-1B-7B-0924
export SPARSEFEDMOE_DATA=/home/fit/qianxueh/WORK/alchem/gfq/SparseMoEFed/data
export SPARSEFEDMOE_DUMMY_MODEL=0
export SPARSEFEDMOE_LOCAL_EPOCHS=1
export SPARSEFEDMOE_BATCH_SIZE=4
export SPARSEFEDMOE_MAX_SEQ_LEN=256
export SPARSEFEDMOE_LR=2e-4
export SPARSEFEDMOE_EVAL_FRAC=0.1

bash scripts/run_simulator.sh jobs/fedavg_olmoe -n 1 -t 1 -w /home/fit/qianxueh/WORK/alchem/gfq/SparseMoEFed/workspace
