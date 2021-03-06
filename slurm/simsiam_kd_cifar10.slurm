#!/bin/bash

#SBATCH -J simsiam_kd_cifar
#SBATCH -o simsiam_kd_cifar%j.out
#SBATCH -e simsiam_kd_cifar%j.err
#SBATCH --mail-user=swhan@mit.edu
#SBATCH --mail-type=ALL
#SBATCH --gres=gpu:4
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --mem=1T
#SBATCH --time=24:00:00
#SBATCH --qos=sched_level_2
#SBATCH --exclusive

## User python environment
HOME2=/nobackup/users/$(whoami)
PYTHON_VIRTUAL_ENVIRONMENT=pytorch1.7
CONDA_ROOT=$HOME2/anaconda3

## Activate WMLCE virtual environment 
source ${CONDA_ROOT}/etc/profile.d/conda.sh
conda activate $PYTHON_VIRTUAL_ENVIRONMENT
ulimit -s unlimited

## Creating SLURM nodes list
export NODELIST=nodelist.$
srun -l bash -c 'hostname' |  sort -k 2 -u | awk -vORS=, '{print $2":4"}' | sed 's/,$//' > $NODELIST

## Number of total processes 
echo " "
echo " Nodelist:= " $SLURM_JOB_NODELIST
echo " Number of nodes:= " $SLURM_JOB_NUM_NODES
echo " NGPUs per node:= " $SLURM_GPUS_PER_NODE 
echo " Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

echo " Running on multiple nodes and GPU devices"
echo ""
echo " Run started at:- "
date

CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --data_dir ../data/ --log_dir ./logs/simsiam_kd_cifar/ -c configs/simsiam_kd_cifar.yaml --ckpt_dir ./experiments/simsiam_kd_cifar/

echo "Run completed at:- "
date
