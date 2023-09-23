#!/bin/bash -l
#SBATCH -A uppmax2023-2-36 # project name
#SBATCH -M snowy # name of system
#SBATCH -p node # request a full node
#SBATCH -N 1 # request 1 node
#SBATCH -t 30:00 # job takes at most 1 hour
#SBATCH --gres=gpu:1 --gpus-per-node=1
#SBATCH -J stream_simd # name of the job
#SBATCH -D ./ # stay in current working directory
./matmul -M 5000 -N 5000  
