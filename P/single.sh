#!/bin/bash -l

#SBATCH -A uppmax2023-2-36
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 0:10:00 --qos=short
#SBATCH --gres=gpu:1
#SBATCH -J Execersie2
#SBATCH -D ./

./main
