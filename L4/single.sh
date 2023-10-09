#!/bin/bash -l

#SBATCH -A uppmax2023-2-36
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 0:10:00 --qos=short
#SBATCH --gres=gpu:1
#SBATCH -J Execersie2
#SBATCH -D ./

export OMP_PROC_BIND=spread OMP_PLACES=threads
#export OMP_NUM_THREADS=2
#./elemmat.host -Nelem 1000 -Nmax 400000000 
export OMP_NUM_THREADS=16
./elemmat.host -Nelem 1000 -Nmax 400000000 
