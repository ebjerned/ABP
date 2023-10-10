#!/bin/bash -l

#SBATCH -A uppmax2023-2-36
#SBATCH -M snowy
#SBATCH -p node
#SBATCH -N 1
#SBATCH -t 1:00:00 
#SBATCH --gres=gpu:1
#SBATCH -J Execersie2
#SBATCH -D ./

export OMP_PROC_BIND=spread OMP_PLACES=threads
export OMP_NUM_THREADS=16
echo "FCL"
./exec/elemmatFCL.host -Nelem 1000 -Nmax 40000000 
echo "FCR"
./exec/elemmatFCR.host -Nelem 1000 -Nmax 40000000 
echo "DCL"
./exec/elemmatDCL.host -Nelem 1000 -Nmax 40000000 
echo "DCR"
./exec/elemmatDCR.host -Nelem 1000 -Nmax 40000000 
echo "FGL"
./exec/elemmatFGL.cuda -Nelem 1000 -Nmax 40000000 
echo "FGR"
./exec/elemmatFGR.cuda -Nelem 1000 -Nmax 40000000 
echo "DGL"
./exec/elemmatDGL.cuda -Nelem 1000 -Nmax 40000000 
echo "DGR"
./exec/elemmatDGR.cuda -Nelem 1000 -Nmax 40000000 

