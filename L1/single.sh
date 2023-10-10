#!/bin/bash -l
#SBATCH -A uppmax2023-2-36 # project name
#SBATCH -M snowy # name of system
#SBATCH -p node # request a full node
#SBATCH -N 1 # request 1 node
#SBATCH -t 15:00 # job takes at most 1 hour
#SBATCH -J stream_simd # name of the job
#SBATCH -D ./ # stay in current working directory


./stream_triadSIMD0ALIGN -min 8 -max 1e8 -align 1 
