module load gcc/13.2.0
module use /sw/EasyBuild/snowy/modules/all/
module load intelcuda/2019b
export OMPI_MCA_btl_openib_allow_ib=1
salloc -A uppmax2023-2-36 -N 1 -M snowy --gres=gpu:1 --gpus-per-node=1 -t 1:0:00
