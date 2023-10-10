module load gcc
module use /sw/EasyBuild/snowy/modules/all/
module load intelcuda/2019b
export OMPI_MCA_btl_openib_allow_ib=1
salloc -A uppmax2023-2-36 -N 1 -M snowy --gres=gpu:1 --gpus-per-node=1 -t 1:0:00



CUDNN_PATH=$(dirname $(python3 -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH


module load gcc/8.4.0
module use /sw/EasyBuild/snowy/modules/all/
module use /sw/EasyBuild/snowy-gpu/modules/all/
module use python_ML_packages/3.9.5-gpu
module load CUDA/11.7.0
