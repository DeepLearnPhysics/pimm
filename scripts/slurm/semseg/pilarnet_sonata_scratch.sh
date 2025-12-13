#!/bin/bash
#SBATCH --job-name=scratch_data_scaling
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=6
#SBATCH --mem=192G
#SBATCH --time=96:00:00
#SBATCH --account=mli:nu-ml-dev
#SBATCH --partition=ampere
#SBATCH --output=slurm_logs/%j_%n_%x_%a.txt
#SBATCH --array=1
set -e

export PYTHONFAULTHANDLER=1

SINGULARITY_IMAGE_PATH=/sdf/group/neutrino/images/develop.sif
export NCCL_SOCKET_IFNAME=^docker0,lo  # Use any interface except docker and loopback

# Get current date and time in format YYYY-MM-DD_HH-MM
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt     

# define configs based on array task ID
if [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-scratch-100"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-scratch-1000"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-scratch-10000"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then   
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-scratch-100000"
elif [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    CONFIG="semseg-pt-v3m2-pilarnet-ft-5cls-scratch-1000000"
fi

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/lar.fm/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 1 -g 4 -d panda/semseg -c ${CONFIG} -n ${CONFIG}-${CURRENT_DATETIME}-seed0"

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"