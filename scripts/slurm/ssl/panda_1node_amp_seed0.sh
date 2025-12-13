#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --nodes=2
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
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
if [ $SLURM_ARRAY_TASK_ID -eq 1 ]; then
    CONFIG="pretrain-sonata-v1m1-pilarnet-100-amp-seed0-smallmask"
    RUN_NAME="pretrain-sonata-pilarnet-100-amp-4GPU"
elif [ $SLURM_ARRAY_TASK_ID -eq 2 ]; then
    CONFIG="pretrain-sonata-v1m1-pilarnet-1k-amp-seed0-smallmask"
    RUN_NAME="pretrain-sonata-pilarnet-1k-amp-4GPU"
elif [ $SLURM_ARRAY_TASK_ID -eq 3 ]; then
    CONFIG="pretrain-sonata-v1m1-pilarnet-10k-amp-seed0-smallmask"
    RUN_NAME="pretrain-sonata-pilarnet-10k-amp-4GPU"
elif [ $SLURM_ARRAY_TASK_ID -eq 4 ]; then
    CONFIG="pretrain-sonata-v1m1-pilarnet-100k-amp-seed0-smallmask"
    RUN_NAME="pretrain-sonata-pilarnet-100k-amp-4GPU"
elif [ $SLURM_ARRAY_TASK_ID -eq 5 ]; then
    CONFIG="pretrain-sonata-v1m1-pilarnet-1m-amp-seed0-smallmask"
    RUN_NAME="pretrain-sonata-pilarnet-1m-amp-4GPU"
fi

TRAIN_PATH=/sdf/home/y/youngsam/sw/dune/representations/lar.fm/scripts/train.sh
COMMAND="sh ${TRAIN_PATH} -m 2 -g 4 -d panda/pretrain -c ${CONFIG} -n ${RUN_NAME}-${CURRENT_DATETIME}-seed0"

srun singularity run --nv -B /sdf,/fs,/sdf/scratch,/lscratch ${SINGULARITY_IMAGE_PATH} \
    bash -c "source ~/.bashrc && mamba activate pointcept-torch2.5.0-cu12.4 && ${COMMAND} $1"