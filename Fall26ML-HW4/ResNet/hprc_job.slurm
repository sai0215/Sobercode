#!/bin/bash
# HW4 Problem 4 on TAMU HPRC: 3 models x 3 learning rates = 9 independent GPU jobs (Slurm job array).
# Submit from the ResNet/ folder:   mkdir -p logs && sbatch hprc_job.slurm
# Run a subset, e.g. only ResNet18:  sbatch --array=0-2 hprc_job.slurm
#
#   task id : 0               1              2              3                 4                5
#             resnet18/0.003  resnet18/0.01  resnet18/0.03  no_residual/0.003 no_residual/0.01 no_residual/0.03
#   task id : 6                  7                 8
#             no_batchnorm/0.003 no_batchnorm/0.01 no_batchnorm/0.03

#SBATCH --job-name=hw4-resnet
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --array=0-8
#SBATCH --output=logs/%x_%A_%a.out
##SBATCH --account=YOUR_ACCOUNT_NUMBER   # uncomment only if you have more than one HPRC allocation

# ---- EDIT ME: the same modules you loaded when you created the venv (see README step 2) ----
module purge
module load GCCcore/12.3.0 Python/3.11.3
VENV="$SCRATCH/hw4-venv"
# ---------------------------------------------------------------------------------------------

VARIANTS=(resnet18 no_residual no_batchnorm)
LRS=(0.003 0.01 0.03)
VARIANT=${VARIANTS[$((SLURM_ARRAY_TASK_ID / 3))]}
LR=${LRS[$((SLURM_ARRAY_TASK_ID % 3))]}

source "$VENV/bin/activate"
cd "$SLURM_SUBMIT_DIR"

echo "host=$(hostname)  variant=$VARIANT  lr=$LR  start=$(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

# Results go to runs_hprc/ so they never mix with the local (Apple-GPU) runs in runs/.
# tr turns the per-batch carriage-return progress bar into newlines; grep drops those lines.
# EPOCHS can be overridden for a quick smoke test:  EPOCHS=1 sbatch --array=0 hprc_job.slurm
# --resume: if this run was killed earlier (crash / time limit), just resubmit the same task id
# and it continues from the last 10-epoch checkpoint (runs_hprc/<run>/train_state.pt).
python -u main.py --variant "$VARIANT" --lr "$LR" --epochs "${EPOCHS:-100}" --resume \
  --modeldir "$PWD/runs_hprc/${VARIANT}_lr${LR}" 2>&1 \
  | stdbuf -oL tr '\r' '\n' | grep --line-buffered -v '^Batch'

echo "end=$(date)"
