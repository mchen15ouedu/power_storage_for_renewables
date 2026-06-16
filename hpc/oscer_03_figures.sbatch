#!/bin/bash
# OSCER stage 3 only: regenerate figures from the existing summary.csv + zonal
# parquet (no re-analysis). Use when only the plotting code changed.
#
#   sbatch hpc/oscer_03_figures.sbatch
#
#SBATCH --job-name=gldas_figures
#SBATCH --partition=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/figures.out

set -euo pipefail
mkdir -p logs

# OSCER has no Mamba module; use the system Anaconda3 and the existing geo2 env.
# geo2's gdal-activate.sh reads unset vars, so suspend nounset around activation.
module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

# $0 points at the SLURM spool copy, not the repo -- hardcode the repo path
REPO=${HOME}/gldas_storage
CFG="${REPO}/hpc/config_oscer.yaml"

python "${REPO}/scripts/03_figures.py" --config "${CFG}"
