#!/bin/bash
# OSCER stages 2-3: analysis (per region + pools) and figures.
# Run after all stage-1 array tasks have finished:
#
#   sbatch hpc/oscer_02_analysis.sbatch
#
#SBATCH --job-name=gldas_analysis
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/analysis.out

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

python "${REPO}/scripts/02_analyze.py" --config "${CFG}"
python "${REPO}/scripts/03_figures.py" --config "${CFG}"
