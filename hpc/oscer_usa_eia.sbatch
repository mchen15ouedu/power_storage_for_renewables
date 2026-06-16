#!/bin/bash
# USA-only storage/feasibility analysis driven by real EIA monthly consumption.
# Reuses the existing GLDAS zonal parquet; analyzes the 51 US states with a
# per-state monthly demand profile, then writes usa_summary.csv + the USA
# feasibility figure.
#
#   sbatch hpc/oscer_usa_eia.sbatch
#
#SBATCH --job-name=gldas_usa_eia
#SBATCH --partition=normal
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/usa_eia.out

set -euo pipefail
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

REPO=${HOME}/gldas_storage
CFG="${REPO}/hpc/config_oscer.yaml"

python "${REPO}/scripts/usa_eia_feasibility.py" --config "${CFG}"
