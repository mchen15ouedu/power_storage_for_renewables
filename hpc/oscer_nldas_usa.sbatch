#!/bin/bash
# NLDAS USA downstream analysis: once all years are reduced to hourly zonal
# parquet, run the same USA/EIA workflow as the GLDAS run, but on NLDAS:
#   1. usa_eia_feasibility.py  -> usa_summary.csv, usa_storage_ranking.csv,
#                                 map_usa_feasibility.png  (% of annual consumption)
#   2. usa_eia_deficit_plots.py-> deficit_{solar,wind,mix}_USA_states_*.pdf,
#                                 usa_months_unmet.csv
# Outputs land in /ourdisk/hpc/caps/mchen15/nldas_analysis/{results,figures}/.
# CONUS-only: Alaska & Hawaii are excluded (no NLDAS coverage).
#
#   sbatch hpc/oscer_nldas_usa.sbatch
#
#SBATCH --job-name=nldas_usa
#SBATCH --partition=normal
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=logs/nldas_usa.out

set -euo pipefail
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

REPO=${HOME}/gldas_storage
CFG="${REPO}/hpc/config_nldas.yaml"

# guard: refuse to run if no zonal parquet exists yet
N_PARQUET=$(find /ourdisk/hpc/caps/mchen15/nldas_full -name 'zonal_*.parquet' 2>/dev/null | wc -l)
echo "NLDAS zonal parquet files present: ${N_PARQUET}"
if [ "${N_PARQUET}" -eq 0 ]; then
    echo "no NLDAS zonal parquet -- stage 1 has not produced anything yet" >&2
    exit 1
fi

python "${REPO}/scripts/usa_eia_feasibility.py"  --config "${CFG}"
python "${REPO}/scripts/usa_eia_deficit_plots.py" --config "${CFG}"
echo "NLDAS USA analysis complete"
