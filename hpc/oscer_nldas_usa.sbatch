#!/bin/bash
# NLDAS USA downstream analysis (REAL UNITS, TWh + NREL 1% land constraint):
#   1. usa_eia_feasibility.py  -> usa_summary.csv (real-units + land cols),
#                                 usa_cannot_meet_demand.csv, usa_storage_ranking.csv,
#                                 map_usa_feasibility.png (feasible within 1% land, TWh)
#   2. usa_eia_deficit_plots.py-> deficit_{solar,wind,mix}_USA_states_{supply,storage,
#                                 normalized}.pdf (TWh; normalized = / monthly EIA),
#                                 usa_months_unmet.csv (real TWh shortfalls)
#   3. colocation_maps.py --land-feasibility -> map_colocation_feasibility.png
#                                 (feasible within 1% land vs co-location k)
# Wind counted at DIRECT footprint (spacing reused). Feasibility = meets demand
# within the land cap; two caps reported: 1% (NREL) and 1.88% (2x onshore oil &
# gas). Outputs in nldas_analysis/{results,figures}/.
# CONUS-only: Alaska & Hawaii excluded (no NLDAS coverage).
#
#   sbatch hpc/oscer_nldas_usa.sbatch
#
#SBATCH --job-name=nldas_usa
#SBATCH --partition=caps_rcm,normal
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

# ENV GUARD: relocate off a node where geo2 failed to load (would crash analysis)
NODE=$(hostname -s); BADFILE="${REPO}/hpc/bad_nodes.txt"
if ! python -c "import geopandas, pyarrow, xarray" 2>/dev/null; then
    echo "nldas_usa: ENV BROKEN on ${NODE} -- relocating" >&2
    printf '%s\t%s\tjob=%s\tnldas_usa env failed\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${SLURM_JOB_ID:-?}" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    sbatch --exclude="${EXCL}" "${REPO}/hpc/oscer_nldas_usa.sbatch" \
        || sbatch "${REPO}/hpc/oscer_nldas_usa.sbatch"
    exit 0
fi

# guard: refuse to run if no zonal parquet exists yet
N_PARQUET=$(find /ourdisk/hpc/caps/mchen15/nldas_full -name 'zonal_*.parquet' 2>/dev/null | wc -l)
echo "NLDAS zonal parquet files present: ${N_PARQUET}"
if [ "${N_PARQUET}" -eq 0 ]; then
    echo "no NLDAS zonal parquet -- stage 1 has not produced anything yet" >&2
    exit 1
fi

# readiness sentinel: the real-units (TWh + 1% land) USA pipeline rewrite is in
# progress. Until the sentinel exists, no-op cleanly so QCagent's auto-trigger
# does NOT run the stale fractional pipeline. Created when the rewrite is done.
READY=${REPO}/hpc/PIPELINE_READY_REALUNITS
if [ ! -f "${READY}" ]; then
    echo "pipeline-ready sentinel ${READY} absent -- real-units rewrite not finished; skipping" >&2
    exit 0
fi

python "${REPO}/scripts/usa_eia_feasibility.py"  --config "${CFG}"
python "${REPO}/scripts/usa_eia_deficit_plots.py" --config "${CFG}"

# co-location feasibility maps (k = 0/17/35/50 %), USA / CONUS, EIA monthly demand
MONTHLY=/ourdisk/hpc/caps/mchen15/nldas_analysis/results/usa_eia_monthly_demand.csv
python "${REPO}/scripts/colocation_maps.py" --config "${CFG}" \
    --summary usa_summary.csv --monthly-csv "${MONTHLY}" --conus --land-feasibility

echo "NLDAS USA analysis complete"
