#!/bin/bash
# Submit the full GLDAS storage pipeline on OSCER.
#
#   ./hpc/submit_oscer.sh                # full run: years 2000-2025, 4 concurrent
#   ./hpc/submit_oscer.sh 2025 2025      # validation run: one year only
#   ./hpc/submit_oscer.sh 2000 2025 6    # custom concurrency throttle
#
# Stage 1 (one array task per year: wget granules -> zonal parquet -> clean
# scratch) runs first; stage 2 (analysis + figures) is submitted with a SLURM
# dependency and starts only after every array task succeeds.
#
# Prerequisites (all already done once on 2026-06-11, checked below anyway):
#   * /scratch/mchen15/gldas/gldas_list.txt  (wget list from GES DISC)
#   * ~/.edl_token                           (Earthdata token, expires ~2026-08-10)
#   * stage 0 outputs in /ourdisk/.../gldas_analysis/data/ (00_regions.py)
#   * conda env geo2 with pyarrow + pyyaml

set -euo pipefail

REPO=${HOME}/gldas_storage
FIRST=${1:-2000}
LAST=${2:-2025}
THROTTLE=${3:-4}

# --- prerequisite checks -----------------------------------------------------
fail() { echo "ERROR: $*" >&2; exit 1; }

[ -f /scratch/mchen15/gldas/gldas_list.txt ] \
    || fail "wget list missing: /scratch/mchen15/gldas/gldas_list.txt"
[ -f "${HOME}/.edl_token" ] || [ -f "${REPO}/hpc/edl_token.txt" ] \
    || fail "no Earthdata token (~/.edl_token); generate at urs.earthdata.nasa.gov"
[ -f /ourdisk/hpc/caps/mchen15/gldas_analysis/data/region_weights.npz ] \
    || fail "stage 0 not run: python scripts/00_regions.py --config hpc/config_oscer.yaml"
[ -x "${HOME}/.conda/envs/geo2/bin/python" ] \
    || fail "conda env geo2 not found"

mkdir -p "${REPO}/logs" /scratch/mchen15/gldas/raw \
         /ourdisk/hpc/caps/mchen15/gldas_full \
         /ourdisk/hpc/caps/mchen15/gldas_analysis/{data,results,figures}

# --- submit ------------------------------------------------------------------
cd "${REPO}"   # sbatch scripts write logs/ relative to the submit directory

JID=$(sbatch --parsable --array=${FIRST}-${LAST}%${THROTTLE} hpc/oscer_01_zonal.sbatch)
echo "stage 1 (zonal):    job ${JID}, years ${FIRST}-${LAST}, ${THROTTLE} concurrent"

JID2=$(sbatch --parsable --dependency=afterok:${JID} hpc/oscer_02_analysis.sbatch)
echo "stage 2 (analysis): job ${JID2}, runs after all of job ${JID} succeeds"

echo
echo "monitor:  squeue -u mchen15        logs: ${REPO}/logs/zonal_<year>.out"
echo "results:  /ourdisk/hpc/caps/mchen15/gldas_analysis/{results,figures}/"
