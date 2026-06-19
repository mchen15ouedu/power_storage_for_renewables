#!/bin/bash
# OSCER NLDAS-2 PER-PIXEL reduction, one SLURM array task per year (additive layer
# for the NREL-style optimal-siting analysis; the zonal pipeline is untouched):
#   1. pull this year's URLs from /scratch/mchen15/nldas/nldas_list.txt (strip CRLF)
#   2. wget them into /scratch/mchen15/nldas/raw/<YEAR>/
#   3. reduce to PER-PIXEL per-year solar/wind capacity factor
#      (scripts/nldas_pixel_reduce.py -> <data_dir>/pixel_cf/pixel_cf_<YEAR>.parquet)
#   4. delete the year's raw granules from scratch (only if download was complete)
#
# The zonal run deletes raw after reducing, so the per-pixel layer must re-download.
# Output is tiny (~80k cells x a few cols per year); the heavy cost is the download.
#
# PREREQUISITE: stage 0 already ran (regions + weights on the NLDAS grid) -- it has,
# since the zonal run used them (nldas_analysis/data/region_weights*.npz).
#
# Submit (throttle to ~4 concurrent so GES DISC + scratch stay happy):
#   sbatch --array=2000-2025%4 hpc/oscer_nldas_pixel.sbatch
#
# Restartable: wget -c resumes partial files; re-running a year just overwrites its
# pixel_cf parquet. Years whose download stays incomplete still write a parquet, but
# the siting script ignores any year with n_steps < ~8000 h.
#
#SBATCH --job-name=nldas_pixel
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/nldas_pixel_%a.out

set -euo pipefail
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

YEAR=${SLURM_ARRAY_TASK_ID}
REPO=${HOME}/gldas_storage
LIST=/scratch/mchen15/nldas/nldas_list.txt
RAW=/scratch/mchen15/nldas/raw/${YEAR}
mkdir -p "${RAW}"

# --- Earthdata Login bearer token (GES DISC token-based authorization) ------
TOKEN_FILE="${EARTHDATA_TOKEN_FILE:-$HOME/.edl_token}"
[ -f "${TOKEN_FILE}" ] || TOKEN_FILE="${REPO}/hpc/edl_token.txt"
if [ ! -f "${TOKEN_FILE}" ]; then
    echo "no Earthdata token found (~/.edl_token or hpc/edl_token.txt)" >&2
    exit 1
fi
TOKEN=$(tr -d '[:space:]' < "${TOKEN_FILE}")

# --- 1+2: download this year's granules (1.14-safe multi-pass; -c resumes) ---
grep "NLDAS_FORA0125_H.2.0/${YEAR}/" "${LIST}" | grep '\.nc' \
    | tr -d '\r' > "${RAW}/list_${YEAR}.txt"
N_URLS=$(wc -l < "${RAW}/list_${YEAR}.txt")
echo "${YEAR}: ${N_URLS} granule URLs"

N_FILES=0
for attempt in 1 2 3 4; do
    wget --header "Authorization: Bearer ${TOKEN}" \
         --tries=3 --waitretry=15 --retry-connrefused \
         -c -nd -np -nv -P "${RAW}" -i "${RAW}/list_${YEAR}.txt" || true
    N_FILES=$(find "${RAW}" -name "*.nc" | wc -l)
    echo "${YEAR}: pass ${attempt} -> ${N_FILES}/${N_URLS} files on disk"
    if [ "${N_FILES}" -ge "${N_URLS}" ]; then break; fi
    sleep 30
done
if [ "${N_FILES}" -lt "${N_URLS}" ]; then
    echo "${YEAR}: download incomplete -- this year may be dropped by the siting step" >&2
fi

# --- 3: per-pixel reduction -------------------------------------------------
python "${REPO}/scripts/nldas_pixel_reduce.py" \
    --config "${REPO}/hpc/config_nldas.yaml" --year "${YEAR}"

# --- 4: free the scratch space (only if the download was complete) ----------
if [ "${N_FILES}" -ge "${N_URLS}" ]; then
    rm -rf "${RAW}"
    echo "${YEAR}: raw granules deleted"
else
    echo "${YEAR}: keeping raw folder for inspection (incomplete download)" >&2
fi
