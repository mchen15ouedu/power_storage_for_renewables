#!/bin/bash
# OSCER stage 1, one SLURM array task per year:
#   1. pull this year's URLs out of /scratch/mchen15/gldas/gldas_list.txt
#   2. wget them into /scratch/mchen15/gldas/raw/<YEAR>/
#   3. reduce to 3-hourly zonal means (monthly parquet) on /ourdisk
#   4. delete the year's raw granules from scratch
#
# PREREQUISITE: stage 0 must have been run once (regions + weights):
#   python scripts/00_regions.py --config hpc/config_oscer.yaml
# (needs one sample granule in /scratch/mchen15/gldas/raw -- wget the first
#  line of the list file; login node is fine)
#
# Submit (throttle to ~4 concurrent so GES DISC and scratch stay happy):
#   sbatch --array=2000-2025%4 hpc/oscer_01_zonal.sbatch
#
# ONE-TIME SETUP (login node), per the GES DISC data-access instructions
# (token-based authorization -- create the token at urs.earthdata.nasa.gov,
#  "Generate Token"; tokens live ~60 days):
#   install -m 600 hpc/edl_token.txt ~/.edl_token   # or paste a fresh token there
# Environment: the existing ~/.conda/envs/geo2 env is used (pyarrow + pyyaml
# were pip-installed into it); no new env needed.
#
# Restartable: wget -c resumes partial files; reduced months are skipped.
#
#SBATCH --job-name=gldas_zonal
#SBATCH --partition=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --output=logs/zonal_%a.out

set -euo pipefail
mkdir -p logs

# OSCER has no Mamba module; use the system Anaconda3 and the existing geo2 env.
# geo2's gdal-activate.sh reads unset vars, so suspend nounset around activation.
module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

YEAR=${SLURM_ARRAY_TASK_ID}
# $0 points at the SLURM spool copy, not the repo -- hardcode the repo path
REPO=${HOME}/gldas_storage
LIST=/scratch/mchen15/gldas/gldas_list.txt
RAW=/scratch/mchen15/gldas/raw/${YEAR}
mkdir -p "${RAW}"

# --- Earthdata Login bearer token (GES DISC token-based authorization) ------
TOKEN_FILE="${EARTHDATA_TOKEN_FILE:-$HOME/.edl_token}"
[ -f "${TOKEN_FILE}" ] || TOKEN_FILE="${REPO}/hpc/edl_token.txt"
if [ ! -f "${TOKEN_FILE}" ]; then
    echo "no Earthdata token found (~/.edl_token or hpc/edl_token.txt)" >&2
    echo "generate one at urs.earthdata.nasa.gov and save it there" >&2
    exit 1
fi
TOKEN=$(tr -d '[:space:]' < "${TOKEN_FILE}")   # strip newlines/CR from Windows edits

# --- 1+2: download this year's granules ------------------------------------
# GLDAS URLs carry the year in the path: .../GLDAS_NOAH025_3H.2.1/<YEAR>/<DOY>/...
grep "GLDAS_NOAH025_3H.2.1/${YEAR}/" "${LIST}" | grep "\.nc4" > "${RAW}/list_${YEAR}.txt"
N_URLS=$(wc -l < "${RAW}/list_${YEAR}.txt")
echo "${YEAR}: ${N_URLS} granule URLs"

# Don't let a transient GES DISC 5xx on a handful of granules abort the whole
# year (set -e + wget exit 8). Retry on server errors, then fall through to the
# N_FILES check below, which decides whether the reduction can proceed.
wget --header "Authorization: Bearer ${TOKEN}" \
     --tries=5 --retry-on-http-error=429,500,502,503,504 --waitretry=15 \
     -c -nd -np -nv -P "${RAW}" -i "${RAW}/list_${YEAR}.txt" || true

N_FILES=$(find "${RAW}" -name "*.nc4" | wc -l)
echo "${YEAR}: ${N_FILES}/${N_URLS} files on disk"
if [ "${N_FILES}" -lt "${N_URLS}" ]; then
    echo "${YEAR}: download incomplete -- reduction may warn about short months" >&2
fi

# --- 3: reduce to 3-hourly zonal means --------------------------------------
python "${REPO}/scripts/01_zonal.py" \
    --config "${REPO}/hpc/config_oscer.yaml" \
    --start "${YEAR}-01-01" --end "${YEAR}-12-31"

# --- 4: free the scratch space ----------------------------------------------
if [ "${N_FILES}" -ge "${N_URLS}" ]; then
    rm -rf "${RAW}"
    echo "${YEAR}: raw granules deleted"
else
    echo "${YEAR}: keeping raw folder for inspection (incomplete download)" >&2
fi
