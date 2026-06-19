#!/bin/bash
# OSCER NLDAS-2 stage 1, one SLURM array task per year:
#   1. pull this year's URLs out of /scratch/mchen15/nldas/nldas_list.txt
#      (the list has Windows CRLF endings -- the trailing \r MUST be stripped or
#       wget appends %0D to every URL and gets an HTML error page)
#   2. wget them into /scratch/mchen15/nldas/raw/<YEAR>/
#   3. reduce to HOURLY zonal means (monthly parquet) on /ourdisk
#   4. ALSO reduce to per-pixel per-year CF (optimal-siting layer)
#   5. KEEP the raw granules on /scratch (it auto-purges after 14 days; we reuse
#      the raw within that window for the per-pixel layer and other analyses --
#      NEVER rm the raw, that is what deleted 24 years' data before).
#
# ENV GUARD: some nodes silently fail `conda activate geo2` and run base python
# (no geopandas) -> the reduction crashes AFTER a full download. We detect that,
# log the node to hpc/bad_nodes.txt (report to OSCER), and resubmit excluding it.
#
# NLDAS_FORA0125_H v2.0: hourly, 0.125 deg, CONUS only. ~8760 granules/year.
#
# PREREQUISITE: stage 0 must have been run once (regions + weights) with a sample
# granule on the NLDAS grid:
#   python scripts/00_regions.py --config hpc/config_nldas.yaml --sample <one.nc>
#
# Submit (throttle to ~4 concurrent so GES DISC and scratch stay happy):
#   sbatch --array=2000-2025%4 hpc/oscer_nldas_zonal.sbatch
#
# Restartable: wget -c resumes partial files; reduced months are skipped.
#
#SBATCH --job-name=nldas_zonal
#SBATCH --partition=caps_rcm,normal
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/nldas_zonal_%a.out

set -euo pipefail
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

YEAR=${SLURM_ARRAY_TASK_ID}
REPO=${HOME}/gldas_storage
SELF="${REPO}/hpc/oscer_nldas_zonal.sbatch"     # $0 is the spool copy; hardcode for resubmit
LIST=/scratch/mchen15/nldas/nldas_list.txt
RAW=/scratch/mchen15/nldas/raw/${YEAR}
mkdir -p "${RAW}"

# --- ENV GUARD (RETRY): verify geo2 actually loaded on THIS node. The first import
#     after `conda activate` on a cold node can fail TRANSIENTLY (NFS/Lustre lag,
#     activation race) -- a no-retry guard once blacklisted 35 good nodes in seconds.
#     So retry with backoff and only relocate after SUSTAINED failure (a real bad node),
#     logging the actual error for the OSCER report. ---
NODE=$(hostname -s)
BADFILE="${REPO}/hpc/bad_nodes.txt"
ENV_OK=0; ENV_ERR=""
for t in 1 2 3 4 5; do
    conda activate geo2 2>/dev/null || true
    if ENV_ERR=$(python -c "import geopandas, pyarrow, xarray" 2>&1); then ENV_OK=1; break; fi
    echo "${YEAR}: env check ${t}/5 on ${NODE} failed: ${ENV_ERR}" >&2
    sleep $((t * 5))
done
if [ "${ENV_OK}" -ne 1 ]; then
    echo "${YEAR}: ENV BROKEN on ${NODE} after 5 tries -- relocating" >&2
    printf '%s\t%s\tyear=%s\tjob=%s\t%s\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${YEAR}" "${SLURM_JOB_ID:-?}" "${ENV_ERR//$'\n'/ }" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    echo "${YEAR}: resubmitting --exclude=${EXCL}" >&2
    sbatch --array="${YEAR}" --exclude="${EXCL}" "${SELF}" \
        || sbatch --array="${YEAR}" "${SELF}"
    exit 0
fi
echo "${YEAR}: env OK on ${NODE} (try ${t})"

# --- Earthdata Login bearer token (GES DISC token-based authorization) ------
TOKEN_FILE="${EARTHDATA_TOKEN_FILE:-$HOME/.edl_token}"
[ -f "${TOKEN_FILE}" ] || TOKEN_FILE="${REPO}/hpc/edl_token.txt"
if [ ! -f "${TOKEN_FILE}" ]; then
    echo "no Earthdata token found (~/.edl_token or hpc/edl_token.txt)" >&2
    exit 1
fi
TOKEN=$(tr -d '[:space:]' < "${TOKEN_FILE}")

# --- 1+2: download this year's granules ------------------------------------
# NLDAS URLs carry the year in the path: .../NLDAS_FORA0125_H.2.0/<YEAR>/<DOY>/...
# tr -d '\r' strips the CRLF carriage returns from the user-provided list.
grep "NLDAS_FORA0125_H.2.0/${YEAR}/" "${LIST}" | grep '\.nc' \
    | tr -d '\r' > "${RAW}/list_${YEAR}.txt"
N_URLS=$(wc -l < "${RAW}/list_${YEAR}.txt")
echo "${YEAR}: ${N_URLS} granule URLs"

# The system wget on the compute nodes is 1.14, which has NO --retry-on-http-error
# (that flag, and an unknown-option error, would make wget print usage and fetch
# nothing). Instead make several full passes with only 1.14-safe flags: -c
# resumes partial files and skips already-complete ones, so a transient GES DISC
# 5xx on one pass is picked up by the next. set -e is tolerated via "|| true".
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
    echo "${YEAR}: download incomplete -- reduction may warn about short months" >&2
fi

# --- 3: reduce to hourly zonal means ----------------------------------------
python "${REPO}/scripts/01_zonal.py" \
    --config "${REPO}/hpc/config_nldas.yaml" \
    --start "${YEAR}-01-01" --end "${YEAR}-12-31"

# --- 4: per-pixel per-year CF (optimal-siting layer), from the same download --
python "${REPO}/scripts/nldas_pixel_reduce.py" \
    --config "${REPO}/hpc/config_nldas.yaml" --year "${YEAR}"

# --- 5: KEEP the raw on /scratch (auto-purges in 14 days) --------------------
# Never `rm -rf` the raw: it is reused for the per-pixel layer and other analyses
# within the 14-day scratch window. The /scratch policy reclaims the space itself.
echo "${YEAR}: done (zonal + pixel); raw kept on scratch (auto-purge in ~14 days)"
