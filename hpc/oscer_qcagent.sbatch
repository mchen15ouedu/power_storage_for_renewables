#!/bin/bash
# QCagent -- long-lived monitor for the NLDAS run. It watches the per-year
# download/reduction array, resubmits years whose job died or whose download
# stalled (server disruption), and once all data is in it runs the figure
# layout QC and writes results/qc_report.txt.
#
#   sbatch hpc/oscer_qcagent.sbatch
#
# Needs squeue/sbatch/scancel from the compute node (available on OSCER).
#
# caps_rcm has no time limit -- keep the babysitter alive for the whole run.
#SBATCH --job-name=QCagent
#SBATCH --partition=caps_rcm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/qcagent.out

set -euo pipefail
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

REPO=${HOME}/gldas_storage
SELF="${REPO}/hpc/oscer_qcagent.sbatch"

# ENV GUARD (RETRY): retry the import on a cold node before relocating (a no-retry
# guard once blacklisted 35 good nodes on transient cold-start import failures).
NODE=$(hostname -s); BADFILE="${REPO}/hpc/bad_nodes.txt"
ENV_OK=0; ENV_ERR=""
for t in 1 2 3 4 5; do
    conda activate geo2 2>/dev/null || true
    if ENV_ERR=$(python -c "import geopandas, pyarrow, xarray" 2>&1); then ENV_OK=1; break; fi
    echo "QCagent: env check ${t}/5 on ${NODE} failed: ${ENV_ERR}" >&2
    sleep $((t * 5))
done
if [ "${ENV_OK}" -ne 1 ]; then
    echo "QCagent: ENV BROKEN on ${NODE} after 5 tries -- relocating" >&2
    printf '%s\t%s\tjob=%s\t%s\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${SLURM_JOB_ID:-?}" "${ENV_ERR//$'\n'/ }" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    sbatch --exclude="${EXCL}" "${SELF}" || sbatch "${SELF}"
    exit 0
fi

# TARGET=zonal (default) babysits the zonal download; TARGET=pixel babysits the
# per-pixel re-download campaign (same sbatch, different completion criterion).
#   sbatch --export=ALL,TARGET=pixel hpc/oscer_qcagent.sbatch
# run unbounded on caps_rcm (no wall-clock cap) so it babysits the whole campaign
python "${REPO}/scripts/qc_agent.py" \
    --config "${REPO}/hpc/config_nldas.yaml" \
    --sbatch "${REPO}/hpc/oscer_nldas_zonal.sbatch" \
    --target "${TARGET:-zonal}" --min-steps "${MIN_STEPS:-8000}" \
    --interval 600 --stall-min 90 --max-hours 100000
