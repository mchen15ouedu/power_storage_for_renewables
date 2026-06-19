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

# ENV GUARD: if geo2 didn't load on this node, the QCagent itself would crash.
# Record the node and relaunch the QCagent excluding all bad nodes seen.
NODE=$(hostname -s); BADFILE="${REPO}/hpc/bad_nodes.txt"
if ! python -c "import geopandas, pyarrow, xarray" 2>/dev/null; then
    echo "QCagent: ENV BROKEN on ${NODE} -- relocating" >&2
    printf '%s\t%s\tjob=%s\tQCagent env failed\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${SLURM_JOB_ID:-?}" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    sbatch --exclude="${EXCL}" "${SELF}" || sbatch "${SELF}"
    exit 0
fi

# run unbounded on caps_rcm (no wall-clock cap) so it babysits the whole campaign
python "${REPO}/scripts/qc_agent.py" \
    --config "${REPO}/hpc/config_nldas.yaml" \
    --sbatch "${REPO}/hpc/oscer_nldas_zonal.sbatch" \
    --interval 600 --stall-min 90 --max-hours 100000
