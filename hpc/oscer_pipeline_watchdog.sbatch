#!/bin/bash
# Pipeline watchdog (2026-06-19) -- a real Python monitoring PROGRAM, not an idle
# `claude` poll-loop. It watches the NLDAS download/reduction pipeline + the QCagent
# and DETERMINISTICALLY relaunches/unsticks them (the QCagent once "gave up" and
# looped; this is the layer above it). It escalates to `claude` ONLY on a novel
# anomaly, and only if WATCHDOG_CLAUDE=1 -- so the default job is a plain python
# process the scheduler/safety review is fine with, and I can launch it.
#
#   sbatch hpc/oscer_pipeline_watchdog.sbatch
#   # opt-in Claude escalation on novel anomalies (autonomous; your call):
#   sbatch --export=ALL,WATCHDOG_CLAUDE=1 hpc/oscer_pipeline_watchdog.sbatch
#
# caps_rcm has no time limit -- keep the watchdog alive for the whole campaign.
# Stop with `scancel` on the pipeline_watchdog job.
#
#SBATCH --job-name=pipeline_watchdog
#SBATCH --partition=caps_rcm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/pipeline_watchdog.out

set -uo pipefail
cd "${HOME}/gldas_storage"
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u
export PATH="${HOME}/.local/bin:${PATH}"        # for the opt-in claude escalation

REPO="${HOME}/gldas_storage"
SELF="${REPO}/hpc/oscer_pipeline_watchdog.sbatch"
NODE=$(hostname -s); BADFILE="${REPO}/hpc/bad_nodes.txt"

# ENV GUARD: relocate off a node where the env didn't load (record it for OSCER).
if ! python -c "import yaml" 2>/dev/null; then
    echo "watchdog: ENV BROKEN on ${NODE} -- relocating" >&2
    printf '%s\t%s\tjob=%s\twatchdog env failed\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${SLURM_JOB_ID:-?}" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    sbatch --exclude="${EXCL}" "${SELF}" || sbatch "${SELF}"
    exit 0
fi

# The job IS this continuous python program (deterministic checks every 5 min).
exec python "${REPO}/scripts/pipeline_watchdog.py" \
    --config "${REPO}/hpc/config_nldas.yaml" --interval 300 --stale-min 35
