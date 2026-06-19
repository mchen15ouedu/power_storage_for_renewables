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
conda activate geo2 || true     # transient activation failures handled by the retry guard
export PATH="${HOME}/.local/bin:${PATH}"        # for the opt-in claude escalation

REPO="${HOME}/gldas_storage"
SELF="${REPO}/hpc/oscer_pipeline_watchdog.sbatch"
NODE=$(hostname -s); BADFILE="${REPO}/hpc/bad_nodes.txt"

# ENV GUARD (RETRY): retry before relocating (transient cold-start import failures).
ENV_OK=0; ENV_ERR=""
for t in 1 2 3 4 5 6 7 8; do
    conda activate geo2 2>/dev/null || true
    if ENV_ERR=$(python -c "import yaml, pandas" 2>&1); then ENV_OK=1; break; fi
    echo "watchdog: env check ${t}/8 on ${NODE} failed: ${ENV_ERR}" >&2
    sleep $((t * 5))
done
if [ "${ENV_OK}" -ne 1 ]; then
    echo "watchdog: ENV BROKEN on ${NODE} after 8 tries -- relocating" >&2
    printf '%s\t%s\tjob=%s\t%s\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${SLURM_JOB_ID:-?}" "${ENV_ERR//$'\n'/ }" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    sbatch --exclude="${EXCL}" "${SELF}" || sbatch "${SELF}"
    exit 0
fi
set -u          # env confirmed good

# The job IS this continuous python program (deterministic checks every 5 min).
# TARGET=zonal (default) watches the zonal download; TARGET=pixel watches the
# per-pixel re-download campaign (relaunches a matching-mode QCagent if it dies).
#   sbatch --export=ALL,TARGET=pixel hpc/oscer_pipeline_watchdog.sbatch
exec python "${REPO}/scripts/pipeline_watchdog.py" \
    --config "${REPO}/hpc/config_nldas.yaml" --target "${TARGET:-zonal}" \
    --interval 300 --stale-min 35
