#!/bin/bash
# Autonomous Claude watchdog for the NLDAS pipeline (2026-06-19).
# Runs on caps_rcm (no time limit) and, every ~20 min, invokes `claude -p` with
# hpc/claude_monitor_prompt.txt to check the QCagent's heartbeat + the SLURM queue
# and relaunch/unstick the pipeline if the QCagent dies or hangs. This is the
# higher-level watchdog the QCagent itself lacked (it once "gave up" and looped).
#
#   sbatch hpc/oscer_claude_monitor.sbatch
#
# Needs: claude CLI at ~/.local/bin (authenticated), squeue/sbatch/scancel.
# Conservative by design -- the prompt caps it to a few actions per run and forbids
# deleting any data. Stop it with `scancel` on the claude_monitor job.
#
#SBATCH --job-name=claude_monitor
#SBATCH --partition=caps_rcm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=logs/claude_monitor.out

set -uo pipefail
cd "${HOME}/gldas_storage"
mkdir -p logs

module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u
export PATH="${HOME}/.local/bin:${PATH}"        # claude CLI lives here

REPO="${HOME}/gldas_storage"
SELF="${REPO}/hpc/oscer_claude_monitor.sbatch"
NODE=$(hostname -s)
BADFILE="${REPO}/hpc/bad_nodes.txt"

# ENV GUARD: if geo2 didn't load or claude is missing on this node, record + move.
if ! command -v claude >/dev/null || ! python -c "import geopandas" 2>/dev/null; then
    echo "claude_monitor: ENV BROKEN on ${NODE} (claude or geo2 missing) -- relocating" >&2
    printf '%s\t%s\tjob=%s\tclaude_monitor env failed\n' \
        "$(date -u +%FT%TZ)" "${NODE}" "${SLURM_JOB_ID:-?}" >> "${BADFILE}"
    EXCL=$(awk -F'\t' 'NF>1{print $2}' "${BADFILE}" | sort -u | paste -sd, -)
    sbatch --exclude="${EXCL}" "${SELF}" || sbatch "${SELF}"
    exit 0
fi

INTERVAL="${INTERVAL:-1200}"                     # seconds between checks (20 min)
PROMPT_FILE="${REPO}/hpc/claude_monitor_prompt.txt"
echo "$(date -u +%FT%TZ) claude_monitor up on ${NODE}, interval ${INTERVAL}s" >> logs/claude_monitor.log

while true; do
    timeout 900 claude -p "$(cat "${PROMPT_FILE}")" \
        --dangerously-skip-permissions \
        >> logs/claude_monitor.log 2>&1 \
        || echo "$(date -u +%FT%TZ) claude invocation failed/timed out" >> logs/claude_monitor.log
    sleep "${INTERVAL}"
done
