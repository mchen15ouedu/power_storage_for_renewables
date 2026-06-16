#!/bin/bash
# Coding-agent robustness harness as its own SLURM job (2026-06-16).
# Continuously byte-compiles, config-validates and runs the pytest suite for the
# gldas_storage codebase, re-checking whenever a source .py changes. Runs
# alongside the NLDAS pipeline + QCagent so code regressions surface early.
#
# Submit from ~/gldas_storage:
#   sbatch hpc/oscer_coding_agent.sbatch
# One-shot (no watch) for a quick gate:
#   sbatch --export=ALL,WATCH=0 hpc/oscer_coding_agent.sbatch
#
#SBATCH --job-name=coding_agent
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=logs/coding_agent.out

set -uo pipefail
cd "${HOME}/gldas_storage"
mkdir -p logs

# OSCER has no Mamba module; use the system Anaconda3 + existing geo2 env.
# geo2's gdal-activate.sh reads unset vars, so suspend nounset around activation.
module load Anaconda3/2022.05 2>/dev/null || true
set +u
source /opt/oscer/software/Anaconda3/2022.05/etc/profile.d/conda.sh
conda activate geo2
set -u

REPO="${HOME}/gldas_storage"
WATCH="${WATCH:-1}"          # 1 = keep watching for source changes, 0 = one-shot
INTERVAL="${INTERVAL:-600}"

if [ "${WATCH}" = "1" ]; then
    python "${REPO}/scripts/coding_agent.py" --watch --interval "${INTERVAL}"
else
    python "${REPO}/scripts/coding_agent.py"
fi
