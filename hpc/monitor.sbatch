#!/bin/bash
# Lightweight watcher for the GLDAS stage-1 rerun + stage-2 resubmit (2026-06-15).
# Runs as its own SLURM job so monitoring outlives any interactive session.
# Polls every 5 min, appends progress, and on stage-2 completion writes a FINAL
# summary + a DONE marker file. Submit from ~/gldas_storage:
#   sbatch hpc/monitor.sbatch
#
#SBATCH --job-name=gldas_monitor
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=logs/monitor_progress.log

ARR=32268259          # stage-1 rerun array (9 failed years)
ST2=32268260          # stage-2 analysis (afterany:ARR)
ZD=/ourdisk/hpc/caps/mchen15/gldas_full
DONE=${HOME}/gldas_storage/logs/MONITOR_DONE.txt
RERUN="2001 2004 2005 2008 2009 2010 2015 2017 2019"
ALL_YEARS=$(seq 2000 2025)

stamp(){ date '+%Y-%m-%d %H:%M:%S'; }
pq_count(){ find "$ZD" -name "zonal_${1:-}*.parquet" 2>/dev/null | wc -l; }

echo "$(stamp) monitor start: watching rerun array $ARR and stage-2 $ST2 (poll 5 min)"

while true; do
    NP=$(pq_count)
    QARR=$(squeue -j "$ARR" -h -t all 2>/dev/null | wc -l)
    miss=""
    for y in $RERUN; do
        c=$(pq_count "$y"); [ "$c" -lt 12 ] && miss="$miss ${y}($c/12)"
    done
    s2=$(sacct -j "$ST2" -n -X --format=State 2>/dev/null | head -1 | tr -d ' ')
    echo "$(stamp) parquet=$NP/312 | rerun tasks queued/running=$QARR | stage2=$s2 | incomplete reruns:${miss:- none}"

    case "$s2" in
        COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|DEADLINE)
            echo "$(stamp) stage-2 ($ST2) reached terminal state: $s2"
            break;;
    esac
    sleep 300
done

# ---- FINAL SUMMARY -----------------------------------------------------------
{
    echo "===== GLDAS pipeline FINAL summary  $(stamp) ====="
    NP=$(pq_count)
    echo "parquet total: ${NP}/312 (26 years x 12 months)"
    echo "--- per-year parquet (flag any != 12) ---"
    for y in $ALL_YEARS; do
        c=$(pq_count "$y"); flag=""; [ "$c" -ne 12 ] && flag="  <-- INCOMPLETE"
        echo "  ${y}: ${c}/12${flag}"
    done
    echo "--- stage-1 rerun ($ARR) task states ---"
    sacct -j "$ARR" -X -n --format=JobID%20,State,ExitCode 2>/dev/null
    echo "--- stage-2 ($ST2) ---"
    sacct -j "$ST2" -X -n --format=JobID%20,State,ExitCode,Elapsed 2>/dev/null
    echo "--- tail of analysis log ---"
    tail -n 20 "${HOME}/gldas_storage/logs/analysis.out" 2>/dev/null || echo "(no analysis.out)"
    echo "===== end ====="
} | tee "$DONE"

echo "$(stamp) monitor done; summary written to $DONE"
