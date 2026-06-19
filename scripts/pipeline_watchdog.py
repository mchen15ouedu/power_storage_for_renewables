"""Pipeline watchdog -- a real monitoring program (not an idle sleep-loop).

Watches the NLDAS download/reduction pipeline and the QCagent that babysits it, and
handles the clear-cut failure modes DETERMINISTICALLY (the QCagent itself once "gave
up" and looped forever; this is the layer above it):

  every cycle (cheap, real work -- squeue + parse heartbeat + count parquets):
    * QCagent gone from the queue while data is incomplete   -> relaunch it
    * QCagent heartbeat stale (json mtime/ts old = hung)      -> scancel + relaunch
    * a year the QCagent "gave up" on, not in the queue       -> resubmit that year
      (the hardened sbatch self-excludes bad nodes on its own)
    * all 26 years reduced but the downstream USA analysis    -> submit it once
      neither ran nor is queued and its figure is missing
    * new bad node recorded                                   -> log it for the OSCER report
    * GES DISC down                                           -> do nothing (don't thrash)

Only a TRULY novel anomaly (none of the above, yet something is wrong and not
progressing) is escalated -- it writes a prominent ALERT file, and, ONLY if
WATCHDOG_CLAUDE=1, asks `claude -p` for a one-shot diagnosis/fix. The Claude
escalation is opt-in precisely because an always-on autonomous agent is the thing
the scheduler/safety review (rightly) objects to.

Usage:
    python scripts/pipeline_watchdog.py --config hpc/config_nldas.yaml [--interval 300]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage.config import load_config                  # noqa: E402 (yaml-only, no geopandas)

log = logging.getLogger("watchdog")
REPO = Path(__file__).resolve().parents[1]
GESDISC = ("https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/"
           "NLDAS_FORA0125_H.2.0/2000/001/")


def sh(cmd: list[str], timeout: int = 120) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except Exception as exc:                                    # pragma: no cover
        log.warning("cmd failed %s: %s", " ".join(cmd), exc)
        return ""


def user() -> str:
    import getpass
    return getpass.getuser()


def jobs_by_name() -> dict[str, list[tuple[str, str]]]:
    """{jobname: [(jobid, state), ...]} for the current user's queue."""
    out = sh(["squeue", "-h", "-u", user(), "-o", "%A|%j|%T"])
    d: dict[str, list[tuple[str, str]]] = {}
    for line in out.splitlines():
        jid, _, rest = line.partition("|")
        name, _, state = rest.partition("|")
        d.setdefault(name.strip(), []).append((jid.strip(), state.strip()))
    return d


def server_up() -> bool:
    out = sh(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "--max-time", "30", "-I", GESDISC])
    return out.strip() in ("200", "301", "302", "401", "403")


def zonal_count(cfg) -> int:
    return len(list(Path(cfg["paths"]["zonal_dir"]).glob("zonal_*.parquet")))


def read_status(cfg, target="zonal") -> tuple[dict, float]:
    """QCagent heartbeat json + its age in seconds (inf if absent), per target."""
    name = "qcagent_status.json" if target == "zonal" else f"qcagent_{target}_status.json"
    f = Path(cfg["paths"]["results_dir"]) / name
    if not f.exists():
        return {}, float("inf")
    try:
        return json.loads(f.read_text()), time.time() - f.stat().st_mtime
    except Exception:
        return {}, float("inf")


def pixel_years_done(cfg, years, min_steps: int = 8000) -> int:
    """Count years whose per-pixel CF parquet exists with a near-full hour count."""
    import pandas as pd
    d = Path(cfg["paths"]["data_dir"]) / "pixel_cf"
    n = 0
    for y in years:
        f = d / f"pixel_cf_{y}.parquet"
        if f.exists():
            try:
                if int(pd.read_parquet(f, columns=["n_steps"])["n_steps"].max()) >= min_steps:
                    n += 1
            except Exception:
                pass
    return n


def bad_nodes() -> list[str]:
    f = REPO / "hpc" / "bad_nodes.txt"
    if not f.exists():
        return []
    return sorted({ln.split("\t")[1] for ln in f.read_text().splitlines()
                   if "\t" in ln and len(ln.split("\t")) > 1})


def submit(args_list: list[str], dry: bool) -> str:
    if dry:
        log.warning("[dry-run] sbatch %s", " ".join(args_list)); return "dry"
    out = sh(["sbatch", "--parsable", *args_list]).strip()
    log.warning("submitted: sbatch %s -> %s", " ".join(args_list), out or "?")
    return out


def cancel(jobid: str, dry: bool) -> None:
    if dry:
        log.warning("[dry-run] scancel %s", jobid); return
    sh(["scancel", jobid]); log.warning("scancelled %s", jobid); time.sleep(8)


def alert(cfg, msg: str) -> None:
    f = Path(cfg["paths"]["results_dir"]) / "watchdog_alert.txt"
    f.parent.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with f.open("a") as fh:
        fh.write(f"{stamp}  {msg}\n")
    log.error("ALERT: %s", msg)


def ask_claude(cfg, context: str) -> None:
    """Opt-in (WATCHDOG_CLAUDE=1) one-shot escalation for a novel anomaly."""
    if os.environ.get("WATCHDOG_CLAUDE") != "1":
        return
    prompt_file = REPO / "hpc" / "claude_monitor_prompt.txt"
    base = prompt_file.read_text() if prompt_file.exists() else ""
    prompt = f"{base}\n\nWATCHDOG ESCALATION -- a novel anomaly the deterministic rules " \
             f"did not resolve:\n{context}\nDiagnose and, if safe, fix it. Never delete data."
    log.warning("escalating to claude (WATCHDOG_CLAUDE=1)")
    sh(["claude", "-p", prompt, "--dangerously-skip-permissions"], timeout=900)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=REPO / "hpc" / "config_nldas.yaml")
    ap.add_argument("--zonal-sbatch", default=REPO / "hpc" / "oscer_nldas_zonal.sbatch")
    ap.add_argument("--qc-sbatch", default=REPO / "hpc" / "oscer_qcagent.sbatch")
    ap.add_argument("--usa-sbatch", default=REPO / "hpc" / "oscer_nldas_usa.sbatch")
    ap.add_argument("--interval", type=int, default=300, help="seconds between checks")
    ap.add_argument("--stale-min", type=int, default=35, help="QCagent heartbeat staleness = hung")
    ap.add_argument("--target", choices=["zonal", "pixel"], default="zonal",
                    help="which campaign to watch: zonal download or the per-pixel re-download")
    ap.add_argument("--min-steps", type=int, default=8000)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    tgt = args.target
    years = range(int(cfg["period"]["start"][:4]), int(cfg["period"]["end"][:4]) + 1)
    total = len(years) if tgt == "pixel" else len(years) * 12
    # relaunch the QCagent in the SAME target mode it is watching
    qc_relaunch = ([f"--export=ALL,TARGET={tgt},MIN_STEPS={args.min_steps}", str(args.qc_sbatch)]
                   if tgt == "pixel" else [str(args.qc_sbatch)])
    seen_bad: set[str] = set(bad_nodes())
    qc_relaunch_at = 0.0          # debounce: don't relaunch QCagent more than ~1/10min
    log.info("watchdog target=%s (total=%d)", tgt, total)

    while True:
        q = jobs_by_name()
        nz = pixel_years_done(cfg, years, args.min_steps) if tgt == "pixel" else zonal_count(cfg)
        status, age = read_status(cfg, tgt)
        up = server_up()
        bn = bad_nodes()
        qc_states = [s for _, s in q.get("QCagent", [])]
        usa_states = [s for _, s in q.get("nldas_usa", [])]
        zonal_q = q.get("nldas_zonal", [])
        gave_up = status.get("gave_up", []) if status else []
        complete = nz >= total

        log.info("%s %d/%d | QCagent=%s (hb %.0fmin) | nldas_zonal jobs=%d | "
                 "gave_up=%s | bad_nodes=%s | GES DISC %s",
                 tgt, nz, total, qc_states or "absent",
                 (age / 60 if age != float("inf") else -1),
                 len(zonal_q), gave_up or [], bn or [], "up" if up else "DOWN")

        # new bad nodes -> surface for the OSCER report
        new_bad = set(bn) - seen_bad
        if new_bad:
            alert(cfg, f"NEW BAD NODE(S) (geo2 failed to load -- report to OSCER): {sorted(new_bad)}")
            seen_bad |= new_bad

        handled_anomaly = True
        if not complete and up:
            # 1. QCagent absent while work remains -> relaunch (debounced), same target
            if not qc_states and (time.monotonic() - qc_relaunch_at) > 600:
                log.warning("QCagent absent and %s incomplete -- relaunching (target=%s)", tgt, tgt)
                submit(qc_relaunch, args.dry_run)
                qc_relaunch_at = time.monotonic()
            # 2. QCagent present but heartbeat stale -> hung; scancel + relaunch
            elif qc_states and age > args.stale_min * 60 and (time.monotonic() - qc_relaunch_at) > 600:
                log.warning("QCagent heartbeat stale (%.0fmin) -- scancel + relaunch", age / 60)
                for jid, _ in q.get("QCagent", []):
                    cancel(jid, args.dry_run)
                submit(qc_relaunch, args.dry_run)
                qc_relaunch_at = time.monotonic()
            # 3. given-up years not in the queue -> resubmit (sbatch self-excludes bad nodes)
            queued_years = {int(k.split("%")[0].split("-")[0]) for k, _ in zonal_q if k[:4].isdigit()}
            for y in gave_up:
                if int(y) not in queued_years:
                    log.warning("resubmitting given-up year %s", y)
                    submit([f"--array={y}", str(args.zonal_sbatch)], args.dry_run)
        elif complete:
            if tgt == "zonal":
                # all zonal in; ensure the downstream USA analysis runs once
                fig = Path(cfg["paths"]["figures_dir"]) / "map_usa_feasibility.png"
                if not fig.exists() and not usa_states:
                    log.warning("data complete, USA analysis missing -- submitting it")
                    submit([str(args.usa_sbatch)], args.dry_run)
                else:
                    log.info("pipeline complete (or downstream running); nothing to do")
            else:
                log.info("all %d per-pixel years complete; siting handled by the QCagent", total)
        elif not up:
            log.info("GES DISC down -- holding (no resubmits)")
        else:
            handled_anomaly = False

        # novel anomaly: not progressing and none of the rules applied
        if not handled_anomaly:
            alert(cfg, f"unclassified {tgt} state: {nz}/{total}, QCagent={qc_states}, gave_up={gave_up}")
            ask_claude(cfg, f"target={tgt} {nz}/{total} QCagent={qc_states} "
                            f"gave_up={gave_up} bad_nodes={bn} server_up={up}")

        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
