"""QCagent -- babysits the NLDAS run.

Two jobs, run in one long-lived loop (submit it as its own SLURM job):

1. DOWNLOAD / REDUCTION WATCH
   For each year 2000-2025 it knows the "done" condition is 12 monthly parquet
   files in zonal_dir. Every cycle it:
     * checks GES DISC reachability (if the server is down it waits -- no point
       thrashing resubmits during an outage);
     * resubmits any year that is incomplete and has NO running/pending array
       task (its job died -- e.g. a wget disruption aborted it), up to a cap;
     * detects a STALL: a year whose array task is running but whose raw-granule
       count has not grown for `stall_min` minutes -> cancels that task and
       resubmits it (a hung/disrupted download), up to the same cap.

2. FIGURE QC
   Once every year is complete and the downstream figures exist, it runs the
   layout checks in ``gldas_storage.qc`` (legend/title overlaps, excess white
   space) on the feasibility PNG and writes a consolidated qc_report.txt, then
   exits 0. Exits non-zero if it hits the wall-clock cap with work outstanding.

Usage:
   python scripts/qc_agent.py --config hpc/config_nldas.yaml \
       --sbatch hpc/oscer_nldas_zonal.sbatch [--interval 600] [--once]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import qc
from gldas_storage.config import load_config

log = logging.getLogger("qcagent")

GESDISC_PROBE = ("https://data.gesdisc.earthdata.nasa.gov/data/NLDAS/"
                 "NLDAS_FORA0125_H.2.0/2000/001/")
JOBNAME = "nldas_zonal"
MAX_RESUBMITS = 5          # per year, safety cap against runaway resubmission
REPO = Path(__file__).resolve().parents[1]


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
    except Exception as exc:                                   # pragma: no cover
        log.warning("command failed %s: %s", " ".join(cmd), exc)
        return ""


def server_up() -> bool:
    out = sh(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}",
              "--max-time", "30", "-I", GESDISC_PROBE])
    return out.strip() in ("200", "301", "302", "401", "403")  # reachable (auth codes still mean "up")


def pixel_done(cfg: dict, year: int, min_steps: int) -> bool:
    """A per-pixel year is done when its pixel_cf parquet exists with a near-full
    hour count (partial-download years have a low n_steps and don't count)."""
    f = Path(cfg["paths"]["data_dir"]) / "pixel_cf" / f"pixel_cf_{year}.parquet"
    if not f.exists():
        return False
    try:
        import pandas as pd
        return int(pd.read_parquet(f, columns=["n_steps"])["n_steps"].max()) >= min_steps
    except Exception:
        return False


def years_done(cfg: dict, years: range, target: str = "zonal",
               min_steps: int = 8000) -> dict[int, bool]:
    zdir = cfg["paths"]["zonal_dir"]
    done = {}
    for y in years:
        if target == "pixel":
            done[y] = pixel_done(cfg, y, min_steps)
        else:
            done[y] = len(list(zdir.glob(f"zonal_{y}*.parquet"))) >= 12
    return done


def progress_count(cfg: dict, year: int, target: str) -> int:
    """A monotone signal that the year is making progress, used to reset the
    resubmit budget: reduced parquets for zonal, downloaded raw granules for pixel
    (whose product is one parquet, so raw growth is the live progress signal)."""
    if target == "pixel":
        return raw_count(cfg, year)
    return len(list(cfg["paths"]["zonal_dir"].glob(f"zonal_{year}*.parquet")))


def bad_nodes() -> list[str]:
    """Distinct compute nodes that failed the env guard (geo2 not active), from
    hpc/bad_nodes.txt -- surfaced so the user can report them to OSCER."""
    f = REPO / "hpc" / "bad_nodes.txt"
    if not f.exists():
        return []
    return sorted({ln.split("\t")[1] for ln in f.read_text().splitlines()
                   if "\t" in ln and len(ln.split("\t")) > 1})


def write_status(cfg: dict, target: str = "zonal", **kv) -> None:
    """Heartbeat the QCagent's state to results/qcagent[_<target>]_status.json so the
    pipeline watchdog can tell if it is healthy or stuck (per-target file so a zonal
    and a pixel QCagent don't clobber each other)."""
    import json
    name = "qcagent_status.json" if target == "zonal" else f"qcagent_{target}_status.json"
    out = cfg["paths"]["results_dir"] / name
    out.parent.mkdir(parents=True, exist_ok=True)
    kv["target"] = target
    kv["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    kv["bad_nodes"] = bad_nodes()
    out.write_text(json.dumps(kv, indent=2))


def _expand_array_indices(field: str) -> set[int]:
    """Expand a squeue %K array-index field into the set of year indices.

    SLURM compresses a throttled/pending array into ranges, e.g. "2006-2025%4",
    and lists multiple chunks comma-separated, e.g. "2006,2008-2010". Missing
    any of these would make the QCagent resubmit years that are actually queued.
    """
    field = field.split("%")[0]            # drop the %<throttle> suffix
    yrs: set[int] = set()
    for chunk in field.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, _, b = chunk.partition("-")
            try:
                yrs.update(range(int(a), int(b) + 1))
            except ValueError:
                pass
        else:
            try:
                yrs.add(int(chunk))
            except ValueError:
                pass
    return yrs


def running_tasks() -> set[int]:
    """Year indices of nldas_zonal array tasks currently in the queue (any state)."""
    out = sh(["squeue", "-h", "-u", _user(), "-n", JOBNAME, "-o", "%K"])
    yrs: set[int] = set()
    for line in out.splitlines():
        yrs |= _expand_array_indices(line.strip())
    return yrs


def queue_states() -> dict[int, str]:
    """year -> queue state ('RUNNING' if any task for that year is running, else
    'PENDING'). A *pending* year (throttled behind the array's %N limit) has no
    raw files yet and must NEVER be treated as a stalled download -- only running
    tasks are eligible for stall detection."""
    out = sh(["squeue", "-h", "-u", _user(), "-n", JOBNAME, "-o", "%K|%T"])
    st: dict[int, str] = {}
    for line in out.splitlines():
        k, _, state = line.partition("|")
        state = state.strip()
        for y in _expand_array_indices(k.strip()):
            if st.get(y) != "RUNNING":      # RUNNING wins over PENDING
                st[y] = state
    return st


def running_jobids(year: int) -> list[str]:
    """All currently-RUNNING task ids (`<jobid>_<idx>`) for a given year. Returns
    every match (there may be more than one if a duplicate slipped through) so the
    stall handler can kill them ALL before resubmitting."""
    out = sh(["squeue", "-h", "-u", _user(), "-n", JOBNAME, "-t", "RUNNING", "-o", "%A %K"])
    ids = []
    for line in out.splitlines():
        a, _, k = line.partition(" ")
        if year in _expand_array_indices(k.strip()):
            ids.append(f"{a.strip()}_{year}")
    return ids


def kill_year(year: int, dry: bool = False) -> list[str]:
    """scancel every running task for `year`; wait briefly so the cancel registers
    before any resubmit (avoids the stalled job and its replacement coexisting)."""
    ids = running_jobids(year)
    for jid in ids:
        if dry:
            log.warning("scancel (dry-run) %s", jid)
        else:
            sh(["scancel", jid])
    if ids and not dry:
        time.sleep(8)        # let SLURM register the cancellation before resubmit
    return ids


def _user() -> str:
    import getpass
    return getpass.getuser()


def raw_count(cfg: dict, year: int) -> int:
    raw = Path(cfg["paths"]["raw_dir"]) / str(year)
    return sum(1 for _ in raw.glob("*.nc")) if raw.exists() else 0


def resubmit(sbatch: Path, year: int, dry: bool = False) -> None:
    if dry:
        log.warning("RESUBMIT (dry-run) year %d: sbatch --array=%d %s", year, year, sbatch)
        return
    out = sh(["sbatch", "--parsable", f"--array={year}", str(sbatch)])
    log.warning("RESUBMIT year %d -> job %s", year, out.strip() or "?")


def usa_job_active() -> bool:
    return bool(sh(["squeue", "-h", "-u", _user(), "-n", "nldas_usa", "-o", "%A"]).strip())


def submit_downstream(usa_sbatch: Path, dry: bool = False) -> None:
    if dry:
        log.warning("DOWNSTREAM (dry-run): sbatch %s", usa_sbatch)
        return
    out = sh(["sbatch", "--parsable", str(usa_sbatch)])
    log.warning("submitted downstream USA analysis -> job %s", out.strip() or "?")


def figure_qc(cfg: dict) -> list[str]:
    figdir = cfg["paths"]["figures_dir"]
    png = figdir / "map_usa_feasibility.png"
    issues = []
    if png.exists():
        issues += qc.raster_issues(png)
        log.info("QC %s: %s", png.name, "OK" if not issues else f"{len(issues)} issue(s)")
    else:
        issues.append(f"missing figure {png}")
    return issues


def write_report(cfg: dict, lines: list[str]) -> None:
    out = cfg["paths"]["results_dir"] / "qc_report.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=REPO / "hpc" / "config_nldas.yaml")
    ap.add_argument("--sbatch", default=REPO / "hpc" / "oscer_nldas_zonal.sbatch")
    ap.add_argument("--usa-sbatch", default=REPO / "hpc" / "oscer_nldas_usa.sbatch",
                    help="downstream USA analysis, auto-submitted once all years are reduced")
    ap.add_argument("--interval", type=int, default=600, help="seconds between cycles")
    ap.add_argument("--stall-min", type=int, default=90,
                    help="minutes of no new granules before a running task is deemed stalled")
    ap.add_argument("--max-hours", type=float, default=120.0, help="overall wall-clock cap")
    ap.add_argument("--target", choices=["zonal", "pixel"], default="zonal",
                    help="completion target: 12 zonal parquets/year, or the per-pixel CF parquet")
    ap.add_argument("--min-steps", type=int, default=8000,
                    help="(pixel) minimum reduced hours for a year to count as complete")
    ap.add_argument("--once", action="store_true", help="run a single cycle and exit")
    ap.add_argument("--dry-run", action="store_true", help="log resubmit/cancel actions but do not execute them")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    sbatch = Path(args.sbatch)
    tgt = args.target
    years = range(int(cfg["period"]["start"][:4]), int(cfg["period"]["end"][:4]) + 1)
    resubmits = {y: 0 for y in years}
    last_count = {y: (-1, time.monotonic()) for y in years}   # (raw_count, time first seen at this count)
    last_pq = {y: progress_count(cfg, y, tgt) for y in years}  # reduction/download progress
    gave_up: set[int] = set()
    deadline = time.monotonic() + args.max_hours * 3600
    log.info("QCagent target=%s (min_steps=%d)", tgt, args.min_steps)

    while True:
        done = years_done(cfg, years, tgt, args.min_steps)
        states = queue_states()                 # year -> RUNNING / PENDING
        queued = set(states)                    # pending OR running = still in the array
        running_now = {y for y, s in states.items() if s == "RUNNING"}
        up = server_up()
        n_done = sum(done.values())
        log.info("cycle: %d/%d complete | running=%s | pending=%d | gave_up=%s | GES DISC %s",
                 n_done, len(years), sorted(running_now),
                 len(queued) - len(running_now), sorted(gave_up) or "none", "up" if up else "DOWN")
        write_status(cfg, tgt, years_done=n_done, years_total=len(years),
                     running=sorted(running_now), pending=len(queued) - len(running_now),
                     gave_up=sorted(gave_up), server_up=up,
                     resubmits={y: c for y, c in resubmits.items() if c})

        if n_done < len(years) and up:
            for y in years:
                if done[y]:
                    continue
                # reduction progress => the sbatch is self-healing (e.g. excluded a
                # bad node and reran); clear the give-up state and resubmit budget.
                pq = progress_count(cfg, y, tgt)
                if pq > last_pq[y]:
                    last_pq[y] = pq
                    if resubmits[y] or y in gave_up:
                        log.info("year %d progressed (%d) -- resetting resubmit budget", y, pq)
                    resubmits[y] = 0
                    gave_up.discard(y)
                if y not in queued:
                    # genuinely gone from the queue (its job died) -> resubmit
                    if resubmits[y] < MAX_RESUBMITS:
                        resubmits[y] += 1
                        log.warning("year %d incomplete and not queued -- resubmitting "
                                    "(#%d)", y, resubmits[y])
                        resubmit(sbatch, y, args.dry_run)
                    elif y not in gave_up:
                        gave_up.add(y)         # log ONCE, not every cycle
                        log.error("year %d incomplete after %d resubmits -- giving up. "
                                  "bad nodes so far: %s", y, MAX_RESUBMITS,
                                  bad_nodes() or "none recorded")
                    continue
                if y not in running_now:
                    # PENDING behind the array's %N throttle -- 0 raw files is
                    # expected, NOT a stall. Leave it alone; reset its stall clock.
                    last_count[y] = (-1, time.monotonic())
                    continue
                # actually RUNNING: stall detection on raw-granule growth
                cnt = raw_count(cfg, y)
                prev, since = last_count[y]
                if cnt != prev:
                    last_count[y] = (cnt, time.monotonic())
                elif cnt > 0 and (time.monotonic() - since) > args.stall_min * 60:
                    # only a running task that downloaded SOME files then went
                    # quiet counts as a real stall (cnt>0 guards against a task
                    # that just started and hasn't written yet). NASA/GES DISC is
                    # spotty, so this WILL fire -- always KILL the stalled task
                    # first, then resubmit a fresh one (never leave both alive).
                    if resubmits[y] < MAX_RESUBMITS:
                        killed = kill_year(y, args.dry_run)
                        log.warning("year %d STALLED at %d granules for >%dmin -- "
                                    "killed %s, now resubmitting (#%d)", y, cnt,
                                    args.stall_min, killed or "(none found)",
                                    resubmits[y] + 1)
                        resubmits[y] += 1
                        resubmit(sbatch, y, args.dry_run)
                        last_count[y] = (-1, time.monotonic())

        # all data in -> trigger the downstream product, then finish
        if n_done == len(years):
            if tgt == "pixel":
                # all per-pixel CF in -> run the optimal-siting analysis + maps once
                map_png = cfg["paths"]["figures_dir"] / "map_usa_siting_landcaps.png"
                if not map_png.exists() or not args.dry_run:
                    log.warning("all %d years' per-pixel CF ready -- running siting + maps", len(years))
                    if not args.dry_run:
                        sh(["python", str(REPO / "scripts" / "nldas_pixel_siting.py"),
                            "--config", str(args.config)])
                        sh(["python", str(REPO / "scripts" / "nldas_siting_maps.py"),
                            "--config", str(args.config)])
                write_report(cfg, [f"NLDAS per-pixel siting complete -- {n_done}/{len(years)} years",
                                   f"bad nodes: {bad_nodes() or 'none'}"])
                log.info("QCagent(pixel) done: all per-pixel years in, siting run")
                return
            png = cfg["paths"]["figures_dir"] / "map_usa_feasibility.png"
            if not png.exists():
                if usa_job_active():
                    log.info("data complete; downstream USA analysis running, awaiting figures")
                else:
                    log.warning("all %d years reduced -- launching downstream USA analysis",
                                len(years))
                    submit_downstream(Path(args.usa_sbatch), args.dry_run)
            else:
                issues = figure_qc(cfg)
                report = [f"NLDAS QC report -- {n_done}/{len(years)} years complete",
                          "resubmits: " + (", ".join(f"{y}:{c}" for y, c in resubmits.items() if c)
                                           or "none"), ""]
                report += issues or ["figures: no layout issues detected"]
                write_report(cfg, report)
                log.info("QCagent done: data complete, figures checked")
                return

        if args.once:
            return
        if time.monotonic() > deadline:
            log.error("QCagent hit wall-clock cap with %d/%d years done", n_done, len(years))
            write_report(cfg, [f"TIMEOUT: {n_done}/{len(years)} years complete"])
            sys.exit(2)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
