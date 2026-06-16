"""Coding agent -- automated robustness harness for the gldas_storage codebase.

This is a DETERMINISTIC test-and-validate harness (not an autonomous LLM): it
exercises every layer of the pipeline that the GLDAS->NLDAS work touched and
emits a pass/fail report. Run it once, or with --watch to keep re-running as a
SLURM babysitter that re-checks whenever any source file changes.

Stages:
  1. compile    -- byte-compile every .py under gldas_storage/, scripts/, tests/
  2. configs    -- load every hpc/config_*.yaml and assert per-dataset invariants
  3. pytest     -- run the unit/regression suite in tests/
  4. legacy     -- run the standalone Deficit.R-equivalence script

Exit code is non-zero if any stage fails, so it can gate a pipeline. The report
is written to <results_dir>/coding_agent_report.txt for the configs found, and
also echoed to stdout (the SLURM log).

Usage:
  python scripts/coding_agent.py
  python scripts/coding_agent.py --watch --interval 600   # re-run on source changes
"""

from __future__ import annotations

import argparse
import compileall
import io
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PKG_DIRS = [ROOT / "gldas_storage", ROOT / "scripts", ROOT / "tests"]
REPORT = ROOT / "logs" / "coding_agent_report.txt"


def _stage(name: str, ok: bool, detail: str = "") -> str:
    head = f"[{'PASS' if ok else 'FAIL'}] {name}"
    return head if not detail else f"{head}\n{detail.rstrip()}"


def stage_compile() -> tuple[bool, str]:
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = all(compileall.compile_dir(str(d), quiet=1, force=True) for d in PKG_DIRS)
    return ok, "" if ok else buf.getvalue()


def stage_configs() -> tuple[bool, str]:
    """Load every config and assert the invariants that keep both datasets honest."""
    sys.path.insert(0, str(ROOT))
    from gldas_storage import analyze
    from gldas_storage.config import load_config

    lines, ok = [], True
    for cfg_path in sorted((ROOT / "hpc").glob("config_*.yaml")):
        try:
            cfg = load_config(cfg_path)
            spy = analyze.steps_per_year(cfg)
            g = cfg["gldas"]
            # universal invariants
            assert g.get("short_name"), "missing short_name"
            assert spy in (2920, 8760), f"unexpected steps_per_year {spy}"
            assert "wind10" not in g["variables"].values() or not g.get("wind_components"), \
                "wind10 both mapped directly and derived"
            # if wind is derived, exactly two components must be named
            if g.get("wind_components"):
                assert len(g["wind_components"]) == 2, "wind_components must be [E, N]"
            # every output column the energy model needs must be producible
            produced = set(g["variables"].values())
            if g.get("wind_components"):
                produced.add("wind10")
            need = {"swdown", "wind10", "tair", "psurf", "qair"}
            missing = need - produced
            assert not missing, f"forcing cannot produce {missing}"
            lines.append(f"  {cfg_path.name}: short={g['short_name']} "
                         f"ext={g.get('file_ext', 'nc4')} steps/yr={spy} "
                         f"derived_wind={bool(g.get('wind_components'))}")
        except Exception as exc:  # noqa: BLE001 -- report, don't crash the harness
            ok = False
            lines.append(f"  {cfg_path.name}: ERROR {type(exc).__name__}: {exc}")
    return ok, "\n".join(lines)


def stage_pytest() -> tuple[bool, str]:
    tests = ROOT / "tests"
    if not any(tests.glob("test_*.py")):
        return True, "  (no tests/ found)"
    proc = subprocess.run([sys.executable, "-m", "pytest", str(tests), "-q",
                           "--no-header", "--disable-warnings",
                           "-p", "no:cacheprovider"],
                          capture_output=True, text=True, cwd=str(ROOT))
    tail = "\n".join(proc.stdout.strip().splitlines()[-12:])
    return proc.returncode == 0, tail


def stage_legacy() -> tuple[bool, str]:
    legacy = ROOT / "tests" / "test_deficit_vs_r.py"
    if not legacy.exists():
        return True, "  (no legacy script)"
    proc = subprocess.run([sys.executable, str(legacy)],
                          capture_output=True, text=True, cwd=str(ROOT))
    tail = "\n".join(proc.stdout.strip().splitlines()[-3:])
    return proc.returncode == 0, tail


STAGES = [("compile", stage_compile), ("configs", stage_configs),
          ("pytest", stage_pytest), ("legacy", stage_legacy)]


def run_once() -> bool:
    report = [f"coding-agent robustness report  (root: {ROOT})", "=" * 64]
    all_ok = True
    for name, fn in STAGES:
        try:
            ok, detail = fn()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, f"  harness error: {type(exc).__name__}: {exc}"
        all_ok &= ok
        report.append(_stage(name, ok, detail))
    report.append("=" * 64)
    report.append("OVERALL: " + ("ALL STAGES PASS" if all_ok else "FAILURES PRESENT"))
    text = "\n".join(report)
    print(text, flush=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text + "\n")
    return all_ok


def _source_mtime() -> float:
    return max((p.stat().st_mtime for d in PKG_DIRS for p in d.rglob("*.py")),
              default=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--watch", action="store_true",
                    help="keep running, re-checking whenever a source file changes")
    ap.add_argument("--interval", type=int, default=600,
                    help="seconds between change checks in --watch mode")
    args = ap.parse_args()

    if not args.watch:
        sys.exit(0 if run_once() else 1)

    last = -1.0
    print(f"coding-agent watch mode: polling every {args.interval}s", flush=True)
    while True:
        mtime = _source_mtime()
        if mtime != last:
            last = mtime
            run_once()
            print(f"--- waiting for source changes (poll {args.interval}s) ---", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
