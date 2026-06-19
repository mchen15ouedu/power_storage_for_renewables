"""Non-invasive integrity check for a year's NLDAS zonal parquet files.

Opens each zonal_YYYYMM.parquet for the year and validates it actually reads and
holds sane data -- catches a silently-corrupt-but-present file that the pipeline's
skip-existing logic would otherwise never regenerate. Read-only: it never deletes
or rewrites anything; it just reports PASS/FAIL per month and exits non-zero if any
month is bad, naming exactly which files to delete so a normal re-run heals them.

Usage:
  python scripts/validate_year_parquet.py --year 2013 \
         --dir /ourdisk/hpc/caps/mchen15/nldas_full
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def check_month(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "MISSING"
    try:
        df = pd.read_parquet(path)
    except Exception as e:                       # corrupt / truncated parquet
        return False, f"UNREADABLE ({type(e).__name__}: {e})"
    if len(df) == 0:
        return False, "EMPTY (0 rows)"
    if "region_id" not in df.columns:
        return False, f"no region_id column (cols={list(df.columns)})"
    n_reg = df["region_id"].nunique()
    num = df.select_dtypes("number").drop(columns=["region_id"], errors="ignore")
    # a data column that is entirely NaN/zero across every region signals a bad reduce
    dead = [c for c in num.columns if not num[c].notna().any()]
    if dead:
        return False, f"all-NaN data column(s): {dead}"
    return True, f"OK ({len(df)} rows, {n_reg} regions, cols={list(df.columns)})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--dir", required=True, help="zonal_dir holding zonal_YYYYMM.parquet")
    args = ap.parse_args()
    zdir = Path(args.dir)

    bad = []
    print(f"=== validating {args.year} zonal parquet in {zdir} ===")
    for m in range(1, 13):
        p = zdir / f"zonal_{args.year}{m:02d}.parquet"
        ok, msg = check_month(p)
        print(f"  {p.name}: {'PASS' if ok else 'FAIL'} - {msg}")
        if not ok:
            bad.append(p)

    if bad:
        print(f"\nRESULT: {len(bad)}/12 month(s) FAILED for {args.year}.")
        print("To heal, delete these and let the normal re-run regenerate them:")
        for p in bad:
            print(f"  rm {p}")
        sys.exit(1)
    print(f"\nRESULT: all 12 months of {args.year} are clean. No action needed.")


if __name__ == "__main__":
    main()
