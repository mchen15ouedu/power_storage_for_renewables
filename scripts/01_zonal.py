"""Stage 1: reduce 3-hourly GLDAS granules to 3-hourly zonal means per region.

Granules come either from earthaccess (download_mode: earthaccess) or are
already on disk under raw_dir (download_mode: local, e.g. wget). Writes one
parquet per month into paths.zonal_dir; existing months are skipped.

Usage:  python scripts/01_zonal.py [--config config.yaml] [--start Y-M-D] [--end Y-M-D]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gldas_storage import process
from gldas_storage import regions as reg
from gldas_storage.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--start", default=None,
                        help="override period start (YYYY-MM-DD), e.g. one year of a SLURM array")
    parser.add_argument("--end", default=None, help="override period end (YYYY-MM-DD)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    if args.start:
        cfg["period"]["start"] = args.start
    if args.end:
        cfg["period"]["end"] = args.end
    ensure_dirs(cfg)

    _, w, land = reg.load(cfg)
    process.run(cfg, w, land)


if __name__ == "__main__":
    main()
