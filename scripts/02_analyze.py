"""Stage 2: capacity factors + all analyses for every region and pool.

Writes results/summary.csv (per region) and results/pooled_summary.csv
(continents + world) -- the global Table-1 analogue plus the added metrics
(diurnal/seasonal storage split, optimal mix, archetypes, dunkelflaute, trends).

Usage:  python scripts/02_analyze.py [--config config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gldas_storage import analyze
from gldas_storage.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    regions = pd.read_csv(cfg["paths"]["regions_csv"])
    analyze.run(cfg, regions)


if __name__ == "__main__":
    main()
