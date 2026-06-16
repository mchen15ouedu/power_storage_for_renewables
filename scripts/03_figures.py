"""Stage 3: variability and deficit/storage figures plus world summary maps.

Usage:  python scripts/03_figures.py [--config config.yaml]
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gldas_storage import figures
from gldas_storage.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    summary = pd.read_csv(cfg["paths"]["results_dir"] / "summary.csv")
    figures.run(cfg, regions, summary)


if __name__ == "__main__":
    main()
