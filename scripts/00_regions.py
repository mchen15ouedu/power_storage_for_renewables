"""Stage 0 (run once, before the big processing run): build the analysis regions
and the region->grid weight matrix.

Needs ONE sample 3-hourly granule for the grid and land mask -- download any
single GLDAS_NOAH025_3H file into raw_dir first (e.g. the first line of the
wget list), or pass --sample.

Usage:  python scripts/00_regions.py [--config config.yaml] [--sample path.nc4]
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gldas_storage import regions as reg
from gldas_storage.config import ensure_dirs, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=Path(__file__).resolve().parents[1] / "config.yaml")
    parser.add_argument("--sample", default=None, help="path to a sample granule (.nc4)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    gdf = reg.build_regions(cfg)
    lat, lon, land = reg.grid_from_granule(cfg, Path(args.sample) if args.sample else None)
    w = reg.build_weights(cfg, gdf, lat, lon, land)
    reg.save(cfg, gdf, w, land)


if __name__ == "__main__":
    main()
