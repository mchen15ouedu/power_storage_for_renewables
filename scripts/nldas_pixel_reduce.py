#!/usr/bin/env python
"""Per-PIXEL NLDAS reduction (NREL-style optimal-siting layer; additive — the zonal
pipeline is untouched).

Instead of collapsing each granule to a per-state zonal mean, this keeps every CONUS
land cell and accumulates its **per-year** solar and wind capacity factor, so the
downstream siting analysis can rank cells best-first and size a worst-year-reliable
build. Output is tiny: one row per (cell, year).

For one year (a SLURM-array task) it scans that year's hourly granules under
``paths.raw_dir/<year>``, converts each cell's forcing to solar/wind CF with the SAME
energy functions the zonal pipeline uses (gldas_storage.energy), and writes
``<data_dir>/pixel_cf/pixel_cf_<year>.parquet`` with columns:
    cell, region_id, lat, lon, area_km2, year, n_steps, solar_cf, wind_cf

Cell→state assignment and the land mask are reused from the saved NLDAS region weights
(regions.load), so pixels map to exactly the same 51 states as the zonal run.

Usage:
    python scripts/nldas_pixel_reduce.py --config hpc/config_nldas.yaml --year 2010
    python scripts/nldas_pixel_reduce.py --config hpc/config_nldas.yaml --sample   # smoke test
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import energy, regions as reg            # noqa: E402
from gldas_storage.config import load_config                # noqa: E402

log = logging.getLogger("nldas_pixel")

DEG_KM = 111.32  # km per degree latitude (and per degree lon at the equator)


def cell_geometry(cfg, w, land):
    """Per-land-cell (lat, lon, area_km2, region_id) aligned to the columns of w.

    Cell order matches regions.build_weights: the row-major nonzero order of `land`.
    region_id = the region holding the largest weight for that column (cells on a
    state border resolve to their majority state)."""
    lat, lon, _ = reg.grid_from_granule(cfg, sample=_sample_granule(cfg))
    iy, ix = np.nonzero(land)
    cell_lat, cell_lon = lat[iy], lon[ix]
    # NLDAS grid spacing (assume uniform)
    dlat = abs(float(lat[1] - lat[0])); dlon = abs(float(lon[1] - lon[0]))
    area = (dlat * DEG_KM) * (dlon * DEG_KM * np.cos(np.deg2rad(cell_lat)))
    # cell -> region via column-argmax of the weight matrix
    wc = w.tocsc()
    region_of = np.full(w.shape[1], -1, dtype=np.int64)
    for c in range(w.shape[1]):
        sl = slice(wc.indptr[c], wc.indptr[c + 1])
        rows, vals = wc.indices[sl], wc.data[sl]
        if rows.size:
            region_of[c] = rows[np.argmax(vals)]
    return pd.DataFrame({"cell": np.arange(cell_lat.size), "region_id": region_of,
                         "lat": cell_lat, "lon": cell_lon, "area_km2": area})


def _sample_granule(cfg):
    pat = f"{cfg['gldas']['short_name']}.A*.{cfg['gldas'].get('file_ext', 'nc')}"
    c = sorted(cfg["paths"]["raw_dir"].rglob(pat))
    if not c:
        raise FileNotFoundError(f"no NLDAS granules under {cfg['paths']['raw_dir']}")
    return c[0]


def year_granules(cfg, year: int):
    pat = f"{cfg['gldas']['short_name']}.A{year}*.{cfg['gldas'].get('file_ext', 'nc')}"
    return sorted(cfg["paths"]["raw_dir"].rglob(pat))


def reduce_files(cfg, files, land):
    """Accumulate per-cell mean solar/wind CF over a list of granules."""
    land_flat = land.ravel()
    varmap = cfg["gldas"]["variables"]           # SWdown->swdown, Tair->tair, ...
    wc = cfg["gldas"]["wind_components"]          # [Wind_E, Wind_N]
    n = int(land_flat.sum())
    sum_s = np.zeros(n); sum_w = np.zeros(n); cnt = 0
    for f in files:
        with xr.open_dataset(f) as ds:
            g = {col: np.nan_to_num(ds[var].values[0].ravel()[land_flat])
                 for var, col in varmap.items()}
            e = ds[wc[0]].values[0].ravel()[land_flat]
            nn = ds[wc[1]].values[0].ravel()[land_flat]
            speed = np.nan_to_num(np.hypot(e, nn))
        sum_s += energy.solar_cf(g["swdown"], cfg)
        sum_w += energy.wind_cf(speed, cfg, tair=g.get("tair"),
                                psurf=g.get("psurf"), qair=g.get("qair"))
        cnt += 1
    return sum_s, sum_w, cnt


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--config", default=repo / "hpc" / "config_nldas.yaml")
    ap.add_argument("--year", type=int, help="calendar year to reduce")
    ap.add_argument("--sample", action="store_true", help="smoke test on one granule")
    ap.add_argument("--out", default=None, help="override output dir (default <data_dir>/pixel_cf)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    _, w, land = reg.load(cfg)
    geom = cell_geometry(cfg, w, land)
    log.info("%d land cells; %d states represented", len(geom),
             geom["region_id"].nunique())

    outdir = Path(args.out) if args.out else Path(cfg["paths"]["data_dir"]) / "pixel_cf"
    outdir.mkdir(parents=True, exist_ok=True)

    if args.sample:
        files = [_sample_granule(cfg)]
        year = pd.Timestamp(xr.open_dataset(files[0])["time"].values[0]).year
        log.info("SAMPLE on %s", files[0].name)
    else:
        if args.year is None:
            ap.error("--year is required unless --sample")
        year = args.year
        files = year_granules(cfg, year)
        if not files:
            log.error("no granules for %d under %s", year, cfg["paths"]["raw_dir"]); return
        log.info("year %d: %d granules", year, len(files))

    sum_s, sum_w, cnt = reduce_files(cfg, files, land)
    out = geom.copy()
    out["year"] = year
    out["n_steps"] = cnt
    out["solar_cf"] = (sum_s / max(cnt, 1)).astype(np.float32)
    out["wind_cf"] = (sum_w / max(cnt, 1)).astype(np.float32)

    name = f"pixel_cf_{year}{'_sample' if args.sample else ''}.parquet"
    dest = outdir / name
    out.to_parquet(dest, index=False)
    log.info("wrote %s (%d cells, %d steps; solar_cf mean %.3f, wind_cf mean %.3f)",
             dest, len(out), cnt, out["solar_cf"].mean(), out["wind_cf"].mean())


if __name__ == "__main__":
    main()
