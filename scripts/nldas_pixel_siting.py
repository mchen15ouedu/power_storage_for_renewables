#!/usr/bin/env python
"""NREL-style optimal-siting land analysis for the USA (additive layer).

Reads the per-pixel NLDAS capacity factors (scripts/nldas_pixel_reduce.py) and, for
each state, answers: *how little land does it take to meet demand if you build on the
best grid cells first?*

Algorithm (per state, per demand scenario):
  1. Each cell builds its BEST resource — solar (30 MW/km²) or wind (100 MW/km² direct
     footprint) — whichever yields more energy per km² (long-term mean).
  2. Rank cells by that energy density, best first.
  3. Add cells until the build is WORST-YEAR reliable: the weakest year's cumulative
     generation across the chosen cells meets that scenario's annual demand.
  4. Land = area of the chosen cells (last one fractional); compare to 1% / 1.88% / 5%
     of the state's land.

Demand scenarios 1× / 2× / 3× real EIA consumption are motivated by Goldman Sachs'
AI/data-center load-growth projection (data-center power demand ~doubles 2025→2027).
Also reports solar-only and wind-only siting as bookends.

Output: <results_dir>/usa_pixel_siting.csv
Usage:  python scripts/nldas_pixel_siting.py --config hpc/config_nldas.yaml
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import landuse                            # noqa: E402
from gldas_storage.config import load_config                 # noqa: E402

log = logging.getLogger("nldas_siting")

HOURS = landuse.HOURS_PER_YEAR
SOLAR_D = landuse.SOLAR_MW_PER_KM2                            # 30 MW/km²
WIND_D = landuse.WIND_DIRECT_MW_PER_KM2                       # 100 MW/km² direct
# source-backed demand-growth scenarios (multiplier on each state's current EIA use):
#   today   1.00x  2024 EIA actual
#   2030ref 1.14x  Goldman ~2.6%/yr total US power-demand growth 2025-2030 (data centers ~half)
#   2030hi  1.25x  high data-center case (EPRI: data centers ~17% of US electricity by 2030)
SCENARIOS = [("today", 1.00, "2024 baseline"),
             ("2030ref", 1.14, "2030 reference (Goldman ~2.6%/yr)"),
             ("2030hi", 1.25, "2030 high-AI (EPRI ~17% DC share)")]
MODES = ["best", "solar", "wind"]


def site_state(area, solar_cf_yr, wind_cf_yr, demand_gwh, mode):
    """Minimum land (km²) to meet `demand_gwh` worst-year, siting best cells first.

    area: (P,) km² per cell; *_cf_yr: (P, Y) per-year capacity factor.
    Returns dict(land_km2, npix, met, solar_share). land=NaN/met=False if even using
    every cell the worst year cannot meet demand (an energy deficit, not a land one)."""
    # per-cell, per-year energy density [GWh/km²/yr] and the best-resource pick
    s_dens = solar_cf_yr * SOLAR_D * HOURS / 1e3
    w_dens = wind_cf_yr * WIND_D * HOURS / 1e3
    s_mean, w_mean = s_dens.mean(1), w_dens.mean(1)
    if mode == "solar":
        dens = s_dens; is_solar = np.ones(len(area), bool)
    elif mode == "wind":
        dens = w_dens; is_solar = np.zeros(len(area), bool)
    else:
        is_solar = s_mean >= w_mean
        dens = np.where(is_solar[:, None], s_dens, w_dens)
    energy_yr = dens * area[:, None]                          # (P,Y) GWh/yr if cell fully built
    rank = np.argsort(-dens.mean(1))                          # best energy density first
    cum = np.cumsum(energy_yr[rank], axis=0)                  # (P,Y)
    worst = cum.min(axis=1)                                   # weakest-year cumulative
    if worst[-1] < demand_gwh:                                # can't meet even with all land
        return {"land_km2": float(np.nan), "npix": len(area), "met": False,
                "solar_share": float(is_solar[rank].mean())}
    k = int(np.searchsorted(worst, demand_gwh))              # first prefix that meets demand
    a = area[rank]
    prev = worst[k - 1] if k > 0 else 0.0
    f = 1.0 if worst[k] <= prev else (demand_gwh - prev) / (worst[k] - prev)
    land = float(a[:k].sum() + f * a[k])
    return {"land_km2": land, "npix": k + f, "met": True,
            "solar_share": float(is_solar[rank][:k + 1].mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--config", default=repo / "hpc" / "config_nldas.yaml")
    ap.add_argument("--usa-summary",
                    default="/ourdisk/hpc/caps/mchen15/gldas_analysis/results/usa_summary.csv",
                    help="source of real EIA demand + equal-area state land (by state name)")
    ap.add_argument("--pixel-dir", default=None)
    ap.add_argument("--min-steps", type=int, default=8000,
                    help="drop years with fewer than this many reduced hours (partial downloads)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    pdir = Path(args.pixel_dir) if args.pixel_dir else Path(cfg["paths"]["data_dir"]) / "pixel_cf"
    files = sorted(p for p in pdir.glob("pixel_cf_*.parquet") if "sample" not in p.name)
    if not files:
        log.error("no pixel_cf parquets in %s — run nldas_pixel_reduce.py first", pdir); return
    px = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    px = px[px["region_id"] >= 0]
    # drop partial-download years (biased annual CF); worst-year needs full years
    if "n_steps" in px.columns:
        full = px.groupby("year")["n_steps"].max()
        keep = full[full >= args.min_steps].index
        dropped = sorted(set(px["year"].unique()) - set(keep))
        if dropped:
            log.warning("dropping %d partial years (<%d h): %s", len(dropped),
                        args.min_steps, dropped)
        px = px[px["year"].isin(keep)]
    years = sorted(px["year"].unique())
    if not years:
        log.error("no complete years available (need >=%d h); run the full reduction first",
                  args.min_steps); return
    log.info("pixels: %d cells x %d complete years (%s-%s)", px["cell"].nunique(),
             len(years), years[0], years[-1])

    # state name + real EIA demand + equal-area land area (dedupe legacy .1 cols)
    rg = gpd.read_file(cfg["paths"]["regions_gpkg"])[["region_id", "name"]]
    name_of = dict(zip(rg["region_id"], rg["name"]))
    us = pd.read_csv(args.usa_summary)
    us = us[[c for c in us.columns if not c.endswith(".1")]]
    demand_of = dict(zip(us["name"], us["annual_consumption_TWh"]))      # TWh/yr (real EIA)
    area_of = dict(zip(us["name"], us["state_land_km2"]))                # km² equal-area

    rows = []
    for rid, grp in px.groupby("region_id"):
        name = name_of.get(rid)
        if name is None or name not in demand_of:
            continue
        cells = grp.groupby("cell")
        area = cells["area_km2"].first().to_numpy(float)
        # (P,Y) per-year CF matrices, aligned cell x year
        piv_s = grp.pivot_table("solar_cf", "cell", "year").reindex(columns=years)
        piv_w = grp.pivot_table("wind_cf", "cell", "year").reindex(columns=years)
        s_yr = piv_s.to_numpy(float); w_yr = piv_w.to_numpy(float)
        area = piv_s.index.map(cells["area_km2"].first()).to_numpy(float)
        st_area = float(area_of[name]); dem_twh = float(demand_of[name])
        rec = {"region_id": int(rid), "name": name,
               "annual_consumption_TWh": round(dem_twh, 2),
               "state_land_km2": round(st_area, 1),
               "grid_land_km2": round(float(area.sum()), 1), "n_cells": len(area)}
        for mode in MODES:
            for skey, mult, _ in SCENARIOS:
                r = site_state(area, s_yr, w_yr, dem_twh * 1e3 * mult, mode)
                tag = f"{mode}_{skey}"
                lp = (r["land_km2"] / st_area * 100) if r["met"] else np.nan
                rec[f"land_pct_{tag}"] = round(lp, 4) if r["met"] else None
                rec[f"land_km2_{tag}"] = round(r["land_km2"], 1) if r["met"] else None
                rec[f"met_{tag}"] = r["met"]
                if mode == "best":
                    rec[f"solar_share_{tag}"] = round(r["solar_share"], 3)
                    for key, frac, _ in landuse.LAND_CAPS:
                        rec[f"feasible_{key}_{tag}"] = bool(r["met"] and lp <= frac * 100)
        rows.append(rec)
        log.info("%-22s demand %.0f TWh  best today %.3f%%  2030hi %.3f%%", name, dem_twh,
                 rec.get("land_pct_best_today") or -1, rec.get("land_pct_best_2030hi") or -1)

    out = pd.DataFrame(rows).sort_values("name")
    P = f"{cfg['usa_tag']}_" if cfg.get("usa_tag") else ""    # e.g. "nldas_"
    dest = Path(cfg["paths"]["results_dir"]) / f"{P}usa_pixel_siting.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    log.info("wrote %s (%d states, years %s-%s)", dest, len(out), years[0], years[-1])
    for skey, mult, slbl in SCENARIOS:
        for key, frac, lbl in landuse.LAND_CAPS:
            n = int(out[f"feasible_{key}_best_{skey}"].sum())
            log.info("  %s (%.2fx), best siting feasible @%s: %d/%d", slbl, mult, lbl, n, len(out))


if __name__ == "__main__":
    main()
