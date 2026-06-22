#!/usr/bin/env python
"""USA land-cap maps for the NREL-style optimal-siting layer.

One map per demand scenario (today / 2030 reference / 2030 high-AI), each coloring
states by the tightest land cap their best-pixel-sited build fits within
(1% / 1.88% / 5% of state land), with a second row showing the continuous land-use %.

Reads <results_dir>/usa_pixel_siting.csv (scripts/nldas_pixel_siting.py) and the US
state geometries; writes <figures_dir>/map_usa_siting_landcaps.png.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.patches import Patch                               # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import landuse                                  # noqa: E402
from gldas_storage.config import load_config                       # noqa: E402

log = logging.getLogger("siting_maps")

TIER_COLORS = ["#1a9850", "#a6d96a", "#fee08b"]                     # 1% / 1.88% / 5%
FAIL_C, NODATA_C = "#d73027", "#f7f7f7"
SCEN = [("today", "1.00× — 2024 baseline"),
        ("2030ref", "1.14× — 2030 reference (Goldman)"),
        ("2030hi", "1.25× — 2030 high-AI (EPRI)")]
# CONUS bounds (drop AK/HI/territories for a readable map)
CONUS = (-125, 24, -66, 50)


def _draw(ax, sub, color):
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=0.2, edgecolor="white")


def tier_map(ax, gdf, skey, title):
    keys = [k for k, _, _ in landuse.LAND_CAPS]
    have = gdf[f"feasible_{keys[0]}_best_{skey}"].notna()
    _draw(ax, gdf[~have], NODATA_C)
    assigned = gdf.index[~have].tolist()
    for (key, _, _), col in zip(landuse.LAND_CAPS, TIER_COLORS):
        sel = have & gdf[f"feasible_{key}_best_{skey}"].eq(True) & ~gdf.index.isin(assigned)
        _draw(ax, gdf[sel], col)
        assigned += gdf.index[sel].tolist()
    _draw(ax, gdf[have & ~gdf.index.isin(assigned)], FAIL_C)        # needs > loosest cap
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def cont_map(ax, gdf, col, title, vmax):
    ok = gdf[col].notna()
    _draw(ax, gdf[~ok], NODATA_C)
    if ok.any():
        gdf[ok].plot(ax=ax, column=col, cmap="magma_r", vmin=0, vmax=vmax,
                     linewidth=0.2, edgecolor="white", legend=True,
                     legend_kwds={"shrink": 0.6, "label": "% of state land"})
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--config", default=repo / "hpc" / "config_nldas.yaml")
    ap.add_argument("--regions-gpkg",
                    default="/ourdisk/hpc/caps/mchen15/gldas_analysis/data/regions.gpkg",
                    help="US state geometries (admin1) joined to the siting CSV by name")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    rdir, figdir = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["figures_dir"])
    P = f"{cfg['usa_tag']}_" if cfg.get("usa_tag") else ""    # e.g. "nldas_"
    sit = pd.read_csv(rdir / f"{P}usa_pixel_siting.csv")

    g = gpd.read_file(args.regions_gpkg)
    us = g[(g["country"] == "United States of America") & (g["level"] == "admin1")].copy()
    us = us.merge(sit, on="name", how="left", suffixes=("", "_s"))
    us = us.cx[CONUS[0]:CONUS[2], CONUS[1]:CONUS[3]]               # clip to CONUS view
    us = us.reset_index(drop=True)

    # shared land-% color scale across scenarios (95th pct, capped)
    lp_cols = [f"land_pct_best_{k}" for k, _ in SCEN]
    allv = pd.concat([us[c] for c in lp_cols if c in us]).dropna()
    vmax = max(2.0, round(float(allv.quantile(0.95)), 1)) if len(allv) else 5.0

    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 3, hspace=0.05, wspace=0.02)
    counts = []
    for i, (skey, slbl) in enumerate(SCEN):
        n1 = int(us[f"feasible_1pct_best_{skey}"].eq(True).sum())
        n2 = int(us[f"feasible_oilgas2x_best_{skey}"].eq(True).sum())
        n5 = int(us[f"feasible_oilgas_all_best_{skey}"].eq(True).sum())
        counts.append((skey, n1, n2, n5))
        tier_map(fig.add_subplot(gs[0, i]), us, skey,
                 f"({'abc'[i]}) {slbl}\n{n1}→{n2}→{n5} feasible @1/1.88/5% land")
        cont_map(fig.add_subplot(gs[1, i]), us, f"land_pct_best_{skey}",
                 f"({'def'[i]}) land used, optimal siting", vmax)

    handles = [Patch(facecolor=c, label=l) for c, l in zip(
        TIER_COLORS, ["feasible ≤1% land", "gained ≤1.88%", "gained ≤5%"])]
    handles += [Patch(facecolor=FAIL_C, label="needs >5% land"),
                Patch(facecolor=NODATA_C, edgecolor="0.6", label="no grid data (AK/HI/DC)")]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("USA optimal-siting land use vs demand growth — best resource per NLDAS grid cell,\n"
                 "worst-year reliable, 1%/1.88%/5% land caps (1× / 1.14× / 1.25× EIA demand)",
                 fontsize=13, y=0.97)
    dest = figdir / f"map_{P}usa_siting_landcaps.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=130, bbox_inches="tight")
    log.info("wrote %s", dest)
    for skey, n1, n2, n5 in counts:
        log.info("  %s: feasible %d/%d (1%%), %d (1.88%%), %d (5%%)", skey, n1, len(us), n2, n5)


if __name__ == "__main__":
    main()
