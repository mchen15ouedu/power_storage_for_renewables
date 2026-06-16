"""Stage 4 (extra): renewable-feasibility maps + storage-requirement rankings.

Reads results/summary.csv (one row per region) and data/regions.gpkg and writes:

  figures/map_feasibility.png      -- a 6-panel overview + bottom legend strip:
      (a,b,c) binary "sustainable" maps where the optimal-mix storage
          requirement is <= 2% / 7% / 10% of annual production;
      (d) continuous optimal-mix storage (% of annual production);
      (e) continuous solar-only storage (% of annual solar production);
      (f) continuous wind-only storage (% of annual wind production);
      panels (d)(e)(f) each carry their own colorbar/scale; the binary legend
      sits in a thin full-width strip between the title and the top row.

  results/storage_ranking_full.csv     -- all regions, ranked by storage need;
  results/storage_ranking_extremes.csv -- the 50 lowest + 50 highest.

Storage is reported on two bases:
  * mix_s_pct_demand     = % of annual consumption/DEMAND  (== mix_s_tot_pct;
        this is the basis the published ~7%-for-100%-renewable figure uses);
  * mix_s_pct_production = % of annual PRODUCTION = demand-basis / overbuild f_adj.
The maps use the production basis (as requested); the demand basis is carried
alongside because for a few equatorial near-zero-wind regions the mix overbuild
factor f_adj is pathological and deflates the production basis.

Usage:  python scripts/04_feasibility.py [--config hpc/config_oscer.yaml]
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage.config import load_config

log = logging.getLogger(__name__)

THRESHOLDS = [2.0, 7.0, 10.0]          # % of annual production
FEASIBLE_C = "#1a9850"                 # green
INFEASIBLE_C = "#d73027"               # red
FAIL_C = "#d9d9d9"                     # light grey (above-threshold)
NODATA_C = "#f7f7f7"                   # near-white (no data)


def _draw(ax, sub, color):
    """Plot a subset only if non-empty (geopandas errors on empty geometry)."""
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=0)


def _binary_map(ax, gdf, mask, title, pass_c, fail_c):
    """Two-tone choropleth: `mask` True -> pass_c, False -> fail_c, NaN -> NODATA."""
    valid = gdf["_metric"].notna()
    _draw(ax, gdf[~valid], NODATA_C)
    _draw(ax, gdf[valid & mask], pass_c)
    _draw(ax, gdf[valid & ~mask], fail_c)
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default=Path(__file__).resolve().parents[1] / "hpc" / "config_oscer.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    results_dir = cfg["paths"]["results_dir"]
    figdir = cfg["paths"]["figures_dir"]

    summary = pd.read_csv(results_dir / "summary.csv")
    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])

    # --- derived metrics ----------------------------------------------------
    # storage as % of annual PRODUCTION = (% of demand) / overbuild factor f_adj
    summary["mix_s_pct_demand"] = summary["mix_s_tot_pct"]
    summary["mix_s_pct_production"] = summary["mix_s_tot_pct"] / summary["mix_f_adj"]
    summary["solar_s_pct_production"] = summary["solar_s_tot_pct"] / summary["solar_f_adj"]
    summary["wind_s_pct_production"] = summary["wind_s_tot_pct"] / summary["wind_f_adj"]
    summary["feasible"] = summary["solar_converged"].astype(bool) & \
        summary["wind_converged"].astype(bool)

    prod = summary["mix_s_pct_production"]
    n = len(summary)
    counts = {t: int((prod <= t).sum()) for t in THRESHOLDS}
    n_feas = int(summary["feasible"].sum())
    log.info("regions: %d | technically feasible: %d", n, n_feas)
    for t in THRESHOLDS:
        log.info("storage <= %g%% of production: %d (%.1f%%)",
                 t, counts[t], 100 * counts[t] / n)

    # --- rankings -----------------------------------------------------------
    rank_cols = ["region_id", "name", "country", "level", "continent",
                 "mix_alpha", "mix_s_pct_production", "mix_s_pct_demand",
                 "mix_f_adj", "solar_cf", "wind_cf", "feasible"]
    ranked = summary[rank_cols].sort_values("mix_s_pct_production").reset_index(drop=True)
    ranked.insert(0, "rank", ranked.index + 1)
    ranked = ranked.round(4)
    ranked.to_csv(results_dir / "storage_ranking_full.csv", index=False)
    log.info("wrote %s (%d rows)", "storage_ranking_full.csv", len(ranked))

    lowest = ranked.head(50).assign(group="lowest-50 (easiest)")
    highest = ranked.tail(50).assign(group="highest-50 (hardest)")
    extremes = pd.concat([lowest, highest], ignore_index=True)
    extremes.to_csv(results_dir / "storage_ranking_extremes.csv", index=False)
    log.info("wrote %s (%d rows)", "storage_ranking_extremes.csv", len(extremes))

    # --- maps ---------------------------------------------------------------
    gdf = regions.merge(
        summary[["region_id", "mix_s_pct_production", "solar_s_pct_production",
                 "wind_s_pct_production", "feasible"]],
        on="region_id", how="left")
    gdf["_metric"] = gdf["mix_s_pct_production"]

    # thin binary-legend strip between the title and the top row of panels,
    # then 2 rows of maps (binary thresholds on top, continuous below).
    # height tuned so each 2:1 world map fills its cell (avoids a big vertical
    # gap between the rows from maps floating in over-tall cells).
    fig = plt.figure(figsize=(19, 7.4))
    gs = fig.add_gridspec(3, 3, height_ratios=[0.12, 1.0, 1.0],
                          hspace=0.02, wspace=0.04,
                          left=0.01, right=0.97, top=0.91, bottom=0.02)
    fig.suptitle("Solar+wind sustainable regions with electric storage requirements "
                 "(optimal mix, GLDAS 2000-2025, flat demand)",
                 fontsize=15, weight="bold")

    # binary legend, full-width strip under the title (applies to panels a-c)
    axleg = fig.add_subplot(gs[0, :])
    axleg.set_axis_off()
    handles = [
        Patch(facecolor=FEASIBLE_C, label="sustainable (≤ threshold)"),
        Patch(facecolor=FAIL_C, label="above storage threshold"),
        Patch(facecolor=NODATA_C, edgecolor="0.6", label="no data"),
    ]
    axleg.legend(handles=handles, loc="center", ncol=3, fontsize=11, frameon=True,
                 title="Panels (a)-(c): optimal-mix storage vs. threshold")

    axmap = [fig.add_subplot(gs[r, c]) for r in (1, 2) for c in (0, 1, 2)]

    # (a,b,c) optimal-mix storage thresholds (binary)
    for k, t in enumerate(THRESHOLDS):
        mask = gdf["_metric"] <= t
        _binary_map(axmap[k], gdf, mask.fillna(False),
                    f"({'abc'[k]}) Mix storage ≤ {t:g}% of production\n"
                    f"{counts[t]}/{n} regions ({100*counts[t]/n:.0f}%)",
                    FEASIBLE_C, FAIL_C)

    # (d,e,f) continuous storage maps -- each with its OWN colorbar/scale
    cont = [
        ("mix_s_pct_production",   "(d) Optimal-mix storage\n% of annual production"),
        ("solar_s_pct_production", "(e) Solar-only storage\n% of annual solar production"),
        ("wind_s_pct_production",  "(f) Wind-only storage\n% of annual wind production"),
    ]
    for axc, (col, title) in zip(axmap[3:], cont):
        vmax = max(5.0, 5.0 * math.ceil(gdf[col].quantile(0.95) / 5.0))
        gdf.plot(column=col, ax=axc, cmap="viridis", vmin=0.0, vmax=vmax,
                 legend=True, missing_kwds={"color": NODATA_C},
                 legend_kwds={"shrink": 0.55, "extend": "max",
                              "label": "% of annual production"})
        axc.set_title(title, fontsize=10)
        axc.set_axis_off()

    out = figdir / "map_feasibility.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
