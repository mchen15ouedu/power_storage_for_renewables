"""Co-location feasibility maps: how the optimal-mix storage requirement shrinks
when solar and wind share the same land (the proportional overbuild scenario).

For every region it reuses the storage-optimal solar share alpha* from
results/summary.csv (which is essentially independent of the co-location factor k,
see colocation_scenario.md), rebuilds the optimal mix from the zonal series, and
recomputes the storage need S_tot with a (1+k) overbuild via
``deficit.analyze_region(..., overbuild=1+k)`` for k in {0, 0.17, 0.35, 0.50}.

Storage is reported as % of annual CONSUMPTION (a fixed denominator -- unlike
"% of production", whose denominator (1+k)*f_adj would itself inflate with k).

Outputs:
  results/colocation_storage.csv          -- per region, S_tot at each k;
  figures/map_colocation_feasibility.png  -- binary "sustainable (<= threshold)"
      world maps, one column per co-location level (baseline + 17/35/50 %),
      one row per storage threshold.

Usage:
  python scripts/colocation_maps.py [--config hpc/config_oscer.yaml]
                                    [--thresholds 5 10 20] [--no-recompute]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import analyze, deficit, process, qc
from gldas_storage.config import load_config

log = logging.getLogger(__name__)

K_LEVELS = [0.0, 0.17, 0.35, 0.50]      # 0 = baseline; 17/35/50 % co-location
FEASIBLE_C, FAIL_C, NODATA_C = "#1a9850", "#d9d9d9", "#f7f7f7"


def kcol(k: float) -> str:
    return "k0" if k == 0 else f"k{int(round(k*100))}"


def recompute(cfg: dict, regions: gpd.GeoDataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per-region optimal-mix S_tot (% of consumption) at each co-location level."""
    spy = analyze.steps_per_year(cfg)
    storage_cfg = cfg["storage"]
    alpha_by_id = dict(zip(summary["region_id"].astype(int), summary["mix_alpha"]))
    meta = regions.set_index("region_id")

    table = process.load_zonal(cfg)
    table = table[table["region_id"].isin(alpha_by_id)]
    rows = []
    for rid, grp in table.groupby("region_id"):
        rid = int(rid)
        alpha = float(alpha_by_id.get(rid, np.nan))
        if not np.isfinite(alpha):
            continue
        cf_df = analyze.capacity_factors(grp, cfg)
        s, w = cf_df["solar"].to_numpy(), cf_df["wind"].to_numpy()
        if s.mean() <= 0 or w.mean() <= 0:
            continue
        mix = alpha * (s / s.mean()) + (1 - alpha) * (w / w.mean())
        demand = deficit.make_demand(cfg, cf_df.index, tair=cf_df["tair"].to_numpy(),
                                     monthly=analyze.monthly_demand_profile(cfg, rid))
        row = {"region_id": rid, "name": meta.loc[rid, "name"], "mix_alpha": alpha}
        for k in K_LEVELS:
            res = deficit.analyze_region(mix, cf_df.index, demand, storage_cfg, spy,
                                         simulate=False, overbuild=1.0 + k)
            # store BOTH bases: % of annual consumption (s_tot) and % of annual
            # production (= consumption / overbuilt installation factor f_adj)
            row[f"{kcol(k)}_cons"] = res.s_tot * 100.0
            row[f"{kcol(k)}_prod"] = (res.s_tot / res.tot_factor * 100.0
                                      if res.tot_factor and res.tot_factor > 0 else np.nan)
        rows.append(row)
        if len(rows) % 100 == 0:
            log.info("  %d regions done", len(rows))
    return pd.DataFrame(rows).round(4)


def _draw(ax, sub, color):
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=0)


def make_figure(gdf: gpd.GeoDataFrame, thresholds: list[float], n: int,
                dataset: str, period: str, out: Path,
                bounds: dict | None = None, demand_label: str = "consumption") -> None:
    nrow, ncol = len(thresholds), len(K_LEVELS)
    fig = plt.figure(figsize=(4.6 * ncol, 2.6 * nrow + 1.2))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.16, wspace=0.03,
                          left=0.04, right=0.99, top=0.84, bottom=0.02)
    fig.suptitle(f"Solar+wind sustainable regions vs. co-location overbuild k\n"
                 f"(optimal-mix storage, {dataset} {period}; storage as % of annual "
                 f"{demand_label})", fontsize=14, weight="bold", y=0.985)
    fig.legend(handles=[Patch(facecolor=FEASIBLE_C, label="sustainable (≤ threshold)"),
                        Patch(facecolor=FAIL_C, label="above threshold"),
                        Patch(facecolor=NODATA_C, edgecolor="0.6", label="no data")],
               loc="center", ncol=3, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.90))

    for ri, t in enumerate(thresholds):
        for ci, k in enumerate(K_LEVELS):
            ax = fig.add_subplot(gs[ri, ci])
            col = kcol(k)
            metric = gdf[col]
            valid = metric.notna()
            mask = metric <= t
            _draw(ax, gdf[~valid], NODATA_C)
            _draw(ax, gdf[valid & mask], FEASIBLE_C)
            _draw(ax, gdf[valid & ~mask], FAIL_C)
            cnt = int((valid & mask).sum())
            klab = "no co-location" if k == 0 else f"{int(round(k*100))}% co-location"
            if ri == 0:
                ax.set_title(f"{klab}\n≤{t:g}%: {cnt}/{n} ({100*cnt/n:.0f}%)", fontsize=9)
            else:
                ax.set_title(f"≤{t:g}%: {cnt}/{n} ({100*cnt/n:.0f}%)", fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"threshold {t:g}%", fontsize=10)
            if bounds:
                ax.set_xlim(*bounds["xlim"]); ax.set_ylim(*bounds["ylim"])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    qc.report(fig=fig, name=out.name)
    plt.close(fig)
    qc.report(png_path=out, name=out.name)
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=Path(__file__).resolve().parents[1] / "hpc" / "config_oscer.yaml")
    ap.add_argument("--thresholds", type=float, nargs=3, default=None,
                    help="three storage thresholds (%% of consumption); default = data percentiles")
    ap.add_argument("--no-recompute", action="store_true",
                    help="reuse results/colocation_storage.csv instead of re-running the deficit calc")
    ap.add_argument("--summary", default="summary.csv",
                    help="per-region summary CSV in results_dir providing mix_alpha (USA: usa_summary.csv)")
    ap.add_argument("--monthly-csv", default=None,
                    help="EIA monthly-demand CSV; if given, demand profile is set to monthly (USA case)")
    ap.add_argument("--conus", action="store_true",
                    help="frame the maps on CONUS (for the USA / NLDAS run)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.monthly_csv:                         # USA: real EIA monthly demand
        cfg["demand"]["profile"] = "monthly"
        cfg["demand"]["monthly_csv"] = str(args.monthly_csv)
    results_dir, figdir = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]
    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    summary = pd.read_csv(results_dir / args.summary)

    csv = results_dir / "colocation_storage.csv"
    if args.no_recompute and csv.exists():
        data = pd.read_csv(csv)
        log.info("reusing %s (%d regions)", csv.name, len(data))
    else:
        data = recompute(cfg, regions, summary)
        data.to_csv(csv, index=False)
        log.info("wrote %s (%d regions)", csv.name, len(data))

    n = len(data)
    # Basis + FIXED thresholds, identical to the existing feasibility maps and
    # constant across all k (NOT data-derived): the USA / EIA run uses
    # % of annual CONSUMPTION at 8/12/15; the world run uses % of annual
    # PRODUCTION at 2/7/10.
    basis = "consumption" if args.monthly_csv else "production"
    suffix = "cons" if basis == "consumption" else "prod"
    thresholds = list(args.thresholds) if args.thresholds else \
        ([8.0, 12.0, 15.0] if basis == "consumption" else [2.0, 7.0, 10.0])
    log.info("basis = %% of annual %s | fixed thresholds %s", basis, thresholds)
    for k in K_LEVELS:
        c = data[f"{kcol(k)}_{suffix}"]
        log.info("  k=%.2f: %s", k,
                 ", ".join(f"≤{t:g}%:{int((c <= t).sum())}" for t in thresholds))

    short = cfg["gldas"]["short_name"]
    dataset = "NLDAS" if "NLDAS" in short else "GLDAS" if "GLDAS" in short else short
    period = f"{cfg['period']['start'][:4]}-{cfg['period']['end'][:4]}"
    # rename the basis-appropriate columns to the bare kcol() the figure expects
    rename = {f"{kcol(k)}_{suffix}": kcol(k) for k in K_LEVELS}
    sub = data[["region_id"] + list(rename)].rename(columns=rename)
    gdf = regions.merge(sub, on="region_id", how="left")
    bounds = {"xlim": (-126, -66), "ylim": (23, 50)} if args.conus else None
    make_figure(gdf, thresholds, n, dataset, period,
                figdir / "map_colocation_feasibility.png", bounds=bounds,
                demand_label=basis)


if __name__ == "__main__":
    main()
