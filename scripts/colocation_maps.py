"""Co-location feasibility maps: how the optimal-mix storage requirement shrinks
when solar and wind share the same land via a co-located overlap area.

The baseline land is already split into the storage-optimal shares alpha* (solar)
and 1 - alpha* (wind); alpha* is optimized ONCE for the baseline and is fixed for
life (reused from results/summary.csv). Co-location adds an overlap area k on top
of that fixed split. Because the infrastructure is built once, the overlap cannot
be divided: the WHOLE of k is dedicated either to solar or to wind. Which one is a
second, BINARY optimization per region -- pick the assignment that minimizes
storage:

    give k to solar:  supply = min(alpha + k, 1) * sol + (1 - alpha)      * win
    give k to wind:   supply =      alpha        * sol + min(1-alpha+k, 1)* win
    (sol, win mean-normalized; total mean = 1 + k either way -- k counted ONCE)

The winning (lower-storage) assignment is kept. Its overlapped series is analysed
at its REAL magnitude (mean 1 + k, referenced to the baseline mix mean of 1 so the
extra production survives as surplus rather than being normalized away); the
excess-installation factor f is re-derived from that series -- k never multiplies
net_factor. k in {0, 0.17, 0.35, 0.50} (k = 0 = baseline, no overlap).

Storage is reported on FIXED, k-independent denominators: % of annual
CONSUMPTION (s_tot) for the USA/NLDAS run, and % of baseline (k=0) annual
PRODUCTION (s_tot / f_adj at k=0) for the world/GLDAS run.

Outputs:
  results/colocation_storage.csv          -- per region, the min storage at each k
      plus which technology won the overlap (kN_pick);
  figures/map_colocation_feasibility.png  -- WORLD storage-threshold maps (default
      path): binary "sustainable (<= threshold)", one column per overlap level
      (baseline + 17/35/50 %), one row per storage threshold. For k > 0 a region is
      colored by the overlap assignment (wind vs solar); k = 0 uses a baseline color.
  figures/map_usa_colocation_feasibility.png -- USA land-feasibility maps written by
      --land-feasibility (distinct filename so it never clobbers the world figure):
      grid of rows = land cap (1/1.88/5 %), cols = overlap k.

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
from gldas_storage import analyze, deficit, landuse, process, qc
from gldas_storage.config import load_config

log = logging.getLogger(__name__)

K_LEVELS = [0.0, 0.17, 0.35, 0.50]      # 0 = baseline; 17/35/50 % overlap area
BASE_C, WIND_C, SOLAR_C = "#1a9850", "#4575b4", "#f1a340"   # k=0 / overlap→wind / →solar
FAIL_C, NODATA_C = "#d9d9d9", "#f7f7f7"


def kcol(k: float) -> str:
    return "k0" if k == 0 else f"k{int(round(k*100))}"


def recompute(cfg: dict, regions: gpd.GeoDataFrame, summary: pd.DataFrame) -> pd.DataFrame:
    """Per-region optimal-mix S_tot at each co-location overlap level k.

    For each k > 0 the WHOLE overlap area goes either to solar (share alpha + k)
    or to wind (share 1 - alpha + k), each capped at full land; the assignment
    that needs less storage is kept (a binary choice -- the overlap can't be
    split). The chosen overlapped supply is analysed at its real magnitude
    (norm_mean=1, the baseline mix mean). Storage is reported on two FIXED
    denominators: % of annual consumption (s_tot) and % of baseline (k=0) annual
    production (s_tot divided by the k=0 loss-adjusted install factor f_adj). The
    winning technology is recorded in kN_pick.
    """
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
        ns, nw = s / s.mean(), w / w.mean()      # mean-normalized (each mean 1)
        demand = deficit.make_demand(cfg, cf_df.index, tair=cf_df["tair"].to_numpy(),
                                     monthly=analyze.monthly_demand_profile(cfg, rid))
        row = {"region_id": rid, "name": meta.loc[rid, "name"], "mix_alpha": alpha}
        base_factor = None                       # f_adj at k=0 -> fixed prod denom
        for k in K_LEVELS:
            if k == 0.0:                         # baseline: no overlap to assign
                chosen = deficit.analyze_region(
                    alpha * ns + (1.0 - alpha) * nw, cf_df.index, demand,
                    storage_cfg, spy, simulate=False, norm_mean=1.0)
                base_factor = chosen.tot_factor
                pick = "base"
            else:
                # BINARY choice: the whole overlap k goes to solar OR to wind
                # (built once, can't be split); keep the lower-storage assignment
                res_s = deficit.analyze_region(
                    min(alpha + k, 1.0) * ns + (1.0 - alpha) * nw, cf_df.index,
                    demand, storage_cfg, spy, simulate=False, norm_mean=1.0)
                res_w = deficit.analyze_region(
                    alpha * ns + min((1.0 - alpha) + k, 1.0) * nw, cf_df.index,
                    demand, storage_cfg, spy, simulate=False, norm_mean=1.0)
                chosen, pick = ((res_s, "solar") if res_s.s_tot <= res_w.s_tot
                                else (res_w, "wind"))
            # store BOTH bases on FIXED denominators plus the winning assignment:
            # % of annual consumption (s_tot) and % of baseline (k=0) production
            row[f"{kcol(k)}_cons"] = chosen.s_tot * 100.0
            row[f"{kcol(k)}_prod"] = (chosen.s_tot / base_factor * 100.0
                                      if base_factor and base_factor > 0 else np.nan)
            row[f"{kcol(k)}_pick"] = pick
        rows.append(row)
        if len(rows) % 100 == 0:
            log.info("  %d regions done", len(rows))
    return pd.DataFrame(rows).round(4)


def land_feasibility(summary: pd.DataFrame) -> pd.DataFrame:
    """USA real-units path: 'meets demand within the land cap' at each (cap, k).

    Two variables sweep on a grid: the land-cap scenario (1% NREL, 1.88% = 2x onshore
    oil & gas) and the co-location overlap k. Co-location lets the same footprint host
    (1+k) generation, so the land the demand-meeting build needs shrinks by 1/(1+k):
    land(k)=land(0)/(1+k); a region is feasible when land(k) <= cap. alpha*, f_adj, cf
    and consumption are held fixed. Uses usa_summary.csv real-units columns (no recompute)."""
    rows = []
    for _, r in summary.iterrows():
        row = {"region_id": int(r["region_id"]), "name": r["name"]}
        for capkey, frac, _ in landuse.LAND_CAPS:
            for k in K_LEVELS:
                res = landuse.assess(
                    alpha=float(r["mix_alpha"]), f_adj=float(r["mix_f_adj"]),
                    solar_cf=float(r["solar_cf"]), wind_cf=float(r["wind_cf"]),
                    annual_consumption_TWh=float(r["annual_consumption_TWh"]),
                    state_area_km2=float(r["state_land_km2"]),
                    s_tot_fraction=float(r["mix_s_tot_pct"]) / 100.0,
                    land_overlap_k=k, land_fraction_cap=frac)
                row[f"{capkey}_{kcol(k)}_feasible"] = bool(res.feasible)
        rows.append(row)
    return pd.DataFrame(rows)


def make_land_figure(gdf: gpd.GeoDataFrame, n: int, dataset: str, period: str,
                     out: Path, bounds: dict) -> None:
    """Grid of binary 'feasible within land cap' maps: rows = land cap, cols = overlap k."""
    fail_c = "#d73027"          # red, matches map_usa_feasibility "needs > cap land"
    caps = landuse.LAND_CAPS
    nrow, ncol = len(caps), len(K_LEVELS)
    fig = plt.figure(figsize=(4.6 * ncol, 3.6 * nrow + 1.0))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.12, wspace=0.03,
                          left=0.05, right=0.99, top=0.84, bottom=0.02)
    fig.suptitle("USA solar+wind feasibility vs. land cap and co-location overlap k\n"
                 f"({dataset} {period}, real EIA consumption; co-location packs (1+k)× "
                 "generation into the same footprint)", fontsize=14, weight="bold", y=0.99)
    fig.legend(handles=[Patch(facecolor=BASE_C, label="feasible within land cap"),
                        Patch(facecolor=fail_c, label="needs more land"),
                        Patch(facecolor=NODATA_C, edgecolor="0.6", label="no data")],
               loc="center", ncol=3, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.89))
    for ri, (capkey, _, caplabel) in enumerate(caps):
        base_cnt = int(gdf[f"{capkey}_{kcol(0.0)}_feasible"].fillna(False).sum())
        for ci, k in enumerate(K_LEVELS):
            ax = fig.add_subplot(gs[ri, ci])
            feas = gdf[f"{capkey}_{kcol(k)}_feasible"]
            valid = feas.notna()
            _draw(ax, gdf[~valid], NODATA_C)
            _draw(ax, gdf[valid & ~feas.fillna(False)], fail_c)
            _draw(ax, gdf[valid & feas.fillna(False)], BASE_C)
            cnt = int(feas.fillna(False).sum())
            delta = "" if k == 0 else f"  (+{cnt - base_cnt})"
            cnt_lab = f"{cnt}/{n} ({100*cnt/n:.0f}%){delta}"
            if ri == 0:
                klab = "no co-location" if k == 0 else f"{int(round(k*100))}% co-location"
                ax.set_title(f"{klab}\n{cnt_lab}", fontsize=10)
            else:
                ax.set_title(cnt_lab, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Land cap: {caplabel}", fontsize=11)
            ax.set_xlim(*bounds["xlim"]); ax.set_ylim(*bounds["ylim"])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    qc.report(fig=fig, name=out.name)
    plt.close(fig)
    qc.report(png_path=out, name=out.name)
    log.info("wrote %s", out)


def _draw(ax, sub, color):
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=0)


def make_figure(gdf: gpd.GeoDataFrame, thresholds: list[float], n: int,
                dataset: str, period: str, out: Path,
                bounds: dict | None = None, demand_label: str = "consumption") -> None:
    nrow, ncol = len(thresholds), len(K_LEVELS)
    fig = plt.figure(figsize=(4.6 * ncol, 3.1 * nrow + 1.2))
    gs = fig.add_gridspec(nrow, ncol, hspace=0.14, wspace=0.03,
                          left=0.05, right=0.99, top=0.86, bottom=0.02)
    fig.suptitle(f"Solar+wind sustainable regions vs. co-location overlap area k\n"
                 f"(optimal-mix storage, {dataset} {period}; storage as % of annual "
                 f"{demand_label})", fontsize=14, weight="bold", y=0.985)
    fig.legend(handles=[Patch(facecolor=BASE_C, label="sustainable — no overlap"),
                        Patch(facecolor=WIND_C, label="sustainable — overlap → wind"),
                        Patch(facecolor=SOLAR_C, label="sustainable — overlap → solar"),
                        Patch(facecolor=FAIL_C, label="above threshold"),
                        Patch(facecolor=NODATA_C, edgecolor="0.6", label="no data")],
               loc="center", ncol=5, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, 0.90))

    for ri, t in enumerate(thresholds):
        for ci, k in enumerate(K_LEVELS):
            ax = fig.add_subplot(gs[ri, ci])
            col = kcol(k)
            metric = gdf[col]
            valid = metric.notna()
            sustain = valid & (metric <= t)
            _draw(ax, gdf[~valid], NODATA_C)
            _draw(ax, gdf[valid & (metric > t)], FAIL_C)
            if k == 0:                            # baseline: single color
                _draw(ax, gdf[sustain], BASE_C)
            else:                                 # color by overlap assignment
                pick = gdf[f"{col}_pick"]
                _draw(ax, gdf[sustain & (pick == "wind")], WIND_C)
                _draw(ax, gdf[sustain & (pick == "solar")], SOLAR_C)
            cnt = int(sustain.sum())
            klab = "no co-location" if k == 0 else f"{int(round(k*100))}% co-location"
            cnt_lab = f"{cnt}/{n} ({100*cnt/n:.0f}%)"   # sustainable / total regions
            # top row carries the co-location-level header above the count; every
            # row carries the count (the threshold itself is on the y-axis label)
            if ri == 0:
                ax.set_title(f"{klab}\n{cnt_lab}", fontsize=10)
            else:
                ax.set_title(cnt_lab, fontsize=9)
            if ci == 0:
                ax.set_ylabel(f"Store {t:g}% of annual {demand_label}", fontsize=11)
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
    ap.add_argument("--land-feasibility", action="store_true",
                    help="USA real-units mode: map 'meets demand within <=1%% land' at "
                         "each co-location k (uses usa_summary.csv real-units columns)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    if args.monthly_csv:                         # USA: real EIA monthly demand
        cfg["demand"]["profile"] = "monthly"
        cfg["demand"]["monthly_csv"] = str(args.monthly_csv)
    results_dir, figdir = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]
    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    summary = pd.read_csv(results_dir / args.summary)

    short = cfg["gldas"]["short_name"]
    dataset = "NLDAS" if "NLDAS" in short else "GLDAS" if "GLDAS" in short else short
    period = f"{cfg['period']['start'][:4]}-{cfg['period']['end'][:4]}"

    # ---- USA real-units land-feasibility path (no storage thresholds) ----
    if args.land_feasibility:
        land = land_feasibility(summary)
        land.to_csv(results_dir / "colocation_land_feasibility.csv", index=False)
        n = len(land)
        for capkey, frac, caplabel in landuse.LAND_CAPS:
            cnts = ", ".join(f"k={k:.2f}:{int(land[f'{capkey}_{kcol(k)}_feasible'].sum())}"
                             for k in K_LEVELS)
            log.info("  cap %s (%.2f%%): %s / %d states", caplabel, frac * 100, cnts, n)
        gdf = regions.merge(land, on=["region_id", "name"], how="right")
        bounds = {"xlim": (-126, -66), "ylim": (23, 50)}
        # distinct filename -- must NOT clobber the world map_colocation_feasibility.png
        make_land_figure(gdf, n, dataset, period,
                         figdir / "map_usa_colocation_feasibility.png", bounds)
        return

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

    # rename the basis-appropriate columns to the bare kcol() the figure expects
    rename = {f"{kcol(k)}_{suffix}": kcol(k) for k in K_LEVELS}
    pickcols = [f"{kcol(k)}_pick" for k in K_LEVELS]
    sub = data[["region_id"] + list(rename) + pickcols].rename(columns=rename)
    gdf = regions.merge(sub, on="region_id", how="left")
    bounds = {"xlim": (-126, -66), "ylim": (23, 50)} if args.conus else None
    make_figure(gdf, thresholds, n, dataset, period,
                figdir / "map_colocation_feasibility.png", bounds=bounds,
                demand_label=basis)


if __name__ == "__main__":
    main()
