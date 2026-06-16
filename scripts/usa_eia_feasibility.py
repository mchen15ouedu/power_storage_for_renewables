"""USA-only storage/feasibility analysis driven by REAL EIA state consumption.

Re-runs the GLDAS deficit/storage framework for the 51 US states (50 + DC),
but instead of flat demand it uses each state's observed monthly electricity
consumption (EIA, retail sales all sectors) as the seasonal demand profile.
Outputs:
  results/usa_eia_monthly_demand.csv   -- per-region 12-month demand profile;
  results/usa_summary.csv              -- per-state metrics (mix/solar/wind);
  results/usa_storage_ranking.csv      -- states ranked by mix storage need;
  figures/map_usa_feasibility.png      -- feasibility panels, CONUS view.

Demand profile = climatological monthly mean consumption over the GLDAS period
(2000-2025); the framework normalizes it to mean 1, so only the seasonal SHAPE
matters (summer-AC vs winter-heating states differ).

Usage: python scripts/usa_eia_feasibility.py [--config hpc/config_oscer.yaml]
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
from gldas_storage import analyze, process, qc
from gldas_storage.config import load_config

log = logging.getLogger(__name__)

EIA_TSV = Path("/ourdisk/hpc/caps/mchen15/gldas_analysis/EIA/"
               "Consumption_Megawatthours_1990_2025.tsv")
THRESHOLDS = [8.0, 12.0, 15.0]             # % of annual consumption
CONUS = dict(xlim=(-126, -66), ylim=(23, 50))   # lower-48 view
FEASIBLE_C, FAIL_C, NODATA_C = "#1a9850", "#d9d9d9", "#f7f7f7"

NAME2ABBR = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "District of Columbia": "DC", "Florida": "FL", "Georgia": "GA", "Hawaii": "HI",
    "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI",
    "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX",
    "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
}


def build_monthly_demand_csv(us_regions: pd.DataFrame, out_csv: Path,
                             year0: int, year1: int) -> None:
    """Per-region 12-month climatological consumption (region_id, month, value)."""
    out_csv = Path(out_csv)
    eia = pd.read_csv(EIA_TSV, sep="\t")
    eia.columns = [c.strip().strip('"') for c in eia.columns]
    eia = eia[(eia["Year"] >= year0) & (eia["Year"] <= year1)]
    rows = []
    for _, r in us_regions.iterrows():
        abbr = NAME2ABBR.get(r["name"])
        if abbr is None or abbr not in eia.columns:
            log.warning("no EIA column for %s", r["name"])
            continue
        clim = eia.groupby("Month")[abbr].mean()
        for m, v in clim.items():
            rows.append({"region_id": int(r["region_id"]),
                         "month": int(m), "value": float(v)})
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    log.info("wrote %s (%d states x 12 months)", out_csv.name, len(rows) // 12)


def run_analysis(cfg: dict, regions: gpd.GeoDataFrame,
                 us_ids: list[int]) -> pd.DataFrame:
    table = process.load_zonal(cfg)
    table = table[table["region_id"].isin(us_ids)]
    meta = regions.set_index("region_id")
    rows = []
    for rid, grp in table.groupby("region_id"):
        cf_df = analyze.capacity_factors(grp, cfg)
        row = {"region_id": int(rid), "name": meta.loc[rid, "name"]}
        row.update(analyze.analyze_one(cf_df, cfg, int(rid)))
        rows.append(row)
        log.info("analyzed %s", row["name"])
    summary = pd.DataFrame(rows).round(4)
    summary["mix_s_pct_production"] = summary["mix_s_tot_pct"] / summary["mix_f_adj"]
    summary["solar_s_pct_production"] = summary["solar_s_tot_pct"] / summary["solar_f_adj"]
    summary["wind_s_pct_production"] = summary["wind_s_tot_pct"] / summary["wind_f_adj"]
    return summary


def _draw(ax, sub, color):
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=0.2, edgecolor="white")


def _binary_map(ax, gdf, mask, title):
    valid = gdf["_metric"].notna()
    _draw(ax, gdf[~valid], NODATA_C)
    _draw(ax, gdf[valid & mask], FEASIBLE_C)
    _draw(ax, gdf[valid & ~mask], FAIL_C)
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*CONUS["xlim"]); ax.set_ylim(*CONUS["ylim"])
    ax.set_axis_off()


def make_figure(gdf: gpd.GeoDataFrame, counts: dict, n: int, out: Path,
                dataset: str = "GLDAS", period: str = "2000-2025") -> None:
    # CONUS spans ~60 deg lon x ~27 deg lat (~2.2:1). With geopandas' equal
    # aspect, a panel much taller than that leaves big empty vertical bands
    # (the "too much space" QC flag); too short and the legend collides with
    # the panel titles. So: size the figure so a 3-col x 2-row block of 2.2:1
    # maps fills its cells, and keep the legend OUT of the grid -- as a figure
    # legend in the clear band between the suptitle and the map titles.
    fig = plt.figure(figsize=(18, 7.6))
    gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.02,
                          left=0.01, right=0.97, top=0.78, bottom=0.04)
    fig.suptitle("USA solar+wind sustainable regions with electric storage "
                 f"requirements\n(optimal mix, {dataset} {period}, with real power "
                 "consumption data from EIA)", fontsize=14, weight="bold", y=0.99)
    fig.legend(handles=[Patch(facecolor=FEASIBLE_C, label="sustainable (≤ threshold)"),
                        Patch(facecolor=FAIL_C, label="above storage threshold"),
                        Patch(facecolor=NODATA_C, edgecolor="0.6", label="no data")],
               loc="center", ncol=3, fontsize=11, frameon=True,
               bbox_to_anchor=(0.5, 0.875),
               title="Panels (a)-(c): optimal-mix storage vs. threshold")
    axmap = [fig.add_subplot(gs[r, c]) for r in (0, 1) for c in (0, 1, 2)]

    for k, t in enumerate(THRESHOLDS):
        mask = gdf["_metric"] <= t
        _binary_map(axmap[k], gdf, mask.fillna(False),
                    f"({'abc'[k]}) Mix storage ≤ {t:g}% of consumption\n"
                    f"{counts[t]}/{n} states ({100*counts[t]/n:.0f}%)")

    cont = [("mix_s_tot_pct",   "(d) Optimal-mix storage\n% of annual consumption"),
            ("solar_s_tot_pct", "(e) Solar-only storage\n% of annual consumption"),
            ("wind_s_tot_pct",  "(f) Wind-only storage\n% of annual consumption")]
    for axc, (col, title) in zip(axmap[3:], cont):
        vmax = max(5.0, 5.0 * math.ceil(gdf[col].quantile(0.95) / 5.0))
        gdf.plot(column=col, ax=axc, cmap="viridis", vmin=0.0, vmax=vmax,
                 legend=True, missing_kwds={"color": NODATA_C},
                 linewidth=0.2, edgecolor="white",
                 legend_kwds={"shrink": 0.55, "extend": "max",
                              "label": "% of annual consumption"})
        axc.set_title(title, fontsize=10)
        axc.set_xlim(*CONUS["xlim"]); axc.set_ylim(*CONUS["ylim"])
        axc.set_axis_off()

    fig.savefig(out, dpi=200, bbox_inches="tight")
    qc.report(fig=fig, name=out.name)        # self-check layout (legend/whitespace)
    plt.close(fig)
    qc.report(png_path=out, name=out.name)   # and the saved raster
    log.info("wrote %s", out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=Path(__file__).resolve().parents[1] / "hpc" / "config_oscer.yaml")
    ap.add_argument("--no-analysis", action="store_true",
                    help="skip the heavy re-analysis; reuse results/usa_summary.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    results_dir, figdir = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]
    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    us = regions[(regions["country"] == "United States of America") &
                 (regions["level"] == "admin1")].copy()
    # drop states with no grid coverage (e.g. AK/HI on the CONUS-only NLDAS grid)
    if "weight_sum" in us.columns:
        dropped = us.loc[us["weight_sum"] <= 0, "name"].tolist()
        us = us[us["weight_sum"] > 0].copy()
        if dropped:
            log.info("excluded %d state(s) with no grid coverage: %s",
                     len(dropped), ", ".join(dropped))
    us_ids = sorted(int(x) for x in us["region_id"])
    log.info("US states: %d", len(us_ids))

    monthly_csv = results_dir / "usa_eia_monthly_demand.csv"
    build_monthly_demand_csv(us, monthly_csv, 2000, 2025)

    summary_path = results_dir / "usa_summary.csv"
    if args.no_analysis and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        cfg["demand"]["profile"] = "monthly"
        cfg["demand"]["monthly_csv"] = str(monthly_csv)
        summary = run_analysis(cfg, regions, us_ids)
        summary.to_csv(summary_path, index=False)
        log.info("wrote %s", summary_path)

    # ranking (by optimal-mix storage as % of annual consumption)
    rank = summary.sort_values("mix_s_tot_pct").reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    rank[["rank", "name", "mix_alpha", "mix_s_tot_pct", "solar_s_tot_pct",
          "wind_s_tot_pct", "solar_cf", "wind_cf", "mix_s_pct_production"]
         ].round(4).to_csv(results_dir / "usa_storage_ranking.csv", index=False)

    # figure
    gdf = us.merge(summary[["region_id", "mix_s_tot_pct",
                            "solar_s_tot_pct", "wind_s_tot_pct"]],
                   on="region_id", how="left")
    gdf["_metric"] = gdf["mix_s_tot_pct"]
    n = len(summary)
    counts = {t: int((summary["mix_s_tot_pct"] <= t).sum()) for t in THRESHOLDS}
    dataset = str(cfg["gldas"]["short_name"]).split("_")[0]   # GLDAS_NOAH025_3H -> GLDAS
    period = f"{str(cfg['period']['start'])[:4]}-{str(cfg['period']['end'])[:4]}"
    make_figure(gdf, counts, n, figdir / "map_usa_feasibility.png", dataset, period)


if __name__ == "__main__":
    main()
