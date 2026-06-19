"""USA storage/feasibility analysis in REAL UNITS (TWh) driven by EIA consumption.

Runs the deficit/storage framework for the lower-48 states (+DC) using each state's
observed monthly EIA electricity consumption as demand, then expresses everything in
real energy (TWh) and applies NREL's ~1% land constraint (cap-at-1% rule; wind counted
at DIRECT-footprint density so turbine spacing stays farmable). See gldas_storage.landuse.

Feasibility criterion = "meets demand within <=1% of state land" (physical, no fractions).

Outputs (results_dir):
  usa_eia_monthly_demand.csv  -- per-region 12-month demand profile (MWh);
  usa_summary.csv             -- per-state metrics incl. real-units + land columns;
  usa_storage_ranking.csv     -- states ranked by mix storage need (TWh);
  figures/map_usa_feasibility.png -- 6-panel real-units CONUS maps.

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
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import analyze, landuse, process, qc
from gldas_storage.config import load_config

log = logging.getLogger(__name__)

EIA_TSV = Path("/ourdisk/hpc/caps/mchen15/gldas_analysis/EIA/"
               "Consumption_Megawatthours_1990_2025.tsv")
CONUS = dict(xlim=(-126, -66), ylim=(23, 50))
# one color per land-cap tier (tightest -> loosest), then red for "needs more than
# the loosest cap". Aligned positionally with landuse.LAND_CAPS.
TIER_COLORS = ["#1a9850", "#a6d96a", "#fee08b"]
FAIL_C, NODATA_C = "#d73027", "#f7f7f7"

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
    """Per-region 12-month climatological consumption (region_id, month, value MWh)."""
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


def annual_consumption_twh(monthly_csv: Path) -> dict:
    """{region_id -> annual climatological consumption, TWh} from the monthly CSV."""
    m = pd.read_csv(monthly_csv)
    ann = m.groupby("region_id")["value"].sum() / 1e6      # MWh -> TWh
    return {int(k): float(v) for k, v in ann.items()}


def run_analysis(cfg: dict, regions: gpd.GeoDataFrame, us_ids: list[int]) -> pd.DataFrame:
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
    return pd.DataFrame(rows).round(6)


def add_real_units(summary: pd.DataFrame, ann_twh: dict, areas: dict) -> pd.DataFrame:
    """Attach real-units + land-constraint columns for the optimal mix.

    Cap-independent quantities (capacity, land footprint) are computed once; the
    feasibility / storage / shortfall are computed for EACH land-cap scenario in
    landuse.LAND_CAPS (1% and 1.88% = 2x onshore oil & gas), with per-cap suffixes.
    """
    out = []
    for _, r in summary.iterrows():
        cons = ann_twh.get(int(r["region_id"]), float("nan"))
        area = areas.get(r["name"], float("nan"))
        common = dict(alpha=float(r["mix_alpha"]), f_adj=float(r["mix_f_adj"]),
                      solar_cf=float(r["solar_cf"]), wind_cf=float(r["wind_cf"]),
                      annual_consumption_TWh=cons, state_area_km2=area,
                      s_tot_fraction=float(r["mix_s_tot_pct"]) / 100.0)
        base = landuse.assess(**common)                  # land footprint is cap-independent
        # single-resource demand-meeting footprints (alpha=1 -> solar only, alpha=0 ->
        # wind only); same landuse.assess source so the deficit plots can cap them too.
        solar_only = landuse.assess(alpha=1.0, f_adj=float(r["solar_f_adj"]),
                                    solar_cf=float(r["solar_cf"]), wind_cf=float(r["wind_cf"]),
                                    annual_consumption_TWh=cons, state_area_km2=area,
                                    s_tot_fraction=float(r["solar_s_tot_pct"]) / 100.0)
        wind_only = landuse.assess(alpha=0.0, f_adj=float(r["wind_f_adj"]),
                                   solar_cf=float(r["solar_cf"]), wind_cf=float(r["wind_cf"]),
                                   annual_consumption_TWh=cons, state_area_km2=area,
                                   s_tot_fraction=float(r["wind_s_tot_pct"]) / 100.0)
        rec = {
            "annual_consumption_TWh": cons,
            "state_land_km2": area,
            "mix_solar_nameplate_GW": base.solar_nameplate_GW,
            "mix_wind_nameplate_GW": base.wind_nameplate_GW,
            "mix_capacity_TWh": (base.solar_nameplate_GW + base.wind_nameplate_GW)
                                * landuse.HOURS_PER_YEAR / 1e3,
            "mix_land_km2": base.land_km2,
            "mix_land_pct": base.land_pct,
            "solar_land_km2": solar_only.land_km2,
            "solar_land_pct": solar_only.land_pct,
            "wind_land_km2": wind_only.land_km2,
            "wind_land_pct": wind_only.land_pct,
        }
        for key, frac, _ in landuse.LAND_CAPS:
            res = landuse.assess(**common, land_fraction_cap=frac)
            rec[f"feasible_{key}"] = res.feasible
            rec[f"mix_storage_TWh_{key}"] = res.storage_TWh
            rec[f"mix_supply_capped_TWh_{key}"] = res.supply_capped_TWh
            rec[f"mix_shortfall_TWh_{key}"] = res.shortfall_TWh
        out.append(rec)
    add = pd.DataFrame(out)
    base = summary.drop(columns=[c for c in add.columns if c in summary.columns]
                        ).reset_index(drop=True)         # idempotent on re-runs
    return pd.concat([base, add], axis=1)


def _draw(ax, sub, color):
    if len(sub):
        sub.plot(ax=ax, color=color, linewidth=0.2, edgecolor="white")


def _tier_map(ax, gdf, title):
    """Color each state by the TIGHTEST land cap that makes it feasible.

    Tier i = feasible only once the cap is loosened to LAND_CAPS[i]; states still
    infeasible at the loosest cap are red. Generalizes to any number of caps."""
    keys = [k for k, _, _ in landuse.LAND_CAPS]
    valid = gdf[f"feasible_{keys[0]}"].notna()
    _draw(ax, gdf[~valid], NODATA_C)
    assigned = valid & False
    for key, color in zip(keys, TIER_COLORS):
        feas = gdf[f"feasible_{key}"].fillna(False)
        _draw(ax, gdf[valid & feas & ~assigned], color)
        assigned = assigned | (valid & feas)
    _draw(ax, gdf[valid & ~assigned], FAIL_C)       # needs more than the loosest cap
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*CONUS["xlim"]); ax.set_ylim(*CONUS["ylim"]); ax.set_axis_off()


def _cont_map(ax, gdf, col, title, label, cmap="viridis", vmax=None, q=0.95):
    valid = gdf[gdf[col].notna()]
    if vmax is None:
        hi = valid[col].quantile(q) if len(valid) else 1.0
        vmax = max(1.0, math.ceil(hi))
    gdf.plot(column=col, ax=ax, cmap=cmap, vmin=0.0, vmax=vmax, legend=True,
             missing_kwds={"color": NODATA_C}, linewidth=0.2, edgecolor="white",
             legend_kwds={"shrink": 0.55, "extend": "max", "label": label})
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*CONUS["xlim"]); ax.set_ylim(*CONUS["ylim"]); ax.set_axis_off()


def make_figure(gdf: gpd.GeoDataFrame, counts: list, n: int,
                out: Path, dataset: str, period: str) -> None:
    # counts: list of (capkey, label, frac, n_feas) tightest -> loosest
    caplabels = " vs ".join(f"{c[2]*100:g}%" for c in counts)
    progression = " → ".join(f"{c[3]} @{c[2]*100:g}%" for c in counts)
    fig = plt.figure(figsize=(18, 7.8))
    gs = fig.add_gridspec(2, 3, hspace=0.16, wspace=0.02,
                          left=0.01, right=0.97, top=0.86, bottom=0.04)
    fig.suptitle(f"USA solar+wind feasibility in real units, land limit {caplabels}\n"
                 f"(optimal mix, {dataset} {period}, real EIA consumption)",
                 fontsize=13, weight="bold", y=0.99)

    # --- upper row: (a) tier map | legend | (b) land used ---
    _tier_map(fig.add_subplot(gs[0, 0]), gdf,
              f"(a) Feasibility by land cap\n{progression} feasible")

    axleg = fig.add_subplot(gs[0, 1]); axleg.set_axis_off()
    handles = [Patch(facecolor=col, label=f"feasible at ≤{c[2]*100:g}% land"
                     + (" (gained)" if i else ""))
               for i, (c, col) in enumerate(zip(counts, TIER_COLORS))]
    handles += [Patch(facecolor=FAIL_C, label=f"needs >{counts[-1][2]*100:g}% land"),
                Patch(facecolor=NODATA_C, edgecolor="0.6", label="no data")]
    axleg.legend(handles=handles, loc="center", fontsize=12, frameon=True)

    _cont_map(fig.add_subplot(gs[0, 2]), gdf, "mix_land_pct",
              "(b) Land used by optimal mix\n% of state land",
              "% of state land", cmap="magma_r", vmax=2.0)

    # --- lower row: storage requirement (TWh) at each land cap, shared color scale ---
    stor_cols = [f"mix_storage_TWh_{c[0]}" for c in counts]
    allv = pd.concat([gdf[c] for c in stor_cols]).dropna()
    vmax = max(1.0, math.ceil(allv.quantile(0.95))) if len(allv) else 1.0
    for i, (col, c) in enumerate(zip(stor_cols, counts)):
        _cont_map(fig.add_subplot(gs[1, i]), gdf, col,
                  f"({'cde'[i]}) Storage need @{c[2]*100:g}% landuse\nTWh",
                  "TWh", vmax=vmax)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    qc.report(fig=fig, name=out.name)
    plt.close(fig)
    qc.report(png_path=out, name=out.name)
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
    results_dir, figdir = Path(cfg["paths"]["results_dir"]), Path(cfg["paths"]["figures_dir"])
    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    us = regions[(regions["country"] == "United States of America") &
                 (regions["level"] == "admin1")].copy()
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
    ann_twh = annual_consumption_twh(monthly_csv)
    areas = landuse.state_areas_km2(us)

    summary_path = results_dir / "usa_summary.csv"
    if args.no_analysis and summary_path.exists():
        summary = pd.read_csv(summary_path)
    else:
        cfg["demand"]["profile"] = "monthly"
        cfg["demand"]["monthly_csv"] = str(monthly_csv)
        summary = run_analysis(cfg, regions, us_ids)

    summary = add_real_units(summary, ann_twh, areas)
    summary.to_csv(summary_path, index=False)
    log.info("wrote %s", summary_path)

    # states whose supply CANNOT meet demand at all within 1% land (annual energy
    # deficit -- storage cannot rescue an energy shortfall). Surface them, and note
    # the TIGHTEST larger cap (if any) that recovers each (shortfall -> 0).
    caps = landuse.LAND_CAPS                              # tightest -> loosest
    short = summary[summary["mix_shortfall_TWh_1pct"] > 0].sort_values(
        "mix_shortfall_TWh_1pct", ascending=False)
    cols = ["name", "annual_consumption_TWh", "mix_land_pct"] + \
        [f"mix_shortfall_TWh_{k}" for k, _, _ in caps]
    short[cols].round(3).to_csv(results_dir / "usa_cannot_meet_demand.csv", index=False)

    def recovering_cap(row):
        for k, frac, _ in caps:
            if row[f"mix_shortfall_TWh_{k}"] <= 0:
                return f"RECOVERED@{frac*100:g}%"
        return f"needs >{caps[-1][1]*100:g}%"

    if len(short):
        rec_counts = {f"{frac*100:g}%": int((short[f"mix_shortfall_TWh_{k}"] <= 0).sum())
                      for k, frac, _ in caps[1:]}
        log.warning("%d states CANNOT meet demand within 1%% land (annual deficit); "
                    "larger caps recover: %s", len(short),
                    ", ".join(f"{v}@{c}" for c, v in rec_counts.items()))
        for _, r in short.iterrows():
            log.warning("   %-20s short %6.1f TWh/yr @1%% (needs %.2f%% land) -> %s",
                        r["name"], r["mix_shortfall_TWh_1pct"], r["mix_land_pct"],
                        recovering_cap(r))
    n_short = len(short)

    # ranking by mix storage need (TWh)
    rank = summary.sort_values("mix_storage_TWh_1pct").reset_index(drop=True)
    rank.insert(0, "rank", rank.index + 1)
    rank[["rank", "name", "mix_alpha", "annual_consumption_TWh", "mix_storage_TWh_1pct",
          "mix_land_pct", "feasible_1pct", "feasible_oilgas2x", "mix_shortfall_TWh_1pct"]
         ].round(4).to_csv(results_dir / "usa_storage_ranking.csv", index=False)

    gdf = us.merge(summary, on=["region_id", "name"], how="left")
    n = int(summary["feasible_1pct"].notna().sum())
    counts = [(key, label, frac,
               int((summary[f"feasible_{key}"] == True).sum()))   # noqa: E712
              for key, frac, label in landuse.LAND_CAPS]
    dataset = str(cfg["gldas"]["short_name"]).split("_")[0]
    period = f"{str(cfg['period']['start'])[:4]}-{str(cfg['period']['end'])[:4]}"
    make_figure(gdf, counts, n,
                figdir / "map_usa_feasibility.png", dataset, period)
    log.info("feasible: %s (of %d); cannot meet @1%%: %d",
             ", ".join(f"{c[3]}@{c[2]*100:g}%" for c in counts), n, n_short)


if __name__ == "__main__":
    main()
