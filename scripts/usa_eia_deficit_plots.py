"""Deficit/storage plots + months-unmet table for the 51 US states (EIA demand).

For every US state (50 + DC) this reproduces the paper's deficit figure family
(paper Fig. 6 / 9-12) -- the same PDFs already produced globally by
``gldas_storage.figures`` -- but driven by each state's REAL observed monthly
electricity consumption (EIA retail sales, all sectors) as the demand profile
instead of flat demand. Three resources are plotted:

    solar-only, wind-only, and the storage-optimal solar+wind mix.

For each resource two paginated PDFs are written (4x3 panels, one per state, in
the same style as the world files), with a cover/legend page:

    figures/deficit_{solar,wind,mix}_USA_states_supply.pdf   (supply vs demand)
    figures/deficit_{solar,wind,mix}_USA_states_storage.pdf  (deficit + storage)

It also writes a stand-alone table counting, for every state, how many calendar
months over 2000-2025 the solar+wind generation falls short of demand:

    results/usa_months_unmet.csv

Months-unmet definition: the solar+wind system is sized so total generation over
the period equals total consumption (mean-normalized, no storage and no
overbuild), aggregated to each of the 312 calendar months; a month is "unmet"
when that month's mean generation is below its mean demand. The storage-optimal
solar share (mix_alpha from results/usa_summary.csv) sets the solar/wind blend.
Solar-only and wind-only counts (same mean-matched sizing) are reported too.

Usage:
    python scripts/usa_eia_deficit_plots.py [--config hpc/config_oscer.yaml]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import analyze, deficit, figures, process
from gldas_storage.config import load_config

# reuse the EIA monthly-demand builder so this script is self-sufficient
from usa_eia_feasibility import build_monthly_demand_csv

log = logging.getLogger(__name__)

GROUP = "USA_states"          # filename tag, mirrors the world groups
PERIOD = (2000, 2025)


def mix_series(cf_df: pd.DataFrame, alpha: float) -> np.ndarray:
    """Mean-normalized solar+wind blend: alpha*S/mean(S) + (1-alpha)*W/mean(W).

    Matches metrics.mix_sweep; the result has mean ~1 (it is the unit-mean,
    no-overbuild sizing where annual generation equals annual consumption)."""
    s = cf_df["solar"].to_numpy(float)
    w = cf_df["wind"].to_numpy(float)
    return alpha * (s / s.mean()) + (1.0 - alpha) * (w / w.mean())


def months_unmet(supply: np.ndarray, demand: np.ndarray,
                 dates: pd.DatetimeIndex) -> tuple[int, int, float]:
    """(#months supply<demand, total months, worst monthly shortfall %).

    supply and demand are both mean-normalized (mean 1). Aggregated to calendar
    months; a month counts as unmet when its mean generation < mean demand."""
    df = pd.DataFrame({"supply": supply, "demand": demand}, index=dates)
    m = df.resample("MS").mean()
    short = m["demand"] - m["supply"]
    n_unmet = int((short > 0).sum())
    worst = float((short / m["demand"]).max() * 100.0)
    return n_unmet, int(len(m)), worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=Path(__file__).resolve().parents[1] / "hpc" / "config_oscer.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    results_dir, figdir = cfg["paths"]["results_dir"], cfg["paths"]["figures_dir"]

    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    us = regions[(regions["country"] == "United States of America") &
                 (regions["level"] == "admin1")].copy().sort_values("name")
    # drop states with no grid coverage (e.g. AK/HI on the CONUS-only NLDAS grid)
    if "weight_sum" in us.columns:
        us = us[us["weight_sum"] > 0].copy()
    us_ids = sorted(int(x) for x in us["region_id"])
    log.info("US states: %d", len(us_ids))

    # demand: real EIA monthly consumption climatology (same file the
    # feasibility run uses; rebuild if absent so the script stands alone)
    monthly_csv = results_dir / "usa_eia_monthly_demand.csv"
    if not monthly_csv.exists():
        build_monthly_demand_csv(us, monthly_csv, PERIOD[0], PERIOD[1])
    cfg["demand"]["profile"] = "monthly"
    cfg["demand"]["monthly_csv"] = str(monthly_csv)

    # storage-optimal solar share per state, from the feasibility summary
    summary = pd.read_csv(results_dir / "usa_summary.csv")
    alpha_by_id = dict(zip(summary["region_id"].astype(int), summary["mix_alpha"]))

    # load only the US slice of the zonal table
    table = process.load_zonal(cfg)
    table = table[table["region_id"].isin(us_ids)]
    by_region = {int(rid): grp for rid, grp in table.groupby("region_id")}

    items = {"solar": [], "wind": [], "mix": []}
    unmet_rows = []
    for _, r in us.iterrows():
        rid = int(r["region_id"])
        if rid not in by_region:
            log.warning("no zonal data for %s", r["name"]); continue
        name, alpha = r["name"], float(alpha_by_id.get(rid, np.nan))
        cf_df = analyze.capacity_factors(by_region[rid], cfg)
        demand = deficit.make_demand(cfg, cf_df.index, tair=cf_df["tair"].to_numpy(),
                                     monthly=analyze.monthly_demand_profile(cfg, rid))

        # per-resource deficit + storage simulation for the plots
        series = {"mix": mix_series(cf_df, alpha)}
        for src in ("solar", "wind"):
            series[src] = cf_df[src].to_numpy(float)
        for src in ("solar", "wind", "mix"):
            res = deficit.analyze_region(series[src], cf_df.index, demand,
                                         cfg["storage"], analyze.steps_per_year(cfg),
                                         simulate=True)
            if res.series is not None:
                items[src].append((name, res))

        # months-unmet table: mean-matched (unit-mean) sizing, no storage
        row = {"name": name, "mix_alpha": round(alpha, 3)}
        for src in ("mix", "solar", "wind"):
            sup = series[src] / series[src].mean()    # enforce mean 1 exactly
            n_unmet, n_tot, worst = months_unmet(sup, demand, cf_df.index)
            tag = "mix" if src == "mix" else src
            row[f"{tag}_months_unmet"] = n_unmet
            if src == "mix":
                row["n_months_total"] = n_tot
                row["mix_pct_months_unmet"] = round(100.0 * n_unmet / n_tot, 1)
                row["mix_worst_month_shortfall_pct"] = round(worst, 1)
        unmet_rows.append(row)
        log.info("processed %s (alpha=%.2f, mix unmet=%d)", name, alpha,
                 row["mix_months_unmet"])

    # ---- deficit/storage PDFs (solar, wind, optimal mix) ----
    for src in ("solar", "wind", "mix"):
        figures.deficit_pdf(figdir / f"deficit_{src}_{GROUP}.pdf",
                            items[src], src, GROUP)
        log.info("wrote deficit_%s_%s_{supply,storage}.pdf (%d states)",
                 src, GROUP, len(items[src]))

    # ---- months-unmet table ----
    cols = ["name", "mix_alpha", "n_months_total", "mix_months_unmet",
            "mix_pct_months_unmet", "mix_worst_month_shortfall_pct",
            "solar_months_unmet", "wind_months_unmet"]
    out = (pd.DataFrame(unmet_rows)[cols]
           .sort_values("mix_months_unmet", ascending=False)
           .reset_index(drop=True))
    out.insert(0, "rank", out.index + 1)
    csv_path = results_dir / "usa_months_unmet.csv"
    out.to_csv(csv_path, index=False)
    log.info("wrote %s (%d states)", csv_path, len(out))


if __name__ == "__main__":
    main()
