"""Stage 2: run all analyses for every region (and pooled continents/world).

Per region and source (solar, wind):
  * base deficit analysis at 3-hourly resolution: cf, f, f_adj, S_net, S_tot;
  * temporal decomposition: S_tot on daily-mean CFs isolates the seasonal tier,
    the 3-hourly excess is the diurnal tier;
  * storage technology archetypes: S_tot re-run per parameter set;
  * dunkelflaute statistics and (optional) annual-CF trends.
Per region: storage-optimal solar+wind mix.
Optionally: the same base metrics for continent + world pools.

Output: results/summary.csv and results/pooled_summary.csv.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from . import deficit, energy, metrics, process

log = logging.getLogger(__name__)

SOURCES = ("solar", "wind")
STEPS_PER_YEAR_3H = 2920   # 365 * 8 (GLDAS 3-hourly default)
STEPS_PER_YEAR_D = 365


def steps_per_year(cfg: dict) -> int:
    """Sub-daily timesteps per year for the configured forcing (3-hourly=2920,
    hourly NLDAS=8760). Falls back to the GLDAS 3-hourly default."""
    return int(cfg.get("gldas", {}).get("steps_per_year", STEPS_PER_YEAR_3H))


def capacity_factors(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """3-hourly capacity-factor DataFrame (solar, wind, + tair) from a zonal slice."""
    return pd.DataFrame({
        "solar": energy.solar_cf(df["swdown"].to_numpy(float), cfg),
        "wind": energy.wind_cf(df["wind10"].to_numpy(float), cfg,
                               tair=df["tair"].to_numpy(float),
                               psurf=df["psurf"].to_numpy(float),
                               qair=df["qair"].to_numpy(float)),
        "tair": df["tair"].to_numpy(float),
    }, index=pd.DatetimeIndex(df["time"]))


def monthly_demand_profile(cfg: dict, region_key) -> np.ndarray | None:
    path = cfg["demand"].get("monthly_csv")
    if cfg["demand"]["profile"] != "monthly" or not path:
        return None
    prof = pd.read_csv(path)
    prof["region_id"] = prof["region_id"].astype(str)
    rows = prof[prof["region_id"] == str(region_key)]
    if rows.empty:
        rows = prof[prof["region_id"].str.upper() == "ALL"]
    return None if rows.empty else rows.sort_values("month")["value"].to_numpy(float)


def analyze_one(cf_df: pd.DataFrame, cfg: dict, region_key) -> dict:
    """All metrics for one region (or pool). cf_df: index time, cols solar/wind/tair."""
    acfg = cfg["analysis"]
    storage_cfg = cfg["storage"]
    spy = steps_per_year(cfg)
    dates = cf_df.index
    demand = deficit.make_demand(cfg, dates, tair=cf_df["tair"].to_numpy(),
                                 monthly=monthly_demand_profile(cfg, region_key))
    daily = cf_df.resample("1D").mean()
    demand_daily = pd.Series(demand, index=dates).resample("1D").mean().to_numpy()

    row = {}
    for source in SOURCES:
        cf = cf_df[source].to_numpy()
        res = deficit.analyze_region(cf, dates, demand, storage_cfg,
                                     spy, simulate=False)
        row.update({
            f"{source}_cf": res.capacity_factor,
            f"{source}_f": res.net_factor,
            f"{source}_f_adj": res.tot_factor,
            f"{source}_s_net_pct": res.s_net * 100.0,
            f"{source}_s_tot_pct": res.s_tot * 100.0,
            f"{source}_converged": res.converged,
        })
        # seasonal tier from daily means of the 3-hourly CFs; diurnal = excess
        res_d = deficit.analyze_region(daily[source].to_numpy(),
                                       daily.index, demand_daily,
                                       storage_cfg, STEPS_PER_YEAR_D, simulate=False)
        row[f"{source}_s_seasonal_pct"] = res_d.s_tot * 100.0
        row[f"{source}_s_diurnal_pct"] = (res.s_tot - res_d.s_tot) * 100.0

        for arch in acfg.get("archetypes", []):
            arch_cfg = {**storage_cfg, **arch}
            res_a = deficit.analyze_region(cf, dates, demand, arch_cfg,
                                           spy, simulate=False)
            row[f"{source}_s_tot_{arch['name']}_pct"] = res_a.s_tot * 100.0

        row.update({f"{source}_{k}": v for k, v in metrics.dunkelflaute(
            cf, spy, acfg["dunkelflaute_cf_frac"],
            acfg["gap_window_days"]).items()})
        if acfg.get("trends", False):
            row.update({f"{source}_{k}": v for k, v in metrics.cf_trend(cf, dates).items()})

    best = metrics.mix_sweep(cf_df["solar"].to_numpy(), cf_df["wind"].to_numpy(),
                             dates, demand, storage_cfg, spy,
                             step=acfg["mix_step"])
    row.update({"mix_alpha": best["mix_alpha"],
                "mix_s_tot_pct": best["mix_s_tot"] * 100.0,
                "mix_f_adj": best["mix_f_adj"]})
    return row


def run(cfg: dict, regions: pd.DataFrame) -> pd.DataFrame:
    table = process.load_zonal(cfg)
    value_cols = list(cfg["gldas"]["variables"].values())
    meta = regions.set_index("region_id")

    rows = []
    grouped = table.groupby("region_id")
    for i, (region_id, grp) in enumerate(grouped, 1):
        cf_df = capacity_factors(grp, cfg)
        row = {"region_id": int(region_id),
               "name": meta.loc[region_id, "name"],
               "country": meta.loc[region_id, "country"],
               "adm0_a3": meta.loc[region_id, "adm0_a3"],
               "continent": meta.loc[region_id, "continent"],
               "level": meta.loc[region_id, "level"]}
        row.update(analyze_one(cf_df, cfg, region_id))
        rows.append(row)
        if i % 50 == 0 or i == grouped.ngroups:
            log.info("analyzed %d/%d regions", i, grouped.ngroups)

    summary = pd.DataFrame(rows).round(4)
    out = cfg["paths"]["results_dir"] / "summary.csv"
    summary.to_csv(out, index=False)
    log.info("wrote %s", out)

    if cfg["analysis"].get("pools", False):
        pooled = metrics.pooled_tables(table, regions, value_cols)
        pool_rows = []
        for pool_name, grp in pooled.groupby("pool"):
            cf_df = capacity_factors(grp, cfg)
            row = {"pool": pool_name}
            row.update(analyze_one(cf_df, cfg, pool_name))
            pool_rows.append(row)
            log.info("analyzed pool: %s", pool_name)
        pooled_summary = pd.DataFrame(pool_rows).round(4)
        out = cfg["paths"]["results_dir"] / "pooled_summary.csv"
        pooled_summary.to_csv(out, index=False)
        log.info("wrote %s", out)

    bad = summary[~(summary["solar_converged"] & summary["wind_converged"])]
    if len(bad):
        log.warning("%d region(s) did not converge: %s",
                    len(bad), ", ".join(bad["name"].astype(str).head(20)))
    return summary
