#!/usr/bin/env python
"""Build the static data bundle for the GLDAS/NLDAS storage-analysis website.

Produces, under ``<repo>/web/data`` (override with --out):

  regions.geojson      simplified world admin0+admin1 polygons + key metrics as
                       feature properties (for the Leaflet choropleth).
  regions_index.json   [{id,name,country,level,continent,has_usa}] lookup list.
  meta.json            dataset/version/period descriptors + metric dictionary.
  regions/<id>.json    per-region numbers (normalized; + USA real-units/nameplate/
                       feasibility) and 12-month climatology series for the charts.

The per-region series are recomputed from the zonal table with the SAME functions
the analysis pipeline uses (analyze.capacity_factors / deficit.analyze_region), then
collapsed to a month-of-year climatology so each file stays a couple of kB.

Scope:
  --scope usa     only the 51 USA admin1 states (fast; world numbers still exported
                  for the map from summary.csv, but no per-region world series).
  --scope world   every region with zonal data (heavy: loads the full zonal table).
  --scope all     world + USA real units (default).

NLDAS-USA outputs do not exist yet; each USA region carries an ``nldas`` slot that is
null until config_nldas.yaml's results are present (re-run with --nldas-config then).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import analyze, deficit, landuse, process       # noqa: E402
from gldas_storage.config import load_config                       # noqa: E402

log = logging.getLogger("build_website")

K_LEVELS = [0, 17, 35, 50]            # co-location overlap k (% of farm reused), see colocation_scenario.md
DOWNLOAD_CSVS = ["summary.csv", "usa_summary.csv", "storage_ranking_full.csv",
                 "colocation_storage.csv", "usa_months_unmet.csv", "usa_storage_ranking.csv"]

# metric dictionary surfaced in the UI (key -> short label + help + unit)
METRIC_INFO = {
    "solar_cf":      ("Solar capacity factor", "Mean solar PV output as a fraction of nameplate.", ""),
    "wind_cf":       ("Wind capacity factor", "Mean wind output as a fraction of nameplate.", ""),
    "mix_alpha":     ("Optimal solar share α*", "Storage-optimal fraction of the build that is solar (rest is wind).", ""),
    "mix_s_tot_pct": ("Optimal-mix storage", "Storage to ride out the worst year, as % of annual consumption.", "% annual"),
    "solar_s_tot_pct": ("Solar-only storage", "Storage for a solar-only build, % of annual consumption.", "% annual"),
    "wind_s_tot_pct":  ("Wind-only storage", "Storage for a wind-only build, % of annual consumption.", "% annual"),
    "mix_f_adj":     ("Overbuild factor", "Excess installation (worst-year reliability) for the optimal mix.", "×"),
    "solar_flaute_days": ("Solar dark spell", "Longest low-solar spell.", "days"),
    "wind_flaute_days":  ("Wind lull spell", "Longest low-wind (Dunkelflaute) spell.", "days"),
}
MAP_METRIC_KEYS = ["mix_s_tot_pct", "mix_alpha", "solar_cf", "wind_cf",
                   "solar_s_tot_pct", "wind_s_tot_pct"]

# columns we copy verbatim into each region's normalized "metrics" block
WORLD_METRIC_COLS = [
    "solar_cf", "wind_cf", "solar_f_adj", "wind_f_adj", "mix_alpha", "mix_f_adj",
    "solar_s_tot_pct", "wind_s_tot_pct", "mix_s_tot_pct",
    "solar_s_seasonal_pct", "solar_s_diurnal_pct",
    "wind_s_seasonal_pct", "wind_s_diurnal_pct",
    "solar_flaute_days", "wind_flaute_days",
    "solar_gap30d_max", "wind_gap30d_max",
    "solar_trend_pct_decade", "wind_trend_pct_decade",
]
# USA real-units block (deduplicated; see note on .1 columns)
USA_REAL_COLS = [
    "annual_consumption_TWh", "state_land_km2",
    "mix_solar_nameplate_GW", "mix_wind_nameplate_GW", "mix_capacity_TWh",
    "mix_land_km2", "mix_land_pct", "solar_land_pct", "wind_land_pct",
    "mix_storage_TWh",
]


def _clean(v):
    """JSON-safe scalar (NaN/inf -> None, numpy -> python)."""
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        f = float(v)
        return None if not math.isfinite(f) else round(f, 6)
    if isinstance(v, str):
        return v
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    return v


def _row_metrics(row: pd.Series, cols) -> dict:
    return {c: _clean(row[c]) for c in cols if c in row.index}


def load_zonal_subset(cfg: dict, region_ids: set[int] | None) -> pd.DataFrame:
    """Like process.load_zonal but optionally keep only some region_ids per file
    (so USA scope doesn't hold the whole world in memory)."""
    if region_ids is None:
        return process.load_zonal(cfg)
    files = sorted(cfg["paths"]["zonal_dir"].glob("zonal_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no zonal files in {cfg['paths']['zonal_dir']}")
    start, end = pd.Timestamp(cfg["period"]["start"]), pd.Timestamp(cfg["period"]["end"])
    parts = []
    for f in files:
        t = pd.read_parquet(f)
        t = t[t["region_id"].isin(region_ids)]
        if len(t):
            parts.append(t)
    table = pd.concat(parts, ignore_index=True)
    table["time"] = pd.to_datetime(table["time"])
    table = table[(table["time"] >= start) & (table["time"] <= end + pd.Timedelta(hours=23))]
    return table.sort_values(["region_id", "time"]).reset_index(drop=True)


def month_climatology(series: pd.Series) -> list:
    """Mean value for each calendar month (Jan..Dec), JSON-safe."""
    clim = series.groupby(series.index.month).mean()
    return [_clean(clim.get(m, np.nan)) for m in range(1, 13)]


def region_series(cf_df: pd.DataFrame, cfg: dict, alpha: float, spy: int,
                  rid, monthly_demand=None) -> dict:
    """Normalized monthly-climatology series for one region: solar/wind/mix supply
    (mean ~1, after worst-year overbuild) vs demand, plus mix storage-state."""
    dates = cf_df.index
    demand = deficit.make_demand(cfg, dates, tair=cf_df["tair"].to_numpy(),
                                 monthly=monthly_demand)
    s = cf_df["solar"].to_numpy(float)
    w = cf_df["wind"].to_numpy(float)
    mix = alpha * (s / s.mean()) + (1 - alpha) * (w / w.mean())
    out = {"months": list(range(1, 13)),
           "demand": month_climatology(pd.Series(demand, index=dates))}
    for name, arr in (("solar", s), ("wind", w), ("mix", mix)):
        res = deficit.analyze_region(arr, dates, demand, cfg["storage"], spy,
                                     simulate=(name == "mix"))
        out[f"{name}_supply"] = month_climatology(
            pd.Series(res.series["supply"] if res.series is not None
                      else arr / arr.mean() * res.tot_factor, index=dates))
        if name == "mix" and res.series is not None:
            out["mix_storage"] = month_climatology(res.series["storage"])
            out["mix_s_tot"] = _clean(res.s_tot)
    return out


def usa_real_series(cf_df: pd.DataFrame, cfg: dict, alpha: float, spy: int,
                    cons_twh: float, land_pct: float, monthly_demand) -> dict:
    """Real-units (TWh) monthly-climatology mix supply with the four land-cap lines
    plus the EIA demand, mirroring figures.deficit_pdf_realunits."""
    dates = cf_df.index
    demand = deficit.make_demand(cfg, dates, tair=cf_df["tair"].to_numpy(),
                                 monthly=monthly_demand)
    s = cf_df["solar"].to_numpy(float); w = cf_df["wind"].to_numpy(float)
    mix = alpha * (s / s.mean()) + (1 - alpha) * (w / w.mean())
    res = deficit.analyze_region(mix, dates, demand, cfg["storage"], spy, simulate=True)
    sup = pd.Series(res.series["supply"], index=dates) * cons_twh / spy
    dem = pd.Series(res.series["demand"], index=dates) * cons_twh / spy
    sup_m = sup.groupby(sup.index.month).sum() / (len(set(dates.year)))
    dem_m = dem.groupby(dem.index.month).sum() / (len(set(dates.year)))
    caps = [("nocap", None)] + [(k, frac) for k, frac, _ in landuse.LAND_CAPS]
    series = {"months": list(range(1, 13)),
              "demand_TWh": [_clean(dem_m.get(m, np.nan)) for m in range(1, 13)]}
    for key, frac in caps:
        scale = 1.0 if frac is None or not (land_pct and land_pct > 0) \
            else min(1.0, frac * 100.0 / land_pct)
        series[f"supply_TWh_{key}"] = [_clean(sup_m.get(m, np.nan) * scale)
                                       for m in range(1, 13)]
    return series


def usa_block(row: pd.Series) -> dict:
    """USA real-units numbers + per-cap feasibility/storage/shortfall."""
    nums = _row_metrics(row, USA_REAL_COLS)
    caps = {}
    for key, frac, label in landuse.LAND_CAPS:
        caps[key] = {
            "label": label, "cap_pct": frac * 100.0,
            "feasible": _clean(row.get(f"feasible_{key}")),
            "storage_TWh": _clean(row.get(f"mix_storage_TWh_{key}")),
            "supply_capped_TWh": _clean(row.get(f"mix_supply_capped_TWh_{key}")),
            "shortfall_TWh": _clean(row.get(f"mix_shortfall_TWh_{key}")),
        }
    nums["caps"] = caps
    return nums


def dedupe_dot1(df: pd.DataFrame) -> pd.DataFrame:
    """Drop pandas '.1' duplicate columns (identical legacy copies)."""
    return df[[c for c in df.columns if not c.endswith(".1")]]


def coloc_block(rid: int, coloc_by_id: dict, cl_by_id: dict) -> dict | None:
    """Per-region co-location result: normalized storage (% of annual consumption) and
    the winning technology as the overlap k rises, + USA cap×k land-feasibility grid."""
    r = coloc_by_id.get(rid)
    if r is None:
        return None
    b = {"k": K_LEVELS,
         "storage_pct_cons": [_clean(r.get(f"k{k}_cons")) for k in K_LEVELS],
         "pick": [r.get(f"k{k}_pick") if isinstance(r.get(f"k{k}_pick"), str) else None
                  for k in K_LEVELS]}
    cr = cl_by_id.get(rid)
    if cr is not None:
        b["usa_feasible"] = {key: [bool(cr.get(f"{key}_k{k}_feasible")) for k in K_LEVELS]
                             for key, _, _ in landuse.LAND_CAPS}
    return b


def region_detail_pdf(path: Path, name: str, cf_df: pd.DataFrame, cfg: dict,
                      alpha: float, spy: int, usa: dict | None = None) -> None:
    """Standalone detailed figure for one region (full 2000-2025 record): normalized
    supply-vs-demand + cumulative deficit/storage; USA adds real-units TWh cap lines.
    Dense lines are rasterized so each PDF stays small."""
    dates = cf_df.index
    demand = deficit.make_demand(cfg, dates, tair=cf_df["tair"].to_numpy(),
                                 monthly=usa["mdem"] if usa else None)
    s = cf_df["solar"].to_numpy(float); w = cf_df["wind"].to_numpy(float)
    mix = alpha * (s / s.mean()) + (1 - alpha) * (w / w.mean())
    res = deficit.analyze_region(mix, dates, demand, cfg["storage"], spy, simulate=True)
    nrows = 3 if usa else 2
    fig, axes = plt.subplots(nrows, 1, figsize=(8.2, 2.5 * nrows))

    sup = pd.Series(res.series["supply"], index=dates).resample("MS").mean()
    dem = pd.Series(res.series["demand"], index=dates).resample("MS").mean()
    ax = axes[0]
    ax.plot(sup.index, sup.values, color="#1f77b4", lw=0.6, label="optimal-mix supply", rasterized=True)
    ax.plot(dem.index, dem.values, color="black", lw=0.8, label="demand")
    ax.axhline(1.0, color="0.6", lw=0.5)
    ax.set_ylabel("supply / mean demand", fontsize=8)
    ax.set_title(f"{name} — normalized supply vs demand (monthly, optimal mix)", fontsize=10)
    ax.legend(fontsize=7, loc="upper right"); ax.tick_params(labelsize=7)

    daily = res.series.resample("1D").mean()
    ax = axes[1]
    ax.plot(daily.index, daily["deficit"], color="#1a5fb4", lw=0.5, rasterized=True)
    ax.set_ylabel("cumulative deficit", fontsize=8, color="#1a5fb4")
    ax.tick_params(labelsize=7)
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["storage"], color="#d73027", lw=0.5, rasterized=True)
    ax2.set_ylabel("storage state", fontsize=8, color="#d73027")
    ax2.tick_params(labelsize=7, colors="#d73027")
    ax.set_title(f"Cumulative deficit & storage state (peak deficit = required storage, "
                 f"S_tot={res.s_tot*100:.1f}% of annual)", fontsize=10)

    if usa is not None:
        cons, lp = usa["cons"], usa["land_pct"]
        sm = (pd.Series(res.series["supply"], index=dates) * cons / spy)
        dm = (pd.Series(res.series["demand"], index=dates) * cons / spy)
        sm_m = sm.resample("MS").sum(); dm_m = dm.resample("MS").sum()
        ax = axes[2]
        cap_lines = [(None, "no land cap", "#1a9850")] + \
            [(f, f"≤{f*100:g}% land", c) for (k, f, _), c in
             zip(landuse.LAND_CAPS, ["#d73027", "#fdae61", "#74add1"])]
        for frac, lbl, color in cap_lines:
            scale = 1.0 if frac is None or not (lp and lp > 0) else min(1.0, frac * 100.0 / lp)
            ax.plot(sm_m.index, sm_m.values * scale, color=color, lw=0.6, label=lbl, rasterized=True)
        ax.plot(dm_m.index, dm_m.values, color="black", lw=0.8, label="EIA demand")
        ax.set_ylabel("TWh / month", fontsize=8)
        ax.set_title(f"Real energy vs EIA demand ({cons:.0f} TWh/yr, {lp:.1f}% land)", fontsize=10)
        ax.legend(fontsize=6, ncol=2, loc="upper right"); ax.tick_params(labelsize=7)

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def load_nldas_usa(nldas_cfg_path: Path, spy_gldas: int):
    """If the NLDAS-USA run exists, return {state_name: nldas_record} where each record
    mirrors the GLDAS USA block (real units + caps) and series_real. Matched downstream by
    NAME (NLDAS uses its own region_id numbering). Returns {} when the run is absent."""
    nl_cfg = load_config(nldas_cfg_path)
    rdir = Path(nl_cfg["paths"]["results_dir"])
    fsum = rdir / "usa_summary.csv"
    if not fsum.exists():
        return {}
    usa = dedupe_dot1(pd.read_csv(fsum))
    ids = set(int(x) for x in usa["region_id"])
    spy = analyze.steps_per_year(nl_cfg)
    table = load_zonal_subset(nl_cfg, ids)
    by_region = {int(rid): grp for rid, grp in table.groupby("region_id")}
    monthly_csv = rdir / "usa_eia_monthly_demand.csv"
    out = {}
    for _, urow in usa.iterrows():
        rid = int(urow["region_id"]); name = urow["name"]
        rec = {"real": usa_block(urow), "metrics": _row_metrics(urow, WORLD_METRIC_COLS)}
        if rid in by_region:
            mcfg = dict(nl_cfg)
            mcfg["demand"] = dict(nl_cfg["demand"], profile="monthly", monthly_csv=str(monthly_csv))
            mdem = analyze.monthly_demand_profile(mcfg, rid)
            cf_df = analyze.capacity_factors(by_region[rid], nl_cfg)
            cons = float(urow["annual_consumption_TWh"]); lp = float(urow["mix_land_pct"])
            rec["series_real"] = usa_real_series(cf_df, mcfg, float(urow["mix_alpha"]),
                                                 spy, cons, lp, mdem)
            rec["series_norm"] = region_series(cf_df, mcfg, float(urow["mix_alpha"]), spy, rid, mdem)
            rec["consumption_TWh"] = _clean(cons)
        out[name] = rec
    log.info("NLDAS-USA: loaded %d states", len(out))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[1]
    ap.add_argument("--config", default=repo / "hpc" / "config_oscer.yaml")
    ap.add_argument("--nldas-config", default=repo / "hpc" / "config_nldas.yaml")
    ap.add_argument("--out", default=repo / "web" / "data")
    ap.add_argument("--scope", choices=["usa", "world", "all"], default="all")
    ap.add_argument("--simplify", type=float, default=0.02,
                    help="Douglas-Peucker tolerance in degrees for web geometry")
    ap.add_argument("--no-series", action="store_true",
                    help="export numbers + geojson only (skip the zonal sim)")
    ap.add_argument("--with-pdfs", action="store_true",
                    help="also render a standalone detailed PDF per region (heavy)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    out = Path(args.out)
    (out / "regions").mkdir(parents=True, exist_ok=True)
    if args.with_pdfs:
        (out / "figures").mkdir(parents=True, exist_ok=True)
    rdir = Path(cfg["paths"]["results_dir"])

    gdf = gpd.read_file(cfg["paths"]["regions_gpkg"])
    summary = pd.read_csv(rdir / "summary.csv")
    usa = dedupe_dot1(pd.read_csv(rdir / "usa_summary.csv"))
    usa_ids = set(int(x) for x in usa["region_id"])
    log.info("regions: %d geom, %d world rows, %d USA rows", len(gdf), len(summary), len(usa))

    sum_by_id = {int(r["region_id"]): r for _, r in summary.iterrows()}
    usa_by_id = {int(r["region_id"]): r for _, r in usa.iterrows()}

    # co-location results (world storage-vs-k; USA land cap×k feasibility)
    coloc_by_id, cl_by_id = {}, {}
    fcoloc = rdir / "colocation_storage.csv"
    if fcoloc.exists():
        coloc_by_id = {int(r["region_id"]): r for _, r in pd.read_csv(fcoloc).iterrows()}
    fcl = rdir / "colocation_land_feasibility.csv"
    if fcl.exists():
        cl_by_id = {int(r["region_id"]): r for _, r in pd.read_csv(fcl).iterrows()}

    # downloadable CSVs
    dl = out / "downloads"; dl.mkdir(parents=True, exist_ok=True)
    copied = []
    for fn in DOWNLOAD_CSVS:
        src = rdir / fn
        if src.exists():
            shutil.copy(src, dl / fn); copied.append(fn)
    log.info("downloads: copied %d CSVs", len(copied))

    # NLDAS-USA (guarded: {} until that run exists)
    nldas_usa = load_nldas_usa(Path(args.nldas_config), analyze.steps_per_year(cfg))

    # ---- simplified geojson with key metrics as properties --------------------
    g = gdf.copy()
    g["geometry"] = g["geometry"].simplify(args.simplify, preserve_topology=True)
    feats = []
    for _, r in g.iterrows():
        rid = int(r["region_id"])
        srow = sum_by_id.get(rid)
        props = {"id": rid, "name": r["name"], "country": r["country"],
                 "level": r["level"], "continent": r.get("continent"),
                 "has_usa": rid in usa_ids,
                 "no_data": bool(r.get("weight_sum", 1) < 1.0)}
        for k in MAP_METRIC_KEYS:
            props[k] = _clean(srow[k]) if srow is not None and k in srow.index else None
        if rid in usa_by_id:
            props["feasible_1pct"] = _clean(usa_by_id[rid].get("feasible_1pct"))
            props["mix_land_pct"] = _clean(usa_by_id[rid].get("mix_land_pct"))
        feats.append({"type": "Feature",
                      "properties": props,
                      "geometry": r["geometry"].__geo_interface__})
    geojson = {"type": "FeatureCollection", "features": feats}
    (out / "regions.geojson").write_text(json.dumps(geojson))
    log.info("wrote regions.geojson (%d features, %.1f MB)", len(feats),
             (out / "regions.geojson").stat().st_size / 1e6)

    # ---- index + meta ---------------------------------------------------------
    index = [{"id": int(r["region_id"]), "name": r["name"], "country": r["country"],
              "level": r["level"], "continent": r.get("continent"),
              "has_usa": int(r["region_id"]) in usa_ids}
             for _, r in gdf.iterrows()]
    (out / "regions_index.json").write_text(json.dumps(index))

    nldas_avail = len(nldas_usa) > 0
    st = cfg["storage"]
    meta = {
        "title": "Solar + Wind Storage Atlas",
        "period": f"{cfg['period']['start'][:4]}–{cfg['period']['end'][:4]}",
        "datasets": {
            "gldas": {"label": "GLDAS · theoretical (3-hourly)", "available": True},
            "nldas": {"label": "NLDAS-USA · EIA (hourly)", "available": bool(nldas_avail)},
        },
        "metric_info": {k: {"label": v[0], "help": v[1], "unit": v[2]}
                        for k, v in METRIC_INFO.items()},
        "map_metrics": MAP_METRIC_KEYS,
        "land_caps": [{"key": k, "cap_pct": f * 100, "label": lbl}
                      for k, f, lbl in landuse.LAND_CAPS],
        "k_levels": K_LEVELS,
        "downloads": copied,
        "assumptions": {
            "Model": "Each region is sized to be self-sufficient on solar+wind+storage alone, "
                     "meeting demand every year including the worst. The optimal mix is the "
                     "solar share α* that minimizes storage.",
            "Overbuild (f_adj)": "Capacity is over-installed by the worst-year reliability "
                     "factor f_adj so the weakest year still meets demand with storage.",
            "Storage": f"Round-trip + self-discharge: charge {st['up_efficiency']}, discharge "
                     f"{st['down_efficiency']}, annual energy retention {st['annual_decay']}.",
            "Land density": f"Solar {landuse.SOLAR_MW_PER_KM2:g} MW/km² (array footprint); "
                     f"wind {landuse.WIND_DIRECT_MW_PER_KM2:g} MW/km² (direct footprint — "
                     "turbine pads + roads only; the ~98% spacing stays farmable).",
            "Land caps": "Feasibility is tested at 1% of state land (NREL's <1% claim), 1.88% "
                     "(2× US onshore oil & gas surface footprint), and 5% (all onshore oil "
                     "& gas production land).",
            "Co-location (k)": "Overlap k lets the same footprint host (1+k)× generation "
                     "(turbines + PV on one parcel), so land needed shrinks and more regions fit "
                     "a cap as k rises (k = 0/17/35/50%).",
            "GLDAS vs NLDAS": "GLDAS: global 3-hourly theoretical forcing with flat (normalized) "
                     "demand. NLDAS-USA: hourly forcing over the CONUS with real monthly EIA "
                     "consumption, so USA results are in real energy (TWh).",
            "Framing": "This quantifies LOCAL self-sufficiency (per-state autarky), not national "
                     "grid optimization — where our land/storage exceeds NREL's <1%, that gap "
                     "is the externalized cost of the national transmission build-out.",
        },
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))

    # ---- per-region numbers (always) + series (unless --no-series) ------------
    spy = analyze.steps_per_year(cfg)
    want_world = args.scope in ("world", "all")
    want_usa = args.scope in ("usa", "all")

    # which region_ids need series this run
    series_ids = set()
    if not args.no_series:
        if want_world:
            series_ids |= set(sum_by_id)
        if want_usa:
            series_ids |= usa_ids

    by_region = {}
    if series_ids:
        sub = None if want_world else usa_ids
        log.info("loading zonal table (%s)...", "world" if sub is None else f"{len(sub)} USA")
        table = load_zonal_subset(cfg, sub)
        table = table[table["region_id"].isin(series_ids)]
        by_region = {int(rid): grp for rid, grp in table.groupby("region_id")}
        log.info("zonal loaded: %d regions with data", len(by_region))

    # EIA monthly demand for USA real-units series
    usa_monthly_csv = rdir / "usa_eia_monthly_demand.csv"

    n = npdf = 0
    for rid, srow in sum_by_id.items():
        is_usa = rid in usa_by_id
        rec = {
            "id": rid, "name": srow["name"], "country": srow["country"],
            "level": srow["level"], "continent": _clean(srow.get("continent")),
            "metrics": _row_metrics(srow, WORLD_METRIC_COLS),
            "colocation": coloc_block(rid, coloc_by_id, cl_by_id),
            "datasets": {"gldas": {}, "nldas": None},
        }
        cf_df = None
        if rid in by_region:
            cf_df = analyze.capacity_factors(by_region[rid], cfg)
        alpha = float(srow["mix_alpha"]) if pd.notna(srow.get("mix_alpha")) else 0.5

        # GLDAS normalized series
        if cf_df is not None:
            rec["datasets"]["gldas"]["series_norm"] = region_series(cf_df, cfg, alpha, spy, rid)

        # USA real units (GLDAS theoretical forcing + EIA consumption)
        usa_pdf = None
        if is_usa:
            urow = usa_by_id[rid]
            rec["usa"] = usa_block(urow)
            if cf_df is not None and not args.no_series:
                mcfg = dict(cfg)
                mcfg["demand"] = dict(cfg["demand"], profile="monthly",
                                      monthly_csv=str(usa_monthly_csv))
                mdem = analyze.monthly_demand_profile(mcfg, rid)
                cons = float(urow["annual_consumption_TWh"]); lp = float(urow["mix_land_pct"])
                rec["datasets"]["gldas"]["series_real"] = usa_real_series(
                    cf_df, mcfg, float(urow["mix_alpha"]), spy, cons, lp, mdem)
                rec["datasets"]["gldas"]["consumption_TWh"] = _clean(cons)
                usa_pdf = {"mdem": mdem, "cons": cons, "land_pct": lp}
            # NLDAS-USA (matched by NAME); null until that run exists
            rec["datasets"]["nldas"] = nldas_usa.get(srow["name"])

        # standalone detailed PDF
        if args.with_pdfs and cf_df is not None:
            try:
                region_detail_pdf(out / "figures" / f"{rid}.pdf", srow["name"], cf_df, cfg,
                                  alpha, spy, usa=usa_pdf if is_usa else None)
                rec["pdf"] = f"figures/{rid}.pdf"
                npdf += 1
            except Exception as e:                       # one bad region must not kill the run
                log.warning("pdf failed for %s (%d): %s", srow["name"], rid, e)

        (out / "regions" / f"{rid}.json").write_text(json.dumps(rec))
        n += 1
    log.info("wrote %d per-region JSON files (%d PDFs) to %s", n, npdf, out / "regions")
    log.info("NLDAS-USA states wired: %d", len(nldas_usa))


if __name__ == "__main__":
    main()
