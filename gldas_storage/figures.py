"""Stage 3: figures.

Reproduces the paper's figure families globally (from the 3-hourly zonal data,
plotted at daily aggregation for legibility):
  * variability figures: annual time series + day-of-year envelopes of the
    forcings (paper Figures 5/7/8) -- paginated PDFs, one panel per region;
  * deficit/storage figures: normalized supply vs demand and the cumulative
    deficit with the simulated storage state (paper Figures 6/9-12);
  * world choropleth maps of the summary metrics, including the new ones
    (optimal mix fraction, mix storage, diurnal/seasonal split, trends).

Figures are grouped the way the paper grouped states: one file for all
stand-alone (admin-0) countries, and one file per large country for its
states/provinces.
"""

from __future__ import annotations

import logging

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from . import analyze, deficit, process

log = logging.getLogger(__name__)

PANELS_PER_PAGE = (4, 3)
ENVELOPE = [("min", "blue"), ("q25", "cyan"), ("median", "black"),
            ("q75", "gold"), ("max", "red")]
# Human-readable legend labels for the day-of-year envelope lines.
ENVELOPE_LABELS = {
    "min": "Daily minimum across all years",
    "q25": "25th percentile across years",
    "median": "Median (typical day)",
    "q75": "75th percentile across years",
    "max": "Daily maximum across all years",
}
FORCING = {"solar": ("swdown", "Solar Rad. [W m$^{-2}$]"),
           "wind": ("wind10", "Wind Speed (10 m) [m s$^{-1}$]")}


def seasonal_stats(series: pd.Series) -> pd.DataFrame:
    doy = series.index.dayofyear
    keep = doy <= 365
    grp = series[keep].groupby(doy[keep])
    return pd.DataFrame({
        "min": grp.min(), "q25": grp.quantile(0.25), "median": grp.median(),
        "q75": grp.quantile(0.75), "max": grp.max(),
    })


def _page_iter(pdf: PdfPages, n_panels: int):
    """Yield axes one at a time, opening/closing 4x3 pages as needed."""
    nrow, ncol = PANELS_PER_PAGE
    per_page = nrow * ncol
    fig = axes = None
    for i in range(n_panels):
        if i % per_page == 0:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
            # constrained_layout keeps the per-panel titles, axis labels and
            # twin-axis labels from overlapping their neighbours on the dense grid.
            fig, axes = plt.subplots(nrow, ncol, figsize=(11, 8.5),
                                     constrained_layout=True)
            axes = axes.ravel()
            for ax in axes:
                ax.set_visible(False)
        ax = axes[i % per_page]
        ax.set_visible(True)
        yield ax
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)


def _legend_page(pdf: PdfPages, title: str, entries: list[tuple[str, float, str]],
                 note: str = "") -> None:
    """Write a single cover/legend page (first page of every multi-panel PDF).

    ``entries`` is a list of (color, linewidth, label) describing each line that
    appears in every panel, so the reader only needs one legend per file.
    """
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.80, title, ha="center", va="center", fontsize=15,
             weight="bold", wrap=True)
    handles = [Line2D([0], [0], color=c, lw=max(lw, 2.0)) for c, lw, _ in entries]
    labels = [lab for _, _, lab in entries]
    leg = fig.legend(handles, labels, loc="center", fontsize=12, frameon=True,
                     title="What the lines mean (same in every panel)")
    leg.get_title().set_fontsize(12)
    leg.get_title().set_fontweight("bold")
    if note:
        fig.text(0.5, 0.16, note, ha="center", va="center", fontsize=10, wrap=True)
    fig.text(0.5, 0.04, "Each subsequent page holds a 4x3 grid of panels; "
             "one panel per region.", ha="center", va="center",
             fontsize=9, style="italic", color="0.3")
    pdf.savefig(fig)
    plt.close(fig)


def variability_pdf(path, items: list[tuple[str, pd.Series]], ylabel: str,
                    source: str, group_name: str) -> None:
    """Annual-mean time series and day-of-year envelopes (daily aggregation)."""
    region_label = group_name.replace("_", " ")
    with PdfPages(path.with_name(path.stem + "_annual.pdf")) as pdf:
        _legend_page(
            pdf,
            f"Inter-annual variability of the {source} resource\n{region_label}",
            [("black", 1.5, f"Annual mean of {ylabel}")],
            note="X axis: calendar year (2000-2025).  Y axis: that year's mean "
                 f"of the {source} forcing.  A rising/falling line is a long-term "
                 "trend in the resource.")
        for ax, (title, series) in zip(_page_iter(pdf, len(items)), items):
            annual = series.groupby(series.index.year).mean()
            ax.plot(annual.index, annual.values, color="black", lw=1.5)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time [year]", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)
    with PdfPages(path.with_name(path.stem + "_seasonal.pdf")) as pdf:
        _legend_page(
            pdf,
            f"Seasonal cycle of the {source} resource\n{region_label}",
            [(color, 1.0, ENVELOPE_LABELS[col]) for col, color in ENVELOPE],
            note="X axis: day of year (1-365).  Y axis: "
                 f"{ylabel}.  Each line is a percentile of the daily value taken "
                 "across all years, so the band width shows year-to-year spread.")
        for ax, (title, series) in zip(_page_iter(pdf, len(items)), items):
            daily = series.resample("1D").mean()
            stats = seasonal_stats(daily)
            for col, color in ENVELOPE:
                ax.plot(stats.index, stats[col], color=color, lw=1.0)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Time [day of year]", fontsize=7)
            ax.set_ylabel(ylabel, fontsize=7)
            ax.tick_params(labelsize=6)


def deficit_pdf(path, items: list[tuple[str, deficit.RegionResult]], source: str,
                group_name: str) -> None:
    """Supply/demand panels and deficit/storage panels (paper Fig. 6 left/right)."""
    region_label = group_name.replace("_", " ")
    with PdfPages(path.with_name(path.stem + "_supply.pdf")) as pdf:
        _legend_page(
            pdf,
            f"Normalized {source} supply vs. demand\n{region_label}",
            [("green", 0.3, "Normalized supply (daily mean) -- left axis"),
             ("black", 1.0, "Normalized demand -- left axis"),
             ("magenta", 1.2, "Annual-mean supply -- right axis")],
            note="Both series are normalized so demand averages 1; supply is the "
                 "capacity-factor series scaled by the excess-installation factor. "
                 "Where green dips below black the region is short of energy.")
        for ax, (title, res) in zip(_page_iter(pdf, len(items)), items):
            s = res.series.resample("1D").mean()
            ax.plot(s.index, s["supply"], color="green", lw=0.3)
            ax.plot(s.index, s["demand"], color="black", lw=1.0)
            annual = s["supply"].groupby(s.index.year).mean()
            ax2 = ax.twinx()
            ax2.plot(pd.to_datetime(annual.index, format="%Y"),
                     annual.values, color="magenta", lw=1.2)
            ax2.set_ylabel("Annual-mean supply", fontsize=7, color="magenta")
            ax2.tick_params(labelsize=6, colors="magenta")
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Date", fontsize=7)
            ax.set_ylabel(f"Normalized {source.title()}", fontsize=7)
            ax.tick_params(labelsize=6)
    with PdfPages(path.with_name(path.stem + "_storage.pdf")) as pdf:
        _legend_page(
            pdf,
            f"Cumulative deficit and storage state ({source})\n{region_label}",
            [("blue", 0.5, "Cumulative deficit [%] -- left axis"),
             ("red", 0.5, "Storage state, % of sized capacity -- right axis")],
            note="The deficit (blue) is the running unmet demand; the simulated "
                 "store (red) starts full and is drawn down to cover it. A store "
                 "sized at S_tot keeps the deficit from growing without bound.")
        for ax, (title, res) in zip(_page_iter(pdf, len(items)), items):
            s = res.series.resample("1D").mean()
            ax.plot(s.index, s["deficit"] * 100, color="blue", lw=0.5)
            ax.set_ylabel(f"{source.title()} Deficit [%]", fontsize=7, color="blue")
            ax.set_xlabel("Date", fontsize=7)
            ax2 = ax.twinx()
            if res.s_tot > 0:
                ax2.plot(s.index, s["storage"] / res.s_tot * 100, color="red", lw=0.5)
            ax2.set_ylabel("Storage [%]", fontsize=7, color="red")
            ax2.tick_params(labelsize=6, colors="red")
            ax.set_title(title, fontsize=9)
            ax.tick_params(labelsize=6)


MAP_METRICS = [
    ("solar_cf", "Solar capacity factor", None),
    ("wind_cf", "Wind capacity factor", None),
    ("solar_f_adj", "Solar adjusted excess installation factor", None),
    ("wind_f_adj", "Wind adjusted excess installation factor", None),
    ("solar_s_tot_pct", "Solar total storage [% of annual consumption]", (0, 60)),
    ("wind_s_tot_pct", "Wind total storage [% of annual consumption]", (0, 60)),
    ("solar_s_diurnal_pct", "Solar diurnal storage tier [%]", None),
    ("wind_flaute_days", "Wind: longest low-resource spell [days]", None),
    ("mix_alpha", "Storage-optimal solar fraction", (0, 1)),
    ("mix_s_tot_pct", "Optimal-mix total storage [% of annual consumption]", (0, 60)),
]


def world_maps(cfg: dict, regions: gpd.GeoDataFrame, summary: pd.DataFrame) -> None:
    drop = [c for c in ("name", "country", "adm0_a3", "continent", "level") if c in summary]
    gdf = regions.merge(summary.drop(columns=drop), on="region_id", how="left")
    for col, label, clim in MAP_METRICS:
        if col not in gdf.columns:
            continue
        fig, ax = plt.subplots(figsize=(12, 6))
        kwargs = {"vmin": clim[0], "vmax": clim[1]} if clim else {}
        gdf.plot(column=col, ax=ax, legend=True, cmap="viridis",
                 missing_kwds={"color": "lightgrey"},
                 legend_kwds={"shrink": 0.6, "label": label}, **kwargs)
        ax.set_title(label)
        ax.set_axis_off()
        out = cfg["paths"]["figures_dir"] / f"map_{col}.png"
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        log.info("wrote %s", out.name)


def _region_groups(regions: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    groups = [("World_countries", regions[regions["level"] == "admin0"].sort_values("name"))]
    for country, grp in regions[regions["level"] == "admin1"].groupby("country"):
        groups.append((country.replace(" ", "_"), grp.sort_values("name")))
    return groups


def run(cfg: dict, regions: gpd.GeoDataFrame, summary: pd.DataFrame) -> None:
    figdir = cfg["paths"]["figures_dir"]
    table = process.load_zonal(cfg)
    storage_cfg = cfg["storage"]

    by_region = {int(rid): grp for rid, grp in table.groupby("region_id")}
    for group_name, grp in _region_groups(regions):
        for source in analyze.SOURCES:
            col, ylabel = FORCING[source]
            items_var, items_def = [], []
            for _, r in grp.iterrows():
                rid = int(r["region_id"])
                if rid not in by_region:
                    continue
                z = by_region[rid]
                series = pd.Series(z[col].to_numpy(float),
                                   index=pd.DatetimeIndex(z["time"]))
                items_var.append((r["name"], series))

                cf_df = analyze.capacity_factors(z, cfg)
                demand = deficit.make_demand(cfg, cf_df.index,
                                             tair=cf_df["tair"].to_numpy(),
                                             monthly=analyze.monthly_demand_profile(cfg, rid))
                res = deficit.analyze_region(cf_df[source].to_numpy(), cf_df.index,
                                             demand, storage_cfg,
                                             analyze.steps_per_year(cfg), simulate=True)
                if res.series is not None:
                    items_def.append((r["name"], res))
            if not items_var:
                continue
            variability_pdf(figdir / f"variability_{source}_{group_name}.pdf",
                            items_var, ylabel, source, group_name)
            deficit_pdf(figdir / f"deficit_{source}_{group_name}.pdf", items_def,
                        source, group_name)
            log.info("figures for %s / %s (%d regions)", group_name, source, len(items_var))

    world_maps(cfg, regions, summary)
