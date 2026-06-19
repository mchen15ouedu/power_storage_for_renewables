"""De-normalized (ACTUAL energy units) supply-vs-consumption check for a few US states.

Everything in the deficit framework is normalized (demand mean -> 1). This script
puts real MWh/TWh back on both sides for a handful of states so the temporal match
between GLDAS-derived generation and real EIA consumption is visible in physical units.

Sizing = annual-energy parity (the assumption-light de-normalization): install enough
nameplate that long-run MEAN annual generation equals that state's MEAN annual EIA
consumption (2000-2025). The framework's worst-year overbuild factor f_adj and the
storage S_tot are reported alongside as the EXTRA capacity/storage the analysis adds.

  solar-only : nameplate sized so annual solar generation   = annual consumption
  wind-only  : nameplate sized so annual wind  generation   = annual consumption
  mix        : alpha*solar_norm + (1-alpha)*wind_norm (alpha = storage-optimal mix_alpha
               from usa_summary.csv), scaled so annual mix generation = annual consumption

Outputs (results_dir):
  actual_supply_vs_eia_monthly.csv  -- per state, 12-month climatology: consumption +
                                       solar/wind/mix generation (GWh) + nameplate (MW)
  actual_supply_vs_eia_summary.csv  -- per state annual totals, CF, nameplate, f_adj, S_tot
  figures: actual_supply_vs_eia.png -- 5 panels, monthly generation vs consumption (TWh)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import geopandas as gpd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gldas_storage import analyze, process
from gldas_storage.config import load_config

log = logging.getLogger(__name__)

EIA_TSV = Path("/ourdisk/hpc/caps/mchen15/gldas_analysis/EIA/"
               "Consumption_Megawatthours_1990_2025.tsv")
STATES = ["California", "Texas", "North Dakota", "Florida", "Washington"]
ABBR = {"California": "CA", "Texas": "TX", "North Dakota": "ND",
        "Florida": "FL", "Washington": "WA"}
HOURS_PER_YEAR = 8760.0

# --- land-use densities (NREL) ------------------------------------------------
# Solar PV, full array footprint: NREL Ong 2013 (TP-6A20-56290) capacity-weighted
#   ~7 acres/MWac; Nature 2025 measured 24.7 MW/km^2. Use 30 MW/km^2 (round NREL).
# Wind: NREL Denholm 2009 (TP-6A2-45834): TOTAL project area ~34.5 ha/MW (= 2.9
#   MW/km^2) but turbines/roads/pads PERMANENTLY remove only ~1 ha/MW direct
#   (= 100 MW/km^2 ~ 3% of project area); the other 97% stays farmable/grazable.
SOLAR_MW_PER_KM2 = 30.0          # solar array footprint
WIND_PROJECT_MW_PER_KM2 = 3.0    # wind spacing/project area (co-usable land)
WIND_DIRECT_MW_PER_KM2 = 100.0   # wind turbines+roads only (land actually removed)
EQUAL_AREA_EPSG = 5070           # USA Contiguous Albers Equal Area (km^2-accurate)
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def eia_monthly_clim(abbr: str, y0: int, y1: int) -> pd.Series:
    """Climatological monthly consumption (MWh), mean over y0..y1."""
    eia = pd.read_csv(EIA_TSV, sep="\t")
    eia.columns = [c.strip().strip('"') for c in eia.columns]
    eia = eia[(eia["Year"] >= y0) & (eia["Year"] <= y1)]
    return eia.groupby("Month")[abbr].mean().reindex(range(1, 13))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default=Path(__file__).resolve().parents[1] / "hpc" / "config_oscer.yaml")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    results_dir = Path(cfg["paths"]["results_dir"])
    figdir = Path(cfg["paths"]["figures_dir"])
    spy = analyze.steps_per_year(cfg)              # 2920 for GLDAS 3-hourly
    y0 = int(str(cfg["period"]["start"])[:4])
    y1 = int(str(cfg["period"]["end"])[:4])

    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    us = regions[(regions["country"] == "United States of America") &
                 (regions["level"] == "admin1")].copy()
    name2id = {r["name"]: int(r["region_id"]) for _, r in us.iterrows()}
    ids = {s: name2id[s] for s in STATES if s in name2id}
    log.info("state region_ids: %s", ids)

    # state land area (km^2) from the analysis geometries, equal-area reprojected
    us_ea = us.to_crs(epsg=EQUAL_AREA_EPSG)
    area_km2 = {r["name"]: float(g.area) / 1e6
                for r, g in zip(us_ea.to_dict("records"), us_ea.geometry)}

    summ = pd.read_csv(results_dir / "usa_summary.csv").set_index("name")

    table = process.load_zonal(cfg)
    table = table[table["region_id"].isin(ids.values())]

    monthly_rows, annual_rows, cap_data = [], [], []
    fig, axes = plt.subplots(1, len(STATES), figsize=(4.2 * len(STATES), 4.2),
                             sharey=False)
    for ax, state in zip(np.atleast_1d(axes), STATES):
        rid = ids[state]
        grp = table[table["region_id"] == rid]
        cf = analyze.capacity_factors(grp, cfg)
        s, w = cf["solar"].to_numpy(), cf["wind"].to_numpy()
        idx = cf.index
        scf, wcf = s.mean(), w.mean()
        alpha = float(summ.loc[state, "mix_alpha"])

        # normalized series (mean 1 each); mix mean 1 by construction
        ns, nw = s / scf, w / wcf
        nmix = alpha * ns + (1 - alpha) * nw

        cons = eia_monthly_clim(ABBR[state], y0, y1)          # MWh / month, clim
        annual_cons = float(cons.sum())                        # MWh / yr
        pbar = annual_cons / HOURS_PER_YEAR                    # mean power, MW

        # energy-parity scale: normalized series * (annual_cons / spy) -> MWh per 3h step,
        # so a mean year integrates to annual_cons. Aggregate to climatological months.
        per_step = annual_cons / spy
        df = pd.DataFrame({"solar": ns * per_step, "wind": nw * per_step,
                           "mix": nmix * per_step}, index=idx)
        ym = df.groupby([idx.year, idx.month]).sum()           # monthly totals per year
        clim = ym.groupby(level=1).mean().reindex(range(1, 13))  # climatological month (MWh)

        for m in range(1, 13):
            monthly_rows.append({
                "state": state, "month": m,
                "consumption_GWh": cons[m] / 1e3,
                "solar_GWh": clim.loc[m, "solar"] / 1e3,
                "wind_GWh": clim.loc[m, "wind"] / 1e3,
                "mix_GWh": clim.loc[m, "mix"] / 1e3,
            })

        # nameplate (power, GW) and its RATED annual energy (TWh/yr = GW * 8760 h),
        # the latter in the SAME unit as consumption. rated/consumption = 1/CF.
        solar_gw = pbar / scf / 1e3
        wind_gw = pbar / wcf / 1e3
        mix_solar_gw = alpha * pbar / scf / 1e3
        mix_wind_gw = (1 - alpha) * pbar / wcf / 1e3
        mix_total_gw = mix_solar_gw + mix_wind_gw
        gw_to_twh = HOURS_PER_YEAR / 1e3                       # GW -> TWh/yr at full output

        # --- land use (NREL densities) -> km^2 and % of state land --------------
        land = area_km2[state]
        mix_solar_km2 = mix_solar_gw * 1e3 / SOLAR_MW_PER_KM2
        mix_wind_project_km2 = mix_wind_gw * 1e3 / WIND_PROJECT_MW_PER_KM2
        mix_wind_direct_km2 = mix_wind_gw * 1e3 / WIND_DIRECT_MW_PER_KM2
        # "full" counts wind spacing area; "direct" counts only land removed from use
        mix_land_full_km2 = mix_solar_km2 + mix_wind_project_km2
        mix_land_direct_km2 = mix_solar_km2 + mix_wind_direct_km2
        annual_rows.append({
            "state": state, "abbr": ABBR[state],
            "annual_consumption_TWh": annual_cons / 1e6,
            "solar_cf": scf, "wind_cf": wcf, "mix_alpha": alpha,
            # nameplate as power
            "solar_nameplate_GW": solar_gw,
            "wind_nameplate_GW": wind_gw,
            "mix_solar_nameplate_GW": mix_solar_gw,
            "mix_wind_nameplate_GW": mix_wind_gw,
            "mix_total_nameplate_GW": mix_total_gw,
            # nameplate as RATED annual energy (TWh/yr) -- same unit as consumption
            "solar_nameplate_TWh": solar_gw * gw_to_twh,
            "wind_nameplate_TWh": wind_gw * gw_to_twh,
            "mix_solar_nameplate_TWh": mix_solar_gw * gw_to_twh,
            "mix_wind_nameplate_TWh": mix_wind_gw * gw_to_twh,
            "mix_total_nameplate_TWh": mix_total_gw * gw_to_twh,
            "mix_f_adj": float(summ.loc[state, "mix_f_adj"]),
            "mix_S_tot_pct_cons": float(summ.loc[state, "mix_s_tot_pct"]),
            # land use of the optimal mix
            "state_land_km2": land,
            "mix_solar_land_km2": mix_solar_km2,
            "mix_wind_project_km2": mix_wind_project_km2,
            "mix_wind_direct_km2": mix_wind_direct_km2,
            "mix_land_full_km2": mix_land_full_km2,
            "mix_land_direct_km2": mix_land_direct_km2,
            "mix_land_full_pct": 100.0 * mix_land_full_km2 / land,
            "mix_land_direct_pct": 100.0 * mix_land_direct_km2 / land,
        })

        # stash for the 1%-land-cap analysis (monthly MWh series, parity-sized)
        cap_data.append({
            "state": state, "abbr": ABBR[state],
            "cons": cons.copy(), "mix_monthly": clim["mix"].copy(),
            "annual_cons": annual_cons,
            "land_direct_km2": mix_land_direct_km2,
            "land_full_km2": mix_land_full_km2, "area": land,
        })

        # plot (TWh/month)
        mm = clim / 1e6
        ax.plot(MONTHS, cons.to_numpy() / 1e6, "k-", lw=2.4, label="EIA consumption")
        ax.plot(MONTHS, mm["solar"], color="#f1a340", lw=1.6, label="solar-only")
        ax.plot(MONTHS, mm["wind"], color="#67a9cf", lw=1.6, label="wind-only")
        ax.plot(MONTHS, mm["mix"], color="#1a9850", lw=2.0,
                label=f"mix (α={alpha:.2f})")
        ax.set_title(f"{state}\n{annual_cons/1e6:.1f} TWh/yr", fontsize=10)
        ax.set_xticks(range(0, 12, 2))
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.3)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("energy per month (TWh)")
    np.atleast_1d(axes)[-1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Actual monthly generation (annual-energy parity, GLDAS 2000-2025) "
                 "vs real EIA consumption", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = figdir / "actual_supply_vs_eia.png"
    fig.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close(fig)

    md = pd.DataFrame(monthly_rows).round(2)
    ad = pd.DataFrame(annual_rows).round(4)
    md.to_csv(results_dir / "actual_supply_vs_eia_monthly.csv", index=False)
    ad.to_csv(results_dir / "actual_supply_vs_eia_summary.csv", index=False)
    log.info("wrote %s", out_png)

    # --- land-use figure: % of state land for the optimal mix -----------------
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(ad))
    ax2.bar(x - 0.2, ad["mix_land_full_pct"], 0.4, color="#bdbdbd",
            label="full project area (incl. wind spacing, stays farmable)")
    ax2.bar(x + 0.2, ad["mix_land_direct_pct"], 0.4, color="#1a9850",
            label="land actually removed (solar array + wind footprint)")
    ax2.axhline(1.0, color="crimson", ls="--", lw=1.2, label="NREL ~1% of land")
    for xi, (f, d) in enumerate(zip(ad["mix_land_full_pct"], ad["mix_land_direct_pct"])):
        ax2.text(xi - 0.2, f, f"{f:.1f}%", ha="center", va="bottom", fontsize=8)
        ax2.text(xi + 0.2, d, f"{d:.2f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_ylim(0, float(ad["mix_land_full_pct"].max()) * 1.15)
    ax2.set_xticks(x); ax2.set_xticklabels(ad["abbr"])
    ax2.set_ylabel("% of state land area")
    ax2.set_title("Land needed for the optimal solar+wind mix\n"
                  "(annual-energy parity, NREL densities: solar 30, wind 3 / 100 MW/km²)",
                  fontsize=11)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.grid(axis="y", alpha=0.3)
    fig2.tight_layout()
    out_png2 = figdir / "actual_supply_land_use.png"
    fig2.savefig(out_png2, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    log.info("wrote %s", out_png2)

    # --- 1%-of-land CAP: what supply does 1% of the state deliver? ------------
    # "Use all the land" => full PROJECT-AREA density binds (wind 3 MW/km^2 spacing,
    # solar 30 MW/km^2), since turbines can't be packed tighter than wake spacing.
    # Keep the storage-optimal alpha* fixed (one variable = the land cap). The
    # parity-sized mix produces exactly `consumption` from `land_full_km2`, so
    # 1% of land yields supply_frac = (0.01*area)/land_full_km2 of consumption.
    # If supply_frac < 1, the state is ENERGY-short within 1% -- no storage can fix it.
    cap_rows = []
    fig3, axes3 = plt.subplots(1, len(cap_data), figsize=(4.2 * len(cap_data), 4.2))
    for ax, cd in zip(np.atleast_1d(axes3), cap_data):
        budget = 0.01 * cd["area"]                         # 1% of state land, km^2
        f_full = budget / cd["land_full_km2"]              # "use all the land" density
        f_direct = budget / cd["land_direct_km2"]          # if wind spacing stays usable
        sup_full = cd["mix_monthly"] * f_full              # MWh/month from 1% land
        sup_direct = cd["mix_monthly"] * f_direct
        cap_rows.append({
            "state": cd["state"], "abbr": cd["abbr"],
            "annual_consumption_TWh": cd["annual_cons"] / 1e6,
            "supply_1pct_full_TWh": cd["annual_cons"] * f_full / 1e6,
            "supply_pct_of_cons_full": 100.0 * f_full,
            "supply_1pct_direct_TWh": cd["annual_cons"] * f_direct / 1e6,
            "supply_pct_of_cons_direct": 100.0 * f_direct,
        })
        ax.plot(MONTHS, cd["cons"].to_numpy() / 1e6, "k-", lw=2.4, label="EIA consumption")
        ax.plot(MONTHS, sup_full.to_numpy() / 1e6, color="#1a9850", lw=2.0,
                label=f"supply, 1% land ({100*f_full:.0f}% of cons.)")
        ax.fill_between(range(12), sup_full.to_numpy() / 1e6, color="#1a9850", alpha=0.18)
        ax.set_title(f"{cd['state']}", fontsize=10)
        ax.set_xticks(range(0, 12, 2)); ax.tick_params(axis="x", labelsize=8)
        ax.grid(alpha=0.3)
        if ax is np.atleast_1d(axes3)[0]:
            ax.set_ylabel("energy per month (TWh)")
        ax.legend(fontsize=7, loc="upper right")
    fig3.suptitle("Supply from ≤1% of state land (optimal mix, wind+solar at full "
                  "land-use density) vs real EIA consumption", fontsize=12, weight="bold")
    fig3.tight_layout(rect=(0, 0, 1, 0.95))
    out_png3 = figdir / "supply_1pct_land_vs_eia.png"
    fig3.savefig(out_png3, dpi=200, bbox_inches="tight")
    plt.close(fig3)
    cap_df = pd.DataFrame(cap_rows).round(3)
    cap_df.to_csv(results_dir / "supply_1pct_land_vs_eia.csv", index=False)
    log.info("wrote %s", out_png3)

    print("\n=== Supply from <=1% of state land vs consumption ===")
    print(cap_df.to_string(index=False))

    print("\n=== Annual summary (energy-parity sizing) ===")
    print(ad[["state", "annual_consumption_TWh", "mix_alpha",
              "mix_total_nameplate_GW", "mix_total_nameplate_TWh"]].to_string(index=False))
    print("\n=== Land use of the optimal mix (NREL densities) ===")
    print(ad[["state", "state_land_km2", "mix_land_full_km2", "mix_land_full_pct",
              "mix_land_direct_km2", "mix_land_direct_pct"]].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
