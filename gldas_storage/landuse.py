"""Real-units land-use reckoning for the USA pipeline (NREL ~1% land constraint).

Single source of truth for: NREL capacity densities, the chosen WIND = DIRECT
FOOTPRINT convention (turbine pads+roads only; the ~97% spacing between turbines
stays farmable/grazable, so it is NOT counted as land used), state areas, and the
"cap sizing at 1% of land" operating rule.

Operating rule (user-confirmed 2026-06-17):
  * size capacity to meet demand with the framework's f_adj overbuild;
  * its land footprint = solar array + wind DIRECT footprint;
  * if footprint <= 1% of state land  -> FEASIBLE (report storage_TWh, land_pct);
  * if footprint  > 1% of state land  -> LAND-CONSTRAINED: scale the build down to
    exactly 1% of land, so generation drops to cap_scale*f_adj of consumption;
    report the 1%-limited supply_TWh and the annual shortfall.

All energy is real (TWh); no normalized fractions for the USA.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd

# --- NREL capacity densities --------------------------------------------------
# Solar PV array footprint: NREL Ong 2013 (TP-6A20-56290) ~7 acres/MWac; Nature
#   2025 measured 24.7 MW/km^2. Use 30 MW/km^2.
# Wind: NREL Denholm 2009 (TP-6A2-45834): TOTAL project area ~34.5 ha/MW (3 MW/km^2)
#   but turbines/roads/pads PERMANENTLY remove only ~1 ha/MW (100 MW/km^2 ~ 3% of
#   project area). We count the DIRECT footprint (spacing reused) -- chosen convention.
SOLAR_MW_PER_KM2 = 30.0
WIND_DIRECT_MW_PER_KM2 = 100.0      # chosen: direct footprint, spacing farmable
WIND_PROJECT_MW_PER_KM2 = 3.0       # full project area, kept for reference only

LAND_FRACTION_CAP = 0.01            # NREL ~1% of land (default scenario)
# Land-cap scenarios (key, fraction, label), policy-anchored. Only this fraction
# varies between scenarios; everything else is held fixed.
#   1%    -- NREL "<1% of land" footprint.
#   1.88% -- 2x the U.S. onshore oil & gas surface footprint (~0.94% of land).
#   5%    -- all U.S. onshore oil & gas PRODUCTION land (federal + private + state).
LAND_CAPS = [
    ("1pct", 0.01, "1% of land (NREL)"),
    ("oilgas2x", 0.0188, "1.88% (2× onshore oil & gas)"),
    ("oilgas_all", 0.05, "5% (all onshore oil & gas land)"),
]
HOURS_PER_YEAR = 8760.0
EQUAL_AREA_EPSG = 5070              # USA Contiguous Albers Equal Area (km^2-accurate)


def state_areas_km2(gdf: gpd.GeoDataFrame, epsg: int = EQUAL_AREA_EPSG) -> dict:
    """{region name -> land area km^2} via equal-area reprojection."""
    ea = gdf.to_crs(epsg=epsg)
    return {r["name"]: float(g.area) / 1e6
            for r, g in zip(ea.to_dict("records"), ea.geometry)}


@dataclass
class LandResult:
    mean_power_MW: float            # annual_consumption / 8760
    solar_nameplate_GW: float       # demand-meeting build (f_adj-sized), per leg
    wind_nameplate_GW: float
    land_km2: float                 # footprint of the demand-meeting build
    land_pct: float                 # as % of state land
    feasible: bool                  # land_pct <= 1%
    cap_scale: float                # 1.0 if feasible else (1% land)/(needed land) < 1
    storage_TWh: float              # s_tot * annual_consumption (if feasible)
    supply_capped_TWh: float        # 1%-land annual generation (if not feasible)
    shortfall_TWh: float            # max(0, consumption - capped supply)


def assess(alpha: float, f_adj: float, solar_cf: float, wind_cf: float,
           annual_consumption_TWh: float, state_area_km2: float,
           s_tot_fraction: float,
           solar_density: float = SOLAR_MW_PER_KM2,
           wind_density: float = WIND_DIRECT_MW_PER_KM2,
           land_overlap_k: float = 0.0,
           land_fraction_cap: float = LAND_FRACTION_CAP) -> LandResult:
    """Cap-at-(land_fraction_cap) assessment for one region's optimal mix, real units.

    ``land_overlap_k`` (co-location) lets the same footprint host (1+k) generation,
    so the land needed for the demand-meeting build shrinks by 1/(1+k).
    ``land_fraction_cap`` is the allowed share of state land (0.01 = NREL 1%,
    0.0188 = 2x onshore oil & gas).
    """
    mean_power_MW = annual_consumption_TWh * 1e6 / HOURS_PER_YEAR    # TWh->MWh / h
    # demand-meeting nameplate per leg (MW): mean_supply_leg = nameplate*cf = share*P*f_adj
    solar_mw = alpha * mean_power_MW * f_adj / solar_cf if solar_cf > 0 else 0.0
    wind_mw = (1 - alpha) * mean_power_MW * f_adj / wind_cf if wind_cf > 0 else 0.0
    land = (solar_mw / solar_density + wind_mw / wind_density) / (1.0 + land_overlap_k)
    cap = land_fraction_cap * state_area_km2                          # allowed land, km^2
    land_pct = 100.0 * land / state_area_km2 if state_area_km2 > 0 else float("nan")
    feasible = land <= cap
    cap_scale = 1.0 if feasible else cap / land
    storage_TWh = s_tot_fraction * annual_consumption_TWh
    # capped build generates cap_scale * f_adj * consumption (f_adj = mean overbuild)
    supply_capped_TWh = cap_scale * f_adj * annual_consumption_TWh
    shortfall_TWh = max(0.0, annual_consumption_TWh - supply_capped_TWh)
    return LandResult(
        mean_power_MW=mean_power_MW,
        solar_nameplate_GW=solar_mw / 1e3, wind_nameplate_GW=wind_mw / 1e3,
        land_km2=land, land_pct=land_pct, feasible=feasible, cap_scale=cap_scale,
        storage_TWh=storage_TWh if feasible else float("nan"),
        supply_capped_TWh=float("nan") if feasible else supply_capped_TWh,
        shortfall_TWh=0.0 if feasible else shortfall_TWh)
