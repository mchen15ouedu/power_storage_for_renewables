"""Conversion of meteorological forcings to power-generation capacity factors.

Solar (paper Eq. 1): piecewise-linear capacity factor between a starting
threshold R_s and a plateauing threshold R_p, applied to the 3-hourly
shortwave flux.

Wind:
  1. Prandtl-von Karman log-profile correction of the 10 m wind to hub height
     (paper Eqs. 2-3; z_d = 0.7 H_veg, z_0 = 0.1 H_veg).
  2. Air-density correction via the IEC 61400-12-1 equivalent wind speed
     u_eq = u_hub (rho/rho0)^(1/3) with rho0 = 1.225 kg m-3, where the moist
     air density rho = P / (R_d T_v) uses the virtual temperature
     T_v = T (1 + 0.608 q) (Wallace & Hobbs 2006). Surface pressure is
     barometrically adjusted to hub height. See also Ulazia et al. (2019),
     Energy 187:115938 for the global relevance of density seasonality.
  3. The same piecewise-linear capacity factor (paper Eq. 1).
"""

from __future__ import annotations

import numpy as np

R_DRY = 287.05    # J kg-1 K-1
GRAVITY = 9.80665  # m s-2


def wind_at_hub(u_ref: np.ndarray, ref_height: float, hub_height: float,
                veg_height: float) -> np.ndarray:
    """Scale wind speed from the reference height to hub height (paper Eq. 3)."""
    zd = 0.7 * veg_height
    z0 = 0.1 * veg_height
    return u_ref * np.log((hub_height - zd) / z0) / np.log((ref_height - zd) / z0)


def virtual_temperature(tair: np.ndarray, qair: np.ndarray) -> np.ndarray:
    """T_v = T (1 + 0.608 q) for specific humidity q [kg/kg]."""
    return tair * (1.0 + 0.608 * qair)


def air_density(psurf: np.ndarray, tair: np.ndarray, qair: np.ndarray,
                dz: float = 0.0) -> np.ndarray:
    """Moist air density [kg m-3] at dz metres above the surface.

    rho = P_hub / (R_d T_v), with the surface pressure barometrically adjusted
    to hub height: P_hub = P exp(-g dz / (R_d T_v)).
    """
    tv = virtual_temperature(tair, qair)
    p_hub = psurf * np.exp(-GRAVITY * dz / (R_DRY * tv))
    return p_hub / (R_DRY * tv)


def density_equivalent_wind(u_hub: np.ndarray, rho: np.ndarray, rho0: float) -> np.ndarray:
    """IEC 61400-12-1 air-density normalization: u_eq = u (rho/rho0)^(1/3)."""
    return u_hub * np.cbrt(rho / rho0)


def capacity_factor(x: np.ndarray, start: float, plateau: float) -> np.ndarray:
    """Piecewise-linear capacity factor (paper Eq. 1): 0 below R_s, 1 above R_p."""
    return np.clip((x - start) / (plateau - start), 0.0, 1.0)


def solar_cf(swdown: np.ndarray, cfg: dict) -> np.ndarray:
    s = cfg["energy"]["solar"]
    return capacity_factor(swdown, s["start_wm2"], s["plateau_wm2"])


def wind_cf(wind10: np.ndarray, cfg: dict,
            tair: np.ndarray | None = None,
            psurf: np.ndarray | None = None,
            qair: np.ndarray | None = None) -> np.ndarray:
    w = cfg["energy"]["wind"]
    dz = w["hub_height_m"] - w["ref_height_m"]
    u = wind_at_hub(wind10, w["ref_height_m"], w["hub_height_m"], w["veg_height_m"])
    if w.get("density_correction", False) and tair is not None:
        rho = air_density(psurf, tair, qair, dz=dz)
        u = density_equivalent_wind(u, rho, w["rho0"])
    return capacity_factor(u, w["start_ms"], w["plateau_ms"])
