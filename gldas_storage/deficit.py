"""Modified cumulative surplus/deficit (S-curve) analysis -- port of Deficit.R.

Implements paper Section 2.4 (Eqs. 4-9):
  * excess installation factor  f = mean(annual means) / min(annual means)   (Eq. 4)
  * net cumulative deficit / storage capacity S_net                          (Eq. 5)
  * loss-adjusted cumulative deficit with recharge (k_R), discharge (k_D)
    and per-step storage decay (k_dS = k_aS ** dt) efficiencies              (Eq. 6-7)
  * iterative adjustment of the excess installation factor f_adj             (Eq. 8-9)
  * storage-operation simulation that mirrors the deficits (Figure 6 right)

The formulation is time-step agnostic: dt = 1/steps_per_year (365 for daily
series as in the paper, 2920 for 3-hourly GLDAS series). All quantities are
normalized: demand averages 1, supply is the capacity-factor series normalized
by its mean and scaled by the (adjusted) excess installation factor, and
storage is expressed as a fraction of annual energy consumption.

The clamped-at-zero recursion d_i = max(0, d_{i-1} + x_i) of Deficit.R is
vectorized via the running-minimum identity d = cumsum(x) - min(0, runmin(cumsum(x))).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def positive_running_sum(x: np.ndarray) -> np.ndarray:
    """Vectorized d_i = max(0, d_{i-1} + x_i) with d_0 = 0."""
    cs = np.cumsum(x)
    return cs - np.minimum.accumulate(np.minimum(cs, 0.0))


@dataclass
class RegionResult:
    capacity_factor: float      # long-term mean of the capacity-factor series
    net_factor: float           # excess installation factor f (Eq. 4)
    tot_factor: float           # adjusted excess installation factor f_adj (Eq. 8-9)
    s_net: float                # storage capacity, fraction of annual consumption (Eq. 5)
    s_tot: float                # total storage capacity incl. losses (Eq. 6-7)
    iterations: int
    converged: bool
    series: pd.DataFrame | None = field(repr=False, default=None)


def make_demand(cfg: dict, dates: pd.DatetimeIndex,
                tair: np.ndarray | None = None,
                monthly: np.ndarray | None = None) -> np.ndarray:
    """Normalized demand series (mean 1).

    flat       -- constant (the paper's assumption);
    monthly    -- a 12-value profile mapped to months;
    degree_day -- all-electric heating/cooling scenario from the region's own
                  air temperature: D = base + HDD + CDD around the balance
                  temperature, scaled so the temperature-independent part is
                  ``base_fraction`` of the mean and the series averages 1.
    """
    profile = cfg["demand"]["profile"]
    if profile == "monthly" and monthly is not None:
        demand = np.asarray(monthly, dtype=float)[dates.month.to_numpy() - 1]
        return demand / demand.mean()
    if profile == "degree_day":
        if tair is None:
            raise ValueError("degree_day demand requires the tair series")
        balance = cfg["demand"]["balance_temp_k"]
        base_frac = cfg["demand"]["base_fraction"]
        dd = np.abs(np.asarray(tair, dtype=float) - balance)  # HDD + CDD per step
        mean_dd = dd.mean()
        if mean_dd <= 0:
            return np.ones(len(dates))
        demand = base_frac + (1.0 - base_frac) * dd / mean_dd
        return demand / demand.mean()
    return np.ones(len(dates))


def analyze_region(cf: np.ndarray, dates: pd.DatetimeIndex, demand: np.ndarray,
                   storage_cfg: dict, steps_per_year: int,
                   simulate: bool = True) -> RegionResult:
    """Run the full net + loss-adjusted cumulative deficit analysis for one region.

    storage_cfg holds up_efficiency, down_efficiency, annual_decay,
    max_iterations, tolerance (the ``storage`` block of the config, or an
    archetype with max_iterations/tolerance merged in).
    """
    dt = 1.0 / steps_per_year
    up = storage_cfg["up_efficiency"]
    down = storage_cfg["down_efficiency"]
    ann_decay = storage_cfg["annual_decay"]
    step_eff = ann_decay ** dt

    cf = np.asarray(cf, dtype=float)
    mean_cf = cf.mean()
    if not np.isfinite(mean_cf) or mean_cf <= 0.0:
        return RegionResult(0.0, np.nan, np.nan, np.nan, np.nan, 0, False)

    # --- net analysis (no losses), Deficit.R lines 18-33 -------------------
    annual = pd.Series(cf, index=dates).groupby(dates.year).mean()
    net_factor = annual.mean() / annual.min()
    supply = cf / mean_cf * net_factor
    net_deficit = positive_running_sum((demand - supply) * dt)
    s_net = net_deficit.max()

    # --- iterative loss-adjusted analysis, Deficit.R lines 34-58 -----------
    tot_storage = s_net
    tot_loss = s_net * ((1 - up) + (1 - down) + (1 - ann_decay))
    tot_factor = net_factor * (1 + tot_loss)
    tot_supply = supply
    tot_deficit = net_deficit
    converged = False
    iterations = 0
    for iterations in range(1, storage_cfg["max_iterations"] + 1):
        tot_factor = net_factor * (1 + tot_loss)
        tot_supply = cf / mean_cf * tot_factor
        diff = demand - tot_supply
        shortfall = diff > 0
        x = np.where(shortfall, diff / down, diff * up) * dt
        tot_deficit = positive_running_sum(x)
        loss = (np.where(shortfall, diff * (1 - down), diff * (up - 1)) * dt
                + tot_deficit * (1 - step_eff))
        old = tot_storage
        tot_storage = tot_deficit.max()
        tot_loss = loss.mean() / dt
        if abs(tot_storage - old) < storage_cfg["tolerance"]:
            converged = True
            break

    series = None
    if simulate:
        # --- storage operation mirroring the deficits, Deficit.R lines 60-67
        storage = _simulate_storage(tot_supply, demand, tot_storage,
                                    up, down, step_eff, dt)
        series = pd.DataFrame({
            "demand": demand, "supply": tot_supply,
            "deficit": tot_deficit, "storage": storage,
        }, index=dates)
    return RegionResult(mean_cf, net_factor, tot_factor, s_net, tot_storage,
                        iterations, converged, series)


def _simulate_storage(supply: np.ndarray, demand: np.ndarray, capacity: float,
                      up: float, down: float, step_eff: float, dt: float) -> np.ndarray:
    """Track the storage state starting from full (Deficit.R lines 60-67).

    Mirrors the R code exactly, including the possibility of (slightly)
    negative excursions when the sized storage falls short of a worst run --
    those dips are diagnostic and show up in the figures, as in the paper.
    """
    out = np.empty_like(supply)
    state = capacity
    for i in range(supply.size):
        d = supply[i] - demand[i]
        state += (d / down if d < 0.0 else d * up) * dt
        if state > capacity:
            state = capacity
        else:
            state *= step_eff
        out[i] = state
    return out
