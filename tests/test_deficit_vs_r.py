"""Verify the vectorized deficit analysis against a literal port of Deficit.R,
plus sanity checks of the air-density correction and the added metrics.

The reference implementation below mirrors the R loops line by line
(Deficit.R lines 27-67) at daily resolution (dt = 1/365, as in the paper).

Run:  python tests/test_deficit_vs_r.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gldas_storage import metrics
from gldas_storage.deficit import analyze_region
from gldas_storage.energy import air_density, density_equivalent_wind

STORAGE_CFG = {"up_efficiency": 0.9, "down_efficiency": 0.8,
               "annual_decay": 0.7, "max_iterations": 20, "tolerance": 1e-3}
DT = 1.0 / 365.0


def reference_r_port(cf, dates, demand):
    """Literal port of the loops in Deficit.R function.MakePlot."""
    up, down, ann = 0.9, 0.8, 0.7
    day_eff = ann ** DT
    n = len(cf)

    annual = pd.Series(cf, index=dates).groupby(dates.year).mean()
    net_factor = annual.mean() / annual.min()
    supply = cf / cf.mean() * net_factor

    net_deficit = np.zeros(n)
    for i in range(n):
        d = 0.0 if i == 0 else net_deficit[i - 1]
        net_deficit[i] = d + (demand[i] - supply[i]) * DT
        if net_deficit[i] < 0.0:
            net_deficit[i] = 0.0
    s_net = net_deficit.max()

    tot_storage = s_net
    tot_factor = net_factor + s_net * ((1 - up) + (1 - down) + (1 - ann))
    tot_loss = tot_factor - net_factor
    tot_deficit = np.zeros(n)
    loss = np.zeros(n)
    for _j in range(20):
        tot_factor = net_factor * (1 + tot_loss)
        tot_supply = cf / cf.mean() * tot_factor
        for i in range(n):
            d = 0.0 if i == 0 else tot_deficit[i - 1]
            if demand[i] > tot_supply[i]:
                tot_deficit[i] = d + (demand[i] - tot_supply[i]) / down * DT
                loss[i] = (demand[i] - tot_supply[i]) * (1 - down) * DT
            else:
                tot_deficit[i] = d + (demand[i] - tot_supply[i]) * up * DT
                loss[i] = (demand[i] - tot_supply[i]) * (up - 1) * DT
            if tot_deficit[i] < 0.0:
                tot_deficit[i] = 0.0
            loss[i] += tot_deficit[i] * (1 - day_eff)
        old = tot_storage
        tot_storage = tot_deficit.max()
        tot_loss = loss.mean() / DT
        if abs(tot_storage - old) < 0.001:
            break

    storage = np.zeros(n)
    for i in range(n):
        s = tot_storage if i == 0 else storage[i - 1]
        if tot_supply[i] < demand[i]:
            storage[i] = s + (tot_supply[i] - demand[i]) / down * DT
        else:
            storage[i] = s + (tot_supply[i] - demand[i]) * up * DT
        if storage[i] > tot_storage:
            storage[i] = tot_storage
        else:
            storage[i] *= day_eff
    return dict(net_factor=net_factor, tot_factor=tot_factor,
                s_net=s_net, s_tot=tot_storage, storage=storage)


def synthetic_cf(kind, n_years=20, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("1995-01-01", periods=n_years * 366, freq="D")
    dates = dates[~((dates.month == 2) & (dates.day == 29))][: n_years * 365]
    doy = dates.dayofyear.to_numpy()
    if kind == "solar":  # strong seasonal cycle, modest noise
        base = 0.45 + 0.30 * np.cos(2 * np.pi * (doy - 172) / 365)
        cf = np.clip(base + rng.normal(0, 0.08, len(dates)), 0.0, 1.0)
    else:  # wind: weak seasonality, heavy noise, inter-annual swings
        yr_amp = (1 + 0.10 * rng.standard_normal(n_years)).repeat(365)[: len(dates)]
        base = 0.30 + 0.05 * np.cos(2 * np.pi * (doy - 30) / 365)
        cf = np.clip(base * yr_amp + rng.normal(0, 0.12, len(dates)), 0.0, 1.0)
    return cf, pd.DatetimeIndex(dates)


def check(name, ok, failures):
    print(f"  {name}: {'OK' if ok else 'MISMATCH'}")
    return failures + (not ok)


def main():
    failures = 0
    for kind in ("solar", "wind"):
        cf, dates = synthetic_cf(kind)
        demand = np.ones(len(dates))
        ref = reference_r_port(cf, dates, demand)
        res = analyze_region(cf, dates, demand, STORAGE_CFG,
                             steps_per_year=365, simulate=True)
        print(f"--- {kind} (R equivalence, dt=1/365) ---")
        for name, want, got in [("f", ref["net_factor"], res.net_factor),
                                ("f_adj", ref["tot_factor"], res.tot_factor),
                                ("S_net", ref["s_net"], res.s_net),
                                ("S_tot", ref["s_tot"], res.s_tot)]:
            failures = check(f"{name} {want:.6f} vs {got:.6f}",
                             np.isclose(want, got, rtol=1e-10), failures)
        failures = check("storage trace", np.allclose(
            ref["storage"], res.series["storage"].to_numpy(), rtol=1e-10), failures)

    print("--- physics sanity ---")
    # ISA sea level, dry air, 15 C -> 1.225 kg/m3
    rho = air_density(np.array([101325.0]), np.array([288.15]), np.array([0.0]))
    failures = check(f"air density {rho[0]:.4f} ~ 1.225", abs(rho[0] - 1.225) < 0.002, failures)
    # warm humid low pressure -> lighter air -> lower equivalent wind
    rho_warm = air_density(np.array([95000.0]), np.array([303.15]), np.array([0.020]))
    u_eq = density_equivalent_wind(np.array([10.0]), rho_warm, 1.225)
    failures = check(f"u_eq {u_eq[0]:.3f} < 10 for light air", u_eq[0] < 10.0, failures)

    print("--- metrics sanity ---")
    cf_s, dates = synthetic_cf("solar")
    cf_w, _ = synthetic_cf("wind", seed=7)
    demand = np.ones(len(dates))
    best = metrics.mix_sweep(cf_s, cf_w, dates, demand, STORAGE_CFG, 365, step=0.25)
    pure_s = analyze_region(cf_s, dates, demand, STORAGE_CFG, 365, simulate=False).s_tot
    pure_w = analyze_region(cf_w, dates, demand, STORAGE_CFG, 365, simulate=False).s_tot
    failures = check(
        f"mix S_tot {best['mix_s_tot']:.4f} <= min(pure) {min(pure_s, pure_w):.4f}",
        best["mix_s_tot"] <= min(pure_s, pure_w) + 1e-12, failures)
    df = metrics.dunkelflaute(cf_w, 365, 0.5, 30)
    failures = check(f"dunkelflaute finite: {df}", np.isfinite(df["flaute_days"]), failures)
    slope, p = metrics.mann_kendall_sen(np.arange(20) + np.random.default_rng(1).normal(0, 0.1, 20))
    failures = check(f"MK detects trend (slope {slope:.3f}, p {p:.2e})",
                     0.8 < slope < 1.2 and p < 0.01, failures)

    print("\nALL CHECKS PASSED" if failures == 0 else f"\n{failures} CHECK(S) FAILED")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
