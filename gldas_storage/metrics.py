"""Analysis methods beyond the original R scripts.

  * solar+wind mix optimization: sweep the blend fraction, find the mix that
    minimizes the total storage requirement;
  * temporal decomposition: S_tot from the 3-hourly series vs from its daily
    means separates the diurnal storage tier from the seasonal one;
  * dunkelflaute statistics: longest low-resource spell and the deepest
    rolling-window energy gap;
  * Mann-Kendall trend test + Sen slope on annual capacity factors;
  * pooled (interconnected) regions: area-weighted aggregation of the zonal
    forcings to continent/world level before the capacity-factor transform.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import deficit


# --------------------------------------------------------------------------
# solar + wind mix
# --------------------------------------------------------------------------

def mix_sweep(cf_solar: np.ndarray, cf_wind: np.ndarray, dates: pd.DatetimeIndex,
              demand: np.ndarray, storage_cfg: dict, steps_per_year: int,
              step: float = 0.05) -> dict:
    """Sweep alpha = solar share of mean generation; return the storage-optimal mix.

    The blend is built from mean-normalized series so alpha is the fraction of
    delivered energy coming from solar: mix = alpha*S/mean(S) + (1-alpha)*W/mean(W).
    """
    alphas = np.round(np.arange(0.0, 1.0 + step / 2, step), 10)
    ns = cf_solar / cf_solar.mean()
    nw = cf_wind / cf_wind.mean()
    best = None
    for alpha in alphas:
        mix = alpha * ns + (1 - alpha) * nw
        res = deficit.analyze_region(mix, dates, demand, storage_cfg,
                                     steps_per_year, simulate=False)
        if best is None or res.s_tot < best["mix_s_tot"]:
            best = {"mix_alpha": float(alpha), "mix_s_tot": res.s_tot,
                    "mix_f_adj": res.tot_factor}
    return best


# --------------------------------------------------------------------------
# dunkelflaute / low-resource event statistics
# --------------------------------------------------------------------------

def _longest_run(mask: np.ndarray) -> int:
    """Length of the longest consecutive run of True."""
    if not mask.any():
        return 0
    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return int((edges[1::2] - edges[::2]).max())


def dunkelflaute(cf: np.ndarray, steps_per_year: int, cf_frac: float,
                 window_days: int) -> dict:
    """Low-resource spell statistics on the mean-normalized resource series.

    longest_days : longest consecutive spell with cf < cf_frac * mean(cf);
    gap_max      : deepest ``window_days`` rolling-mean shortfall below the
                   mean resource level (fraction of mean supply, 0..1).
    """
    steps_per_day = steps_per_year / 365.0
    norm = cf / cf.mean()
    longest = _longest_run(norm < cf_frac) / steps_per_day

    window = max(1, int(round(window_days * steps_per_day)))
    shortfall = np.maximum(0.0, 1.0 - norm)
    rolling = pd.Series(shortfall).rolling(window, min_periods=window).mean()
    gap = float(np.nanmax(rolling.to_numpy()))
    return {"flaute_days": float(longest), f"gap{window_days}d_max": gap}


# --------------------------------------------------------------------------
# trends (Mann-Kendall + Sen slope) on annual means
# --------------------------------------------------------------------------

def mann_kendall_sen(values: np.ndarray) -> tuple[float, float]:
    """Return (sen_slope_per_year, two-sided p) of the Mann-Kendall test."""
    x = np.asarray(values, dtype=float)
    n = x.size
    if n < 8:
        return np.nan, np.nan
    i, j = np.triu_indices(n, k=1)
    diffs = x[j] - x[i]
    s = np.sign(diffs).sum()
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    z = (s - np.sign(s)) / math.sqrt(var_s) if s != 0 else 0.0
    p = math.erfc(abs(z) / math.sqrt(2.0))
    slope = float(np.median(diffs / (j - i)))
    return slope, p


def cf_trend(cf: np.ndarray, dates: pd.DatetimeIndex) -> dict:
    annual = pd.Series(cf, index=dates).groupby(dates.year).mean()
    slope, p = mann_kendall_sen(annual.to_numpy())
    mean = annual.mean()
    pct_decade = slope / mean * 1000.0 if mean > 0 else np.nan  # % per decade
    return {"trend_pct_decade": pct_decade, "trend_p": p}


# --------------------------------------------------------------------------
# pooled (interconnected) regions
# --------------------------------------------------------------------------

def pooled_tables(table: pd.DataFrame, regions: pd.DataFrame,
                  value_cols: list[str]) -> pd.DataFrame:
    """Aggregate zonal forcings to continent + world pools.

    Weighted by each region's land area (``weight_sum``), which reconstructs
    the zonal mean over the union of the member regions. Pooling is done at
    forcing level (before the capacity-factor transform), consistent with how
    individual regions are treated. Returns a long table with a ``pool``
    column instead of ``region_id``.
    """
    meta = regions[["region_id", "continent", "weight_sum"]].copy()
    merged = table.merge(meta, on="region_id", how="left")
    merged = merged[merged["weight_sum"] > 0]

    def _aggregate(df: pd.DataFrame, label: str) -> pd.DataFrame:
        # vectorized weighted mean: sum(w*x)/sum(w) per time
        w = df["weight_sum"].to_numpy()
        g = df.assign(**{f"_w_{c}": df[c] * w for c in value_cols}, _w=w).groupby("time")
        sums = g[[f"_w_{c}" for c in value_cols] + ["_w"]].sum()
        agg = pd.DataFrame({c: sums[f"_w_{c}"] / sums["_w"] for c in value_cols})
        agg["pool"] = label
        return agg.reset_index()

    pools = [_aggregate(grp, cont) for cont, grp in merged.groupby("continent")]
    pools.append(_aggregate(merged, "World"))
    return pd.concat(pools, ignore_index=True)
