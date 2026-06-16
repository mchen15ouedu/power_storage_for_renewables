"""Stage 1: 3-hourly GLDAS granules -> 3-hourly zonal means per region.

No gridded data is stored: each granule is reduced to area-weighted regional
means of the configured variables immediately, and written as one parquet per
month into ``paths.zonal_dir`` (columns: time, region_id, swdown, wind10,
tair, psurf, qair). Months whose parquet already exists are skipped, so the
stage is fully restartable.

Granule acquisition (``gldas.download_mode``):
  * ``local``       -- granules were pre-downloaded (e.g. wget from a GES DISC
    list file) and are found by filename under ``paths.raw_dir`` (recursive);
  * ``earthaccess`` -- search + download each month via the earthaccess API.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr

log = logging.getLogger(__name__)


def month_range(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(pd.Timestamp(start).normalize().replace(day=1), end, freq="MS")


def monthly_file(zonal_dir: Path, month: pd.Timestamp) -> Path:
    return zonal_dir / f"zonal_{month:%Y%m}.parquet"


def local_month_files(cfg: dict, month: pd.Timestamp) -> list[Path]:
    """Find already-downloaded granules for one month under raw_dir (recursive)."""
    raw_dir = cfg["paths"]["raw_dir"]
    ext = cfg["gldas"].get("file_ext", "nc4")
    pattern = f"{cfg['gldas']['short_name']}.A{month:%Y%m}*.{ext}"
    return sorted(raw_dir.rglob(pattern))


def download_month(cfg: dict, month: pd.Timestamp) -> list[Path]:
    """Download all granules of one month via earthaccess; returns local paths."""
    import earthaccess

    t0 = month
    t1 = month + pd.offsets.MonthEnd(0) + pd.Timedelta(hours=23, minutes=59)
    results = earthaccess.search_data(
        short_name=cfg["gldas"]["short_name"],
        version=cfg["gldas"]["version"],
        temporal=(t0.isoformat(), t1.isoformat()),
    )
    files = earthaccess.download(results, str(cfg["paths"]["raw_dir"]))
    return [Path(f) for f in files]


def zonal_month(cfg: dict, month: pd.Timestamp, files: list[Path],
                w: sp.csr_matrix, land: np.ndarray) -> pd.DataFrame:
    """Reduce one month of granules to 3-hourly zonal means (long format)."""
    varmap = cfg["gldas"]["variables"]  # granule var -> column
    wind_comp = cfg["gldas"].get("wind_components")  # e.g. [Wind_E, Wind_N] -> wind10
    land_flat = land.ravel()
    n_regions = w.shape[0]

    columns = {col: [] for col in varmap.values()}
    if wind_comp:
        columns["wind10"] = []
    times = []
    for f in files:
        with xr.open_dataset(f) as ds:
            times.append(pd.Timestamp(ds["time"].values[0]))
            for var, col in varmap.items():
                cells = np.nan_to_num(ds[var].values[0].ravel()[land_flat])
                columns[col].append(w @ cells)
            if wind_comp:
                # NLDAS delivers wind as E/N components; reduce the SPEED, not the
                # vector mean (averaging components would cancel opposing winds).
                e = ds[wind_comp[0]].values[0].ravel()[land_flat]
                n = ds[wind_comp[1]].values[0].ravel()[land_flat]
                speed = np.nan_to_num(np.hypot(e, n))
                columns["wind10"].append(w @ speed)

    expected = cfg["gldas"]["granules_per_day"] * (month + pd.offsets.MonthEnd(0)).day
    if len(times) < expected:
        log.warning("%s: %d granules reduced, expected %d",
                    month.strftime("%Y-%m"), len(times), expected)

    order = np.argsort(np.asarray(times))
    out = pd.DataFrame({
        "time": np.repeat(pd.DatetimeIndex(times)[order].values, n_regions),
        "region_id": np.tile(np.arange(n_regions, dtype=np.int32), len(times)),
    })
    for col, rows in columns.items():
        out[col] = np.concatenate([rows[i] for i in order]).astype(np.float32)
    return out


def run(cfg: dict, w: sp.csr_matrix, land: np.ndarray) -> None:
    """Process every month in the configured period (acquire -> zonal -> parquet)."""
    mode = cfg["gldas"].get("download_mode", "earthaccess")
    if mode == "earthaccess":
        import earthaccess
        earthaccess.login(persist=True)

    zonal_dir = cfg["paths"]["zonal_dir"]
    for month in month_range(cfg["period"]["start"], cfg["period"]["end"]):
        out = monthly_file(zonal_dir, month)
        if out.exists():
            log.info("%s exists, skipping", out.name)
            continue
        files = local_month_files(cfg, month) if mode == "local" else download_month(cfg, month)
        if not files:
            log.error("%s: no granules found, skipping", month.strftime("%Y-%m"))
            continue
        table = zonal_month(cfg, month, files, w, land)
        tmp = out.with_suffix(".parquet.tmp")
        table.to_parquet(tmp, index=False)
        tmp.replace(out)
        log.info("wrote %s (%d timesteps)", out.name, table["time"].nunique())
        if cfg["gldas"]["delete_raw"]:
            for f in files:
                f.unlink(missing_ok=True)


def load_zonal(cfg: dict) -> pd.DataFrame:
    """Concatenate the monthly zonal parquets over the configured period."""
    files = sorted(cfg["paths"]["zonal_dir"].glob("zonal_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no zonal files in {cfg['paths']['zonal_dir']}; run stage 1 first")
    table = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    table["time"] = pd.to_datetime(table["time"])
    start, end = pd.Timestamp(cfg["period"]["start"]), pd.Timestamp(cfg["period"]["end"])
    table = table[(table["time"] >= start) & (table["time"] <= end + pd.Timedelta(hours=23))]
    return table.sort_values(["region_id", "time"]).reset_index(drop=True)
