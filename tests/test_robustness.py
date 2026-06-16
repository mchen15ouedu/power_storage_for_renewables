"""Robustness / regression tests for the GLDAS->NLDAS generalization.

These lock down the forcing-agnostic changes (multi-dataset config, hourly
steps, derived wind speed, the nearest-cell distance cap) AND guarantee the
original GLDAS behaviour is unchanged. They use tiny synthetic granules and
grids, so they run in seconds with no downloaded data.

Run:  pytest -q tests/test_robustness.py
  or: python tests/test_robustness.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

xr = pytest.importorskip("xarray")
gpd = pytest.importorskip("geopandas")
sp = pytest.importorskip("scipy.sparse")
from shapely.geometry import Point, box  # noqa: E402

from gldas_storage import analyze, process, regions  # noqa: E402
from gldas_storage.config import load_config  # noqa: E402
from gldas_storage import energy  # noqa: E402

GLDAS_CFG = ROOT / "hpc" / "config_oscer.yaml"
NLDAS_CFG = ROOT / "hpc" / "config_nldas.yaml"


# --------------------------------------------------------------------------
# config: forcing-block aliasing and per-dataset invariants
# --------------------------------------------------------------------------
def test_gldas_config_invariants():
    cfg = load_config(GLDAS_CFG)
    assert cfg["gldas"]["short_name"] == "GLDAS_NOAH025_3H"
    assert cfg["gldas"].get("file_ext", "nc4") == "nc4"          # default preserved
    assert analyze.steps_per_year(cfg) == 2920                   # 3-hourly default
    assert cfg["gldas"].get("wind_components") is None           # GLDAS = scalar wind


def test_nldas_config_invariants():
    cfg = load_config(NLDAS_CFG)
    # the "forcing:" block must be aliased to cfg["gldas"]
    assert cfg["gldas"]["short_name"] == "NLDAS_FORA0125_H"
    assert cfg["gldas"]["file_ext"] == "nc"
    assert analyze.steps_per_year(cfg) == 8760                   # hourly
    assert cfg["gldas"]["wind_components"] == ["Wind_E", "Wind_N"]
    assert "wind10" not in cfg["gldas"]["variables"].values()    # wind10 is derived
    assert cfg["regions"]["include_countries"] == ["USA"]
    assert cfg["regions"]["nearest_cell_max_deg"] == 0.5


def test_steps_per_year_defaults_without_forcing_block():
    assert analyze.steps_per_year({}) == analyze.STEPS_PER_YEAR_3H
    assert analyze.steps_per_year({"gldas": {}}) == 2920


# --------------------------------------------------------------------------
# process: filename pattern + zonal reduction with/without derived wind
# --------------------------------------------------------------------------
def _make_granule(path: Path, varvals: dict, ny=2, nx=2):
    lat = np.linspace(25.0, 26.0, ny)
    lon = np.linspace(-100.0, -99.0, nx)
    data = {v: (("time", "lat", "lon"), np.full((1, ny, nx), val, float))
            for v, val in varvals.items()}
    ds = xr.Dataset(data, coords={"lat": lat, "lon": lon,
                                  "time": [np.datetime64("2000-01-01T00:00")]})
    ds.to_netcdf(path)
    return lat, lon


def _unit_weight(n_cells):
    """1 region covering all cells with equal (row-stochastic) weight."""
    w = np.full((1, n_cells), 1.0 / n_cells)
    return sp.csr_matrix(w)


def test_local_month_files_extension(tmp_path):
    cfg = {"paths": {"raw_dir": tmp_path},
           "gldas": {"short_name": "NLDAS_FORA0125_H", "file_ext": "nc"}}
    good = tmp_path / "NLDAS_FORA0125_H.A20000101.0000.020.nc"
    good.write_text("x")
    (tmp_path / "NLDAS_FORA0125_H.A20000101.0000.020.nc4").write_text("x")  # wrong ext
    found = process.local_month_files(cfg, pd.Timestamp("2000-01-01"))
    assert found == [good]


def test_zonal_month_derives_wind_speed(tmp_path):
    """NLDAS path: wind10 must be sqrt(E^2+N^2) (3,4 -> 5), not the component mean."""
    f = tmp_path / "g.nc"
    _make_granule(f, {"SWdown": 200.0, "Tair": 290.0, "PSurf": 9e4,
                      "Qair": 0.01, "Wind_E": 3.0, "Wind_N": 4.0})
    land = np.ones((2, 2), bool)
    cfg = {"gldas": {"variables": {"SWdown": "swdown", "Tair": "tair",
                                   "PSurf": "psurf", "Qair": "qair"},
                     "wind_components": ["Wind_E", "Wind_N"],
                     "granules_per_day": 24}}
    out = process.zonal_month(cfg, pd.Timestamp("2000-01-01"), [f],
                              _unit_weight(4), land)
    assert set(["swdown", "tair", "psurf", "qair", "wind10"]).issubset(out.columns)
    assert out["wind10"].iloc[0] == pytest.approx(5.0)
    assert out["swdown"].iloc[0] == pytest.approx(200.0)


def test_zonal_month_gldas_scalar_wind_unchanged(tmp_path):
    """GLDAS path (no wind_components): scalar Wind_f_inst maps straight through."""
    f = tmp_path / "g.nc4"
    _make_granule(f, {"SWdown_f_tavg": 150.0, "Wind_f_inst": 7.0, "Tair_f_inst": 280.0,
                      "Psurf_f_inst": 9e4, "Qair_f_inst": 0.008})
    land = np.ones((2, 2), bool)
    cfg = {"gldas": {"variables": {"SWdown_f_tavg": "swdown", "Wind_f_inst": "wind10",
                                   "Tair_f_inst": "tair", "Psurf_f_inst": "psurf",
                                   "Qair_f_inst": "qair"},
                     "granules_per_day": 8}}
    out = process.zonal_month(cfg, pd.Timestamp("2000-01-01"), [f],
                              _unit_weight(4), land)
    assert out["wind10"].iloc[0] == pytest.approx(7.0)
    assert "wind_components" not in cfg["gldas"]            # GLDAS cfg untouched


# --------------------------------------------------------------------------
# regions: nearest-cell distance cap excludes off-grid regions (AK/HI)
# --------------------------------------------------------------------------
def _grid_and_regions():
    lat = np.arange(25.0, 30.01, 0.25)
    lon = np.arange(-100.0, -94.99, 0.25)
    land = np.ones((lat.size, lon.size), bool)
    near = box(-97.06, 27.06, -96.94, 27.19)     # tiny box, no cell centre, <0.5 deg away
    far = box(-0.5, -0.5, 0.5, 0.5)              # off the grid by >>0.5 deg (mock AK/HI)
    gdf = gpd.GeoDataFrame(
        {"region_id": [0, 1], "name": ["near", "far"]},
        geometry=[near, far], crs="EPSG:4326")
    return lat, lon, land, gdf


def test_nearest_cell_cap_excludes_far_region():
    lat, lon, land, gdf = _grid_and_regions()
    cfg = {"regions": {"nearest_cell_fallback": True, "nearest_cell_max_deg": 0.5}}
    w = regions.build_weights(cfg, gdf, lat, lon, land)
    assert w.shape[0] == 2
    assert gdf.loc[gdf["name"] == "far", "weight_sum"].iloc[0] == 0.0     # left empty
    assert gdf.loc[gdf["name"] == "near", "weight_sum"].iloc[0] > 0.0     # snapped


def test_nearest_cell_no_cap_snaps_everything():
    """Backward-compat: without the cap (GLDAS default) even the far region snaps."""
    lat, lon, land, gdf = _grid_and_regions()
    cfg = {"regions": {"nearest_cell_fallback": True}}      # no nearest_cell_max_deg
    regions.build_weights(cfg, gdf, lat, lon, land)
    assert (gdf["weight_sum"] > 0).all()


def test_grid_from_granule_uses_mask_var_and_ext(tmp_path):
    f = tmp_path / "NLDAS_FORA0125_H.A20000101.0000.020.nc"
    lat = np.linspace(25.0, 26.0, 3)
    lon = np.linspace(-100.0, -99.0, 4)
    tair = np.full((1, 3, 4), 290.0)
    tair[0, 0, 0] = np.nan                       # one ocean cell
    xr.Dataset({"Tair": (("time", "lat", "lon"), tair)},
               coords={"lat": lat, "lon": lon,
                       "time": [np.datetime64("2000-01-01")]}).to_netcdf(f)
    cfg = {"paths": {"raw_dir": tmp_path},
           "gldas": {"short_name": "NLDAS_FORA0125_H", "file_ext": "nc",
                     "land_mask_var": "Tair", "variables": {"Tair": "tair"}}}
    glat, glon, land = regions.grid_from_granule(cfg)
    assert glat.size == 3 and glon.size == 4
    assert land.sum() == 11 and not land[0, 0]   # the NaN cell is masked out


# --------------------------------------------------------------------------
# energy: derived wind speed consistency + capacity-factor clipping
# --------------------------------------------------------------------------
def test_capacity_factor_clips_to_unit_interval():
    x = np.array([-5.0, 0.0, 3.0, 6.0, 20.0])
    cf = energy.capacity_factor(x, start=3.0, plateau=12.0)
    assert cf.min() == 0.0 and cf.max() == 1.0
    assert cf[2] == pytest.approx(0.0)           # at start threshold
    assert np.all(np.diff(cf) >= 0)              # monotonic


def test_hypot_matches_manual_speed():
    e = np.array([3.0, 0.0, -6.0])
    n = np.array([4.0, 5.0, 8.0])
    assert np.allclose(np.hypot(e, n), np.sqrt(e ** 2 + n ** 2))


# --------------------------------------------------------------------------
# figures: the supply/storage PDF pipeline still renders after label edits
# --------------------------------------------------------------------------
def test_deficit_pdf_renders(tmp_path):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from gldas_storage import deficit, figures

    dates = pd.date_range("2000-01-01", periods=365 * 3, freq="D")
    rng = np.random.default_rng(0)
    cf = np.clip(0.4 + 0.3 * np.cos(2 * np.pi * dates.dayofyear / 365)
                 + rng.normal(0, 0.05, len(dates)), 0.01, 1.0)
    res = deficit.analyze_region(cf, dates, np.ones(len(dates)),
                                 {"up_efficiency": 0.9, "down_efficiency": 0.8,
                                  "annual_decay": 0.7, "max_iterations": 20,
                                  "tolerance": 1e-3}, 365, simulate=True)
    out = tmp_path / "deficit_solar_TEST.pdf"
    figures.deficit_pdf(out, [("Testland", res)], "solar", "TEST_group")
    supply = out.with_name(out.stem + "_supply.pdf")
    storage = out.with_name(out.stem + "_storage.pdf")
    assert supply.exists() and supply.stat().st_size > 0
    assert storage.exists() and storage.stat().st_size > 0


def main():
    raise SystemExit(pytest.main([__file__, "-q"]))


if __name__ == "__main__":
    main()
