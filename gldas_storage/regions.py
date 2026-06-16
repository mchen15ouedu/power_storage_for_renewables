"""Stage 2a: build the analysis regions and their grid-cell weight matrix.

Regions come from Natural Earth 10m cultural vectors:
  * admin-0 countries (one region per country) by default,
  * admin-1 states/provinces for countries larger than ``regions.admin1_area_km2``
    (or explicitly listed in ``regions.admin1_force``) -- "the bigger countries,
    we still use state/province lines".

The mapping from regions to GLDAS grid cells is stored as a sparse row-stochastic
matrix W (n_regions x n_land_cells): zonal_mean = W @ field[land_cells], with
cos(latitude) area weighting. Regions too small to contain any 0.25-degree cell
center are assigned their nearest land cell.
"""

from __future__ import annotations

import io
import logging
import urllib.request
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import xarray as xr
from scipy.spatial import cKDTree
from shapely.geometry import Point

log = logging.getLogger(__name__)

NE_BASE = "https://naciscdn.org/naturalearth/10m/cultural"
NE_ADMIN0 = "ne_10m_admin_0_countries"
NE_ADMIN1 = "ne_10m_admin_1_states_provinces"
EQUAL_AREA = "EPSG:6933"


def _fetch_natural_earth(name: str, cache_dir: Path) -> gpd.GeoDataFrame:
    """Download (once) and read a Natural Earth shapefile; lower-case the columns."""
    shp_dir = cache_dir / name
    if not shp_dir.exists():
        url = f"{NE_BASE}/{name}.zip"
        log.info("downloading %s", url)
        req = urllib.request.Request(url, headers={"User-Agent": "gldas-storage/0.1"})
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
        shp_dir.mkdir(parents=True)
        zipfile.ZipFile(io.BytesIO(data)).extractall(shp_dir)
    gdf = gpd.read_file(next(shp_dir.glob("*.shp")))
    gdf.columns = [c.lower() for c in gdf.columns]
    return gdf


def build_regions(cfg: dict) -> gpd.GeoDataFrame:
    """Assemble the region GeoDataFrame (admin-0 + admin-1 for large countries)."""
    rcfg = cfg["regions"]
    # cache inside the repo so bundled shapefiles travel with it (no internet needed on HPC)
    cache = cfg["root"] / "data" / "naturalearth"

    a0 = _fetch_natural_earth(NE_ADMIN0, cache)[["admin", "adm0_a3", "continent", "geometry"]]
    a0 = a0[~a0["adm0_a3"].isin(rcfg["exclude"])]
    if rcfg["include_countries"]:
        a0 = a0[a0["adm0_a3"].isin(rcfg["include_countries"])]
    continent_of = dict(zip(a0["adm0_a3"], a0["continent"]))

    area_km2 = a0.to_crs(EQUAL_AREA).geometry.area / 1e6
    split = set(a0.loc[area_km2 >= float(rcfg["admin1_area_km2"]), "adm0_a3"])
    split |= set(rcfg["admin1_force"])
    log.info("countries split into admin-1: %s", ", ".join(sorted(split)) or "none")

    parts = [
        a0[~a0["adm0_a3"].isin(split)]
        .rename(columns={"admin": "name"})
        .assign(country=lambda d: d["name"], level="admin0")
    ]
    if split:
        a1 = _fetch_natural_earth(NE_ADMIN1, cache)
        a1 = a1[a1["adm0_a3"].isin(split)][["name", "name_en", "adm1_code", "admin", "adm0_a3", "geometry"]]
        a1["name"] = a1["name"].fillna(a1["name_en"]).fillna(a1["adm1_code"])
        parts.append(
            a1[["name", "admin", "adm0_a3", "geometry"]]
            .rename(columns={"admin": "country"})
            .assign(level="admin1",
                    continent=lambda d: d["adm0_a3"].map(continent_of))
        )

    gdf = pd.concat(parts, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, crs=a0.crs).reset_index(drop=True)
    gdf["region_id"] = gdf.index.astype(int)
    log.info("%d regions (%d admin-0, %d admin-1)", len(gdf),
             (gdf["level"] == "admin0").sum(), (gdf["level"] == "admin1").sum())
    return gdf[["region_id", "name", "country", "adm0_a3", "continent", "level", "geometry"]]


def grid_from_granule(cfg: dict, sample: Path | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lat, lon, land) from a sample 3-hourly granule; land marks valid cells."""
    ext = cfg["gldas"].get("file_ext", "nc4")
    if sample is None:
        pattern = f"{cfg['gldas']['short_name']}.A*.{ext}"
        candidates = sorted(cfg["paths"]["raw_dir"].rglob(pattern))
        if not candidates:
            raise FileNotFoundError(
                f"no sample granule under {cfg['paths']['raw_dir']}; download one first "
                "(any single granule works, it only provides the grid and land mask)")
        sample = candidates[0]
    mask_var = cfg["gldas"].get("land_mask_var", next(iter(cfg["gldas"]["variables"])))
    with xr.open_dataset(sample) as ds:
        lat = ds["lat"].values
        lon = ds["lon"].values
        land = ds[mask_var].isel(time=0).notnull().values
    log.info("grid %dx%d from %s", lat.size, lon.size, Path(sample).name)
    return lat, lon, land


def build_weights(cfg: dict, regions: gpd.GeoDataFrame,
                  lat: np.ndarray, lon: np.ndarray, land: np.ndarray) -> sp.csr_matrix:
    """Sparse (n_regions x n_land_cells) area-weight matrix via point-in-polygon."""
    lon2d, lat2d = np.meshgrid(lon, lat)
    iy, ix = np.nonzero(land)
    cell_lat, cell_lon = lat2d[iy, ix], lon2d[iy, ix]
    n_cells = cell_lat.size
    log.info("%d land cells on the GLDAS grid", n_cells)

    cells = gpd.GeoDataFrame(
        {"cell": np.arange(n_cells)},
        geometry=[Point(x, y) for x, y in zip(cell_lon, cell_lat)],
        crs=regions.crs,
    )
    joined = gpd.sjoin(cells, regions[["region_id", "geometry"]], predicate="intersects")
    rows = joined["region_id"].to_numpy()
    cols = joined["cell"].to_numpy()

    # nearest-land-cell fallback for regions that caught no cell (small islands etc.)
    matched = set(rows.tolist())
    missing = regions.loc[~regions["region_id"].isin(matched)]
    if len(missing) and cfg["regions"]["nearest_cell_fallback"]:
        tree = cKDTree(np.column_stack([cell_lon, cell_lat]))
        pts = missing.geometry.representative_point()
        dist, nearest = tree.query(np.column_stack([pts.x, pts.y]))
        # Cap the snap distance so regions that lie OUTSIDE the grid coverage
        # (e.g. Alaska/Hawaii for a CONUS-only NLDAS grid) are not snapped to a
        # far-away edge cell; they stay empty -> all-NaN -> excluded downstream.
        max_deg = cfg["regions"].get("nearest_cell_max_deg")
        near = (dist <= float(max_deg)) if max_deg is not None else np.ones(len(dist), bool)
        snapped = missing[near]
        if (~near).any():
            log.warning("%d region(s) beyond %s deg of any cell, left empty: %s",
                        int((~near).sum()), max_deg,
                        ", ".join(missing[~near]["name"].head(20)))
        log.info("%d region(s) assigned their nearest land cell: %s",
                 len(snapped), ", ".join(snapped["name"].head(20)))
        rows = np.concatenate([rows, snapped["region_id"].to_numpy()])
        cols = np.concatenate([cols, nearest[near]])

    weights = np.cos(np.deg2rad(cell_lat))[cols]
    w = sp.csr_matrix((weights, (rows, cols)), shape=(len(regions), n_cells))
    row_sum = np.asarray(w.sum(axis=1)).ravel()
    # pre-normalization weight sums are proportional to land area -> used for pooling
    regions["weight_sum"] = row_sum
    empty = row_sum == 0
    if empty.any():
        log.warning("%d region(s) have no grid cells and will be all-NaN: %s",
                    empty.sum(), ", ".join(regions.loc[empty, "name"]))
        row_sum = row_sum.copy()
        row_sum[empty] = 1.0
    w = sp.diags(1.0 / row_sum) @ w
    return w.tocsr()


def save(cfg: dict, regions: gpd.GeoDataFrame, w: sp.csr_matrix, land: np.ndarray) -> None:
    regions.to_file(cfg["paths"]["regions_gpkg"], driver="GPKG")
    regions.drop(columns="geometry").to_csv(cfg["paths"]["regions_csv"], index=False)
    sp.save_npz(cfg["paths"]["weights_npz"].with_suffix(".weights.npz"), w)
    np.savez_compressed(cfg["paths"]["weights_npz"], land=land)
    log.info("saved %s, %s", cfg["paths"]["regions_gpkg"].name, cfg["paths"]["weights_npz"].name)


def load(cfg: dict) -> tuple[gpd.GeoDataFrame, sp.csr_matrix, np.ndarray]:
    regions = gpd.read_file(cfg["paths"]["regions_gpkg"])
    w = sp.load_npz(cfg["paths"]["weights_npz"].with_suffix(".weights.npz"))
    land = np.load(cfg["paths"]["weights_npz"])["land"]
    return regions, w, land
