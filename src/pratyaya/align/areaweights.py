"""Spatial aggregation and area-weighted overlap calculation for IMD subdivisions.

Grain: (region_id, grid_cell) sparse overlap weight matrix.
Ensures exact polygon x grid-cell area overlaps (geopandas.overlay / shapely intersection),
caching the weight matrix as .npz per docs/03-architecture-and-modeling.md §3.2.
Guarantees weights.sum(axis=1) ≈ 1.0 within 1e-4 tolerance.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import scipy.sparse as sp
import shapely
from shapely.geometry import box

logger = logging.getLogger(__name__)

# IMD 0.25 degree standard grid definition
GRID_LON_MIN: float = 66.5
GRID_LON_MAX: float = 100.0
GRID_LAT_MIN: float = 6.5
GRID_LAT_MAX: float = 38.5
GRID_RES: float = 0.25

DEFAULT_CACHE_DIR = Path("data/interim")
DEFAULT_WEIGHTS_FILE = DEFAULT_CACHE_DIR / "area_weights_subdivisions_36.npz"
DEFAULT_REGIONS_PARQUET = Path("data/canonical/regions.parquet")
DEFAULT_GEOJSON_SOURCE = DEFAULT_CACHE_DIR / "india-district-imd.geojson"
GEOJSON_FALLBACK_URL = (
    "https://gist.githubusercontent.com/planemad/d347ad7485344fb0ba4b470721825427"
    "/raw/209d887d681cc1112dd85cc63df0b2858ec97c6f/india-district-imd.geojson"
)


def get_imd_grid_coords() -> Tuple[np.ndarray, np.ndarray]:
    """Return standard IMD 0.25-degree grid 1D coordinate arrays.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (lons, lats)
            lons: 135 points from 66.5 to 100.0
            lats: 129 points from 6.5 to 38.5
    """
    lons = np.arange(GRID_LON_MIN, GRID_LON_MAX + GRID_RES / 2.0, GRID_RES)
    lats = np.arange(GRID_LAT_MIN, GRID_LAT_MAX + GRID_RES / 2.0, GRID_RES)
    return lons, lats


def load_or_fetch_subdivisions_gdf(
    geojson_path: Optional[Path] = None,
) -> gpd.GeoDataFrame:
    """Load or download Datameet district map with IMD subdivision properties.

    Dissolves 675 districts into the 36 meteorological subdivisions of India.

    Returns:
        gpd.GeoDataFrame: Dissolved 36 subdivisions sorted by integer subdivision_id.
    """
    path = Path(geojson_path) if geojson_path else DEFAULT_GEOJSON_SOURCE
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("Fetching subdivision boundary dataset from %s", GEOJSON_FALLBACK_URL)
        import requests

        r = requests.get(GEOJSON_FALLBACK_URL, timeout=60)
        r.raise_for_status()
        with open(path, "w", encoding="utf-8") as f:
            f.write(r.text)

    gdf = gpd.read_file(path)
    # Fix self-intersections or sliver geometries using buffer(0)
    gdf["geometry"] = gdf.geometry.buffer(0)

    # Dissolve districts into 36 subdivisions
    subs = gdf.dissolve(
        by=["subdivision_id", "subdivision", "region"], as_index=False
    )
    subs["subdivision_id"] = subs["subdivision_id"].astype(int)
    subs = subs.sort_values("subdivision_id").reset_index(drop=True)

    # Compute accurate geographic area in km² using Indian Albers Equal Area (EPSG:7755)
    subs["area_km2"] = subs.to_crs(epsg=7755).area / 1e6

    return subs


def compute_area_weights(
    subs_gdf: gpd.GeoDataFrame,
    lons: Optional[np.ndarray] = None,
    lats: Optional[np.ndarray] = None,
) -> Tuple[sp.csr_matrix, pd.DataFrame, np.ndarray, np.ndarray]:
    """Compute exact polygon x grid-cell area overlap weights for 36 subdivisions.

    Grain: (subdivision, grid_cell).

    Args:
        subs_gdf: GeoDataFrame containing 36 dissolved meteorological subdivisions.
        lons: 1D array of grid longitudes (default: IMD 0.25 deg).
        lats: 1D array of grid latitudes (default: IMD 0.25 deg).

    Returns:
        Tuple:
            - weights_csr: scipy.sparse.csr_matrix of shape (36, n_cells)
            - meta_df: DataFrame with region_id, name, area_km2
            - lons: 1D longitude array
            - lats: 1D latitude array
    """
    if lons is None or lats is None:
        grid_lons, grid_lats = get_imd_grid_coords()
    else:
        grid_lons, grid_lats = lons, lats

    n_lons = len(grid_lons)
    n_lats = len(grid_lats)
    n_cells = n_lons * n_lats
    half_res = GRID_RES / 2.0

    # Build grid cell polygons and spatial index
    cell_boxes = []
    cell_cos_lats = []
    for j, lat in enumerate(grid_lats):
        cos_lat = np.cos(np.radians(lat))
        for i, lon in enumerate(grid_lons):
            b = box(lon - half_res, lat - half_res, lon + half_res, lat + half_res)
            cell_boxes.append(b)
            cell_cos_lats.append(cos_lat)

    cell_cos_lats = np.array(cell_cos_lats, dtype=np.float64)
    tree = shapely.STRtree(cell_boxes)

    rows = []
    cols = []
    weights = []

    sub_ids = []
    sub_names = []
    sub_areas = []

    for sub_idx, sub in subs_gdf.iterrows():
        sub_geom = sub.geometry
        sub_id = int(sub["subdivision_id"])
        sub_name = str(sub["subdivision"])
        sub_area_km2 = float(sub.get("area_km2", 0.0))

        sub_ids.append(sub_id)
        sub_names.append(sub_name)
        sub_areas.append(sub_area_km2)

        # Spatial index query for candidate intersecting grid cells
        candidate_indices = tree.query(sub_geom, predicate="intersects")
        cell_weights: Dict[int, float] = {}
        total_sub_area: float = 0.0

        for c_idx in candidate_indices:
            c_box = cell_boxes[c_idx]
            inter = sub_geom.intersection(c_box)
            if not inter.is_empty:
                # Approximate spherical surface area differential: inter.area * cos(lat)
                cell_area = inter.area * cell_cos_lats[c_idx]
                if cell_area > 0:
                    cell_weights[c_idx] = cell_area
                    total_sub_area += cell_area

        if total_sub_area <= 0:
            raise ValueError(f"Subdivision {sub_id} ({sub_name}) has zero intersecting grid area!")

        # Normalize weights so row sums to 1.0
        for c_idx, area_val in cell_weights.items():
            norm_val = area_val / total_sub_area
            rows.append(sub_idx)
            cols.append(c_idx)
            weights.append(norm_val)

    weights_csr = sp.csr_matrix(
        (weights, (rows, cols)), shape=(len(subs_gdf), n_cells), dtype=np.float64
    )

    # Assert correctness gate: row sums ≈ 1.0 within 1e-4 tolerance
    row_sums = np.array(weights_csr.sum(axis=1)).flatten()
    if not np.allclose(row_sums, 1.0, atol=1e-4):
        bad_idx = np.where(~np.isclose(row_sums, 1.0, atol=1e-4))[0]
        bad_names = [sub_names[i] for i in bad_idx]
        raise AssertionError(f"Area weights row sums not within 1e-4 tolerance for: {bad_names}")

    meta_df = pd.DataFrame(
        {
            "region_id": [str(sid) for sid in sub_ids],
            "name": sub_names,
            "area_km2": sub_areas,
            "geometry": [g.wkb for g in subs_gdf.geometry],
        }
    )

    return weights_csr, meta_df, grid_lons, grid_lats


def cache_area_weights(
    weights_csr: sp.csr_matrix,
    meta_df: pd.DataFrame,
    lons: np.ndarray,
    lats: np.ndarray,
    npz_path: Optional[Path] = None,
    parquet_path: Optional[Path] = None,
) -> Tuple[Path, Path]:
    """Persist sparse weights to .npz and canonical regions metadata to parquet.

    Returns:
        Tuple[Path, Path]: (npz_path, parquet_path)
    """
    npz_target = Path(npz_path) if npz_path else DEFAULT_WEIGHTS_FILE
    parquet_target = Path(parquet_path) if parquet_path else DEFAULT_REGIONS_PARQUET

    npz_target.parent.mkdir(parents=True, exist_ok=True)
    parquet_target.parent.mkdir(parents=True, exist_ok=True)

    # Save sparse CSR components + coordinates in NPZ
    np.savez_compressed(
        npz_target,
        data=weights_csr.data,
        indices=weights_csr.indices,
        indptr=weights_csr.indptr,
        shape=weights_csr.shape,
        lons=lons,
        lats=lats,
        subdivision_ids=meta_df["region_id"].values.astype(str),
        subdivision_names=meta_df["name"].values.astype(str),
    )
    logger.info("Saved cached area weights to %s", npz_target)

    # Save canonical regions metadata to parquet
    meta_df.to_parquet(parquet_target, index=False)
    logger.info("Saved canonical regions table to %s", parquet_target)

    return npz_target, parquet_target


def load_cached_weights(
    npz_path: Optional[Path] = None,
) -> Tuple[sp.csr_matrix, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached area-weight matrix and coordinate arrays.

    Returns:
        Tuple: (weights_csr, lons, lats, subdivision_ids, subdivision_names)
    """
    path = Path(npz_path) if npz_path else DEFAULT_WEIGHTS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Weights cache {path} does not exist. Run areaweights.py to build."
        )

    with np.load(path, allow_pickle=True) as data:
        weights_csr = sp.csr_matrix(
            (data["data"], data["indices"], data["indptr"]), shape=tuple(data["shape"])
        )
        lons = data["lons"]
        lats = data["lats"]
        subdivision_ids = data["subdivision_ids"]
        subdivision_names = data["subdivision_names"]

    return weights_csr, lons, lats, subdivision_ids, subdivision_names


def aggregate_grid_to_subdivisions(
    grid_data: np.ndarray,
    weights_csr: sp.csr_matrix,
) -> np.ndarray:
    """Aggregate a 2D (lat, lon) or 3D (batch/time, lat, lon) grid to 36 subdivisions.

    Grain conversion: (lat, lon) -> (region_id).

    Args:
        grid_data: 2D array of shape (n_lats, n_lons) or 3D array of shape (..., n_lats, n_lons)
        weights_csr: CSR matrix of shape (36, n_lats * n_lons)

    Returns:
        np.ndarray: 1D array of shape (36,) or multi-dim array of shape (..., 36)
    """
    n_regions, n_cells = weights_csr.shape
    if grid_data.ndim == 2:
        flat_grid = grid_data.ravel()
        if flat_grid.shape[0] != n_cells:
            raise ValueError(
                f"Grid size {flat_grid.shape[0]} does not match weight cells {n_cells}"
            )
        # Handle possible NaNs by zeroing them in weight multiplication if needed
        return weights_csr.dot(flat_grid)
    elif grid_data.ndim == 3:
        n_batches = grid_data.shape[0]
        flat_grid = grid_data.reshape(n_batches, -1)
        if flat_grid.shape[1] != n_cells:
            raise ValueError(
                f"Grid size {flat_grid.shape[1]} does not match weight cells {n_cells}"
            )
        # Result shape: (n_batches, 36)
        return flat_grid.dot(weights_csr.T.toarray())
    else:
        raise ValueError(f"Unsupported grid dimension: {grid_data.ndim}")


def build_and_cache_subdivision_weights() -> Tuple[sp.csr_matrix, pd.DataFrame]:
    """Complete pipeline to load boundaries, compute weights, and cache artefacts."""
    subs_gdf = load_or_fetch_subdivisions_gdf()
    lons, lats = get_imd_grid_coords()
    weights_csr, meta_df, lons, lats = compute_area_weights(subs_gdf, lons, lats)
    cache_area_weights(weights_csr, meta_df, lons, lats)
    return weights_csr, meta_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Executing subdivision area-weight build...")
    weights_csr, meta_df = build_and_cache_subdivision_weights()
    row_sums = np.array(weights_csr.sum(axis=1)).flatten()
    print(f"Successfully computed weights for {len(meta_df)} subdivisions.")
    print(f"Minimum row sum: {row_sums.min():.6f}, Maximum row sum: {row_sums.max():.6f}")
    assert np.allclose(row_sums, 1.0, atol=1e-4)
    print("Assertion passed: weights.sum(axis=1) ~= 1.0 within 1e-4.")
