"""Unit tests for area-weighted subdivision aggregation (Node F3).

Enforces:
- Exact 36 meteorological subdivisions of India.
- Grid matches IMD 0.25° coordinates (66.5°E to 100.0°E, 6.5°N to 38.5°N).
- Exact area weights row sums ≈ 1.0 within 1e-4 tolerance.
- Aggregation preserves constant fields exactly.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from pratyaya.align.areaweights import (
    DEFAULT_REGIONS_PARQUET,
    DEFAULT_WEIGHTS_FILE,
    aggregate_grid_to_subdivisions,
    get_imd_grid_coords,
    load_cached_weights,
)


@pytest.fixture(scope="module")
def cached_weights():
    """Load cached area weights; if not present, build them."""
    if not DEFAULT_WEIGHTS_FILE.exists():
        from pratyaya.align.areaweights import build_and_cache_subdivision_weights
        build_and_cache_subdivision_weights()

    weights_csr, lons, lats, sub_ids, sub_names = load_cached_weights()
    return weights_csr, lons, lats, sub_ids, sub_names


def test_subdivision_count_and_grid_dims(cached_weights):
    weights_csr, lons, lats, sub_ids, sub_names = cached_weights

    # Must contain exactly 36 meteorological subdivisions
    assert weights_csr.shape[0] == 36
    assert len(sub_ids) == 36
    assert len(sub_names) == 36

    # IMD 0.25° grid centers check
    assert np.isclose(lons.min(), 66.5)
    assert np.isclose(lons.max(), 100.0)
    assert len(lons) == 135

    assert np.isclose(lats.min(), 6.5)
    assert np.isclose(lats.max(), 38.5)
    assert len(lats) == 129

    expected_cells = 135 * 129
    assert weights_csr.shape[1] == expected_cells


def test_area_weights_row_sums_approx_one(cached_weights):
    """Core correctness gate: assert weights.sum(axis=1) ≈ 1.0 within 1e-4 tolerance."""
    weights_csr, _, _, sub_ids, sub_names = cached_weights
    row_sums = np.array(weights_csr.sum(axis=1)).flatten()

    # Strict check across all 36 subdivisions
    assert np.allclose(row_sums, 1.0, atol=1e-4), (
        f"Row sums deviate from 1.0. Min: {row_sums.min()}, Max: {row_sums.max()}"
    )

    # Check non-negativity of weights
    assert weights_csr.data.min() >= 0.0, "Weights must be non-negative"


def test_uniform_field_aggregation(cached_weights):
    """Aggregation of a constant field must equal that constant value for every subdivision."""
    weights_csr, lons, lats, _, _ = cached_weights
    constant_val = 42.0

    # 2D grid
    grid_2d = np.full((len(lats), len(lons)), constant_val, dtype=np.float64)
    res_2d = aggregate_grid_to_subdivisions(grid_2d, weights_csr)
    assert res_2d.shape == (36,)
    assert np.allclose(res_2d, constant_val, atol=1e-5)

    # 3D grid (e.g. 5 time steps or ensemble members)
    grid_3d = np.full((5, len(lats), len(lons)), constant_val, dtype=np.float64)
    res_3d = aggregate_grid_to_subdivisions(grid_3d, weights_csr)
    assert res_3d.shape == (5, 36)
    assert np.allclose(res_3d, constant_val, atol=1e-5)


def test_canonical_regions_schema():
    """Verify canonical regions table exists and follows docs/04-data-and-api.md §2.1."""
    assert DEFAULT_REGIONS_PARQUET.exists(), f"Regions parquet missing at {DEFAULT_REGIONS_PARQUET}"
    df = pd.read_parquet(DEFAULT_REGIONS_PARQUET)

    assert len(df) == 36
    required_cols = {"region_id", "name", "area_km2", "geometry"}
    assert required_cols.issubset(set(df.columns))

    # All IDs should be unique strings
    assert df["region_id"].nunique() == 36
    assert df["name"].nunique() == 36
    assert (df["area_km2"] > 0).all()
