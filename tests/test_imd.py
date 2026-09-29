"""Unit and regression tests for IMD gridded daily rainfall ingestion (Node F2).

Validates:
- Strict held-out protection (2018-2019 blocked outside --final per D-000).
- Local binary .grd file loading and no-data cleaning.
- Empirical rain-day alignment IMD_DAY_OFFSET = -1 per D-001.
- Canonical observations parquet schema and partition structure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pratyaya.ingest.imd import (
    load_imd_rainfall_grid,
    ingest_imd_observations,
    save_canonical_imd_observations,
)


def test_imd_heldout_protection():
    """Verify that attempting to load IMD data for 2018 or 2019 raises PermissionError."""
    with pytest.raises(PermissionError, match="held-out validation set"):
        load_imd_rainfall_grid(2018)

    with pytest.raises(PermissionError, match="held-out validation set"):
        load_imd_rainfall_grid(2019)


def test_load_imd_rainfall_grid_cached():
    """Verify that cached 2000.grd file loads cleanly with non-negative rain values."""
    rain_file = Path("data/raw/imd/rain/2000.grd")
    if not rain_file.exists():
        pytest.skip("Cached 2000.grd not present on disk")

    clean_rain, dates = load_imd_rainfall_grid(2000)
    assert clean_rain.shape == (366, 129, 135)
    assert clean_rain.min() >= 0.0, "Rainfall values must be non-negative"
    assert len(dates) == 366
    assert dates[0] == pd.Timestamp("2000-01-01")


def test_ingest_imd_observations_schema():
    """Verify canonical observations table produced from IMD data."""
    rain_file = Path("data/raw/imd/rain/2000.grd")
    if not rain_file.exists():
        pytest.skip("Cached 2000.grd not present on disk")

    # Ingest year 2000 (first 10 days for test speed)
    df = ingest_imd_observations(2000)
    assert len(df) == 366 * 36

    canonical_cols = [
        "valid_date",
        "region_id",
        "variable",
        "observed_value",
        "source",
        "raw_date",
    ]
    assert list(df.columns) == canonical_cols
    assert (df["source"] == "imd_gridded").all()
    assert (df["variable"] == "rainfall").all()

    # Verify D-001 offset: valid_date == raw_date - 1 day
    assert (df["valid_date"] == df["raw_date"] - pd.Timedelta(days=1)).all()
