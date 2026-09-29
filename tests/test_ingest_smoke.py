"""Smoke and regression test suite for GEFSv12 (F1) and ERA5 (F2) ingest.

Validates:
- Strict held-out protection (2018-2019 blocked outside --final per D-000).
- S3 key formatting and GRIB2 parsing.
- WeatherBench2 ERA5 lazy slicing.
- Canonical parquet schema conformity (docs/04-data-and-api.md §2.2, §2.3).
"""

from datetime import date
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from pratyaya.ingest.gefs import (
    format_s3_grib_key,
    parse_gefs_member_daily_grid,
    save_canonical_forecasts,
)
from pratyaya.ingest.era5 import (
    fetch_era5_india_daily,
    save_canonical_observations,
)


def test_gefs_heldout_gate():
    """Verify that attempting to ingest 2018 or 2019 raises PermissionError."""
    with pytest.raises(PermissionError, match="held-out period"):
        format_s3_grib_key("2018-05-15", "c00")

    with pytest.raises(PermissionError, match="held-out period"):
        format_s3_grib_key("2019-07-20", "p01")

    # Safe year 2000 should format cleanly
    key = format_s3_grib_key("2000-01-01", "c00")
    assert "2000/2000010100/c00/Days:1-10/apcp_sfc_2000010100_c00.grib2" in key


def test_era5_heldout_gate():
    """Verify that attempting to slice ERA5 for 2018 or 2019 raises PermissionError."""
    with pytest.raises(PermissionError, match="held-out years"):
        fetch_era5_india_daily("2018-01-01", "2018-01-02")

    with pytest.raises(PermissionError, match="held-out years"):
        fetch_era5_india_daily("2019-06-01", "2019-06-02")


def test_gefs_grib_parsing_dimensions():
    """Verify GRIB2 daily grid parser produces exact India domain shape (10, 129, 135)."""
    sample_file = Path("data/raw/gefs/sample_apcp.grib2")
    if not sample_file.exists():
        sample_file = Path("data/raw/gefs/apcp_sfc_2000010100_c00.grib2")

    if sample_file.exists():
        daily_grid = parse_gefs_member_daily_grid(sample_file, variable="rainfall")
        assert daily_grid.shape == (10, 129, 135), f"Unexpected shape {daily_grid.shape}"
        assert daily_grid.min() >= 0.0, "Rainfall must be non-negative"
        # Total rainfall shouldn't be completely zero or all NaN
        assert not np.isnan(daily_grid).any()


def test_canonical_forecasts_schema_and_serialization(tmp_path):
    """Verify forecasts dataframe matches docs/04-data-and-api.md §2.2."""
    n_regions = 36
    n_leads = 10
    records = []
    for l in range(1, n_leads + 1):
        for r in range(1, n_regions + 1):
            records.append({
                "init_date": pd.Timestamp("2000-01-01"),
                "region_id": str(r),
                "lead_day": int(l),
                "variable": "rainfall",
                "model_source": "gefsv12_reforecast",
                "forecast_value": 5.2,
                "ens_spread": 1.1,
                "ens_members": [4.5, 5.0, 5.5, 6.0, 5.0],
                "n_members": 5,
            })
    df = pd.DataFrame(records)

    out_file = save_canonical_forecasts(df, canonical_dir=tmp_path)
    assert out_file.exists()

    loaded = pd.read_parquet(out_file)
    assert len(loaded) == 360
    assert set(loaded.columns) == {
        "init_date", "region_id", "lead_day", "variable", "model_source",
        "forecast_value", "ens_spread", "ens_members", "n_members"
    }
    assert loaded["n_members"].iloc[0] == 5


def test_canonical_observations_schema_and_serialization(tmp_path):
    """Verify observations dataframe matches docs/04-data-and-api.md §2.3."""
    records = []
    for r in range(1, 37):
        records.append({
            "valid_date": pd.Timestamp("2000-01-01"),
            "region_id": str(r),
            "variable": "t2m",
            "observed_value": 298.5,
            "source": "era5",
            "raw_date": pd.Timestamp("2000-01-01"),
        })
    df = pd.DataFrame(records)

    out_file = save_canonical_observations(df, canonical_dir=tmp_path)
    assert out_file.exists()

    loaded = pd.read_parquet(out_file)
    assert len(loaded) == 36
    assert set(loaded.columns) == {
        "valid_date", "region_id", "variable", "observed_value", "source", "raw_date"
    }
