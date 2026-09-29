"""ERA5 Atmospheric State and Diagnostic Truth Ingest (Node F2).

Grain: (valid_date, region_id, variable)
Lazily queries WeatherBench2 ERA5 0.25° 6-hourly Zarr dataset from Google Cloud Storage,
subsets to the India domain, aggregates 6-hourly steps to daily diagnostics,
computes subdivision area-weighted averages using F3 cached weights, and outputs
canonical observations parquet tables per docs/04-data-and-api.md §2.3.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gcsfs
import numpy as np
import pandas as pd
import xarray as xr

from pratyaya.align.areaweights import (
    aggregate_grid_to_subdivisions,
    get_imd_grid_coords,
    load_cached_weights,
)

logger = logging.getLogger(__name__)

WB2_GCS_ZARR_PATH = "weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
SOURCE_NAME = "era5"
DEFAULT_CANONICAL_DIR = Path("data/canonical/observations")


def open_wb2_era5_dataset() -> xr.Dataset:
    """Open WeatherBench2 ERA5 Zarr store on GCS anonymously and lazily."""
    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(WB2_GCS_ZARR_PATH)
    ds = xr.open_zarr(mapper, consolidated=True)
    return ds


def fetch_era5_india_daily(
    start_date: Union[date, datetime, str],
    end_date: Union[date, datetime, str],
    variable: str = "2m_temperature",
    stat: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract daily gridded values for India domain from WeatherBench2 ERA5 Zarr.

    Grain: (time, lat, lon).
    Reindexes latitudes to ascending (6.5 to 38.5).

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        variable: ERA5 variable name in WB2 (e.g. '2m_temperature').
        stat: Daily aggregation method ('mean' or 'max').

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - daily_grid: 3D array of shape (n_days, 129, 135)
            - dates: 1D array of daily timestamps
    """
    # Guard against touching held-out period (2018-2019)
    s_dt = pd.to_datetime(start_date)
    e_dt = pd.to_datetime(end_date)
    if any(y in (2018, 2019) for y in (s_dt.year, e_dt.year)):
        raise PermissionError(
            f"Requested date range {start_date} to {end_date} touches held-out years (2018-2019)."
        )

    ds = open_wb2_era5_dataset()

    # Slice time range + India bounding box (lat descending 38.5 to 6.5, lon 66.5 to 100.0)
    var_da = ds[variable].sel(
        time=slice(str(start_date), str(end_date)),
        latitude=slice(38.5, 6.5),
        longitude=slice(66.5, 100.0),
    )

    # Compute daily statistic
    if stat == "mean":
        daily_da = var_da.resample(time="1D").mean()
    elif stat == "max":
        daily_da = var_da.resample(time="1D").max()
    else:
        raise ValueError(f"Unsupported daily statistic: {stat}")

    # Load values into memory (small domain: ~129x135xN_days floats)
    daily_vals = daily_da.values
    daily_dates = pd.to_datetime(daily_da.time.values)

    # Reverse latitude axis to ascending (6.5 to 38.5)
    daily_grid_asc = daily_vals[:, ::-1, :]

    return daily_grid_asc, daily_dates


def ingest_era5_observations(
    start_date: Union[date, datetime, str],
    end_date: Union[date, datetime, str],
    variable: str = "2m_temperature",
    canonical_var_name: str = "t2m",
    stat: str = "mean",
    weights_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Ingest ERA5 slice and aggregate to 36 subdivisions.

    Grain: (valid_date, region_id, variable).

    Returns:
        pd.DataFrame: Canonical observations schema table.
    """
    daily_grid, daily_dates = fetch_era5_india_daily(start_date, end_date, variable, stat)
    weights_csr, lons, lats, sub_ids, sub_names = load_cached_weights(weights_path)

    # Aggregate (n_days, 129, 135) to (n_days, 36)
    sub_vals = aggregate_grid_to_subdivisions(daily_grid, weights_csr)

    records = []
    for d_idx, dt in enumerate(daily_dates):
        raw_dt = pd.Timestamp(dt)
        for r_idx, reg_id in enumerate(sub_ids):
            val = float(sub_vals[d_idx, r_idx])
            records.append(
                {
                    "valid_date": raw_dt,  # Raw date before IMD_DAY_OFFSET (D-001)
                    "region_id": str(reg_id),
                    "variable": canonical_var_name,
                    "observed_value": val,
                    "source": SOURCE_NAME,
                    "raw_date": raw_dt,
                }
            )

    df = pd.DataFrame(records)
    return df


def save_canonical_observations(
    df: pd.DataFrame,
    canonical_dir: Optional[Path] = None,
) -> Path:
    """Save canonical observations partitioned by year."""
    target_dir = Path(canonical_dir) if canonical_dir else DEFAULT_CANONICAL_DIR
    year = df["valid_date"].iloc[0].year
    year_dir = target_dir / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)

    var_name = df["variable"].iloc[0]
    out_file = year_dir / f"obs_era5_{var_name}_{year}.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Saved canonical observations to %s (%d rows)", out_file, len(df))
    return out_file


def smoke_verify_era5(
    start_date: str = "2000-01-01",
    end_date: str = "2000-01-07",
) -> Tuple[pd.DataFrame, Path]:
    """Execute smoke ingest for 1 week sample of year 2000 and verify canonical schema."""
    df = ingest_era5_observations(
        start_date, end_date, variable="2m_temperature", canonical_var_name="t2m"
    )
    out_file = save_canonical_observations(df)

    n_days = len(pd.date_range(start_date, end_date))
    expected_rows = n_days * 36
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    required_cols = {
        "valid_date",
        "region_id",
        "variable",
        "observed_value",
        "source",
        "raw_date",
    }
    assert required_cols.issubset(set(df.columns)), f"Missing cols: {required_cols - set(df.columns)}"

    return df, out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running ERA5 smoke ingest verification for 2000-01-01 to 2000-01-07...")
    df, out_path = smoke_verify_era5("2000-01-01", "2000-01-07")
    print(f"Verified {len(df)} rows saved to {out_path}!")
    print(df.head())
