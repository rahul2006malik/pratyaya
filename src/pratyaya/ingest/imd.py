"""IMD Gridded Daily Rainfall Ingestion Module (Node F2).

Grain: (valid_date, region_id, variable)
Parses IMD 0.25° gridded daily rainfall data (via imdlib or cached binary .grd files),
handles no-data masks (-999.0), aggregates to the 36 meteorological subdivisions of India
using cached exact area weights (Node F3), applies empirical rain-day alignment
IMD_DAY_OFFSET = -1 (Node F2, Decision D-001), and persists canonical observations tables
per docs/04-data-and-api.md §2.3.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import imdlib
import numpy as np
import pandas as pd
import xarray as xr

from pratyaya.align.areaweights import (
    aggregate_grid_to_subdivisions,
    get_imd_grid_coords,
    load_cached_weights,
)
from pratyaya.align.rainday import IMD_DAY_OFFSET, apply_imd_day_offset

logger = logging.getLogger(__name__)

DEFAULT_RAW_IMD_DIR = Path("data/raw/imd")
DEFAULT_CANONICAL_DIR = Path("data/canonical/observations")
SOURCE_NAME = "imd_gridded"


def load_imd_rainfall_grid(
    year: int,
    raw_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, pd.DatetimeIndex]:
    """Load IMD 0.25° gridded daily rainfall for a single year.

    Checks for cached {year}.grd file under raw_dir/rain/ first.
    If absent, attempts retrieval via imdlib.get_data.

    Args:
        year: Calendar year (e.g. 2000).
        raw_dir: Directory containing 'rain' subdirectory with .grd files.

    Returns:
        Tuple[np.ndarray, pd.DatetimeIndex]:
            - daily_grid: 3D array of shape (n_days, 129, 135) with values in mm (clean, >= 0.0)
            - dates: 1D array of daily reporting timestamps (0830 IST convention)

    Raises:
        PermissionError: If year is in held-out period (2018-2019).
        FileNotFoundError: If binary data cannot be found or downloaded.
    """
    if year in (2018, 2019):
        raise PermissionError(
            f"Year {year} is in held-out validation set (2018-2019). "
            "Ingestion blocked outside --final per D-000."
        )

    base_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_IMD_DIR
    rain_file = base_dir / "rain" / f"{year}.grd"

    if rain_file.exists():
        logger.info("Opening cached IMD rain file at %s", rain_file)
        data = imdlib.open_data("rain", year, year, fn_format="yearwise", file_dir=str(base_dir))
    else:
        logger.info("IMD rain file %s not found locally; attempting imdlib.get_data...", rain_file)
        try:
            data = imdlib.get_data("rain", year, year, fn_format="yearwise", file_dir=str(base_dir))
        except Exception as e:
            raise FileNotFoundError(
                f"Failed to load or download IMD rain file for year {year}: {e}"
            ) from e

    ds = data.get_xarray()
    raw_rain = ds["rain"].values  # shape: (n_days, 129, 135)
    dates = pd.to_datetime(ds["time"].values)

    # Clean missing values (-999.0 outside India domain -> 0.0)
    clean_rain = np.where((raw_rain < 0) | np.isnan(raw_rain), 0.0, raw_rain)

    return clean_rain, dates


def ingest_imd_observations(
    year: int,
    raw_dir: Optional[Path] = None,
    weights_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Ingest IMD daily rainfall for a year, aggregate to subdivisions, and apply D-001 offset.

    Grain: (valid_date, region_id, variable).

    Returns:
        pd.DataFrame: Canonical observations table rows.
    """
    clean_rain, raw_dates = load_imd_rainfall_grid(year, raw_dir=raw_dir)
    weights_csr, lons, lats, sub_ids, sub_names = load_cached_weights(weights_path)

    # Area-aggregate (n_days, 129, 135) to (n_days, 36)
    sub_vals = aggregate_grid_to_subdivisions(clean_rain, weights_csr)

    records = []
    for d_idx, r_dt in enumerate(raw_dates):
        raw_ts = pd.Timestamp(r_dt)
        for r_idx, reg_id in enumerate(sub_ids):
            val = float(sub_vals[d_idx, r_idx])
            records.append({
                "raw_date": raw_ts,
                "region_id": str(reg_id),
                "variable": "rainfall",
                "observed_value": val,
                "source": SOURCE_NAME,
            })

    raw_df = pd.DataFrame(records)

    # Apply empirical IMD_DAY_OFFSET = -1 per D-001
    aligned_df = apply_imd_day_offset(raw_df)

    canonical_cols = [
        "valid_date",
        "region_id",
        "variable",
        "observed_value",
        "source",
        "raw_date",
    ]
    return aligned_df[canonical_cols]


def save_canonical_imd_observations(
    df: pd.DataFrame,
    canonical_dir: Optional[Path] = None,
) -> List[Path]:
    """Save canonical IMD observations partitioned by year."""
    target_dir = Path(canonical_dir) if canonical_dir else DEFAULT_CANONICAL_DIR
    out_files = []
    for year, group in df.groupby(df["valid_date"].dt.year):
        year_dir = target_dir / f"year={year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        out_file = year_dir / f"obs_imd_rainfall_{year}.parquet"
        if out_file.exists():
            existing_group = pd.read_parquet(out_file)
            combined = pd.concat([existing_group, group], ignore_index=True)
            combined = combined.drop_duplicates(subset=["valid_date", "region_id", "variable"])
            combined = combined.sort_values(["valid_date", "region_id"]).reset_index(drop=True)
            combined.to_parquet(out_file, index=False)
            logger.info("Updated canonical IMD observations at %s (%d rows)", out_file, len(combined))
        else:
            group.to_parquet(out_file, index=False)
            logger.info("Saved canonical IMD observations to %s (%d rows)", out_file, len(group))
        out_files.append(out_file)
    return out_files
