"""NOAA GEFSv12 Reforecast Ingestion Module (Node F1).

Grain: (init_date, region_id, lead_day, variable)
Fetches NOAA GEFSv12 retrospective forecasts from AWS S3 (anonymous),
parses surface diagnostics using cfgrib/xarray, subsets to the India domain,
aggregates to 36 IMD subdivisions using cached area weights, and converts
to canonical parquet schema per docs/04-data-and-api.md §2.2.
"""

from __future__ import annotations

import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import s3fs
import xarray as xr

from pratyaya.align.areaweights import (
    aggregate_grid_to_subdivisions,
    get_imd_grid_coords,
    load_cached_weights,
)

logger = logging.getLogger(__name__)

S3_BUCKET = "noaa-gefs-retrospective"
MODEL_SOURCE = "gefsv12_reforecast"
DEFAULT_RAW_DIR = Path("data/raw/gefs")
DEFAULT_CANONICAL_DIR = Path("data/canonical/forecasts")

# Ensemble members: c00 (control), p01..p04 (perturbed)
STANDARD_MEMBERS = ["c00", "p01", "p02", "p03", "p04"]


def format_s3_grib_key(
    init_date: Union[date, datetime, str],
    member: str,
    variable_file: str = "apcp_sfc",
    lead_range: str = "Days:1-10",
) -> str:
    """Format canonical S3 key for GEFSv12 reforecast GRIB2 file.

    Pattern:
    noaa-gefs-retrospective/GEFSv12/reforecast/{YYYY}/{YYYYMMDD00}/{member}/{lead_range}/{var}_{YYYYMMDD00}_{member}.grib2
    """
    if isinstance(init_date, str):
        init_date = datetime.strptime(init_date[:10], "%Y-%m-%d").date()
    elif isinstance(init_date, datetime):
        init_date = init_date.date()

    # Held-out protection gate: never ingest 2018-2019 outside --final
    if init_date.year in (2018, 2019):
        raise PermissionError(
            f"Init date {init_date} is in held-out period (2018-2019). "
            "Ingestion blocked outside --final per D-000."
        )

    date_str = init_date.strftime("%Y%m%d00")
    year_str = init_date.strftime("%Y")

    return (
        f"{S3_BUCKET}/GEFSv12/reforecast/{year_str}/{date_str}/{member}/"
        f"{lead_range}/{variable_file}_{date_str}_{member}.grib2"
    )


def fetch_gefs_grib(
    init_date: Union[date, datetime, str],
    member: str,
    variable_file: str = "apcp_sfc",
    output_dir: Optional[Path] = None,
) -> Path:
    """Download single GEFSv12 GRIB2 file from S3 anonymous bucket.

    Returns:
        Path: Local path to downloaded GRIB2 file.
    """
    s3_key = format_s3_grib_key(init_date, member, variable_file)
    target_dir = Path(output_dir) if output_dir else DEFAULT_RAW_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = Path(s3_key).name
    local_path = target_dir / filename

    if not local_path.exists():
        logger.info("Downloading s3://%s -> %s", s3_key, local_path)
        fs = s3fs.S3FileSystem(anon=True)
        fs.get(s3_key, str(local_path))
    else:
        logger.debug("Using cached GRIB2 file at %s", local_path)

    return local_path


def parse_gefs_member_daily_grid(
    grib_path: Path,
    member: str = "c00",
    variable: str = "rainfall",
) -> np.ndarray:
    """Extract 10 lead days of gridded data over the India 0.25° domain.

    Reindexes latitudes to ascending (6.5 to 38.5) to match IMD grid conventions.

    Returns:
        np.ndarray: shape (10, 129, 135) for lead_days 1..10.
    """
    data_type = "cf" if member == "c00" else "pf"
    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"dataType": data_type}},
    )

    if variable == "rainfall":
        # Sum the four 6h accumulation intervals for each 24h lead day (6h, 12h, 18h, 24h)
        daily_leads = np.zeros((10, 129, 135), dtype=np.float64)
        sub_ds = ds["tp"].sel(
            latitude=slice(38.5, 6.5),
            longitude=slice(66.5, 100.0),
        )
        for lead_idx in range(10):
            lead_day = lead_idx + 1
            steps = [np.timedelta64((lead_day - 1) * 24 + h, "h") for h in (6, 12, 18, 24)]
            day_accum = sub_ds.sel(step=steps).sum(dim="step").values
            # Reverse latitude axis to ascending (6.5 to 38.5) to match IMD grid
            daily_leads[lead_idx] = day_accum[::-1, :]
        return daily_leads
    else:
        raise NotImplementedError(f"GEFS variable '{variable}' parser not implemented.")


def ingest_gefs_cycle(
    init_date: Union[date, datetime, str],
    variable: str = "rainfall",
    members: Optional[List[str]] = None,
    raw_dir: Optional[Path] = None,
    weights_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Ingest complete ensemble for one initialization date and aggregate to subdivisions.

    Grain: (init_date, region_id, lead_day, variable).

    Returns:
        pd.DataFrame: Canonical forecasts table rows.
    """
    if members is None:
        members = STANDARD_MEMBERS

    if isinstance(init_date, str):
        init_dt = datetime.strptime(init_date[:10], "%Y-%m-%d")
    elif isinstance(init_date, date) and not isinstance(init_date, datetime):
        init_dt = datetime(init_date.year, init_date.month, init_date.day)
    else:
        init_dt = init_date

    weights_csr, lons, lats, sub_ids, sub_names = load_cached_weights(weights_path)
    n_regions = len(sub_ids)
    n_leads = 10
    n_members = len(members)

    # Array to hold member values: shape (n_members, n_leads, 36)
    member_sub_vals = np.zeros((n_members, n_leads, n_regions), dtype=np.float64)

    for m_idx, member in enumerate(members):
        grib_file = fetch_gefs_grib(init_dt, member, variable_file="apcp_sfc", output_dir=raw_dir)
        daily_grid = parse_gefs_member_daily_grid(grib_file, member=member, variable=variable)
        # Aggregate each lead day grid (129, 135) to 36 subdivisions
        for l_idx in range(n_leads):
            member_sub_vals[m_idx, l_idx, :] = aggregate_grid_to_subdivisions(
                daily_grid[l_idx], weights_csr
            )

    records = []
    for l_idx in range(n_leads):
        lead_day = l_idx + 1
        for r_idx, reg_id in enumerate(sub_ids):
            mem_vals = member_sub_vals[:, l_idx, r_idx].tolist()
            f_mean = float(np.mean(mem_vals))
            f_spread = float(np.std(mem_vals, ddof=1)) if n_members > 1 else 0.0

            records.append(
                {
                    "init_date": pd.Timestamp(init_dt),
                    "region_id": str(reg_id),
                    "lead_day": int(lead_day),
                    "variable": variable,
                    "model_source": MODEL_SOURCE,
                    "forecast_value": f_mean,
                    "ens_spread": f_spread,
                    "ens_members": mem_vals,
                    "n_members": int(n_members),
                }
            )

    df = pd.DataFrame(records)
    return df


def save_canonical_forecasts(
    df: pd.DataFrame,
    canonical_dir: Optional[Path] = None,
) -> Path:
    """Save canonical forecasts partitioned by year."""
    target_dir = Path(canonical_dir) if canonical_dir else DEFAULT_CANONICAL_DIR
    year = df["init_date"].iloc[0].year
    year_dir = target_dir / f"year={year}"
    year_dir.mkdir(parents=True, exist_ok=True)

    date_str = df["init_date"].iloc[0].strftime("%Y%m%d")
    out_file = year_dir / f"forecasts_{date_str}.parquet"
    df.to_parquet(out_file, index=False)
    logger.info("Saved canonical forecasts to %s (%d rows)", out_file, len(df))
    return out_file


def smoke_verify_gefs(sample_date: str = "2000-01-01") -> Tuple[pd.DataFrame, Path]:
    """Execute smoke ingest for 1 cycle (2000-01-01) and verify canonical schema."""
    df = ingest_gefs_cycle(sample_date, variable="rainfall", members=STANDARD_MEMBERS[:2])
    out_file = save_canonical_forecasts(df)

    # Validate output schema and row count
    expected_rows = 10 * 36  # 10 leads * 36 subdivisions
    assert len(df) == expected_rows, f"Expected {expected_rows} rows, got {len(df)}"

    required_cols = {
        "init_date",
        "region_id",
        "lead_day",
        "variable",
        "model_source",
        "forecast_value",
        "ens_spread",
        "ens_members",
        "n_members",
    }
    assert required_cols.issubset(set(df.columns)), f"Missing cols: {required_cols - set(df.columns)}"

    return df, out_file


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("Running GEFSv12 smoke ingest verification for 2000-01-01...")
    df, out_path = smoke_verify_gefs("2000-01-01")
    print(f"Verified {len(df)} rows saved to {out_path}!")
    print(df.head())
