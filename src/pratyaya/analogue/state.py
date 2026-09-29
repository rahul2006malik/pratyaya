"""Wide-Domain ERA5 Synoptic State Representation (Node G1, docs/03 §6.1).

Defines the synoptic state vector on a 1.5° grid covering 30°E–120°E, 10°S–45°N.
Extracts 5 core synoptic fields from WeatherBench2 / ERA5:
- Z500: Geopotential at 500 hPa
- U850: U-component of wind at 850 hPa
- V850: V-component of wind at 850 hPa
- MSLP: Mean sea level pressure
- TCWV: Total column water vapour

Computes standardized anomalies strictly against training-fold climatology.
All methods enforce zero access to held-out years (2018–2019) per Decision D-000.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Bounding box covering 30°E–120°E, 10°S–45°N per docs/03 §6.1
WIDE_DOMAIN_LON_MIN: float = 30.0
WIDE_DOMAIN_LON_MAX: float = 120.0
WIDE_DOMAIN_LAT_MIN: float = -10.0  # 10°S in decimal degrees
WIDE_DOMAIN_LAT_MAX: float = 45.0   # 45°N in decimal degrees
GRID_RESOLUTION: float = 1.5

# Canonical 5 synoptic fields
DEFAULT_SYNOPTIC_FIELDS: Tuple[str, ...] = ("z500", "u850", "v850", "mslp", "tcwv")

# WeatherBench2 ERA5 mapping: (variable_name, vertical_level_hPa)
WB2_FIELD_SPECS: Dict[str, Tuple[str, Optional[int]]] = {
    "z500": ("geopotential", 500),
    "u850": ("u_component_of_wind", 850),
    "v850": ("v_component_of_wind", 850),
    "mslp": ("mean_sea_level_pressure", None),
    "tcwv": ("total_column_water_vapour", None),
    "t850": ("temperature", 850),  # Optional 6th field per docs/03 §6.1
}

WB2_GCS_ZARR_PATH: str = "weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr"
HELDOUT_YEARS: Tuple[int, int] = (2018, 2019)


def check_heldout_date_protection(dates: Union[Sequence[Union[date, datetime, pd.Timestamp, str]], pd.DatetimeIndex]) -> None:
    """Enforce Decision D-000 and Risk R-10: never access or process 2018-2019 held-out data."""
    dt_index = pd.to_datetime(list(dates))
    heldout_present = dt_index[dt_index.year.isin(HELDOUT_YEARS)]
    if len(heldout_present) > 0:
        sample_dates = [d.strftime("%Y-%m-%d") for d in heldout_present[:3]]
        raise PermissionError(
            f"Access to held-out years {HELDOUT_YEARS} is strictly prohibited outside guarded "
            f"--final runs (found {len(heldout_present)} forbidden dates, e.g. {sample_dates}). "
            f"Enforced per Decision D-000 and Risk R-10."
        )


def get_wide_domain_coords(
    ascending_lat: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate the 1.5° grid coordinates for the Indian synoptic wide domain.

    Domain spans:
      Longitude: 30.0°E to 120.0°E (step 1.5° -> 61 points)
      Latitude:  -10.5° to 45.0°N (step 1.5° -> 38 points, covering 10°S to 45°N and 0° Equator)

    Returns:
        Tuple[np.ndarray, np.ndarray]: (lats, lons) 1D coordinate arrays.
    """
    lons = np.arange(WIDE_DOMAIN_LON_MIN, WIDE_DOMAIN_LON_MAX + 1e-5, GRID_RESOLUTION)
    # -10.5 covers 10°S (-10.0°) with exact 1.5° steps aligned with 0.0° (Equator) and 45.0°N
    lats = np.arange(-10.5, WIDE_DOMAIN_LAT_MAX + 1e-5, GRID_RESOLUTION)
    if not ascending_lat:
        lats = lats[::-1]
    return lats, lons


@dataclass
class WideDomainGrid:
    """Metadata describing the wide domain 1.5° grid dimensions and bounding box."""
    lon_min: float = WIDE_DOMAIN_LON_MIN
    lon_max: float = WIDE_DOMAIN_LON_MAX
    lat_min: float = WIDE_DOMAIN_LAT_MIN
    lat_max: float = WIDE_DOMAIN_LAT_MAX
    resolution: float = GRID_RESOLUTION
    n_lons: int = 61
    n_lats: int = 38
    n_grid_points: int = 61 * 38  # 2318
    fields: Tuple[str, ...] = DEFAULT_SYNOPTIC_FIELDS

    @property
    def total_features(self) -> int:
        """Total flattened vector length across all synoptic fields."""
        return self.n_grid_points * len(self.fields)


class WideDomainClimatology:
    """Gridpoint climatology and standardization computer fitted strictly on training data.

    Per docs/03 §6.1: Each field is computed as an anomaly against a training-years
    per-gridpoint climatology, then standardized per gridpoint.
    """

    def __init__(self, epsilon: float = 1e-6) -> None:
        self.epsilon = epsilon
        self.mean_: Optional[np.ndarray] = None   # shape: (n_fields, n_lats, n_lons) or (n_features,)
        self.std_: Optional[np.ndarray] = None    # shape: (n_fields, n_lats, n_lons) or (n_features,)
        self.fitted_training_dates_: Optional[pd.DatetimeIndex] = None
        self.is_fitted: bool = False

    def fit(
        self,
        states: np.ndarray,
        dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
    ) -> WideDomainClimatology:
        """Fit climatological mean and standard deviation strictly on training dates.

        Args:
            states: Array of shape (n_days, n_fields, n_lats, n_lons) or (n_days, n_features).
            dates: Sequence of observation timestamps.

        Raises:
            PermissionError: If any date is in held-out years (2018, 2019).
            ValueError: If empty states are provided.
        """
        check_heldout_date_protection(dates)
        dt_index = pd.to_datetime(list(dates))

        if len(dt_index) == 0 or len(states) == 0:
            raise ValueError("Cannot fit climatology on empty state records.")
        if len(dt_index) != len(states):
            raise ValueError(f"Length mismatch: {len(dt_index)} dates vs {len(states)} state samples.")

        # Compute mean and standard deviation along the temporal axis (axis 0)
        self.mean_ = np.nanmean(states, axis=0)
        self.std_ = np.nanstd(states, axis=0)

        # Guard against zero or NaN variance
        self.std_ = np.where(np.isnan(self.std_) | (self.std_ < self.epsilon), 1.0, self.std_)
        self.mean_ = np.where(np.isnan(self.mean_), 0.0, self.mean_)

        self.fitted_training_dates_ = dt_index
        self.is_fitted = True
        logger.info(
            "Fitted WideDomainClimatology on %d training days (years: %s)",
            len(dt_index),
            sorted(dt_index.year.unique().tolist()),
        )
        return self

    def transform(self, states: np.ndarray) -> np.ndarray:
        """Transform raw states into standardized anomalies: (x - mu) / (sigma + eps).

        Args:
            states: Array of raw states matching the fitted spatial/feature dimensions.

        Returns:
            np.ndarray: Standardized anomaly array with same shape as input.
        """
        if not self.is_fitted or self.mean_ is None or self.std_ is None:
            raise RuntimeError("WideDomainClimatology must be fitted on training fold before transform.")

        anomalies = (states - self.mean_) / (self.std_ + self.epsilon)
        return anomalies

    def fit_transform(
        self,
        states: np.ndarray,
        dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
    ) -> np.ndarray:
        """Fit on training fold and return standardized anomalies."""
        return self.fit(states, dates).transform(states)


def build_synoptic_state_vector(
    field_grids: Dict[str, np.ndarray],
    fields_order: Sequence[str] = DEFAULT_SYNOPTIC_FIELDS,
) -> np.ndarray:
    """Stack and flatten multiple 2D/3D synoptic grids into a 1D or 2D state vector.

    Args:
        field_grids: Dict mapping field name to array of shape (n_days, n_lats, n_lons)
                     or (n_lats, n_lons).
        fields_order: Order in which fields are concatenated.

    Returns:
        np.ndarray: Flattened array of shape (n_days, total_features) or (total_features,).
    """
    for f in fields_order:
        if f not in field_grids:
            raise KeyError(f"Required field '{f}' missing from field_grids (present: {list(field_grids.keys())})")

    first_arr = field_grids[fields_order[0]]
    is_batch = (first_arr.ndim == 3)

    flattened_fields = []
    for f in fields_order:
        arr = field_grids[f]
        if is_batch:
            n_samples = arr.shape[0]
            flat = arr.reshape(n_samples, -1)
        else:
            flat = arr.reshape(-1)
        flattened_fields.append(flat)

    if is_batch:
        return np.concatenate(flattened_fields, axis=1)
    else:
        return np.concatenate(flattened_fields, axis=0)


def extract_era5_wide_domain_synoptic(
    start_date: Union[date, datetime, str],
    end_date: Union[date, datetime, str],
    fields: Sequence[str] = DEFAULT_SYNOPTIC_FIELDS,
    zarr_path: str = WB2_GCS_ZARR_PATH,
) -> Tuple[Dict[str, np.ndarray], pd.DatetimeIndex, Tuple[np.ndarray, np.ndarray]]:
    """Extract and regrid synoptic fields from WeatherBench2 ERA5 Zarr store.

    Extracts:
    - Z500: geopotential at level 500
    - U850: u_component_of_wind at level 850
    - V850: v_component_of_wind at level 850
    - MSLP: mean_sea_level_pressure
    - TCWV: total_column_water_vapour

    Subsets to 30°E–120°E, 10°S–45°N and regrids to 1.5° resolution.

    Args:
        start_date: Start date string (YYYY-MM-DD).
        end_date: End date string (YYYY-MM-DD).
        fields: Subset of fields to extract.
        zarr_path: GCS path to WeatherBench2 ERA5 Zarr store.

    Returns:
        Tuple[Dict[str, np.ndarray], pd.DatetimeIndex, Tuple[np.ndarray, np.ndarray]]:
            - Dict of {field_name: array of shape (n_days, n_lats, n_lons)}
            - DatetimeIndex of extracted days
            - (target_lats, target_lons)
    """
    check_heldout_date_protection([start_date, end_date])
    date_range = pd.date_range(start_date, end_date, freq="D")
    check_heldout_date_protection(date_range)

    import gcsfs
    import xarray as xr

    target_lats, target_lons = get_wide_domain_coords(ascending_lat=False)

    fs = gcsfs.GCSFileSystem(token="anon")
    mapper = fs.get_mapper(zarr_path)
    ds = xr.open_zarr(mapper, consolidated=True)

    extracted_fields: Dict[str, np.ndarray] = {}
    time_index: Optional[pd.DatetimeIndex] = None

    for field_name in fields:
        if field_name not in WB2_FIELD_SPECS:
            raise ValueError(f"Unknown synoptic field '{field_name}'. Supported: {list(WB2_FIELD_SPECS.keys())}")

        wb2_var, level = WB2_FIELD_SPECS[field_name]
        var_da = ds[wb2_var]

        # Select vertical pressure level if 3D field
        if level is not None and "level" in var_da.dims:
            var_da = var_da.sel(level=level)

        # Slice wide-domain bounding box (WB2 latitude is descending: 90 to -90)
        # 45.0°N to -10.5° (covering 10°S), 30.0°E to 120.0°E
        sub_da = var_da.sel(
            time=slice(str(start_date), str(end_date)),
            latitude=slice(WIDE_DOMAIN_LAT_MAX, -10.5),
            longitude=slice(WIDE_DOMAIN_LON_MIN, WIDE_DOMAIN_LON_MAX),
        )

        # Daily mean aggregation from 6-hourly steps
        daily_da = sub_da.resample(time="1D").mean()

        # Interpolate / regrid to exact 1.5° wide domain coordinates
        regrid_da = daily_da.interp(
            latitude=target_lats,
            longitude=target_lons,
            method="linear",
        )

        field_vals = regrid_da.values
        if time_index is None:
            time_index = pd.to_datetime(regrid_da.time.values)

        extracted_fields[field_name] = field_vals

    return extracted_fields, time_index if time_index is not None else date_range, (target_lats, target_lons)
