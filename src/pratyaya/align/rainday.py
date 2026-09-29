"""Empirical Rain-Day / UTC Alignment Module (Node F2, Decision D-001).

Grain: (valid_date, region_id, variable)

Defines the single empirical constant IMD_DAY_OFFSET governing alignment between
IMD 0830 IST gauge accumulation windows and model/reanalysis UTC daily cycles.

Physical Rationale:
-------------------
IMD daily rainfall is reported at 08:30 IST on calendar day T (raw_date), accumulating
rainfall from 08:30 IST on day T-1 to 08:30 IST on day T.
In UTC coordinates:
- 08:30 IST (T-1) = 03:00 UTC (T-1)
- 08:30 IST (T)   = 03:00 UTC (T)
Thus, 21 out of 24 hours (87.5%) of the IMD accumulation window belong to UTC day T-1,
while only 3 hours (12.5%) belong to UTC day T.

Empirical Evidence:
-------------------
Cross-correlation between IMD daily subdivision series and ERA5 daily UTC total precipitation
across training years (2000, 2005) during the southwest monsoon (JJAS) confirms peak correlation
at Lag -1 on high-signal subdivisions (e.g. Coastal Karnataka r=0.8323/0.8737, Kerala r=0.8722).
Aligning at Lag 0 (same calendar date) lags the true weather by nearly 24 hours, manufacturing
spurious forecast busts out of clock conventions.

Governed by docs/DECISIONS.md D-001 and docs/03-architecture-and-modeling.md §3.3.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Empirical integer day lag offset: valid_date = raw_date + Timedelta(days=IMD_DAY_OFFSET)
# D-001: IMD raw_date T aligns with UTC valid_date T - 1.
IMD_DAY_OFFSET: int = -1

# High-signal subdivisions for monsoon cross-correlation verification
DEFAULT_TEST_SUBDIVISIONS: List[Tuple[str, str]] = [
    ("7", "Konkan & Goa"),        # Datameet ID 7 / IMD Met ID 9
    ("13", "Kerala"),             # Datameet ID 13 / IMD Met ID 35
    ("33", "Coastal Karnataka"),  # Datameet ID 33 / IMD Met ID 31
]


def align_imd_observation_date(
    raw_date: Union[date, datetime, pd.Timestamp, str],
    offset_days: int = IMD_DAY_OFFSET,
) -> pd.Timestamp:
    """Align an IMD gauge date (0830 IST reporting date) to the canonical UTC valid_date.

    Args:
        raw_date: The original IMD reporting date (e.g. '2000-06-02').
        offset_days: Integer day offset (defaults to IMD_DAY_OFFSET = -1).

    Returns:
        pd.Timestamp: Normalized valid_date (e.g. '2000-06-01').
    """
    ts = pd.to_datetime(raw_date)
    return ts + pd.Timedelta(days=offset_days)


def apply_imd_day_offset(
    df: pd.DataFrame,
    date_col: str = "raw_date",
    target_col: str = "valid_date",
    offset_days: int = IMD_DAY_OFFSET,
) -> pd.DataFrame:
    """Apply IMD_DAY_OFFSET to a DataFrame, populating or overwriting target_col.

    Grain: preserves existing row grain.
    Both raw_date and valid_date are kept per docs/04-data-and-api.md §2.3
    so that date alignment is always fully auditable.

    Args:
        df: Input DataFrame containing date_col.
        date_col: Column with pre-offset dates (default: 'raw_date').
        target_col: Column to receive post-offset valid dates (default: 'valid_date').
        offset_days: Integer day shift (default: IMD_DAY_OFFSET = -1).

    Returns:
        pd.DataFrame: DataFrame with updated target_col.
    """
    out = df.copy()
    out[target_col] = pd.to_datetime(out[date_col]) + pd.Timedelta(days=offset_days)
    return out


def compute_cross_correlation_series(
    imd_series: pd.Series,
    era5_series: pd.Series,
    lags: Sequence[int] = (-2, -1, 0, 1, 2),
) -> Dict[int, float]:
    """Compute Pearson cross-correlation between IMD and ERA5 series across specified day lags.

    Convention:
        lag = k means comparing IMD(t) with ERA5(t + k).
        For lag = -1, IMD(t) is compared with ERA5(t - 1).

    Args:
        imd_series: Daily IMD rainfall Series indexed by Timestamp.
        era5_series: Daily ERA5 UTC precipitation Series indexed by Timestamp.
        lags: Sequence of integer day lags.

    Returns:
        Dict[int, float]: Mapping from lag to Pearson correlation coefficient.
    """
    results: Dict[int, float] = {}
    for lag in lags:
        # Shifting ERA5 by -lag aligns index t with t + lag
        shifted_era5 = era5_series.shift(-lag).reindex(imd_series.index)
        valid = imd_series.notna() & shifted_era5.notna()
        if valid.sum() > 10:
            corr = float(imd_series[valid].corr(shifted_era5[valid]))
        else:
            corr = float("nan")
        results[lag] = corr
    return results


def summarize_correlation_table(
    correlation_dict: Dict[str, Dict[int, float]],
) -> pd.DataFrame:
    """Format correlation dictionary into a clean summary DataFrame.

    Returns:
        pd.DataFrame: Indexed by subdivision, columns correspond to lags.
    """
    df = pd.DataFrame.from_dict(correlation_dict, orient="index")
    df.columns = [f"Lag {c:+d}" for c in df.columns]
    return df
