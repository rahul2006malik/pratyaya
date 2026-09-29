"""Causal Forecast and Observation Alignment Joins (Node F2, Node F5).

Grain: (init_date, region_id, lead_day, variable)

Enforces non-negotiable causality rules per docs/03-architecture-and-modeling.md §3.1:
- Every feature must be computed from information available at initialization time t0.
- Atmospheric state features join on init_date, NEVER on valid_date.
- Forecast verification alignment joins valid_date = init_date + lead_day (after applying IMD_DAY_OFFSET),
  but any observational feature input strictly rejects valid_date > init_date.
"""

from __future__ import annotations

import logging
from typing import Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_target_valid_date(
    init_date: pd.Timestamp,
    lead_day: int,
) -> pd.Timestamp:
    """Compute the target valid_date for a forecast issued at init_date with lead_day.

    Convention:
        GEFSv12 24h accumulation for lead_day L (where L in 1..10) spans:
        init_date + (L - 1) days to init_date + L days.
        The weather occurred during UTC calendar date: init_date + (L - 1) days.
        (e.g. init 2000-06-01 lead 1 -> weather on 2000-06-01).

    Args:
        init_date: Forecast initialization timestamp (00 UTC).
        lead_day: Forecast lead day horizon (1 to 10).

    Returns:
        pd.Timestamp: Target verification date in UTC coordinates.
    """
    return pd.to_datetime(init_date) + pd.Timedelta(days=lead_day - 1)


def join_forecasts_with_verification_obs(
    forecasts_df: pd.DataFrame,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join forecasts with truth observations for label evaluation.

    Grain: (init_date, region_id, lead_day, variable).

    Args:
        forecasts_df: Canonical forecasts schema dataframe.
        observations_df: Canonical observations schema dataframe (post-IMD_DAY_OFFSET valid_date).

    Returns:
        pd.DataFrame: Merged dataframe with forecast_value and observed_value.
    """
    f_df = forecasts_df.copy()
    o_df = observations_df.copy()

    f_df["init_date"] = pd.to_datetime(f_df["init_date"])
    f_df["region_id"] = f_df["region_id"].astype(str)
    f_df["lead_day"] = f_df["lead_day"].astype(int)

    # Compute target valid_date for each forecast row
    f_df["valid_date"] = f_df["init_date"] + pd.to_timedelta(f_df["lead_day"] - 1, unit="D")

    o_df["valid_date"] = pd.to_datetime(o_df["valid_date"])
    o_df["region_id"] = o_df["region_id"].astype(str)

    # Merge on (valid_date, region_id, variable)
    merged = pd.merge(
        f_df,
        o_df[["valid_date", "region_id", "variable", "observed_value"]],
        on=["valid_date", "region_id", "variable"],
        how="inner",
    )

    expected_cols = {
        "init_date",
        "region_id",
        "lead_day",
        "variable",
        "forecast_value",
        "observed_value",
    }
    if not expected_cols.issubset(set(merged.columns)):
        raise ValueError(f"Merged dataframe missing required columns: {expected_cols - set(merged.columns)}")

    return merged


def join_causal_prior_observations(
    forecasts_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    lookback_days: int = 1,
) -> pd.DataFrame:
    """Join historical observations strictly prior to or at initialization time t0.

    Non-Negotiable Rule:
        Observations with valid_date > init_date are future-dated and MUST be filtered out.
        If any observation row with valid_date > init_date is present in the join target,
        a ValueError is raised.

    Args:
        forecasts_df: Canonical forecasts table.
        observations_df: Observations table.
        lookback_days: Number of days prior to t0 to extract prior observation.

    Returns:
        pd.DataFrame: Forecasts dataframe augmented with causal prior observation.
    """
    f_df = forecasts_df.copy()
    o_df = observations_df.copy()

    f_df["init_date"] = pd.to_datetime(f_df["init_date"])
    f_df["region_id"] = f_df["region_id"].astype(str)
    o_df["valid_date"] = pd.to_datetime(o_df["valid_date"])
    o_df["region_id"] = o_df["region_id"].astype(str)

    # Required target observation date is strictly before t0
    f_df["prior_obs_date"] = f_df["init_date"] - pd.Timedelta(days=lookback_days)

    # Strict causality filter: drop any future observation
    # Any attempt to join future data raises or is masked
    max_allowed_date = f_df["init_date"].max()
    causal_obs = o_df[o_df["valid_date"] <= max_allowed_date].copy()

    merged = pd.merge(
        f_df,
        causal_obs[["valid_date", "region_id", "variable", "observed_value"]],
        left_on=["prior_obs_date", "region_id", "variable"],
        right_on=["valid_date", "region_id", "variable"],
        how="left",
    )

    # Verify no row joined on valid_date > init_date
    if (merged["valid_date"] > merged["init_date"]).any():
        raise RuntimeError("Causality violation: observations joined with valid_date > init_date!")

    merged = merged.rename(columns={"observed_value": f"obs_lag_{lookback_days}d"})
    merged = merged.drop(columns=["valid_date", "prior_obs_date"])

    return merged
