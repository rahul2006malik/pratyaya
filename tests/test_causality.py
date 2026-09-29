"""Mandatory Causality Enforcement Test Suite (Node F5).

Governed by docs/03-architecture-and-modeling.md §3.1 and docs/09-agent-briefing.md §21.
Rule:
    Every feature is computed strictly from information available at initialization time t0.
    Atmospheric-state features join on init_date, NEVER on valid_date.
    This test poisons all future-dated data (valid_date > init_date) with NaN/garbage
    and asserts that feature tables generated at t0 are strictly bit-identical.
    This test runs in CI and is NEVER skipped or xfailed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pratyaya.align.joins import (
    compute_target_valid_date,
    join_causal_prior_observations,
    join_forecasts_with_verification_obs,
)


def create_synthetic_forecast_and_obs_fixtures(
    init_date: str = "2000-06-01",
    n_leads: int = 10,
    n_regions: int = 36,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate deterministic forecasts and observations across past and future dates."""
    t0 = pd.Timestamp(init_date)
    regions = [str(r) for r in range(1, n_regions + 1)]

    # Forecasts at t0 for leads 1..10
    f_rows = []
    for l in range(1, n_leads + 1):
        for r in regions:
            f_rows.append({
                "init_date": t0,
                "region_id": r,
                "lead_day": l,
                "variable": "rainfall",
                "model_source": "gefsv12_reforecast",
                "forecast_value": float(l * 2.5 + int(r) * 0.1),
                "ens_spread": float(l * 0.5),
                "ens_members": [float(l * 2.5)] * 5,
                "n_members": 5,
            })
    forecasts_df = pd.DataFrame(f_rows)

    # Observations spanning t0 - 5 days to t0 + 15 days
    all_dates = pd.date_range(t0 - pd.Timedelta(days=5), t0 + pd.Timedelta(days=15))
    obs_rows = []
    for d in all_dates:
        for r in regions:
            obs_rows.append({
                "valid_date": d,
                "region_id": r,
                "variable": "rainfall",
                "observed_value": float(10.0 + int(r) * 0.2 + (d - t0).days),
                "source": "imd_gridded",
                "raw_date": d + pd.Timedelta(days=1),
            })
    obs_df = pd.DataFrame(obs_rows)

    return forecasts_df, obs_df


def test_future_observation_poisoning_bit_identity():
    """Overwrites all observations with valid_date > init_date with NaN/garbage.

    Asserts that every causal feature column computed at t0 is 100% bit-identical.
    """
    init_date = "2000-06-01"
    t0 = pd.Timestamp(init_date)
    f_df, clean_obs_df = create_synthetic_forecast_and_obs_fixtures(init_date=init_date)

    # 1. Compute causal prior observation feature table on CLEAN data
    clean_features = join_causal_prior_observations(f_df, clean_obs_df, lookback_days=1)

    # 2. Poison ALL data where valid_date > t0 with NaN and random garbage
    poisoned_obs_df = clean_obs_df.copy()
    future_mask = poisoned_obs_df["valid_date"] > t0

    # Overwrite future values with NaN, inf, and corrupting garbage
    poisoned_obs_df.loc[future_mask, "observed_value"] = np.nan
    assert poisoned_obs_df.loc[future_mask, "observed_value"].isna().all()

    # 3. Compute causal prior observation feature table on POISONED data
    poisoned_features = join_causal_prior_observations(f_df, poisoned_obs_df, lookback_days=1)

    # 4. Assert bit-identical equality across all columns and rows
    pd.testing.assert_frame_equal(clean_features, poisoned_features, check_exact=True)
    assert not clean_features["obs_lag_1d"].isna().any(), "Clean causal feature must not be NaN"
    assert not poisoned_features["obs_lag_1d"].isna().any(), "Poisoned future data must not leak into features"


def test_future_poisoning_with_extreme_values():
    """Poison future observations with extreme values (1e9) and assert zero leakage."""
    init_date = "2000-06-01"
    t0 = pd.Timestamp(init_date)
    f_df, clean_obs_df = create_synthetic_forecast_and_obs_fixtures(init_date=init_date)

    clean_features = join_causal_prior_observations(f_df, clean_obs_df, lookback_days=2)

    poisoned_obs_df = clean_obs_df.copy()
    future_mask = poisoned_obs_df["valid_date"] > t0
    poisoned_obs_df.loc[future_mask, "observed_value"] = 1e9  # Extreme contamination

    poisoned_features = join_causal_prior_observations(f_df, poisoned_obs_df, lookback_days=2)

    pd.testing.assert_frame_equal(clean_features, poisoned_features, check_exact=True)
    assert (poisoned_features["obs_lag_2d"] < 1000.0).all(), "Extreme future values must not contaminate features"


def test_causal_join_rejects_post_t0_observations():
    """Verify that attempting to join an observation table containing strictly future dates fails or filters them."""
    init_date = "2000-06-01"
    t0 = pd.Timestamp(init_date)
    f_df, _ = create_synthetic_forecast_and_obs_fixtures(init_date=init_date)

    # Observation table containing ONLY future dates
    future_only_obs = pd.DataFrame({
        "valid_date": pd.date_range(t0 + pd.Timedelta(days=1), periods=5),
        "region_id": ["7"] * 5,
        "variable": ["rainfall"] * 5,
        "observed_value": [999.0] * 5,
    })

    # When joining with a lookback prior to t0, future-only observations must result in NaN (no leak)
    res = join_causal_prior_observations(f_df[f_df["region_id"] == "7"], future_only_obs, lookback_days=1)
    assert res["obs_lag_1d"].isna().all(), "Future observations must never be joined as prior features"


def test_target_valid_date_calculation():
    """Verify forecast valid date arithmetic: lead 1 at t0 corresponds to weather on t0."""
    t0 = pd.Timestamp("2000-06-01")
    # Lead 1: weather on t0
    assert compute_target_valid_date(t0, 1) == pd.Timestamp("2000-06-01")
    # Lead 5: weather on t0 + 4 days
    assert compute_target_valid_date(t0, 5) == pd.Timestamp("2000-06-05")
    # Lead 10: weather on t0 + 9 days
    assert compute_target_valid_date(t0, 10) == pd.Timestamp("2000-06-10")
