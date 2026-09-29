"""Unit tests for empirical rain-day alignment module (Node F2, D-001).

Validates:
- Single exported constant IMD_DAY_OFFSET == -1.
- Date alignment arithmetic and retention of raw_date alongside valid_date.
- Cross-correlation calculation across day lags.
"""

import numpy as np
import pandas as pd
import pytest

from pratyaya.align.rainday import (
    IMD_DAY_OFFSET,
    align_imd_observation_date,
    apply_imd_day_offset,
    compute_cross_correlation_series,
    summarize_correlation_table,
)


def test_imd_day_offset_constant():
    """Verify IMD_DAY_OFFSET is defined as -1 per D-001."""
    assert IMD_DAY_OFFSET == -1, f"Expected IMD_DAY_OFFSET == -1, got {IMD_DAY_OFFSET}"


def test_align_imd_observation_date():
    """Verify single date alignment shifts raw 0830 IST reporting date by -1 day."""
    raw = pd.Timestamp("2000-06-02")
    valid = align_imd_observation_date(raw)
    assert valid == pd.Timestamp("2000-06-01")

    # Works with strings and dates
    assert align_imd_observation_date("2005-07-15") == pd.Timestamp("2005-07-14")


def test_apply_imd_day_offset_dataframe():
    """Verify apply_imd_day_offset creates valid_date column while preserving raw_date."""
    dates = pd.date_range("2000-06-01", periods=5)
    df = pd.DataFrame({
        "raw_date": dates,
        "region_id": ["7"] * 5,
        "observed_value": [10.0, 20.0, 30.0, 40.0, 50.0],
    })

    result = apply_imd_day_offset(df)
    assert "valid_date" in result.columns
    assert "raw_date" in result.columns
    assert (result["valid_date"] == result["raw_date"] - pd.Timedelta(days=1)).all()
    assert len(result) == len(df)


def test_compute_cross_correlation_series():
    """Verify cross-correlation identifies known artificial lag peak."""
    rng = np.random.default_rng(42)
    n = 100
    dates = pd.date_range("2000-06-01", periods=n)
    signal = rng.standard_normal(n)

    # True truth at time t
    era5_s = pd.Series(signal, index=dates)
    # IMD recorded at t+1 (Lag -1 in our convention: IMD(t) = ERA5(t-1))
    imd_s = pd.Series(signal, index=dates).shift(1).bfill()

    corrs = compute_cross_correlation_series(imd_s, era5_s, lags=[-2, -1, 0, 1, 2])
    # Peak must be at lag -1
    best_lag = max(corrs.items(), key=lambda x: x[1])[0]
    assert best_lag == -1, f"Expected peak at lag -1, got {best_lag} with {corrs}"

    table = summarize_correlation_table({"Test Region": corrs})
    assert "Lag -1" in table.columns
    assert table.loc["Test Region", "Lag -1"] > 0.95
