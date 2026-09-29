"""Unit tests for NCMRWF circulation verification module (Node F8).

Tests:
- Z500 Geopotential Height Anomaly Correlation Coefficient (ACC)
- 850 hPa Wind Vector RMSE
- Lead- and season-standardized error normalization
- Multi-lead error trajectories (D+3 -> D+7)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pratyaya.evaluate.circulation import (
    compute_geopotential_acc,
    compute_lead_season_standardized_error,
    compute_multilead_trajectory_metrics,
    compute_wind_vector_rmse,
)


def test_geopotential_acc_identical():
    """Identical anomaly field must produce ACC = 1.0."""
    z_obs = np.array([[5800.0, 5820.0], [5840.0, 5860.0]])
    z_fcst = z_obs.copy()
    clim = np.array([[5700.0, 5700.0], [5700.0, 5700.0]])
    lats = np.array([20.0, 30.0])

    acc = compute_geopotential_acc(z_fcst, z_obs, clim, lats=lats)
    assert np.isclose(acc, 1.0, atol=1e-5)


def test_geopotential_acc_inverted():
    """Exact opposite anomaly field must produce ACC = -1.0."""
    clim = np.array([[5700.0, 5700.0], [5700.0, 5700.0]])
    z_obs = clim + np.array([[10.0, -10.0], [20.0, -20.0]])
    z_fcst = clim - np.array([[10.0, -10.0], [20.0, -20.0]])

    acc = compute_geopotential_acc(z_fcst, z_obs, clim)
    assert np.isclose(acc, -1.0, atol=1e-5)


def test_geopotential_acc_cosine_weighting():
    """Equatorial cells should receive higher weight than polar cells under cosine weighting."""
    lats = np.array([0.0, 60.0])  # cos(0) = 1.0, cos(60) = 0.5
    clim = np.zeros((2, 1))

    # Pattern A matches at equator, disagrees at 60N
    z_obs = np.array([[10.0], [10.0]])
    z_fcst = np.array([[10.0], [-10.0]])

    acc = compute_geopotential_acc(z_fcst, z_obs, clim, lats=lats)
    # Equatorial contribution: 1.0 * 10 * 10 = +100
    # 60N contribution: 0.5 * 10 * (-10) = -50
    # Net numerator is positive (+50)
    assert acc > 0.0


def test_wind_vector_rmse_zero():
    """Identical wind fields must yield 0.0 vector RMSE."""
    u = np.array([12.5, 14.0, 18.2])
    v = np.array([-4.0, -2.5, 0.5])

    rmse = compute_wind_vector_rmse(u, v, u, v)
    assert np.isclose(rmse, 0.0, atol=1e-6)


def test_wind_vector_rmse_known():
    """Vector differences with magnitude 3 and 4 should yield sqrt(3^2 + 4^2) = 5.0."""
    u_fc = np.array([13.0])
    v_fc = np.array([4.0])
    u_obs = np.array([10.0])
    v_obs = np.array([0.0])

    rmse = compute_wind_vector_rmse(u_fc, v_fc, u_obs, v_obs)
    assert np.isclose(rmse, 5.0, atol=1e-5)


def test_lead_season_standardized_error():
    """Standardized error should center errors at 0 with std ~ 1, flagging tail busts."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "init_date": pd.date_range("2000-06-01", periods=n, freq="D"),
        "region_id": 1,
        "lead_day": 3,
        "error_signed": np.random.normal(loc=0.0, scale=1.0, size=n),
    })

    # Inject extreme tail bust at index 0 (+4.0 sigma)
    df.loc[0, "error_signed"] = 4.0

    res = compute_lead_season_standardized_error(df, z_tail_threshold=2.0)
    assert "error_z_score" in res.columns
    assert "is_tail_bust" in res.columns
    assert res.loc[0, "is_tail_bust"] == 1
    assert res.loc[0, "error_z_score"] > 2.0


def test_multilead_trajectory_metrics():
    """Should correctly identify monotonic compounding error trajectories."""
    # Create synthetic trajectory for 1 subdivision
    records = []
    # Valid date: 2000-06-10
    target_valid = pd.Timestamp("2000-06-10")
    for lead in [3, 4, 5, 6, 7]:
        init = target_valid - pd.Timedelta(days=lead)
        # Compounding error: 2, 4, 6, 8, 10 mm
        records.append({
            "init_date": init,
            "region_id": 1,
            "lead_day": lead,
            "error_signed": float(lead * 2.0),
        })

    df = pd.DataFrame(records)
    res_df, summary = compute_multilead_trajectory_metrics(df, leads=(3, 4, 5, 6, 7))

    assert len(res_df) == 1
    assert summary.n_trajectories == 1
    assert summary.monotonically_compounding_fraction == 1.0
    assert np.isclose(summary.mean_spearman_rho, 1.0, atol=1e-5)
    assert np.isclose(summary.mean_error_growth_rate, 2.0, atol=1e-5)
