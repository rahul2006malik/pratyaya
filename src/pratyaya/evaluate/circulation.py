"""NCMRWF Atmospheric Circulation & Multi-Lead Verification Framework.

Project Pratyaya (SIH26079) - Incorporating expert recommendations from
NCMRWF Senior Scientists (Dr. Mohana Satyanarayana et al.).

Key Scientific Capabilities:
1. Geopotential Height Anomaly Correlation Coefficient (Z500 ACC):
   Standard WMO latitude-weighted ACC between forecast and reanalysis anomalies.
2. 850 hPa Wind Vector RMSE (V850 RMSE):
   Low-level monsoon wind circulation vector root-mean-squared error.
3. Lead- and Season-Standardized Error Tails:
   Normalizes forecast error per (region_id, lead_day, month) cell:
   z = (e - mu_{r,L,m}) / sigma_{r,L,m}.
   Identifies true dynamic forecast busts in distribution tails (|z| >= 2.0).
4. Multi-Lead Error Trajectory Dynamics (D+3 -> D+7):
   Tracks compounding error growth targeting fixed valid dates across medium-range
   leads to distinguish systematic dynamical tracking breakdown from stochastic noise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def compute_geopotential_acc(
    z_fcst: np.ndarray,
    z_obs: np.ndarray,
    clim_mean: np.ndarray,
    lats: Optional[np.ndarray] = None,
) -> float:
    """Compute latitude-weighted Anomaly Correlation Coefficient (ACC) for Z500.

    Follows WMO standard verification formula:
        ACC = sum(w * f' * a') / sqrt(sum(w * (f')^2) * sum(w * (a')^2))
    where f' = z_fcst - clim_mean, a' = z_obs - clim_mean,
    and w = cos(latitude) accounts for grid area convergence towards poles.

    Args:
        z_fcst: Forecast geopotential height grid (2D or flattened).
        z_obs: Observed / reanalysis geopotential height grid.
        clim_mean: Climatological mean geopotential height field.
        lats: Latitude array matching grid rows if 2D, or 1D array of cell latitudes.

    Returns:
        float: ACC score in [-1.0, 1.0].
    """
    f_prime = np.asarray(z_fcst, dtype=float) - np.asarray(clim_mean, dtype=float)
    a_prime = np.asarray(z_obs, dtype=float) - np.asarray(clim_mean, dtype=float)

    # Flatten arrays
    f_flat = f_prime.ravel()
    a_flat = a_prime.ravel()

    # Determine area weighting
    if lats is not None:
        if z_fcst.ndim == 2 and len(lats) == z_fcst.shape[0]:
            # 2D grid: tile cosine weights across columns
            cos_w = np.cos(np.radians(lats))
            w = np.repeat(cos_w[:, np.newaxis], z_fcst.shape[1], axis=1).ravel()
        elif len(lats) == len(f_flat):
            w = np.cos(np.radians(lats))
        else:
            w = np.ones_like(f_flat)
    else:
        w = np.ones_like(f_flat)

    # Remove any NaN elements
    valid_mask = ~(np.isnan(f_flat) | np.isnan(a_flat) | np.isnan(w))
    f_v = f_flat[valid_mask]
    a_v = a_flat[valid_mask]
    w_v = w[valid_mask]

    if len(f_v) == 0:
        return 0.0

    numerator = np.sum(w_v * f_v * a_v)
    denom_f = np.sum(w_v * (f_v ** 2))
    denom_a = np.sum(w_v * (a_v ** 2))
    denominator = np.sqrt(denom_f * denom_a)

    if denominator < 1e-12:
        return 0.0

    acc = float(numerator / denominator)
    return float(np.clip(acc, -1.0, 1.0))


def compute_wind_vector_rmse(
    u_fcst: np.ndarray,
    v_fcst: np.ndarray,
    u_obs: np.ndarray,
    v_obs: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> float:
    """Compute root-mean-squared vector error (RMSE) for 850 hPa wind fields.

    Formula:
        RMSE_V = sqrt( (1 / sum(w)) * sum( w * ((u_fc - u_obs)^2 + (v_fc - v_obs)^2) ) )

    Captures low-level monsoon Somali Jet and cyclonic vortex tracking errors.

    Args:
        u_fcst: Forecast zonal (U) wind component (m/s).
        v_fcst: Forecast meridional (V) wind component (m/s).
        u_obs: Observed zonal wind component (m/s).
        v_obs: Observed meridional wind component (m/s).
        weights: Optional spatial area weights.

    Returns:
        float: Wind vector RMSE in m/s.
    """
    diff_u = np.asarray(u_fcst, dtype=float) - np.asarray(u_obs, dtype=float)
    diff_v = np.asarray(v_fcst, dtype=float) - np.asarray(v_obs, dtype=float)

    vector_diff_sq = (diff_u ** 2) + (diff_v ** 2)
    v_flat = vector_diff_sq.ravel()

    if weights is not None:
        w_flat = np.asarray(weights, dtype=float).ravel()
        valid = ~(np.isnan(v_flat) | np.isnan(w_flat))
        if np.sum(w_flat[valid]) < 1e-12:
            return 0.0
        mean_sq = np.sum(w_flat[valid] * v_flat[valid]) / np.sum(w_flat[valid])
    else:
        valid = ~np.isnan(v_flat)
        if np.sum(valid) == 0:
            return 0.0
        mean_sq = np.mean(v_flat[valid])

    return float(np.sqrt(max(0.0, mean_sq)))


def compute_lead_season_standardized_error(
    df: pd.DataFrame,
    error_col: str = "error_signed",
    date_col: str = "init_date",
    region_col: str = "region_id",
    lead_col: str = "lead_day",
    z_tail_threshold: float = 2.0,
    training_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute lead- and season-standardized forecast error tails per NCMRWF guidance.

    Normalizes errors against the climatological error distribution fitted
    strictly for each (region_id, lead_day, calendar_month) cell:
        z(r, L, m) = (error - mu_{r,L,m}) / (sigma_{r,L,m} + eps)

    This isolates anomalous dynamic failure from the expected seasonal and lead-time
    degradation, preventing wet regions (e.g. Konkan) or high leads (Day 7) from
    dominating raw-threshold error tallies.

    Args:
        df: Input DataFrame with forecast errors.
        error_col: Column containing signed error (forecast - observed).
        date_col: Initialization date column.
        region_col: Subdivision identifier column.
        lead_col: Forecast lead day column (1..10).
        z_tail_threshold: Number of standard deviations defining a tail bust (|z| >= threshold).
        training_mask: Optional boolean series defining training records to fit (mu, sigma).
                       If None, fits on all records in df.

    Returns:
        pd.DataFrame: Copy of df with added columns:
            ['error_cell_mean', 'error_cell_std', 'error_z_score', 'is_tail_bust'].
    """
    out = df.copy()
    dt = pd.to_datetime(out[date_col])
    out["month"] = dt.dt.month

    # Subset for fitting cell parameters
    fit_df = out[training_mask] if training_mask is not None else out

    stats = (
        fit_df.groupby([region_col, lead_col, "month"])[error_col]
        .agg(error_cell_mean="mean", error_cell_std="std")
        .reset_index()
    )

    # Guard against zero or NaN std
    stats["error_cell_std"] = stats["error_cell_std"].fillna(1.0)
    stats["error_cell_std"] = np.where(stats["error_cell_std"] < 1e-4, 1.0, stats["error_cell_std"])

    # Merge cell statistics
    merged = pd.merge(out, stats, on=[region_col, lead_col, "month"], how="left")
    merged["error_cell_mean"] = merged["error_cell_mean"].fillna(0.0)
    merged["error_cell_std"] = merged["error_cell_std"].fillna(1.0)

    # Compute standardized z-score
    merged["error_z_score"] = (merged[error_col] - merged["error_cell_mean"]) / merged["error_cell_std"]
    merged["is_tail_bust"] = (merged["error_z_score"].abs() >= z_tail_threshold).astype(int)

    return merged


@dataclass
class TrajectoryDynamicsResult:
    """Summary metrics of multi-lead forecast error compounding."""
    n_trajectories: int
    mean_error_growth_rate: float
    monotonically_compounding_fraction: float
    mean_spearman_rho: float


def compute_multilead_trajectory_metrics(
    df: pd.DataFrame,
    date_col: str = "init_date",
    lead_col: str = "lead_day",
    region_col: str = "region_id",
    error_col: str = "error_signed",
    leads: Sequence[int] = (3, 4, 5, 6, 7),
) -> Tuple[pd.DataFrame, TrajectoryDynamicsResult]:
    """Analyze multi-lead error compounding targeting identical valid dates (D+3 -> D+7).

    Distinguishes systematic dynamical breakdown (monotonic compounding error growth)
    from stochastic daily variance.

    For each valid date V = init_date + lead_day and subdivision R:
        Tracks |error| as lead increases from 3 to 7.

    Args:
        df: Input DataFrame containing forecast errors.
        date_col: Initialization date column.
        lead_col: Forecast lead day column.
        region_col: Subdivision identifier column.
        error_col: Signed or absolute error column.
        leads: Sequence of lead days to trace (default: 3, 4, 5, 6, 7).

    Returns:
        Tuple[pd.DataFrame, TrajectoryDynamicsResult]:
            - DataFrame with trajectory metrics per (valid_date, region_id).
            - TrajectoryDynamicsResult object with aggregated compounding statistics.
    """
    sub = df[df[lead_col].isin(leads)].copy()
    init_dt = pd.to_datetime(sub[date_col])
    sub["valid_date"] = init_dt + pd.to_timedelta(sub[lead_col], unit="D")
    sub["abs_error"] = sub[error_col].abs()

    # Pivot to track lead progression for each (valid_date, region_id)
    pivot = sub.pivot_table(
        index=["valid_date", region_col],
        columns=lead_col,
        values="abs_error",
    ).dropna()

    if pivot.empty:
        empty_res = TrajectoryDynamicsResult(
            n_trajectories=0,
            mean_error_growth_rate=0.0,
            monotonically_compounding_fraction=0.0,
            mean_spearman_rho=0.0,
        )
        return pd.DataFrame(), empty_res

    lead_cols = sorted(list(leads))
    vals = pivot[lead_cols].values  # shape: (n_cases, len(leads))

    # Error growth rate: (Error_Lmax - Error_Lmin) / (Lmax - Lmin)
    l_min, l_max = lead_cols[0], lead_cols[-1]
    growth_rates = (vals[:, -1] - vals[:, 0]) / float(l_max - l_min)

    # Monotonic compounding: is each step non-decreasing?
    is_monotonic = np.all(np.diff(vals, axis=1) >= 0.0, axis=1)

    # Spearman correlation along leads
    lead_arr = np.array(lead_cols)
    rhos = []
    for row in vals:
        if np.std(row) < 1e-8:
            rhos.append(0.0)
        else:
            r, _ = spearmanr(lead_arr, row)
            rhos.append(float(r) if not np.isnan(r) else 0.0)

    rhos_arr = np.array(rhos)

    result_df = pivot.reset_index()
    result_df["error_growth_rate"] = growth_rates
    result_df["is_monotonic_compounding"] = is_monotonic
    result_df["spearman_rho"] = rhos_arr

    summary = TrajectoryDynamicsResult(
        n_trajectories=len(result_df),
        mean_error_growth_rate=float(np.mean(growth_rates)),
        monotonically_compounding_fraction=float(np.mean(is_monotonic)),
        mean_spearman_rho=float(np.mean(rhos_arr)),
    )

    logger.info(
        "Multi-lead trajectory analysis: %d valid cases, %.1f%% monotonically compounding, "
        "mean Spearman rho = %.3f, mean growth rate = %.2f mm/day",
        summary.n_trajectories,
        summary.monotonically_compounding_fraction * 100,
        summary.mean_spearman_rho,
        summary.mean_error_growth_rate,
    )
    return result_df, summary
