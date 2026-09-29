"""Paired Block Bootstrap Module (Node F8).

Grain: Whole initialization dates (init_date) or independent events (event_id).

Implements the block bootstrap confidence interval procedure per
docs/05-evaluation-protocol.md §4.2:
- NEVER bootstrap individual rows: weather errors are spatially and temporally
  autocorrelated. Resampling individual rows understates standard errors by ~7×.
- Resample whole initialization dates (or independent meteorological events) with
  replacement across B = 1,000 bootstrap iterations.
- Supports paired block bootstrap comparison:
    Delta_b = metric(model, sample_b) - metric(baseline, sample_b)
  evaluated on identical resampled dates, producing the exact test for
  "did we beat the baseline" with honest 95% confidence intervals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pratyaya.evaluate.metrics import compute_pr_auc

logger = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    """Result of a single-model block bootstrap estimation."""
    metric_name: str
    point_estimate: float
    median: float
    ci_lower: float
    ci_upper: float
    std_err: float
    n_resamples: int
    n_clusters: int
    resamples: np.ndarray = field(repr=False)

    @property
    def ci_tuple(self) -> Tuple[float, float]:
        return (self.ci_lower, self.ci_upper)

    def summary(self) -> str:
        return (
            f"{self.metric_name}: {self.point_estimate:.4f} "
            f"[95% CI: {self.ci_lower:.4f}, {self.ci_upper:.4f}] "
            f"(SE={self.std_err:.4f}, N_clusters={self.n_clusters})"
        )


@dataclass
class PairedBootstrapResult:
    """Result of a paired-difference block bootstrap comparison between two models."""
    metric_name: str
    point_estimate_model: float
    point_estimate_baseline: float
    delta_point_estimate: float
    delta_median: float
    delta_ci_lower: float
    delta_ci_upper: float
    delta_std_err: float
    p_value: float
    ci_excludes_zero: bool
    n_resamples: int
    n_clusters: int
    delta_resamples: np.ndarray = field(repr=False)

    def summary(self) -> str:
        sig_str = "Excludes 0 (Statistically Significant)" if self.ci_excludes_zero else "Includes 0 (Not Significant)"
        return (
            f"{self.metric_name} Margin: {self.delta_point_estimate:+.4f} "
            f"[95% CI: {self.delta_ci_lower:+.4f}, {self.delta_ci_upper:+.4f}] "
            f"({sig_str}, p={self.p_value:.4f}, N_clusters={self.n_clusters})"
        )


def block_bootstrap_metric(
    df: pd.DataFrame,
    y_true_col: str = "is_bust",
    y_prob_col: str = "prob_bust",
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = compute_pr_auc,
    metric_name: str = "PR-AUC",
    group_col: str = "init_date",
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> BootstrapResult:
    """Compute confidence intervals by block bootstrap resampling whole cluster units (init dates).

    Args:
        df: Input DataFrame containing evaluated records.
        y_true_col: Binary target label column.
        y_prob_col: Forecast probability or score column.
        metric_fn: Function (y_true, y_prob) -> float. Default is compute_pr_auc.
        metric_name: Readable name of the metric (default: 'PR-AUC').
        group_col: Resampling block unit column (default: 'init_date').
        n_resamples: Number of bootstrap iterations (default: 1,000 per docs/05 §4.2).
        ci_level: Nominal confidence level (default: 0.95).
        random_state: Random seed for reproducibility.

    Returns:
        BootstrapResult: Dataclass containing point estimate, CI bounds, and bootstrap distribution.
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty for bootstrap estimation.")

    y_true = np.asarray(df[y_true_col].values, dtype=float)
    y_prob = np.asarray(df[y_prob_col].values, dtype=float)
    point_est = float(metric_fn(y_true, y_prob))

    # Identify cluster units (e.g. unique init dates)
    groups = df[group_col].values
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    n_clusters = len(unique_groups)

    if n_clusters < 2:
        return BootstrapResult(
            metric_name=metric_name,
            point_estimate=point_est,
            median=point_est,
            ci_lower=point_est,
            ci_upper=point_est,
            std_err=0.0,
            n_resamples=n_resamples,
            n_clusters=n_clusters,
            resamples=np.full(n_resamples, point_est),
        )

    # Pre-index row positions per cluster for fast vectorized extraction
    cluster_to_rows: List[np.ndarray] = [
        np.where(group_indices == g_idx)[0] for g_idx in range(n_clusters)
    ]

    rng = np.random.default_rng(random_state)
    boot_estimates = np.empty(n_resamples, dtype=float)

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for b in range(n_resamples):
        # Draw clusters with replacement
        sampled_clusters = rng.choice(n_clusters, size=n_clusters, replace=True)
        # Concatenate rows from sampled clusters
        sampled_rows = np.concatenate([cluster_to_rows[c] for c in sampled_clusters])

        sample_y_true = y_true[sampled_rows]
        sample_y_prob = y_prob[sampled_rows]

        boot_estimates[b] = float(metric_fn(sample_y_true, sample_y_prob))

    ci_lower = float(np.percentile(boot_estimates, lower_pct))
    ci_upper = float(np.percentile(boot_estimates, upper_pct))
    if metric_name.upper() in ("PR-AUC", "AUROC", "ROC-AUC", "AVERAGE PRECISION"):
        ci_lower = max(0.0, min(1.0, ci_lower))
        ci_upper = max(0.0, min(1.0, ci_upper))
        point_est = max(0.0, min(1.0, point_est))
    median_val = float(np.median(boot_estimates))
    std_err = float(np.std(boot_estimates, ddof=1))

    return BootstrapResult(
        metric_name=metric_name,
        point_estimate=point_est,
        median=median_val,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std_err=std_err,
        n_resamples=n_resamples,
        n_clusters=n_clusters,
        resamples=boot_estimates,
    )


def paired_block_bootstrap_comparison(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_model_col: str,
    y_prob_baseline_col: str,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = compute_pr_auc,
    metric_name: str = "PR-AUC",
    group_col: str = "init_date",
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_state: int = 42,
) -> PairedBootstrapResult:
    """Perform paired block bootstrap difference comparison between two models.

    Computes Delta = metric(model) - metric(baseline) on identical resampled blocks.
    Strictly follows docs/05 §4.2:
    "For paired comparisons (M2 vs B4), bootstrap the difference on the same resampled
    dates — a paired bootstrap is much tighter and is the correct test for 'did we
    beat the baseline.' If the 95% CI on your margin includes zero, say so."

    Args:
        df: Input DataFrame.
        y_true_col: Binary target label column.
        y_prob_model_col: Forecast probabilities for candidate model (e.g. M2).
        y_prob_baseline_col: Forecast probabilities for baseline model (e.g. B3 or B4).
        metric_fn: Metric evaluation function (y_true, y_prob) -> float.
        metric_name: Display metric name.
        group_col: Cluster grouping column (default: 'init_date').
        n_resamples: Number of bootstrap draws (default: 1,000).
        ci_level: Confidence level (default: 0.95).
        random_state: Random seed.

    Returns:
        PairedBootstrapResult: Structured comparison with margin, 95% CI, p-value,
                               and significance flag.
    """
    if df.empty:
        raise ValueError("DataFrame cannot be empty for paired bootstrap.")

    y_true = np.asarray(df[y_true_col].values, dtype=float)
    y_m = np.asarray(df[y_prob_model_col].values, dtype=float)
    y_b = np.asarray(df[y_prob_baseline_col].values, dtype=float)

    point_m = float(metric_fn(y_true, y_m))
    point_b = float(metric_fn(y_true, y_b))
    delta_point = point_m - point_b

    groups = df[group_col].values
    unique_groups, group_indices = np.unique(groups, return_inverse=True)
    n_clusters = len(unique_groups)

    if n_clusters < 2:
        return PairedBootstrapResult(
            metric_name=metric_name,
            point_estimate_model=point_m,
            point_estimate_baseline=point_b,
            delta_point_estimate=delta_point,
            delta_median=delta_point,
            delta_ci_lower=delta_point,
            delta_ci_upper=delta_point,
            delta_std_err=0.0,
            p_value=0.5 if delta_point == 0 else (0.0 if delta_point > 0 else 1.0),
            ci_excludes_zero=(delta_point != 0),
            n_resamples=n_resamples,
            n_clusters=n_clusters,
            delta_resamples=np.full(n_resamples, delta_point),
        )

    cluster_to_rows: List[np.ndarray] = [
        np.where(group_indices == g_idx)[0] for g_idx in range(n_clusters)
    ]

    rng = np.random.default_rng(random_state)
    delta_estimates = np.empty(n_resamples, dtype=float)

    alpha = 1.0 - ci_level
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    for b in range(n_resamples):
        sampled_clusters = rng.choice(n_clusters, size=n_clusters, replace=True)
        sampled_rows = np.concatenate([cluster_to_rows[c] for c in sampled_clusters])

        sample_y_true = y_true[sampled_rows]
        sample_y_m = y_m[sampled_rows]
        sample_y_b = y_b[sampled_rows]

        val_m = float(metric_fn(sample_y_true, sample_y_m))
        val_b = float(metric_fn(sample_y_true, sample_y_b))
        delta_estimates[b] = val_m - val_b

    ci_lower = float(np.percentile(delta_estimates, lower_pct))
    ci_upper = float(np.percentile(delta_estimates, upper_pct))
    median_delta = float(np.median(delta_estimates))
    std_err = float(np.std(delta_estimates, ddof=1))

    # Two-sided empirical p-value for hypothesis Delta == 0
    p_le_zero = float(np.mean(delta_estimates <= 0.0))
    p_ge_zero = float(np.mean(delta_estimates >= 0.0))
    p_value = float(2.0 * min(p_le_zero, p_ge_zero))

    # CI excludes zero if both bounds have the same sign (and neither is zero)
    ci_excludes_zero = bool((ci_lower > 0.0 and ci_upper > 0.0) or (ci_lower < 0.0 and ci_upper < 0.0))

    return PairedBootstrapResult(
        metric_name=metric_name,
        point_estimate_model=point_m,
        point_estimate_baseline=point_b,
        delta_point_estimate=delta_point,
        delta_median=median_delta,
        delta_ci_lower=ci_lower,
        delta_ci_upper=ci_upper,
        delta_std_err=std_err,
        p_value=min(1.0, p_value),
        ci_excludes_zero=ci_excludes_zero,
        n_resamples=n_resamples,
        n_clusters=n_clusters,
        delta_resamples=delta_estimates,
    )
