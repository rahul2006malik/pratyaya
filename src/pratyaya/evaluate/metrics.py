"""Metric Calculation Module (Node F8).

Grain: 1D ground-truth labels y_true and forecast probabilities/scores y_prob.

Implements the evaluation metrics suite per docs/05-evaluation-protocol.md §3:
1. Discrimination:
   - PR-AUC (Average Precision score, primary metric under ~4–10% base rate).
   - AUROC (Area under the ROC curve).
2. Probabilistic Calibration:
   - Brier Score (mean squared probability error).
   - Brier Skill Score (BSS vs B1 conditional climatology reference).
   - Reliability diagram calibration curves with sample counts and observed frequencies.
   - Murphy (1973) Brier score resolution/reliability/uncertainty decomposition:
     BS = Reliability - Resolution + Uncertainty.
3. Decision-Oriented Evaluation:
   - Relative economic value V(alpha) as a function of cost-loss ratio alpha in (0, 1)
     per Murphy (1977) and Richardson (2000).
4. Stratified Breakdowns:
   - Evaluates performance stratified by lead day (1–10), season, and region group.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

logger = logging.getLogger(__name__)


def compute_pr_auc(y_true: Sequence[Union[bool, int, float]], y_prob: Sequence[float]) -> float:
    """Compute Precision-Recall Area Under the Curve (PR-AUC / Average Precision).

    Primary discrimination metric per docs/05 §3.1.
    Handles single-class or degenerate edge cases safely.

    Args:
        y_true: Binary ground truth indicators (0 or 1, False or True).
        y_prob: Predicted probability of bust in [0.0, 1.0].

    Returns:
        float: PR-AUC score in [0.0, 1.0].
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0:
        return 0.0

    n_pos = int(y_t.sum())
    if n_pos == 0:
        # No positives present: baseline is 0.0
        return 0.0
    if n_pos == len(y_t):
        # All positives present: baseline is 1.0
        return 1.0

    return float(average_precision_score(y_t, y_p))


def compute_auroc(y_true: Sequence[Union[bool, int, float]], y_prob: Sequence[float]) -> float:
    """Compute Area Under Receiver Operating Characteristic curve (AUROC).

    Reported for completeness per docs/05 §3.1.

    Args:
        y_true: Binary ground truth indicators.
        y_prob: Predicted probability or ranking score.

    Returns:
        float: AUROC score in [0.0, 1.0]. Returns 0.5 if single class.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_prob, dtype=float)

    if len(y_t) == 0 or len(np.unique(y_t)) < 2:
        return 0.5

    try:
        return float(roc_auc_score(y_t, y_p))
    except ValueError:
        return 0.5


def compute_brier_score(y_true: Sequence[Union[bool, int, float]], y_prob: Sequence[float]) -> float:
    """Compute Brier Score (mean squared error of probability forecasts).

    BS = (1/N) * sum((p_i - y_i)^2). Range [0.0, 1.0], lower is better.

    Args:
        y_true: Binary ground truth indicators.
        y_prob: Predicted probabilities in [0.0, 1.0].

    Returns:
        float: Brier score.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

    if len(y_t) == 0:
        return 0.0

    return float(np.mean((y_p - y_t) ** 2))


def compute_brier_skill_score(
    y_true: Sequence[Union[bool, int, float]],
    y_prob: Sequence[float],
    y_ref: Sequence[float],
) -> float:
    """Compute Brier Skill Score (BSS) against a reference climatology.

    BSS = 1 - (BS_model / BS_ref).
    Positive BSS indicates skill superior to the reference forecast.
    Per docs/05 §3.1, reference should be B1 conditional climatology.

    Args:
        y_true: Binary ground truth indicators.
        y_prob: Model predicted probabilities.
        y_ref: Reference predicted probabilities (e.g. B1 conditional climatology).

    Returns:
        float: Brier Skill Score. Returns 0.0 if BS_ref is 0.
    """
    bs_model = compute_brier_score(y_true, y_prob)
    bs_ref = compute_brier_score(y_true, y_ref)

    if bs_ref <= 1e-12:
        return 0.0 if bs_model <= 1e-12 else -float("inf")

    return float(1.0 - (bs_model / bs_ref))


def compute_reliability_diagram(
    y_true: Sequence[Union[bool, int, float]],
    y_prob: Sequence[float],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """Compute reliability calibration curve and Murphy (1973) decomposition.

    Decomposes Brier score into:
        BS = Reliability - Resolution + Uncertainty
    where:
        Uncertainty = o_bar * (1 - o_bar)
        Reliability = (1/N) * sum_k N_k * (p_k_bar - o_k_bar)^2   (lower is better)
        Resolution  = (1/N) * sum_k N_k * (o_k_bar - o_bar)^2     (higher is better)

    Args:
        y_true: Binary ground truth indicators.
        y_prob: Model predicted probabilities in [0.0, 1.0].
        n_bins: Number of probability bins (default: 10).
        strategy: 'uniform' for equal-width bins, or 'quantile' for equal-count bins.

    Returns:
        Dict[str, Any]: Dictionary containing:
            - 'prob_pred': Mean predicted probability per bin.
            - 'prob_true': Observed relative frequency per bin.
            - 'bin_counts': Sample count per bin.
            - 'bin_edges': Bin interval edges.
            - 'brier_score': Total Brier score.
            - 'reliability': Murphy reliability component.
            - 'resolution': Murphy resolution component.
            - 'uncertainty': Murphy uncertainty component.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_t)

    if n == 0:
        return {
            "prob_pred": np.array([]),
            "prob_true": np.array([]),
            "bin_counts": np.array([]),
            "bin_edges": np.linspace(0.0, 1.0, n_bins + 1),
            "brier_score": 0.0,
            "reliability": 0.0,
            "resolution": 0.0,
            "uncertainty": 0.0,
        }

    o_bar = float(np.mean(y_t))
    uncertainty = o_bar * (1.0 - o_bar)

    if strategy == "quantile":
        # Quantile binning with uniqueness check
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.unique(np.quantile(y_p, quantiles))
        if len(bin_edges) < 2:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    # Assign each sample to a bin
    bin_indices = np.digitize(y_p, bin_edges[1:-1])

    prob_pred_list: List[float] = []
    prob_true_list: List[float] = []
    bin_counts_list: List[int] = []

    reliability = 0.0
    resolution = 0.0
    within_bin_var = 0.0

    actual_n_bins = len(bin_edges) - 1
    for k in range(actual_n_bins):
        mask = (bin_indices == k)
        n_k = int(mask.sum())
        if n_k > 0:
            p_k_bar = float(np.mean(y_p[mask]))
            o_k_bar = float(np.mean(y_t[mask]))
            prob_pred_list.append(p_k_bar)
            prob_true_list.append(o_k_bar)
            bin_counts_list.append(n_k)

            reliability += (n_k / n) * ((p_k_bar - o_k_bar) ** 2)
            resolution += (n_k / n) * ((o_k_bar - o_bar) ** 2)
            within_bin_var += float(np.sum((y_p[mask] - p_k_bar) ** 2)) / n
        else:
            midpoint = float(0.5 * (bin_edges[k] + bin_edges[k + 1]))
            prob_pred_list.append(midpoint)
            prob_true_list.append(0.0)
            bin_counts_list.append(0)

    bs = float(np.mean((y_p - y_t) ** 2))
    binned_bs = float(reliability - resolution + uncertainty)

    return {
        "prob_pred": np.array(prob_pred_list),
        "prob_true": np.array(prob_true_list),
        "bin_counts": np.array(bin_counts_list),
        "bin_edges": bin_edges,
        "brier_score": bs,
        "binned_brier_score": binned_bs,
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "within_bin_var": float(within_bin_var),
    }


def compute_cost_loss_value_curve(
    y_true: Sequence[Union[bool, int, float]],
    y_prob: Sequence[float],
    alpha_grid: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute Richardson (2000) & Murphy (1977) relative economic value curve V(alpha).

    For an operational user with cost-loss ratio alpha = C / L in (0, 1):
        E_clim(alpha) = min(alpha, o_bar)
        E_perf(alpha) = alpha * o_bar
        E_fcst(alpha) = alpha * (o_bar * H + (1 - o_bar) * F) + o_bar * (1 - H)
        V(alpha)      = (E_clim - E_fcst) / (E_clim - E_perf)
    where:
        o_bar is the base rate (climatological frequency)
        H is the hit rate (sensitivity / recall) at decision threshold p_crit = alpha
        F is the false alarm rate (1 - specificity) at decision threshold p_crit = alpha

    Args:
        y_true: Binary ground truth indicators.
        y_prob: Predicted bust probabilities in [0.0, 1.0].
        alpha_grid: 1D array of cost-loss ratios in (0, 1) to evaluate.
                    Default: 99 points from 0.01 to 0.99.

    Returns:
        Dict[str, Any]:
            - 'alpha': Array of cost-loss ratios.
            - 'value': Array of relative economic value V(alpha) in [-inf, 1.0].
            - 'max_value': Peak economic value.
            - 'optimal_alpha': Alpha where peak value occurs.
            - 'positive_range': (alpha_min, alpha_max) where V(alpha) > 0.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    n = len(y_t)

    if alpha_grid is None:
        alpha_grid = np.linspace(0.01, 0.99, 99)

    if n == 0 or y_t.sum() == 0 or y_t.sum() == n:
        return {
            "alpha": alpha_grid,
            "value": np.zeros_like(alpha_grid),
            "max_value": 0.0,
            "optimal_alpha": 0.5,
            "positive_range": (0.0, 0.0),
        }

    o_bar = float(y_t.mean())
    n_pos = int(y_t.sum())
    n_neg = n - n_pos

    v_list: List[float] = []

    for alpha in alpha_grid:
        # Climatological and perfect forecast expected loss per unit L
        e_clim = min(alpha, o_bar)
        e_perf = alpha * o_bar

        denom = e_clim - e_perf
        if denom <= 1e-12:
            v_list.append(0.0)
            continue

        # Rational decision threshold for cost-loss ratio alpha is p_crit = alpha
        # Alert is issued if p >= alpha
        pred_alert = (y_p >= alpha)

        tp = int((pred_alert & (y_t == 1)).sum())
        fp = int((pred_alert & (y_t == 0)).sum())

        h = tp / n_pos if n_pos > 0 else 0.0
        f = fp / n_neg if n_neg > 0 else 0.0

        # Forecast expected loss per unit L: C*(TP+FP)/N + L*FN/N
        # = alpha * (o_bar * H + (1 - o_bar) * F) + o_bar * (1 - H)
        e_fcst = alpha * (o_bar * h + (1.0 - o_bar) * f) + o_bar * (1.0 - h)

        v_alpha = (e_clim - e_fcst) / denom
        v_list.append(float(v_alpha))

    v_arr = np.array(v_list)
    max_idx = int(np.argmax(v_arr))
    max_v = float(v_arr[max_idx])
    opt_alpha = float(alpha_grid[max_idx])

    # Find range where value is positive
    pos_mask = v_arr > 0.0
    if pos_mask.any():
        pos_alphas = alpha_grid[pos_mask]
        pos_range = (float(pos_alphas[0]), float(pos_alphas[-1]))
    else:
        pos_range = (0.0, 0.0)

    return {
        "alpha": alpha_grid,
        "value": v_arr,
        "max_value": max_v,
        "optimal_alpha": opt_alpha,
        "positive_range": pos_range,
    }


def compute_all_metrics(
    y_true: Sequence[Union[bool, int, float]],
    y_prob: Sequence[float],
    y_ref: Optional[Sequence[float]] = None,
) -> Dict[str, Any]:
    """Compute complete suite of primary and secondary evaluation metrics.

    Args:
        y_true: Binary ground-truth labels.
        y_prob: Predicted model probabilities.
        y_ref: Optional reference probabilities for BSS calculation.

    Returns:
        Dict[str, Any]: Summary dictionary of all metrics.
    """
    pr = compute_pr_auc(y_true, y_prob)
    roc = compute_auroc(y_true, y_prob)
    bs = compute_brier_score(y_true, y_prob)
    bss = compute_brier_skill_score(y_true, y_prob, y_ref) if y_ref is not None else None
    rel_diag = compute_reliability_diagram(y_true, y_prob)
    vl = compute_cost_loss_value_curve(y_true, y_prob)

    out = {
        "pr_auc": pr,
        "auroc": roc,
        "brier_score": bs,
        "brier_skill_score": bss,
        "reliability": rel_diag["reliability"],
        "resolution": rel_diag["resolution"],
        "uncertainty": rel_diag["uncertainty"],
        "max_value_v": vl["max_value"],
        "optimal_cost_loss_alpha": vl["optimal_alpha"],
        "positive_value_band": vl["positive_range"],
        "n_samples": len(y_true),
        "base_rate": float(np.mean(y_true)) if len(y_true) > 0 else 0.0,
    }
    return out


def compute_stratified_metrics(
    df: pd.DataFrame,
    y_true_col: str = "is_bust",
    y_prob_col: str = "prob_bust",
    y_ref_col: Optional[str] = None,
    group_col: str = "lead_day",
) -> pd.DataFrame:
    """Compute primary evaluation metrics stratified by a grouping column (e.g. lead_day).

    Args:
        df: Input DataFrame.
        y_true_col: Binary truth label column name.
        y_prob_col: Predicted model probability column name.
        y_ref_col: Optional reference probability column name for BSS.
        group_col: Column to group by (default: 'lead_day').

    Returns:
        pd.DataFrame: Table with per-stratum metrics.
    """
    rows = []
    for grp_val, grp_df in df.groupby(group_col):
        y_t = grp_df[y_true_col]
        y_p = grp_df[y_prob_col]
        y_r = grp_df[y_ref_col] if y_ref_col and y_ref_col in grp_df.columns else None

        metrics = compute_all_metrics(y_t, y_p, y_r)
        metrics[group_col] = grp_val
        rows.append(metrics)

    res_df = pd.DataFrame(rows)
    cols = [group_col] + [c for c in res_df.columns if c != group_col]
    return res_df[cols]
