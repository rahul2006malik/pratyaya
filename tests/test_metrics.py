"""Unit tests for Evaluation Metrics and Paired Block Bootstrap (Node F8, docs/05 §3, §4)."""

import numpy as np
import pandas as pd
import pytest

from pratyaya.evaluate.bootstrap import (
    block_bootstrap_metric,
    paired_block_bootstrap_comparison,
)
from pratyaya.evaluate.metrics import (
    compute_all_metrics,
    compute_auroc,
    compute_brier_score,
    compute_brier_skill_score,
    compute_cost_loss_value_curve,
    compute_pr_auc,
    compute_reliability_diagram,
    compute_stratified_metrics,
)


def test_discrimination_metrics() -> None:
    """Verify PR-AUC and AUROC behavior under known ground truth."""
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    # Perfect ranking
    y_perfect = np.array([0.9, 0.8, 0.1, 0.1, 0.05, 0.05, 0.02, 0.01, 0.01, 0.0])
    # Inverse ranking
    y_inverted = np.array([0.01, 0.02, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    assert compute_pr_auc(y_true, y_perfect) == 1.0
    assert compute_auroc(y_true, y_perfect) == 1.0

    assert compute_pr_auc(y_true, y_inverted) < 0.25
    assert compute_auroc(y_true, y_inverted) == 0.0

    # Single-class edge case
    assert compute_pr_auc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.0
    assert compute_auroc([0, 0, 0], [0.1, 0.2, 0.3]) == 0.5


def test_brier_score_and_bss() -> None:
    """Verify Brier Score and Brier Skill Score against reference forecast."""
    y_true = np.array([1, 0, 1, 0])

    # Perfect forecast
    assert compute_brier_score(y_true, [1.0, 0.0, 1.0, 0.0]) == 0.0

    # Constant 0.5 forecast
    assert compute_brier_score(y_true, [0.5, 0.5, 0.5, 0.5]) == 0.25

    # BSS: Perfect model vs 0.5 climatology
    bss_perfect = compute_brier_skill_score(
        y_true,
        y_prob=[1.0, 0.0, 1.0, 0.0],
        y_ref=[0.5, 0.5, 0.5, 0.5],
    )
    assert bss_perfect == 1.0

    # BSS: Climatology vs Climatology
    bss_identical = compute_brier_skill_score(
        y_true,
        y_prob=[0.5, 0.5, 0.5, 0.5],
        y_ref=[0.5, 0.5, 0.5, 0.5],
    )
    assert bss_identical == 0.0


def test_murphy_brier_decomposition() -> None:
    """Verify the exact Murphy (1973) identity: BS == Reliability - Resolution + Uncertainty."""
    rng = np.random.default_rng(123)
    y_true = rng.binomial(1, 0.1, size=1000)
    y_prob = np.clip(rng.beta(1, 10, size=1000), 0.0, 1.0)

    rel_diag = compute_reliability_diagram(y_true, y_prob, n_bins=10)

    binned_bs = rel_diag["binned_brier_score"]
    bs = rel_diag["brier_score"]
    rel = rel_diag["reliability"]
    res = rel_diag["resolution"]
    unc = rel_diag["uncertainty"]

    # Murphy (1973) exact identity on binned probabilities: BS_binned = Rel - Res + Unc
    reconstructed_bs = rel - res + unc
    assert abs(binned_bs - reconstructed_bs) < 1e-12
    # Continuous Brier score matches binned Brier score within discretization variance
    assert abs(bs - binned_bs) < 1e-3


def test_cost_loss_value_curve() -> None:
    """Verify Richardson/Murphy relative economic value curve V(alpha)."""
    y_true = np.array([1] * 10 + [0] * 90)  # 10% base rate
    # A model with high discrimination
    y_prob = np.array([0.8] * 10 + [0.05] * 90)

    vl = compute_cost_loss_value_curve(y_true, y_prob)

    assert "alpha" in vl
    assert "value" in vl
    assert vl["max_value"] > 0.0
    # Peak value should occur near base rate
    assert 0.05 <= vl["optimal_alpha"] <= 0.25


def test_stratified_metrics() -> None:
    """Verify metric computation grouped across lead days."""
    df = pd.DataFrame({
        "lead_day": [1, 1, 2, 2, 3, 3],
        "is_bust": [0, 1, 0, 1, 0, 1],
        "prob_bust": [0.2, 0.8, 0.3, 0.7, 0.4, 0.6],
    })

    strat = compute_stratified_metrics(df, y_true_col="is_bust", y_prob_col="prob_bust", group_col="lead_day")
    assert len(strat) == 3
    assert "lead_day" in strat.columns
    assert "pr_auc" in strat.columns
    assert "brier_score" in strat.columns


def test_block_bootstrap_init_dates() -> None:
    """Verify block bootstrap over initialization dates returns valid 95% CIs."""
    records = []
    # 20 distinct init dates, 10 rows per date
    for d_idx in range(20):
        dt = f"2000-06-{d_idx + 1:02d}"
        for r_idx in range(10):
            # Synthetic signal: probability correlates with bust
            prob = (d_idx + r_idx) / 30.0
            is_bust = int(prob > 0.6)
            records.append({
                "init_date": dt,
                "region_id": str(r_idx),
                "is_bust": is_bust,
                "prob_m": prob,
                "prob_b": 0.05,  # constant baseline
            })
    df = pd.DataFrame(records)

    # 1. Single model bootstrap
    boot_res = block_bootstrap_metric(
        df,
        y_true_col="is_bust",
        y_prob_col="prob_m",
        metric_name="PR-AUC",
        n_resamples=100,  # fast test
        random_state=42,
    )
    assert boot_res.n_clusters == 20
    assert 0.0 <= boot_res.ci_lower <= boot_res.point_estimate <= boot_res.ci_upper <= 1.0

    # 2. Paired difference bootstrap (Model vs Baseline)
    paired_res = paired_block_bootstrap_comparison(
        df,
        y_true_col="is_bust",
        y_prob_model_col="prob_m",
        y_prob_baseline_col="prob_b",
        metric_name="PR-AUC",
        n_resamples=100,
        random_state=42,
    )
    assert paired_res.delta_point_estimate > 0.0
    assert paired_res.ci_excludes_zero is True
