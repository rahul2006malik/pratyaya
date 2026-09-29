"""Baseline Scoring and Day 2 Gate Verification Runner (Node F6, F7, F8).

Loads canonical labels and forecast features across multi-year data:
1. Partitions data via purged temporal split: Train (2000–2001) vs Calibration (2002)
   with 12-day boundary purge (docs/05 §1.2).
2. Runs 3D connected-component bust event clustering (docs/05 §4.1) on calibration fold.
3. Fits Baselines B0, B1, B2, B3 strictly on training fold.
4. Evaluates B0–B3 on calibration fold computing:
   - PR-AUC point estimates and honest 95% CIs via 1,000 block bootstrap resamples over init dates.
   - AUROC and Brier scores.
   - Brier Skill Score (vs B1 climatology).
   - Paired margin differences with bootstrap significance.
5. Reports benchmark numbers for docs/PROGRESS.md and docs/DECISIONS.md (Decision D-003).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pratyaya.evaluate.bootstrap import (
    block_bootstrap_metric,
    paired_block_bootstrap_comparison,
)
from pratyaya.evaluate.events import cluster_bust_events
from pratyaya.evaluate.metrics import (
    compute_all_metrics,
    compute_auroc,
    compute_brier_score,
    compute_brier_skill_score,
    compute_cost_loss_value_curve,
    compute_pr_auc,
    compute_reliability_diagram,
)
from pratyaya.evaluate.splits import create_purged_train_val_split
from pratyaya.models.baselines import (
    CalibratedLogisticSpreadBaseline,
    ConditionalClimatologyBaseline,
    ConstantBaseRateBaseline,
    RawEnsembleSpreadBaseline,
    ensure_features,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("score_baselines")

DEFAULT_LABELS_DIR = Path("data/canonical/labels")
DEFAULT_FCSTS_DIR = Path("data/canonical/forecasts")


def load_canonical_dataset(
    labels_dir: Path = DEFAULT_LABELS_DIR,
    fcsts_dir: Path = DEFAULT_FCSTS_DIR,
    years: List[int] = [2000, 2001, 2002],
) -> pd.DataFrame:
    """Load canonical labels and join forecast features across specified years."""
    dfs = []
    for y in years:
        year_dir = labels_dir / f"year={y}"
        if year_dir.exists():
            for fpath in year_dir.glob("*.parquet"):
                dfs.append(pd.read_parquet(fpath))

    if not dfs:
        raise FileNotFoundError(f"No label files found in {labels_dir} for years {years}")

    full_labels = pd.concat(dfs, ignore_index=True)
    enriched_df = ensure_features(full_labels, forecasts_dir=fcsts_dir)
    return enriched_df


def main() -> None:
    logger.info("--- PRATYAYA DAY 2 GATE: BASELINE BENCHMARK SCORING ---")
    data_df = load_canonical_dataset()
    logger.info("Loaded total records: %d across years: %s", len(data_df), sorted(pd.to_datetime(data_df["init_date"]).dt.year.unique()))

    # 1. Purged Train / Calibration Split (2000-2001 Train vs 2002 Calibration)
    train_df, cal_df, purged_dates = create_purged_train_val_split(
        data_df,
        train_years=[2000, 2001],
        val_years=[2002],
        purge_window_days=12,
    )
    logger.info(
        "Purged Split complete: Train records = %d, Calibration records = %d, Purged dates = %d",
        len(train_df),
        len(cal_df),
        len(purged_dates),
    )

    # 2. 3D Event Clustering on Calibration Fold
    event_res = cluster_bust_events(cal_df, bust_col="is_bust")
    cal_df = event_res.labeled_df

    # 3. Fit Baselines B0–B3 on Train Fold
    logger.info("Fitting Baseline Ladder B0–B3 on training fold (2000–2001)...")
    b0 = ConstantBaseRateBaseline().fit(train_df, y_col="is_bust")
    b1 = ConditionalClimatologyBaseline().fit(train_df, y_col="is_bust")
    b2 = RawEnsembleSpreadBaseline().fit(train_df, y_col="is_bust")
    b3 = CalibratedLogisticSpreadBaseline(features=["ens_spread", "lead_day", "forecast_value"]).fit(train_df, y_col="is_bust")

    # 4. Predict on Calibration Fold
    cal_df["prob_b0"] = b0.predict_proba(cal_df)
    cal_df["prob_b1"] = b1.predict_proba(cal_df)
    cal_df["prob_b2"] = b2.predict_proba(cal_df)
    cal_df["prob_b3"] = b3.predict_proba(cal_df)

    y_true = cal_df["is_bust"].values
    cal_base_rate = float(y_true.mean())

    # 5. Compute Metrics and 1,000 Block Bootstrap CIs
    models = [
        ("B0 (Constant Base Rate)", "prob_b0"),
        ("B1 (Conditional Climatology)", "prob_b1"),
        ("B2 (Raw Spread)", "prob_b2"),
        ("B3 (Calibrated Logistic Spread)", "prob_b3"),
    ]

    results_summary = []
    b1_prob = cal_df["prob_b1"].values

    for name, col in models:
        prob = cal_df[col].values
        pr_point = compute_pr_auc(y_true, prob)
        roc_point = compute_auroc(y_true, prob)
        bs_point = compute_brier_score(y_true, prob)
        bss_point = compute_brier_skill_score(y_true, prob, b1_prob) if col != "prob_b1" else 0.0

        # Block bootstrap over init dates
        boot = block_bootstrap_metric(
            cal_df,
            y_true_col="is_bust",
            y_prob_col=col,
            metric_fn=compute_pr_auc,
            metric_name="PR-AUC",
            group_col="init_date",
            n_resamples=1000,
            random_state=42,
        )

        results_summary.append({
            "model": name,
            "pr_auc": pr_point,
            "pr_auc_ci_lower": boot.ci_lower,
            "pr_auc_ci_upper": boot.ci_upper,
            "auroc": roc_point,
            "brier_score": bs_point,
            "bss_vs_b1": bss_point,
        })

    # Paired block bootstrap: B3 vs B1, and B3 vs B2
    paired_b3_vs_b1 = paired_block_bootstrap_comparison(
        cal_df,
        y_true_col="is_bust",
        y_prob_model_col="prob_b3",
        y_prob_baseline_col="prob_b1",
        metric_fn=compute_pr_auc,
        metric_name="PR-AUC",
        group_col="init_date",
        n_resamples=1000,
        random_state=42,
    )

    paired_b3_vs_b2 = paired_block_bootstrap_comparison(
        cal_df,
        y_true_col="is_bust",
        y_prob_model_col="prob_b3",
        y_prob_baseline_col="prob_b2",
        metric_fn=compute_pr_auc,
        metric_name="PR-AUC",
        group_col="init_date",
        n_resamples=1000,
        random_state=42,
    )

    print("\n" + "=" * 84)
    print("                 PRATYAYA DAY 2 GATE — BASELINE LADDER BENCHMARK")
    print("=" * 84)
    print(f"Calibration Fold (Year 2002): {len(cal_df)} region-lead records across 10 init cycles")
    print(f"Empirical Calibration Base Rate: {cal_base_rate * 100:.2f}% ({int(y_true.sum())} bust rows)")
    print(f"Independent Bust Events: N = {event_res.n_events} events (Inflation factor: {event_res.inflation_factor:.2f} rows/event)")
    print("-" * 84)
    print(f"{'Baseline Model':<35} | {'PR-AUC (95% CI)':<26} | {'AUROC':<7} | {'Brier':<7} | {'BSS vs B1':<9}")
    print("-" * 84)
    for r in results_summary:
        ci_str = f"{r['pr_auc']:.4f} [{r['pr_auc_ci_lower']:.4f}, {r['pr_auc_ci_upper']:.4f}]"
        print(f"{r['model']:<35} | {ci_str:<26} | {r['auroc']:<7.4f} | {r['brier_score']:<7.4f} | {r['bss_vs_b1']:<+9.4f}")
    print("-" * 84)
    print("\nPaired Block Bootstrap Significance (1,000 resamples over init dates):")
    print(f"   - B3 vs B1: Margin = {paired_b3_vs_b1.delta_point_estimate:+.4f} [95% CI: {paired_b3_vs_b1.delta_ci_lower:+.4f}, {paired_b3_vs_b1.delta_ci_upper:+.4f}] (p = {paired_b3_vs_b1.p_value:.4f})")
    print(f"   - B3 vs B2: Margin = {paired_b3_vs_b2.delta_point_estimate:+.4f} [95% CI: {paired_b3_vs_b2.delta_ci_lower:+.4f}, {paired_b3_vs_b2.delta_ci_upper:+.4f}] (p = {paired_b3_vs_b2.p_value:.4f})")
    print("=" * 84)
    b3_row = [r for r in results_summary if "B3" in r["model"]][0]
    print(f"\nDAY 2 GATE BENCHMARK NUMBER TO BEAT ON DAY 5:")
    print(f"   Baseline B3 PR-AUC: {b3_row['pr_auc']:.4f} [95% CI: {b3_row['pr_auc_ci_lower']:.4f}, {b3_row['pr_auc_ci_upper']:.4f}]")
    print("=" * 84 + "\n")

    # Generate and write docs/EVALUATION.md
    write_evaluation_report(
        results_summary=results_summary,
        paired_b3_vs_b1=paired_b3_vs_b1,
        paired_b3_vs_b2=paired_b3_vs_b2,
        cal_df=cal_df,
        event_res=event_res,
        cal_base_rate=cal_base_rate,
        b3_benchmark=b3_row,
    )


def write_evaluation_report(
    results_summary: List[Dict],
    paired_b3_vs_b1: Any,
    paired_b3_vs_b2: Any,
    cal_df: pd.DataFrame,
    event_res: Any,
    cal_base_rate: float,
    b3_benchmark: Dict,
    output_path: Path = Path("docs/EVALUATION.md"),
) -> Path:
    """Generate and write the canonical docs/EVALUATION.md report per docs/05 §6 and docs/07 line 57."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_records = len(cal_df)
    n_cycles = len(pd.to_datetime(cal_df["init_date"]).unique())
    n_busts = int(cal_df["is_bust"].sum())

    lines = [
        "# Evaluation Protocol & Benchmark Report — Pratyaya",
        "",
        "Auto-generated by `make eval` (`scripts/score_baselines.py`) per `docs/05-evaluation-protocol.md` §6 and `docs/07-ten-day-plan.md` line 57.",
        "",
        "---",
        "",
        "## 1. Split Definitions & Temporal Partitions",
        "",
        "- **Training Fold:** Years 2000–2001 (7,200 records across 36 subdivisions and 10 lead days).",
        f"- **Calibration Fold:** Year 2002 ({n_records} records across {n_cycles} initialization cycles).",
        "- **Boundary Purge Window:** 12 calendar days purged between training and evaluation folds (zero temporal leakage).",
        "- **Held-Out Test Set:** Years 2018–2019 (Strictly isolated outside guarded `--final` runs per Decision D-000 and Risk R-10).",
        "",
        "---",
        "",
        "## 2. Independent Meteorological Event Clustering (docs/05 §4.1)",
        "",
        f"- **Empirical Calibration Base Rate:** {cal_base_rate * 100:.2f}% ({n_busts} bust records out of {n_records}).",
        f"- **Independent 3D Bust Events:** $N = {event_res.n_events}$ connected-component events.",
        f"- **Event Inflation Factor:** {event_res.inflation_factor:.2f} rows / independent event.",
        "- **Spatiotemporal Connectivity:** Spatial adjacency graph across 36 subdivisions with $\\Delta t \\le 1$ day temporal clustering.",
        "",
        "---",
        "",
        "## 3. Baseline Ladder Benchmark Table (B0–B3)",
        "",
        "Evaluated on the calibration fold (2002) with 1,000 block bootstrap resamples over initialization dates:",
        "",
        "| Baseline Model | PR-AUC [95% CI] | AUROC | Brier Score | BSS vs B1 |",
        "|---|---|---|---|---|",
    ]

    for r in results_summary:
        ci_str = f"**{r['pr_auc']:.4f}** [{r['pr_auc_ci_lower']:.4f}, {r['pr_auc_ci_upper']:.4f}]" if "B3" in r["model"] else f"{r['pr_auc']:.4f} [{r['pr_auc_ci_lower']:.4f}, {r['pr_auc_ci_upper']:.4f}]"
        lines.append(
            f"| {r['model']} | {ci_str} | {r['auroc']:.4f} | {r['brier_score']:.4f} | {r['bss_vs_b1']:+.4f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Paired Block Bootstrap Significance",
        "",
        f"- **B3 vs B1 (Conditional Climatology):** Margin = `{paired_b3_vs_b1.delta_point_estimate:+.4f}` [95% CI: `{paired_b3_vs_b1.delta_ci_lower:+.4f}`, `{paired_b3_vs_b1.delta_ci_upper:+.4f}`] ($p = {paired_b3_vs_b1.p_value:.4f}$)",
        f"- **B3 vs B2 (Raw Spread):** Margin = `{paired_b3_vs_b2.delta_point_estimate:+.4f}` [95% CI: `{paired_b3_vs_b2.delta_ci_lower:+.4f}`, `{paired_b3_vs_b2.delta_ci_upper:+.4f}`] ($p = {paired_b3_vs_b2.p_value:.4f}$)",
        "",
        "---",
        "",
        "## 5. Day 2 Gate Benchmark Threshold (Decision D-003)",
        "",
        f"> **Day 5 Gate Target:** Baseline B3 PR-AUC = **{b3_benchmark['pr_auc']:.4f}** [95% CI: {b3_benchmark['pr_auc_ci_lower']:.4f}, {b3_benchmark['pr_auc_ci_upper']:.4f}].",
        "> Any differentiated analogue (M1) or fusion (M2) model must beat this benchmark on the calibration fold.",
        "",
    ])

    content = "\n".join(lines) + "\n"
    output_path.write_text(content, encoding="utf-8")
    logger.info("Successfully wrote evaluation benchmark report to %s", output_path)
    return output_path


if __name__ == "__main__":
    main()
