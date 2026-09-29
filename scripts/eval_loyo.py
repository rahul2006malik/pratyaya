"""3-Fold Purged Leave-One-Year-Out Cross-Validation (LOYO-CV) Engine.

Pratyaya Atmospheric Research Framework.

Evaluates the dense 3-year empirical benchmark (2000–2002 JJAS, 120 cycles,
43,200 canonical rows) across 3 folds:
  - Fold 1: Train 2000 + 2001 (80 cycles), Test 2002 (40 cycles)
  - Fold 2: Train 2000 + 2002 (80 cycles), Test 2001 (40 cycles)
  - Fold 3: Train 2001 + 2002 (80 cycles), Test 2000 (40 cycles)

Strict boundary purges and >=15-day analogue temporal exclusions enforced.
Zero synthetic rows permitted.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from pratyaya.analogue.eof import EOFReducer
from pratyaya.analogue.metric import (
    LowRankMahalanobisMetric,
    build_outcome_vectors,
    sample_triplets,
)
from pratyaya.analogue.retrieve import AnalogueRetriever, TEMPORAL_EXCLUSION_DAYS
from pratyaya.analogue.state import (
    WideDomainClimatology,
    build_synoptic_state_vector,
    get_wide_domain_coords,
)
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
from pratyaya.evaluate.circulation import compute_multilead_trajectory_metrics
from pratyaya.evaluate.splits import create_purged_train_val_split
from pratyaya.models.baselines import (
    CalibratedLogisticSpreadBaseline,
    ConditionalClimatologyBaseline,
    ConstantBaseRateBaseline,
    EMOSBaseline,
    RawEnsembleSpreadBaseline,
    TabularLightGBMBaseline,
    ensure_features,
)
from pratyaya.models.calibrate import ProbabilityCalibrator
from pratyaya.models.fusion import (
    AnalogueNWPFusionModel,
    generate_out_of_fold_analogue_features,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("eval_loyo")

DEFAULT_LABELS_DIR = Path("data/canonical/labels")
DEFAULT_FCSTS_DIR = Path("data/canonical/forecasts")
DEFAULT_SYNOPTIC_CACHE = Path("data/interim/synoptic_states_2000_2002.npz")


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


def murphy_decomposition(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """Compute Murphy (1973) Brier score resolution and reliability decomposition."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    base_rate = float(np.mean(y_true))
    uncertainty = base_rate * (1.0 - base_rate)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    n_samples = len(y_true)
    reliability = 0.0
    resolution = 0.0

    for b in range(n_bins):
        mask = bin_indices == b
        n_k = np.sum(mask)
        if n_k > 0:
            p_bar_k = float(np.mean(y_prob[mask]))
            o_bar_k = float(np.mean(y_true[mask]))
            reliability += (n_k / n_samples) * ((p_bar_k - o_bar_k) ** 2)
            resolution += (n_k / n_samples) * ((o_bar_k - base_rate) ** 2)

    brier = reliability - resolution + uncertainty
    return {
        "reliability": float(reliability),
        "resolution": float(resolution),
        "uncertainty": float(uncertainty),
        "brier_reconstructed": float(brier),
    }


def run_single_fold(
    fold_idx: int,
    train_years: List[int],
    test_year: int,
    data_df: pd.DataFrame,
    synoptic_cache: Dict[str, Any],
    extra_library_embeddings: Optional[np.ndarray] = None,
    extra_library_dates: Optional[Sequence] = None,
) -> pd.DataFrame:
    """Train models on train_years and predict out-of-fold on test_year."""
    logger.info("=" * 70)
    logger.info("STARTING FOLD %d: Train %s -> Test %d", fold_idx, train_years, test_year)
    logger.info("=" * 70)

    train_df, test_df, purged_dates = create_purged_train_val_split(
        data_df,
        train_years=train_years,
        val_years=[test_year],
        purge_window_days=12,
    )
    logger.info(
        "Fold %d Split: Train records = %d, Test records = %d, Purged dates = %d",
        fold_idx, len(train_df), len(test_df), len(purged_dates),
    )

    # 1. Fit Baselines B0, B1, B2, B3, B4, B5
    logger.info("Fitting Baselines B0–B5...")
    train_df = ensure_features(train_df)
    test_df = ensure_features(test_df)

    b0 = ConstantBaseRateBaseline().fit(train_df, y_col="is_bust")
    b1 = ConditionalClimatologyBaseline().fit(train_df, y_col="is_bust")
    b2 = RawEnsembleSpreadBaseline().fit(train_df, y_col="is_bust")
    b3 = CalibratedLogisticSpreadBaseline(
        features=["ens_spread", "lead_day", "forecast_value"]
    ).fit(train_df, y_col="is_bust")
    b4 = EMOSBaseline().fit(train_df, y_col="is_bust")
    b5 = TabularLightGBMBaseline().fit(train_df, y_col="is_bust")

    test_df["prob_b0"] = b0.predict_proba(test_df)
    test_df["prob_b1"] = b1.predict_proba(test_df)
    test_df["prob_b2"] = b2.predict_proba(test_df)

    # Decision D-020: Equivalent Post-Hoc Isotonic Calibration across Benchmark Ladder
    cal_b3 = ProbabilityCalibrator(method="isotonic").fit(train_df["is_bust"].values, b3.predict_proba(train_df))
    test_df["prob_b3_raw"] = b3.predict_proba(test_df)
    test_df["prob_b3"] = cal_b3.predict(test_df["prob_b3_raw"].values)

    cal_b4 = ProbabilityCalibrator(method="isotonic").fit(train_df["is_bust"].values, b4.predict_proba(train_df))
    test_df["prob_b4_raw"] = b4.predict_proba(test_df)
    test_df["prob_b4"] = cal_b4.predict(test_df["prob_b4_raw"].values)

    cal_b5 = ProbabilityCalibrator(method="isotonic").fit(train_df["is_bust"].values, b5.predict_proba(train_df))
    test_df["prob_b5_raw"] = b5.predict_proba(test_df)
    test_df["prob_b5"] = cal_b5.predict(test_df["prob_b5_raw"].values)

    # 2. Extract synoptic states for training cycles and test cycles
    all_daily_dates = pd.to_datetime(synoptic_cache["training_daily_dates"])
    all_daily_states = synoptic_cache["training_daily_states"]

    # Filter daily states to train years
    train_mask = np.isin(all_daily_dates.year, train_years)
    train_daily_states = all_daily_states[train_mask]
    train_daily_dates = all_daily_dates[train_mask]

    clim = WideDomainClimatology()
    clim.fit(train_daily_states, train_daily_dates)
    X_train_daily_anom = clim.transform(train_daily_states)

    eof = EOFReducer(n_components=100)
    eof.fit(X_train_daily_anom, train_daily_dates)

    # Candidate cycle dates from training fold
    train_cycle_dates = sorted(pd.to_datetime(train_df["init_date"].unique()))
    test_cycle_dates = sorted(pd.to_datetime(test_df["init_date"].unique()))

    # Build or lookup cycle states
    # For dates present in daily cache, index directly
    date_to_idx = {pd.Timestamp(d).strftime("%Y-%m-%d"): i for i, d in enumerate(all_daily_dates)}

    Z_cand_list = []
    valid_train_cycles = []
    for dt in train_cycle_dates:
        ds = dt.strftime("%Y-%m-%d")
        if ds in date_to_idx:
            raw_s = all_daily_states[date_to_idx[ds]]
            anom_s = clim.transform(raw_s.reshape(1, -1))
            z_eof = eof.transform(anom_s)[0]
            Z_cand_list.append(z_eof)
            valid_train_cycles.append(dt)

    Z_cand_eof = np.array(Z_cand_list)

    Z_test_list = []
    valid_test_cycles = []
    for dt in test_cycle_dates:
        ds = dt.strftime("%Y-%m-%d")
        if ds in date_to_idx:
            raw_s = all_daily_states[date_to_idx[ds]]
            anom_s = clim.transform(raw_s.reshape(1, -1))
            z_eof = eof.transform(anom_s)[0]
            Z_test_list.append(z_eof)
            valid_test_cycles.append(dt)

    Z_test_eof = np.array(Z_test_list)

    # 3. Fit Low-Rank Mahalanobis Metric (Node G2)
    logger.info("Sampling triplets and training Low-Rank Mahalanobis metric...")
    outcome_vecs, _ = build_outcome_vectors(
        train_df,
        valid_train_cycles,
        n_leads=10,
        standardize=True,
    )
    triplets = sample_triplets(
        outcome_vecs,
        valid_train_cycles,
        n_triplets=min(2000, len(valid_train_cycles) * 25),
        min_separation_days=15,
        seed=42 + fold_idx,
    )

    metric = LowRankMahalanobisMetric(
        n_components=32,
        input_dim=100,
        margin=1.0,
        learning_rate=0.005,
        n_epochs=60,
        batch_size=64,
        l2_reg=1e-4,
        seed=42 + fold_idx,
    )
    metric.fit(Z_cand_eof, triplets)

    # 4. Model M1 v2 Retriever
    logger.info("Scoring Model M1 v2 on test fold...")
    retriever_v2 = AnalogueRetriever(
        k_analogues=8,
        exclusion_days=TEMPORAL_EXCLUSION_DAYS,
        metric=metric,
    )
    retriever_v2.fit(Z_cand_eof, valid_train_cycles, outcomes_df=train_df)

    pred_m1_list = []
    for i, q_date in enumerate(valid_test_cycles):
        q_emb = Z_test_eof[i]
        res = retriever_v2.retrieve(q_emb, q_date, k=8)
        p_df = retriever_v2.predict_bust_probabilities(q_emb, q_date, k=8)
        p_df["init_date_dt"] = q_date
        p_df["d_k"] = res.d_k
        pred_m1_list.append(p_df)

    pred_m1_df = pd.concat(pred_m1_list, ignore_index=True)

    # 4b. Model M1 Extended (Extra Library Expansion per Decision D-016)
    pred_m1_ext_df = None
    if extra_library_embeddings is not None and extra_library_dates is not None:
        logger.info("Scoring Model M1 Extended (extra library pool) on test fold...")
        retriever_ext = AnalogueRetriever(
            k_analogues=8,
            exclusion_days=TEMPORAL_EXCLUSION_DAYS,
            metric=metric,
        )
        retriever_ext.fit(
            Z_cand_eof,
            valid_train_cycles,
            outcomes_df=train_df,
            extra_library_embeddings=extra_library_embeddings,
            extra_library_dates=extra_library_dates,
        )
        pred_m1_ext_list = []
        for i, q_date in enumerate(valid_test_cycles):
            q_emb = Z_test_eof[i]
            _ = retriever_ext.retrieve(q_emb, q_date, k=8)
            p_ext_df = retriever_ext.predict_bust_probabilities(q_emb, q_date, k=8)
            p_ext_df["init_date_dt"] = q_date
            p_ext_df["unlabeled_neighbor_fraction"] = retriever_ext.last_query_unlabeled_fraction_
            pred_m1_ext_list.append(p_ext_df)
        pred_m1_ext_df = pd.concat(pred_m1_ext_list, ignore_index=True)
        pred_m1_ext_df["region_id_str"] = pred_m1_ext_df["region_id"].astype(str)

    # 5. Out-of-fold Analogue Features -> Fit Model M2 (Fusion)
    logger.info("Generating out-of-fold features for training fold and fitting M2...")
    train_fusion_df = generate_out_of_fold_analogue_features(
        train_df=train_df,
        cycle_embeddings=Z_cand_eof,
        cycle_dates=valid_train_cycles,
        retriever_factory=lambda k_analogues: AnalogueRetriever(
            k_analogues=k_analogues,
            exclusion_days=TEMPORAL_EXCLUSION_DAYS,
            metric=metric,
        ),
        k_analogues=8,
    )

    fusion_model = AnalogueNWPFusionModel(backend="lightgbm")
    fusion_model.fit(train_fusion_df, y_col="is_bust")

    # Fit Isotonic Calibrator strictly on training fold
    calibrator = ProbabilityCalibrator(method="isotonic")
    train_p_raw = fusion_model.predict_proba(train_fusion_df)
    calibrator.fit(y_true=train_fusion_df["is_bust"].values, y_prob=train_p_raw)

    # Merge M1 and test_df
    test_df["init_date_dt"] = pd.to_datetime(test_df["init_date"])
    test_df["region_id_str"] = test_df["region_id"].astype(str)
    pred_m1_df["region_id_str"] = pred_m1_df["region_id"].astype(str)

    test_merged = pd.merge(
        test_df,
        pred_m1_df[["init_date_dt", "region_id_str", "lead_day", "p_analogue", "d_k", "signed_expect", "analogue_dispersion"]],
        on=["init_date_dt", "region_id_str", "lead_day"],
        how="inner",
    )

    if pred_m1_ext_df is not None:
        pred_m1_ext_df = pred_m1_ext_df.rename(columns={"p_analogue": "p_analogue_ext"})
        test_merged = pd.merge(
            test_merged,
            pred_m1_ext_df[["init_date_dt", "region_id_str", "lead_day", "p_analogue_ext", "unlabeled_neighbor_fraction"]],
            on=["init_date_dt", "region_id_str", "lead_day"],
            how="left",
        )
        test_merged["prob_m1_ext"] = test_merged["p_analogue_ext"]
    else:
        test_merged["prob_m1_ext"] = np.nan
        test_merged["unlabeled_neighbor_fraction"] = 0.0

    # Predict M1 and M2 on test fold
    test_merged["prob_m1"] = test_merged["p_analogue"]
    test_merged["prob_m2_raw"] = fusion_model.predict_proba(test_merged)
    test_merged["prob_m2"] = calibrator.predict(test_merged["prob_m2_raw"].values)
    test_merged["fold"] = fold_idx

    logger.info("Fold %d Out-of-Fold scoring complete: %d records.", fold_idx, len(test_merged))
    return test_merged


def main() -> None:
    parser = argparse.ArgumentParser(description="3-Fold Purged LOYO-CV Evaluation Engine.")
    parser.add_argument("--n-bootstrap", type=int, default=1000, help="Number of block bootstrap iterations.")
    parser.add_argument("--output-json", type=Path, default=Path("models/loyo_benchmark_results.json"))
    parser.add_argument("--force-overwrite-canonical", action="store_true", help="Force overwrite canonical results even with low n-bootstrap.")
    args = parser.parse_args()

    # Guard against accidental smoke test overwriting canonical results
    canonical_path = Path("models/loyo_benchmark_results.json").resolve()
    if args.output_json.resolve() == canonical_path and args.n_bootstrap < 100 and not args.force_overwrite_canonical:
        raise ValueError(
            f"Refusing to overwrite canonical benchmark results at {args.output_json} "
            f"with only {args.n_bootstrap} bootstrap iterations. "
            f"Use --n-bootstrap >= 1000 for canonical benchmark, or pass --force-overwrite-canonical."
        )

    logger.info("=" * 80)
    logger.info("PROJECT PRATYAYA: 3-FOLD PURGED LOYO-CV EVALUATION (2000–2002 JJAS)")
    logger.info("=" * 80)

    data_df = load_canonical_dataset()
    logger.info(
        "Loaded canonical dataset: %d records, %d cycles across years: %s",
        len(data_df),
        data_df["init_date"].nunique(),
        sorted(pd.to_datetime(data_df["init_date"]).dt.year.unique()),
    )
    assert len(data_df) == 43200, f"Expected 43,200 rows, found {len(data_df)}"

    synoptic_cache = np.load(DEFAULT_SYNOPTIC_CACHE, allow_pickle=True)

    # Run the 3 Folds
    folds_config = [
        (1, [2000, 2001], 2002),
        (2, [2000, 2002], 2001),
        (3, [2001, 2002], 2000),
    ]

    extra_npz_path = Path("models/extra_library_embeddings.npz")
    extra_embs = None
    extra_dates = None
    if extra_npz_path.exists():
        logger.info("Found extra library embeddings at %s", extra_npz_path)
        extra_data = np.load(extra_npz_path, allow_pickle=True)
        extra_embs = extra_data["embeddings"]
        extra_dates = extra_data["dates"]

    fold_dfs = []
    for fold_idx, train_years, test_year in folds_config:
        fold_res = run_single_fold(
            fold_idx=fold_idx,
            train_years=train_years,
            test_year=test_year,
            data_df=data_df,
            synoptic_cache=synoptic_cache,
            extra_library_embeddings=extra_embs,
            extra_library_dates=extra_dates,
        )
        fold_dfs.append(fold_res)

    pooled_df = pd.concat(fold_dfs, ignore_index=True)
    logger.info("Pooled out-of-fold records: %d across %d cycles.", len(pooled_df), pooled_df["init_date_dt"].nunique())
    assert len(pooled_df) == 43200, f"Pooled dataset has {len(pooled_df)} != 43,200 rows"

    # Multi-lead circulation error trajectory diagnostics
    traj_df, traj_summary = compute_multilead_trajectory_metrics(pooled_df, leads=(3, 4, 5, 6, 7))

    # Cluster 3D meteorological events per docs/05 §4.1
    logger.info("Clustering 3D connected-component bust events across the 43,200 pooled rows...")
    event_res = cluster_bust_events(pooled_df, bust_col="is_bust")
    pooled_df = event_res.labeled_df
    n_events = event_res.n_events
    inflation = event_res.inflation_factor
    logger.info("Event clustering complete: %d independent events (sample inflation: %.2fx)", n_events, inflation)

    # Evaluate all models with 1,000 block bootstrap iterations
    logger.info("Running %d block bootstrap iterations over initialization dates...", args.n_bootstrap)
    models = [
        "prob_b0",
        "prob_b1",
        "prob_b2",
        "prob_b3",
        "prob_b4",
        "prob_b5",
        "prob_m1",
        "prob_m2_raw",
        "prob_m2",
    ]
    model_labels = {
        "prob_b0": "B0 (Constant Base Rate)",
        "prob_b1": "B1 (Conditional Climatology)",
        "prob_b2": "B2 (Raw Ensemble Spread)",
        "prob_b3": "B3 (Calibrated Logistic Spread)",
        "prob_b4": "B4 (EMOS Spread-Calibrated)",
        "prob_b5": "B5 (Tabular LightGBM NWP)",
        "prob_m1": "M1 v2 (Pure Analogue Memory)",
        "prob_m2_raw": "M2 Raw (Analogue-NWP Uncalibrated)",
        "prob_m2": "M2 (Analogue-NWP Calibrated)",
    }

    results: Dict[str, Any] = {
        "n_records": len(pooled_df),
        "n_cycles": int(pooled_df["init_date_dt"].nunique()),
        "n_events": int(n_events),
        "sample_inflation": float(inflation),
        "primary_base_rate": float(pooled_df["is_bust"].mean()),
        "n_bust_rows": int(pooled_df["is_bust"].sum()),
        "models": {},
        "circulation_diagnostics": {
            "n_trajectories": traj_summary.n_trajectories,
            "mean_error_growth_rate": traj_summary.mean_error_growth_rate,
            "monotonically_compounding_fraction": traj_summary.monotonically_compounding_fraction,
            "mean_spearman_rho": traj_summary.mean_spearman_rho,
            "z500_acc": "synoptic_cache_missing_required_fields",
        },
    }

    y_true = pooled_df["is_bust"].values

    for m_col in models:
        m_prob = pooled_df[m_col].values
        # PR-AUC point estimate + bootstrap CI
        pr_res = block_bootstrap_metric(
            pooled_df,
            y_true_col="is_bust",
            y_prob_col=m_col,
            group_col="init_date_dt",
            metric_fn=compute_pr_auc,
            n_resamples=args.n_bootstrap,
            ci_level=0.95,
            random_state=42,
        )
        auroc = compute_auroc(y_true, m_prob)
        brier = compute_brier_score(y_true, m_prob)
        murphy = murphy_decomposition(y_true, m_prob)
        # Lead stratification: compute PR-AUC for lead days 1 to 10
        lead_pr_aucs = []
        for l in range(1, 11):
            l_mask = pooled_df["lead_day"] == l
            if l_mask.sum() > 0 and pooled_df.loc[l_mask, "is_bust"].sum() > 0:
                lead_pr_aucs.append(round(float(compute_pr_auc(pooled_df.loc[l_mask, "is_bust"].values, pooled_df.loc[l_mask, m_col].values)), 4))
            else:
                lead_pr_aucs.append(0.0)

        results["models"][m_col] = {
            "name": model_labels[m_col],
            "pr_auc": float(pr_res.point_estimate),
            "pr_auc_ci": [float(pr_res.ci_lower), float(pr_res.ci_upper)],
            "auroc": float(auroc),
            "brier": float(brier),
            "reliability": murphy["reliability"],
            "resolution": murphy["resolution"],
            "uncertainty": murphy["uncertainty"],
            "pr_auc_by_lead": lead_pr_aucs,
        }

    # Compute BSS vs B1 for all models
    b1_brier = results["models"]["prob_b1"]["brier"]
    for m_col in models:
        m_brier = results["models"][m_col]["brier"]
        bss = (b1_brier - m_brier) / b1_brier if b1_brier > 0 else 0.0
        results["models"][m_col]["bss_vs_b1"] = float(bss)

    # Paired block bootstrap: M2 vs B3
    m2_vs_b3 = paired_block_bootstrap_comparison(
        pooled_df,
        y_true_col="is_bust",
        y_prob_model_col="prob_m2",
        y_prob_baseline_col="prob_b3",
        group_col="init_date_dt",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        random_state=42,
    )
    p_val_m2_b3 = float(m2_vs_b3.p_value)
    p_val_m2_b3_clean = max(p_val_m2_b3, 1.0 / args.n_bootstrap) if p_val_m2_b3 == 0.0 else p_val_m2_b3
    results["margin_m2_vs_b3"] = {
        "diff_point": float(m2_vs_b3.delta_point_estimate),
        "ci_lower": float(m2_vs_b3.delta_ci_lower),
        "ci_upper": float(m2_vs_b3.delta_ci_upper),
        "p_value": p_val_m2_b3_clean,
        "p_value_note": "bootstrap n=1000; exact value unresolved below 0.001, stored as ceiling" if p_val_m2_b3 == 0.0 else None,
        "significant": bool(m2_vs_b3.ci_excludes_zero),
    }

    # Paired block bootstrap: M2 vs B1
    m2_vs_b1 = paired_block_bootstrap_comparison(
        pooled_df,
        y_true_col="is_bust",
        y_prob_model_col="prob_m2",
        y_prob_baseline_col="prob_b1",
        group_col="init_date_dt",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        random_state=42,
    )
    p_val_m2_b1 = float(m2_vs_b1.p_value)
    p_val_m2_b1_clean = max(p_val_m2_b1, 1.0 / args.n_bootstrap) if p_val_m2_b1 == 0.0 else p_val_m2_b1
    results["margin_m2_vs_b1"] = {
        "diff_point": float(m2_vs_b1.delta_point_estimate),
        "ci_lower": float(m2_vs_b1.delta_ci_lower),
        "ci_upper": float(m2_vs_b1.delta_ci_upper),
        "p_value": p_val_m2_b1_clean,
        "p_value_note": "bootstrap n=1000; exact value unresolved below 0.001, stored as ceiling" if p_val_m2_b1 == 0.0 else None,
        "significant": bool(m2_vs_b1.ci_excludes_zero),
    }

    # Checkpoint 1 Diagnostics (M1 Baseline vs Extended Library per Decision D-016)
    m1_pr_auc_baseline_only = float(results["models"]["prob_m1"]["pr_auc"])
    if "prob_m1_ext" in pooled_df.columns and not pooled_df["prob_m1_ext"].isna().all():
        m1_pr_auc_extended = float(compute_pr_auc(y_true, pooled_df["prob_m1_ext"].values))
        unlabeled_neighbor_fraction = float(pooled_df["unlabeled_neighbor_fraction"].mean())
    else:
        m1_pr_auc_extended = None
        unlabeled_neighbor_fraction = 0.0

    results["diagnostic"] = {
        "m1_pr_auc_baseline_only": float(m1_pr_auc_baseline_only),
        "m1_pr_auc_extended": float(m1_pr_auc_extended) if m1_pr_auc_extended is not None else None,
        "unlabeled_neighbor_fraction": float(unlabeled_neighbor_fraction),
    }

    # Task P4.2: Sensitivity Analysis (Decisions D-015 & D-017)
    logger.info("Computing 10-day sensitivity analysis alongside primary 15-day window (Decision D-017)...")
    event_res_10d = cluster_bust_events(pooled_df, bust_col="is_bust", temporal_window_days=10)
    n_events_10d = int(event_res_10d.n_events)
    inflation_factor_10d = float(event_res_10d.inflation_factor)

    m2_pr_auc_primary = float(results["models"]["prob_m2"]["pr_auc"])
    b3_pr_auc_primary = float(results["models"]["prob_b3"]["pr_auc"])

    # Genuine 10-day block bootstrap (grouping initialization dates into 10-day blocks)
    pooled_df["block_10d"] = pooled_df["init_date_dt"].apply(lambda d: f"{d.year}_{d.dayofyear // 10}")

    m2_boot_10d = block_bootstrap_metric(
        pooled_df,
        y_true_col="is_bust",
        y_prob_col="prob_m2",
        group_col="block_10d",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        ci_level=0.95,
        random_state=42,
    )
    b3_boot_10d = block_bootstrap_metric(
        pooled_df,
        y_true_col="is_bust",
        y_prob_col="prob_b3",
        group_col="block_10d",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        ci_level=0.95,
        random_state=42,
    )
    m2_vs_b3_10d = paired_block_bootstrap_comparison(
        pooled_df,
        y_true_col="is_bust",
        y_prob_model_col="prob_m2",
        y_prob_baseline_col="prob_b3",
        group_col="block_10d",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        random_state=42,
    )

    # Genuine Alternate Threshold: is_bust_p90 (Dr. Anumeha Checklist Item 5)
    logger.info("Computing alternate threshold analysis (is_bust_p90)...")
    y_true_p90 = pooled_df["is_bust_p90"].values
    n_bust_p90 = int(y_true_p90.sum())
    p90_base_rate = float(np.mean(y_true_p90))
    event_res_p90 = cluster_bust_events(pooled_df, bust_col="is_bust_p90", temporal_window_days=15)
    n_events_p90 = int(event_res_p90.n_events)
    inflation_p90 = float(event_res_p90.inflation_factor)

    m2_boot_p90 = block_bootstrap_metric(
        pooled_df,
        y_true_col="is_bust_p90",
        y_prob_col="prob_m2",
        group_col="init_date_dt",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        ci_level=0.95,
        random_state=42,
    )
    b3_boot_p90 = block_bootstrap_metric(
        pooled_df,
        y_true_col="is_bust_p90",
        y_prob_col="prob_b3",
        group_col="init_date_dt",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        ci_level=0.95,
        random_state=42,
    )
    m2_vs_b3_p90 = paired_block_bootstrap_comparison(
        pooled_df,
        y_true_col="is_bust_p90",
        y_prob_model_col="prob_m2",
        y_prob_baseline_col="prob_b3",
        group_col="init_date_dt",
        metric_fn=compute_pr_auc,
        n_resamples=args.n_bootstrap,
        random_state=42,
    )

    results["sensitivity_analysis"] = {
        "primary_window_days": 15,
        "primary_n_events": int(n_events),
        "primary_inflation_factor": float(inflation),
        "alternate_window_days": 10,
        "alternate_n_events": int(n_events_10d),
        "alternate_inflation_factor": float(inflation_factor_10d),
        "m2_pr_auc_primary": float(m2_pr_auc_primary),
        "m2_pr_auc_primary_ci": results["models"]["prob_m2"]["pr_auc_ci"],
        "m2_pr_auc_alternate": float(m2_boot_10d.point_estimate),
        "m2_pr_auc_alternate_ci": [float(m2_boot_10d.ci_lower), float(m2_boot_10d.ci_upper)],
        "b3_pr_auc_primary": float(b3_pr_auc_primary),
        "b3_pr_auc_primary_ci": results["models"]["prob_b3"]["pr_auc_ci"],
        "b3_pr_auc_alternate": float(b3_boot_10d.point_estimate),
        "b3_pr_auc_alternate_ci": [float(b3_boot_10d.ci_lower), float(b3_boot_10d.ci_upper)],
        "margin_10d": {
            "diff_point": float(m2_vs_b3_10d.delta_point_estimate),
            "ci_lower": float(m2_vs_b3_10d.delta_ci_lower),
            "ci_upper": float(m2_vs_b3_10d.delta_ci_upper),
            "p_value": float(m2_vs_b3_10d.p_value),
            "significant": bool(m2_vs_b3_10d.ci_excludes_zero),
        },
        "alternate_threshold_p90": {
            "n_bust_rows": n_bust_p90,
            "base_rate": p90_base_rate,
            "n_events": n_events_p90,
            "inflation_factor": inflation_p90,
            "m2_pr_auc": float(m2_boot_p90.point_estimate),
            "m2_pr_auc_ci": [float(m2_boot_p90.ci_lower), float(m2_boot_p90.ci_upper)],
            "m2_auroc": float(compute_auroc(y_true_p90, pooled_df["prob_m2"].values)),
            "m2_brier": float(compute_brier_score(y_true_p90, pooled_df["prob_m2"].values)),
            "b3_pr_auc": float(b3_boot_p90.point_estimate),
            "b3_pr_auc_ci": [float(b3_boot_p90.ci_lower), float(b3_boot_p90.ci_upper)],
            "b3_auroc": float(compute_auroc(y_true_p90, pooled_df["prob_b3"].values)),
            "b3_brier": float(compute_brier_score(y_true_p90, pooled_df["prob_b3"].values)),
            "margin_m2_vs_b3": {
                "diff_point": float(m2_vs_b3_p90.delta_point_estimate),
                "ci_lower": float(m2_vs_b3_p90.delta_ci_lower),
                "ci_upper": float(m2_vs_b3_p90.delta_ci_upper),
                "p_value": float(m2_vs_b3_p90.p_value),
                "significant": bool(m2_vs_b3_p90.ci_excludes_zero),
            },
        },
    }

    # Task P4.3: Cost-Loss Economic Value Curve V(alpha)
    logger.info("Computing Richardson/Murphy cost-loss economic value curve V(alpha)...")
    cost_loss_grid = [0.01, 0.05, 0.10, 0.20, 0.30, 0.50]
    dense_alpha_grid = np.array([0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50])
    v_dense_m2 = compute_cost_loss_value_curve(y_true, pooled_df["prob_m2"].values, alpha_grid=dense_alpha_grid)
    v_dense_b3 = compute_cost_loss_value_curve(y_true, pooled_df["prob_b3"].values, alpha_grid=dense_alpha_grid)

    v_grid_m2 = compute_cost_loss_value_curve(y_true, pooled_df["prob_m2"].values, alpha_grid=np.array(cost_loss_grid))
    v_grid_b3 = compute_cost_loss_value_curve(y_true, pooled_df["prob_b3"].values, alpha_grid=np.array(cost_loss_grid))

    results["economic_value"] = {
        "max_value_m2": float(v_dense_m2["max_value"]),
        "max_value_b3": float(v_dense_b3["max_value"]),
        "cost_loss_grid": cost_loss_grid,
        "v_alpha_m2": [float(v) for v in v_grid_m2["value"]],
        "v_alpha_b3": [float(v) for v in v_grid_b3["value"]],
        "M2": {
            "max_value": float(v_dense_m2["max_value"]),
            "values": [float(v) for v in v_grid_m2["value"]],
            "alphas": cost_loss_grid,
        },
        "B3": {
            "max_value": float(v_dense_b3["max_value"]),
            "values": [float(v) for v in v_grid_b3["value"]],
            "alphas": cost_loss_grid,
        },
    }

    n_bust_rows = int(pooled_df["is_bust"].sum())

    # Print Summary Table
    print("\n" + "=" * 96)
    print("       PROJECT PRATYAYA: 3-FOLD PURGED LOYO-CV BENCHMARK RESULTS (2000–2002 JJAS)       ")
    print("=" * 96)
    print("SECTION 1: NCMRWF 5-POINT REPORTING CHECKLIST (Dr. Anumeha Dube, Sep 23 2026)")
    print(f"  1. Total Precipitation Forecast Bust Rows: {n_bust_rows:,} rows (Base Rate: {results['primary_base_rate']*100:.2f}%)")
    print(f"  2. N Effectively Independent Bust Events:  {n_events} events (Sample Inflation: {inflation:.2f}x)")
    print(f"  3. Independence Definition:               3D spatiotemporal connected components with 15-day temporal exclusion")
    print(f"  4. 95% Confidence Intervals:              {args.n_bootstrap} block bootstrap iterations over initialization dates")
    print(f"  5. Sensitivity under Alternate Threshold:  10-day separation window yields N = {n_events_10d} events ({inflation_factor_10d:.2f}x inflation)\n")

    print("=" * 96)
    print("SECTION 2: 9-MODEL BENCHMARK LADDER PERFORMANCE")
    print("-" * 96)
    print(f"{'Model':<36} | {'PR-AUC [95% CI]':<26} | {'AUROC':<7} | {'Brier':<7} | {'Rel Err':<7} | {'BSS vs B1':<9}")
    print("-" * 96)
    for m_col in models:
        m_data = results["models"][m_col]
        ci_str = f"[{m_data['pr_auc_ci'][0]:.4f}, {m_data['pr_auc_ci'][1]:.4f}]"
        print(
            f"{m_data['name']:<36} | {m_data['pr_auc']:.4f} {ci_str:<20} | "
            f"{m_data['auroc']:.4f}  | {m_data['brier']:.4f}  | {m_data['reliability']:.4f}  | {m_data['bss_vs_b1']:+.4f}"
        )
    print("-" * 96)

    print("\nSECTION 3: PAIRED BLOCK BOOTSTRAP STATISTICAL SIGNIFICANCE")
    margin3 = results["margin_m2_vs_b3"]
    p_str3 = "p < 0.001" if margin3['p_value'] < 0.001 else f"p = {margin3['p_value']:.4f}"
    print(f"  Model M2 vs Baseline B3 (Physical Spread Benchmark):")
    print(f"    - Delta PR-AUC: {margin3['diff_point']:+.4f} [95% CI: {margin3['ci_lower']:+.4f}, {margin3['ci_upper']:+.4f}]")
    print(f"    - p-value: {p_str3} | Significant (CI excludes zero): {margin3['significant']}")
    margin1 = results["margin_m2_vs_b1"]
    p_str1 = "p < 0.001" if margin1['p_value'] < 0.001 else f"p = {margin1['p_value']:.4f}"
    print(f"  Model M2 vs Baseline B1 (Operational Climatology Benchmark):")
    print(f"    - Delta PR-AUC: {margin1['diff_point']:+.4f} [95% CI: {margin1['ci_lower']:+.4f}, {margin1['ci_upper']:+.4f}]")
    print(f"    - p-value: {p_str1} | Significant (CI excludes zero): {margin1['significant']}")

    print("\nSECTION 4: COST-LOSS ECONOMIC VALUE V(alpha) (Murphy 1977 / Richardson 2000)")
    ev = results["economic_value"]
    print(f"  - Peak Economic Value M2: {ev['max_value_m2']:.4f}")
    print(f"  - Peak Economic Value B3: {ev['max_value_b3']:.4f}")
    for a, vm, vb in zip(ev["cost_loss_grid"], ev["v_alpha_m2"], ev["v_alpha_b3"]):
        print(f"    alpha = {a:4.2f}: V(M2) = {vm:+.4f} | V(B3) = {vb:+.4f}")

    print("\nSECTION 5: SENSITIVITY ANALYSIS (Decisions D-015 & D-017)")
    sa = results["sensitivity_analysis"]
    print(f"  5.1 Alternate 10-Day Window (Block Resampling & Independence Consolidation):")
    print(f"    - Primary Window (15 days):   N = {sa['primary_n_events']} events ({sa['primary_inflation_factor']:.2f}x) | M2: {sa['m2_pr_auc_primary']:.4f} [{sa['m2_pr_auc_primary_ci'][0]:.4f}, {sa['m2_pr_auc_primary_ci'][1]:.4f}] | B3: {sa['b3_pr_auc_primary']:.4f} [{sa['b3_pr_auc_primary_ci'][0]:.4f}, {sa['b3_pr_auc_primary_ci'][1]:.4f}]")
    print(f"    - Alternate Window (10 days): N = {sa['alternate_n_events']} events ({sa['alternate_inflation_factor']:.2f}x) | M2: {sa['m2_pr_auc_alternate']:.4f} [{sa['m2_pr_auc_alternate_ci'][0]:.4f}, {sa['m2_pr_auc_alternate_ci'][1]:.4f}] | B3: {sa['b3_pr_auc_alternate']:.4f} [{sa['b3_pr_auc_alternate_ci'][0]:.4f}, {sa['b3_pr_auc_alternate_ci'][1]:.4f}]")
    m10 = sa["margin_10d"]
    p_str10 = "p < 0.001" if m10['p_value'] < 0.001 else f"p = {m10['p_value']:.4f}"
    print(f"    - 10-day Block Paired Margin (M2 - B3): {m10['diff_point']:+.4f} [{m10['ci_lower']:+.4f}, {m10['ci_upper']:+.4f}], p = {p_str10}, significant = {m10['significant']}")
    p90 = sa["alternate_threshold_p90"]
    print(f"  5.2 Alternate Bust Threshold (P90: is_bust_p90, Base Rate = {p90['base_rate']*100:.2f}%, N = {p90['n_events']} events):")
    print(f"    - Model M2: PR-AUC = {p90['m2_pr_auc']:.4f} [{p90['m2_pr_auc_ci'][0]:.4f}, {p90['m2_pr_auc_ci'][1]:.4f}] | AUROC = {p90['m2_auroc']:.4f} | Brier = {p90['m2_brier']:.4f}")
    print(f"    - Baseline B3: PR-AUC = {p90['b3_pr_auc']:.4f} [{p90['b3_pr_auc_ci'][0]:.4f}, {p90['b3_pr_auc_ci'][1]:.4f}] | AUROC = {p90['b3_auroc']:.4f} | Brier = {p90['b3_brier']:.4f}")
    m_p90 = p90["margin_m2_vs_b3"]
    p_str_p90 = "p < 0.001" if m_p90['p_value'] < 0.001 else f"p = {m_p90['p_value']:.4f}"
    print(f"    - P90 Paired Margin (M2 - B3): {m_p90['diff_point']:+.4f} [{m_p90['ci_lower']:+.4f}, {m_p90['ci_upper']:+.4f}], p = {p_str_p90}, significant = {m_p90['significant']}")

    print("\nSECTION 6: CHECKPOINT 1 DIAGNOSTICS (Analogue Library Expansion, Decision D-016)")
    print(f"  - M1 PR-AUC (Baseline 120-date pool): {results['diagnostic']['m1_pr_auc_baseline_only']:.4f}")
    if results['diagnostic']['m1_pr_auc_extended'] is not None:
        print(f"  - M1 PR-AUC (Extended 1,200+ pool):  {results['diagnostic']['m1_pr_auc_extended']:.4f}")
    else:
        print(f"  - M1 PR-AUC (Extended 1,200+ pool):  None")
    print(f"  - Unlabeled Neighbor Fraction:        {results['diagnostic']['unlabeled_neighbor_fraction']:.4f}")
    print("=" * 96 + "\n")

    # Save to JSON
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved benchmark results to %s", args.output_json)


if __name__ == "__main__":
    main()
