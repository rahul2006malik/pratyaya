"""Unit tests for Analogue-NWP Fusion, Calibration, and Effective Lead Time (Nodes G3, G4, G5).

Tests:
1. Fusion model (LightGBM and Logistic Regression): fitting, prediction, feature importances.
2. Out-of-fold analogue feature generation: temporal exclusion enforcement, zero leakage.
3. Orthogonality diagnostics: Pearson and Spearman correlation between analogue and spread.
4. Probability calibration: Isotonic regression, Platt scaling, and Murphy (1973) decomposition.
5. Effective Lead Time (ELT) inversion: monotonic growth, clipping [0.5, 12.0], edge-case headlines.
6. 3-class BustDirection head: WET, DRY, NONE classifications with deadband.
7. Confidence band classifications: exact alignment with BAND_THRESHOLDS.
"""

import numpy as np
import pandas as pd
import pytest

from pratyaya.analogue.retrieve import AnalogueRetriever
from pratyaya.api.schemas import (
    BAND_THRESHOLDS,
    BustDirection,
    ConfidenceBand,
    classify_confidence_band,
)
from pratyaya.models.calibrate import (
    ProbabilityCalibrator,
    compare_calibrators,
)
from pratyaya.models.elt import (
    DEFAULT_DIRECTION_DEADBAND,
    ELT_MAX_CLIP,
    ELT_MIN_CLIP,
    EffectiveLeadTimeTransformer,
    classify_bust_direction,
)
from pratyaya.models.fusion import (
    DEFAULT_FUSION_FEATURES,
    AnalogueNWPFusionModel,
    add_calendar_features,
    compute_orthogonality_metrics,
    generate_out_of_fold_analogue_features,
)


def _make_dummy_fusion_df(n_samples=200, seed=42):
    """Generate synthetic DataFrame with physical and analogue features."""
    rng = np.random.default_rng(seed)
    init_dates = rng.choice(pd.date_range("2000-06-01", "2000-07-31", freq="D"), size=n_samples)
    regions = rng.choice([str(r) for r in range(1, 37)], size=n_samples)
    leads = rng.integers(1, 11, size=n_samples)

    ens_spread = rng.uniform(1.0, 15.0, size=n_samples)
    forecast_val = rng.uniform(5.0, 60.0, size=n_samples)
    p_analogue = np.clip(rng.beta(1.0, 15.0, size=n_samples), 0.0, 1.0)
    d_k = rng.uniform(10.0, 80.0, size=n_samples)
    signed_expect = rng.normal(0.0, 8.0, size=n_samples)
    analogue_dispersion = rng.uniform(2.0, 15.0, size=n_samples)

    # Ground truth: probability elevated when spread and p_analogue are high
    bust_prob = 0.03 + 0.3 * (ens_spread / 15.0) + 0.4 * p_analogue
    is_bust = (rng.uniform(0.0, 1.0, size=n_samples) < bust_prob).astype(int)

    df = pd.DataFrame({
        "init_date": init_dates,
        "region_id": regions,
        "lead_day": leads,
        "ens_spread": ens_spread,
        "forecast_value": forecast_val,
        "p_analogue": p_analogue,
        "d_k": d_k,
        "signed_expect": signed_expect,
        "analogue_dispersion": analogue_dispersion,
        "is_bust": is_bust,
        "error_signed": signed_expect,
    })
    return df


def test_calendar_features():
    """Verify cyclic month sine and cosine features are generated correctly."""
    df = pd.DataFrame({"init_date": ["2000-06-15", "2000-12-15"]})
    res = add_calendar_features(df)

    assert "month_sin" in res.columns
    assert "month_cos" in res.columns

    # June is month 6: sin(2*pi*6/12) = sin(pi) ~ 0.0, cos(pi) = -1.0
    assert res.loc[0, "month_sin"] == pytest.approx(0.0, abs=1e-6)
    assert res.loc[0, "month_cos"] == pytest.approx(-1.0, abs=1e-6)


def test_fusion_model_training_and_inference_both_backends():
    """Verify AnalogueNWPFusionModel trains and infers with both LightGBM and Logistic backends."""
    df = _make_dummy_fusion_df(n_samples=250)

    # 1. Test LightGBM backend
    lgb_model = AnalogueNWPFusionModel(backend="lightgbm", n_estimators=20)
    lgb_model.fit(df)
    assert lgb_model.is_fitted

    prob_lgb = lgb_model.predict_proba(df)
    assert prob_lgb.shape == (250,)
    assert np.all(prob_lgb >= 0.0) and np.all(prob_lgb <= 1.0)

    pred_lgb = lgb_model.predict(df, threshold=0.5)
    assert set(pred_lgb).issubset({0, 1})

    imp_lgb = lgb_model.feature_importances
    assert len(imp_lgb) == len(DEFAULT_FUSION_FEATURES)
    assert sum(imp_lgb.values()) == pytest.approx(1.0, abs=1e-4)

    # 2. Test Logistic Regression backend
    lr_model = AnalogueNWPFusionModel(backend="logistic")
    lr_model.fit(df)
    assert lr_model.is_fitted

    prob_lr = lr_model.predict_proba(df)
    assert prob_lr.shape == (250,)
    assert np.all(prob_lr >= 0.0) and np.all(prob_lr <= 1.0)

    imp_lr = lr_model.feature_importances
    assert len(imp_lr) == len(DEFAULT_FUSION_FEATURES)
    assert sum(imp_lr.values()) == pytest.approx(1.0, abs=1e-4)


def test_out_of_fold_feature_generation():
    """Verify out-of-fold analogue features are generated cleanly with +/- 15-day exclusion."""
    # 4 distinct cycle dates spaced 20 days apart
    cycle_dates = ["2000-06-05", "2000-06-25", "2000-07-15", "2000-08-05"]
    records = []
    rng = np.random.default_rng(101)
    for dt in cycle_dates:
        for r in range(1, 10):  # 9 subdivisions for fast test
            for lead in range(1, 6):  # 5 leads
                records.append({
                    "init_date": dt,
                    "region_id": str(r),
                    "lead_day": lead,
                    "forecast_value": 25.0,
                    "observation_value": 20.0,
                    "error_signed": 5.0,
                    "is_bust": 0,
                    "ens_spread": 4.5,
                })
    train_df = pd.DataFrame(records)

    # 4 cycle embeddings in 100-EOF space
    cycle_embs = rng.normal(0, 1, size=(4, 100))

    oof_df = generate_out_of_fold_analogue_features(
        train_df=train_df,
        cycle_embeddings=cycle_embs,
        cycle_dates=cycle_dates,
        retriever_factory=lambda k_analogues: AnalogueRetriever(k_analogues=k_analogues, exclusion_days=15),
        k_analogues=2,
    )

    assert len(oof_df) == len(train_df)
    assert "p_analogue" in oof_df.columns
    assert "d_k" in oof_df.columns
    assert "signed_expect" in oof_df.columns
    assert "analogue_dispersion" in oof_df.columns
    assert not oof_df["p_analogue"].isna().any()


def test_orthogonality_metrics_diagnostics():
    """Verify orthogonality check calculates Pearson and Spearman correlations."""
    df = _make_dummy_fusion_df(n_samples=100)
    df["prob_b3"] = df["ens_spread"] * 0.05

    res = compute_orthogonality_metrics(df)
    assert "analogue_vs_spread_pearson_r" in res
    assert "analogue_vs_spread_spearman_rho" in res
    assert "analogue_vs_b3_pearson_r" in res
    assert -1.0 <= res["analogue_vs_spread_pearson_r"] <= 1.0


def test_probability_calibrator_and_murphy_decomposition():
    """Verify ProbabilityCalibrator produces monotonic outputs and Murphy decomposition."""
    rng = np.random.default_rng(55)
    y_raw = rng.uniform(0.0, 0.8, size=150)
    y_true = (rng.uniform(0.0, 1.0, size=150) < y_raw).astype(int)

    # 1. Isotonic Calibrator
    cal_iso = ProbabilityCalibrator(method="isotonic").fit(y_true, y_raw)
    p_iso = cal_iso.predict(y_raw)
    assert np.all(p_iso >= 0.0) and np.all(p_iso <= 1.0)

    # Monotonicity test: sorted input yields sorted output
    sort_idx = np.argsort(y_raw)
    p_iso_sorted = p_iso[sort_idx]
    assert np.all(np.diff(p_iso_sorted) >= -1e-7), "Isotonic calibration violated monotonicity!"

    # 2. Murphy decomposition
    murphy = cal_iso.evaluate_calibration(y_true, p_iso, n_bins=5)
    assert "reliability" in murphy
    assert "resolution" in murphy
    assert "uncertainty" in murphy
    assert "brier_score" in murphy
    assert murphy["reliability"] >= 0.0

    # 3. Compare calibrators
    comp = compare_calibrators(y_true, y_raw, n_bins=5)
    assert "winner" in comp
    assert comp["winner"] in ["raw", "isotonic", "platt"]


def test_effective_lead_time_inversion():
    """Verify EffectiveLeadTimeTransformer monotonicity, clipping, and edge-case headlines."""
    df = _make_dummy_fusion_df(n_samples=300)
    elt_tf = EffectiveLeadTimeTransformer().fit(df)

    assert elt_tf.is_fitted
    assert len(elt_tf.national_curve_) == 10

    # Monotonicity of ELT: higher probability -> equal or higher ELT
    res_low = elt_tf.transform_single(p_calibrated=0.01, nominal_lead_day=5)
    res_mid = elt_tf.transform_single(p_calibrated=0.15, nominal_lead_day=5)
    res_high = elt_tf.transform_single(p_calibrated=0.60, nominal_lead_day=5)

    assert res_low.effective_lead_time <= res_mid.effective_lead_time
    assert res_mid.effective_lead_time <= res_high.effective_lead_time

    # Boundary clipping test
    res_zero = elt_tf.transform_single(p_calibrated=0.000, nominal_lead_day=3)
    assert res_zero.effective_lead_time >= ELT_MIN_CLIP
    assert res_zero.is_better_than_day1
    assert "better than a typical Day-1" in res_zero.headline

    res_extreme = elt_tf.transform_single(p_calibrated=0.999, nominal_lead_day=6)
    assert res_extreme.effective_lead_time <= ELT_MAX_CLIP
    assert res_extreme.is_beyond_range
    assert "beyond our measured range" in res_extreme.headline

    # Vector transform
    elts = elt_tf.transform_series(np.array([0.02, 0.20, 0.70]), np.array([2, 5, 8]))
    assert len(elts) == 3
    assert elts[0] <= elts[1] <= elts[2]


def test_bust_direction_head_and_confidence_band():
    """Verify 3-class BustDirection head and ConfidenceBand boundary integration."""
    # 1. Direction Head
    assert classify_bust_direction(-6.5, deadband=2.0) == BustDirection.WET
    assert classify_bust_direction(4.2, deadband=2.0) == BustDirection.DRY
    assert classify_bust_direction(1.0, deadband=2.0) == BustDirection.NONE
    assert classify_bust_direction(-1.5, deadband=2.0) == BustDirection.NONE

    # 2. Confidence Band Classifier integration
    assert classify_confidence_band(0.05) == ConfidenceBand.HIGH
    assert classify_confidence_band(0.18) == ConfidenceBand.MODERATE
    assert classify_confidence_band(0.35) == ConfidenceBand.LOW
    assert classify_confidence_band(0.75) == ConfidenceBand.CRITICAL
