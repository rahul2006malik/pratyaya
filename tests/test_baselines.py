"""Unit tests for Baseline Models Ladder B0–B5 (Node F6, docs/05 §2)."""

import numpy as np
import pandas as pd
import pytest

from pratyaya.models.baselines import (
    CalibratedLogisticSpreadBaseline,
    ConditionalClimatologyBaseline,
    ConstantBaseRateBaseline,
    EMOSBaseline,
    RawEnsembleSpreadBaseline,
    TabularLightGBMBaseline,
)


@pytest.fixture
def synthetic_train_df() -> pd.DataFrame:
    """Create a synthetic training dataset with known characteristics."""
    records = []
    # 5 regions, 10 leads, June/July
    for r in range(1, 6):
        for l in range(1, 11):
            for m in [6, 7]:
                # Signal: higher spread and higher lead -> more busts
                spread = float(r * 0.5 + l * 0.8)
                is_bust = int(spread > 6.0)
                records.append({
                    "init_date": pd.Timestamp(f"2000-{m:02d}-01"),
                    "region_id": str(r),
                    "lead_day": l,
                    "variable": "rainfall",
                    "ens_spread": spread,
                    "forecast_value": float(r * 5.0),
                    "is_bust": is_bust,
                })
    return pd.DataFrame(records)


def test_b0_constant_base_rate(synthetic_train_df: pd.DataFrame) -> None:
    """Verify B0 predicts constant training base rate."""
    model = ConstantBaseRateBaseline()
    model.fit(synthetic_train_df, y_col="is_bust")

    expected_rate = float(synthetic_train_df["is_bust"].mean())
    assert abs(model.fitted_rate_ - expected_rate) < 1e-6

    test_df = synthetic_train_df.iloc[:10].copy()
    probas = model.predict_proba(test_df)
    assert len(probas) == 10
    assert np.allclose(probas, expected_rate)


def test_b1_conditional_climatology(synthetic_train_df: pd.DataFrame) -> None:
    """Verify B1 stratifies by (region, lead, month) and handles fallback."""
    model = ConditionalClimatologyBaseline()
    model.fit(synthetic_train_df, y_col="is_bust")

    # Predict on exact training cell
    sample_row = pd.DataFrame([{
        "init_date": pd.Timestamp("2001-06-15"),  # month 6
        "region_id": "1",
        "lead_day": 1,
    }])
    p_seen = model.predict_proba(sample_row)
    assert 0.0 <= p_seen[0] <= 1.0

    # Predict on unseen region -> should fall back to global rate
    unseen_row = pd.DataFrame([{
        "init_date": pd.Timestamp("2001-12-15"),  # month 12
        "region_id": "99",
        "lead_day": 10,
    }])
    p_unseen = model.predict_proba(unseen_row)
    assert abs(p_unseen[0] - model.global_rate_) < 1e-6


def test_b2_raw_ensemble_spread(synthetic_train_df: pd.DataFrame) -> None:
    """Verify B2 normalizes spread to [0, 1] preserving rank ordering."""
    model = RawEnsembleSpreadBaseline(spread_col="ens_spread")
    model.fit(synthetic_train_df)

    test_df = pd.DataFrame({
        "ens_spread": [model.min_spread_, (model.min_spread_ + model.max_spread_) / 2.0, model.max_spread_]
    })
    probas = model.predict_proba(test_df)
    assert np.allclose(probas, [0.0, 0.5, 1.0])


def test_b3_calibrated_logistic_spread(synthetic_train_df: pd.DataFrame) -> None:
    """Verify B3 fits logistic regression on spread and produces calibrated probabilities."""
    model = CalibratedLogisticSpreadBaseline(features=["ens_spread"])
    model.fit(synthetic_train_df, y_col="is_bust")

    assert model.model_ is not None

    # Positive spread coefficient: probability must increase monotonically with spread
    test_df = pd.DataFrame({
        "ens_spread": [1.0, 5.0, 10.0]
    })
    probas = model.predict_proba(test_df)
    assert len(probas) == 3
    assert probas[0] < probas[1] < probas[2]
    assert np.all(probas >= 0.0) and np.all(probas <= 1.0)


def test_b4_b5_scaffolds(synthetic_train_df: pd.DataFrame) -> None:
    """Verify B4 (EMOS) and B5 (LightGBM) models instantiate, fit, and predict real probabilities."""
    b4 = EMOSBaseline()
    b4.fit(synthetic_train_df)
    p4 = b4.predict_proba(synthetic_train_df.iloc[:5])
    assert len(p4) == 5
    assert np.all(p4 >= 0.0) and np.all(p4 <= 1.0)
    assert not any(abs(x - 0.043) < 0.001 for x in p4), "B4 returned constant scaffold"

    b5 = TabularLightGBMBaseline()
    b5.fit(synthetic_train_df)
    p5 = b5.predict_proba(synthetic_train_df.iloc[:5])
    assert len(p5) == 5
    assert np.all(p5 >= 0.0) and np.all(p5 <= 1.0)
    assert not any(abs(x - 0.043) < 0.001 for x in p5), "B5 returned constant scaffold"
