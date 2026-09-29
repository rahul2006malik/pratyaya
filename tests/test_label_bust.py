"""Comprehensive Test Suite for Bust Labeling Module (Node F4, Decision D-002).

Validates:
1. IMD operational rainfall category classification.
2. Primary decision-relevant label construction (magnitude + alert flip + significance).
3. Secondary percentile-of-own-error label (P90 + 5.0 mm floor).
4. Four mandatory sanity checks per docs/03 §4.3:
   - Check 1: Monotonic bust rate with lead day 1 to 10.
   - Check 2: Error magnitude floor (|e| >= 10.0 mm).
   - Check 3: Primary base rate ~3.5%–5.5%.
   - Check 4: Secondary base rate ~10% by construction.
5. Strict held-out protection: fitting thresholds on 2018-2019 raises PermissionError.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pratyaya.label.bust import (
    RAIN_FLOOR_MAG,
    RAIN_SIG_THRESHOLD,
    SECONDARY_FLOOR_RAIN,
    ThresholdModel,
    classify_imd_rainfall_category,
    compute_bust_labels,
    fit_bust_thresholds,
    verify_sanity_checks,
)


def test_classify_imd_rainfall_category():
    """Verify IMD operational 24h rainfall category boundaries."""
    assert classify_imd_rainfall_category(0.0) == "No Rain"
    assert classify_imd_rainfall_category(0.05) == "No Rain"
    assert classify_imd_rainfall_category(0.1) == "Very Light Rain"
    assert classify_imd_rainfall_category(2.4) == "Very Light Rain"
    assert classify_imd_rainfall_category(2.5) == "Light Rain"
    assert classify_imd_rainfall_category(15.5) == "Light Rain"
    assert classify_imd_rainfall_category(15.6) == "Moderate Rain"
    assert classify_imd_rainfall_category(64.4) == "Moderate Rain"
    assert classify_imd_rainfall_category(64.5) == "Heavy Rain"
    assert classify_imd_rainfall_category(115.5) == "Heavy Rain"
    assert classify_imd_rainfall_category(115.6) == "Very Heavy Rain"
    assert classify_imd_rainfall_category(204.4) == "Very Heavy Rain"
    assert classify_imd_rainfall_category(204.5) == "Extremely Heavy Rain"
    assert classify_imd_rainfall_category(350.0) == "Extremely Heavy Rain"


def test_heldout_fitting_protection():
    """Verify that attempting to fit thresholds on 2018 or 2019 data raises PermissionError."""
    df_heldout = pd.DataFrame({
        "init_date": pd.date_range("2018-06-01", periods=10),
        "region_id": ["7"] * 10,
        "lead_day": [1] * 10,
        "variable": ["rainfall"] * 10,
        "forecast_value": [10.0] * 10,
        "observed_value": [5.0] * 10,
    })

    with pytest.raises(PermissionError, match="held-out years"):
        fit_bust_thresholds(df_heldout)


def test_primary_label_decision_relevance_components():
    """Verify primary bust requires magnitude AND alert flip AND significant weather."""
    thresholds = ThresholdModel(
        p95_lookup={("7", 6, 1, "rainfall"): 8.0},  # Below floor_mag of 10.0 -> max(10, 8) = 10.0
        p90_lookup={("7", 6, 1, "rainfall"): 4.0},
        global_p95={(1, "rainfall"): 10.0},
        global_p90={(1, "rainfall"): 5.0},
    )

    # Case 1: Large error (12 mm), category flip (Light -> Moderate), significant weather (O = 20 mm)
    # F = 8 mm (Light), O = 20 mm (Moderate), |e| = 12 mm >= 10 mm -> BUST (wet)
    df1 = pd.DataFrame([{
        "init_date": pd.Timestamp("2000-06-01"),
        "region_id": "7",
        "lead_day": 1,
        "variable": "rainfall",
        "forecast_value": 8.0,
        "observed_value": 20.0,
    }])
    res1 = compute_bust_labels(df1, thresholds)
    assert bool(res1["is_bust"].iloc[0]) is True
    assert res1["bust_direction"].iloc[0] == "wet"
    assert bool(res1["is_category_flip"].iloc[0]) is True

    # Case 2: Large error (14 mm), but NO significant weather (both < 15.6 mm)
    # F = 0.5 mm (Very Light), O = 14.5 mm (Light), max(F, O) = 14.5 < 15.6 -> NOT BUST
    df2 = pd.DataFrame([{
        "init_date": pd.Timestamp("2000-06-01"),
        "region_id": "7",
        "lead_day": 1,
        "variable": "rainfall",
        "forecast_value": 0.5,
        "observed_value": 14.5,
    }])
    res2 = compute_bust_labels(df2, thresholds)
    assert bool(res2["is_bust"].iloc[0]) is False
    assert res2["bust_direction"].iloc[0] == "none"

    # Case 3: Large error (12 mm), significant weather, but NO category flip
    # F = 25 mm (Moderate), O = 37 mm (Moderate) -> same category! NOT BUST
    df3 = pd.DataFrame([{
        "init_date": pd.Timestamp("2000-06-01"),
        "region_id": "7",
        "lead_day": 1,
        "variable": "rainfall",
        "forecast_value": 25.0,
        "observed_value": 37.0,
    }])
    res3 = compute_bust_labels(df3, thresholds)
    assert bool(res3["is_bust"].iloc[0]) is False
    assert bool(res3["is_category_flip"].iloc[0]) is False

    # Case 4: Category flip and significant, but magnitude too small (< 10 mm)
    # F = 15.0 mm (Light), O = 16.5 mm (Moderate), |e| = 1.5 mm < 10 mm -> NOT BUST
    df4 = pd.DataFrame([{
        "init_date": pd.Timestamp("2000-06-01"),
        "region_id": "7",
        "lead_day": 1,
        "variable": "rainfall",
        "forecast_value": 15.0,
        "observed_value": 16.5,
    }])
    res4 = compute_bust_labels(df4, thresholds)
    assert bool(res4["is_bust"].iloc[0]) is False


def test_secondary_p90_label_construction():
    """Verify secondary label is_bust_p90 triggers when error exceeds P90 and 5.0 mm floor."""
    thresholds = ThresholdModel(
        p95_lookup={},
        p90_lookup={("7", 6, 1, "rainfall"): 7.0},
        global_p95={(1, "rainfall"): 12.0},
        global_p90={(1, "rainfall"): 7.0},
    )

    # Error = 8.0 mm >= 7.0 mm (P90) and >= 5.0 mm floor -> secondary bust
    df = pd.DataFrame([{
        "init_date": pd.Timestamp("2000-06-01"),
        "region_id": "7",
        "lead_day": 1,
        "variable": "rainfall",
        "forecast_value": 20.0,
        "observed_value": 28.0,
    }])
    res = compute_bust_labels(df, thresholds)
    assert bool(res["is_bust_p90"].iloc[0]) is True


def test_four_sanity_checks_end_to_end():
    """Generate realistic synthetic training evaluation dataset and verify all 4 sanity checks."""
    rng = np.random.default_rng(42)
    n_days = 300
    n_regions = 36
    dates = pd.date_range("2000-05-01", periods=n_days)
    records = []

    for d in dates:
        # Synoptic daily rainfall regime
        base_rain = rng.exponential(scale=14.0, size=n_regions)
        for r_idx in range(1, n_regions + 1):
            reg_id = str(r_idx)
            obs_val = float(base_rain[r_idx - 1])
            for l in range(1, 11):
                # Error std increases monotonically with lead day: sigma = 1.5 + 1.6 * (l ** 1.15)
                error_sigma = 1.5 + 1.6 * (l ** 1.15)
                error = float(rng.normal(0.0, error_sigma))
                fcst_val = max(0.0, obs_val + error)
                records.append({
                    "init_date": d,
                    "region_id": reg_id,
                    "lead_day": l,
                    "variable": "rainfall",
                    "forecast_value": fcst_val,
                    "observed_value": obs_val,
                })

    df = pd.DataFrame(records)

    # 1. Fit thresholds on training data
    thresholds = fit_bust_thresholds(df, fit_period_label="2000-sample")

    # 2. Compute bust labels
    labeled_df = compute_bust_labels(df, thresholds)

    # 3. Execute all 4 sanity checks
    results = verify_sanity_checks(labeled_df)

    # Assert Check 1: Monotonicity
    assert results["check_1_monotonic_pass"] is True
    rates = results["lead_bust_rates"]
    assert rates[1] < rates[10], f"Lead 10 rate ({rates[10]}) must exceed lead 1 rate ({rates[1]})"

    # Assert Check 2: Magnitude floor
    assert results["check_2_magnitude_floor_pass"] is True
    assert results["min_bust_abs_error"] >= 10.0 - 1e-4

    # Assert Check 3: Primary base rate ~3.5% - 5.5%
    assert results["check_3_primary_base_rate_pass"] is True
    assert 0.030 <= results["primary_base_rate"] <= 0.060

    # Assert Check 4: Secondary base rate ~10%
    assert results["check_4_secondary_base_rate_pass"] is True
    assert 0.080 <= results["secondary_base_rate"] <= 0.120
