"""Unit tests for Linear Low-Rank Mahalanobis Metric Learning (Node G2, docs/03 §6.2).

Tests:
1. Outcome vector construction: R^(36 x 10) = R^360 dimensions, standardization.
2. Triplet sampling: mandatory >= 15-day calendar separation between anchor and positive.
3. Strict held-out date quarantine (rejection of 2018-2019 dates).
4. LowRankMahalanobisMetric: projection dimensions (32 x 100), distance axioms, and Adam training convergence.
5. Save and load weight persistence roundtrip.
"""

import numpy as np
import pandas as pd
import pytest

from pratyaya.analogue.metric import (
    DEFAULT_EOF_DIM,
    DEFAULT_METRIC_COMPONENTS,
    LowRankMahalanobisMetric,
    build_outcome_vectors,
    sample_triplets,
)


def _make_dummy_outcomes_df(dates, n_regions=36, n_leads=10):
    """Generate synthetic canonical outcomes DataFrame for testing."""
    records = []
    rng = np.random.default_rng(123)
    for dt in dates:
        for r in range(1, n_regions + 1):
            for lead in range(1, n_leads + 1):
                fcst = float(rng.uniform(5.0, 50.0))
                obs = float(fcst + rng.normal(0.0, 10.0))
                err = fcst - obs
                records.append({
                    "init_date": dt,
                    "region_id": str(r),
                    "lead_day": lead,
                    "forecast_value": fcst,
                    "observation_value": obs,
                    "error_signed": err,
                    "is_bust": 1 if abs(err) > 15.0 else 0,
                })
    return pd.DataFrame(records)


def test_outcome_vector_construction_and_standardization():
    """Verify outcome vector construction achieves exact 360-dim shape and standardization."""
    dates = [
        "2000-06-01", "2000-06-15", "2000-07-01", "2000-07-20",
        "2001-06-05", "2001-06-25", "2001-07-10", "2001-08-01",
    ]
    df = _make_dummy_outcomes_df(dates, n_regions=36, n_leads=10)

    outcome_mat, stats = build_outcome_vectors(df, dates, n_leads=10, standardize=True)

    # Dimension check: 8 dates x (36 regions * 10 leads = 360)
    assert outcome_mat.shape == (8, 360)
    assert stats["dim"] == 360
    assert len(stats["regions"]) == 36
    assert stats["leads"] == list(range(1, 11))

    # Standardized: columns have approximately 0 mean and 1 variance
    col_means = np.mean(outcome_mat, axis=0)
    col_stds = np.std(outcome_mat, axis=0)
    np.testing.assert_allclose(col_means, 0.0, atol=1e-7)
    np.testing.assert_allclose(col_stds, 1.0, atol=1e-5)


def test_outcome_vector_heldout_protection():
    """Verify outcome vector constructor strictly rejects held-out dates (2018–2019)."""
    heldout_dates = ["2018-06-01", "2018-07-15"]
    df = _make_dummy_outcomes_df(heldout_dates)

    with pytest.raises(PermissionError, match="Access to held-out years"):
        build_outcome_vectors(df, heldout_dates)


def test_triplet_sampling_enforces_temporal_separation():
    """Verify sampled triplets strictly enforce >= 15-day calendar separation between anchor and positive."""
    dates = pd.date_range("2000-06-01", "2000-08-31", freq="5D")
    rng = np.random.default_rng(42)
    # Synthetic outcome vectors
    N = len(dates)
    outcome_vecs = rng.normal(0, 1, size=(N, 360))

    triplets = sample_triplets(
        outcome_vecs,
        dates,
        n_triplets=100,
        min_separation_days=15,
        seed=42,
    )

    assert len(triplets) == 100

    for a_idx, p_idx, n_idx in triplets:
        a_date = dates[a_idx]
        p_date = dates[p_idx]
        delta_days = abs((a_date - p_date).days)

        # Mandatory >= 15-day temporal separation constraint
        assert delta_days >= 15, f"Anchor {a_date} and positive {p_date} only {delta_days} days apart!"

        # Outcome space distance constraint: d(a, p) < d(a, n)
        d_ap = np.linalg.norm(outcome_vecs[a_idx] - outcome_vecs[p_idx])
        d_an = np.linalg.norm(outcome_vecs[a_idx] - outcome_vecs[n_idx])
        assert d_ap < d_an, f"Outcome ordering violated: d_ap={d_ap:.4f} >= d_an={d_an:.4f}"


def test_triplet_sampling_heldout_protection():
    """Verify triplet sampler strictly rejects held-out years."""
    heldout_dates = [pd.Timestamp("2018-06-01"), pd.Timestamp("2018-07-01"), pd.Timestamp("2018-08-01")]
    vecs = np.random.randn(3, 360)

    with pytest.raises(PermissionError, match="Access to held-out years"):
        sample_triplets(vecs, heldout_dates, n_triplets=5)


def test_low_rank_mahalanobis_metric_dimensions_and_distance():
    """Verify low-rank metric projection dimensions (32 x 100) and Mahalanobis distance properties."""
    metric = LowRankMahalanobisMetric(n_components=32, input_dim=100)

    # Synthetic training EOF data (20 samples x 100 EOFs)
    rng = np.random.default_rng(99)
    X = rng.normal(0, 1, size=(20, 100))

    # Construct trivial triplets
    triplets = [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
    ]

    metric.fit(X, triplets)
    assert metric.is_fitted
    assert metric.W_ is not None
    assert metric.W_.shape == (32, 100)
    assert metric.metric_matrix.shape == (100, 100)

    # Projection test
    Z = metric.transform(X)
    assert Z.shape == (20, 32)

    # 1D projection test
    z0 = metric.transform(X[0])
    assert z0.shape == (32,)

    # Distance axioms
    d_same = metric.mahalanobis_distance(X[0], X[0])
    assert d_same == pytest.approx(0.0, abs=1e-6)

    d_diff = metric.mahalanobis_distance(X[0], X[1])
    d_sym = metric.mahalanobis_distance(X[1], X[0])
    assert d_diff > 0.0
    assert d_diff == pytest.approx(d_sym, rel=1e-5)

    # Triangle inequality: d(x, z) <= d(x, y) + d(y, z)
    d_xz = metric.mahalanobis_distance(X[0], X[2])
    d_xy = metric.mahalanobis_distance(X[0], X[1])
    d_yz = metric.mahalanobis_distance(X[1], X[2])
    assert d_xz <= (d_xy + d_yz + 1e-6)


def test_metric_training_reduces_loss():
    """Verify that mini-batch Adam optimization decreases triplet ranking loss over epochs."""
    rng = np.random.default_rng(42)
    N = 30
    X = rng.normal(0, 1, size=(N, 100))

    # Generate synthetic triplets
    triplets = []
    for i in range(100):
        a = rng.integers(0, N)
        p = (a + 5) % N
        n = (a + 12) % N
        triplets.append((int(a), int(p), int(n)))

    metric = LowRankMahalanobisMetric(
        n_components=32,
        input_dim=100,
        margin=1.0,
        learning_rate=0.01,
        n_epochs=40,
        batch_size=32,
        seed=42,
    )
    metric.fit(X, triplets)

    assert len(metric.loss_history_) == 40
    # Average of last 5 epochs should be lower than average of first 5 epochs
    initial_loss = np.mean(metric.loss_history_[:5])
    final_loss = np.mean(metric.loss_history_[-5:])
    assert final_loss < initial_loss, f"Loss did not decrease: initial={initial_loss}, final={final_loss}"


def test_metric_save_and_load_roundtrip(tmp_path):
    """Verify that LowRankMahalanobisMetric can be persisted and loaded losslessly."""
    rng = np.random.default_rng(77)
    X = rng.normal(0, 1, size=(10, 100))
    triplets = [(0, 1, 2), (2, 3, 4)]

    metric = LowRankMahalanobisMetric(n_components=32, input_dim=100, n_epochs=5)
    metric.fit(X, triplets)

    save_path = tmp_path / "metric_test.npz"
    metric.save(save_path)

    loaded = LowRankMahalanobisMetric.load(save_path)
    assert loaded.is_fitted
    assert loaded.n_components == 32
    assert loaded.input_dim == 100
    np.testing.assert_allclose(metric.W_, loaded.W_)

    # Verify transform identical
    Z_orig = metric.transform(X)
    Z_loaded = loaded.transform(X)
    np.testing.assert_allclose(Z_orig, Z_loaded)
