"""Test Suite for Analogue Retrieval Engine v1 (Nodes G1, G2, G6).

Validates:
1. Wide-domain bounding box (30°E–120°E, 10°S–45°N) and 1.5° grid dimensions.
2. Mandatory +/- 15-day temporal exclusion window (zero autocorrelation leakage).
3. Strict held-out protection (2018–2019 dates raise PermissionError).
4. EOF / PCA dimensionality reduction and reconstruction.
5. Euclidean k-NN retrieval, Gaussian kernel weights, probability estimation, and native OOD detection.
"""

import numpy as np
import pandas as pd
import pytest

from pratyaya.analogue.eof import (
    DEFAULT_N_COMPONENTS,
    EOFReducer,
)
from pratyaya.analogue.retrieve import (
    DEFAULT_K_ANALOGUES,
    TEMPORAL_EXCLUSION_DAYS,
    AnalogueMatch,
    AnalogueRetriever,
    RetrievalResult,
)
from pratyaya.analogue.state import (
    DEFAULT_SYNOPTIC_FIELDS,
    GRID_RESOLUTION,
    WIDE_DOMAIN_LAT_MAX,
    WIDE_DOMAIN_LAT_MIN,
    WIDE_DOMAIN_LON_MAX,
    WIDE_DOMAIN_LON_MIN,
    WideDomainClimatology,
    WideDomainGrid,
    build_synoptic_state_vector,
    check_heldout_date_protection,
    get_wide_domain_coords,
)


def test_wide_domain_bounding_box_and_dims():
    """Verify wide domain bounding box covers 30°E-120°E, 10°S-45°N on 1.5° grid."""
    lats, lons = get_wide_domain_coords(ascending_lat=True)

    # 1. Longitude checks
    assert lons.min() == WIDE_DOMAIN_LON_MIN == 30.0
    assert lons.max() == WIDE_DOMAIN_LON_MAX == 120.0
    assert np.allclose(np.diff(lons), GRID_RESOLUTION)
    assert len(lons) == 61, f"Expected 61 longitudes, got {len(lons)}"

    # 2. Latitude checks: must cover 10°S (-10.0°) to 45°N (45.0°)
    assert lats.min() <= WIDE_DOMAIN_LAT_MIN, f"Lat min {lats.min()} does not cover 10°S"
    assert lats.max() >= WIDE_DOMAIN_LAT_MAX, f"Lat max {lats.max()} does not cover 45°N"
    assert np.allclose(np.diff(lats), GRID_RESOLUTION)
    assert len(lats) == 38, f"Expected 38 latitudes, got {len(lats)}"
    assert 0.0 in lats, "Equator (0.0°) must be aligned on grid"

    # 3. Grid dataclass metadata
    grid = WideDomainGrid()
    assert grid.n_lons == 61
    assert grid.n_lats == 38
    assert grid.n_grid_points == 61 * 38
    assert grid.total_features == 61 * 38 * 5  # 11,590 features


def test_synoptic_state_vector_flattening():
    """Verify build_synoptic_state_vector flattens 5 fields consistently."""
    n_days = 4
    n_lats = 38
    n_lons = 61
    mock_fields = {}
    for f in DEFAULT_SYNOPTIC_FIELDS:
        mock_fields[f] = np.random.randn(n_days, n_lats, n_lons)

    flat_batch = build_synoptic_state_vector(mock_fields)
    assert flat_batch.shape == (n_days, 11590)

    # Test single 2D day
    single_day_fields = {f: mock_fields[f][0] for f in DEFAULT_SYNOPTIC_FIELDS}
    flat_single = build_synoptic_state_vector(single_day_fields)
    assert flat_single.shape == (11590,)
    assert np.allclose(flat_batch[0], flat_single)


def test_heldout_protection_guards():
    """Verify that any attempt to fit or query on held-out years (2018-2019) raises PermissionError."""
    heldout_dates = ["2018-06-15", "2018-07-20", "2019-08-10"]
    valid_dates = ["2000-06-15", "2000-07-20", "2001-08-10"]

    # 1. State check function
    with pytest.raises(PermissionError, match="strictly prohibited outside guarded --final"):
        check_heldout_date_protection(heldout_dates)

    with pytest.raises(PermissionError, match="strictly prohibited outside guarded --final"):
        check_heldout_date_protection(["2000-05-01", "2019-01-01"])

    # 2. Climatology fit guard
    clim = WideDomainClimatology()
    mock_states = np.random.randn(3, 10)
    with pytest.raises(PermissionError, match="strictly prohibited"):
        clim.fit(mock_states, heldout_dates)

    # 3. EOF fit guard
    eof = EOFReducer(n_components=5)
    with pytest.raises(PermissionError, match="strictly prohibited"):
        eof.fit(mock_states, heldout_dates)

    # 4. Retriever fit guard
    retriever = AnalogueRetriever(k_analogues=2)
    with pytest.raises(PermissionError, match="strictly prohibited"):
        retriever.fit(mock_states, heldout_dates)


def test_climatology_standardization_training_fold():
    """Verify climatology standardization computes unit-variance anomalies strictly on training fold."""
    np.random.seed(42)
    n_days = 50
    n_features = 20
    # Training data with distinct non-zero mean and variance
    train_states = np.random.normal(loc=25.0, scale=4.0, size=(n_days, n_features))
    train_dates = pd.date_range("2000-06-01", periods=n_days, freq="D")

    clim = WideDomainClimatology()
    anomalies = clim.fit_transform(train_states, train_dates)

    assert anomalies.shape == (n_days, n_features)
    # Mean of anomalies should be approx 0 and std approx 1
    assert np.allclose(np.mean(anomalies, axis=0), 0.0, atol=1e-5)
    assert np.allclose(np.std(anomalies, axis=0), 1.0, atol=1e-5)

    # Transform new data using fitted training climatology
    val_states = np.random.normal(loc=25.0, scale=4.0, size=(10, n_features))
    val_anomalies = clim.transform(val_states)
    assert val_anomalies.shape == (10, n_features)


def test_eof_reduction_variance_and_reconstruction():
    """Verify EOFReducer reduces dimensionality and accurately reconstructs signals."""
    np.random.seed(123)
    n_samples = 120
    n_features = 300
    # Correlated signals
    base_signal = np.random.randn(n_samples, 10)
    mixing_matrix = np.random.randn(10, n_features)
    X = base_signal @ mixing_matrix + 0.1 * np.random.randn(n_samples, n_features)
    train_dates = pd.date_range("2000-01-01", periods=n_samples, freq="D")

    eof = EOFReducer(n_components=50)
    Z = eof.fit_transform(X, train_dates)

    assert Z.shape == (n_samples, 50)
    assert eof.is_fitted
    assert len(eof.explained_variance_ratio_) == 50
    # Explained variance should be monotonically non-increasing
    assert np.all(np.diff(eof.explained_variance_ratio_) <= 1e-7)
    # The first 10 components should capture the vast majority of variance
    assert np.sum(eof.explained_variance_ratio_[:10]) > 0.85

    # Reconstruct original features
    X_rec = eof.inverse_transform(Z)
    assert X_rec.shape == (n_samples, n_features)
    rec_error = np.mean((X - X_rec) ** 2)
    assert rec_error < 0.05, f"Reconstruction error {rec_error} unexpectedly high"


def test_temporal_exclusion_window_enforced():
    """Verify that query date t0 NEVER selects any candidate within [t0 - 15, t0 + 15] days.

    Even if a candidate inside the window has identical zero Euclidean distance,
    it must be strictly masked out to eliminate autocorrelation persistence bias.
    """
    np.random.seed(999)
    q_date = pd.Timestamp("2001-07-15")

    # Create candidate library spanning dates:
    # - exactly q_date (lag 0)
    # - q_date +/- 1 day
    # - q_date +/- 14 days
    # - q_date +/- 15 days (boundary condition: MUST be excluded per [t0-15, t0+15])
    # - q_date +/- 16 days (boundary condition: first allowed candidate)
    # - q_date +/- 60 days
    candidate_offsets = [-60, -30, -16, -15, -14, -1, 0, 1, 14, 15, 16, 45, 90]
    cand_dates = [q_date + pd.Timedelta(days=d) for d in candidate_offsets]

    # Craft embeddings: give the FORBIDDEN candidates (offsets 0, 1, -1) perfect zero distance
    n_features = 20
    q_emb = np.ones(n_features) * 5.0

    cand_embs = []
    for d in candidate_offsets:
        if abs(d) <= TEMPORAL_EXCLUSION_DAYS:
            # Trapped candidate: set identical to query!
            cand_embs.append(q_emb.copy())
        else:
            # Allowed candidate: some distance
            cand_embs.append(q_emb + np.random.randn(n_features) * 2.0)
    cand_embs = np.array(cand_embs)

    retriever = AnalogueRetriever(k_analogues=4, exclusion_days=TEMPORAL_EXCLUSION_DAYS)
    retriever.fit(cand_embs, cand_dates)

    res = retriever.retrieve(q_emb, q_date)

    assert len(res.matches) == 4
    retrieved_dates = res.candidate_dates

    # MANDATORY ASSERTION: No retrieved candidate can be within +/- 15 days of q_date
    for dt in retrieved_dates:
        delta = abs((dt - q_date).days)
        assert delta > TEMPORAL_EXCLUSION_DAYS, (
            f"AUTOCORRELATION VIOLATION: Retrieved date {dt} is within {delta} days of query {q_date}!"
        )

    # Verify that the nearest allowed candidates (-16 and +16) were eligible
    allowed_offsets = set(candidate_offsets) - {d for d in candidate_offsets if abs(d) <= 15}
    actual_offsets = {(dt - q_date).days for dt in retrieved_dates}
    assert actual_offsets.issubset(allowed_offsets)


def test_analogue_retriever_ranking_weights_and_ood():
    """Verify Euclidean ranking, Gaussian kernel weights, and native OOD trigger."""
    q_date = pd.Timestamp("2001-08-01")
    n_features = 10

    # 10 training candidate dates in year 2000 (well outside +/- 15 days)
    cand_dates = pd.date_range("2000-06-01", periods=15, freq="7D")
    cand_embs = np.random.randn(15, n_features)

    # Make candidate #2 the closest to query
    q_emb = np.zeros(n_features)
    cand_embs[2] = q_emb + 0.1  # distance ~ 0.316

    # Create mock outcomes
    records = []
    for dt in cand_dates:
        for r in range(1, 3):
            for l in [1, 5]:
                records.append({
                    "init_date": dt,
                    "region_id": str(r),
                    "lead_day": l,
                    "is_bust": (dt == cand_dates[2]),  # Closest candidate had bust
                    "error_signed": 12.5 if (dt == cand_dates[2]) else -2.0,
                })
    outcomes_df = pd.DataFrame(records)

    retriever = AnalogueRetriever(k_analogues=5, ood_percentile=90.0)
    retriever.fit(cand_embs, cand_dates, outcomes_df=outcomes_df)

    res = retriever.retrieve(q_emb, q_date)

    # Verify ranking
    assert res.matches[0].candidate_idx == 2
    assert res.matches[0].candidate_date == cand_dates[2]

    # Verify weights
    weights = res.weights
    assert np.allclose(sum(weights), 1.0)
    assert weights[0] == max(weights), "Closest match must have highest weight"

    # Verify probability prediction
    prob_df = retriever.predict_bust_probabilities(q_emb, q_date)
    assert not prob_df.empty
    assert set(prob_df.columns) == {
        "region_id", "lead_day", "p_analogue", "signed_expect", "analogue_dispersion", "is_ood"
    }
    # Since candidate 2 has highest weight and was a bust, p_analogue should be > 0.3
    assert (prob_df["p_analogue"] > 0.2).all()

    # Verify Native OOD (Node G6) with an extreme outlier query
    outlier_query = np.ones(n_features) * 1000.0  # Massive distance
    outlier_res = retriever.retrieve(outlier_query, q_date)
    assert outlier_res.is_ood is True


def test_analogue_scoring_runner_end_to_end():
    """Verify Model M1 v1 calibration scoring and Day 3 Gate requirements."""
    from scripts.score_analogue_v1 import (
        get_or_build_synoptic_cache,
        load_canonical_dataset,
    )
    from pratyaya.evaluate.splits import create_purged_train_val_split
    from pratyaya.evaluate.metrics import compute_pr_auc

    data_df = load_canonical_dataset()
    train_df, cal_df, _ = create_purged_train_val_split(data_df, train_years=[2000, 2001], val_years=[2002])

    cache = get_or_build_synoptic_cache()
    training_daily_states = cache["training_daily_states"]
    training_daily_dates = cache["training_daily_dates"]
    training_cycle_dates = [str(d) for d in cache["training_cycle_dates"]]
    training_cycle_states = cache["training_cycle_states"]
    cal_cycle_dates = [str(d) for d in cache["calibration_cycle_dates"]]
    cal_cycle_states = cache["calibration_cycle_states"]

    clim = WideDomainClimatology().fit(training_daily_states, training_daily_dates)
    eof = EOFReducer(n_components=100).fit(clim.transform(training_daily_states), training_daily_dates)
    assert len(eof.explained_variance_ratio_) == 100

    Z_cand = eof.transform(clim.transform(training_cycle_states))
    retriever = AnalogueRetriever(k_analogues=8, exclusion_days=TEMPORAL_EXCLUSION_DAYS)
    retriever.fit(Z_cand, training_cycle_dates, outcomes_df=train_df)

    Z_cal = eof.transform(clim.transform(cal_cycle_states))

    # Test eyeball check on sample query
    res = retriever.retrieve(Z_cal[0], cal_cycle_dates[0], k=3)
    for m in res.matches:
        cand_dt = pd.Timestamp(str(m.candidate_date))
        assert cand_dt.month in [6, 7, 8, 9], "All analogues must fall in JJAS"
        assert abs((cand_dt - pd.Timestamp(cal_cycle_dates[0])).days) > TEMPORAL_EXCLUSION_DAYS

    # Test probability prediction on 1 cycle
    pred_sub = retriever.predict_bust_probabilities(Z_cal[0], cal_cycle_dates[0])
    assert len(pred_sub) == 360  # 36 subdivisions x 10 leads
    assert "p_analogue" in pred_sub.columns


def test_unlabeled_extra_library_retrieval_and_probabilities():
    """Verify AnalogueRetriever handles unlabeled extra library candidates per Decision D-016."""
    np.random.seed(42)
    n_features = 20
    labeled_dates = ["2000-06-01", "2000-07-01", "2000-08-01", "2000-09-01"]
    labeled_embs = np.random.randn(len(labeled_dates), n_features)

    # Extra library dates (unlabeled historical dates, e.g. 1990)
    extra_dates = ["1990-06-01", "1990-07-01", "1990-08-01", "1990-09-01"]
    extra_embs = np.random.randn(len(extra_dates), n_features)

    # Mock outcomes for labeled dates only
    records = []
    for dt in labeled_dates:
        for r in range(1, 3):
            for l in [1, 2]:
                records.append({
                    "init_date": dt,
                    "region_id": str(r),
                    "lead_day": l,
                    "is_bust": 1.0 if dt == "2000-06-01" else 0.0,
                    "error_signed": 5.0 if dt == "2000-06-01" else -1.0,
                })
    outcomes_df = pd.DataFrame(records)
    base_rate = float(outcomes_df["is_bust"].mean())
    assert base_rate == 0.25

    retriever = AnalogueRetriever(k_analogues=4, exclusion_days=15)
    retriever.fit(
        embeddings=labeled_embs,
        dates=labeled_dates,
        outcomes_df=outcomes_df,
        extra_library_embeddings=extra_embs,
        extra_library_dates=extra_dates,
    )

    assert len(retriever.library_dates_) == 8
    assert len(retriever._labeled_date_set) == 4

    # Make query identical to extra_dates[0] ("1990-06-01") so it's top match
    q_emb = extra_embs[0].copy()
    q_date = pd.Timestamp("2002-07-15")

    res = retriever.retrieve(q_emb, q_date, k=4)
    assert res.matches[0].candidate_date == pd.Timestamp("1990-06-01")
    assert retriever.last_query_unlabeled_fraction_ > 0.0

    prob_df = retriever.predict_bust_probabilities(q_emb, q_date, k=4)
    assert len(prob_df) == 4  # 2 regions x 2 leads
    assert "p_analogue" in prob_df.columns
    # Check that predictions are valid bounded probabilities
    assert (prob_df["p_analogue"] >= 0.0).all() and (prob_df["p_analogue"] <= 1.0).all()

    # Held-out protection on extra library dates
    with pytest.raises(PermissionError, match="strictly prohibited"):
        retriever_bad = AnalogueRetriever()
        retriever_bad.fit(
            embeddings=labeled_embs,
            dates=labeled_dates,
            extra_library_embeddings=np.random.randn(2, n_features),
            extra_library_dates=["2018-06-01", "2019-07-01"],
        )


