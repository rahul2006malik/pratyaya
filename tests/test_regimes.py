"""Unit test suite for Project Pratyaya Day 7 Synoptic Regime Discovery (Node G7).

Validates:
1. National bust burden calculation restricted to medium-range forecast leads 3–7.
2. Top quintile (P80) high-burden initialization cycle identification.
3. Gaussian temporal smoothing from cycle burden to daily synoptic dates.
4. Objective silhouette score analysis over candidate range k in [3, 8] selecting argmax silhouette.
5. Composite raw ERA5 synoptic anomaly fields (u850, v850, z500, tcwv, mslp) spatial dimensions (38, 61).
6. External validation against independent event catalogues computing enrichment ratio (E = Obs / Exp),
   bootstrap 95% confidence intervals, and honest classification ('validated' vs 'unvalidated_exploratory').
7. Physical cluster characterization: named mechanisms, JJAS seasonality, threatened subdivisions,
   vulnerable leads, and wet/dry signed error bias.
8. End-to-end `SynopticRegimeClusterer` execution.
9. Strict enforcement of Decision D-000 and Risk R-10 (zero access to held-out years 2018–2019).
10. FastAPI endpoint `GET /api/regimes` and `ProductionModelServer.get_regimes()`.
"""

from __future__ import annotations

from typing import Dict, List, Set

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from pratyaya.analogue.regimes import (
    AVAILABLE_CATALOGUES,
    ClusterCharacterization,
    EnrichmentResult,
    RegimeClusteringResult,
    SynopticRegimeClusterer,
    characterize_cluster,
    composite_raw_synoptic_anomalies,
    compute_national_bust_burden,
    evaluate_catalogue_enrichment,
    identify_high_burden_dates,
    select_optimal_k,
    smooth_burden_to_daily_dates,
    summarize_composite_anomalies,
)
from pratyaya.analogue.state import WideDomainClimatology
from pratyaya.api.main import app
from pratyaya.api.schemas import RegimesResponse
from pratyaya.models.serving import ProductionModelServer, get_model_server

client = TestClient(app)


# ==============================================================================
# Fixtures
# ==============================================================================
@pytest.fixture
def mock_labels_df() -> pd.DataFrame:
    """Create synthetic training-year (2000-2001) labelled forecast outcomes."""
    rng = np.random.RandomState(42)
    init_dates = [
        "2000-06-05", "2000-06-15", "2000-07-01", "2000-07-10", "2000-08-01",
        "2001-06-08", "2001-06-20", "2001-07-05", "2001-07-15", "2001-08-10",
    ]
    records = []
    for d in init_dates:
        for r in range(1, 11):  # 10 regions
            for lead in range(1, 11):  # Leads 1-10
                # Higher bust rate on certain dates and leads 3-7
                p_bust = 0.6 if d in ("2000-07-01", "2001-07-05") and 3 <= lead <= 7 else 0.15
                is_bust = 1 if rng.rand() < p_bust else 0
                error_signed = float(rng.normal(loc=2.0 if is_bust else 0.0, scale=1.5))
                records.append({
                    "init_date": d,
                    "region_id": str(r),
                    "lead_day": lead,
                    "is_bust": is_bust,
                    "error_signed": error_signed,
                })
    return pd.DataFrame(records)


@pytest.fixture
def mock_synoptic_dataset() -> tuple:
    """Create synthetic daily synoptic states and fitted climatology (38 x 61 x 5)."""
    rng = np.random.RandomState(42)
    n_days = 60
    n_features = 38 * 61 * 5  # 11,590 features
    states = rng.normal(loc=10.0, scale=2.0, size=(n_days, n_features)).astype(np.float32)

    # Inject distinct patterns into subgroups of days to form identifiable clusters
    states[0:15, :1000] += 8.0  # Cluster A
    states[15:30, 1000:2000] += 8.0  # Cluster B
    states[30:45, 2000:3000] += 8.0  # Cluster C
    states[45:60, 3000:4000] += 8.0  # Cluster D

    dates = [pd.Timestamp("2000-06-01") + pd.Timedelta(days=i) for i in range(n_days)]

    # Mock climatology
    clim = WideDomainClimatology()
    clim.fit(states, dates)
    return states, dates, clim


# ==============================================================================
# Tests
# ==============================================================================
def test_compute_national_bust_burden(mock_labels_df):
    """Verify burden calculation restricts to leads 3–7 and aggregates correctly."""
    burden_df = compute_national_bust_burden(mock_labels_df, leads=(3, 7))

    assert not burden_df.empty
    assert "init_date" in burden_df.columns
    assert "bust_count" in burden_df.columns
    assert "bust_rate" in burden_df.columns
    assert "mean_signed_error" in burden_df.columns

    # High-bust dates should be at the top
    top_dates = [str(d)[:10] for d in burden_df.head(2)["init_date"]]
    assert "2000-07-01" in top_dates or "2001-07-05" in top_dates

    # Out-of-bounds lead range should raise ValueError
    with pytest.raises(ValueError):
        compute_national_bust_burden(mock_labels_df, leads=(15, 20))


def test_identify_high_burden_dates(mock_labels_df):
    """Verify top quintile (P80) high-burden date selection."""
    burden_df = compute_national_bust_burden(mock_labels_df, leads=(3, 7))
    high_dates = identify_high_burden_dates(burden_df, quantile=0.80)

    assert len(high_dates) > 0
    assert len(high_dates) <= len(burden_df)
    # P80 should select approximately top 20-30% of dates
    assert len(high_dates) >= 2


def test_smooth_burden_to_daily_dates():
    """Verify temporal smoothing maps cycle burden to daily calendar."""
    cycle_map = {"2000-07-01": 25.0, "2000-07-15": 10.0}
    daily_dates = [pd.Timestamp("2000-06-25") + pd.Timedelta(days=i) for i in range(30)]

    daily_burdens, high_dates, threshold = smooth_burden_to_daily_dates(
        cycle_burden_map=cycle_map,
        daily_dates=daily_dates,
        kernel_sigma_days=2.0,
        quantile=0.80,
    )

    assert len(daily_burdens) == 30
    assert len(high_dates) >= 1
    assert threshold > 0.0
    # Peak burden should be on or adjacent to 2000-07-01
    peak_idx = int(np.argmax(daily_burdens))
    assert daily_dates[peak_idx] == pd.Timestamp("2000-07-01")


def test_select_optimal_k_silhouette_objective():
    """Verify objective silhouette score analysis over candidate k in [3, 8]."""
    rng = np.random.RandomState(42)
    # Generate 4 clearly separated Gaussian clusters in 32-D space
    n_per_cluster = 15
    c1 = rng.normal(loc=[5.0] * 32, scale=0.5, size=(n_per_cluster, 32))
    c2 = rng.normal(loc=[-5.0] * 32, scale=0.5, size=(n_per_cluster, 32))
    c3 = rng.normal(loc=[0.0] * 16 + [5.0] * 16, scale=0.5, size=(n_per_cluster, 32))
    c4 = rng.normal(loc=[0.0] * 16 + [-5.0] * 16, scale=0.5, size=(n_per_cluster, 32))
    embeddings = np.vstack([c1, c2, c3, c4])

    optimal_k, sil_scores = select_optimal_k(embeddings, k_range=(3, 6), random_state=42)

    assert 3 <= optimal_k <= 6
    assert len(sil_scores) == 4  # k=3, 4, 5, 6
    assert all(k in sil_scores for k in [3, 4, 5, 6])
    # Cluster structure has 4 true clusters, so k=4 should achieve a high silhouette score (>0.4)
    assert sil_scores[4] > 0.4
    assert optimal_k == 4


def test_composite_raw_synoptic_anomalies():
    """Verify raw ERA5 synoptic anomaly composite field dimensions and statistics."""
    n_samples = 20
    n_features = 38 * 61 * 5
    raw_states = np.ones((n_samples, n_features), dtype=np.float32) * 5.0
    clim_mean = np.ones(n_features, dtype=np.float32) * 3.0  # Anomaly should be +2.0
    cluster_labels = np.array([0] * 10 + [1] * 10)

    composites = composite_raw_synoptic_anomalies(
        raw_states=raw_states,
        clim_mean=clim_mean,
        cluster_labels=cluster_labels,
        n_clusters=2,
    )

    assert 0 in composites and 1 in composites
    for f in ("u850", "v850", "z500", "tcwv", "mslp"):
        assert f in composites[0]
        grid = composites[0][f]
        assert grid.shape == (38, 61)
        assert np.isclose(grid.mean(), 2.0, atol=1e-3)

    summary = summarize_composite_anomalies(composites[0])
    assert "z500" in summary
    assert summary["z500"]["mean"] == 2.0


def test_evaluate_catalogue_enrichment():
    """Verify enrichment ratio, bootstrap 95% CIs, and statistical classification."""
    all_dates = [f"2000-06-{i:02d}" for i in range(1, 31)]
    # Target catalogue: days 10 through 15 (6 days out of 30, base rate = 0.20)
    cat_dates = {f"2000-06-{i:02d}" for i in range(10, 16)}

    # Case 1: Strongly enriched cluster (5 out of 6 members in catalogue)
    enriched_cluster = [f"2000-06-{i:02d}" for i in range(10, 16)]
    res_enriched = evaluate_catalogue_enrichment(
        cluster_dates=enriched_cluster,
        all_candidate_dates=all_dates,
        catalogue_dates=cat_dates,
        catalogue_name="Test Active Monsoon",
        n_bootstrap=500,
    )

    assert res_enriched.observed_count == 6
    assert res_enriched.enrichment_ratio > 2.0
    assert res_enriched.p_value < 0.05
    assert res_enriched.is_significant is True
    assert res_enriched.ci_95[0] > 1.0

    # Case 2: Unenriched cluster (0 members in catalogue)
    unenriched_cluster = [f"2000-06-{i:02d}" for i in range(1, 6)]
    res_unenriched = evaluate_catalogue_enrichment(
        cluster_dates=unenriched_cluster,
        all_candidate_dates=all_dates,
        catalogue_dates=cat_dates,
        catalogue_name="Test Active Monsoon",
        n_bootstrap=500,
    )

    assert res_unenriched.observed_count == 0
    assert res_unenriched.enrichment_ratio == 0.0
    assert res_unenriched.is_significant is False


def test_characterize_cluster(mock_labels_df):
    """Verify physical characterization, seasonality, vulnerable leads, and honest validation."""
    member_dates = ["2000-07-01", "2000-07-02", "2000-07-03"]
    all_dates = ["2000-06-15", "2000-07-01", "2000-07-02", "2000-07-03", "2000-08-01"]
    composite_dict = {
        f: np.zeros((38, 61), dtype=np.float32)
        for f in ("u850", "v850", "z500", "tcwv", "mslp")
    }

    char = characterize_cluster(
        cluster_id=0,
        member_dates=member_dates,
        all_candidate_dates=all_dates,
        labels_df=mock_labels_df,
        composite_dict=composite_dict,
    )

    assert isinstance(char, ClusterCharacterization)
    assert char.cluster_id == 0
    assert char.n_members == 3
    assert char.seasonality["July"] == 1.0  # All dates in July
    assert char.signed_bias in ("wet", "dry")
    assert char.validation_status in ("validated", "unvalidated_exploratory")
    assert len(char.validation) == len(AVAILABLE_CATALOGUES)


def test_synoptic_regime_clusterer_fit(mock_labels_df, mock_synoptic_dataset):
    """Verify end-to-end fitting of SynopticRegimeClusterer."""
    states, dates, clim = mock_synoptic_dataset
    clusterer = SynopticRegimeClusterer(k_range=(3, 5), quantile=0.80, random_state=42)

    clusterer.fit(
        labels_df=mock_labels_df,
        synoptic_states=states,
        synoptic_dates=dates,
        climatology=clim,
    )

    assert clusterer.is_fitted is True
    res = clusterer.result_
    assert isinstance(res, RegimeClusteringResult)
    assert 3 <= res.optimal_k <= 5
    assert len(res.clusters) == res.optimal_k
    assert res.n_high_burden_dates > 0
    assert res.composite_anomaly_grids is not None
    assert len(res.composite_anomaly_grids) == res.optimal_k


def test_regimes_heldout_protection(mock_labels_df, mock_synoptic_dataset):
    """Verify strict held-out year protection raises PermissionError for 2018–2019 (Decision D-000)."""
    tainted_labels = mock_labels_df.copy()
    tainted_labels.loc[0, "init_date"] = "2018-07-01"

    with pytest.raises(PermissionError, match="Decision D-000"):
        compute_national_bust_burden(tainted_labels)

    states, dates, clim = mock_synoptic_dataset
    tainted_dates = list(dates)
    tainted_dates[0] = pd.Timestamp("2019-06-15")

    clusterer = SynopticRegimeClusterer()
    with pytest.raises(PermissionError, match="Decision D-000"):
        clusterer.fit(
            labels_df=mock_labels_df,
            synoptic_states=states,
            synoptic_dates=tainted_dates,
            climatology=clim,
        )


def test_api_regimes_endpoint():
    """Verify FastAPI GET /api/regimes endpoint returns valid schema and status."""
    response = client.get("/api/regimes")
    assert response.status_code == 200
    data = response.json()

    assert "optimal_k" in data
    assert "silhouette_scores" in data
    assert "clusters" in data
    assert "model_version" in data

    clusters = data["clusters"]
    assert len(clusters) == data["optimal_k"]
    assert len(clusters) >= 3

    # Check structure of first cluster
    c0 = clusters[0]
    assert "cluster_id" in c0
    assert "name" in c0
    assert "mechanism_summary" in c0
    assert "seasonality" in c0
    assert "threatened_subdivisions" in c0
    assert "validation_status" in c0
    assert c0["validation_status"] in ("validated", "unvalidated_exploratory")


def test_production_serving_get_regimes():
    """Verify ProductionModelServer provides cached or live regime discovery metadata."""
    server = get_model_server()
    regimes = server.get_regimes()

    assert isinstance(regimes, RegimesResponse)
    assert regimes.optimal_k >= 3
    assert len(regimes.clusters) == regimes.optimal_k
    assert regimes.is_live is True
    assert "pratyaya" in regimes.model_version
