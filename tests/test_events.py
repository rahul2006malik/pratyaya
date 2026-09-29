"""Unit tests for Independent Bust Event Clustering (Node F8, docs/05 §4.1)."""

import pandas as pd
import pytest

from pratyaya.evaluate.events import (
    EventClusteringResult,
    cluster_bust_events,
    compute_subdivision_adjacency,
    haversine_distance_km,
)


def test_haversine_distance() -> None:
    """Verify haversine distance calculation between known geographic coordinates."""
    # Delhi (28.61, 77.21) to Mumbai (19.08, 72.88) ~ 1148 km
    d = haversine_distance_km(28.61, 77.21, 19.08, 72.88)
    assert 1100.0 < d < 1200.0


def test_subdivision_adjacency_built() -> None:
    """Verify subdivision adjacency topology produces non-empty neighbour sets."""
    adj = compute_subdivision_adjacency()
    assert len(adj) >= 36
    # Each region must be adjacent to itself
    for r in ["1", "5", "10", "36"]:
        if r in adj:
            assert r in adj[r]


def test_spatiotemporal_event_clustering() -> None:
    """Verify that temporally and spatially adjacent bust rows merge into single event IDs."""
    # Define explicit custom adjacency: 1 adjacent to 2; 2 adjacent to 3; 30 isolated
    custom_adj = {
        "1": {"1", "2"},
        "2": {"2", "1", "3"},
        "3": {"3", "2"},
        "30": {"30"},
    }

    records = [
        # Event 1: Monsoon depression traversing regions 1, 2, 3 across July 1-3
        {"valid_date": "2000-07-01", "region_id": "1", "is_bust": True, "lead_day": 3},
        {"valid_date": "2000-07-01", "region_id": "2", "is_bust": True, "lead_day": 4},
        {"valid_date": "2000-07-02", "region_id": "2", "is_bust": True, "lead_day": 5},
        {"valid_date": "2000-07-03", "region_id": "3", "is_bust": True, "lead_day": 6},

        # Event 2: Independent isolated bust 2 weeks later in Region 30
        {"valid_date": "2000-07-20", "region_id": "30", "is_bust": True, "lead_day": 5},

        # Non-bust rows
        {"valid_date": "2000-07-01", "region_id": "3", "is_bust": False, "lead_day": 3},
        {"valid_date": "2000-07-10", "region_id": "1", "is_bust": False, "lead_day": 1},
    ]
    df = pd.DataFrame(records)

    result = cluster_bust_events(
        df,
        custom_adjacency=custom_adj,
        temporal_window_days=1,
    )

    labeled_df = result.labeled_df

    # 1. Non-bust rows must receive event_id = -1
    non_bust_events = labeled_df.loc[~labeled_df["is_bust"], "event_id"].unique()
    assert list(non_bust_events) == [-1]

    # 2. Event count must be exactly 2
    assert result.n_events == 2
    assert result.n_bust_rows == 5

    # 3. Inflation factor: 5 bust rows / 2 events = 2.5
    assert result.inflation_factor == 2.5

    # 4. First 4 rows should all share the same event_id
    event_1_id = labeled_df.loc[0, "event_id"]
    assert event_1_id > 0
    assert labeled_df.loc[1, "event_id"] == event_1_id
    assert labeled_df.loc[2, "event_id"] == event_1_id
    assert labeled_df.loc[3, "event_id"] == event_1_id

    # 5. Row 4 (Region 30 on July 20) must have a different event_id
    event_2_id = labeled_df.loc[4, "event_id"]
    assert event_2_id > 0
    assert event_2_id != event_1_id


def test_zero_bust_handling() -> None:
    """Verify behavior when dataset has zero positive busts."""
    df = pd.DataFrame({
        "valid_date": ["2000-06-01", "2000-06-02"],
        "region_id": ["1", "2"],
        "is_bust": [False, False],
        "lead_day": [1, 2],
    })

    result = cluster_bust_events(df)
    assert result.n_events == 0
    assert result.n_bust_rows == 0
    assert result.inflation_factor == 1.0
    assert (result.labeled_df["event_id"] == -1).all()
