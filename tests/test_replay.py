"""Unit test suite for Replay Demonstration Engine (Node F15 / Screen 4).

Tests canonical replay scenarios, frame transitions, schema validation,
dynamic model-scored caption synthesis, and FastAPI demonstration endpoints.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pratyaya.api.main import app
from pratyaya.api.replay import (
    get_available_scenarios,
    get_replay_scenario,
    synthesize_caption_strip,
)
from pratyaya.api.schemas import (
    BustDirection,
    ConfidenceBand,
    ReplayFrame,
    ReplayScenario,
    ScenarioSummary,
)


@pytest.fixture
def api_client() -> TestClient:
    return TestClient(app)


def test_get_available_scenarios():
    """Verify scenario catalog returns both pre-committed canonical demonstration cases."""
    scenarios = get_available_scenarios()
    assert len(scenarios) == 2
    ids = {s.id for s in scenarios}
    assert "july2002_break" in ids
    assert "july2001_depression" in ids

    for s in scenarios:
        assert isinstance(s, ScenarioSummary)
        assert len(s.title) > 0
        assert len(s.description) > 0
        assert len(s.cycle_date) == 10
        assert "Lead Days" in s.leads


def test_get_replay_scenario_case1_july2002_break():
    """Verify Case 1: July 2002 Break Monsoon Onset frames and metrics."""
    scenario = get_replay_scenario("july2002_break", max_lead=7)
    assert isinstance(scenario, ReplayScenario)
    assert scenario.scenario_id == "july2002_break"
    assert scenario.cycle_date == "2002-07-07"
    assert len(scenario.frames) == 7

    for idx, frame in enumerate(scenario.frames, start=1):
        assert isinstance(frame, ReplayFrame)
        assert frame.lead_day == idx
        assert frame.cycle_date == "2002-07-07"
        assert len(frame.subdivisions) == 36
        assert len(frame.caption_strip) > 20
        assert f"Day {idx} lead:" in frame.caption_strip
        assert frame.max_p_bust >= 0.0
        assert frame.mean_effective_lead_time >= 0.0
        assert len(frame.triage_queue) <= 10
        assert frame.confidence_map.lead_day == idx

        # Verify subdivision prediction metrics
        for sub in frame.subdivisions:
            assert 0.0 <= sub.p_bust <= 1.0
            assert sub.effective_lead_time_days >= 0.0
            assert sub.nwp_rainfall >= 0.0
            assert sub.confidence_band in ConfidenceBand
            assert sub.error_direction in BustDirection


def test_get_replay_scenario_case2_july2001_depression():
    """Verify Case 2: July 2001 Active Monsoon Depression frames and metrics."""
    scenario = get_replay_scenario("july2001_depression", max_lead=7)
    assert isinstance(scenario, ReplayScenario)
    assert scenario.scenario_id == "july2001_depression"
    assert scenario.cycle_date == "2001-07-01"
    assert len(scenario.frames) == 7

    # Verify frame progression and dynamic captions
    for idx, frame in enumerate(scenario.frames, start=1):
        assert frame.lead_day == idx
        assert frame.cycle_date == "2001-07-01"
        assert len(frame.subdivisions) == 36
        assert f"Day {idx} lead:" in frame.caption_strip
        assert "%" in frame.caption_strip
        assert "mm" in frame.caption_strip


def test_get_replay_scenario_aliases():
    """Verify case1 and case2 aliases route to canonical scenarios."""
    s1 = get_replay_scenario("case1", max_lead=3)
    assert s1.scenario_id == "july2002_break"
    assert len(s1.frames) == 3

    s2 = get_replay_scenario("case2", max_lead=3)
    assert s2.scenario_id == "july2001_depression"
    assert len(s2.frames) == 3


def test_get_replay_scenario_invalid_id():
    """Verify ValueError is raised for unrecognized scenario ID."""
    with pytest.raises(ValueError, match="Unknown scenario_id"):
        get_replay_scenario("kerala_2018_invalid")


def test_synthesize_caption_strip():
    """Verify that caption strip is synthesized strictly from model numbers."""
    # Test dry bust scenario
    caption_dry = synthesize_caption_strip(
        lead_day=5,
        top_region_name="West Madhya Pradesh",
        p_bust=0.78,
        confidence_band=ConfidenceBand.LOW,
        effective_lead_time=9.2,
        nwp_rainfall=18.2,
        obs_rainfall=1.4,
        direction=BustDirection.DRY,
        ens_spread=2.8,
        scenario_id="july2002_break",
    )
    assert "Day 5 lead:" in caption_dry
    assert "18.2 mm/day" in caption_dry
    assert "78% probability of dry bust" in caption_dry
    assert "(actual: 1.4 mm/day)" in caption_dry
    assert "Day 9.2" in caption_dry
    assert "LOW" in caption_dry

    # Test wet bust scenario
    caption_wet = synthesize_caption_strip(
        lead_day=3,
        top_region_name="Kerala",
        p_bust=0.65,
        confidence_band=ConfidenceBand.CRITICAL,
        effective_lead_time=8.5,
        nwp_rainfall=25.0,
        obs_rainfall=88.0,
        direction=BustDirection.WET,
        ens_spread=3.1,
        scenario_id="july2001_depression",
    )
    assert "Day 3 lead:" in caption_wet
    assert "wet bust surge" in caption_wet
    assert "65%" in caption_wet
    assert "CRITICAL" in caption_wet

    # Test stable low-risk scenario
    caption_stable = synthesize_caption_strip(
        lead_day=1,
        top_region_name="Punjab",
        p_bust=0.03,
        confidence_band=ConfidenceBand.HIGH,
        effective_lead_time=1.1,
        nwp_rainfall=5.0,
        obs_rainfall=4.8,
        direction=BustDirection.NONE,
        ens_spread=0.8,
        scenario_id="july2002_break",
    )
    assert "Day 1 lead:" in caption_stable
    assert "stable monsoon" in caption_stable
    assert "HIGH" in caption_stable


def test_api_demo_replay_scenarios_endpoint(api_client: TestClient):
    """Verify GET /api/demo/replay/scenarios endpoint."""
    response = api_client.get("/api/demo/replay/scenarios")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["id"] == "july2002_break"


def test_api_demo_replay_endpoint(api_client: TestClient):
    """Verify GET /api/demo/replay returns full ReplayScenario."""
    response = api_client.get("/api/demo/replay?scenario_id=july2002_break&max_lead=5")
    assert response.status_code == 200
    data = response.json()
    assert data["scenario_id"] == "july2002_break"
    assert len(data["frames"]) == 5
    assert data["frames"][0]["lead_day"] == 1
    assert len(data["frames"][0]["subdivisions"]) == 36
    assert len(data["frames"][0]["caption_strip"]) > 0


def test_api_demo_replay_invalid_endpoint(api_client: TestClient):
    """Verify GET /api/demo/replay returns HTTP 404 for invalid scenario."""
    response = api_client.get("/api/demo/replay?scenario_id=unknown_scenario_xyz")
    assert response.status_code == 404
    assert "Unknown scenario_id" in response.json()["detail"]
