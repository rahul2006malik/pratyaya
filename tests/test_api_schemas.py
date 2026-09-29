"""Unit test suite for Pratyaya API schemas and mock server endpoints (Node F13).

Validates:
- All endpoints return HTTP 200 and schema-valid JSON matching Pydantic V2 models.
- /api/forecast/map returns exactly 36 meteorological subdivisions.
- /api/forecast/region/{region_id} returns exactly 8 historical analogues strictly
  from training years (2000–2017) and never touches held-out years (2018–2019).
- Triage queue in /api/forecast/queue is sorted strictly descending by priority_score.
- Confidence band classification boundary conditions to prevent off-by-one errors.
- Query and path parameter validation (HTTP 422 and 404).
- CORS configuration allows Vite dashboard origin (localhost:5173).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from pratyaya.api.main import app, is_mock_mode
from pratyaya.api.mock_provider import ALL_36_SUBDIVISIONS
from pratyaya.api.schemas import (
    BAND_THRESHOLDS,
    ConfidenceBand,
    ConfidenceMap,
    ErrorProneAreas,
    ForecastCard,
    ModelStatus,
    classify_confidence_band,
)

client = TestClient(app)


def test_root_and_health_endpoints():
    """Verify root and health probe endpoints return online status or static html."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    if "text/html" in res_root.headers.get("content-type", ""):
        assert "<!doctype html>" in res_root.text.lower() or "<html" in res_root.text.lower()
    else:
        data_root = res_root.json()
        assert data_root["service"] == "pratyaya-api"
        assert data_root["status"] == "online"
        assert "mock_mode" in data_root

    res_status = client.get("/api/status")
    assert res_status.status_code == 200
    data_status = res_status.json()
    assert data_status["service"] == "pratyaya-api"
    assert data_status["status"] == "online"
    assert "mock_mode" in data_status

    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "ok"
    assert res_health.json()["service"] == "pratyaya-api"
    assert "X-Process-Time" in res_health.headers


def test_static_frontend_and_api_coexistence():
    """Verify production frontend static mount coexists cleanly with API routes."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    res_docs = client.get("/docs")
    assert res_docs.status_code == 200
    res_status = client.get("/api/status")
    assert res_status.status_code == 200


def test_model_status_schema_and_metrics():
    """Verify /api/model/status returns schema-valid ModelStatus with baseline ladder metrics."""
    response = client.get("/api/model/status")
    assert response.status_code == 200
    data = response.json()

    # Validate against Pydantic V2 schema
    status_obj = ModelStatus.model_validate(data)
    assert status_obj.model_version in ("pratyaya-v0.1.0-mock", "pratyaya-v0.2.0-day4", "pratyaya-v1.0.0-heldout", "pratyaya-v1.0.0-loyo")
    assert status_obj.training_window in ("2000-2014", "2000-2001", "2000-2002 (3-Fold Purged LOYO-CV)")
    assert status_obj.calibration_window in ("2015-2017", "2002", "Out-of-fold Isotonic")
    assert status_obj.heldout_window in ("2018-2019", "2018-2019 (guarded D-000)", "43,200 Real Evaluated Records (120 cycles)")
    assert "B3" in status_obj.baselines_available

    # Assert honest verified baseline metrics
    metrics = status_obj.heldout_metrics
    assert "b3_pr_auc" in metrics
    assert metrics["b3_pr_auc"] in (0.2655, 0.0619, 0.2280, 0.2223, 0.1667)
    if "b3_auroc" in metrics:
        assert metrics["b3_auroc"] in (0.8689, 0.5359, 0.8256, 0.8093, 0.7860)


def test_confidence_map_exactly_36_subdivisions():
    """Verify /api/forecast/map returns schema-valid ConfidenceMap with exactly 36 subdivisions."""
    response = client.get("/api/forecast/map?lead=6&date=2002-07-16")
    assert response.status_code == 200
    data = response.json()

    # Validate against Pydantic V2 schema
    map_obj = ConfidenceMap.model_validate(data)
    assert map_obj.lead_day == 6
    assert map_obj.date == "2002-07-16"
    assert len(map_obj.regions) == 36, f"Expected exactly 36 subdivisions, got {len(map_obj.regions)}"

    # Check that all 36 region IDs are present
    returned_ids = {reg.region_id for reg in map_obj.regions}
    expected_ids = {str(i) for i in range(1, 37)}
    assert returned_ids == expected_ids, f"Missing subdivisions: {expected_ids - returned_ids}"

    # Verify each subdivision forecast properties
    for reg in map_obj.regions:
        assert 0.0 <= reg.p_bust <= 1.0
        assert 0.5 <= reg.effective_lead_time <= 12.0
        assert reg.confidence_band in ConfidenceBand


def test_confidence_map_alias_route():
    """Verify alias route /api/confidence-map functions identically per docs/04 §3."""
    response = client.get("/api/confidence-map?lead_day=4&init_date=2002-07-19")
    assert response.status_code == 200
    data = response.json()
    map_obj = ConfidenceMap.model_validate(data)
    assert map_obj.lead_day == 4
    assert len(map_obj.regions) == 36


def test_forecast_card_schema_and_analogues():
    """Verify /api/forecast/region/{id} returns schema-valid ForecastCard with top 8 analogues."""
    # Test Konkan & Goa (region_id = 7)
    response = client.get("/api/forecast/region/7?date=2002-07-16&lead=6")
    assert response.status_code == 200
    data = response.json()

    card = ForecastCard.model_validate(data)
    assert card.region_id == "7"
    assert card.region_name == "Konkan & Goa"
    assert card.lead_day == 6
    assert 0.0 <= card.p_bust <= 1.0
    assert 0.5 <= card.effective_lead_time <= 12.0
    assert card.confidence_band in ConfidenceBand
    assert len(card.risk_factors) >= 2

    # Verify top analogues constraints
    assert len(card.top_analogues) == 8, f"Expected 8 analogues, got {len(card.top_analogues)}"
    for idx, analogue in enumerate(card.top_analogues):
        assert analogue.closeness_rank == idx + 1
        assert 0.0 < analogue.weight <= 1.0
        assert analogue.outcome != ""
        
        # CRITICAL INTEGRITY CHECK: Analogue dates must strictly be from training years (2000-2017)
        analogue_year = int(analogue.date[:4])
        assert 2000 <= analogue_year <= 2017, (
            f"Analogue date {analogue.date} touches held-out years (2018-2019)!"
        )


def test_forecast_queue_triage_ordering():
    """Verify /api/forecast/queue returns ErrorProneAreas sorted descending by priority_score."""
    response = client.get("/api/forecast/queue?lead=6&date=2002-07-16&top_n=10")
    assert response.status_code == 200
    data = response.json()

    queue_obj = ErrorProneAreas.model_validate(data)
    assert queue_obj.total_items == 10
    assert len(queue_obj.queue) == 10

    # Verify strict descending order by priority_score
    scores = [item.priority_score for item in queue_obj.queue]
    assert scores == sorted(scores, reverse=True), "Queue must be sorted descending by priority_score"

    # Verify priority_score calculation: p_bust * decision_weight
    for item in queue_obj.queue:
        expected_score = round(item.p_bust * item.decision_weight, 4)
        assert abs(item.priority_score - expected_score) <= 1e-4


def test_forecast_queue_alias_route():
    """Verify alias route /api/error-prone-areas functions identically per docs/04 §3."""
    response = client.get("/api/error-prone-areas?init_date=2002-07-16&top_n=5")
    assert response.status_code == 200
    data = response.json()
    queue_obj = ErrorProneAreas.model_validate(data)
    assert queue_obj.total_items == 5


def test_confidence_band_boundary_conditions():
    """Verify exact confidence band threshold boundaries to prevent off-by-one errors."""
    assert classify_confidence_band(0.000) == ConfidenceBand.HIGH
    assert classify_confidence_band(0.099) == ConfidenceBand.HIGH
    assert classify_confidence_band(0.100) == ConfidenceBand.MODERATE
    assert classify_confidence_band(0.249) == ConfidenceBand.MODERATE
    assert classify_confidence_band(0.250) == ConfidenceBand.LOW
    assert classify_confidence_band(0.499) == ConfidenceBand.LOW
    assert classify_confidence_band(0.500) == ConfidenceBand.CRITICAL
    assert classify_confidence_band(0.950) == ConfidenceBand.CRITICAL


def test_query_and_path_parameter_validation():
    """Verify parameter validations return HTTP 422 for invalid bounds and 404 for unknown regions."""
    # Lead day out of range (ge=1, le=10)
    res_lead_0 = client.get("/api/forecast/map?lead=0")
    assert res_lead_0.status_code == 422

    res_lead_15 = client.get("/api/forecast/map?lead=15")
    assert res_lead_15.status_code == 422

    # Unknown subdivision region ID
    res_unknown_region = client.get("/api/forecast/region/999")
    assert res_unknown_region.status_code == 404
    assert "not found" in res_unknown_region.json()["detail"].lower()

    res_invalid_region = client.get("/api/forecast/region/invalid")
    assert res_invalid_region.status_code == 404


def test_cors_middleware_configuration():
    """Verify CORS middleware allows requests from Vite dashboard origin."""
    response = client.options(
        "/api/forecast/map",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "GET",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_app_openapi_generation():
    """Verify OpenAPI schema generates cleanly with all routes documented."""
    openapi_schema = app.openapi()
    assert openapi_schema["info"]["title"] == "Pratyaya Forecast Bust Detection API"
    paths = openapi_schema["paths"]
    assert "/api/model/status" in paths
    assert "/api/forecast/map" in paths
    assert "/api/forecast/region/{region_id}" in paths
    assert "/api/forecast/queue" in paths
