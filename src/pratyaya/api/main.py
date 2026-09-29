"""FastAPI application for Pratyaya forecast bust detection (Node F13).

Provides read-only access to national confidence choropleth maps,
error-prone areas review queues, plain-language forecast explanations,
and live model status per docs/04 §3 and docs/06 §2.

Guarantees:
- Strict Pydantic V2 schema validation for all endpoints.
- Automatic mock fallback when model artifacts in models/current.json are absent.
- Full CLI support for --mock and --demo modes.
- Permissive CORS for localhost development with Vite dashboard (localhost:5173).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles

from pratyaya.api.mock_provider import (
    ALL_36_SUBDIVISIONS,
    get_mock_confidence_map,
    get_mock_error_prone_areas,
    get_mock_forecast_card,
    get_mock_model_status,
    get_mock_regimes,
)
from pratyaya.api.schemas import (
    CANONICAL_MODEL_VERSION,
    ConfidenceMap,
    ErrorProneAreas,
    ForecastCard,
    ModelStatus,
    RegimesResponse,
    ReplayScenario,
    ScenarioSummary,
)
from pratyaya.api.replay import (
    get_available_scenarios,
    get_replay_scenario,
)
from pratyaya.models.serving import get_model_server

logger = logging.getLogger("pratyaya.api")

# Determine whether mock mode is enforced via environment or CLI arguments
def is_mock_mode() -> bool:
    """Return True if mock mode is requested or if model artifacts are absent."""
    if os.environ.get("PRATYAYA_MOCK", "").lower() in ("1", "true", "yes"):
        return True
    if "--mock" in sys.argv or "--demo" in sys.argv:
        return True
    current_json = Path("models/current.json")
    if not current_json.exists():
        return True
    return False


# Initialize FastAPI application
app = FastAPI(
    title="Pratyaya Forecast Bust Detection API",
    description=(
        "AI-Based Medium-Range Forecast Bust Detection & Historical Analogue Retrieval "
        "for Meteorological Subdivisions of India (SIH26079)."
    ),
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allowed origins for CORS (including Vite default port 5173 per coordinator guardrail)
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Calculate request execution latency and attach X-Process-Time response header."""
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time"] = f"{duration_ms:.1f}ms"
    return response


VALID_REGION_IDS = {reg_id for reg_id, _ in ALL_36_SUBDIVISIONS}
DIST_DIR = Path(__file__).resolve().parents[3] / "dashboard" / "dist"


@app.get("/api/status", tags=["Health"])
def api_status():
    """API service status and operation mode."""
    return {
        "service": "pratyaya-api",
        "version": "0.1.0",
        "status": "online",
        "mock_mode": is_mock_mode(),
        "documentation": "/docs",
    }


if not (DIST_DIR.exists() and (DIST_DIR / "index.html").exists()):
    @app.get("/", tags=["Health"])
    def root():
        """Root endpoint returning service status and mock operation mode."""
        return api_status()


@app.get("/health", tags=["Health"])
def health_check():
    """Liveness probe returning operational service metadata."""
    now_iso = datetime.now(timezone.utc).isoformat()
    return {
        "status": "ok",
        "service": "pratyaya-api",
        "model_version": CANONICAL_MODEL_VERSION,
        "run_id": "run-loyo-benchmark-2000-2002",
        "data_as_of": now_iso,
    }


@app.get(
    "/api/model/status",
    response_model=ModelStatus,
    summary="Get model version, run status, and verified baseline metrics",
    tags=["Model"],
)
def get_model_status() -> ModelStatus:
    """Return model version, active run ID, and verified metrics table."""
    if not is_mock_mode():
        server = get_model_server()
        if server is not None:
            try:
                return server.get_status()
            except Exception as exc:
                logger.warning("Error fetching real model status: %s. Falling back to mock.", exc)
    return get_mock_model_status()


DEFAULT_CYCLE_DATE = "2002-07-16"


def _validate_cycle_date(date_str: str, allow_ood: bool = False) -> None:
    """Validate that date_str is a verified forecast initialization cycle date in the operational store."""
    try:
        dt_year = int(str(date_str)[:4])
        if allow_ood and (dt_year < 2000 or dt_year > 2002):
            return
    except Exception:
        pass

    if not is_mock_mode():
        server = get_model_server()
        if server is not None:
            if not server.is_cycle_date(date_str):
                nearest = server.get_nearest_cycle_dates(date_str, n=3)
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Date '{date_str}' is not a valid forecast initialization cycle date in the operational store. "
                        f"Operational forecast cycles run every 3 days during JJAS (days 01, 04, 07, 10, 13, 16, 19, 22, 25, 28). "
                        f"Nearest valid cycle dates are: {nearest}."
                    ),
                )


@app.get(
    "/api/forecast/map",
    response_model=ConfidenceMap,
    summary="Get national confidence choropleth map across all 36 subdivisions",
    tags=["Forecast"],
)
def get_forecast_map(
    lead: int = Query(default=6, ge=1, le=10, description="Forecast lead day (1 to 10)"),
    date: str = Query(default=DEFAULT_CYCLE_DATE, description="Forecast init date (YYYY-MM-DD)"),
    variable: str = Query(default="rainfall", description="Forecast variable (rainfall or t2m)"),
) -> ConfidenceMap:
    """Return effective lead time and bust probability for all 36 subdivisions."""
    _validate_cycle_date(date)
    if not is_mock_mode():
        server = get_model_server()
        if server is not None:
            try:
                return server.predict_map(date=date, lead_day=lead, variable=variable)
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                logger.warning("Error computing real confidence map: %s. Falling back to mock.", exc)
    return get_mock_confidence_map(lead_day=lead, date=date, variable=variable)


# Alias route per docs/04 §3
# Bug fix (2026-09-25): Previously only accepted ?lead_day=N; the canonical param is ?lead=N
# (matching /api/forecast/map). Now accepts both; ?lead takes precedence.
@app.get("/api/confidence-map", response_model=ConfidenceMap, include_in_schema=False)
def get_confidence_map_alias(
    lead: Optional[int] = Query(default=None, ge=1, le=10, description="Forecast lead day (canonical, matches /api/forecast/map)"),
    lead_day: Optional[int] = Query(default=None, ge=1, le=10, description="Forecast lead day (legacy alias)"),
    init_date: str = Query(default=DEFAULT_CYCLE_DATE, alias="init_date"),
    variable: str = Query(default="rainfall", alias="variable"),
) -> ConfidenceMap:
    """Alias for /api/forecast/map. Accepts both ?lead=N (canonical) and ?lead_day=N (legacy)."""
    effective_lead = lead if lead is not None else (lead_day if lead_day is not None else 6)
    return get_forecast_map(lead=effective_lead, date=init_date, variable=variable)


@app.get(
    "/api/forecast/region/{region_id}",
    response_model=ForecastCard,
    summary="Get forecast explanation card and top historical analogues for a subdivision",
    tags=["Forecast"],
)
def get_forecast_region(
    region_id: str,
    date: str = Query(default=DEFAULT_CYCLE_DATE, description="Forecast init date (YYYY-MM-DD)"),
    lead: int = Query(default=6, ge=1, le=10, description="Forecast lead day (1 to 10)"),
) -> ForecastCard:
    """Return explanation card with top 8 historical analogues and meteorological risk factors."""
    if region_id not in VALID_REGION_IDS:
        raise HTTPException(
            status_code=404,
            detail=f"Subdivision ID '{region_id}' not found. Valid IDs: 1 to 36.",
        )
    _validate_cycle_date(date, allow_ood=True)
    if not is_mock_mode():
        server = get_model_server()
        if server is not None:
            try:
                return server.predict_region_card(region_id=region_id, date=date, lead_day=lead)
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                logger.warning("Error computing real forecast card for region %s: %s. Falling back to mock.", region_id, exc)
    return get_mock_forecast_card(region_id=region_id, date=date, lead_day=lead)


@app.get(
    "/api/forecast/queue",
    response_model=ErrorProneAreas,
    summary="Get ranked triage queue of error-prone areas sorted by priority score",
    tags=["Forecast"],
)
def get_forecast_queue(
    lead: Optional[int] = Query(default=None, ge=1, le=10, description="Filter by lead day (1 to 10)"),
    date: str = Query(default=DEFAULT_CYCLE_DATE, description="Forecast init date (YYYY-MM-DD)"),
    top_n: int = Query(default=10, ge=1, le=360, description="Number of top ranked items to return"),
) -> ErrorProneAreas:
    """Return ranked review queue sorted descending by priority_score = p_bust * decision_weight."""
    _validate_cycle_date(date)
    if not is_mock_mode():
        server = get_model_server()
        if server is not None:
            try:
                return server.predict_triage_queue(date=date, lead_day=lead, top_n=top_n)
            except HTTPException:
                raise
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc))
            except Exception as exc:
                logger.warning("Error computing real triage queue: %s. Falling back to mock.", exc)
    return get_mock_error_prone_areas(date=date, lead_day=lead, top_n=top_n)


# Alias route per docs/04 §3
@app.get("/api/error-prone-areas", response_model=ErrorProneAreas, include_in_schema=False)
def get_error_prone_areas_alias(
    init_date: str = Query(default=DEFAULT_CYCLE_DATE, alias="init_date"),
    top_n: int = Query(default=10, ge=1, le=360, alias="top_n"),
) -> ErrorProneAreas:
    """Alias for /api/forecast/queue supporting docs/04 query parameters."""
    return get_forecast_queue(lead=None, date=init_date, top_n=top_n)



@app.get(
    "/api/demo/replay/scenarios",
    response_model=List[ScenarioSummary],
    summary="Get catalog of available replay demonstration scenarios",
    tags=["Demo"],
)
def list_replay_scenarios() -> List[ScenarioSummary]:
    """Return available canonical demonstration scenarios."""
    return get_available_scenarios()


@app.get(
    "/api/demo/replay",
    response_model=ReplayScenario,
    summary="Get progressive replay frames and dynamic model-scored captions for demo scenario",
    tags=["Demo"],
)
def get_demo_replay(
    scenario_id: str = Query(
        default="july2002_break",
        description="Scenario identifier: july2002_break (Case 1) or july2001_depression (Case 2)",
    ),
    max_lead: int = Query(
        default=7,
        ge=1,
        le=10,
        description="Maximum lead day to generate (default 7, up to 10)",
    ),
) -> ReplayScenario:
    """Generate progressive Day 1-10 replay frames with model-scored caption strips."""
    try:
        return get_replay_scenario(scenario_id=scenario_id, max_lead=max_lead)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get(
    "/api/regimes",
    response_model=RegimesResponse,
    summary="Get discovered synoptic precursor regimes and external validation statistics",
    tags=["Regimes"],
)
def get_regimes() -> RegimesResponse:
    """Return discovered synoptic regimes, composite anomaly summaries, and external validation statistics.

    Always attempts to read models/regimes_loyo.json directly first — this is the canonical
    artifact (optimal_k=9, silhouette_score=0.0959) produced by scripts/discover_regimes_loyo.py.
    Mock fallback only fires if the artifact is absent or unreadable.
    """
    # Priority 1: Read directly from canonical artifact (independent of mock/live mode)
    regimes_artifact = Path("models/regimes_loyo.json")
    if regimes_artifact.exists():
        try:
            import json as _json
            with open(regimes_artifact, "r", encoding="utf-8") as _f:
                _data = _json.load(_f)
            return RegimesResponse.model_validate(_data)
        except Exception as exc:
            logger.warning("Error reading models/regimes_loyo.json: %s. Trying live server.", exc)

    # Priority 2: Live model server (also reads regimes_loyo.json, but via its own path candidates)
    if not is_mock_mode():
        server = get_model_server()
        if server is not None:
            try:
                return server.get_regimes()
            except Exception as exc:
                logger.warning("Error fetching real regimes: %s. Falling back to mock.", exc)

    # Priority 3: Mock fallback — only if artifact is absent
    return get_mock_regimes()



# Mount compiled production React dashboard at root if present
if DIST_DIR.exists() and (DIST_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(DIST_DIR), html=True), name="static")


def main():
    """CLI entry point for running the API server directly."""
    parser = argparse.ArgumentParser(description="Pratyaya Forecast Bust API Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--mock", action="store_true", help="Enforce realistic mock data generation")
    parser.add_argument("--demo", action="store_true", help="Launch in offline demo mode")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn hot reloading")
    args = parser.parse_args()

    if args.mock or args.demo:
        os.environ["PRATYAYA_MOCK"] = "1"
        print(f"[Pratyaya API] Launching in {'DEMO' if args.demo else 'MOCK'} mode...")

    if args.demo:
        print("[Pratyaya Demo] Open http://localhost:8000 in your browser to view the Mission Control Dashboard & Replay Demo.")

    uvicorn.run("pratyaya.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
