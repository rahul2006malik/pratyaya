# API Service & Data Contracts — Feature Deep-Dive (Node F13)

Implementation notes for the read-only FastAPI service, Pydantic V2 data contracts, and realistic mock server for Pratyaya per `docs/04-data-and-api.md` §2.5, §3 and `docs/06-design-ux.md` §1–§4.

---

## 1. Architecture & Boundaries

The API service (`src/pratyaya/api/main.py`) acts as the frozen contract boundary between the backend Python ML/eval pipeline and the offline React TypeScript dashboard (`dashboard/`).

- **Protocol**: Read-only HTTP REST.
- **Contract Freeze**: Pydantic V2 models enforce strict schemas with `extra="forbid"`.
- **Offline & Mock Mode**: Operates automatically in mock mode when `models/current.json` is not present, or when invoked with `--mock` / `PRATYAYA_MOCK=1`.
- **CORS Support**: Permissive CORS for localhost development, explicitly enabling Vite dashboard on `http://localhost:5173`.
- **Implementation Pointer**: See [main.py](file:///c:/WORK/pratyaya/src/pratyaya/api/main.py#L42-L188) and [mock_provider.py](file:///c:/WORK/pratyaya/src/pratyaya/api/mock_provider.py#L86-L288).

---

## 2. Pydantic V2 Data Contracts

The complete schemas defined in [schemas.py](file:///c:/WORK/pratyaya/src/pratyaya/api/schemas.py#L15-L210) are pasted in full below:

```python
class ConfidenceBand(str, Enum):
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


class BustDirection(str, Enum):
    WET = "wet"
    DRY = "dry"
    NONE = "none"


BAND_THRESHOLDS: Dict[str, float] = {
    "HIGH_MAX": 0.10,      # p_bust < 0.10 -> HIGH
    "MODERATE_MAX": 0.25,  # 0.10 <= p_bust < 0.25 -> MODERATE
    "LOW_MAX": 0.50,       # 0.25 <= p_bust < 0.50 -> LOW
}                          # p_bust >= 0.50 -> CRITICAL


class AnalogueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(..., description="Historical analogue initialization date (YYYY-MM-DD)")
    weight: float = Field(..., ge=0.0, le=1.0, description="Kernel similarity weight in (0, 1]")
    outcome: str = Field(..., description="Historical verification outcome (e.g. 'BUST (wet)', 'no bust')")
    observed_value: Optional[float] = Field(None, description="Observed rainfall/value in mm")
    forecast_value: Optional[float] = Field(None, description="Forecast rainfall/value in mm")
    distance: Optional[float] = Field(None, ge=0.0, description="Mahalanobis distance in synoptic feature space")
    bust_direction: Optional[BustDirection] = Field(None, description="Bust direction if bust occurred")
    is_bust: Optional[bool] = Field(None, description="Whether the analogue forecast busted")
    closeness_rank: Optional[int] = Field(None, ge=1, le=40, description="Rank among analogues (1=closest)")


class ForecastCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effective_lead_time: float = Field(..., description="Effective lead time in days clipped to [0.5, 12.0]")
    p_bust: float = Field(..., ge=0.0, le=1.0, description="Calibrated fused bust probability")
    p_bust_p90: float = Field(..., ge=0.0, le=1.0, description="Secondary 90th percentile error bust probability")
    confidence_band: ConfidenceBand = Field(..., description="Confidence band: HIGH, MODERATE, LOW, CRITICAL")
    top_analogues: List[AnalogueItem] = Field(..., description="Top 8 historical dates with outcomes")
    risk_factors: List[str] = Field(..., description="Supporting plain-language meteorological risk factors")
    bust_direction: BustDirection = Field(..., description="Direction of potential bust: wet, dry, or none")
    unprecedented_flag: bool = Field(..., description="True if synoptic distance exceeds 99th percentile threshold")
    region_id: Optional[str] = Field(None, description="IMD subdivision ID (1-36)")
    region_name: Optional[str] = Field(None, description="Subdivision name")
    lead_day: Optional[int] = Field(None, ge=1, le=10, description="Forecast lead day (1 to 10)")
    init_date: Optional[str] = Field(None, description="Forecast initialization date (YYYY-MM-DD)")
    headline: Optional[str] = Field(None, description="Forecaster headline sentence")
    observed_bust_rate_in_band: Optional[float] = Field(None, ge=0.0, le=1.0)
    model_version: str = Field("pratyaya-v0.1.0-mock", description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(False, description="Whether data is from live ECMWF cycle")


class SubdivisionForecast(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(..., description="IMD subdivision ID (1-36)")
    name: str = Field(..., description="Subdivision name")
    effective_lead_time: float = Field(..., description="Effective lead time in days")
    p_bust: float = Field(..., ge=0.0, le=1.0, description="Calibrated bust probability in [0, 1]")
    confidence_band: ConfidenceBand = Field(..., description="Confidence band: HIGH, MODERATE, LOW, CRITICAL")
    p_bust_p90: Optional[float] = Field(None, ge=0.0, le=1.0, description="Secondary 90th percentile probability")
    bust_direction: Optional[BustDirection] = Field(BustDirection.NONE, description="Bust direction")
    unprecedented_flag: bool = Field(False, description="OOD flag")


class ConfidenceMap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lead_day: int = Field(..., ge=1, le=10, description="Forecast lead day (1 to 10)")
    date: str = Field(..., description="Forecast initialization date (YYYY-MM-DD)")
    regions: List[SubdivisionForecast] = Field(..., description="Forecasts for all 36 subdivisions of India")
    variable: str = Field("rainfall", description="Target variable (rainfall or t2m)")
    model_version: str = Field("pratyaya-v0.1.0-mock", description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(False, description="Whether data is live")


class QueueItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rank: int = Field(..., ge=1, description="Rank in triage queue (1-based)")
    region_id: str = Field(..., description="Subdivision region ID (1-36)")
    region_name: str = Field(..., description="Subdivision name")
    lead_day: int = Field(..., ge=1, le=10, description="Forecast lead day (1 to 10)")
    p_bust: float = Field(..., ge=0.0, le=1.0, description="Calibrated bust probability")
    decision_weight: float = Field(..., gt=0.0, description="Operational decision-cost weight")
    priority_score: float = Field(..., ge=0.0, description="p_bust * decision_weight")
    effective_lead_time: float = Field(..., description="Effective lead time in days")
    confidence_band: ConfidenceBand = Field(..., description="Confidence band")
    bust_direction: BustDirection = Field(..., description="Expected bust direction")
    top_risk_factor: Optional[str] = Field(None, description="Leading meteorological risk reason")


class ErrorProneAreas(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: str = Field(..., description="Forecast initialization date (YYYY-MM-DD)")
    lead_day: Optional[int] = Field(None, ge=1, le=10, description="Filtered lead day if specified")
    total_items: int = Field(..., ge=0, description="Total items in queue")
    queue: List[QueueItem] = Field(..., description="Ranked triage queue sorted by p_bust * decision_weight descending")
    model_version: str = Field("pratyaya-v0.1.0-mock", description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(False, description="Whether data is live")


class ModelStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(..., description="Model version identifier")
    run_id: str = Field(..., description="Artefact run ID in models/")
    heldout_metrics: Dict[str, Any] = Field(..., description="Verified metric table and benchmark ladder")
    calibration_date: str = Field(..., description="Date of last calibration run (YYYY-MM-DD)")
    status: str = Field("ready", description="Model serving status: ready, training, or degraded")
    training_window: str = Field("2000-2014", description="Training period")
    calibration_window: str = Field("2015-2017", description="Calibration period")
    heldout_window: str = Field("2018-2019", description="Held-out evaluation period")
    baselines_available: List[str] = Field(default_factory=lambda: ["B0", "B1", "B2", "B3", "B4", "B5"])
    is_live: bool = Field(False, description="Whether live real-time cycle is active")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
```

---

## 3. Endpoints & Route Definitions

All endpoints are served from [src/pratyaya/api/main.py](file:///c:/WORK/pratyaya/src/pratyaya/api/main.py):

| Method | Route | Description | Query / Path Parameters | Output Schema |
|---|---|---|---|---|
| `GET` | `/api/model/status` | Model version and live verified evaluation metrics | None | `ModelStatus` |
| `GET` | `/api/forecast/map` | National choropleth map across all 36 subdivisions | `lead` (1..10), `date` (YYYY-MM-DD), `variable` | `ConfidenceMap` |
| `GET` | `/api/forecast/region/{region_id}` | Forecast explanation card with 8 analogues | `region_id` (path: 1..36), `date`, `lead` | `ForecastCard` |
| `GET` | `/api/forecast/queue` | Ranked triage queue sorted by priority score | `lead` (optional 1..10), `date`, `top_n` | `ErrorProneAreas` |
| `GET` | `/api/confidence-map` | Alias for `/api/forecast/map` | `lead_day`, `init_date`, `variable` | `ConfidenceMap` |
| `GET` | `/api/error-prone-areas` | Alias for `/api/forecast/queue` | `init_date`, `top_n` | `ErrorProneAreas` |

---

## 4. Operational Commands

- **Launch server with automatic mock fallback**:
  ```bash
  uvicorn pratyaya.api.main:app --host 0.0.0.0 --port 8000 --reload
  ```
  or:
  ```bash
  python -m pratyaya.api.main --mock --port 8000
  ```
  or:
  ```bash
  make serve
  ```

---

### Gotchas

- **Risk R-7 (Frontend/Pipeline Divergence)**: All schemas use Pydantic V2 with `extra="forbid"`. Any field addition or alteration requires an explicit contract update and must be verified against `tests/test_api_schemas.py` before committing. See `docs/08-risks-and-kill-criteria.md` §R-7.
- **Strict 36 Subdivision Invariant**: The national confidence map (`/api/forecast/map`) MUST return exactly 36 subdivisions. Never return a partial list or skip offshore subdivisions (Lakshadweep, A & N Island).
- **Historical Analogue Isolation**: Analogue dates served by the mock provider or retrieval engine MUST be sampled strictly from training and calibration years ($2000 \le \text{year} \le 2017$). Touching held-out years ($2018–2019$) violates Decision `D-000` and `docs/HELDOUT_LOG.md`.
- **Confidence Band Boundary Consistency**: Off-by-one errors in confidence bands alter operational alerting. The boundary conditions ($p < 0.10 \to \text{HIGH}$, $0.10 \le p < 0.25 \to \text{MODERATE}$, $0.25 \le p < 0.50 \to \text{LOW}$, $p \ge 0.50 \to \text{CRITICAL}$) are fixed constants covered by regression unit tests.
- **Triage Queue Priority Score**: Error-prone areas are sorted by `priority_score = p_bust * decision_weight` descending, NEVER by `p_bust` alone.
