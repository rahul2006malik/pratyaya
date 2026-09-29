"""Pydantic V2 Schemas for Pratyaya API (Node F13).

Defines the frozen API data contract per docs/04-data-and-api.md §2.5, §3
and docs/06-design-ux.md §1–§4.
Guarantees strict schema validation for dashboard consumption.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field

CANONICAL_MODEL_VERSION = "pratyaya-v1.0.0-loyo"


class ConfidenceBand(str, Enum):
    """Categorical risk confidence bands per docs/04 §2.5 and graph.json CONFIDENCE_BANDS.

    HIGH: Low bust probability (high confidence in forecast)
    MODERATE: Elevated bust probability
    LOW: High bust probability (low confidence in forecast)
    CRITICAL: Severe bust risk (forecast likely badly wrong)
    """

    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    CRITICAL = "CRITICAL"


class BustDirection(str, Enum):
    """Direction of potential bust per docs/04 §2.4 and docs/06 §1.1."""

    WET = "wet"
    DRY = "dry"
    NONE = "none"


# Exact threshold boundaries for confidence band categorization
# Guaranteed by unit tests in tests/test_api_schemas.py to avoid off-by-one errors
BAND_THRESHOLDS: Dict[str, float] = {
    "HIGH_MAX": 0.10,      # p_bust < 0.10 -> HIGH
    "MODERATE_MAX": 0.25,  # 0.10 <= p_bust < 0.25 -> MODERATE
    "LOW_MAX": 0.50,       # 0.25 <= p_bust < 0.50 -> LOW
}                          # p_bust >= 0.50 -> CRITICAL

# Empirical observed bust rates in calibration set per band
BAND_OBSERVED_BUST_RATES: Dict[ConfidenceBand, float] = {
    ConfidenceBand.HIGH: 0.015,
    ConfidenceBand.MODERATE: 0.082,
    ConfidenceBand.LOW: 0.285,
    ConfidenceBand.CRITICAL: 0.584,
}


def classify_confidence_band(p_bust: float) -> ConfidenceBand:
    """Classify calibrated bust probability into confidence band.

    Args:
        p_bust: Calibrated bust probability in [0.0, 1.0].

    Returns:
        ConfidenceBand: Categorical band enum.
    """
    if p_bust < BAND_THRESHOLDS["HIGH_MAX"]:
        return ConfidenceBand.HIGH
    elif p_bust < BAND_THRESHOLDS["MODERATE_MAX"]:
        return ConfidenceBand.MODERATE
    elif p_bust < BAND_THRESHOLDS["LOW_MAX"]:
        return ConfidenceBand.LOW
    else:
        return ConfidenceBand.CRITICAL


class AnalogueItem(BaseModel):
    """Historical analogue retrieved from synoptic state history per docs/06 §1.2."""

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
    mechanism: Optional[str] = Field(None, description="Synoptic circulation diagnostic or physical mechanism")


class ForecastCard(BaseModel):
    """Region detail forecast explanation payload per docs/04 §2.5 and docs/06 §1–§4."""

    model_config = ConfigDict(extra="forbid")

    effective_lead_time: float = Field(
        ...,
        description="Effective lead time in days clipped to [0.5, 12.0] per docs/06 §1.1",
    )
    p_bust: float = Field(..., ge=0.0, le=1.0, description="Calibrated fused bust probability")
    p_bust_p90: float = Field(..., ge=0.0, le=1.0, description="Secondary 90th percentile error bust probability")
    confidence_band: ConfidenceBand = Field(..., description="Confidence band: HIGH, MODERATE, LOW, CRITICAL")
    top_analogues: List[AnalogueItem] = Field(..., description="Top 8 historical dates with outcomes")
    risk_factors: List[str] = Field(..., description="Supporting plain-language meteorological risk factors")
    bust_direction: BustDirection = Field(..., description="Direction of potential bust: wet, dry, or none")
    unprecedented_flag: bool = Field(
        ...,
        description="True if synoptic distance exceeds 99th percentile threshold (OOD refusal)",
    )

    # Context and envelope metadata
    region_id: Optional[str] = Field(None, description="IMD subdivision ID (1-36)")
    region_name: Optional[str] = Field(None, description="Subdivision name")
    lead_day: Optional[int] = Field(None, ge=1, le=10, description="Forecast lead day (1 to 10)")
    init_date: Optional[str] = Field(None, description="Forecast initialization date (YYYY-MM-DD)")
    headline: Optional[str] = Field(None, description="Forecaster headline sentence per docs/06 §1.1")
    observed_bust_rate_in_band: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Measured observed bust rate for this band in validation set per docs/04 §3",
    )
    model_version: str = Field(CANONICAL_MODEL_VERSION, description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(False, description="Whether data is from live ECMWF cycle")


class SubdivisionForecast(BaseModel):
    """Forecast summary for a single meteorological subdivision."""

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
    """National confidence choropleth map payload per docs/04 §3 and docs/06 §2."""

    model_config = ConfigDict(extra="forbid")

    lead_day: int = Field(..., ge=1, le=10, description="Forecast lead day (1 to 10)")
    date: str = Field(..., description="Forecast initialization date (YYYY-MM-DD)")
    regions: List[SubdivisionForecast] = Field(
        ...,
        description="Forecasts for all 36 subdivisions of India",
    )
    variable: str = Field("rainfall", description="Target variable (rainfall or t2m)")
    model_version: str = Field(CANONICAL_MODEL_VERSION, description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(False, description="Whether data is live")


class QueueItem(BaseModel):
    """Ranked triage item in error-prone areas review queue."""

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
    """Error-prone areas triage queue sorted by p_bust * decision_weight per docs/04 §3."""

    model_config = ConfigDict(extra="forbid")

    date: str = Field(..., description="Forecast initialization date (YYYY-MM-DD)")
    lead_day: Optional[int] = Field(None, ge=1, le=10, description="Filtered lead day if specified")
    total_items: int = Field(..., ge=0, description="Total items in queue")
    queue: List[QueueItem] = Field(
        ...,
        description="Ranked triage queue sorted by p_bust * decision_weight descending",
    )
    model_version: str = Field(CANONICAL_MODEL_VERSION, description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(False, description="Whether data is live")


class ModelStatus(BaseModel):
    """Live model status and verified held-out evaluation metrics per docs/04 §3."""

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
    benchmark_ladder: Optional[List[BenchmarkEntry]] = None
    n_independent_events: Optional[int] = None
    ood_refusal_rate: Optional[float] = None


class SubdivisionReplayPrediction(BaseModel):
    """Subdivision forecast and verification in a replay frame."""

    model_config = ConfigDict(extra="forbid")

    region_id: str = Field(..., description="Subdivision ID (1-36)")
    name: str = Field(..., description="Subdivision name")
    p_bust: float = Field(..., ge=0.0, le=1.0, description="Calibrated bust probability")
    confidence_band: ConfidenceBand = Field(..., description="Confidence band: HIGH, MODERATE, LOW, CRITICAL")
    nwp_rainfall: float = Field(..., ge=0.0, description="NWP forecast rainfall in mm/day")
    obs_rainfall: Optional[float] = Field(None, ge=0.0, description="Observed rainfall in mm/day")
    effective_lead_time_days: float = Field(..., description="Effective lead time in days")
    error_direction: BustDirection = Field(..., description="Bust direction: wet, dry, or none")
    unprecedented_flag: bool = Field(False, description="Whether synoptic flow is OOD")
    p_bust_p90: Optional[float] = Field(None, ge=0.0, le=1.0, description="Secondary P90 bust probability")


class ReplayFrame(BaseModel):
    """A single lead-day frame in a replay demonstration sequence."""

    model_config = ConfigDict(extra="forbid")

    cycle_date: str = Field(..., description="Model initialization/cycle date (YYYY-MM-DD)")
    lead_day: int = Field(..., ge=1, le=10, description="Forecast lead day (1 to 10)")
    valid_date: str = Field(..., description="Forecast target valid date (YYYY-MM-DD)")
    subdivisions: List[SubdivisionReplayPrediction] = Field(
        ...,
        description="Predictions for all 36 subdivisions",
    )
    caption_strip: str = Field(
        ...,
        description="Dynamic meteorological caption synthesized strictly from model metrics",
    )
    triage_queue: List[QueueItem] = Field(
        ...,
        description="Top high-risk subdivisions sorted by priority score",
    )
    confidence_map: ConfidenceMap = Field(
        ...,
        description="National confidence choropleth for this lead day",
    )
    top_risk_region_id: str = Field(..., description="ID of region with highest priority score")
    top_risk_region_name: str = Field(..., description="Name of highest risk region")
    max_p_bust: float = Field(..., ge=0.0, le=1.0, description="Highest bust probability in frame")
    mean_effective_lead_time: float = Field(..., description="National mean effective lead time")


class ReplayScenario(BaseModel):
    """Full demonstration scenario containing progressive lead-day frames."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str = Field(..., description="Scenario identifier (e.g. july2002_break)")
    title: str = Field(..., description="Scenario title")
    description: str = Field(..., description="Scenario meteorological description")
    cycle_date: str = Field(..., description="Initialization cycle date (YYYY-MM-DD)")
    lead_days: List[int] = Field(..., description="List of lead days included in scenario")
    frames: List[ReplayFrame] = Field(..., description="Progressive sequence of replay frames")
    model_version: str = Field(CANONICAL_MODEL_VERSION, description="Model version")
    is_live: bool = Field(True, description="Whether generated from live model engine")


class ScenarioSummary(BaseModel):
    """Catalog summary for an available demonstration scenario."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Scenario ID")
    title: str = Field(..., description="Scenario display title")
    description: str = Field(..., description="Short meteorological summary")
    cycle_date: str = Field(..., description="Cycle initialization date (YYYY-MM-DD)")
    date_range: str = Field(..., description="Date range span")
    leads: str = Field(..., description="Lead day span")


class EnrichmentStats(BaseModel):
    """External validation enrichment statistics against an independent catalogue."""

    model_config = ConfigDict(extra="forbid")

    catalogue_name: str = Field(..., description="Name of independent meteorological catalogue")
    observed_count: int = Field(..., description="Observed events matching catalogue")
    expected_count: float = Field(..., description="Expected events under baseline")
    enrichment_ratio: float = Field(..., description="E = Observed / Expected")
    ci_95: List[float] = Field(..., description="95% bootstrap confidence interval [lower, upper]")
    chi2_stat: float = Field(..., description="Chi-squared or Fisher statistic")
    p_value: float = Field(..., description="Significance p-value")
    is_significant: bool = Field(..., description="Whether enrichment is statistically significant (p < 0.05)")


class RegimeClusterSummary(BaseModel):
    """Full operational and scientific summary of a discovered synoptic precursor regime."""

    model_config = ConfigDict(extra="forbid")

    cluster_id: int = Field(..., description="Cluster index (0 to k-1)")
    name: str = Field(..., description="Named meteorological mechanism")
    mechanism_summary: str = Field(..., description="Detailed physical mechanism summary")
    n_members: int = Field(..., description="Number of high-burden member dates in cluster")
    member_dates: List[str] = Field(..., description="List of historical member dates (YYYY-MM-DD)")
    seasonality: Dict[str, float] = Field(..., description="Proportion of events across JJAS months")
    threatened_subdivisions: List[Dict[str, Any]] = Field(..., description="Top subdivisions with highest bust rates")
    vulnerable_leads: List[int] = Field(..., description="Forecast leads (Days 3-7) most susceptible to busts")
    signed_bias: str = Field(..., description="Dominant error bias: 'wet' (underforecast) or 'dry' (overforecast)")
    mean_error_mm: float = Field(..., description="Mean signed error across cluster in mm")
    composite_summary: Dict[str, Dict[str, float]] = Field(..., description="Spatial summary (min, max, mean) for synoptic fields")
    validation_status: str = Field(..., description="Status: 'validated' or 'unvalidated_exploratory'")
    is_validated: bool = Field(..., description="Whether cluster meets external validation significance")
    validation: List[EnrichmentStats] = Field(default_factory=list, description="Catalogue enrichment test results")
    n_independent_events: Optional[int] = Field(None, description="Number of independent events")


class RegimesResponse(BaseModel):
    """Full payload for GET /api/regimes per Node G7, docs/03 §8, and docs/06 §2."""

    model_config = ConfigDict(extra="ignore")

    optimal_k: int = Field(..., description="Objective optimal cluster count determined by silhouette score")
    silhouette_scores: Dict[str, float] = Field(..., description="Silhouette scores across candidate k in [3, 8]")
    silhouette_score: Optional[float] = Field(None, description="Silhouette score of optimal k")
    silhouette_ladder: Optional[Dict[str, float]] = Field(None, description="Silhouette ladder across candidate k")
    clusters: List[RegimeClusterSummary] = Field(..., description="Discovered synoptic precursor regimes")
    n_high_burden_dates: int = Field(..., description="Total high-bust-burden dates evaluated")
    n_high_burden_days: Optional[int] = Field(None, description="Total high-bust-burden days evaluated")
    high_burden_threshold: float = Field(..., description="National bust burden cutoff threshold")
    total_candidate_dates: int = Field(..., description="Total training daily synoptic dates")
    model_version: str = Field(CANONICAL_MODEL_VERSION, description="Model version")
    data_as_of: Optional[str] = Field(None, description="Data timestamp")
    is_live: bool = Field(True, description="Whether generated from live model engine")


class BenchmarkEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    name: str
    type: str
    pr_auc: float
    pr_auc_ci_lower: float
    pr_auc_ci_upper: float
    auroc: float
    brier: float
    reliability: float
    bss_vs_b1: float
    role: str
    badge: str
    badge_class: str
    pr_auc_by_lead: Optional[List[float]] = None


ModelStatus.model_rebuild()

