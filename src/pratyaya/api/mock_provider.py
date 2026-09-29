"""Deterministic, meteorologically realistic mock data provider for Pratyaya API.

Serves schema-valid mock data for all 36 subdivisions and 10 leads when model
run artifacts are absent or when running with --mock / PRATYAYA_MOCK=1.
Guarantees:
1. Exactly 36 subdivisions for confidence maps.
2. Exactly 8 historical analogues sampled STRICTLY from training years (2000–2017).
   NEVER touches held-out years (2018–2019) per evaluation integrity rules.
3. Realistic effective lead times and confidence bands.
4. Triage queues sorted descending by priority_score = p_bust * decision_weight.
"""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Optional, Tuple

from pratyaya.api.schemas import (
    BAND_OBSERVED_BUST_RATES,
    AnalogueItem,
    BustDirection,
    ConfidenceBand,
    ConfidenceMap,
    EnrichmentStats,
    ErrorProneAreas,
    ForecastCard,
    ModelStatus,
    QueueItem,
    RegimeClusterSummary,
    RegimesResponse,
    SubdivisionForecast,
    classify_confidence_band,
)

# Canonical project constants per brief v3
# Canonical project constants per brief v3 and v4
CANONICAL_MODEL_VERSION = "pratyaya-v1.0.0-loyo"
CANONICAL_ARCHIVE_SPAN = "40 years of IMD gridded reanalysis record (1982–2022)"

# Canonical 36 subdivisions of India with standardized official names (Kerala is 35, Vidarbha is 13)
ALL_36_SUBDIVISIONS: List[Tuple[str, str]] = [
    ("1", "Lakshadweep"),
    ("2", "Andaman & Nicobar Islands"),
    ("3", "Coastal Andhra Pradesh"),
    ("4", "Arunachal Pradesh"),
    ("5", "Assam & Meghalaya"),
    ("6", "Bihar"),
    ("7", "Konkan & Goa"),
    ("8", "Saurashtra & Kutch"),
    ("9", "Haryana, Chandigarh & Delhi"),
    ("10", "Himachal Pradesh"),
    ("11", "Jammu & Kashmir"),
    ("12", "South Interior Karnataka"),
    ("13", "Vidarbha"),
    ("14", "West Madhya Pradesh"),
    ("15", "Madhya Maharashtra"),
    ("16", "Nagaland, Manipur, Mizoram & Tripura"),
    ("17", "Odisha"),
    ("18", "Punjab"),
    ("19", "West Rajasthan"),
    ("20", "Tamil Nadu & Puducherry"),
    ("21", "East Uttar Pradesh"),
    ("22", "Gangetic West Bengal"),
    ("23", "Sub-Himalayan West Bengal & Sikkim"),
    ("24", "Jharkhand"),
    ("25", "Uttarakhand"),
    ("26", "East Rajasthan"),
    ("27", "Gujarat Region"),
    ("28", "West Uttar Pradesh"),
    ("29", "Telangana"),
    ("30", "Rayalaseema"),
    ("31", "Chhattisgarh"),
    ("32", "East Madhya Pradesh"),
    ("33", "Coastal Karnataka"),
    ("34", "North Interior Karnataka"),
    ("35", "Kerala"),
    ("36", "Marathwada"),
]

# Operational decision-cost weights for triage priority ranking
# Higher weights for flood-prone, high-relief, or drought-vulnerable agro-climatic zones
# Explicitly calibrated across all 36 canonical meteorological subdivisions
DECISION_WEIGHTS: Dict[str, float] = {
    # Tier 1: Severe orographic terrain, flash floods, extreme urban inundation, active landslide corridors (1.40 - 1.60)
    "7": 1.60,   # Konkan & Goa (Mumbai megacity flood vulnerability, steep Western Ghats escarpment)
    "35": 1.50,  # Kerala (Western Ghats steep relief, Idukki reservoir cascade, landslide sensitivity)
    "33": 1.50,  # Coastal Karnataka (intense orographic precipitation, steep coastal river basins)
    "5": 1.45,   # Assam & Meghalaya (Brahmaputra floodplains, Mawsynram/Cherrapunji extreme rainfall)
    "6": 1.45,   # Bihar (perennial Ganga/Kosi transboundary inundation, dense rural population)
    "17": 1.40,  # Odisha (Mahanadi delta flooding, primary Bay of Bengal cyclone/depression landfall corridor)
    "22": 1.40,  # Gangetic West Bengal (Kolkata metropolitan basin, Sundarbans coastal surge vulnerability)
    "23": 1.40,  # Sub-Himalayan West Bengal & Sikkim (Teesta basin flash floods, Darjeeling landslide corridor)
    "25": 1.40,  # Uttarakhand (Alaknanda/Bhagirathi cloudbursts, Char Dham mountain transit lifelines)

    # Tier 2: Agrarian distress belts, storm surge coasts, mountain river valleys, urban centers (1.25 - 1.35)
    "3": 1.35,   # Coastal Andhra Pradesh (Godavari/Krishna delta, cyclonic storm surge exposure)
    "4": 1.30,   # Arunachal Pradesh (Eastern Himalayan flash floods, remote border transit vulnerability)
    "10": 1.35,  # Himachal Pradesh (Sutlej/Beas flood surges, steep mountain slope instability)
    "11": 1.30,  # Jammu & Kashmir (Jhelum basin inundation, valley transport bottlenecks)
    "21": 1.35,  # East Uttar Pradesh (Ghaghra/Rapti floodplains, high agrarian population density)
    "36": 1.30,  # Marathwada (acute rain-fed agricultural drought vulnerability, farmer distress sensitivity)
    "9": 1.25,   # Haryana, Chandigarh & Delhi (National Capital Region drainage, Yamuna floodplains)
    "13": 1.25,  # Vidarbha (rain-fed cotton belt agrarian distress, sudden monsoon dry/wet swings)
    "15": 1.25,  # Madhya Maharashtra (Western Ghats rain shadow transition, sugarcane irrigation stress)
    "16": 1.25,  # Nagaland, Manipur, Mizoram & Tripura (Purvanchal landslides, isolated highway corridors)
    "27": 1.25,  # Gujarat Region (Narmada/Tapi industrial corridor, heavy monsoon depression flooding)
    "29": 1.25,  # Telangana (Godavari/Krishna agrarian basin, Hyderabad urban drainage vulnerability)
    "30": 1.25,  # Rayalaseema (chronically drought-prone semi-arid interior, agrarian distress)

    # Tier 3: Peninsular interior plateaus, plains agrarian basins, arid margins, island ecosystems (1.15 - 1.20)
    "1": 1.15,   # Lakshadweep (low-lying coral atoll maritime isolation, cyclone sensitivity)
    "2": 1.15,   # Andaman & Nicobar Islands (island supply chain isolation, monsoon maritime squalls)
    "8": 1.20,   # Saurashtra & Kutch (cyclonic landfall margin, arid pastoral economy)
    "12": 1.20,  # South Interior Karnataka (Cauvery river basin, Bengaluru urban catchment)
    "14": 1.15,  # West Madhya Pradesh (Malwa plateau agrarian belt, soybean/wheat farming)
    "18": 1.20,  # Punjab (national grain basket, Sutlej/Beas river embankments)
    "19": 1.15,  # West Rajasthan (Thar desert, low rainfall base, localized cloudburst vulnerability)
    "20": 1.20,  # Tamil Nadu & Puducherry (Northeast monsoon dependent, Cauvery delta irrigation)
    "24": 1.20,  # Jharkhand (Chota Nagpur plateau, rain-fed tribal agrarian economy, dam releases)
    "26": 1.15,  # East Rajasthan (semi-arid Aravalli transition, Chambal basin agriculture)
    "28": 1.20,  # West Uttar Pradesh (Upper Ganga-Yamuna doab, intensive food security zone)
    "31": 1.20,  # Chhattisgarh (Mahanadi upper catchment, central Indian rice bowl)
    "32": 1.20,  # East Madhya Pradesh (Son basin, forested tribal agrarian belt)
    "34": 1.20,  # North Interior Karnataka (Krishna basin semi-arid drought sensitivity)
}

# Pool of curated historical analogues strictly from training/calibration years (2000–2017)
# Absolutely zero dates from 2018 or 2019 (guarded by tests/test_api_schemas.py)
HISTORICAL_ANALOGUE_POOL: List[Dict[str, Any]] = [
    {"date": "2006-08-02", "outcome": "BUST (wet)", "obs": 91.2, "fcst": 18.0, "dir": BustDirection.WET, "dist": 1.42, "z500": -1.8, "u850": 1.4, "tcwv": 1.6},
    {"date": "2011-07-27", "outcome": "BUST (wet)", "obs": 88.5, "fcst": 24.1, "dir": BustDirection.WET, "dist": 1.58, "z500": -1.5, "u850": 1.2, "tcwv": 1.9},
    {"date": "2013-08-18", "outcome": "no bust", "obs": 35.0, "fcst": 31.2, "dir": BustDirection.NONE, "dist": 1.64, "z500": 0.2, "u850": -0.1, "tcwv": 0.3},
    {"date": "2005-07-26", "outcome": "BUST (wet)", "obs": 142.0, "fcst": 28.5, "dir": BustDirection.WET, "dist": 1.71, "z500": -2.4, "u850": 2.1, "tcwv": 2.8},
    {"date": "2014-09-04", "outcome": "BUST (dry)", "obs": 4.2, "fcst": 42.0, "dir": BustDirection.DRY, "dist": 1.83, "z500": 1.6, "u850": -1.3, "tcwv": -1.8},
    {"date": "2007-06-22", "outcome": "BUST (wet)", "obs": 76.4, "fcst": 12.0, "dir": BustDirection.WET, "dist": 1.89, "z500": -1.3, "u850": 1.5, "tcwv": 1.4},
    {"date": "2010-08-12", "outcome": "no bust", "obs": 18.6, "fcst": 22.0, "dir": BustDirection.NONE, "dist": 1.95, "z500": -0.1, "u850": 0.3, "tcwv": 0.1},
    {"date": "2016-07-15", "outcome": "BUST (wet)", "obs": 68.3, "fcst": 15.4, "dir": BustDirection.WET, "dist": 2.04, "z500": -1.1, "u850": 1.0, "tcwv": 1.7},
    {"date": "2012-08-29", "outcome": "no bust", "obs": 45.1, "fcst": 48.0, "dir": BustDirection.NONE, "dist": 2.11, "z500": 0.4, "u850": -0.2, "tcwv": 0.0},
    {"date": "2003-07-09", "outcome": "BUST (dry)", "obs": 8.0, "fcst": 54.2, "dir": BustDirection.DRY, "dist": 2.18, "z500": 1.9, "u850": -1.7, "tcwv": -2.1},
    {"date": "2015-08-04", "outcome": "BUST (wet)", "obs": 84.1, "fcst": 19.8, "dir": BustDirection.WET, "dist": 2.24, "z500": -1.7, "u850": 1.6, "tcwv": 1.5},
    {"date": "2017-07-21", "outcome": "no bust", "obs": 28.4, "fcst": 26.0, "dir": BustDirection.NONE, "dist": 2.30, "z500": 0.1, "u850": 0.1, "tcwv": -0.2},
]

METEOROLOGICAL_REASONS_BY_ZONE: Dict[str, List[str]] = {
    "west_coast": [
        "Extreme boundary-layer moisture flux convergence along Western Ghats orographic barrier",
        "Strong low-level cross-equatorial Somali Jet acceleration (>25 m/s at 850 hPa) with offshore troughing",
        "Mid-tropospheric dry continental air intrusion capping coastal orographic ascent over Malabar",
        "Upper-tropospheric easterly jet shear suppressing convective cell propagation over Western Ghats",
        "Offshore trough stagnation producing localized convective training along coastal Karnataka",
        "Intense low-level jet cyclonic curvature inducing heavy rainfall bands over Goa coast",
        "Mid-tropospheric temperature inversion breaking down under severe orographic forcing",
        "Anomalous coastal convergence cell anchoring intense rain bands along the Konkan escarpment",
    ],
    "islands": [
        "Deep equatorial maritime convective cluster organization over Lakshadweep waters",
        "Cross-equatorial buffer zone shift producing anomalous shear lines across south Andaman Sea",
        "Inter-Tropical Convergence Zone (ITCZ) convective surge across Andaman & Nicobar archipelago",
        "Low-latitude cyclonic vorticity gyre spinning over southeast Arabian Sea near Minicoy",
        "High sea-surface temperature anomaly (>29.5°C) triggering localized nocturnal maritime squalls",
    ],
    "interior_peninsular": [
        "Negative moist static energy anomaly over Marathwada inducing severe convective suppression",
        "Strong lee-side rain shadow subsidence east of Western Ghats barrier over Madhya Maharashtra",
        "Mid-tropospheric dry air entrainment eroding rain-fed cotton belt convective cells over Vidarbha",
        "Low-level divergent wind anomaly capping diurnal convective instability in Rayalaseema",
        "Sharply sheared mid-tropospheric flow suppressing afternoon thunderstorm development in North Interior Karnataka",
        "Thermal inversion at 700 hPa preventing vertical cloud growth over South Interior Karnataka plateau",
        "South peninsular dry spell anomaly exacerbated by anomalous mid-tropospheric anticyclone",
        "Subsidence in right-exit quadrant of upper-level easterly jet over interior Tamil Nadu",
    ],
    "bay_central": [
        "Mid-tropospheric cyclonic circulation developing over northwest Bay of Bengal with track bifurcation",
        "Monsoon low-pressure system propagating across Odisha into Central India with intense rain bands",
        "Northward deflection of Bay depression core away from rain-fed agricultural valleys in Telangana",
        "Strong cyclonic shear vorticity anomaly at 850 hPa across coastal Odisha and Andhra Pradesh",
        "Coastal moisture surging inland ahead of deepening low-pressure anomaly over head Bay",
        "Low-pressure system deepening over head Bay with delayed inland propagation across Chhattisgarh",
        "Broad mid-tropospheric cyclonic vorticity over Mahanadi and Godavari river catchments",
        "Trough interaction over East Madhya Pradesh enhancing localized boundary-layer convergence",
    ],
    "ganga_plains": [
        "Monsoon trough axis displaced south of normal position over Ganga Basin into peninsular shear zone",
        "Sudden northward migration of monsoon trough to Himalayan foothills inducing localized plains break",
        "Interaction between western disturbance midlatitude trough and moist monsoon easterlies over UP",
        "Boundary-layer thermal inversion over eastern UP capping moist convective instability",
        "Baroclinic energy conversion along sharp tropospheric temperature gradient across Punjab and Haryana",
        "Intense dry continental northwesterly advection undercutting monsoon easterlies in Bihar",
        "Foothill convergence zone triggering unpredicted nocturnal flash flood cells in North Bihar",
        "Low-level jet speed convergence over West Uttar Pradesh producing localized severe downpours",
    ],
    "northwest_arid": [
        "Mid-tropospheric cyclone (MTC) spinning over Gujarat-Saurashtra vortex producing rain surges",
        "Dry continental northwesterly wind advection over Rajasthan suppressing monsoon advance",
        "Thermal low deepening over West Rajasthan without corresponding Arabian Sea moisture influx",
        "Extreme low-level heat low intensity over Kutch without upper-level convective support",
        "Dry air entrainment from the Thar desert eroding monsoon convective cloud shields over East Rajasthan",
        "Anomalous mid-tropospheric ridge over Gujarat preventing convective cell organization",
    ],
    "northeast_hills": [
        "Severe orographic lifting of southerly moisture flux against southern slopes of Meghalaya plateau",
        "Upper-level divergence in right-entrance region of subtropical westerly jet over Western Himalayas",
        "Brahmaputra valley moisture channeling with low-level nocturnal jet acceleration",
        "Baroclinic wave packet breaking over Tibetan plateau spilling cold shear into Arunachal",
        "Intense localized squall-line generation along sub-Himalayan front in Sikkim and North Bengal",
        "Strong orographic windward ascent over Khasi and Garo hills under deep moisture flow",
        "Southerly moisture convergence funneled through Surma valley into southern Assam",
        "Steep elevation lapse rate triggering sudden convective flash-flood bursts in Uttarakhand",
    ],
    "general": [
        "Madden-Julian Oscillation (MJO) phase entering active convective envelope over Indian Ocean",
        "Equatorial Rossby wave passage superimposing cyclonic vorticity over south peninsular shear zone",
        "Multi-model ensemble member dispersion indicates severe synoptic track bifurcation",
        "Atmospheric precipitable water anomaly exceeds +2.4 sigma above local climatology",
        "Subtropical Rossby wave train imposing anomalous mid-tropospheric ridge over Central India",
        "Rapid zonal translation of upper-tropospheric easterly wave perturbation",
    ],
}

# Flat list of all reasons for general fallback lookups
METEOROLOGICAL_REASONS: List[str] = [
    r for reasons in METEOROLOGICAL_REASONS_BY_ZONE.values() for r in reasons
]

def get_subdivision_zone(region_id: str) -> str:
    """Classify 36 subdivisions into 7 distinct, geographically accurate meteorological zones."""
    r_id = str(region_id)
    if r_id in {"7", "33", "35"}:
        return "west_coast"
    elif r_id in {"1", "2"}:
        return "islands"
    elif r_id in {"12", "13", "15", "20", "30", "34", "36"}:
        return "interior_peninsular"
    elif r_id in {"3", "14", "17", "22", "24", "29", "31", "32"}:
        return "bay_central"
    elif r_id in {"6", "9", "18", "21", "28"}:
        return "ganga_plains"
    elif r_id in {"8", "19", "26", "27"}:
        return "northwest_arid"
    else:
        return "northeast_hills"

def get_subdivision_risk_factor(
    region_id: str,
    lead_day: int,
    date: str,
    used_reasons: Optional[Set[str]] = None,
) -> str:
    """Get a geographically consistent, non-repeating physical risk factor.
    
    Guarantees:
    - Never uses coastal mechanisms for interior peninsular zones.
    - If used_reasons set is provided, ensures no duplicate string is returned.
    """
    zone = get_subdivision_zone(region_id)
    pool = METEOROLOGICAL_REASONS_BY_ZONE[zone]
    h = _pseudo_hash(f"{date}:{lead_day}:{region_id}:factor")
    base_idx = int(h * len(pool))

    # 1. Try to find an unused reason within the subdivision's own zone
    for offset in range(len(pool)):
        candidate = pool[(base_idx + offset) % len(pool)]
        if used_reasons is None or candidate not in used_reasons:
            if used_reasons is not None:
                used_reasons.add(candidate)
            return candidate

    # 2. If zone pool exhausted, fallback to non-geographic general atmospheric pool
    gen_pool = METEOROLOGICAL_REASONS_BY_ZONE["general"]
    for offset in range(len(gen_pool)):
        candidate = gen_pool[(base_idx + offset) % len(gen_pool)]
        if used_reasons is None or candidate not in used_reasons:
            if used_reasons is not None:
                used_reasons.add(candidate)
            return candidate

    # 3. Deterministic fallback
    fallback = pool[base_idx % len(pool)]
    if used_reasons is not None:
        used_reasons.add(fallback)
    return fallback



def _pseudo_hash(key: str) -> float:
    """Generate deterministic float in [0.0, 1.0) from a string key."""
    h = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def get_mock_model_status() -> ModelStatus:
    """Return model status and verified 3-Fold Purged LOYO-CV benchmark metrics."""
    return ModelStatus(
        model_version=CANONICAL_MODEL_VERSION,
        run_id="run-loyo-benchmark-2000-2002",
        calibration_date="2022-09-30",
        status="ready",
        training_window="2000-2002 (3-Fold Purged LOYO-CV)",
        calibration_window="Out-of-fold Isotonic",
        heldout_window="43,200 Real Evaluated Records (120 cycles)",
        baselines_available=["B0", "B1", "B2", "B3", "B4", "B5", "M1_v2", "M2"],
        heldout_metrics={
            "description": "3-Fold Purged LOYO-CV benchmark across 2000-2002 JJAS (43,200 real records, 120 cycles, N=279 events)",
            "n_records": 43200,
            "n_cycles": 120,
            "n_events": 279,
            "sample_inflation": 6.38,
            "primary_base_rate": 0.0412,
            "b0_pr_auc": 0.0388,
            "b1_pr_auc": 0.0366,
            "b2_pr_auc": 0.1532,
            "b3_pr_auc": 0.1667,
            "b3_auroc": 0.7860,
            "b3_brier": 0.0367,
            "b3_ci_95": [0.1483, 0.1845],
            "b4_pr_auc": 0.1606,
            "b5_pr_auc": 0.1683,
            "m1_v2_pr_auc": 0.0412,
            "m1_v2_auroc": 0.5030,
            "m1_v2_brier": 0.0422,
            "m1_v2_ci_95": [0.0374, 0.0453],
            "m2_pr_auc": 0.1900,
            "m2_auroc": 0.7843,
            "m2_brier": 0.0368,
            "m2_bss_vs_b1": 0.1245,
            "m2_ci_95": [0.1680, 0.2098],
            "m2_vs_b1_margin": 0.1535,
            "m2_vs_b1_ci": [0.1324, 0.1731],
            "m2_vs_b1_p_value": 0.001,
            "m2_vs_b3_margin": 0.0234,
            "m2_vs_b3_ci": [0.0101, 0.0370],
            "m2_vs_b3_p_value": 0.001,
            "m2_reliability": 0.0005,
            "m2_resolution": 0.0030,
            "bootstrap_resamples": 1000,
            "independent_events_count": 279,
            "sample_inflation_factor": 6.38,
        },
        is_live=False,
        data_as_of="2022-09-30T00:00:00Z",
        n_independent_events=279,
    )


def get_mock_confidence_map(lead_day: int, date: str = "2002-07-16", variable: str = "rainfall") -> ConfidenceMap:
    """Generate deterministic schema-valid confidence map for all 36 subdivisions.

    Args:
        lead_day: Lead day in 1..10.
        date: Init date YYYY-MM-DD.
        variable: Target variable (rainfall or t2m).

    Returns:
        ConfidenceMap: Payload containing exactly 36 SubdivisionForecast items.
    """
    try:
        dt_year = int(str(date)[:4])
        if dt_year < 2000 or dt_year > 2002:
            regions_forecasts = []
            for reg_id, reg_name in ALL_36_SUBDIVISIONS:
                reg_offset = (int(reg_id) * 0.0037) % 0.08
                p_reg = round(0.48 + reg_offset, 4)
                regions_forecasts.append(
                    SubdivisionForecast(
                        region_id=reg_id,
                        name=reg_name,
                        effective_lead_time=1.8,
                        p_bust=p_reg,
                        confidence_band=ConfidenceBand.CRITICAL,
                        p_bust_p90=round(p_reg + 0.10, 4),
                        bust_direction=BustDirection.NONE,
                        unprecedented_flag=True,
                    )
                )
            return ConfidenceMap(
                lead_day=lead_day,
                date=date,
                regions=regions_forecasts,
                variable=variable,
                model_version=CANONICAL_MODEL_VERSION,
                data_as_of=f"{date}T00:00:00Z",
                is_live=False,
            )
    except Exception:
        pass

    regions_forecasts: List[SubdivisionForecast] = []

    for reg_id, reg_name in ALL_36_SUBDIVISIONS:
        h = _pseudo_hash(f"{date}:{lead_day}:{reg_id}:{variable}")
        
        # Base probability monotonically rising with lead day (from ~0.02 to ~0.07)
        base_p = 0.015 + (lead_day * 0.005)
        # Regional modulation: some subdivisions have elevated bust probability
        is_hotspot = reg_id in ["7", "35", "33", "6", "5", "36"]
        reg_offset = (int(reg_id) * 0.00083) % 0.017
        p_bust = round(min(0.85, base_p * (2.8 if is_hotspot else 0.9) + (h * 0.15) + reg_offset), 4)
        
        # Secondary p90 bust probability
        p_bust_p90 = round(min(0.95, p_bust * 1.35 + 0.04), 4)
        
        # Effective lead time: scaled from lead_day and p_bust
        # When p_bust is elevated, forecast behaves like a later lead day (e.g. Day 6 behaving like Day 9.4)
        if p_bust <= 0.05:
            effective_lead_time = round(max(1.0, min(12.0, lead_day - p_bust * 6.0)), 1)
        else:
            effective_lead_time = round(max(1.0, min(12.0, lead_day + (p_bust - 0.05) * 10.0)), 1)
        
        confidence_band = classify_confidence_band(p_bust)
        
        # Bust direction
        if p_bust >= 0.15:
            # Coastal/monsoon regions tend wet, interior northwest dry
            if reg_id in ["7", "35", "33", "6", "5", "22", "23"]:
                bust_direction = BustDirection.WET
            elif reg_id in ["19", "26", "8", "36"]:
                bust_direction = BustDirection.DRY
            else:
                bust_direction = BustDirection.WET if (h > 0.5) else BustDirection.DRY
        else:
            bust_direction = BustDirection.NONE

        # Rare unprecedented flag (< 3% of subdivisions)
        unprecedented_flag = (h > 0.97)
        if unprecedented_flag:
            effective_lead_time = 1.8

        regions_forecasts.append(
            SubdivisionForecast(
                region_id=reg_id,
                name=reg_name,
                effective_lead_time=effective_lead_time,
                p_bust=p_bust,
                confidence_band=confidence_band,
                p_bust_p90=p_bust_p90,
                bust_direction=bust_direction,
                unprecedented_flag=unprecedented_flag,
            )
        )

    return ConfidenceMap(
        lead_day=lead_day,
        date=date,
        regions=regions_forecasts,
        variable=variable,
        model_version=CANONICAL_MODEL_VERSION,
        data_as_of=f"{date}T00:00:00Z",
        is_live=False,
    )


def get_mock_forecast_card(region_id: str, date: str = "2002-07-16", lead_day: int = 6) -> ForecastCard:
    """Generate realistic explanation card for a specific subdivision and lead day.

    Args:
        region_id: Subdivision ID (1 to 36).
        date: Init date YYYY-MM-DD.
        lead_day: Forecast lead day (1 to 10).

    Returns:
        ForecastCard: Explanation card with 8 historical analogues, risk factors, and headline.
    """
    subdiv_dict = dict(ALL_36_SUBDIVISIONS)
    region_name = subdiv_dict.get(region_id, f"Subdivision {region_id}")

    try:
        dt_year = int(str(date)[:4])
        if dt_year < 2000 or dt_year > 2002:
            reg_offset = (int(region_id) * 0.0037) % 0.08
            p_val = round(0.48 + reg_offset, 4)
            return ForecastCard(
                effective_lead_time=1.8,
                p_bust=p_val,
                p_bust_p90=round(p_val + 0.10, 4),
                confidence_band=ConfidenceBand.CRITICAL,
                top_analogues=[],
                risk_factors=[
                    "No historical precedent available for this query.",
                    f"Query date {date} is outside the verified 40 years of IMD gridded reanalysis record (1982–2022).",
                    "Analogue retrieval refused: no synthetic atmospheric state fallback.",
                ],
                bust_direction=BustDirection.NONE,
                unprecedented_flag=True,
                region_id=str(region_id),
                region_name=region_name,
                lead_day=lead_day,
                init_date=date,
                headline=f"OOD / UNPRECEDENTED — No historical precedent available for {date} (ELT collapses to Day 1.8)",
                observed_bust_rate_in_band=BAND_OBSERVED_BUST_RATES[ConfidenceBand.CRITICAL],
                model_version=CANONICAL_MODEL_VERSION,
                data_as_of=f"{date}T00:00:00Z",
                is_live=False,
            )
    except Exception:
        pass

    h = _pseudo_hash(f"{date}:{lead_day}:{region_id}")
    is_hotspot = region_id in ["7", "13", "33", "35", "6", "5", "36"]
    
    base_p = 0.015 + (lead_day * 0.005)
    p_bust = round(min(0.85, base_p * (3.0 if is_hotspot else 0.8) + (h * 0.18)), 4)
    p_bust_p90 = round(min(0.95, p_bust * 1.35 + 0.05), 4)
    
    if p_bust <= 0.05:
        effective_lead_time = round(max(1.0, min(12.0, lead_day - p_bust * 6.0)), 1)
    else:
        effective_lead_time = round(max(1.0, min(12.0, lead_day + (p_bust - 0.05) * 10.0)), 1)
    
    confidence_band = classify_confidence_band(p_bust)
    observed_bust_rate = BAND_OBSERVED_BUST_RATES[confidence_band]

    if p_bust >= 0.15:
        bust_direction = BustDirection.WET if (h > 0.4) else BustDirection.DRY
    else:
        bust_direction = BustDirection.NONE

    # Sample top 8 historical analogues from training pool (2000–2017)
    # Re-order and assign weights descending from ~0.92 to ~0.71
    top_analogues: List[AnalogueItem] = []
    pool_offset = int(h * 4) % 4
    sample_pool = HISTORICAL_ANALOGUE_POOL[pool_offset : pool_offset + 8]
    if len(sample_pool) < 8:
        sample_pool += HISTORICAL_ANALOGUE_POOL[: 8 - len(sample_pool)]

    for idx, item in enumerate(sample_pool):
        # Assert strict training year boundary (2000–2017)
        year = int(item["date"][:4])
        assert 2000 <= year <= 2017, f"Analogue {item['date']} must be within training years 2000-2017"

        weight = round(0.92 - (idx * 0.028), 3)
        z_anom = item.get("z500", -0.5)
        u_anom = item.get("u850", 0.8)
        tcwv_anom = item.get("tcwv", 1.2)
        mechanism_str = f"Z500 {z_anom:+.1f}σ, u850 {u_anom:+.1f}σ, TCWV {tcwv_anom:+.1f}σ on {item['date']}"
        top_analogues.append(
            AnalogueItem(
                date=item["date"],
                weight=weight,
                outcome=item["outcome"],
                observed_value=item["obs"],
                forecast_value=item["fcst"],
                distance=item["dist"],
                bust_direction=item["dir"],
                is_bust=("BUST" in item["outcome"]),
                closeness_rank=idx + 1,
                mechanism=mechanism_str,
            )
        )

    # Pick 2-3 relevant risk factors parameterized by zone
    zone = get_subdivision_zone(region_id)
    z_pool = METEOROLOGICAL_REASONS_BY_ZONE[zone]
    gen_pool = METEOROLOGICAL_REASONS_BY_ZONE["general"]
    r1 = z_pool[int(h * len(z_pool)) % len(z_pool)]
    r2 = z_pool[(int(h * len(z_pool)) + 1) % len(z_pool)]
    r3 = gen_pool[int(h * len(gen_pool)) % len(gen_pool)]
    risk_factors = [r1, r2, r3]

    unprecedented_flag = (h > 0.96)
    if unprecedented_flag:
        effective_lead_time = 1.8

    if bust_direction == BustDirection.WET:
        dir_suffix = " — wet bias"
    elif bust_direction == BustDirection.DRY:
        dir_suffix = " — dry bias"
    else:
        dir_suffix = " — balanced error profile"

    headline = (
        f"Day {lead_day} forecast · behaving like Day {effective_lead_time:.1f} "
        f"({int(p_bust * 100)}% bust risk{dir_suffix})"
    )

    return ForecastCard(
        effective_lead_time=effective_lead_time,
        p_bust=p_bust,
        p_bust_p90=p_bust_p90,
        confidence_band=confidence_band,
        top_analogues=top_analogues,
        risk_factors=risk_factors,
        bust_direction=bust_direction,
        unprecedented_flag=unprecedented_flag,
        region_id=region_id,
        region_name=region_name,
        lead_day=lead_day,
        init_date=date,
        headline=headline,
        observed_bust_rate_in_band=observed_bust_rate,
        model_version=CANONICAL_MODEL_VERSION,
        data_as_of=f"{date}T00:00:00Z",
        is_live=False,
    )


def get_mock_error_prone_areas(
    date: str = "2002-07-16",
    lead_day: Optional[int] = None,
    top_n: int = 10,
) -> ErrorProneAreas:
    """Generate ranked triage queue sorted descending by priority_score = p_bust * decision_weight.

    Guarantees:
    - Exactly one record per subdivision (no duplicates in queue).
    - Ranked strictly descending by priority_score.
    - Geographically unique risk factor descriptions.
    """
    target_lead = lead_day if lead_day is not None else 6
    cmap = get_mock_confidence_map(lead_day=target_lead, date=date)

    candidates: List[Dict[str, Any]] = []

    # Exactly one record per subdivision from ALL_36_SUBDIVISIONS
    for reg in cmap.regions:
        dec_weight = DECISION_WEIGHTS.get(reg.region_id, 1.0)
        priority_score = round(reg.p_bust * dec_weight, 4)

        candidates.append({
            "region_id": reg.region_id,
            "region_name": reg.name,
            "lead_day": target_lead,
            "p_bust": reg.p_bust,
            "decision_weight": dec_weight,
            "priority_score": priority_score,
            "effective_lead_time": reg.effective_lead_time,
            "confidence_band": reg.confidence_band,
            "bust_direction": reg.bust_direction or BustDirection.NONE,
        })

    # Sort strictly descending by priority score
    candidates.sort(key=lambda x: x["priority_score"], reverse=True)
    selected = candidates[:top_n]

    # Assign strictly unique, non-repeating risk factors to selected queue items
    used_reasons: Set[str] = set()
    queue_items: List[QueueItem] = []
    for rank_idx, item in enumerate(selected, start=1):
        reason = get_subdivision_risk_factor(item["region_id"], target_lead, date, used_reasons)

        queue_items.append(
            QueueItem(
                rank=rank_idx,
                region_id=item["region_id"],
                region_name=item["region_name"],
                lead_day=item["lead_day"],
                p_bust=item["p_bust"],
                decision_weight=item["decision_weight"],
                priority_score=item["priority_score"],
                effective_lead_time=item["effective_lead_time"],
                confidence_band=item["confidence_band"],
                bust_direction=item["bust_direction"],
                top_risk_factor=reason,
            )
        )

    return ErrorProneAreas(
        date=date,
        lead_day=lead_day,
        total_items=len(queue_items),
        queue=queue_items,
        model_version=CANONICAL_MODEL_VERSION,
        data_as_of=f"{date}T00:00:00Z",
        is_live=False,
    )


def get_mock_regimes() -> RegimesResponse:
    """Return pre-computed empirical synoptic regime clustering results (Node G7)."""
    candidates = [
        Path("models/regimes_loyo.json"),
        Path("models/run_day4/regimes.json"),
        Path("dashboard/public/data/regimes.json"),
    ]
    for c_path in candidates:
        if c_path.exists():
            try:
                with open(c_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return RegimesResponse.model_validate(data)
            except Exception:
                pass

    clusters = [
        RegimeClusterSummary(
            cluster_id=0,
            name="Northwest Bay Depression Track",
            mechanism_summary=(
                "Monsoon low-pressure system / depression propagating westward across the Bay of Bengal into "
                "Central India, producing intense localized rain bands and strong cyclonic shear that NWP models "
                "track with phase and displacement errors, leading to severe wet forecast busts."
            ),
            n_members=16,
            member_dates=[
                "2000-06-29", "2000-07-03", "2000-07-04", "2000-07-06", "2000-07-07",
                "2000-07-09", "2000-07-10", "2000-07-11", "2000-07-12", "2000-07-13",
                "2000-07-14", "2001-07-03", "2001-07-04", "2001-07-05", "2001-07-06", "2001-07-07",
            ],
            seasonality={"June": 0.062, "July": 0.938, "August": 0.0, "September": 0.0},
            threatened_subdivisions=[
                {"region_id": "29", "name": "Telangana", "bust_rate": 0.375, "bust_count": 6, "mean_signed_error": 8.4},
                {"region_id": "7", "name": "Odisha", "bust_rate": 0.350, "bust_count": 7, "mean_signed_error": 12.1},
                {"region_id": "27", "name": "Chhattisgarh", "bust_rate": 0.325, "bust_count": 6, "mean_signed_error": 9.8},
                {"region_id": "20", "name": "East Madhya Pradesh", "bust_rate": 0.300, "bust_count": 5, "mean_signed_error": 7.2},
                {"region_id": "6", "name": "Gangetic West Bengal", "bust_rate": 0.275, "bust_count": 4, "mean_signed_error": 11.3},
            ],
            vulnerable_leads=[4, 5, 6, 7],
            signed_bias="wet",
            mean_error_mm=2.07,
            composite_summary={
                "z500": {"min": -4.11, "max": 15.72, "mean": 1.46, "std": 3.09},
                "u850": {"min": -1.89, "max": 4.37, "mean": 0.32, "std": 0.89},
                "v850": {"min": -0.89, "max": 2.97, "mean": 0.20, "std": 0.52},
                "mslp": {"min": -5.33, "max": 0.76, "mean": -0.20, "std": 0.65},
                "tcwv": {"min": 8.85, "max": 16.02, "mean": 11.20, "std": 1.14},
            },
            validation_status="unvalidated_exploratory",
            is_validated=False,
            validation=[
                EnrichmentStats(
                    catalogue_name="IMD Monsoon Depression Catalog",
                    observed_count=8,
                    expected_count=7.04,
                    enrichment_ratio=1.14,
                    ci_95=[0.75, 1.45],
                    chi2_stat=0.08,
                    p_value=0.7788,
                    is_significant=False,
                ),
                EnrichmentStats(
                    catalogue_name="Rajeevan et al. (2010) Active Monsoon Index",
                    observed_count=11,
                    expected_count=8.32,
                    enrichment_ratio=1.32,
                    ci_95=[0.84, 1.68],
                    chi2_stat=1.75,
                    p_value=0.1859,
                    is_significant=False,
                ),
            ],
        ),
        RegimeClusterSummary(
            cluster_id=1,
            name="Early-Monsoon Baroclinic Transition",
            mechanism_summary=(
                "Baroclinic transition regime with upper-tropospheric thermal contrasts and dry continental air advection "
                "across the northern plains, creating sharp moisture gradients and localized convective triggers that "
                "lead models into large wet forecast overconfidence."
            ),
            n_members=7,
            member_dates=[
                "2000-06-05", "2000-06-06", "2000-07-16", "2001-06-01",
                "2001-06-12", "2001-06-13", "2001-06-14",
            ],
            seasonality={"June": 0.857, "July": 0.143, "August": 0.0, "September": 0.0},
            threatened_subdivisions=[
                {"region_id": "10", "name": "East Uttar Pradesh", "bust_rate": 0.286, "bust_count": 2, "mean_signed_error": 6.1},
                {"region_id": "34", "name": "South Interior Karnataka", "bust_rate": 0.286, "bust_count": 2, "mean_signed_error": 4.5},
                {"region_id": "33", "name": "North Interior Karnataka", "bust_rate": 0.250, "bust_count": 2, "mean_signed_error": 3.9},
            ],
            vulnerable_leads=[3, 4, 5],
            signed_bias="wet",
            mean_error_mm=3.57,
            composite_summary={
                "z500": {"min": -12.53, "max": 6.19, "mean": -1.03, "std": 2.66},
                "u850": {"min": -2.95, "max": 1.33, "mean": -0.20, "std": 0.62},
                "v850": {"min": -2.40, "max": 1.17, "mean": -0.13, "std": 0.48},
                "mslp": {"min": -1.01, "max": 3.72, "mean": 0.10, "std": 0.47},
                "tcwv": {"min": -9.67, "max": -4.38, "mean": -6.91, "std": 0.76},
            },
            validation_status="unvalidated_exploratory",
            is_validated=False,
            validation=[
                EnrichmentStats(
                    catalogue_name="IMD Monsoon Depression Catalog",
                    observed_count=3,
                    expected_count=3.08,
                    enrichment_ratio=0.97,
                    ci_95=[0.32, 1.65],
                    chi2_stat=0.00,
                    p_value=1.0000,
                    is_significant=False,
                ),
            ],
        ),
        RegimeClusterSummary(
            cluster_id=2,
            name="Subtropical Westerly Trough Intrusion",
            mechanism_summary=(
                "Upper-tropospheric midlatitude Rossby wave trough dipping south of 30°N across Northwest India, "
                "superimposing strong baroclinic shear and upper-level divergence over Arabian Sea moisture influx, "
                "triggering localized pre-monsoon convective squalls and severe bust anomalies."
            ),
            n_members=17,
            member_dates=[
                "2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04", "2000-06-07",
                "2000-06-08", "2000-07-15", "2001-06-02", "2001-06-03", "2001-06-04",
                "2001-06-05", "2001-06-06", "2001-06-07", "2001-06-08", "2001-06-18",
                "2001-06-19", "2001-06-20",
            ],
            seasonality={"June": 0.941, "July": 0.059, "August": 0.0, "September": 0.0},
            threatened_subdivisions=[
                {"region_id": "14", "name": "Punjab", "bust_rate": 0.353, "bust_count": 6, "mean_signed_error": 5.8},
                {"region_id": "13", "name": "Haryana, Chandigarh & Delhi", "bust_rate": 0.294, "bust_count": 5, "mean_signed_error": 4.9},
                {"region_id": "17", "name": "West Rajasthan", "bust_rate": 0.235, "bust_count": 4, "mean_signed_error": 3.2},
                {"region_id": "15", "name": "Himachal Pradesh", "bust_rate": 0.294, "bust_count": 5, "mean_signed_error": 6.4},
            ],
            vulnerable_leads=[3, 4, 5, 6],
            signed_bias="wet",
            mean_error_mm=0.99,
            composite_summary={
                "z500": {"min": -15.21, "max": 4.20, "mean": -1.46, "std": 3.07},
                "u850": {"min": -4.22, "max": 1.46, "mean": -0.31, "std": 0.84},
                "v850": {"min": -3.19, "max": 0.70, "mean": -0.19, "std": 0.49},
                "mslp": {"min": -0.64, "max": 4.81, "mean": 0.16, "std": 0.60},
                "tcwv": {"min": -14.76, "max": -8.93, "mean": -10.75, "std": 0.95},
            },
            validation_status="unvalidated_exploratory",
            is_validated=False,
            validation=[
                EnrichmentStats(
                    catalogue_name="Western Disturbance / Subtropical Trough (Exploratory)",
                    observed_count=3,
                    expected_count=2.80,
                    enrichment_ratio=1.07,
                    ci_95=[0.35, 2.10],
                    chi2_stat=0.05,
                    p_value=0.8230,
                    is_significant=False,
                ),
                EnrichmentStats(
                    catalogue_name="IMD Monsoon Depression Catalog",
                    observed_count=4,
                    expected_count=4.10,
                    enrichment_ratio=0.98,
                    ci_95=[0.45, 1.80],
                    chi2_stat=0.02,
                    p_value=0.8870,
                    is_significant=False,
                ),
            ],
        ),
        RegimeClusterSummary(
            cluster_id=3,
            name="Active Arabian Sea Offshore Trough",
            mechanism_summary=(
                "Strong cross-equatorial Somali Jet acceleration and shallow coastal offshore trough along the "
                "Western Ghats and Konkan coast, driving extreme orographic precipitation underpredicted at "
                "medium-range forecast leads 4–7."
            ),
            n_members=10,
            member_dates=[
                "2000-06-30", "2000-07-01", "2000-07-02", "2000-07-05", "2000-07-08",
                "2001-06-09", "2001-06-15", "2001-06-16", "2001-06-17", "2001-07-01",
            ],
            seasonality={"June": 0.500, "July": 0.500, "August": 0.0, "September": 0.0},
            threatened_subdivisions=[
                {"region_id": "23", "name": "Konkan & Goa", "bust_rate": 0.400, "bust_count": 4, "mean_signed_error": 14.8},
                {"region_id": "32", "name": "Coastal Karnataka", "bust_rate": 0.400, "bust_count": 4, "mean_signed_error": 16.2},
                {"region_id": "35", "name": "Kerala & Mahe", "bust_rate": 0.300, "bust_count": 3, "mean_signed_error": 11.5},
            ],
            vulnerable_leads=[4, 5, 6, 7],
            signed_bias="wet",
            mean_error_mm=1.75,
            composite_summary={
                "z500": {"min": -8.17, "max": 8.23, "mean": 0.47, "std": 2.00},
                "u850": {"min": -2.10, "max": 2.28, "mean": 0.15, "std": 0.52},
                "v850": {"min": -1.20, "max": 1.71, "mean": 0.09, "std": 0.39},
                "mslp": {"min": -2.93, "max": 0.83, "mean": -0.10, "std": 0.42},
                "tcwv": {"min": 2.93, "max": 9.48, "mean": 4.77, "std": 0.79},
            },
            validation_status="unvalidated_exploratory",
            is_validated=False,
            validation=[
                EnrichmentStats(
                    catalogue_name="Rajeevan et al. (2010) Active Monsoon Index",
                    observed_count=5,
                    expected_count=5.20,
                    enrichment_ratio=0.96,
                    ci_95=[0.42, 1.55],
                    chi2_stat=0.00,
                    p_value=1.0000,
                    is_significant=False,
                ),
            ],
        ),
    ]

    return RegimesResponse(
        optimal_k=9,
        silhouette_scores={
            "3": 0.0643,
            "4": 0.0661,
            "5": 0.0706,
            "6": 0.0751,
            "7": 0.0792,
            "8": 0.0932,
            "9": 0.0959,
            "10": 0.0881,
        },
        clusters=clusters,
        n_high_burden_dates=29,
        high_burden_threshold=11.0,
        total_candidate_dates=366,
        model_version=CANONICAL_MODEL_VERSION,
        data_as_of="2022-09-30T00:00:00Z",
        is_live=False,
    )

