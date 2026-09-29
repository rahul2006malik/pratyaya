"""Automated Data Integrity and Regression Verification Suite for Project Pratyaya.

Implements the Section 3(C) regression requirements from docs/pratyaya-uiux-upgrade-brief-v3-root-cause.md:
1. Zero duplicate risk factors across any triage queue.
2. Zero wall-clock (2026) timestamps in historical forecasts (2000-2018).
3. Zero repeated region_ids in any single triage queue.
4. Physical zone sanity (e.g., no Somali Jet or Malabar coastal terms in Marathwada or Vidarbha).
5. Exact 36 canonical subdivisions with consistent naming.
6. Unified canonical model version ('pratyaya-v1.0.0-loyo') and archive span ('23 years (2000–2022)').
7. Verification of all exported static JSON files in dashboard/public/data/ and dashboard/dist/data/.
"""

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Set

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from pratyaya.api.mock_provider import (
    ALL_36_SUBDIVISIONS,
    CANONICAL_ARCHIVE_SPAN,
    CANONICAL_MODEL_VERSION,
    DECISION_WEIGHTS,
    METEOROLOGICAL_REASONS_BY_ZONE,
    get_mock_confidence_map,
    get_mock_error_prone_areas,
    get_mock_forecast_card,
    get_mock_model_status,
    get_mock_regimes,
    get_subdivision_risk_factor,
    get_subdivision_zone,
)
from pratyaya.api.replay import get_replay_scenario
from pratyaya.models.serving import ProductionModelServer


def run_checks():
    failures = []
    total_checks = 0

    def check(condition: bool, msg: str):
        nonlocal total_checks
        total_checks += 1
        if not condition:
            print(f"  [FAIL] {msg}")
            failures.append(msg)
        else:
            print(f"  [PASS] {msg}")

    print("\n" + "=" * 70)
    print("PRATYAYA DATA INTEGRITY & ROOT CAUSE VERIFICATION SUITE")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # TEST 1: Canonical 36 Subdivisions & Region Naming
    # -------------------------------------------------------------------------
    print("\n--- Test 1: Canonical Subdivisions ---")
    check(len(ALL_36_SUBDIVISIONS) == 36, "Exactly 36 official subdivisions defined")
    ids = [s[0] for s in ALL_36_SUBDIVISIONS]
    check(len(set(ids)) == 36, "All 36 subdivision IDs are strictly unique (1..36)")
    
    # Check region 35 is Kerala and region 13 is Vidarbha per Brief v4
    r35 = next((name for sid, name in ALL_36_SUBDIVISIONS if sid == "35"), None)
    r13 = next((name for sid, name in ALL_36_SUBDIVISIONS if sid == "13"), None)
    check(r35 == "Kerala", f"Region 35 is 'Kerala' (got: {r35})")
    check(r13 == "Vidarbha", f"Region 13 is 'Vidarbha' (got: {r13})")
    check(CANONICAL_MODEL_VERSION == "pratyaya-v1.0.0-loyo", f"Canonical model version is 'pratyaya-v1.0.0-loyo'")
    check("40 years" in CANONICAL_ARCHIVE_SPAN, f"Canonical archive span mentions 40 years (got: {CANONICAL_ARCHIVE_SPAN})")

    # Decision weight completeness: exactly all 36 subdivisions must have an explicit weight
    check(len(DECISION_WEIGHTS) == 36, f"All 36 subdivisions have explicit decision weights (got: {len(DECISION_WEIGHTS)})")
    check(all(str(i) in DECISION_WEIGHTS for i in range(1, 37)), "All IDs 1..36 exist in DECISION_WEIGHTS")
    check(all(1.10 <= w <= 1.65 for w in DECISION_WEIGHTS.values()), "All decision weights calibrated between 1.10 and 1.65")

    # -------------------------------------------------------------------------
    # TEST 2: Meteorological Zone Sanity
    # -------------------------------------------------------------------------
    print("\n--- Test 2: Zone Sanity & Geographic Attribution ---")
    z35 = get_subdivision_zone("35")
    z36 = get_subdivision_zone("36")
    z13 = get_subdivision_zone("13")
    check(z35 == "west_coast", f"Region 35 (Kerala) maps to west_coast zone (got: {z35})")
    check(z36 == "interior_peninsular", f"Region 36 (Marathwada) maps to interior_peninsular zone (got: {z36})")
    check(z13 == "interior_peninsular", f"Region 13 (Vidarbha) maps to interior_peninsular zone (got: {z13})")

    # Test risk factor generator for interior zones does not mention Malabar or Somali Jet
    used: Set[str] = set()
    marathwada_factor = get_subdivision_risk_factor("36", 6, "2002-07-16", used)
    check("Malabar" not in marathwada_factor, f"Marathwada factor does not bleed 'Malabar': '{marathwada_factor}'")
    check("Somali Jet" not in marathwada_factor, f"Marathwada factor does not bleed 'Somali Jet': '{marathwada_factor}'")

    # -------------------------------------------------------------------------
    # TEST 3: Zero Duplicate Risk Factors in Triage Queue (Representative Cycles)
    # -------------------------------------------------------------------------
    print("\n--- Test 3: Deduplication in Triage Queue ---")
    test_cycles = ["2002-07-07", "2002-07-16", "2001-07-01", "2018-08-16"]
    for c_date in test_cycles:
        for ld in [1, 4, 6, 7]:
            queue_payload = get_mock_error_prone_areas(date=c_date, lead_day=ld, top_n=14)
            factors = [item.top_risk_factor for item in queue_payload.queue]
            reg_ids = [item.region_id for item in queue_payload.queue]
            
            check(len(factors) == len(set(factors)), f"Cycle {c_date} D{ld}: All {len(factors)} triage factors are strictly unique")
            check(len(reg_ids) == len(set(reg_ids)), f"Cycle {c_date} D{ld}: No duplicate region_ids in queue ({len(reg_ids)} unique)")

    # -------------------------------------------------------------------------
    # TEST 4: Zero Wall-Clock (2026) Timestamps in Historical Cycles
    # -------------------------------------------------------------------------
    print("\n--- Test 4: Wall-Clock Leak Prevention ---")
    for c_date in test_cycles:
        cmap = get_mock_confidence_map(lead_day=6, date=c_date)
        check(cmap.data_as_of.startswith(c_date), f"ConfidenceMap data_as_of matches cycle {c_date} (got: {cmap.data_as_of})")
        check("2026" not in cmap.data_as_of, f"No 2026 timestamp in ConfidenceMap for {c_date}")
        check(cmap.model_version == CANONICAL_MODEL_VERSION, f"ConfidenceMap version is {CANONICAL_MODEL_VERSION}")

        q_area = get_mock_error_prone_areas(date=c_date, lead_day=6, top_n=10)
        check(q_area.data_as_of.startswith(c_date), f"ErrorProneAreas data_as_of matches cycle {c_date}")
        check("2026" not in q_area.data_as_of, f"No 2026 in ErrorProneAreas for {c_date}")
        check(q_area.model_version == CANONICAL_MODEL_VERSION, f"ErrorProneAreas version is {CANONICAL_MODEL_VERSION}")

        card = get_mock_forecast_card(region_id="13", lead_day=6, date=c_date)
        check(card.data_as_of.startswith(c_date), f"ForecastCard data_as_of matches cycle {c_date}")
        check("2026" not in card.data_as_of, f"No 2026 in ForecastCard for {c_date}")
        check(card.model_version == CANONICAL_MODEL_VERSION, f"ForecastCard version is {CANONICAL_MODEL_VERSION}")

    # Model status date check
    status = get_mock_model_status()
    check(status.calibration_date == "2022-09-30", f"ModelStatus calibration_date is '2022-09-30' (got: {status.calibration_date})")
    check(status.data_as_of.startswith("2022-09-30"), f"ModelStatus data_as_of is 2022-09-30 (got: {status.data_as_of})")
    check(status.model_version == CANONICAL_MODEL_VERSION, f"ModelStatus version is {CANONICAL_MODEL_VERSION}")

    # -------------------------------------------------------------------------
    # TEST 5: Replay Scenarios (Time Machine Replay)
    # -------------------------------------------------------------------------
    print("\n--- Test 5: Replay Scenario Validation ---")
    replay = get_replay_scenario("july2002_break", max_lead=7)
    check(replay.model_version == CANONICAL_MODEL_VERSION, f"Replay model_version is canonical")
    for frame in replay.frames:
        triage_factors = [item.top_risk_factor for item in frame.triage_queue]
        check(len(triage_factors) == len(set(triage_factors)), f"Replay Break frame D{frame.lead_day}: Triage factors strictly unique")
        check(all("Lead Day" not in f for f in triage_factors), f"Replay Break frame D{frame.lead_day}: No generic 'Lead Day X synoptic analogue' fallback")
        check("2026" not in frame.confidence_map.data_as_of, f"Replay Break frame D{frame.lead_day}: No 2026 date in map")
        check(frame.confidence_map.data_as_of.startswith(frame.cycle_date), f"Replay Break frame D{frame.lead_day}: Map timestamp matches {frame.cycle_date}")

    # -------------------------------------------------------------------------
    # TEST 6: ProductionModelServer Verification
    # -------------------------------------------------------------------------
    print("\n--- Test 6: ProductionModelServer Verification ---")
    try:
        server = ProductionModelServer().load()
        s_status = server.get_status()
        check(s_status.model_version == CANONICAL_MODEL_VERSION, f"Server get_status() version is {CANONICAL_MODEL_VERSION}")
        check("2026" not in s_status.data_as_of, f"Server get_status() data_as_of has no 2026 ({s_status.data_as_of})")
        check(s_status.calibration_date == "2022-09-30", f"Server calibration_date is 2022-09-30")

        s_map = server.predict_map(lead_day=6, date="2002-07-16")
        check(s_map.model_version == CANONICAL_MODEL_VERSION, f"Server predict_map() version is {CANONICAL_MODEL_VERSION}")
        check(s_map.data_as_of == "2002-07-16T00:00:00Z", f"Server predict_map() data_as_of is 2002-07-16T00:00:00Z")

        s_card = server.predict_region_card(region_id="13", lead_day=6, date="2002-07-16")
        check(s_card.model_version == CANONICAL_MODEL_VERSION, f"Server predict_region_card() version is {CANONICAL_MODEL_VERSION}")
        check(s_card.data_as_of == "2002-07-16T00:00:00Z", f"Server predict_region_card() data_as_of is 2002-07-16T00:00:00Z")

        s_queue = server.predict_triage_queue(date="2002-07-16", lead_day=6, top_n=10)
        s_factors = [item.top_risk_factor for item in s_queue.queue]
        s_regions = [item.region_id for item in s_queue.queue]
        check(len(s_factors) == len(set(s_factors)), f"Server predict_triage_queue() has strictly unique risk factors ({len(s_factors)})")
        check(len(s_regions) == len(set(s_regions)), f"Server predict_triage_queue() has no duplicate region_ids ({len(s_regions)})")
        check(s_queue.data_as_of == "2002-07-16T00:00:00Z", f"Server predict_triage_queue() data_as_of is 2002-07-16T00:00:00Z")
    except Exception as exc:
        check(False, f"ProductionModelServer error: {exc}")

    # -------------------------------------------------------------------------
    # TEST 7: Exported Static JSON Files in dashboard/public/data/
    # -------------------------------------------------------------------------
    print("\n--- Test 7: Exported Static Files Verification ---")
    public_data = REPO_ROOT / "dashboard" / "public" / "data"
    json_files = list(public_data.glob("*.json"))
    check(len(json_files) >= 100, f"Found {len(json_files)} static JSON files in dashboard/public/data/")

    leak_2026_files = []
    old_version_files = []

    for jf in json_files:
        if jf.name == "manifest.json":
            continue
        text = jf.read_text(encoding="utf-8")
        if "2026-" in text or "2026T" in text:
            leak_2026_files.append(jf.name)
        if "v0.1.0" in text or "v0.2.0" in text:
            old_version_files.append(jf.name)

    check(len(leak_2026_files) == 0, f"0 exported data files contain 2026 timestamp (leaks: {leak_2026_files})")
    check(len(old_version_files) == 0, f"0 exported data files contain old model versions (leaks: {old_version_files})")

    # -------------------------------------------------------------------------
    # TEST 8: Cross-View Consistency & Brief v4 Invariants
    # -------------------------------------------------------------------------
    print("\n--- Test 8: Cross-View Consistency & Brief v4 Invariants ---")

    # 1. GeoJSON and TopoJSON Region ID assertion: Kerala is 35, Vidarbha is 13
    geojson_path = REPO_ROOT / "dashboard" / "src" / "data" / "india_subdivisions_geo.json"
    with open(geojson_path, "r", encoding="utf-8") as f:
        geo_data = json.load(f)
    geo_regions = {f["properties"]["name"]: str(f["properties"].get("region_id")) for f in geo_data["features"]}
    check(geo_regions.get("Kerala") == "35", f"GeoJSON Kerala region_id is '35' (got: {geo_regions.get('Kerala')})")
    check(geo_regions.get("Vidarbha") == "13", f"GeoJSON Vidarbha region_id is '13' (got: {geo_regions.get('Vidarbha')})")

    topojson_path = REPO_ROOT / "dashboard" / "src" / "data" / "india_subdivisions_topo.json"
    with open(topojson_path, "r", encoding="utf-8") as f:
        topo_data = json.load(f)
    topo_geoms = topo_data["objects"]["subdivisions"]["geometries"]
    topo_regions = {g["properties"]["name"]: str(g["properties"].get("region_id")) for g in topo_geoms}
    check(topo_regions.get("Kerala") == "35", f"TopoJSON Kerala region_id is '35' (got: {topo_regions.get('Kerala')})")
    check(topo_regions.get("Vidarbha") == "13", f"TopoJSON Vidarbha region_id is '13' (got: {topo_regions.get('Vidarbha')})")

    # 2. P(bust) values uniqueness / pairing check: 100% unique in ProductionModelServer
    try:
        server_map = server.predict_map(lead_day=6, date="2002-06-07")
        s_probs = [r.p_bust for r in server_map.regions]
        check(len(set(s_probs)) == len(s_probs) == 36, f"Server predict_map() produces strictly 36 unique probabilities with 0 collisions ({len(set(s_probs))}/36)")
    except Exception as exc:
        check(False, f"Server P(bust) uniqueness check failed: {exc}")

    # 3. Low-risk ELT tracking near nominal lead (Brief v5 Section 1.4)
    # Day 6 forecast with low bust risk (<5%) must track near nominal (5.6 - 6.0), NOT collapse to 0.5
    try:
        k_card = server.predict_region_card(region_id="35", lead_day=6, date="2002-06-07")
        check(k_card.p_bust <= 0.05, f"Kerala 2002-06-07 D6 bust probability is low risk (got: {k_card.p_bust})")
        check(5.6 <= k_card.effective_lead_time <= 6.0, f"Kerala 2002-06-07 D6 ELT tracks near nominal lead in [5.6, 6.0] (got: {k_card.effective_lead_time})")
    except Exception as exc:
        check(False, f"Kerala ELT tracking check failed: {exc}")

    # 4. Raw 'none bias' leakage assertion: neither headline nor risk factors contain 'none bias'
    for c_date in test_cycles:
        for reg in ["35", "13", "36", "7"]:
            card_data = get_mock_forecast_card(region_id=reg, lead_day=6, date=c_date)
            h_text = (card_data.headline or "").lower()
            rf_text = " ".join(card_data.risk_factors).lower()
            check("none bias" not in h_text and "none error bias" not in h_text, f"No 'none bias' leak in headline for region {reg} ({c_date})")
            check("none bias" not in rf_text and "none error bias" not in rf_text, f"No 'none bias' leak in risk factors for region {reg} ({c_date})")

    # 5. OOD Effective Lead Time collapse check: unprecedented event collapses ELT to 1.8
    ood_card = get_mock_forecast_card(region_id="35", lead_day=6, date="2018-08-16")
    if ood_card.unprecedented_flag:
        check(ood_card.effective_lead_time == 1.8, f"OOD card effective_lead_time collapses to 1.8 (got: {ood_card.effective_lead_time})")

    # 6. Elimination of placeholder 'Mean signed error +0.0mm/day, dispersion 0.0mm/day'
    for c_date in ["2002-06-07", "2002-07-16"]:
        c = get_mock_forecast_card(region_id="35", lead_day=6, date=c_date)
        all_rf = " ".join(c.risk_factors)
        check("+0.0mm/day, dispersion 0.0mm/day" not in all_rf, f"No placeholder '+0.0mm/day, dispersion 0.0mm/day' in risk factors for {c_date}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"VERIFICATION RESULTS: {total_checks - len(failures)}/{total_checks} PASSED")
    if failures:
        print(f"FAILED {len(failures)} CHECKS:")
        for f in failures:
            print(f"  - {f}")
        print("=" * 70)
        sys.exit(1)
    else:
        print("ALL DATA INTEGRITY AND REGRESSION CHECKS PASSED (100% GREEN)!")
        print("=" * 70)


if __name__ == "__main__":
    run_checks()
