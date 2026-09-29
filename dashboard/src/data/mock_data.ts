/**
 * Deterministic Mock Data Provider for Pratyaya Dashboard
 * Replicates the deterministic algorithms in src/pratyaya/api/mock_provider.py
 * to guarantee 100% offline demo capability without network dependencies.
 */

import {
  AnalogueItem,
  BAND_OBSERVED_BUST_RATES,
  BAND_THRESHOLDS,
  BustDirection,
  ConfidenceBand,
  ConfidenceMap,
  ErrorProneAreas,
  ForecastCard,
  ModelStatus,
  QueueItem,
  RegimesResponse,
  ReplayFrame,
  ReplayScenario,
  ScenarioSummary,
  SubdivisionForecast,
  SubdivisionReplayPrediction,
} from '../api/types';

export const CANONICAL_MODEL_VERSION = 'pratyaya-v1.0.0-loyo';
export const CANONICAL_ARCHIVE_SPAN = '40 years (1982–2022)';

export const ALL_36_SUBDIVISIONS: Array<[string, string]> = [
  ['1', 'Lakshadweep'],
  ['2', 'Andaman & Nicobar Islands'],
  ['3', 'Coastal Andhra Pradesh'],
  ['4', 'Arunachal Pradesh'],
  ['5', 'Assam & Meghalaya'],
  ['6', 'Bihar'],
  ['7', 'Konkan & Goa'],
  ['8', 'Saurashtra & Kutch'],
  ['9', 'Haryana, Chandigarh & Delhi'],
  ['10', 'Himachal Pradesh'],
  ['11', 'Jammu & Kashmir'],
  ['12', 'South Interior Karnataka'],
  ['13', 'Vidarbha'],
  ['14', 'West Madhya Pradesh'],
  ['15', 'Madhya Maharashtra'],
  ['16', 'Nagaland, Manipur, Mizoram & Tripura'],
  ['17', 'Odisha'],
  ['18', 'Punjab'],
  ['19', 'West Rajasthan'],
  ['20', 'Tamil Nadu & Puducherry'],
  ['21', 'East Uttar Pradesh'],
  ['22', 'Gangetic West Bengal'],
  ['23', 'Sub-Himalayan West Bengal & Sikkim'],
  ['24', 'Jharkhand'],
  ['25', 'Uttarakhand'],
  ['26', 'East Rajasthan'],
  ['27', 'Gujarat Region'],
  ['28', 'West Uttar Pradesh'],
  ['29', 'Telangana'],
  ['30', 'Rayalaseema'],
  ['31', 'Chhattisgarh'],
  ['32', 'East Madhya Pradesh'],
  ['33', 'Coastal Karnataka'],
  ['34', 'North Interior Karnataka'],
  ['35', 'Kerala'],
  ['36', 'Marathwada'],
];

export const METEOROLOGICAL_REASONS_BY_ZONE: Record<string, string[]> = {
  west_coast: [
    'Extreme boundary-layer moisture flux convergence along Western Ghats orographic barrier',
    'Strong low-level cross-equatorial Somali Jet acceleration (>25 m/s at 850 hPa) with offshore troughing',
    'Mid-tropospheric dry continental air intrusion capping coastal orographic ascent over Malabar',
    'Upper-tropospheric easterly jet shear suppressing convective cell propagation over Western Ghats',
    'Offshore trough stagnation producing localized convective training along coastal Karnataka',
    'Intense low-level jet cyclonic curvature inducing heavy rainfall bands over Goa coast',
    'Mid-tropospheric temperature inversion breaking down under severe orographic forcing',
    'Anomalous coastal convergence cell anchoring intense rain bands along the Konkan escarpment',
  ],
  islands: [
    'Deep equatorial maritime convective cluster organization over Lakshadweep waters',
    'Cross-equatorial buffer zone shift producing anomalous shear lines across south Andaman Sea',
    'Inter-Tropical Convergence Zone (ITCZ) convective surge across Andaman & Nicobar archipelago',
    'Low-latitude cyclonic vorticity gyre spinning over southeast Arabian Sea near Minicoy',
    'High sea-surface temperature anomaly (>29.5°C) triggering localized nocturnal maritime squalls',
  ],
  interior_peninsular: [
    'Negative moist static energy anomaly over Marathwada inducing severe convective suppression',
    'Strong lee-side rain shadow subsidence east of Western Ghats barrier over Madhya Maharashtra',
    'Mid-tropospheric dry air entrainment eroding rain-fed cotton belt convective cells over Vidarbha',
    'Low-level divergent wind anomaly capping diurnal convective instability in Rayalaseema',
    'Sharply sheared mid-tropospheric flow suppressing afternoon thunderstorm development in North Interior Karnataka',
    'Thermal inversion at 700 hPa preventing vertical cloud growth over South Interior Karnataka plateau',
    'South peninsular dry spell anomaly exacerbated by anomalous mid-tropospheric anticyclone',
    'Subsidence in right-exit quadrant of upper-level easterly jet over interior Tamil Nadu',
  ],
  bay_central: [
    'Mid-tropospheric cyclonic circulation developing over northwest Bay of Bengal with track bifurcation',
    'Monsoon low-pressure system propagating across Odisha into Central India with intense rain bands',
    'Northward deflection of Bay depression core away from rain-fed agricultural valleys in Telangana',
    'Strong cyclonic shear vorticity anomaly at 850 hPa across coastal Odisha and Andhra Pradesh',
    'Coastal moisture surging inland ahead of deepening low-pressure anomaly over head Bay',
    'Low-pressure system deepening over head Bay with delayed inland propagation across Chhattisgarh',
    'Broad mid-tropospheric cyclonic vorticity over Mahanadi and Godavari river catchments',
    'Trough interaction over East Madhya Pradesh enhancing localized boundary-layer convergence',
  ],
  ganga_plains: [
    'Monsoon trough axis displaced south of normal position over Ganga Basin into peninsular shear zone',
    'Sudden northward migration of monsoon trough to Himalayan foothills inducing localized plains break',
    'Interaction between western disturbance midlatitude trough and moist monsoon easterlies over UP',
    'Boundary-layer thermal inversion over eastern UP capping moist convective instability',
    'Baroclinic energy conversion along sharp tropospheric temperature gradient across Punjab and Haryana',
    'Intense dry continental northwesterly advection undercutting monsoon easterlies in Bihar',
    'Foothill convergence zone triggering unpredicted nocturnal flash flood cells in North Bihar',
    'Low-level jet speed convergence over West Uttar Pradesh producing localized severe downpours',
  ],
  northwest_arid: [
    'Mid-tropospheric cyclone (MTC) spinning over Gujarat-Saurashtra vortex producing rain surges',
    'Dry continental northwesterly wind advection over Rajasthan suppressing monsoon advance',
    'Thermal low deepening over West Rajasthan without corresponding Arabian Sea moisture influx',
    'Extreme low-level heat low intensity over Kutch without upper-level convective support',
    'Dry air entrainment from the Thar desert eroding monsoon convective cloud shields over East Rajasthan',
    'Anomalous mid-tropospheric ridge over Gujarat preventing convective cell organization',
  ],
  northeast_hills: [
    'Severe orographic lifting of southerly moisture flux against southern slopes of Meghalaya plateau',
    'Upper-level divergence in right-entrance region of subtropical westerly jet over Western Himalayas',
    'Brahmaputra valley moisture channeling with low-level nocturnal jet acceleration',
    'Baroclinic wave packet breaking over Tibetan plateau spilling cold shear into Arunachal',
    'Intense localized squall-line generation along sub-Himalayan front in Sikkim and North Bengal',
    'Strong orographic windward ascent over Khasi and Garo hills under deep moisture flow',
    'Southerly moisture convergence funneled through Surma valley into southern Assam',
    'Steep elevation lapse rate triggering sudden convective flash-flood bursts in Uttarakhand',
  ],
  general: [
    'Madden-Julian Oscillation (MJO) phase entering active convective envelope over Indian Ocean',
    'Equatorial Rossby wave passage superimposing cyclonic vorticity over south peninsular shear zone',
    'Multi-model ensemble member dispersion indicates severe synoptic track bifurcation',
    'Atmospheric precipitable water anomaly exceeds +2.4 sigma above local climatology',
    'Subtropical Rossby wave train imposing anomalous mid-tropospheric ridge over Central India',
    'Rapid zonal translation of upper-tropospheric easterly wave perturbation',
  ],
};

export function getSubdivisionZone(regionId: string): string {
  const rId = String(regionId);
  if (['7', '33', '35'].includes(rId)) return 'west_coast';
  if (['1', '2'].includes(rId)) return 'islands';
  if (['12', '13', '15', '20', '30', '34', '36'].includes(rId)) return 'interior_peninsular';
  if (['3', '14', '17', '22', '24', '29', '31', '32'].includes(rId)) return 'bay_central';
  if (['6', '9', '18', '21', '28'].includes(rId)) return 'ganga_plains';
  if (['8', '19', '26', '27'].includes(rId)) return 'northwest_arid';
  return 'northeast_hills';
}

export function getSubdivisionRiskFactor(
  regionId: string,
  leadDay: number,
  date: string,
  usedReasons?: Set<string>
): string {
  const zone = getSubdivisionZone(regionId);
  const pool = METEOROLOGICAL_REASONS_BY_ZONE[zone] || METEOROLOGICAL_REASONS_BY_ZONE.general;
  const h = deterministicHash(`${date}:${leadDay}:${regionId}:factor`);
  const baseIdx = h % pool.length;

  for (let offset = 0; offset < pool.length; offset++) {
    const candidate = pool[(baseIdx + offset) % pool.length];
    if (!usedReasons || !usedReasons.has(candidate)) {
      usedReasons?.add(candidate);
      return candidate;
    }
  }

  const genPool = METEOROLOGICAL_REASONS_BY_ZONE.general;
  for (let offset = 0; offset < genPool.length; offset++) {
    const candidate = genPool[(baseIdx + offset) % genPool.length];
    if (!usedReasons || !usedReasons.has(candidate)) {
      usedReasons?.add(candidate);
      return candidate;
    }
  }

  const fallback = pool[baseIdx % pool.length];
  usedReasons?.add(fallback);
  return fallback;
}

// Simple deterministic hash for browser JS (djb2 style)
function deterministicHash(seed: string): number {
  let hash = 5381;
  for (let i = 0; i < seed.length; i++) {
    hash = (hash * 33) ^ seed.charCodeAt(i);
  }
  return Math.abs(hash);
}

function classifyConfidenceBand(pBust: number): ConfidenceBand {
  if (pBust < BAND_THRESHOLDS.HIGH_MAX) return 'HIGH';
  if (pBust < BAND_THRESHOLDS.MODERATE_MAX) return 'MODERATE';
  if (pBust < BAND_THRESHOLDS.LOW_MAX) return 'LOW';
  return 'CRITICAL';
}

export function getMockConfidenceMap(
  leadDay: number = 6,
  date: string = '2002-07-16',
  variable: string = 'rainfall'
): ConfidenceMap {
  const regions: SubdivisionForecast[] = ALL_36_SUBDIVISIONS.map(([regId, name]) => {
    const hashVal = deterministicHash(`${date}:${leadDay}:${regId}`) % 10000;
    const baseRate = 0.04 + (leadDay - 1) * 0.035; // scales with lead
    const noise = (hashVal / 10000.0) * 0.45;
    
    const is2018 = date === '2018-08-16' || date.startsWith('2018-08');
    // Inject known synoptic hot spots for realism (e.g. West Coast, Gujarat, Bay of Bengal depression tracks)
    const isHotspot = ['7', '8', '35', '30', '36', '6', '14', '33'].includes(regId);
    // Regional offset guarantees 100% unique pBust down triage list (no duplicate pairings)
    const regOffset = ((parseInt(regId, 10) * 19) % 100) / 10000;
    let pBust = is2018
      ? Math.min(0.96, Math.max(0.52, 0.65 + noise * 0.4 + (isHotspot ? 0.18 : 0.04) + regOffset))
      : Math.min(0.92, Math.max(0.02, baseRate + noise + (isHotspot ? 0.22 : -0.05) + regOffset));
    pBust = Math.round(pBust * 1000) / 1000;

    // Pre-committed Kerala benchmark cases (Decision D-023) - ONLY subdivision 35
    let unprecedented = is2018 ? (regId === '35' || hashVal % 4 === 0) : hashVal % 97 === 0;
    if (regId === '35') {
      if (is2018) {
        pBust = 0.88; // extreme
        unprecedented = true;
      } else {
        pBust = 0.324; // 32.4% calibrated bust probability
        unprecedented = false;
      }
    }

    const band = classifyConfidenceBand(pBust);
    let elt = unprecedented ? 1.8 : Math.min(12.0, Math.max(0.5, Math.round((leadDay + (pBust * 5.2) - 0.4) * 10) / 10));
    if (regId === '35') {
      elt = (date === '2018-08-16' || date.startsWith('2018-08')) ? 1.8 : 8.8;
    }

    // Direction
    const dirHash = deterministicHash(`${date}:dir:${regId}`) % 100;
    const bustDir: BustDirection = regId === '35'
      ? ((date === '2018-08-16' || date.startsWith('2018-08')) ? 'wet' : 'dry')
      : pBust < 0.1 ? 'none' : dirHash > 40 ? 'wet' : 'dry';

    return {
      region_id: regId,
      name,
      effective_lead_time: elt,
      p_bust: pBust,
      confidence_band: band,
      p_bust_p90: Math.min(1.0, Math.round((pBust * 1.25) * 100) / 100),
      bust_direction: bustDir,
      unprecedented_flag: unprecedented,
    };
  });

  return {
    lead_day: leadDay,
    date,
    regions,
    variable,
    model_version: CANONICAL_MODEL_VERSION,
    data_as_of: `${date}T00:00:00Z`,
    is_live: false,
  };
}

const HISTORICAL_SYNOPTIC_DATES = [
  '2006-08-02', '2011-07-27', '2013-08-18', '2005-07-26',
  '2007-06-21', '2012-09-04', '2016-08-10', '2004-07-14',
  '2008-08-09', '2015-07-19', '2003-08-28', '2010-09-12',
  '2014-06-18', '2009-07-30', '2017-08-29', '2001-08-15'
];

export function getMockForecastCard(
  regionId: string = '30',
  leadDay: number = 6,
  date: string = '2002-07-16'
): ForecastCard {
  const regTuple = ALL_36_SUBDIVISIONS.find(([id]) => id === regionId) || ['30', 'Rayalaseema'];
  const regionName = regTuple[1];

  const mapData = getMockConfidenceMap(leadDay, date);
  const regForecast = mapData.regions.find((r) => r.region_id === regionId) || {
    region_id: regionId,
    name: regionName,
    effective_lead_time: 9.4,
    p_bust: 0.58,
    confidence_band: 'CRITICAL' as ConfidenceBand,
    bust_direction: 'wet' as BustDirection,
    unprecedented_flag: false,
  };

  // Pre-committed Kerala benchmarks (Decision D-023) - ONLY Subdivision 35
  if (regionId === '35') {
    if (date === '2018-08-16' || date.startsWith('2018-08')) {
      // Case B: August 2018 OOD safety refusal
      return {
        region_id: regionId,
        region_name: regionName,
        lead_day: leadDay,
        init_date: date,
        effective_lead_time: 1.8,
        p_bust: 0.88,
        p_bust_p90: 0.95,
        confidence_band: 'CRITICAL',
        bust_direction: 'wet',
        unprecedented_flag: true,
        headline: 'OOD Refusal: Circulation vector exceeds historical distribution',
        top_analogues: [],
        risk_factors: [
          'Circulation vector Mahalanobis distance exceeds 99.9th percentile threshold.',
          'Unprecedented August 2018 extreme moisture influx anomaly (>95th percentile).',
          'System safely refuses analogue matching to avoid deceptive guidance.',
        ],
        observed_bust_rate_in_band: BAND_OBSERVED_BUST_RATES['CRITICAL'],
        model_version: CANONICAL_MODEL_VERSION,
        data_as_of: `${date}T00:00:00Z`,
        is_live: false,
      };
    } else {
      // Case A: June 7, 2002 D6 benchmark case
      return {
        region_id: regionId,
        region_name: regionName,
        lead_day: leadDay,
        init_date: date,
        effective_lead_time: 8.8,
        p_bust: 0.324,
        p_bust_p90: 0.421,
        confidence_band: 'LOW',
        bust_direction: 'dry',
        unprecedented_flag: false,
        headline: `Day ${leadDay} forecast · behaving like Day 8.8 (32.4% bust risk — dry bias)`,
        top_analogues: [
          {
            date: '2000-06-18',
            weight: 0.285,
            outcome: 'BUST (dry)',
            observed_value: 14.7,
            forecast_value: 75.4,
            distance: 2.14,
            bust_direction: 'dry',
            is_bust: true,
            closeness_rank: 1,
            mechanism: 'GEFS over-intensified Arabian Sea offshore trough moisture convergence (75.4 mm NWP vs 14.7 mm Obs).',
          },
          {
            date: '2001-07-04',
            weight: 0.242,
            outcome: 'BUST (dry)',
            observed_value: 12.4,
            forecast_value: 71.0,
            distance: 2.38,
            bust_direction: 'dry',
            is_bust: true,
            closeness_rank: 2,
            mechanism: 'Low-level cross-equatorial Somali jet core displacement offshore; convective squalls remained maritime.',
          },
          {
            date: '2000-07-12',
            weight: 0.198,
            outcome: 'BUST (dry)',
            observed_value: 18.0,
            forecast_value: 64.5,
            distance: 2.85,
            bust_direction: 'dry',
            is_bust: true,
            closeness_rank: 3,
            mechanism: 'Mid-tropospheric dry continental air intrusion capping coastal orographic ascent over Western Ghats.',
          },
          {
            date: '2002-06-02',
            weight: 0.145,
            outcome: 'no bust',
            observed_value: 39.8,
            forecast_value: 42.0,
            distance: 3.12,
            bust_direction: 'none',
            is_bust: false,
            closeness_rank: 4,
            mechanism: 'Balanced vorticity advection and verified precipitation envelope along Malabar coast.',
          },
          {
            date: '2001-06-28',
            weight: 0.130,
            outcome: 'no bust',
            observed_value: 51.2,
            forecast_value: 55.0,
            distance: 3.45,
            bust_direction: 'none',
            is_bust: false,
            closeness_rank: 5,
            mechanism: 'Active coastal convergence with accurate deterministic phase alignment.',
          },
        ],
        risk_factors: [
          'GEFS over-intensified Arabian Sea offshore trough convergence leading to severe false alarm dry bust.',
          'Upper-tropospheric easterly jet shear suppressed active convective cell propagation over Western Ghats.',
          'Mahalanobis metric distance 2.14 matches June 2000 & July 2001 false alarm signatures.',
        ],
        observed_bust_rate_in_band: BAND_OBSERVED_BUST_RATES['LOW'],
        model_version: CANONICAL_MODEL_VERSION,
        data_as_of: `${date}T00:00:00Z`,
        is_live: false,
      };
    }
  }

  // Generate top 8 analogues strictly from training window (2000-2017)
  const analogues: AnalogueItem[] = [];
  const baseSeed = deterministicHash(`${date}:${regionId}:${leadDay}`);

  for (let i = 0; i < 8; i++) {
    const aDate = HISTORICAL_SYNOPTIC_DATES[(baseSeed + i * 3) % HISTORICAL_SYNOPTIC_DATES.length];
    const weight = Math.round((0.96 - i * 0.038) * 1000) / 1000;
    const distance = Math.round((1.8 + i * 0.42 + (baseSeed % 10) * 0.05) * 100) / 100;
    
    // Near misses vs busts (rules: must show outcomes, not just dates, and show near misses too!)
    const isBust = i === 2 || i === 5 ? false : true;
    const fcVal = Math.round((15 + ((baseSeed + i * 7) % 35)) * 10) / 10;
    let obsVal = fcVal;
    let outcome = 'no bust';
    let bustDir: BustDirection = 'none';

    if (isBust) {
      if ((baseSeed + i) % 2 === 0) {
        obsVal = Math.round((fcVal * 3.8 + 20) * 10) / 10;
        outcome = 'BUST (wet)';
        bustDir = 'wet';
      } else {
        obsVal = Math.round(Math.max(1.0, fcVal * 0.2) * 10) / 10;
        outcome = 'BUST (dry)';
        bustDir = 'dry';
      }
    } else {
      obsVal = Math.round((fcVal + ((baseSeed % 5) - 2) * 2.1) * 10) / 10;
    }

    analogues.push({
      date: aDate,
      weight,
      outcome,
      observed_value: obsVal,
      forecast_value: fcVal,
      distance,
      bust_direction: bustDir,
      is_bust: isBust,
      closeness_rank: i + 1,
    });
  }

  // Supporting plain-language risk factors parameterized by zone (no generic duplicates)
  const zone = getSubdivisionZone(regionId);
  const zPool = METEOROLOGICAL_REASONS_BY_ZONE[zone] || METEOROLOGICAL_REASONS_BY_ZONE.general;
  const genPool = METEOROLOGICAL_REASONS_BY_ZONE.general;
  const r1 = zPool[baseSeed % zPool.length];
  const r2 = zPool[(baseSeed + 1) % zPool.length];
  const r3 = genPool[(baseSeed + 2) % genPool.length];
  const riskFactors = [r1, r2, r3];

  const dirStr = regForecast.bust_direction === 'wet' ? ' — wet bias' : regForecast.bust_direction === 'dry' ? ' — dry bias' : ' — balanced error profile';
  const headline = `Day ${leadDay} forecast · behaving like Day ${regForecast.effective_lead_time.toFixed(1)} (${(regForecast.p_bust * 100).toFixed(0)}% bust risk${dirStr})`;

  return {
    region_id: regionId,
    region_name: regionName,
    lead_day: leadDay,
    init_date: date,
    effective_lead_time: regForecast.effective_lead_time,
    p_bust: regForecast.p_bust,
    p_bust_p90: Math.min(1.0, Math.round(regForecast.p_bust * 1.3 * 100) / 100),
    confidence_band: regForecast.confidence_band,
    bust_direction: regForecast.bust_direction || 'wet',
    unprecedented_flag: regForecast.unprecedented_flag,
    headline,
    top_analogues: analogues,
    risk_factors: riskFactors,
    observed_bust_rate_in_band: BAND_OBSERVED_BUST_RATES[regForecast.confidence_band],
    model_version: CANONICAL_MODEL_VERSION,
    data_as_of: `${date}T00:00:00Z`,
    is_live: false,
  };
}

export function getMockReviewQueue(
  leadDay?: number,
  date: string = '2002-07-16',
  topN: number = 10
): ErrorProneAreas {
  // Target active lead (defaults to 6)
  const targetLead = leadDay ?? 6;
  const cmap = getMockConfidenceMap(targetLead, date);

  // Exactly one entry per subdivision from the 36 canonical subdivisions
  const rawItems: Array<Omit<QueueItem, 'top_risk_factor'>> = [];

  for (const reg of cmap.regions) {
    // Decision weights: high agricultural/population subdivisions carry higher operational weight
    const decisionWeight = [
      '6', '7', '8', '14', '15', '20', '21', '22', '27', '29', '30', '35'
    ].includes(reg.region_id) ? 1.45 : 1.0;

    const priorityScore = Math.round((reg.p_bust * decisionWeight) * 1000) / 1000;

    rawItems.push({
      rank: 0,
      region_id: reg.region_id,
      region_name: reg.name,
      lead_day: targetLead,
      p_bust: reg.p_bust,
      decision_weight: decisionWeight,
      priority_score: priorityScore,
      effective_lead_time: reg.effective_lead_time,
      confidence_band: reg.confidence_band,
      bust_direction: reg.bust_direction || 'wet',
    });
  }

  // Sort strictly descending by priority_score = p_bust * decision_weight
  rawItems.sort((a, b) => b.priority_score - a.priority_score);

  // Assign strictly unique, non-repeating risk factors to selected queue items
  const usedReasons = new Set<string>();

  const topQueue: QueueItem[] = rawItems.slice(0, topN).map((item, idx) => {
    const reason = getSubdivisionRiskFactor(item.region_id, targetLead, date, usedReasons);
    return {
      ...item,
      rank: idx + 1,
      top_risk_factor: reason,
    };
  });

  return {
    date,
    lead_day: targetLead,
    total_items: topQueue.length,
    queue: topQueue,
    model_version: CANONICAL_MODEL_VERSION,
    data_as_of: `${date}T00:00:00Z`,
    is_live: false,
  };
}

export function getMockModelStatus(): ModelStatus {
  return {
    model_version: CANONICAL_MODEL_VERSION,
    run_id: 'run-loyo-benchmark-2000-2002',
    calibration_date: '2022-09-30',
    status: 'ready',
    training_window: '2000-2002 JJAS (3-Fold Purged LOYO-CV)',
    calibration_window: 'Out-of-fold Isotonic Regression',
    heldout_window: '43,200 Real Evaluated Records (120 cycles)',
    baselines_available: ['B0', 'B1', 'B2', 'B3', 'M1_v2', 'M2'],
    is_live: false,
    data_as_of: '2022-09-30T00:00:00Z',
    heldout_metrics: {
      B0: {
        pr_auc: 0.0423,
        pr_auc_ci: [0.0379, 0.0470],
        auroc: 0.4744,
        brier_score: 0.0427,
        reliability: 0.0000,
        resolution: 0.0000,
        bss: '+0.0613',
        description: 'Constant climatological base rate floor (4.47%)',
      },
      B1: {
        pr_auc: 0.0366,
        pr_auc_ci: [0.0329, 0.0404],
        auroc: 0.4227,
        brier_score: 0.0420,
        reliability: 0.0020,
        resolution: 0.0002,
        bss: '0.0000',
        description: 'Conditional climatology (Subdivision/lead-day bar)',
      },
      B2: {
        pr_auc: 0.1532,
        pr_auc_ci: [0.1340, 0.1735],
        auroc: 0.7633,
        brier_score: 0.0378,
        reliability: 0.0005,
        resolution: 0.0019,
        bss: '+0.1008',
        description: 'Raw GEFS ensemble spread (uncalibrated std dev)',
      },
      B3: {
        pr_auc: 0.1667,
        pr_auc_ci: [0.1483, 0.1845],
        auroc: 0.7860,
        brier_score: 0.0367,
        reliability: 0.0001,
        resolution: 0.0026,
        bss: '+0.1282',
        description: 'Calibrated logistic spread benchmark (Decision D-020)',
      },
      M1_v2: {
        pr_auc: 0.0412,
        pr_auc_ci: [0.0374, 0.0453],
        auroc: 0.5030,
        brier_score: 0.0422,
        reliability: 0.0027,
        resolution: 0.0000,
        bss: '-0.0048',
        description: 'Pure Analogue Memory Engine (Low-rank metric, zero ensemble)',
      },
      M2_Pratyaya: {
        pr_auc: 0.1900,
        pr_auc_ci: [0.1680, 0.2098],
        auroc: 0.7843,
        brier_score: 0.0368,
        reliability: 0.0005,
        resolution: 0.0030,
        bss: '+0.1245',
        description: 'Pratyaya Analogue-NWP Fused (Isotonic Calibrated Benchmark)',
      },
    },
  };
}

export function getMockReplayScenarios(): ScenarioSummary[] {
  return [
    {
      id: 'july2002_break',
      title: 'Case 1: July 2002 Break Monsoon Onset',
      description:
        'Severe drought and break monsoon onset. NWP models predicted active monsoon continuation across Central India, but synoptic analogue engine detected catastrophic suppression 4–7 days ahead.',
      cycle_date: '2002-07-07',
      date_range: '2002-07-07 to 2002-07-14',
      leads: 'Lead Days 1 to 7',
    },
    {
      id: 'july2001_depression',
      title: 'Case 2: July 2001 Active Monsoon Depression',
      description:
        'Deep monsoon depression developing over northwest Bay of Bengal and tracking west-northwestward with heavy orographic enhancement along Western Ghats.',
      cycle_date: '2001-07-01',
      date_range: '2001-07-01 to 2001-07-08',
      leads: 'Lead Days 1 to 7',
    },
  ];
}

export function getMockReplay(
  scenarioId: string = 'july2002_break',
  maxLead: number = 7
): ReplayScenario {
  const isCase2 = scenarioId.includes('2001') || scenarioId === 'case2';
  const canonicalId = isCase2 ? 'july2001_depression' : 'july2002_break';
  const title = isCase2
    ? 'Case 2: July 2001 Active Monsoon Depression'
    : 'Case 1: July 2002 Break Monsoon Onset';
  const desc = isCase2
    ? 'Deep monsoon depression developing over northwest Bay of Bengal and tracking west-northwestward with heavy orographic enhancement along Western Ghats.'
    : 'Severe drought and break monsoon onset. NWP models predicted active monsoon continuation across Central India, but synoptic analogue engine detected catastrophic suppression 4–7 days ahead.';
  const cycleDate = isCase2 ? '2001-07-01' : '2002-07-07';

  const numLeads = Math.min(Math.max(maxLead, 1), 10);
  const leadDays: number[] = Array.from({ length: numLeads }, (_, i) => i + 1);

  const frames: ReplayFrame[] = leadDays.map((ld) => {
    const validDate = `200${isCase2 ? '1' : '2'}-07-${String(ld + (isCase2 ? 1 : 8)).padStart(2, '0')}`;
    const map = getMockConfidenceMap(ld, cycleDate);
    const queue = getMockReviewQueue(ld, cycleDate, 10);

    const subdivisions: SubdivisionReplayPrediction[] = map.regions.map((reg) => {
      const isDry = reg.bust_direction === 'dry';
      const nwpRain = +(15.0 + ld * 2.2).toFixed(1);
      const obsRain = +(isDry ? nwpRain * 0.15 : nwpRain * 2.4).toFixed(1);
      return {
        region_id: reg.region_id,
        name: reg.name,
        p_bust: reg.p_bust,
        confidence_band: reg.confidence_band,
        nwp_rainfall: nwpRain,
        obs_rainfall: obsRain,
        effective_lead_time_days: reg.effective_lead_time,
        error_direction: reg.bust_direction || 'none',
        unprecedented_flag: reg.unprecedented_flag,
        p_bust_p90: reg.p_bust_p90,
      };
    });

    const topSubdiv = [...subdivisions].sort((a, b) => b.p_bust - a.p_bust)[0];
    const meanElt = +(
      subdivisions.reduce((sum, s) => sum + s.effective_lead_time_days, 0) /
      subdivisions.length
    ).toFixed(1);

    const pPct = Math.round(topSubdiv.p_bust * 100);
    const spread = +(1.2 + ld * 0.45).toFixed(1);

    let caption = '';
    if (topSubdiv.p_bust >= 0.25) {
      if (topSubdiv.error_direction === 'dry') {
        caption = `Day ${ld} lead: NWP predicts ${topSubdiv.nwp_rainfall} mm/day in ${topSubdiv.name}, but synoptic analogues indicate ${pPct}% probability of dry bust (actual: ${topSubdiv.obs_rainfall} mm/day). Effective Lead Time jumps from Day ${ld}.0 to Day ${topSubdiv.effective_lead_time_days.toFixed(1)} ahead of raw NWP spread (${spread} mm). Forecaster confidence downgraded to ${topSubdiv.confidence_band}.`;
      } else {
        caption = `Day ${ld} lead: NWP predicts ${topSubdiv.nwp_rainfall} mm/day in ${topSubdiv.name}, but analogue engine flags ${pPct}% probability of wet bust surge (actual: ${topSubdiv.obs_rainfall} mm/day). Effective Lead Time expands to Day ${topSubdiv.effective_lead_time_days.toFixed(1)} (NWP spread: ${spread} mm). Forecaster confidence downgraded to ${topSubdiv.confidence_band}.`;
      }
    } else {
      caption = `Day ${ld} lead: Synoptic analogues confirm stable monsoon flow in ${topSubdiv.name} (NWP: ${topSubdiv.nwp_rainfall} mm/day, p_bust = ${pPct}%). Effective Lead Time calibrated at Day ${topSubdiv.effective_lead_time_days.toFixed(1)}. Forecaster confidence remains ${topSubdiv.confidence_band}.`;
    }

    return {
      cycle_date: cycleDate,
      lead_day: ld,
      valid_date: validDate,
      subdivisions,
      caption_strip: caption,
      triage_queue: queue.queue,
      confidence_map: map,
      top_risk_region_id: topSubdiv.region_id,
      top_risk_region_name: topSubdiv.name,
      max_p_bust: topSubdiv.p_bust,
      mean_effective_lead_time: meanElt,
    };
  });

  return {
    scenario_id: canonicalId,
    title,
    description: desc,
    cycle_date: cycleDate,
    lead_days: leadDays,
    frames,
    model_version: CANONICAL_MODEL_VERSION,
    is_live: false,
  };
}

export function getMockRegimes(): RegimesResponse {
  return {
    optimal_k: 9,
    silhouette_scores: {
      '3': 0.0643,
      '4': 0.0661,
      '5': 0.0706,
      '6': 0.0751,
      '7': 0.0792,
      '8': 0.0932,
      '9': 0.0959,
      '10': 0.0881,
    },
    n_high_burden_dates: 29,
    high_burden_threshold: 11.0,
    total_candidate_dates: 366,
    model_version: CANONICAL_MODEL_VERSION,
    data_as_of: '2022-09-30T00:00:00Z',
    is_live: false,
    clusters: [
      {
        cluster_id: 0,
        name: 'Northwest Bay Depression Track',
        mechanism_summary:
          'Monsoon low-pressure system / depression propagating westward across the Bay of Bengal into Central India, producing intense localized rain bands and strong cyclonic shear that NWP models track with phase and displacement errors, leading to severe wet forecast busts.',
        n_members: 16,
        member_dates: ['2000-06-29', '2000-07-03', '2000-07-04', '2000-07-06', '2000-07-07'],
        seasonality: { June: 0.062, July: 0.938, August: 0.0, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '29', name: 'Telangana', bust_rate: 0.375, bust_count: 6, mean_signed_error: 8.4 },
          { region_id: '7', name: 'Odisha', bust_rate: 0.35, bust_count: 7, mean_signed_error: 12.1 },
        ],
        vulnerable_leads: [4, 5, 6, 7],
        signed_bias: 'wet',
        mean_error_mm: 2.07,
        composite_summary: {
          z500: { min: -4.11, max: 15.72, mean: 1.46, std: 3.09 },
          u850: { min: -1.89, max: 4.37, mean: 0.32, std: 0.89 },
          v850: { min: -0.89, max: 2.97, mean: 0.2, std: 0.52 },
          mslp: { min: -5.33, max: 0.76, mean: -0.2, std: 0.65 },
          tcwv: { min: 8.85, max: 16.02, mean: 11.2, std: 1.14 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 1,
        name: 'Subtropical Westerly Trough Intrusion',
        mechanism_summary:
          'Upper-tropospheric midlatitude Rossby wave trough dipping south of 30°N across Northwest India, superimposing strong baroclinic shear and upper-level divergence over Arabian Sea moisture influx, triggering unpredicted convective squalls.',
        n_members: 17,
        member_dates: ['2000-06-01', '2000-06-02', '2000-06-03', '2000-06-04'],
        seasonality: { June: 0.941, July: 0.059, August: 0.0, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '14', name: 'Punjab', bust_rate: 0.353, bust_count: 6, mean_signed_error: 5.8 },
          { region_id: '18', name: 'Haryana', bust_rate: 0.294, bust_count: 5, mean_signed_error: 4.9 },
        ],
        vulnerable_leads: [3, 4, 5, 6],
        signed_bias: 'wet',
        mean_error_mm: 0.99,
        composite_summary: {
          z500: { min: -15.21, max: 4.2, mean: -1.46, std: 3.07 },
          u850: { min: -4.22, max: 1.46, mean: -0.31, std: 0.84 },
          v850: { min: -3.19, max: 0.7, mean: -0.19, std: 0.49 },
          mslp: { min: -0.64, max: 4.81, mean: 0.16, std: 0.6 },
          tcwv: { min: -14.76, max: -8.93, mean: -10.75, std: 0.95 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 2,
        name: 'Northward-Shifted Monsoon Trough / Break Onset',
        mechanism_summary:
          'Northward shift of the monsoon trough to the Himalayan foothills, inducing a widespread break-monsoon phase across Central and Northwest India with convective suppression, positive MSLP anomalies, and severe dry (overpredicted rainfall) forecast busts.',
        n_members: 14,
        member_dates: ['2002-07-04', '2002-07-05', '2002-07-06', '2002-07-07'],
        seasonality: { June: 0.0, July: 1.0, August: 0.0, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '6', name: 'Bihar', bust_rate: 0.42, bust_count: 6, mean_signed_error: -14.2 },
          { region_id: '14', name: 'West Madhya Pradesh', bust_rate: 0.38, bust_count: 5, mean_signed_error: -11.6 },
        ],
        vulnerable_leads: [4, 5, 6, 7],
        signed_bias: 'dry',
        mean_error_mm: -6.4,
        composite_summary: {
          z500: { min: -2.1, max: 8.5, mean: 1.2, std: 1.8 },
          u850: { min: -3.4, max: 0.8, mean: -1.2, std: 0.7 },
          v850: { min: -1.8, max: 0.6, mean: -0.4, std: 0.4 },
          mslp: { min: 0.5, max: 4.2, mean: 1.8, std: 0.6 },
          tcwv: { min: -12.4, max: -3.2, mean: -7.5, std: 1.2 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 3,
        name: 'Mid-Tropospheric Cyclone over Gujarat',
        mechanism_summary:
          'Quasi-stationary mid-tropospheric cyclone (MTC) centered near 700 hPa over Gujarat and Northeast Arabian Sea, driving torrential localized downpours that medium-range global models underestimate.',
        n_members: 8,
        member_dates: ['2000-07-18', '2000-07-19', '2001-08-04'],
        seasonality: { June: 0.125, July: 0.625, August: 0.25, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '8', name: 'Saurashtra & Kutch', bust_rate: 0.5, bust_count: 4, mean_signed_error: 18.5 },
          { region_id: '27', name: 'Gujarat Region', bust_rate: 0.45, bust_count: 4, mean_signed_error: 14.1 },
        ],
        vulnerable_leads: [3, 4, 5],
        signed_bias: 'wet',
        mean_error_mm: 5.2,
        composite_summary: {
          z500: { min: -6.8, max: 3.4, mean: -1.8, std: 2.1 },
          u850: { min: -1.5, max: 4.8, mean: 1.4, std: 1.1 },
          v850: { min: -2.1, max: 3.9, mean: 0.8, std: 0.9 },
          mslp: { min: -4.2, max: 0.4, mean: -1.6, std: 0.8 },
          tcwv: { min: 4.5, max: 14.2, mean: 8.6, std: 1.4 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 4,
        name: 'Cross-Equatorial Somali Jet Surge',
        mechanism_summary:
          'Intense acceleration of the low-level cross-equatorial Somali Jet (>28 m/s at 850 hPa) impacting the Western Ghats, generating intense orographic moisture flux convergence and widespread wet forecast errors along the western coast.',
        n_members: 12,
        member_dates: ['2000-06-15', '2000-06-16', '2001-07-12'],
        seasonality: { June: 0.5, July: 0.417, August: 0.083, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '7', name: 'Konkan & Goa', bust_rate: 0.58, bust_count: 7, mean_signed_error: 22.4 },
          { region_id: '33', name: 'Coastal Karnataka', bust_rate: 0.5, bust_count: 6, mean_signed_error: 19.8 },
          { region_id: '13', name: 'Kerala', bust_rate: 0.42, bust_count: 5, mean_signed_error: 15.6 },
        ],
        vulnerable_leads: [4, 5, 6, 7],
        signed_bias: 'wet',
        mean_error_mm: 6.8,
        composite_summary: {
          z500: { min: -5.4, max: 4.2, mean: -0.8, std: 1.6 },
          u850: { min: 1.2, max: 7.8, mean: 3.6, std: 1.4 },
          v850: { min: -0.4, max: 3.1, mean: 1.1, std: 0.6 },
          mslp: { min: -3.8, max: 0.2, mean: -1.4, std: 0.7 },
          tcwv: { min: 6.2, max: 18.5, mean: 12.1, std: 1.8 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 5,
        name: 'Active Arabian Sea Offshore Trough',
        mechanism_summary:
          'Shallow coastal offshore trough developing along the Konkan-Goa and Malabar coast, concentrating intense boundary-layer moisture convergence that deterministic NWP models displace offshore.',
        n_members: 10,
        member_dates: ['2000-06-30', '2000-07-01', '2001-06-09'],
        seasonality: { June: 0.5, July: 0.5, August: 0.0, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '7', name: 'Konkan & Goa', bust_rate: 0.4, bust_count: 4, mean_signed_error: 14.8 },
          { region_id: '33', name: 'Coastal Karnataka', bust_rate: 0.4, bust_count: 4, mean_signed_error: 16.2 },
        ],
        vulnerable_leads: [4, 5, 6, 7],
        signed_bias: 'wet',
        mean_error_mm: 1.75,
        composite_summary: {
          z500: { min: -8.17, max: 8.23, mean: 0.47, std: 2.0 },
          u850: { min: -2.1, max: 2.28, mean: 0.15, std: 0.52 },
          v850: { min: -1.2, max: 1.71, mean: 0.09, std: 0.39 },
          mslp: { min: -2.93, max: 0.83, mean: -0.1, std: 0.42 },
          tcwv: { min: 2.93, max: 9.48, mean: 4.77, std: 0.79 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 6,
        name: 'Peninsular Wind Shear Disruption',
        mechanism_summary:
          'Meridional wind shear disruption across the southern peninsula with suppressed zonal westerlies, causing widespread rainfall deficits and persistent dry forecast busts over Tamil Nadu and Interior Karnataka.',
        n_members: 9,
        member_dates: ['2001-07-22', '2001-07-23', '2002-08-11'],
        seasonality: { June: 0.11, July: 0.56, August: 0.33, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '20', name: 'Tamilnadu & Pondichery', bust_rate: 0.44, bust_count: 4, mean_signed_error: -8.2 },
          { region_id: '12', name: 'S. I. Karnataka', bust_rate: 0.33, bust_count: 3, mean_signed_error: -6.5 },
        ],
        vulnerable_leads: [3, 4, 5],
        signed_bias: 'dry',
        mean_error_mm: -3.8,
        composite_summary: {
          z500: { min: -1.5, max: 5.6, mean: 1.1, std: 1.2 },
          u850: { min: -4.2, max: -0.4, mean: -2.1, std: 0.8 },
          v850: { min: -1.6, max: 1.4, mean: -0.2, std: 0.5 },
          mslp: { min: -0.2, max: 2.8, mean: 1.1, std: 0.5 },
          tcwv: { min: -9.8, max: -2.1, mean: -5.4, std: 1.1 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 7,
        name: 'Brahmaputra Valley Orographic Funneling',
        mechanism_summary:
          'Deep tropospheric southerly moisture flux funneling into the Meghalaya plateau and southern Assam, driving severe orographic precipitation bursts misforecast in coarse global models.',
        n_members: 11,
        member_dates: ['2000-06-22', '2001-07-18', '2002-06-28'],
        seasonality: { June: 0.55, July: 0.36, August: 0.09, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '5', name: 'Assam & Meghalaya', bust_rate: 0.55, bust_count: 6, mean_signed_error: 24.1 },
          { region_id: '4', name: 'Arunachal Pradesh', bust_rate: 0.45, bust_count: 5, mean_signed_error: 17.8 },
        ],
        vulnerable_leads: [2, 3, 4],
        signed_bias: 'wet',
        mean_error_mm: 5.9,
        composite_summary: {
          z500: { min: -4.2, max: 6.8, mean: 0.4, std: 1.8 },
          u850: { min: -1.2, max: 5.4, mean: 1.8, std: 1.2 },
          v850: { min: 0.8, max: 6.2, mean: 3.1, std: 1.3 },
          mslp: { min: -3.4, max: 0.8, mean: -1.2, std: 0.6 },
          tcwv: { min: 5.1, max: 19.2, mean: 11.8, std: 2.1 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
      {
        cluster_id: 8,
        name: 'Land Depression Retrograde Track',
        mechanism_summary:
          'Slow-moving land depression over East Central India undergoing retrograde westward propagation, producing anomalous multi-day heavy rainfall corridors across Vidarbha and West MP.',
        n_members: 7,
        member_dates: ['2001-08-14', '2001-08-15', '2002-08-20'],
        seasonality: { June: 0.0, July: 0.286, August: 0.714, September: 0.0 },
        threatened_subdivisions: [
          { region_id: '35', name: 'Vidarbha', bust_rate: 0.57, bust_count: 4, mean_signed_error: 19.3 },
          { region_id: '14', name: 'West Madhya Pradesh', bust_rate: 0.43, bust_count: 3, mean_signed_error: 15.2 },
        ],
        vulnerable_leads: [4, 5, 6, 7],
        signed_bias: 'wet',
        mean_error_mm: 4.4,
        composite_summary: {
          z500: { min: -7.5, max: 2.1, mean: -2.4, std: 1.9 },
          u850: { min: -2.8, max: 3.9, mean: 0.8, std: 1.1 },
          v850: { min: -1.9, max: 3.4, mean: 0.6, std: 0.9 },
          mslp: { min: -5.6, max: -0.4, mean: -2.8, std: 1.0 },
          tcwv: { min: 3.4, max: 15.6, mean: 9.2, std: 1.6 },
        },
        validation_status: 'unvalidated_exploratory',
        is_validated: false,
        validation: [],
      },
    ],
  };
}

