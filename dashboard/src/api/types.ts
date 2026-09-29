/**
 * TypeScript API Types for Project Pratyaya
 * Strictly mirrors Pydantic V2 Schemas in src/pratyaya/api/schemas.py
 */

export type ConfidenceBand = 'HIGH' | 'MODERATE' | 'LOW' | 'CRITICAL';

export type BustDirection = 'wet' | 'dry' | 'none';

export const BAND_THRESHOLDS = {
  HIGH_MAX: 0.10,      // p_bust < 0.10 -> HIGH
  MODERATE_MAX: 0.25,  // 0.10 <= p_bust < 0.25 -> MODERATE
  LOW_MAX: 0.50,       // 0.25 <= p_bust < 0.50 -> LOW
} as const;

export const BAND_OBSERVED_BUST_RATES: Record<ConfidenceBand, number> = {
  HIGH: 0.015,
  MODERATE: 0.082,
  LOW: 0.285,
  CRITICAL: 0.584,
};

export const CONFIDENCE_COLORS: Record<ConfidenceBand, string> = {
  HIGH: '#10B981',      // High confidence (low bust probability)
  MODERATE: '#F59E0B',  // Elevated bust probability
  LOW: '#EF4444',       // Low confidence (high bust probability)
  CRITICAL: '#7F1D1D',  // Critical bust risk
};

export interface AnalogueItem {
  date: string;
  weight: number;
  outcome: string;
  observed_value?: number;
  forecast_value?: number;
  distance?: number;
  bust_direction?: BustDirection;
  is_bust?: boolean;
  closeness_rank?: number;
  mechanism?: string;
}

export interface ForecastCard {
  effective_lead_time: number;
  p_bust: number;
  p_bust_p90: number;
  confidence_band: ConfidenceBand;
  top_analogues: AnalogueItem[];
  risk_factors: string[];
  bust_direction: BustDirection;
  unprecedented_flag: boolean;
  region_id?: string;
  region_name?: string;
  lead_day?: number;
  init_date?: string;
  headline?: string;
  observed_bust_rate_in_band?: number;
  model_version: string;
  data_as_of?: string;
  is_live: boolean;
}

export interface SubdivisionForecast {
  region_id: string;
  name: string;
  effective_lead_time: number;
  p_bust: number;
  confidence_band: ConfidenceBand;
  p_bust_p90?: number;
  bust_direction?: BustDirection;
  unprecedented_flag: boolean;
}

export interface ConfidenceMap {
  lead_day: number;
  date: string;
  regions: SubdivisionForecast[];
  variable: string;
  model_version: string;
  data_as_of?: string;
  is_live: boolean;
}

export interface QueueItem {
  rank: number;
  region_id: string;
  region_name: string;
  lead_day: number;
  p_bust: number;
  decision_weight: number;
  priority_score: number;
  effective_lead_time: number;
  confidence_band: ConfidenceBand;
  bust_direction: BustDirection;
  top_risk_factor?: string;
}

export interface ErrorProneAreas {
  date: string;
  lead_day?: number;
  total_items: number;
  queue: QueueItem[];
  model_version: string;
  data_as_of?: string;
  is_live: boolean;
}

export interface BenchmarkMetrics {
  pr_auc: number;
  auroc: number;
  brier_score: number;
  ece: number;
  description: string;
}

export interface BenchmarkEntry {
  id: string;
  name: string;
  type: string;
  pr_auc: number;
  pr_auc_ci_lower: number;
  pr_auc_ci_upper: number;
  auroc: number;
  brier: number;
  reliability: number;
  bss_vs_b1: number;
  role: string;
  badge: string;
  badge_class: string;
  pr_auc_by_lead?: number[];
}

export interface ModelStatus {
  model_version: string;
  run_id: string;
  heldout_metrics: Record<string, BenchmarkMetrics | any>;
  calibration_date: string;
  status: string;
  training_window: string;
  calibration_window: string;
  heldout_window: string;
  baselines_available: string[];
  is_live: boolean;
  data_as_of?: string;
  benchmark_ladder?: BenchmarkEntry[];
  n_independent_events?: number;
  ood_refusal_rate?: number;
}

export interface SubdivisionReplayPrediction {
  region_id: string;
  name: string;
  p_bust: number;
  confidence_band: ConfidenceBand;
  nwp_rainfall: number;
  obs_rainfall?: number;
  effective_lead_time_days: number;
  error_direction: BustDirection;
  unprecedented_flag: boolean;
  p_bust_p90?: number;
}

export interface ReplayFrame {
  cycle_date: string;
  lead_day: number;
  valid_date: string;
  subdivisions: SubdivisionReplayPrediction[];
  caption_strip: string;
  triage_queue: QueueItem[];
  confidence_map: ConfidenceMap;
  top_risk_region_id: string;
  top_risk_region_name: string;
  max_p_bust: number;
  mean_effective_lead_time: number;
}

export interface ReplayScenario {
  scenario_id: string;
  title: string;
  description: string;
  cycle_date: string;
  lead_days: number[];
  frames: ReplayFrame[];
  model_version: string;
  is_live: boolean;
}

export interface ScenarioSummary {
  id: string;
  title: string;
  description: string;
  cycle_date: string;
  date_range: string;
  leads: string;
}

export interface EnrichmentStats {
  catalogue_name: string;
  observed_count: number;
  expected_count: number;
  enrichment_ratio: number;
  ci_95: [number, number];
  chi2_stat: number;
  p_value: number;
  is_significant: boolean;
}

export interface ThreatenedSubdivision {
  region_id: string;
  name: string;
  bust_rate: number;
  bust_count: number;
  mean_signed_error: number;
}

export interface RegimeClusterSummary {
  cluster_id: number;
  name: string;
  mechanism_summary: string;
  n_members: number;
  member_dates: string[];
  seasonality: Record<string, number>;
  threatened_subdivisions: ThreatenedSubdivision[];
  vulnerable_leads: number[];
  signed_bias: 'wet' | 'dry' | string;
  mean_error_mm: number;
  composite_summary: Record<string, { min: number; max: number; mean: number; std?: number }>;
  validation_status: 'validated' | 'unvalidated_exploratory' | string;
  is_validated: boolean;
  validation: EnrichmentStats[];
}

export interface RegimesResponse {
  optimal_k: number;
  silhouette_scores: Record<string, number>;
  clusters: RegimeClusterSummary[];
  n_high_burden_dates: number;
  high_burden_threshold: number;
  total_candidate_dates: number;
  model_version: string;
  data_as_of?: string;
  is_live: boolean;
}

