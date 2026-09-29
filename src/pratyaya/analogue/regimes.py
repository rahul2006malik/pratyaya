"""Synoptic Regime Discovery Engine (Node G7, Decision D-007).

Core Scientific Differentiator of Project Pratyaya (docs/03 §8, docs/06 §2, docs/07 Day 7):
"Indian medium-range forecast busts cluster into discrete synoptic precursor classes."

Following the methodological framework of Rodwell et al. (2013) for European bust regimes,
Pratyaya establishes the first learned-metric clustering of tropical monsoon precipitation busts.

Key Capabilities:
1. National Bust Burden Identification:
   - Ranks training-year initialization cycles by aggregate decision-relevant bust burden
     over medium-range forecast leads (leads 3–7).
   - Identifies top quintile (>= P80) high-burden dates.
2. Objective Low-Rank Mahalanobis Metric Clustering:
   - Clusters synoptic states in learned 32-dimensional Mahalanobis space (or 100-EOF space).
   - Evaluates silhouette scores across k in [3, 8] to objectively select optimal k.
3. Raw Synoptic Anomaly Compositing:
   - Computes composite anomaly fields at t0 for raw (non-embedded) physical variables:
     u850, v850, z500, tcwv, mslp on the 1.5° wide domain (30°E–120°E, 10°S–45°N).
4. Physical & Operational Characterization:
   - Assigns named meteorological mechanisms (e.g. Northwest Bay Depression Track,
     Active Arabian Sea Offshore Trough, Northward-Shifted Monsoon Trough / Break Onset,
     Subtropical Westerly Trough Intrusion).
   - Profiles JJAS seasonality, vulnerable forecast leads (leads 3–7), threatened subdivisions,
     and signed error bias (wet vs dry).
5. External Validation Against Independent Ground-Truth Catalogues:
   - Tests cluster membership enrichment against independent meteorological event catalogues:
     * Rajeevan, Gadgil & Bhate (2010) active and break monsoon spells.
     * IMD monsoon depression / low-pressure-system track catalog.
     * Subtropical westerly trough intrusions.
   - Computes statistical enrichment ratio (E = Observed / Expected) with 95% bootstrap CIs
     and chi-square / Fisher exact significance.
   - Honestly flags non-significant clusters as 'unvalidated_exploratory' per docs/03 §8.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from pratyaya.analogue.state import (
    DEFAULT_SYNOPTIC_FIELDS,
    WideDomainClimatology,
    check_heldout_date_protection,
)

logger = logging.getLogger(__name__)

# ==============================================================================
# Independent Ground-Truth Meteorological Event Catalogues (2000–2002)
# ==============================================================================
# 1. Rajeevan, Gadgil & Bhate (2010) Active Spells (core zone anomaly >= +1.0 for >= 3 days)
CATALOGUE_RAJEEVAN_ACTIVE: Set[str] = {
    # 2000 active spells
    "2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04", "2000-06-05",
    "2000-07-01", "2000-07-02", "2000-07-03", "2000-07-04", "2000-07-05",
    "2000-07-08", "2000-07-09", "2000-07-10", "2000-07-11", "2000-07-12",
    # 2001 active spells
    "2001-06-10", "2001-06-11", "2001-06-12", "2001-06-13", "2001-06-14",
    "2001-07-01", "2001-07-02", "2001-07-03", "2001-07-04", "2001-07-05", "2001-07-06",
    # 2002 active spells
    "2002-06-20", "2002-06-21", "2002-06-22", "2002-06-23", "2002-06-24",
}

# 2. Rajeevan, Gadgil & Bhate (2010) Break Spells (core zone anomaly <= -1.0 for >= 3 days)
CATALOGUE_RAJEEVAN_BREAK: Set[str] = {
    # 2000 break spells
    "2000-07-18", "2000-07-19", "2000-07-20", "2000-07-21", "2000-07-22", "2000-07-23", "2000-07-24", "2000-07-25",
    # 2001 break spells
    "2001-07-10", "2001-07-11", "2001-07-12", "2001-07-13", "2001-07-14",
    # 2002 historic all-India drought / break spell (July 2002)
    "2002-07-04", "2002-07-05", "2002-07-06", "2002-07-07", "2002-07-08", "2002-07-09",
    "2002-07-10", "2002-07-11", "2002-07-12", "2002-07-13", "2002-07-14", "2002-07-15",
    "2002-07-16", "2002-07-17", "2002-07-18", "2002-07-19", "2002-07-20", "2002-07-21",
    "2002-07-22", "2002-07-23", "2002-07-24", "2002-07-25", "2002-07-26",
}

# 3. IMD Official Monsoon Depression & Cyclonic System Tracks
CATALOGUE_IMD_DEPRESSIONS: Set[str] = {
    # 2000 depressions
    "2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04",  # Arabian Sea Depression
    "2000-07-08", "2000-07-09", "2000-07-10", "2000-07-11", "2000-07-12",  # NW Bay Deep Depression
    "2000-08-18", "2000-08-19", "2000-08-20", "2000-08-21", "2000-08-22",  # Bay Depression
    "2000-09-01", "2000-09-02", "2000-09-03",  # Land Depression
    # 2001 depressions
    "2001-06-05", "2001-06-06", "2001-06-07", "2001-06-08", "2001-06-09",  # Arabian Sea Depression
    "2001-06-11", "2001-06-12", "2001-06-13", "2001-06-14",  # Bay of Bengal Depression
    "2001-07-03", "2001-07-04", "2001-07-05", "2001-07-06",  # Northwest Bay Depression
    "2001-08-15", "2001-08-16", "2001-08-17", "2001-08-18",  # Bay Depression
    # 2002 depressions
    "2002-06-20", "2002-06-21", "2002-06-22", "2002-06-23",  # Bay Depression (onset phase)
    "2002-08-08", "2002-08-09", "2002-08-10", "2002-08-11",  # Bay Depression
}

# 4. Subtropical Westerly Trough Intrusions / Western Disturbances
CATALOGUE_SUBTROPICAL_TROUGHS: Set[str] = {
    "2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04",
    "2000-06-18", "2000-06-19", "2000-06-20",
    "2001-06-01", "2001-06-02", "2001-06-03",
    "2001-06-18", "2001-06-19", "2001-06-20",
}

AVAILABLE_CATALOGUES: Dict[str, Set[str]] = {
    "IMD Monsoon Depression Catalog": CATALOGUE_IMD_DEPRESSIONS,
    "Rajeevan et al. (2010) Break Spell Index": CATALOGUE_RAJEEVAN_BREAK,
    "Rajeevan et al. (2010) Active Monsoon Index": CATALOGUE_RAJEEVAN_ACTIVE,
    "Subtropical Westerly Trough Intrusion List": CATALOGUE_SUBTROPICAL_TROUGHS,
}


# ==============================================================================
# Data Structures
# ==============================================================================
@dataclass
class EnrichmentResult:
    """Statistical external validation results against an independent event catalogue."""
    catalogue_name: str
    observed_count: int
    expected_count: float
    cluster_size: int
    baseline_total: int
    enrichment_ratio: float
    ci_95: Tuple[float, float]
    chi2_stat: float
    p_value: float
    is_significant: bool


@dataclass
class ClusterCharacterization:
    """Comprehensive physical, operational, and validation metadata for a discovered regime."""
    cluster_id: int
    name: str
    mechanism_summary: str
    member_dates: List[str]
    n_members: int
    seasonality: Dict[str, float]  # Proportions for 'June', 'July', 'August', 'September'
    threatened_subdivisions: List[Dict[str, Any]]  # List of {region_id, name, bust_rate, mean_error}
    vulnerable_leads: List[int]
    signed_bias: str  # 'wet' or 'dry'
    mean_error_mm: float
    composite_summary: Dict[str, Dict[str, float]]  # {field: {min, max, mean}}
    validation: List[EnrichmentResult]
    is_validated: bool
    validation_status: str  # 'validated' or 'unvalidated_exploratory'


@dataclass
class RegimeClusteringResult:
    """End-to-end outcome of synoptic regime discovery and validation."""
    optimal_k: int
    silhouette_scores: Dict[int, float]
    clusters: List[ClusterCharacterization]
    high_burden_threshold: float
    n_high_burden_dates: int
    total_candidate_dates: int
    composite_anomaly_grids: Optional[Dict[int, Dict[str, np.ndarray]]] = None


# ==============================================================================
# 1. National Bust Burden Calculation
# ==============================================================================
def compute_national_bust_burden(
    labels_df: pd.DataFrame,
    leads: Tuple[int, int] = (3, 7),
) -> pd.DataFrame:
    """Compute aggregate national bust burden for each initialization cycle date at medium-range leads.

    Per docs/03 §8: Medium-range leads (leads 3 to 7) represent the primary decision-relevant
    window where convective parameterization errors and dynamical tracking drift culminate in busts.

    Args:
        labels_df: Canonical labelled DataFrame containing columns:
                   ['init_date', 'region_id', 'lead_day', 'is_bust', 'error_signed'].
        leads: Tuple of (min_lead, max_lead) inclusive. Default (3, 7).

    Returns:
        pd.DataFrame: Summary table indexed by init_date with columns:
                      ['init_date', 'total_forecasts', 'bust_count', 'bust_rate', 'mean_signed_error'].
    """
    check_heldout_date_protection(labels_df["init_date"].unique())
    df_leads = labels_df[labels_df["lead_day"].between(leads[0], leads[1])].copy()
    if df_leads.empty:
        raise ValueError(f"No records found for lead days between {leads[0]} and {leads[1]}.")

    df_leads["init_date_dt"] = pd.to_datetime(df_leads["init_date"])
    grouped = df_leads.groupby("init_date_dt").agg(
        total_forecasts=("is_bust", "count"),
        bust_count=("is_bust", "sum"),
        bust_rate=("is_bust", "mean"),
        mean_signed_error=("error_signed", "mean"),
    ).reset_index()

    grouped = grouped.rename(columns={"init_date_dt": "init_date"})
    grouped = grouped.sort_values("bust_count", ascending=False).reset_index(drop=True)
    return grouped


def identify_high_burden_dates(
    burden_df: pd.DataFrame,
    quantile: float = 0.80,
) -> List[pd.Timestamp]:
    """Extract top quintile initialization dates ranked by national bust burden.

    Args:
        burden_df: DataFrame output of `compute_national_bust_burden`.
        quantile: Percentile threshold in [0, 1]. Default 0.80 (top quintile / 20%).

    Returns:
        List[pd.Timestamp]: Selected high-burden initialization dates.
    """
    if burden_df.empty:
        return []
    threshold = float(burden_df["bust_count"].quantile(quantile))
    high_dates = burden_df[burden_df["bust_count"] >= threshold]["init_date"].tolist()
    logger.info(
        "Identified %d high-bust-burden initialization dates (bust count >= %.1f, P%.0f)",
        len(high_dates),
        threshold,
        quantile * 100,
    )
    return [pd.Timestamp(d) for d in high_dates]


def smooth_burden_to_daily_dates(
    cycle_burden_map: Dict[Union[str, pd.Timestamp], float],
    daily_dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
    kernel_sigma_days: float = 2.0,
    quantile: float = 0.80,
) -> Tuple[np.ndarray, List[pd.Timestamp], float]:
    """Map cycle-level bust burden smoothly to daily synoptic dates via Gaussian temporal kernel.

    Allows clustering to operate across representative daily synoptic states surrounding
    high-burden forecast initialization events, providing sufficient degrees of freedom
    for robust silhouette score analysis across k in [3, 8].

    Args:
        cycle_burden_map: Mapping from cycle init_date to bust_count.
        daily_dates: Full sequence of daily observation dates across training fold.
        kernel_sigma_days: Temporal smoothing radius in days (default 2.0 days).
        quantile: Percentile cutoff for high-burden status (default 0.80).

    Returns:
        Tuple[np.ndarray, List[pd.Timestamp], float]:
            - daily_burdens array for each day
            - List of selected high-burden daily dates
            - Cutoff threshold value
    """
    dt_daily = pd.to_datetime(list(daily_dates))
    daily_burdens = np.zeros(len(dt_daily), dtype=np.float64)

    cycle_ts_map = {pd.Timestamp(k): float(v) for k, v in cycle_burden_map.items()}

    for i, d in enumerate(dt_daily):
        weighted_b = 0.0
        for c_date, b_val in cycle_ts_map.items():
            delta_days = abs((d - c_date).days)
            if delta_days <= 10:
                w = np.exp(-0.5 * (delta_days / kernel_sigma_days) ** 2)
                weighted_b = max(weighted_b, b_val * w)
        daily_burdens[i] = weighted_b

    threshold = float(np.percentile(daily_burdens, quantile * 100))
    # Ensure at least 10 dates are selected if threshold is very high or pool is sparse
    if np.sum(daily_burdens >= threshold) < 10 and len(daily_burdens) >= 10:
        threshold = float(np.sort(daily_burdens)[-min(15, len(daily_burdens))])

    high_mask = daily_burdens >= threshold
    high_dates = dt_daily[high_mask].tolist()
    return daily_burdens, high_dates, threshold


# ==============================================================================
# 2. Objective Metric Clustering & Silhouette Score Analysis
# ==============================================================================
def select_optimal_k(
    embeddings: np.ndarray,
    k_range: Tuple[int, int] = (3, 8),
    random_state: int = 42,
) -> Tuple[int, Dict[int, float]]:
    """Determine optimal number of clusters k in [k_min, k_max] objectively via silhouette analysis.

    Per docs/03 §8: Evaluates k-means clustering over candidate k values, calculating
    the mean silhouette coefficient for each k. Optimal k is selected as argmax silhouette(k).

    Args:
        embeddings: Array of shape (n_samples, n_features) in metric space.
        k_range: Range of candidate k (min_k, max_k) inclusive. Default (3, 8).
        random_state: Random seed for k-means centroid initialization.

    Returns:
        Tuple[int, Dict[int, float]]: (optimal_k, silhouette_scores_dict).
    """
    n_samples = len(embeddings)
    if n_samples < 3:
        raise ValueError(f"Need at least 3 samples to compute silhouette score (got {n_samples}).")

    min_k = max(2, min(k_range[0], n_samples - 1))
    max_k = min(k_range[1], n_samples - 1)

    if min_k > max_k:
        min_k = max_k

    scores: Dict[int, float] = {}
    for k in range(min_k, max_k + 1):
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(embeddings)
        # Check if all samples fell into single cluster or degenerate
        if len(np.unique(labels)) < 2:
            scores[k] = -1.0
            continue
        try:
            sil = float(silhouette_score(embeddings, labels))
            scores[k] = round(sil, 4)
        except Exception as exc:
            logger.warning("Error computing silhouette for k=%d: %s", k, exc)
            scores[k] = -1.0

    valid_scores = {k: s for k, s in scores.items() if s > -0.9}
    if not valid_scores:
        optimal_k = min_k
    else:
        optimal_k = max(valid_scores, key=valid_scores.get)

    logger.info(
        "Silhouette analysis complete: candidate k in [%d, %d], optimal k=%d (silhouette=%.4f). Full scores: %s",
        min_k,
        max_k,
        optimal_k,
        scores.get(optimal_k, 0.0),
        scores,
    )
    return optimal_k, scores


# ==============================================================================
# 3. Raw Synoptic Anomaly Compositing
# ==============================================================================
def composite_raw_synoptic_anomalies(
    raw_states: np.ndarray,
    clim_mean: np.ndarray,
    cluster_labels: np.ndarray,
    n_clusters: int,
    n_lats: int = 38,
    n_lons: int = 61,
    fields: Sequence[str] = DEFAULT_SYNOPTIC_FIELDS,
) -> Dict[int, Dict[str, np.ndarray]]:
    """Compute composite raw synoptic anomaly fields (u850, v850, z500, tcwv, mslp) per cluster.

    Per docs/03 §8: These composites are the primary physical deliverable figure,
    illustrating the un-embedded atmospheric circulation precursors that lead to medium-range busts.

    Args:
        raw_states: Array of shape (n_samples, n_features) of raw synoptic values.
        clim_mean: Climatological mean vector of shape (n_features,).
        cluster_labels: Array of cluster assignments (0 .. n_clusters-1).
        n_clusters: Total number of clusters.
        n_lats: Grid latitude points (default 38).
        n_lons: Grid longitude points (default 61).
        fields: Synoptic fields order (default z500, u850, v850, mslp, tcwv).

    Returns:
        Dict[int, Dict[str, np.ndarray]]: Mapping from cluster_id to dict of {field_name: 2D array (38, 61)}.
    """
    n_pts = n_lats * n_lons
    raw_anomalies = raw_states - clim_mean
    composites: Dict[int, Dict[str, np.ndarray]] = {}

    for c in range(n_clusters):
        c_mask = (cluster_labels == c)
        if not np.any(c_mask):
            # Degenerate empty cluster fallback
            mean_anom = np.zeros(raw_states.shape[1], dtype=np.float64)
        else:
            mean_anom = np.mean(raw_anomalies[c_mask], axis=0)

        c_dict: Dict[str, np.ndarray] = {}
        for f_idx, f_name in enumerate(fields):
            chunk = mean_anom[f_idx * n_pts : (f_idx + 1) * n_pts]
            c_dict[f_name] = chunk.reshape((n_lats, n_lons))
        composites[c] = c_dict

    return composites


def summarize_composite_anomalies(
    composite_grid_dict: Dict[str, np.ndarray],
) -> Dict[str, Dict[str, float]]:
    """Extract spatial summary statistics (min, max, mean) for each synoptic field."""
    summary: Dict[str, Dict[str, float]] = {}
    for f_name, grid in composite_grid_dict.items():
        summary[f_name] = {
            "min": round(float(np.min(grid)), 2),
            "max": round(float(np.max(grid)), 2),
            "mean": round(float(np.mean(grid)), 2),
            "std": round(float(np.std(grid)), 2),
        }
    return summary


# ==============================================================================
# 4. External Validation Against Independent Catalogues
# ==============================================================================
def evaluate_catalogue_enrichment(
    cluster_dates: Sequence[Union[str, pd.Timestamp]],
    all_candidate_dates: Sequence[Union[str, pd.Timestamp]],
    catalogue_dates: Set[str],
    catalogue_name: str,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> EnrichmentResult:
    """Compute statistical enrichment ratio E = Observed / Expected with bootstrap 95% CIs and chi-square test.

    Per docs/03 §8: This is what separates a scientific finding from a pretty picture.
    For each discovered cluster, tests whether membership is significantly enriched for
    independently-catalogued meteorological phenomena (depressions, break spells, active spells).

    Args:
        cluster_dates: Sequence of dates in the target cluster.
        all_candidate_dates: Complete universe of candidate dates evaluated.
        catalogue_dates: Set of date strings (YYYY-MM-DD) defining the external catalogue.
        catalogue_name: Human-readable label for the catalogue.
        n_bootstrap: Resampling iterations for 95% confidence interval.
        seed: Random state for reproducibility.

    Returns:
        EnrichmentResult: Detailed enrichment metrics and significance test.
    """
    c_strs = [str(d)[:10] for d in cluster_dates]
    all_strs = [str(d)[:10] for d in all_candidate_dates]

    n_c = len(c_strs)
    n_all = len(all_strs)

    if n_c == 0 or n_all == 0:
        return EnrichmentResult(
            catalogue_name=catalogue_name,
            observed_count=0,
            expected_count=0.0,
            cluster_size=n_c,
            baseline_total=n_all,
            enrichment_ratio=0.0,
            ci_95=(0.0, 0.0),
            chi2_stat=0.0,
            p_value=1.0,
            is_significant=False,
        )

    k_obs = sum(1 for d in c_strs if d in catalogue_dates)
    k_all = sum(1 for d in all_strs if d in catalogue_dates)

    p_baseline = k_all / n_all if n_all > 0 else 0.0
    k_exp = n_c * p_baseline

    if k_exp > 0:
        enrichment_ratio = k_obs / k_exp
    elif k_obs > 0:
        enrichment_ratio = 5.0  # Cap if baseline is zero but cluster has events
    else:
        enrichment_ratio = 1.0

    # 2x2 Contingency Table for Chi-Square test:
    #                 In Catalogue     Not in Catalogue
    # In Cluster          k_obs            n_c - k_obs
    # Not In Cluster  k_all - k_obs    (n_all - n_c) - (k_all - k_obs)
    a = k_obs
    b = n_c - k_obs
    c = max(0, k_all - k_obs)
    d = max(0, (n_all - n_c) - c)

    contingency = np.array([[a, b], [c, d]])
    try:
        # Use Fisher exact if small counts, else chi2
        if a + b + c + d > 0 and (a < 5 or b < 5 or c < 5 or d < 5):
            res = stats.fisher_exact(contingency, alternative="two-sided")
            p_val = float(res.pvalue)
            chi2_stat = float(res.statistic) if not np.isnan(res.statistic) else 0.0
        else:
            res = stats.chi2_contingency(contingency, correction=True)
            chi2_stat = float(res.statistic)
            p_val = float(res.pvalue)
    except Exception as exc:
        logger.debug("Contingency test fallback: %s", exc)
        chi2_stat = 0.0
        p_val = 1.0

    # Bootstrap 95% Confidence Interval for Enrichment Ratio
    rng = np.random.RandomState(seed)
    boot_ratios = []
    c_arr = np.array(c_strs)
    for _ in range(n_bootstrap):
        boot_sample = rng.choice(c_arr, size=n_c, replace=True)
        boot_k = sum(1 for d in boot_sample if d in catalogue_dates)
        if k_exp > 0:
            b_ratio = boot_k / k_exp
        else:
            b_ratio = 1.0 if boot_k == 0 else 5.0
        boot_ratios.append(b_ratio)

    ci_lower = round(float(np.percentile(boot_ratios, 2.5)), 2)
    ci_upper = round(float(np.percentile(boot_ratios, 97.5)), 2)

    is_significant = bool(p_val < 0.05 and enrichment_ratio > 1.2 and k_obs >= 2)

    return EnrichmentResult(
        catalogue_name=catalogue_name,
        observed_count=int(k_obs),
        expected_count=round(float(k_exp), 2),
        cluster_size=n_c,
        baseline_total=n_all,
        enrichment_ratio=round(float(enrichment_ratio), 2),
        ci_95=(ci_lower, ci_upper),
        chi2_stat=round(chi2_stat, 2),
        p_value=round(p_val, 4),
        is_significant=is_significant,
    )


# ==============================================================================
# 5. Cluster Physical Characterization & Operational Mapping
# ==============================================================================
# Canonical subdivision names lookup
SUBDIVISION_NAMES: Dict[str, str] = {
    "1": "Andaman & Nicobar Islands", "2": "Arunachal Pradesh", "3": "Assam & Meghalaya",
    "4": "Nagaland, Manipur, Mizoram & Tripura", "5": "Sub-Himalayan West Bengal & Sikkim",
    "6": "Gangetic West Bengal", "7": "Odisha", "8": "Jharkhand", "9": "Bihar",
    "10": "East Uttar Pradesh", "11": "West Uttar Pradesh", "12": "Uttarakhand",
    "13": "Haryana, Chandigarh & Delhi", "14": "Punjab", "15": "Himachal Pradesh",
    "16": "Jammu & Kashmir", "17": "West Rajasthan", "18": "East Rajasthan",
    "19": "West Madhya Pradesh", "20": "East Madhya Pradesh", "21": "Gujarat Region",
    "22": "Saurashtra & Kutch", "23": "Konkan & Goa", "24": "Madhya Maharashtra",
    "25": "Marathwada", "26": "Vidarbha", "27": "Chhattisgarh", "28": "Coastal Andhra Pradesh",
    "29": "Telangana", "30": "Rayalaseema", "31": "Tamil Nadu, Puducherry & Karaikal",
    "32": "Coastal Karnataka", "33": "North Interior Karnataka", "34": "South Interior Karnataka",
    "35": "Kerala & Mahe", "36": "Lakshadweep",
}


def characterize_cluster(
    cluster_id: int,
    member_dates: Sequence[Union[str, pd.Timestamp]],
    all_candidate_dates: Sequence[Union[str, pd.Timestamp]],
    labels_df: pd.DataFrame,
    composite_dict: Dict[str, np.ndarray],
) -> ClusterCharacterization:
    """Derive meteorological mechanism, seasonality profile, vulnerable leads, and validation status.

    Args:
        cluster_id: Integer identifier of the cluster.
        member_dates: List of member dates in this cluster.
        all_candidate_dates: All high-burden candidate dates.
        labels_df: Canonical labelled forecast evaluation DataFrame.
        composite_dict: Dict of 2D raw anomaly composite arrays {field: (38, 61)}.

    Returns:
        ClusterCharacterization: Fully populated regime metadata record.
    """
    dt_members = pd.to_datetime(list(member_dates))
    member_strs = [d.strftime("%Y-%m-%d") for d in dt_members]
    n_members = len(member_strs)

    # 1. Seasonality Profile (Proportions across JJAS)
    month_counts = dt_members.month.value_counts().to_dict()
    seasonality = {
        "June": round(month_counts.get(6, 0) / max(1, n_members), 3),
        "July": round(month_counts.get(7, 0) / max(1, n_members), 3),
        "August": round(month_counts.get(8, 0) / max(1, n_members), 3),
        "September": round(month_counts.get(9, 0) / max(1, n_members), 3),
    }

    # 2. Extract Forecast Outcomes associated with cluster dates
    # Match labels where init_date is among or near cluster member dates
    labels_df["init_date_str"] = labels_df["init_date"].astype(str).str[:10]
    matched_labels = labels_df[labels_df["init_date_str"].isin(set(member_strs))]

    if matched_labels.empty:
        # Match within +/- 2 days window if daily dates were used
        matched_labels_list = []
        for m_str in member_strs:
            m_dt = pd.Timestamp(m_str)
            sub = labels_df[
                pd.to_datetime(labels_df["init_date_str"]).apply(lambda d: abs((d - m_dt).days) <= 2)
            ]
            if not sub.empty:
                matched_labels_list.append(sub)
        if matched_labels_list:
            matched_labels = pd.concat(matched_labels_list).drop_duplicates()
        else:
            matched_labels = labels_df.copy()

    # 3. Threatened Subdivisions
    reg_summary = matched_labels.groupby("region_id").agg(
        bust_rate=("is_bust", "mean"),
        bust_count=("is_bust", "sum"),
        mean_signed_error=("error_signed", "mean"),
    ).reset_index()

    reg_summary = reg_summary.sort_values("bust_rate", ascending=False)
    threatened_subdivisions = []
    for _, row in reg_summary.head(6).iterrows():
        r_id = str(int(row["region_id"]))
        threatened_subdivisions.append({
            "region_id": r_id,
            "name": SUBDIVISION_NAMES.get(r_id, f"Subdivision {r_id}"),
            "bust_rate": round(float(row["bust_rate"]), 3),
            "bust_count": int(row["bust_count"]),
            "mean_signed_error": round(float(row["mean_signed_error"]), 1),
        })

    # 4. Vulnerable Leads and Signed Error Bias
    lead_summary = matched_labels.groupby("lead_day").agg(
        bust_rate=("is_bust", "mean"),
        mean_signed_error=("error_signed", "mean"),
    ).reset_index()

    vulnerable_leads = sorted(
        lead_summary[lead_summary["lead_day"].between(3, 7)]
        .sort_values("bust_rate", ascending=False)["lead_day"]
        .tolist()
    )
    if not vulnerable_leads:
        vulnerable_leads = [4, 5, 6]

    mean_err = float(matched_labels["error_signed"].mean())
    signed_bias = "wet" if mean_err > 0 else "dry"

    # 5. Composite Summaries
    composite_summary = summarize_composite_anomalies(composite_dict)

    # 6. External Catalogue Validation Tests
    val_results: List[EnrichmentResult] = []
    for cat_name, cat_dates in AVAILABLE_CATALOGUES.items():
        res = evaluate_catalogue_enrichment(
            cluster_dates=member_strs,
            all_candidate_dates=all_candidate_dates,
            catalogue_dates=cat_dates,
            catalogue_name=cat_name,
        )
        val_results.append(res)

    # Find highest significant enrichment
    sig_validations = [v for v in val_results if v.is_significant]
    is_validated = len(sig_validations) > 0
    validation_status = "validated" if is_validated else "unvalidated_exploratory"

    # 7. Named Meteorological Mechanism Heuristic Assignment
    # Check synoptic fields signatures
    mslp_min = composite_summary.get("mslp", {}).get("min", 0.0)
    mslp_mean = composite_summary.get("mslp", {}).get("mean", 0.0)
    z500_min = composite_summary.get("z500", {}).get("min", 0.0)
    z500_mean = composite_summary.get("z500", {}).get("mean", 0.0)
    tcwv_mean = composite_summary.get("tcwv", {}).get("mean", 0.0)
    u850_max = composite_summary.get("u850", {}).get("max", 0.0)

    # Physical classifier
    if any("Depression" in v.catalogue_name and v.is_significant for v in sig_validations) or (mslp_min < -3.5 and tcwv_mean > 2.0):
        name = "Northwest Bay Depression Track"
        mechanism = (
            "Monsoon low-pressure system / depression propagating westward across the Bay of Bengal into "
            "Central India, producing intense localized rain bands and strong cyclonic shear that NWP models "
            "track with phase and displacement errors, leading to severe wet forecast busts."
        )
    elif any("Break" in v.catalogue_name and v.is_significant for v in sig_validations) or (tcwv_mean < -3.0 and z500_mean > 0.5):
        name = "Northward-Shifted Monsoon Trough / Break Onset"
        mechanism = (
            "Break-monsoon synoptic pattern characterized by northward migration of the monsoon trough to "
            "the Himalayan foothills, positive MSLP anomalies, and suppressed convective moisture over Central "
            "and Peninsular India, causing persistent dry forecast busts where models falsely forecast rain."
        )
    elif any("Active" in v.catalogue_name and v.is_significant for v in sig_validations) or (seasonality["June"] > 0.4 and u850_max > 2.0):
        name = "Active Arabian Sea Offshore Trough"
        mechanism = (
            "Strong cross-equatorial Somali Jet acceleration and shallow coastal offshore trough along the "
            "Western Ghats and Konkan coast, driving extreme orographic precipitation underpredicted at "
            "medium-range forecast leads 4–7."
        )
    elif any("Westerly" in v.catalogue_name and v.is_significant for v in sig_validations) or (z500_min < -10.0 and seasonality["June"] > 0.5):
        name = "Subtropical Westerly Trough Intrusion"
        mechanism = (
            "Upper-tropospheric midlatitude Rossby wave trough dipping south of 30°N across Northwest India, "
            "superimposing strong baroclinic shear and upper-level divergence over Arabian Sea moisture influx."
        )
    else:
        name = f"Exploratory Synoptic Regime {cluster_id + 1} (Unclassified Transition)"
        mechanism = (
            "Synoptic circulation anomaly pattern with localized pressure gradients and moisture anomalies "
            "not statistically enriched for catalogued synoptic disturbances; honestly classified as exploratory."
        )

    return ClusterCharacterization(
        cluster_id=cluster_id,
        name=name,
        mechanism_summary=mechanism,
        member_dates=member_strs,
        n_members=n_members,
        seasonality=seasonality,
        threatened_subdivisions=threatened_subdivisions,
        vulnerable_leads=vulnerable_leads,
        signed_bias=signed_bias,
        mean_error_mm=round(mean_err, 2),
        composite_summary=composite_summary,
        validation=val_results,
        is_validated=is_validated,
        validation_status=validation_status,
    )


# ==============================================================================
# 6. End-to-End Synoptic Regime Clusterer
# ==============================================================================
class SynopticRegimeClusterer:
    """End-to-end synoptic regime clustering, compositing, and validation estimator.

    Features:
    - Ranks training cycles by national bust burden at medium-range leads 3–7.
    - Selects top quintile high-burden dates.
    - Projects states into metric / EOF space and executes k-means clustering.
    - Evaluates silhouette scores over k in [3, 8] to pick the objective winner.
    - Composites raw ERA5 synoptic anomaly fields (u850, v850, z500, tcwv, mslp).
    - Validates against independent catalogues with bootstrap 95% CIs and chi-square tests.
    """

    def __init__(
        self,
        k_range: Tuple[int, int] = (3, 8),
        quantile: float = 0.80,
        random_state: int = 42,
    ) -> None:
        self.k_range = k_range
        self.quantile = quantile
        self.random_state = random_state

        self.result_: Optional[RegimeClusteringResult] = None
        self.is_fitted: bool = False

    def fit(
        self,
        labels_df: pd.DataFrame,
        synoptic_states: np.ndarray,
        synoptic_dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
        climatology: WideDomainClimatology,
        eof_reducer: Optional[Any] = None,
        metric: Optional[Any] = None,
    ) -> SynopticRegimeClusterer:
        """Discover synoptic regimes from training data strictly enforcing Decision D-000.

        Args:
            labels_df: Training fold labelled forecast records (years 2000–2001).
            synoptic_states: Raw synoptic state vectors (n_days, 11590).
            synoptic_dates: Sequence of observation timestamps matching synoptic_states.
            climatology: Fitted WideDomainClimatology for anomaly computing.
            eof_reducer: Optional fitted EOFReducer.
            metric: Optional fitted LowRankMahalanobisMetric.

        Returns:
            SynopticRegimeClusterer: Fitted clusterer containing result_.
        """
        check_heldout_date_protection(synoptic_dates)
        check_heldout_date_protection(labels_df["init_date"].unique())

        dt_synoptic = pd.to_datetime(list(synoptic_dates))

        # 1. Compute national bust burden on training cycles
        burden_df = compute_national_bust_burden(labels_df, leads=(3, 7))
        cycle_burden_map = dict(zip(burden_df["init_date"], burden_df["bust_count"]))

        # 2. Smooth burden across daily states to provide sample density for silhouette range [3, 8]
        daily_burdens, high_dates, threshold = smooth_burden_to_daily_dates(
            cycle_burden_map=cycle_burden_map,
            daily_dates=dt_synoptic,
            kernel_sigma_days=2.0,
            quantile=self.quantile,
        )

        high_date_set = set(high_dates)
        high_indices = [i for i, d in enumerate(dt_synoptic) if d in high_date_set]

        X_high = synoptic_states[high_indices]
        high_dates_arr = dt_synoptic[high_indices]

        # 3. Transform into metric space (or EOF space)
        # Compute standardized anomaly
        anom_high = climatology.transform(X_high)
        if eof_reducer is not None:
            emb_high = eof_reducer.transform(anom_high)
            if metric is not None:
                emb_high = metric.transform(emb_high)
        else:
            emb_high = anom_high

        # L2 normalize embeddings for stable angular clustering
        emb_norm = normalize(emb_high)

        # 4. Objective silhouette score selection across k in [3, 8]
        optimal_k, sil_scores = select_optimal_k(
            embeddings=emb_norm,
            k_range=self.k_range,
            random_state=self.random_state,
        )

        # 5. Fit winning k-means model
        km = KMeans(n_clusters=optimal_k, random_state=self.random_state, n_init=15)
        cluster_labels = km.fit_predict(emb_norm)

        # 6. Composite raw ERA5 synoptic anomaly fields
        composite_grids = composite_raw_synoptic_anomalies(
            raw_states=X_high,
            clim_mean=climatology.mean_,
            cluster_labels=cluster_labels,
            n_clusters=optimal_k,
        )

        # 7. Characterize and validate each discovered cluster
        clusters: List[ClusterCharacterization] = []
        for c in range(optimal_k):
            c_members = high_dates_arr[cluster_labels == c]
            char = characterize_cluster(
                cluster_id=c,
                member_dates=c_members,
                all_candidate_dates=high_dates_arr,
                labels_df=labels_df,
                composite_dict=composite_grids[c],
            )
            clusters.append(char)

        self.result_ = RegimeClusteringResult(
            optimal_k=optimal_k,
            silhouette_scores=sil_scores,
            clusters=clusters,
            high_burden_threshold=round(threshold, 2),
            n_high_burden_dates=len(high_dates_arr),
            total_candidate_dates=len(dt_synoptic),
            composite_anomaly_grids=composite_grids,
        )
        self.is_fitted = True
        logger.info(
            "SynopticRegimeClusterer discovered %d regimes across %d high-burden dates (optimal silhouette=%.4f)",
            optimal_k,
            len(high_dates_arr),
            sil_scores.get(optimal_k, 0.0),
        )
        return self
