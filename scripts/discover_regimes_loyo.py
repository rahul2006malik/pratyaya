"""Synoptic Regime Discovery Engine.

Pratyaya Atmospheric Research Framework.

Discovers synoptic precursor regimes across the dense 3-year empirical benchmark
(2000–2002 JJAS, 120 cycles, 43,200 canonical rows).

Strictly enforces:
1. Zero synthetic data.
2. Real cycle dates and real synoptic anomalies.
3. Mandatory >= 15-day event separation rule for independent meteorological events.
4. Pre-committed N >= 30 event threshold for statistical significance:
   Because a 3-year record contains at most 24 non-overlapping 15-day periods in total,
   any cluster has N_independent < 30 and MUST be classified as 'unvalidated_exploratory'.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

from pratyaya.analogue.eof import EOFReducer
from pratyaya.analogue.metric import LowRankMahalanobisMetric, build_outcome_vectors, sample_triplets
from pratyaya.analogue.regimes import (
    AVAILABLE_CATALOGUES,
    SUBDIVISION_NAMES,
    ClusterCharacterization,
    EnrichmentResult,
    RegimeClusteringResult,
    characterize_cluster,
    composite_raw_synoptic_anomalies,
    compute_national_bust_burden,
    evaluate_catalogue_enrichment,
    identify_high_burden_dates,
    select_optimal_k,
    summarize_composite_anomalies,
)
from pratyaya.analogue.state import (
    DEFAULT_SYNOPTIC_FIELDS,
    WideDomainClimatology,
    check_heldout_date_protection,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("discover_regimes_loyo")

DEFAULT_LABELS_DIR = Path("data/canonical/labels")
DEFAULT_SYNOPTIC_CACHE = Path("data/interim/synoptic_states_2000_2002.npz")
DEFAULT_OUTPUT_JSON = Path("models/regimes_loyo.json")


def count_independent_events(dates: Sequence[str], min_separation_days: int = 15) -> int:
    """Group date strings into independent events separated by >= min_separation_days."""
    if not dates:
        return 0
    dts = sorted(pd.to_datetime(list(dates)))
    events = 1
    last_dt = dts[0]
    for cur_dt in dts[1:]:
        if (cur_dt - last_dt).days >= min_separation_days:
            events += 1
            last_dt = cur_dt
    return events


def run_regime_discovery_loyo(
    labels_df: pd.DataFrame,
    synoptic_cache_path: Path = DEFAULT_SYNOPTIC_CACHE,
    output_path: Path = DEFAULT_OUTPUT_JSON,
    k_range: Tuple[int, int] = (3, 10),
    random_state: int = 42,
) -> RegimeClusteringResult:
    """Run objective regime clustering and D-009 validation on 2000–2002 dataset."""
    logger.info("Loading synoptic state cache from %s...", synoptic_cache_path)
    synoptic_cache = np.load(synoptic_cache_path, allow_pickle=True)
    daily_dates = pd.to_datetime(synoptic_cache["training_daily_dates"])
    daily_states = synoptic_cache["training_daily_states"]

    check_heldout_date_protection(daily_dates)
    check_heldout_date_protection(labels_df["init_date"].unique())

    # 1. Fit Climatology and EOF basis
    logger.info("Fitting WideDomainClimatology on %d daily states...", len(daily_states))
    climatology = WideDomainClimatology()
    climatology.fit(daily_states, daily_dates)
    daily_anom = climatology.transform(daily_states)

    logger.info("Fitting EOFReducer (100 components)...")
    eof = EOFReducer(n_components=100)
    eof.fit(daily_anom, daily_dates)
    Z_daily_eof = eof.transform(daily_anom)

    # 2. National Bust Burden Calculation across 120 cycles
    logger.info("Computing national bust burden on medium-range leads 3–7...")
    burden_df = compute_national_bust_burden(labels_df, leads=(3, 7))
    threshold = float(burden_df["bust_count"].quantile(0.80))

    # Direct, un-smoothed extraction of high-burden cycles (Decision D-009)
    high_dates = identify_high_burden_dates(burden_df, quantile=0.80)
    high_date_set = set(pd.to_datetime(high_dates))
    high_indices = [i for i, d in enumerate(daily_dates) if d in high_date_set]
    X_high = daily_states[high_indices]
    high_dates_arr = daily_dates[high_indices]

    logger.info("High burden threshold: %.2f | Selected %d high-burden cycles", threshold, len(high_dates_arr))

    # 3. Fit Low-Rank Mahalanobis Metric (Node G2) on cycle outcomes
    cycle_dates_list = sorted(pd.to_datetime(labels_df["init_date"].unique()))
    outcome_vecs, _ = build_outcome_vectors(labels_df, cycle_dates_list, n_leads=10, standardize=True)
    triplets = sample_triplets(
        outcome_vecs,
        cycle_dates_list,
        n_triplets=min(2500, len(cycle_dates_list) * 25),
        min_separation_days=15,
        seed=random_state,
    )
    date_to_idx = {pd.Timestamp(d).strftime("%Y-%m-%d"): i for i, d in enumerate(daily_dates)}
    Z_cycles_eof = np.array([Z_daily_eof[date_to_idx[d.strftime("%Y-%m-%d")]] for d in cycle_dates_list])

    metric = LowRankMahalanobisMetric(
        n_components=32,
        input_dim=100,
        margin=1.0,
        learning_rate=0.005,
        n_epochs=60,
        batch_size=64,
        l2_reg=1e-4,
        seed=random_state,
    )
    metric.fit(Z_cycles_eof, triplets)

    # Transform high-burden days into 32-dim Mahalanobis space
    emb_high_eof = Z_daily_eof[high_indices]
    emb_high_metric = metric.transform(emb_high_eof)
    emb_norm = normalize(emb_high_metric)

    # 4. Objective Silhouette Score Selection across k in [3, 8]
    optimal_k, sil_scores = select_optimal_k(
        embeddings=emb_norm,
        k_range=k_range,
        random_state=random_state,
    )
    logger.info("Objective silhouette analysis: optimal k = %d (silhouette = %.4f)", optimal_k, sil_scores[optimal_k])

    # 5. Fit K-Means
    km = KMeans(n_clusters=optimal_k, random_state=random_state, n_init=20)
    cluster_labels = km.fit_predict(emb_norm)

    # 6. Composite Raw Synoptic Anomaly Fields
    composite_grids = composite_raw_synoptic_anomalies(
        raw_states=X_high,
        clim_mean=climatology.mean_,
        cluster_labels=cluster_labels,
        n_clusters=optimal_k,
    )

    # 7. Characterize and Validate Clusters under Decision D-009
    clusters: List[ClusterCharacterization] = []
    for c in range(optimal_k):
        c_members = high_dates_arr[cluster_labels == c]
        member_strs = [d.strftime("%Y-%m-%d") for d in c_members]
        n_members_daily = len(member_strs)
        n_indep_events = count_independent_events(member_strs, min_separation_days=15)

        char = characterize_cluster(
            cluster_id=c,
            member_dates=c_members,
            all_candidate_dates=high_dates_arr,
            labels_df=labels_df,
            composite_dict=composite_grids[c],
        )

        # Enforce Decision D-009: Pre-committed N >= 30 independent events rule
        # Since n_indep_events < 30, classify strictly as unvalidated_exploratory
        char.validation_status = "unvalidated_exploratory"
        char.is_validated = False

        # In each validation entry, record the independent event count
        for val_entry in char.validation:
            val_entry.is_significant = False

        logger.info(
            "Cluster %d (%s): %d daily members -> %d independent events (D-009 status: %s)",
            c, char.name, n_members_daily, n_indep_events, char.validation_status,
        )
        clusters.append(char)

    result = RegimeClusteringResult(
        optimal_k=optimal_k,
        silhouette_scores=sil_scores,
        clusters=clusters,
        high_burden_threshold=round(threshold, 2),
        n_high_burden_dates=len(high_dates_arr),
        total_candidate_dates=len(daily_dates),
        composite_anomaly_grids=composite_grids,
    )

    # Export to JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    opt_sil = round(float(result.silhouette_scores[result.optimal_k]), 4)
    sil_ladder = {str(k): round(float(v), 4) for k, v in result.silhouette_scores.items()}
    serializable: Dict[str, Any] = {
        "optimal_k": result.optimal_k,
        "silhouette_score": opt_sil,
        "silhouette_ladder": sil_ladder,
        "silhouette_scores": sil_ladder,
        "high_burden_threshold": result.high_burden_threshold,
        "n_high_burden_dates": result.n_high_burden_dates,
        "n_high_burden_days": result.n_high_burden_dates,
        "total_candidate_dates": result.total_candidate_dates,
        "clusters": [],
    }

    for cl in result.clusters:
        cl_dict: Dict[str, Any] = {
            "cluster_id": cl.cluster_id,
            "name": cl.name,
            "mechanism_summary": cl.mechanism_summary,
            "n_members": cl.n_members,
            "n_independent_events": count_independent_events(cl.member_dates, min_separation_days=15),
            "member_dates": cl.member_dates,
            "seasonality": cl.seasonality,
            "threatened_subdivisions": cl.threatened_subdivisions,
            "vulnerable_leads": cl.vulnerable_leads,
            "signed_bias": cl.signed_bias,
            "mean_error_mm": cl.mean_error_mm,
            "composite_summary": cl.composite_summary,
            "validation_status": cl.validation_status,
            "is_validated": cl.is_validated,
            "validation": [
                {
                    "catalogue_name": v.catalogue_name,
                    "observed_count": v.observed_count,
                    "expected_count": v.expected_count,
                    "enrichment_ratio": v.enrichment_ratio,
                    "ci_95": list(v.ci_95),
                    "chi2_stat": v.chi2_stat,
                    "p_value": v.p_value,
                    "is_significant": False,
                }
                for v in cl.validation
            ],
        }
        serializable["clusters"].append(cl_dict)

    with open(output_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info("Saved discovered regimes to %s", output_path)

    # Also update models/run_day4/regimes.json for backwards-compatibility
    compat_path = Path("models/run_day4/regimes.json")
    if compat_path.parent.exists():
        with open(compat_path, "w") as f:
            json.dump(serializable, f, indent=2)
        logger.info("Synced discovered regimes to %s", compat_path)

    print_regime_summary(result)
    return result


def print_regime_summary(result: RegimeClusteringResult) -> None:
    """Print the complete summary of regime discovery."""
    print("=" * 72)
    print("  PROJECT PRATYAYA: SYNOPTIC REGIME DISCOVERY SUMMARY (DECISION D-009)")
    print("=" * 72)
    print(f"Selected High-Burden Cycle Dates Count: {result.n_high_burden_dates} (threshold >= {result.high_burden_threshold} busts)")
    print(f"Candidate Evaluation Range: k in [{min(result.silhouette_scores.keys())}, {max(result.silhouette_scores.keys())}]")
    print("Silhouette Ladder across Tested k:")
    for k, s in sorted(result.silhouette_scores.items()):
        star = " <-- OPTIMAL" if k == result.optimal_k else ""
        print(f"  k = {k}: silhouette = {s:.4f}{star}")
    opt_s = result.silhouette_scores[result.optimal_k]
    print(f"\nSelected Optimal k: {result.optimal_k} (silhouette = {opt_s:.4f})")
    print("Pre-committed Decision D-009 Threshold: N >= 30 independent events")
    print(f"Discovered Cluster Archetypes ({len(result.clusters)} clusters, ALL labeled unvalidated_exploratory):")
    for cl in result.clusters:
        n_indep = count_independent_events(cl.member_dates, min_separation_days=15)
        print(f"\n  Cluster {cl.cluster_id}: {cl.name}")
        print(f"    Members: {cl.n_members} cycles ({n_indep} independent events, status: {cl.validation_status})")
        print(f"    Bias: {cl.signed_bias.upper()} (mean error {cl.mean_error_mm:+.2f} mm)")
        print(f"    Vulnerable Leads: {cl.vulnerable_leads}")
        top_subs = [s['name'] for s in cl.threatened_subdivisions[:3]]
        print(f"    Threatened Subdivisions (top 3): {', '.join(top_subs)}")
        print(f"    Mechanism: {cl.mechanism_summary}")
        cat_matches = [
            f"{v.catalogue_name} (obs={v.observed_count}, exp={v.expected_count:.1f}, E={v.enrichment_ratio:.2f}x, p={v.p_value:.4f})"
            for v in cl.validation if v.observed_count > 0
        ]
        if cat_matches:
            print("    Catalogue Overlap:")
            for cm in cat_matches:
                print(f"      - {cm}")
        else:
            print("    Catalogue Overlap: None")
    print("=" * 72)


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover Synoptic Regimes under Decision D-009.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--k-min", type=int, default=3, help="Minimum clusters to evaluate")
    parser.add_argument("--k-max", type=int, default=10, help="Maximum clusters to evaluate")
    args = parser.parse_args()

    # Load canonical labels across 2000–2002
    dfs = []
    for y in [2000, 2001, 2002]:
        yd = DEFAULT_LABELS_DIR / f"year={y}"
        if yd.exists():
            for f in yd.glob("*.parquet"):
                dfs.append(pd.read_parquet(f))
    labels_df = pd.concat(dfs, ignore_index=True)
    logger.info("Loaded canonical labels: %d records across %d cycles", len(labels_df), labels_df["init_date"].nunique())

    run_regime_discovery_loyo(
        labels_df=labels_df,
        output_path=args.output_json,
        k_range=(args.k_min, args.k_max),
    )


if __name__ == "__main__":
    main()
