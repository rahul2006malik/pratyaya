"""Independent Bust Event Clustering (Node F8).

Grain: (valid_date, region_id) -> spatiotemporal connected components.

Implements 3D connected-component bust event clustering across (time, lat, lon)
per docs/05-evaluation-protocol.md §4.1:
- A single synoptic event (e.g. a poorly-forecast monsoon depression) produces busts
  across multiple neighbouring subdivisions, consecutive valid dates, and multiple
  forecast leads.
- Treating each positive region-lead row as independent inflates sample size by ~7×
  and produces spuriously narrow confidence intervals.
- This module groups correlated bust rows into independent meteorological bust events
  by spatiotemporal connected-component analysis:
  Adjacency rule:
    1. Temporal proximity: |valid_date_1 - valid_date_2| <= temporal_window_days (default 1 day).
    2. Spatial proximity: neighbouring subdivisions (topological bordering) or same region.

Outputs:
- Adds 'event_id' column to DataFrame (1..N for bust events, -1 for non-busts).
- Computes sample inflation factor: N_bust_rows / N_events.
- Summarises event durations and geographic footprints.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_REGIONS_PARQUET = Path("data/canonical/regions.parquet")


@dataclass
class EventClusteringResult:
    """Results from spatiotemporal bust event clustering."""
    labeled_df: pd.DataFrame
    n_events: int
    n_bust_rows: int
    inflation_factor: float
    events_summary: pd.DataFrame
    metadata: Dict[str, Any] = field(default_factory=dict)


def compute_subdivision_adjacency(
    regions_parquet_path: Path = DEFAULT_REGIONS_PARQUET,
    centroid_fallback_dist_km: float = 350.0,
) -> Dict[str, Set[str]]:
    """Build topological adjacency mapping for the 36 meteorological subdivisions.

    Each subdivision is adjacent to itself and to all subdivisions with which it
    shares a boundary. For island or coastal subdivisions, distance between centroids
    is used as a secondary check.

    Args:
        regions_parquet_path: Path to canonical regions.parquet table.
        centroid_fallback_dist_km: Maximum centroid distance to consider adjacent.

    Returns:
        Dict[str, Set[str]]: Mapping from region_id (str) to set of adjacent region_ids (strs).
    """
    adjacency: Dict[str, Set[str]] = defaultdict(set)

    if regions_parquet_path.exists():
        try:
            import shapely.wkb

            df_reg = pd.read_parquet(regions_parquet_path)
            geoms = {}
            centroids = {}
            for _, row in df_reg.iterrows():
                rid = str(row["region_id"])
                g = shapely.wkb.loads(row["geometry"])
                geoms[rid] = g
                c = g.centroid
                centroids[rid] = (c.y, c.x)  # (lat, lon)
                adjacency[rid].add(rid)

            all_rids = list(geoms.keys())
            for i in range(len(all_rids)):
                r1 = all_rids[i]
                g1 = geoms[r1]
                lat1, lon1 = centroids[r1]
                for j in range(i + 1, len(all_rids)):
                    r2 = all_rids[j]
                    g2 = geoms[r2]
                    # Check boundary intersection / touching
                    if g1.touches(g2) or g1.intersects(g2):
                        adjacency[r1].add(r2)
                        adjacency[r2].add(r1)
                    else:
                        # Haversine distance on centroids
                        lat2, lon2 = centroids[r2]
                        d_km = haversine_distance_km(lat1, lon1, lat2, lon2)
                        if d_km <= centroid_fallback_dist_km:
                            adjacency[r1].add(r2)
                            adjacency[r2].add(r1)
            logger.info("Computed subdivision adjacency topology for %d regions.", len(adjacency))
            return dict(adjacency)
        except Exception as e:
            logger.warning("Could not compute geometric adjacency from %s: %s. Using default topology.", regions_parquet_path, e)

    # Fallback default: numeric integer adjacency (+/- 1) and self
    for i in range(1, 37):
        sid = str(i)
        adjacency[sid] = {sid}
        if i > 1:
            adjacency[sid].add(str(i - 1))
        if i < 36:
            adjacency[sid].add(str(i + 1))
    return dict(adjacency)


def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance in kilometres between two points."""
    r_km = 6371.0
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    a = (np.sin(delta_phi / 2.0) ** 2 +
         np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r_km * c)


def cluster_bust_events(
    df: pd.DataFrame,
    bust_col: str = "is_bust",
    valid_date_col: str = "valid_date",
    init_date_col: str = "init_date",
    lead_day_col: str = "lead_day",
    region_col: str = "region_id",
    temporal_window_days: int = 1,
    regions_parquet_path: Path = DEFAULT_REGIONS_PARQUET,
    custom_adjacency: Optional[Dict[str, Set[str]]] = None,
) -> EventClusteringResult:
    """Perform 3D connected-component bust event clustering across time and space.

    Args:
        df: Input DataFrame containing forecast evaluation / label records.
        bust_col: Boolean or integer column indicating bust occurrence (default: 'is_bust').
        valid_date_col: Column indicating verification target date. If missing, computed from
                        init_date + (lead_day - 1).
        init_date_col: Column containing initialization date.
        lead_day_col: Column containing forecast lead day.
        region_col: Column containing subdivision/region identifier.
        temporal_window_days: Maximum calendar day difference for temporal connection (default: 1 day).
        regions_parquet_path: Path to regions.parquet for subdivision adjacency graph.
        custom_adjacency: Optional explicit adjacency graph mapping region_id to set of neighbours.

    Returns:
        EventClusteringResult: Contains labeled DataFrame with 'event_id', event count,
                              inflation factor, and per-event summary metrics.
    """
    df_out = df.copy()

    # Ensure valid_date exists
    if valid_date_col not in df_out.columns:
        if init_date_col in df_out.columns and lead_day_col in df_out.columns:
            inits = pd.to_datetime(df_out[init_date_col])
            leads = df_out[lead_day_col].astype(int)
            df_out[valid_date_col] = inits + pd.to_timedelta(leads - 1, unit="D")
        else:
            raise ValueError(
                f"Cannot determine '{valid_date_col}'. DataFrame must contain '{valid_date_col}' "
                f"or both '{init_date_col}' and '{lead_day_col}'."
            )

    df_out[valid_date_col] = pd.to_datetime(df_out[valid_date_col])
    df_out[region_col] = df_out[region_col].astype(str)
    df_out["event_id"] = -1

    # Filter to positive bust cases
    bust_mask = df_out[bust_col].astype(bool)
    n_bust_rows = int(bust_mask.sum())

    if n_bust_rows == 0:
        empty_summary = pd.DataFrame(columns=[
            "event_id", "start_valid_date", "end_valid_date",
            "duration_days", "n_regions", "regions", "n_rows"
        ])
        return EventClusteringResult(
            labeled_df=df_out,
            n_events=0,
            n_bust_rows=0,
            inflation_factor=1.0,
            events_summary=empty_summary,
            metadata={"temporal_window_days": temporal_window_days},
        )

    # Build or retrieve subdivision adjacency graph
    adjacency = custom_adjacency or compute_subdivision_adjacency(regions_parquet_path)

    # 1. Group positive bust records by unique (valid_date, region_id) node
    bust_indices = df_out.index[bust_mask].tolist()
    node_to_rows: Dict[Tuple[pd.Timestamp, str], List[int]] = defaultdict(list)
    for idx in bust_indices:
        v_dt = df_out.at[idx, valid_date_col]
        r_id = df_out.at[idx, region_col]
        node_to_rows[(v_dt, r_id)].append(idx)

    nodes = list(node_to_rows.keys())
    n_nodes = len(nodes)

    # 2. Build graph between positive (valid_date, region_id) nodes
    # For fast spatial-temporal neighbour search:
    date_to_node_indices: Dict[pd.Timestamp, List[int]] = defaultdict(list)
    for node_idx, (v_dt, r_id) in enumerate(nodes):
        date_to_node_indices[v_dt].append(node_idx)

    adj_list: Dict[int, List[int]] = defaultdict(list)
    unique_dates = sorted(date_to_node_indices.keys())

    for u_dt in unique_dates:
        # Check current day and adjacent days within temporal_window_days
        node_indices_today = date_to_node_indices[u_dt]
        neighbor_dates = [
            d for d in unique_dates
            if 0 <= (d - u_dt).days <= temporal_window_days
        ]

        for i_idx in node_indices_today:
            r1 = nodes[i_idx][1]
            r1_neighbors = adjacency.get(r1, {r1})

            for n_dt in neighbor_dates:
                for j_idx in date_to_node_indices[n_dt]:
                    if i_idx == j_idx:
                        continue
                    r2 = nodes[j_idx][1]
                    if r2 in r1_neighbors or r1 == r2:
                        adj_list[i_idx].append(j_idx)
                        adj_list[j_idx].append(i_idx)

    # 3. Connected-component labeling via Breadth-First Search (BFS)
    visited: Set[int] = set()
    component_labels: Dict[int, int] = {}
    current_event_id = 1

    for start_node in range(n_nodes):
        if start_node not in visited:
            queue = deque([start_node])
            visited.add(start_node)
            component_labels[start_node] = current_event_id

            while queue:
                curr = queue.popleft()
                for neighbor in adj_list[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        component_labels[neighbor] = current_event_id
                        queue.append(neighbor)

            current_event_id += 1

    n_events = current_event_id - 1

    # 4. Map event_id back to individual DataFrame rows
    event_rows: Dict[int, List[int]] = defaultdict(list)
    for node_idx, ev_id in component_labels.items():
        row_indices = node_to_rows[nodes[node_idx]]
        df_out.loc[row_indices, "event_id"] = ev_id
        event_rows[ev_id].extend(row_indices)

    # 5. Build summary table of independent bust events
    event_records = []
    for ev_id in range(1, n_events + 1):
        rows = event_rows[ev_id]
        ev_slice = df_out.loc[rows]
        v_dates = ev_slice[valid_date_col]
        regions_affected = sorted(ev_slice[region_col].unique())
        min_date = v_dates.min()
        max_date = v_dates.max()
        duration = int((max_date - min_date).days) + 1

        event_records.append({
            "event_id": ev_id,
            "start_valid_date": min_date.strftime("%Y-%m-%d"),
            "end_valid_date": max_date.strftime("%Y-%m-%d"),
            "duration_days": duration,
            "n_regions": len(regions_affected),
            "regions": regions_affected,
            "n_rows": len(rows),
        })

    events_summary = pd.DataFrame(event_records).sort_values("n_rows", ascending=False).reset_index(drop=True)
    inflation_factor = float(n_bust_rows) / max(1.0, float(n_events))

    logger.info(
        "Clustered %d bust rows into N = %d independent meteorological events "
        "(inflation factor: %.2f rows/event).",
        n_bust_rows,
        n_events,
        inflation_factor,
    )

    return EventClusteringResult(
        labeled_df=df_out,
        n_events=n_events,
        n_bust_rows=n_bust_rows,
        inflation_factor=inflation_factor,
        events_summary=events_summary,
        metadata={
            "temporal_window_days": temporal_window_days,
            "mean_duration_days": float(events_summary["duration_days"].mean()) if not events_summary.empty else 0.0,
            "mean_regions_per_event": float(events_summary["n_regions"].mean()) if not events_summary.empty else 0.0,
        },
    )
