"""Blocked and Purged Temporal Splitter (Node F7).

Grain: Any DataFrame containing an 'init_date' column.

Implements the three-way temporal fold split and boundary purging per
docs/05-evaluation-protocol.md §1:
- Train: 2000–2014 (TRAIN_YEARS)
- Calibration / model selection: 2015–2017 (CALIBRATION_YEARS)
- Held-out test set: 2018–2019 (HELDOUT_YEARS) — strictly guarded behind --final

Boundary Purging (docs/05 §1.2):
A forecast initialised on 28 Dec 2014 at lead 10 verifies on 7 Jan 2015.
Without purging, that verification window sits inside the calibration fold
and folds are not independent.
Rule: between any two adjacent folds, drop all init dates within
max_lead + 2 = 12 days of the boundary.

Held-Out Guard (docs/05 §1.1, D-000):
Held-out years (2018–2019) are protected. Any attempt to access or split
them without final=True raises PermissionError. When final=True is passed,
a non-empty justification string is mandatory and is recorded to docs/HELDOUT_LOG.md.
"""

from __future__ import annotations

import logging
import os
import subprocess
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Canonical Split Years per docs/05 §1.1 and graph.json
TRAIN_YEARS: Tuple[int, ...] = tuple(range(2000, 2015))        # 2000–2014
CALIBRATION_YEARS: Tuple[int, ...] = tuple(range(2015, 2018))  # 2015–2017
HELDOUT_YEARS: Tuple[int, ...] = tuple(range(2018, 2020))      # 2018–2019
PURGE_WINDOW_DAYS: int = 12                                    # max_lead (10) + 2 days

HELDOUT_LOG_PATH: Path = Path("docs/HELDOUT_LOG.md")


@dataclass
class SplitResult:
    """Container holding temporal fold splits and boundary purge audit records."""
    train: pd.DataFrame
    calibration: pd.DataFrame
    heldout: Optional[pd.DataFrame] = None
    purged_train_dates: List[pd.Timestamp] = field(default_factory=list)
    purged_calibration_dates: List[pd.Timestamp] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_train(self) -> int:
        return len(self.train)

    @property
    def n_calibration(self) -> int:
        return len(self.calibration)

    @property
    def n_heldout(self) -> int:
        return len(self.heldout) if self.heldout is not None else 0

    @property
    def total_purged_dates(self) -> int:
        return len(self.purged_train_dates) + len(self.purged_calibration_dates)


def log_heldout_access(
    reason: str,
    target: str = "heldout_split",
    log_path: Optional[Path] = None,
) -> None:
    """Record an authorised --final access to the held-out dataset in docs/HELDOUT_LOG.md.

    Args:
        reason: Mandatory explicit justification string.
        target: Description of the evaluated target.
        log_path: Optional path to HELDOUT_LOG.md file. If None, uses module HELDOUT_LOG_PATH.

    Raises:
        ValueError: If reason is empty or whitespace-only.
    """
    if log_path is None:
        log_path = HELDOUT_LOG_PATH
    cleaned_reason = reason.strip()
    if not cleaned_reason:
        raise ValueError(
            "A non-empty justification 'reason' is strictly required when accessing "
            "the held-out set with final=True per D-000 and docs/05 §1.1."
        )

    # Gather environment metadata
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    branch = os.environ.get("GIT_BRANCH", "day2/eval-baselines")
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:
        git_commit = "unknown"

    log_entry = f"| {now_utc} | {branch} | {git_commit} | {target} | {cleaned_reason} | pending |\n"

    if log_path.exists():
        content = log_path.read_text(encoding="utf-8")
        if "| *No touches logged* |" in content:
            content = content.replace("| *No touches logged* | | | | | |\n", "")
        content += log_entry
        log_path.write_text(content, encoding="utf-8")
    else:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            "# Held-Out Set Touch Log — Pratyaya\n\n"
            "This file is an immutable log of every invocation that accesses or evaluates "
            "against the held-out dataset (years 2018–2019) via `--final`.\n\n"
            "| Timestamp (UTC) | Session / Branch | Git Commit | Evaluated Target | Justification / Reason | Outcome / PR-AUC |\n"
            "|---|---|---|---|---|---|\n"
        )
        log_path.write_text(header + log_entry, encoding="utf-8")

    logger.warning("HELD-OUT SET ACCESSED with final=True: %s", cleaned_reason)


def apply_boundary_purging(
    df: pd.DataFrame,
    fold_end_date: pd.Timestamp,
    purge_window_days: int = PURGE_WINDOW_DAYS,
    init_date_col: str = "init_date",
) -> Tuple[pd.DataFrame, List[pd.Timestamp]]:
    """Purge trailing initialization dates within purge_window_days before fold boundary.

    Eliminates verification window overlap into adjacent subsequent folds:
    Drops all records where init_date > (fold_end_date - purge_window_days).

    Args:
        df: Input DataFrame for the preceding fold.
        fold_end_date: Maximum boundary date of this fold (e.g. 2014-12-31).
        purge_window_days: Number of trailing boundary days to purge (default: 12).
        init_date_col: Column name containing forecast initialization dates.

    Returns:
        Tuple[pd.DataFrame, List[pd.Timestamp]]:
            Filtered DataFrame and list of dropped unique initialization dates.
    """
    if df.empty:
        return df, []

    df_work = df.copy()
    init_series = pd.to_datetime(df_work[init_date_col])

    # Purge threshold: drop any date that could verify after fold_end_date
    purge_cutoff = pd.to_datetime(fold_end_date) - pd.Timedelta(days=purge_window_days)
    purge_mask = (init_series > purge_cutoff) & (init_series <= fold_end_date)

    purged_dates = sorted(init_series[purge_mask].unique())
    clean_df = df_work[~purge_mask].copy()

    if len(purged_dates) > 0:
        logger.info(
            "Boundary purging: dropped %d rows across %d init dates in window (%s to %s]",
            int(purge_mask.sum()),
            len(purged_dates),
            purge_cutoff.strftime("%Y-%m-%d"),
            pd.to_datetime(fold_end_date).strftime("%Y-%m-%d"),
        )

    return clean_df, [pd.Timestamp(d) for d in purged_dates]


def split_temporal_folds(
    df: pd.DataFrame,
    init_date_col: str = "init_date",
    train_years: Iterable[int] = TRAIN_YEARS,
    calibration_years: Iterable[int] = CALIBRATION_YEARS,
    heldout_years: Iterable[int] = HELDOUT_YEARS,
    purge_window_days: int = PURGE_WINDOW_DAYS,
    final: bool = False,
    reason: Optional[str] = None,
) -> SplitResult:
    """Partition dataset into Train, Calibration, and guarded Held-out folds with boundary purging.

    Args:
        df: Input DataFrame with init_date_col.
        init_date_col: Name of column containing initialization timestamps.
        train_years: Years assigned to training fold (default: 2000–2014).
        calibration_years: Years assigned to calibration fold (default: 2015–2017).
        heldout_years: Years assigned to held-out validation fold (default: 2018–2019).
        purge_window_days: Purge window size in days (default: 12 = max_lead + 2).
        final: Explicit boolean flag authorizing held-out access.
        reason: Required explanation string when final=True.

    Returns:
        SplitResult: Dataclass containing split frames, purge logs, and metadata.

    Raises:
        PermissionError: If data contains held-out years or held-out split is accessed
                         without final=True per docs/05 §1.1 and D-000.
        ValueError: If init_date_col is missing or invalid.
    """
    if init_date_col not in df.columns:
        raise ValueError(f"Required column '{init_date_col}' not found in DataFrame columns: {list(df.columns)}")

    train_set = set(int(y) for y in train_years)
    cal_set = set(int(y) for y in calibration_years)
    heldout_set = set(int(y) for y in heldout_years)

    df_all = df.copy()
    df_all["_dt"] = pd.to_datetime(df_all[init_date_col])
    years_present = set(df_all["_dt"].dt.year.unique())

    # Check for held-out data presence
    heldout_detected = years_present.intersection(heldout_set)
    if heldout_detected and not final:
        raise PermissionError(
            f"Input data contains held-out years {sorted(heldout_detected)} (2018–2019). "
            "Access to the held-out set is strictly prohibited outside the --final protocol "
            "per docs/05 §1.1 and Decision D-000."
        )

    # If final=True is requested, record to HELDOUT_LOG.md
    if final:
        if not reason:
            raise ValueError(
                "A non-empty justification string 'reason' is required when passing "
                "final=True per docs/05 §1.1 and Decision D-000."
            )
        log_heldout_access(reason=reason, target=f"split_temporal_folds({sorted(heldout_set)})")

    # 1. Extract Training Fold
    train_mask = df_all["_dt"].dt.year.isin(train_set)
    raw_train = df_all[train_mask].drop(columns=["_dt"])

    # Apply 12-day boundary purge at end of training fold (e.g. 2014-12-31)
    if not raw_train.empty:
        max_train_year = max(train_set)
        train_boundary = pd.Timestamp(year=max_train_year, month=12, day=31)
        clean_train, purged_train_dates = apply_boundary_purging(
            raw_train,
            fold_end_date=train_boundary,
            purge_window_days=purge_window_days,
            init_date_col=init_date_col,
        )
    else:
        clean_train = raw_train
        purged_train_dates = []

    # 2. Extract Calibration Fold
    cal_mask = df_all["_dt"].dt.year.isin(cal_set)
    raw_cal = df_all[cal_mask].drop(columns=["_dt"])

    # Apply boundary purge at end of calibration fold (e.g. 2017-12-31)
    if not raw_cal.empty:
        max_cal_year = max(cal_set)
        cal_boundary = pd.Timestamp(year=max_cal_year, month=12, day=31)
        clean_cal, purged_cal_dates = apply_boundary_purging(
            raw_cal,
            fold_end_date=cal_boundary,
            purge_window_days=purge_window_days,
            init_date_col=init_date_col,
        )
    else:
        clean_cal = raw_cal
        purged_cal_dates = []

    # 3. Extract Held-Out Fold (only if final=True)
    heldout_df: Optional[pd.DataFrame] = None
    if final:
        heldout_mask = df_all["_dt"].dt.year.isin(heldout_set)
        heldout_df = df_all[heldout_mask].drop(columns=["_dt"])

    metadata = {
        "train_years": sorted(train_set),
        "calibration_years": sorted(cal_set),
        "heldout_years": sorted(heldout_set),
        "purge_window_days": purge_window_days,
        "final_authorised": final,
        "n_raw_train": len(raw_train),
        "n_purged_train": len(raw_train) - len(clean_train),
        "n_raw_cal": len(raw_cal),
        "n_purged_cal": len(raw_cal) - len(clean_cal),
    }

    return SplitResult(
        train=clean_train,
        calibration=clean_cal,
        heldout=heldout_df,
        purged_train_dates=purged_train_dates,
        purged_calibration_dates=purged_cal_dates,
        metadata=metadata,
    )


def create_purged_train_val_split(
    df: pd.DataFrame,
    train_years: Iterable[int],
    val_years: Iterable[int],
    purge_window_days: int = PURGE_WINDOW_DAYS,
    init_date_col: str = "init_date",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
    """Convenience utility to partition within training years (e.g. 2000–2001 train vs 2002 val).

    Applies boundary purging of purge_window_days at the end of the train period.

    Args:
        df: Input DataFrame.
        train_years: Calendar years for training split.
        val_years: Calendar years for validation/calibration split.
        purge_window_days: Days before boundary to purge (default: 12).
        init_date_col: Column name containing initialization dates.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp]]:
            (clean_train_df, val_df, list_of_purged_train_dates)
    """
    train_set = set(int(y) for y in train_years)
    val_set = set(int(y) for y in val_years)

    # Protect against accidental inclusion of 2018–2019
    overlap_heldout = (train_set | val_set).intersection(set(HELDOUT_YEARS))
    if overlap_heldout:
        raise PermissionError(
            f"Attempted to split held-out years {sorted(overlap_heldout)} outside --final."
        )

    df_work = df.copy()
    df_work["_dt"] = pd.to_datetime(df_work[init_date_col])

    raw_train = df_work[df_work["_dt"].dt.year.isin(train_set)].drop(columns=["_dt"])
    val_df = df_work[df_work["_dt"].dt.year.isin(val_set)].drop(columns=["_dt"])

    if not raw_train.empty:
        max_train_year = max(train_set)
        train_boundary = pd.Timestamp(year=max_train_year, month=12, day=31)
        clean_train, purged_dates = apply_boundary_purging(
            raw_train,
            fold_end_date=train_boundary,
            purge_window_days=purge_window_days,
            init_date_col=init_date_col,
        )
    else:
        clean_train = raw_train
        purged_dates = []

    return clean_train, val_df, purged_dates
