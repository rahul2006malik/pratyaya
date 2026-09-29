"""Unit tests for Blocked and Purged Splitter (Node F7, docs/05 §1)."""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from pratyaya.evaluate.splits import (
    CALIBRATION_YEARS,
    HELDOUT_LOG_PATH,
    HELDOUT_YEARS,
    PURGE_WINDOW_DAYS,
    TRAIN_YEARS,
    apply_boundary_purging,
    create_purged_train_val_split,
    split_temporal_folds,
)


@pytest.fixture
def sample_multiyear_df() -> pd.DataFrame:
    """Create a synthetic DataFrame spanning train and calibration years."""
    dates = [
        # Train years (2014)
        "2014-06-01",
        "2014-07-15",
        "2014-12-10",
        "2014-12-18",
        "2014-12-20",  # within 12 days of 2014-12-31 -> should be purged
        "2014-12-28",  # within 12 days -> purged
        "2014-12-31",  # within 12 days -> purged
        # Calibration years (2015, 2016)
        "2015-01-05",
        "2015-07-20",
        "2015-12-15",
        "2016-08-10",
        "2017-06-01",
        "2017-12-25",  # within 12 days of 2017-12-31 -> should be purged
    ]
    records = []
    for d in dates:
        for r in ["1", "2"]:
            records.append({
                "init_date": pd.Timestamp(d),
                "region_id": r,
                "lead_day": 5,
                "is_bust": False,
            })
    return pd.DataFrame(records)


def test_canonical_year_constants() -> None:
    """Verify canonical split year ranges per docs/05 §1.1."""
    assert 2000 in TRAIN_YEARS
    assert 2014 in TRAIN_YEARS
    assert 2015 not in TRAIN_YEARS

    assert 2015 in CALIBRATION_YEARS
    assert 2017 in CALIBRATION_YEARS
    assert 2018 not in CALIBRATION_YEARS

    assert 2018 in HELDOUT_YEARS
    assert 2019 in HELDOUT_YEARS
    assert 2020 not in HELDOUT_YEARS

    assert PURGE_WINDOW_DAYS == 12  # max_lead (10) + 2


def test_boundary_purging_logic() -> None:
    """Verify trailing 12-day boundary purge removes overlapping verification windows."""
    dates = pd.date_range("2014-12-01", "2014-12-31", freq="D")
    df = pd.DataFrame({
        "init_date": dates,
        "region_id": "1",
        "lead_day": 10,
    })

    fold_end = pd.Timestamp("2014-12-31")
    clean_df, purged_dates = apply_boundary_purging(df, fold_end_date=fold_end, purge_window_days=12)

    # 12 days before 2014-12-31: cutoff is 2014-12-19
    # Purged dates should be 2014-12-20 through 2014-12-31 (12 days)
    assert len(purged_dates) == 12
    assert max(clean_df["init_date"]) == pd.Timestamp("2014-12-19")

    # Critical causality assertion: Any lead 10 verification from clean_df must verify BEFORE or ON fold_end
    latest_init = max(clean_df["init_date"])
    latest_verification = latest_init + pd.Timedelta(days=10)
    assert latest_verification <= fold_end + pd.Timedelta(days=1)


def test_split_temporal_folds_purging(sample_multiyear_df: pd.DataFrame) -> None:
    """Verify split_temporal_folds partitions train and calibration with boundary purging."""
    res = split_temporal_folds(sample_multiyear_df, final=False)

    assert res.heldout is None
    assert res.n_train > 0
    assert res.n_calibration > 0

    # Train should have purged 2014-12-20, 2014-12-28, 2014-12-31
    train_dates = set(pd.to_datetime(res.train["init_date"]))
    assert pd.Timestamp("2014-12-18") in train_dates
    assert pd.Timestamp("2014-12-20") not in train_dates
    assert pd.Timestamp("2014-12-31") not in train_dates

    # Calibration should have purged 2017-12-25
    cal_dates = set(pd.to_datetime(res.calibration["init_date"]))
    assert pd.Timestamp("2017-06-01") in cal_dates
    assert pd.Timestamp("2017-12-25") not in cal_dates


def test_heldout_protection_blocks_unauthorized_access() -> None:
    """Verify that any dataset containing 2018–2019 raises PermissionError without final=True."""
    df_heldout = pd.DataFrame({
        "init_date": [pd.Timestamp("2018-07-01"), pd.Timestamp("2019-08-01")],
        "region_id": ["1", "2"],
        "is_bust": [False, True],
    })

    with pytest.raises(PermissionError, match="held-out years"):
        split_temporal_folds(df_heldout, final=False)


def test_heldout_access_requires_reason(tmp_path: Path) -> None:
    """Verify that final=True requires non-empty justification string and logs touch."""
    df_heldout = pd.DataFrame({
        "init_date": [pd.Timestamp("2018-07-01")],
        "region_id": ["1"],
        "is_bust": [False],
    })

    # Empty reason must raise ValueError
    with pytest.raises(ValueError, match="justification.*reason"):
        split_temporal_folds(df_heldout, final=True, reason="")

    # Valid reason succeeds
    test_log = tmp_path / "test_heldout.log"
    import pratyaya.evaluate.splits as splits_mod
    orig_log = splits_mod.HELDOUT_LOG_PATH
    splits_mod.HELDOUT_LOG_PATH = test_log
    try:
        res = split_temporal_folds(df_heldout, final=True, reason="Final Day 8 Gate verification test.")
        assert res.heldout is not None
        assert len(res.heldout) == 1
        assert test_log.exists()
        log_text = test_log.read_text(encoding="utf-8")
        assert "Final Day 8 Gate verification test." in log_text
    finally:
        splits_mod.HELDOUT_LOG_PATH = orig_log


def test_create_purged_train_val_split() -> None:
    """Verify convenience function for training-internal evaluation splits."""
    dates = pd.date_range("2000-01-01", "2002-12-31", freq="W")
    df = pd.DataFrame({
        "init_date": dates,
        "region_id": "1",
        "lead_day": 10,
        "is_bust": False,
    })

    train_df, val_df, purged = create_purged_train_val_split(
        df,
        train_years=[2000, 2001],
        val_years=[2002],
        purge_window_days=12,
    )

    assert max(pd.to_datetime(train_df["init_date"])).year <= 2001
    assert min(pd.to_datetime(val_df["init_date"])).year == 2002
    assert len(purged) > 0
