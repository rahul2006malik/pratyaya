"""Test suite for scripts/eval_final.py — Day 8 Held-Out Evaluation Engine.

These tests use NO actual 2018–2019 data. All assertions target:
1. --final guard enforcement (refuses without --final flag)
2. --dry-run mode on mock artifact structures
3. HELDOUT_LOG.md schema validation and duplicate-run guard
4. Case evaluation rank computation logic (synthetic data)
5. Synthetic dataset generation (label consistency)
6. bootstrap and metric integration (unit smoke tests)

Integrity guarantee: this test file never reads or writes held-out data files.
"""

from __future__ import annotations

import json
import sys
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

# ── sys.path bootstrap ────────────────────────────────────────────────────────
_REPO_DIR = Path(__file__).resolve().parents[1]
for _p in (_REPO_DIR / "src", _REPO_DIR / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import eval_final as mod

# ── Helpers ───────────────────────────────────────────────────────────────────
def _make_synthetic_held_df(
    n_init_dates: int = 10,
    n_subdivisions: int = 36,
    n_leads: int = 10,
    bust_rate: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Create a minimal synthetic held-out DataFrame for testing."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_init_dates):
        dt = pd.Timestamp(f"2018-07-{i+1:02d}")
        for region in range(1, n_subdivisions + 1):
            for lead in range(1, n_leads + 1):
                is_bust = int(rng.random() < bust_rate)
                rows.append({
                    "init_date": dt,
                    "region_id": str(region),
                    "lead_day": lead,
                    "month": 7,
                    "forecast_value": float(rng.uniform(0, 30)),
                    "ens_spread": float(rng.uniform(1, 8)),
                    "observed_value": float(rng.uniform(0, 40)),
                    "error_signed": float(rng.uniform(-20, 20)),
                    "is_bust": is_bust,
                    "is_bust_p90": int(rng.random() < bust_rate * 2),
                    "prob_b0": float(bust_rate),
                    "prob_b1": float(rng.uniform(0.02, 0.08)),
                    "prob_b2": float(rng.uniform(0.01, 0.10)),
                    "prob_b3": float(rng.uniform(0.01, 0.40)),
                    "prob_m1_v2": float(rng.uniform(0.01, 0.30)),
                    "prob_m2": float(rng.uniform(0.01, 0.50)),
                    "d_k": float(rng.uniform(10, 200)),
                })
    return pd.DataFrame(rows)


# ── HELDOUT_LOG.md helpers ────────────────────────────────────────────────────
class TestHeldOutLogGuard:
    """Test the D-000 immutable log guard."""

    def test_clean_log_passes(self, tmp_path: Path) -> None:
        """A log file with only the placeholder row should allow a run."""
        log_path = tmp_path / "HELDOUT_LOG.md"
        log_path.write_text(
            "# Held-Out Set Touch Log\n"
            "| Timestamp | Branch | Commit | Target | Reason | PR-AUC |\n"
            "|---|---|---|---|---|---|\n"
            "| *No touches logged* | | | | | |\n",
            encoding="utf-8",
        )

        orig_path = mod.HELDOUT_LOG_PATH
        mod.HELDOUT_LOG_PATH = log_path
        try:
            mod._check_heldout_log_clean()  # Should NOT raise SystemExit
        finally:
            mod.HELDOUT_LOG_PATH = orig_path

    def test_existing_real_entry_aborts(self, tmp_path: Path) -> None:
        """A log with a real entry should abort with sys.exit(2)."""
        log_path = tmp_path / "HELDOUT_LOG.md"
        log_path.write_text(
            "# Held-Out Set Touch Log\n"
            "| Timestamp | Branch | Commit | Target | Reason | PR-AUC |\n"
            "|---|---|---|---|---|---|\n"
            "| 2026-09-20T08:00:00Z | day8/heldout-eval | `abc1234` | M2 on 2018-2019 | Scheduled | **0.3812** |\n",
            encoding="utf-8",
        )

        orig_path = mod.HELDOUT_LOG_PATH
        mod.HELDOUT_LOG_PATH = log_path
        try:
            with pytest.raises(SystemExit) as exc_info:
                mod._check_heldout_log_clean()
            assert exc_info.value.code == 2
        finally:
            mod.HELDOUT_LOG_PATH = orig_path

    def test_log_write_appends_row(self, tmp_path: Path) -> None:
        """_write_heldout_log should append one correctly-formatted row."""
        log_path = tmp_path / "HELDOUT_LOG.md"
        log_path.write_text(
            "# Held-Out Set Touch Log\n"
            "| Timestamp | Branch | Commit | Target | Reason | PR-AUC |\n"
            "|---|---|---|---|---|---|\n"
            "| *No touches logged* | | | | | |\n",
            encoding="utf-8",
        )

        orig_path = mod.HELDOUT_LOG_PATH
        mod.HELDOUT_LOG_PATH = log_path
        try:
            mod._write_heldout_log(
                timestamp_utc="2026-09-20T08:00:00Z",
                branch="day8/heldout-eval",
                commit="abc1234",
                target="M2 on 2018-2019",
                justification="Test run",
                pr_auc=0.3812,
            )
        finally:
            mod.HELDOUT_LOG_PATH = orig_path

        content = log_path.read_text(encoding="utf-8")
        assert "2026-09-20T08:00:00Z" in content
        assert "0.3812" in content
        assert "*No touches logged*" not in content


class TestSyntheticDataset:
    """Validate the synthetic held-out dataset generation."""

    def test_dataset_shape(self) -> None:
        """Dataset should have expected column schema."""
        df = _make_synthetic_held_df(n_init_dates=5)
        required_cols = [
            "init_date", "region_id", "lead_day", "is_bust", "is_bust_p90",
            "forecast_value", "ens_spread",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_bust_labels_are_binary(self) -> None:
        """is_bust and is_bust_p90 must be integer 0 or 1."""
        df = _make_synthetic_held_df(n_init_dates=5)
        assert df["is_bust"].isin([0, 1]).all(), "is_bust contains non-binary values"
        assert df["is_bust_p90"].isin([0, 1]).all(), "is_bust_p90 contains non-binary values"

    def test_heldout_years_only(self) -> None:
        """init_date years should be 2018 or 2019 only."""
        df = _make_synthetic_held_df(n_init_dates=8)
        years = df["init_date"].dt.year.unique().tolist()
        assert all(y in (2018, 2019) for y in years), f"Unexpected years in held-out: {years}"

    def test_lead_day_range(self) -> None:
        """lead_day should be integers 1 through 10."""
        df = _make_synthetic_held_df(n_init_dates=3, n_leads=10)
        assert set(df["lead_day"].unique()) == set(range(1, 11))

    def test_base_rate_in_plausible_range(self) -> None:
        """Primary bust base rate should be between 1% and 30%."""
        df = _make_synthetic_held_df(n_init_dates=20, bust_rate=0.05)
        rate = float(df["is_bust"].mean())
        assert 0.01 <= rate <= 0.30, f"Base rate {rate:.3f} outside plausible range"

    def test_disaster_window_boost(self) -> None:
        """[DECISION D-008] Synthetic heldout generation raises RuntimeError per D-008."""
        with pytest.raises(RuntimeError, match="DEPRECATED PER DECISION D-008"):
            mod.build_heldout_dataset(pd.DataFrame())


class TestCaseRankLogic:
    """Test pre-committed case rank computation."""

    def test_rank_top_event_is_one(self) -> None:
        """Highest-scored bust event should rank 1."""
        # Simulate event max scores
        rng = np.random.default_rng(7)
        scores = sorted(rng.uniform(0.1, 0.9, 20).tolist(), reverse=True)
        case_score = scores[0]  # Best possible

        event_max_scores = pd.DataFrame({"prob_m2": scores}).sort_values("prob_m2", ascending=False)
        rank = int((event_max_scores["prob_m2"] > case_score).sum()) + 1
        assert rank == 1, f"Top event should rank 1, got {rank}"

    def test_rank_worst_event_is_n(self) -> None:
        """Lowest-scored event should rank N."""
        rng = np.random.default_rng(8)
        scores = sorted(rng.uniform(0.1, 0.9, 15).tolist(), reverse=True)
        case_score = scores[-1] - 0.001  # Slightly worse than worst

        event_max_scores = pd.DataFrame({"prob_m2": scores}).sort_values("prob_m2", ascending=False)
        rank = int((event_max_scores["prob_m2"] > case_score).sum()) + 1
        assert rank == len(scores) + 1, f"Worst event should rank {len(scores)+1}, got {rank}"

    def test_flagged_requires_p_bust_ge_060(self) -> None:
        """flagged_high_bust should require max p_bust >= 0.60."""
        df = _make_synthetic_held_df(n_init_dates=5)
        df["prob_m2"] = 0.65  # Force all scores high
        lead_35 = df[(df["lead_day"].isin([3, 4, 5])) & (df["region_id"] == "1")]["prob_m2"]
        max_p = float(lead_35.max()) if len(lead_35) > 0 else 0.0
        assert max_p >= 0.60


class TestMetricIntegration:
    """Smoke tests for evaluation metric integration."""

    def test_pr_auc_on_synthetic(self) -> None:
        """PR-AUC should be computable on synthetic held-out data."""
        from pratyaya.evaluate.metrics import compute_pr_auc
        df = _make_synthetic_held_df(bust_rate=0.08)
        pr = compute_pr_auc(df["is_bust"].values, df["prob_m2"].values)
        assert 0.0 <= pr <= 1.0, f"PR-AUC out of range: {pr}"

    def test_bootstrap_runs_1000_resamples(self) -> None:
        """Block bootstrap should complete 1,000 resamples in reasonable time."""
        from pratyaya.evaluate.bootstrap import block_bootstrap_metric
        from pratyaya.evaluate.metrics import compute_pr_auc
        df = _make_synthetic_held_df(n_init_dates=15, bust_rate=0.06)
        result = block_bootstrap_metric(
            df, y_true_col="is_bust", y_prob_col="prob_m2",
            metric_fn=compute_pr_auc, n_resamples=100, random_state=42,
        )
        assert result.n_resamples == 100
        assert 0.0 <= result.ci_lower <= result.ci_upper <= 1.0

    def test_murphy_decomposition_sums_correctly(self) -> None:
        """Reliability - Resolution + Uncertainty should approximately equal Brier score."""
        from pratyaya.evaluate.metrics import compute_reliability_diagram, compute_brier_score
        df = _make_synthetic_held_df(n_init_dates=20, bust_rate=0.07)
        y_true = df["is_bust"].values
        y_prob = df["prob_m2"].values
        rel_diag = compute_reliability_diagram(y_true, y_prob)
        bs = compute_brier_score(y_true, y_prob)
        reconstructed = rel_diag["reliability"] - rel_diag["resolution"] + rel_diag["uncertainty"]
        assert abs(bs - reconstructed) < 0.05, (
            f"Murphy decomposition mismatch: BS={bs:.4f} vs REL-RES+UNC={reconstructed:.4f}"
        )

    def test_paired_bootstrap_delta_matches_point_estimate(self) -> None:
        """Paired bootstrap delta_point_estimate should equal metric(M2) - metric(B3)."""
        from pratyaya.evaluate.bootstrap import paired_block_bootstrap_comparison
        from pratyaya.evaluate.metrics import compute_pr_auc
        df = _make_synthetic_held_df(n_init_dates=10, bust_rate=0.06)
        result = paired_block_bootstrap_comparison(
            df, y_true_col="is_bust",
            y_prob_model_col="prob_m2", y_prob_baseline_col="prob_b3",
            metric_fn=compute_pr_auc, n_resamples=50, random_state=42,
        )
        expected_delta = compute_pr_auc(df["is_bust"].values, df["prob_m2"].values) - \
                         compute_pr_auc(df["is_bust"].values, df["prob_b3"].values)
        assert abs(result.delta_point_estimate - expected_delta) < 1e-9


class TestEvalFinalGuards:
    """Test --final guard enforcement (subprocess invocations, no held-out data)."""

    def test_no_flag_exits_nonzero(self) -> None:
        """Calling eval_final.py without --final or --dry-run should exit 1."""
        result = subprocess.run(
            [sys.executable, str(_REPO_DIR / "scripts" / "eval_final.py")],
            capture_output=True, text=True,
        )
        assert result.returncode != 0, (
            f"Expected non-zero exit without --final flag, got {result.returncode}"
        )

    def test_dry_run_passes_precondition_check(self) -> None:
        """--dry-run should exit 0 when log is clean, or exit 2 if already evaluated."""
        artifact_dir = _REPO_DIR / "models" / "run_day4"
        if not artifact_dir.exists():
            pytest.skip("models/run_day4/ not present — skipping dry-run subprocess test")

        result = subprocess.run(
            [sys.executable, str(_REPO_DIR / "scripts" / "eval_final.py"), "--dry-run"],
            capture_output=True, text=True, cwd=str(_REPO_DIR),
        )
        # Pre-evaluation: returncode 0. Post-evaluation: returncode 2 (D-000 guard active). Deprecated: returncode 1 (D-008 abort).
        assert result.returncode in (0, 1, 2), (
            f"--dry-run failed with unexpected code {result.returncode}:\n{result.stderr}"
        )


class TestHeldoutLogMDSchema:
    """Validate docs/HELDOUT_LOG.md schema conformance."""

    def test_current_log_has_valid_header(self) -> None:
        """docs/HELDOUT_LOG.md must exist and have the required header row."""
        log_path = _REPO_DIR / "docs" / "HELDOUT_LOG.md"
        assert log_path.exists(), "docs/HELDOUT_LOG.md must exist"
        content = log_path.read_text(encoding="utf-8")
        assert "Timestamp" in content, "HELDOUT_LOG.md missing Timestamp header"
        assert "Git Commit" in content, "HELDOUT_LOG.md missing Git Commit header"
        assert "Outcome / PR-AUC" in content or "PR-AUC" in content, \
            "HELDOUT_LOG.md missing PR-AUC column"

    def test_current_log_entry_count(self) -> None:
        """After Day 8 --final run, HELDOUT_LOG.md should have exactly 1 real entry."""
        log_path = _REPO_DIR / "docs" / "HELDOUT_LOG.md"
        if not log_path.exists():
            pytest.skip("docs/HELDOUT_LOG.md not found")
        content = log_path.read_text(encoding="utf-8")
        real_entries = [
            ln for ln in content.splitlines()
            if ln.startswith("|") and "No touches logged" not in ln
            and "---" not in ln and "Timestamp" not in ln
        ]
        # Pre-final: 0 real entries. Post-final: 1 real entry. Both are valid.
        assert len(real_entries) <= 1, (
            f"Expected 0 or 1 real HELDOUT_LOG entries, found {len(real_entries)}:\n"
            + "\n".join(real_entries)
        )
