"""Probability Calibration & Reliability Curve Analysis (Node G4, docs/03 §7.2).

Calibrates raw forecast bust probabilities using Isotonic Regression or Platt Scaling (logistic)
fitted strictly on the calibration fold (e.g. 2002).
Decomposes Brier score into Murphy (1973) Resolution, Reliability, and Uncertainty components.
Ensures the reliability diagram sits cleanly on the diagonal.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pratyaya.evaluate.metrics import (
    compute_brier_score,
    compute_brier_skill_score,
    compute_reliability_diagram,
)

logger = logging.getLogger(__name__)


class ProbabilityCalibrator:
    """Monotonic probability calibration using Isotonic Regression or Platt Scaling (docs/03 §7.2).

    Fits strictly on calibration fold predictions (not in-fold, not held-out).
    Guarantees outputs in [0.0, 1.0] and monotonicity.
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method.lower()
        if self.method not in ["isotonic", "platt", "sigmoid"]:
            raise ValueError(f"Unknown calibration method '{method}'. Supported: 'isotonic', 'platt'.")

        self.calibrator: Optional[Union[IsotonicRegression, LogisticRegression]] = None
        self.is_fitted: bool = False

    def fit(self, y_true: np.ndarray, y_prob: np.ndarray) -> ProbabilityCalibrator:
        """Fit calibration model on calibration fold predictions.

        Args:
            y_true: Binary ground truth outcomes in {0, 1}.
            y_prob: Raw predicted probabilities in [0.0, 1.0].

        Returns:
            ProbabilityCalibrator: Self instance.
        """
        y_t = np.asarray(y_true, dtype=int)
        y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

        if len(y_t) == 0:
            raise ValueError("Cannot fit calibrator on empty data.")

        if self.method == "isotonic":
            # Isotonic regression enforces monotonic non-decreasing mapping
            self.calibrator = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self.calibrator.fit(y_p, y_t)
        elif self.method in ["platt", "sigmoid"]:
            # Platt scaling: 1D logistic regression logit(p_cal) = a * logit(p_raw) + b
            eps = 1e-6
            p_clipped = np.clip(y_p, eps, 1.0 - eps)
            logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)

            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(logit_p, y_t)
            self.calibrator = lr

        self.is_fitted = True
        logger.info("Fitted ProbabilityCalibrator (method=%s, N=%d)", self.method, len(y_t))
        return self

    def predict(self, y_prob: np.ndarray) -> np.ndarray:
        """Apply calibration mapping to raw probabilities.

        Args:
            y_prob: Raw probabilities in [0.0, 1.0].

        Returns:
            np.ndarray: Calibrated probabilities in [0.0, 1.0].
        """
        if not self.is_fitted or self.calibrator is None:
            raise RuntimeError("ProbabilityCalibrator must be fitted before predict.")

        y_p = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)

        if self.method == "isotonic":
            cal_p = self.calibrator.predict(y_p)
        elif self.method in ["platt", "sigmoid"]:
            eps = 1e-6
            p_clipped = np.clip(y_p, eps, 1.0 - eps)
            logit_p = np.log(p_clipped / (1.0 - p_clipped)).reshape(-1, 1)
            cal_p = self.calibrator.predict_proba(logit_p)[:, 1]

        return np.clip(cal_p, 0.0, 1.0)

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Compute reliability diagram curve and Murphy (1973) decomposition.

        Args:
            y_true: Ground truth binary outcomes.
            y_prob: Predicted probabilities.
            n_bins: Number of probability bins (default: 10).

        Returns:
            Dict[str, Any]: Murphy decomposition containing reliability, resolution, uncertainty, Brier score.
        """
        return compute_reliability_diagram(y_true, y_prob, n_bins=n_bins, strategy="uniform")


def compare_calibrators(
    y_true: np.ndarray,
    y_prob_raw: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Fit and compare Raw vs Isotonic vs Platt scaling on calibration fold (docs/03 §7.2).

    Reports Brier score, Murphy reliability (lower is better), and resolution (higher is better).

    Args:
        y_true: Ground truth binary labels.
        y_prob_raw: Raw model predicted probabilities.
        n_bins: Number of bins for reliability diagram.

    Returns:
        Dict[str, Any]: Comparison dictionary with metrics per method and winning method.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.clip(np.asarray(y_prob_raw, dtype=float), 0.0, 1.0)

    # 1. Raw evaluation
    raw_eval = compute_reliability_diagram(y_t, y_p, n_bins=n_bins)

    # 2. Isotonic calibrator
    iso = ProbabilityCalibrator(method="isotonic").fit(y_t, y_p)
    p_iso = iso.predict(y_p)
    iso_eval = compute_reliability_diagram(y_t, p_iso, n_bins=n_bins)

    # 3. Platt calibrator
    platt = ProbabilityCalibrator(method="platt").fit(y_t, y_p)
    p_platt = platt.predict(y_p)
    platt_eval = compute_reliability_diagram(y_t, p_platt, n_bins=n_bins)

    results = {
        "raw": {
            "brier_score": float(raw_eval["brier_score"]),
            "reliability": float(raw_eval["reliability"]),
            "resolution": float(raw_eval["resolution"]),
            "uncertainty": float(raw_eval["uncertainty"]),
        },
        "isotonic": {
            "brier_score": float(iso_eval["brier_score"]),
            "reliability": float(iso_eval["reliability"]),
            "resolution": float(iso_eval["resolution"]),
            "uncertainty": float(iso_eval["uncertainty"]),
            "brier_delta_vs_raw": float(iso_eval["brier_score"] - raw_eval["brier_score"]),
        },
        "platt": {
            "brier_score": float(platt_eval["brier_score"]),
            "reliability": float(platt_eval["reliability"]),
            "resolution": float(platt_eval["resolution"]),
            "uncertainty": float(platt_eval["uncertainty"]),
            "brier_delta_vs_raw": float(platt_eval["brier_score"] - raw_eval["brier_score"]),
        },
    }

    # Best method by Brier score (lower is better)
    brier_scores = {
        "raw": results["raw"]["brier_score"],
        "isotonic": results["isotonic"]["brier_score"],
        "platt": results["platt"]["brier_score"],
    }
    winner = min(brier_scores, key=brier_scores.get)  # type: ignore
    results["winner"] = winner
    logger.info(
        "Calibration comparison: Raw BS=%.4f (rel=%.4f) | Isotonic BS=%.4f (rel=%.4f) | Platt BS=%.4f (rel=%.4f) -> Winner: %s",
        results["raw"]["brier_score"],
        results["raw"]["reliability"],
        results["isotonic"]["brier_score"],
        results["isotonic"]["reliability"],
        results["platt"]["brier_score"],
        results["platt"]["reliability"],
        winner.upper(),
    )
    return results
