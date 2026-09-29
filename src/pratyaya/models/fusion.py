"""Analogue-NWP Fusion Model (Node G3, docs/03 §7.1, graph.json F10, G3).

Combines physical NWP forecast signals (ensemble spread, forecast value, lead day) with
data-driven analogue predictability signals (p_analogue, d_k, signed_expect, analogue_dispersion).

Supports both LightGBM (LGBMClassifier) and Logistic Regression backends.
Enforces strict zero-leakage training using out-of-fold analogue features generated with
the mandatory +/- 15-day temporal exclusion window.
Provides orthogonality diagnostics demonstrating non-redundancy between analogue and ensemble spread signals.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Primary features per docs/03 §7.1 and PRD
DEFAULT_FUSION_FEATURES: List[str] = [
    "ens_spread",
    "forecast_value",
    "lead_day",
    "p_analogue",
    "d_k",
    "signed_expect",
    "analogue_dispersion",
]

EXTENDED_FUSION_FEATURES: List[str] = [
    "ens_spread",
    "forecast_value",
    "lead_day",
    "p_analogue",
    "d_k",
    "signed_expect",
    "analogue_dispersion",
    "month_sin",
    "month_cos",
]

# Publication fusion features per NCMRWF guidance D-013.
# NOTE & DISCLOSURE (Decision D-013):
# - 'delta_spread' and 'spread_error_ratio' are live features derived from ensemble spread.
# - 'acc_trajectory_slope' and 'standardized_acc' are zero-variance architectural stubs
#   (defaulting to 0.0) reserved for future operational ingestion of 3D pressure-level
#   circulation fields (Z500, U850, V850). In the current build, they provide zero signal
#   and do not contribute to model skill in any benchmark.
PUBLICATION_FUSION_FEATURES: List[str] = [
    "ens_spread",
    "forecast_value",
    "lead_day",
    "p_analogue",
    "d_k",
    "signed_expect",
    "analogue_dispersion",
    "acc_trajectory_slope",
    "delta_spread",
    "standardized_acc",
    "spread_error_ratio",
]



def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add month cyclic sine and cosine features to DataFrame if init_date is present."""
    out = df.copy()
    if "init_date" in out.columns:
        dt = pd.to_datetime(out["init_date"])
        month = dt.dt.month
        out["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        out["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    return out


def generate_out_of_fold_analogue_features(
    train_df: pd.DataFrame,
    cycle_embeddings: np.ndarray,
    cycle_dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
    retriever_factory: Any,
    k_analogues: int = 8,
) -> pd.DataFrame:
    """Generate out-of-fold analogue features strictly on training fold without leakage.

    For each cycle t in cycle_dates, retrieves analogues from the remaining candidate library
    enforcing the mandatory +/- 15-day temporal exclusion window.

    Args:
        train_df: Training fold canonical records.
        cycle_embeddings: Array of shape (n_cycles, n_features) (EOF or Mahalanobis).
        cycle_dates: List of training cycle dates.
        retriever_factory: Callable or class to instantiate AnalogueRetriever.
        k_analogues: Number of nearest analogues to retrieve.

    Returns:
        pd.DataFrame: train_df merged with out-of-fold ['p_analogue', 'd_k', 'signed_expect', 'analogue_dispersion', 'is_ood'].
    """
    dt_cycles = [pd.Timestamp(str(d)) for d in cycle_dates]
    n_cycles = len(dt_cycles)

    # Instantiate retriever fitted on all training cycles
    retriever = retriever_factory(k_analogues=k_analogues)
    retriever.fit(cycle_embeddings, dt_cycles, outcomes_df=train_df)

    oof_predictions = []
    for i, q_date in enumerate(dt_cycles):
        q_emb = cycle_embeddings[i]
        # retrieve() strictly excludes any candidate within +/- 15 days of q_date
        res = retriever.retrieve(q_emb, q_date, k=k_analogues)
        pred_sub = retriever.predict_bust_probabilities(q_emb, q_date, k=k_analogues)
        pred_sub["init_date_dt"] = q_date
        pred_sub["d_k"] = res.d_k
        oof_predictions.append(pred_sub)

    oof_df = pd.concat(oof_predictions, ignore_index=True)

    df_out = train_df.copy()
    df_out["init_date_dt"] = pd.to_datetime(df_out["init_date"])
    df_out["region_id_str"] = df_out["region_id"].astype(str)
    oof_df["region_id_str"] = oof_df["region_id"].astype(str)

    merged = pd.merge(
        df_out,
        oof_df,
        on=["init_date_dt", "region_id_str", "lead_day"],
        how="inner",
        suffixes=("", "_dup"),
    )
    # Clean up duplicate columns if any
    cols_to_drop = [c for c in merged.columns if c.endswith("_dup")]
    if cols_to_drop:
        merged = merged.drop(columns=cols_to_drop)

    logger.info("Generated out-of-fold analogue features: %d rows merged", len(merged))
    return merged


def compute_orthogonality_metrics(
    df: pd.DataFrame,
    analogue_col: str = "p_analogue",
    spread_col: str = "ens_spread",
    b3_prob_col: str = "prob_b3",
) -> Dict[str, Any]:
    """Demonstrate orthogonality: analogue predictions are weakly correlated with spread signals (Node G3).

    Computes Pearson and Spearman correlation between analogue bust probability and ensemble spread.

    Args:
        df: DataFrame containing predictions and features.
        analogue_col: Column name for analogue probability.
        spread_col: Column name for ensemble spread.
        b3_prob_col: Column name for Baseline B3 probability if present.

    Returns:
        Dict[str, Any]: Correlation coefficients and p-values.
    """
    valid = df.dropna(subset=[analogue_col, spread_col])
    if len(valid) == 0:
        return {"pearson_r": 0.0, "spearman_rho": 0.0, "n_samples": 0}

    std_a = np.std(valid[analogue_col].values)
    std_s = np.std(valid[spread_col].values)

    if std_a < 1e-8 or std_s < 1e-8:
        p_val, p_p = 0.0, 1.0
        s_val, s_p = 0.0, 1.0
    else:
        p_val, p_p = pearsonr(valid[analogue_col], valid[spread_col])
        s_val, s_p = spearmanr(valid[analogue_col], valid[spread_col])

    result: Dict[str, Any] = {
        "n_samples": len(valid),
        "analogue_vs_spread_pearson_r": float(p_val),
        "analogue_vs_spread_pearson_p": float(p_p),
        "analogue_vs_spread_spearman_rho": float(s_val),
        "analogue_vs_spread_spearman_p": float(s_p),
    }

    if b3_prob_col in valid.columns:
        std_b3 = np.std(valid[b3_prob_col].values)
        if std_a < 1e-8 or std_b3 < 1e-8:
            b3_p_val, b3_s_val = 0.0, 0.0
        else:
            b3_p_val, _ = pearsonr(valid[analogue_col], valid[b3_prob_col])
            b3_s_val, _ = spearmanr(valid[analogue_col], valid[b3_prob_col])
        result.update({
            "analogue_vs_b3_pearson_r": float(b3_p_val),
            "analogue_vs_b3_spearman_rho": float(b3_s_val),
        })

    logger.info(
        "Orthogonality check (N=%d): Pearson r(analogue, spread) = %.4f (p=%.2e), Spearman rho = %.4f",
        len(valid),
        result["analogue_vs_spread_pearson_r"],
        result["analogue_vs_spread_pearson_p"],
        result["analogue_vs_spread_spearman_rho"],
    )
    return result


class AnalogueNWPFusionModel:
    """Analogue-NWP Fusion Model (Node G3, docs/03 §7.1).

    Fuses physical forecast signals with analogue error predictability signals.
    Supports LightGBM and Calibrated Logistic Regression backends.
    """

    def __init__(
        self,
        backend: str = "lightgbm",
        features: Optional[List[str]] = None,
        random_state: int = 42,
        **model_params: Any,
    ) -> None:
        self.backend = backend.lower()
        if self.backend not in ["lightgbm", "logistic"]:
            raise ValueError(f"Unknown backend '{backend}'. Supported: 'lightgbm', 'logistic'.")

        self.features = features if features is not None else list(DEFAULT_FUSION_FEATURES)
        self.random_state = random_state
        self.model_params = model_params

        self.model: Optional[Any] = None
        self.scaler: Optional[StandardScaler] = None
        self.is_fitted: bool = False

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and clean feature matrix."""
        df_feat = add_calendar_features(df)
        missing = [f for f in self.features if f not in df_feat.columns]
        if missing:
            raise ValueError(f"DataFrame is missing required fusion features: {missing}")

        X = df_feat[self.features].copy()
        # Fill any missing values with 0
        X = X.fillna(0.0).values
        return np.asarray(X, dtype=np.float64)

    def fit(
        self,
        train_df: pd.DataFrame,
        y_col: str = "is_bust",
    ) -> AnalogueNWPFusionModel:
        """Train the fusion model on out-of-fold training records.

        Args:
            train_df: Training DataFrame containing features and target column.
            y_col: Binary target label column (default: 'is_bust').

        Returns:
            AnalogueNWPFusionModel: Self instance.
        """
        X = self._prepare_features(train_df)
        y = train_df[y_col].astype(int).values

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        scale_pos = (neg_count / max(1, pos_count))

        if self.backend == "lightgbm":
            try:
                import lightgbm as lgb
            except ImportError as err:
                raise ImportError("lightgbm is required for backend='lightgbm'. Install lightgbm.") from err

            params: Dict[str, Any] = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "n_estimators": 100,
                "learning_rate": 0.05,
                "num_leaves": 15,
                "max_depth": 4,
                "min_child_samples": 20,
                "scale_pos_weight": scale_pos,
                "random_state": self.random_state,
                "verbose": -1,
            }
            params.update(self.model_params)
            self.model = lgb.LGBMClassifier(**params)
            self.model.fit(X, y)

        elif self.backend == "logistic":
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)

            params = {
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": self.random_state,
                "solver": "lbfgs",
            }
            params.update(self.model_params)
            self.model = LogisticRegression(**params)
            self.model.fit(X_scaled, y)

        self.is_fitted = True
        logger.info(
            "Fitted AnalogueNWPFusionModel (backend=%s, N=%d, positives=%d, scale_pos_weight=%.2f)",
            self.backend,
            len(y),
            pos_count,
            scale_pos,
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict calibrated bust probability in [0.0, 1.0]."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predict_proba.")

        X = self._prepare_features(df)
        if self.backend == "logistic":
            assert self.scaler is not None
            X = self.scaler.transform(X)

        proba = self.model.predict_proba(X)[:, 1]
        return np.clip(proba, 0.0, 1.0)

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary bust classification."""
        prob = self.predict_proba(df)
        return (prob >= threshold).astype(int)

    @property
    def feature_importances(self) -> Dict[str, float]:
        """Return feature importance dictionary."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model must be fitted before feature_importances.")

        if self.backend == "lightgbm":
            raw_imp = self.model.feature_importances_
            total = float(np.sum(raw_imp)) if np.sum(raw_imp) > 0 else 1.0
            return {f: float(imp / total) for f, imp in zip(self.features, raw_imp)}
        elif self.backend == "logistic":
            raw_coef = np.abs(self.model.coef_[0])
            total = float(np.sum(raw_coef)) if np.sum(raw_coef) > 0 else 1.0
            return {f: float(c / total) for f, c in zip(self.features, raw_coef)}
        return {}
