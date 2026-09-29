"""Baseline Models Ladder B0–B5 (Node F6).

Grain: (init_date, region_id, lead_day, variable) -> bust probability in [0.0, 1.0].

Implements the baseline ladder per docs/05-evaluation-protocol.md §2:
- B0: Constant base rate (~4%).
      The absolute floor. Predicts empirical training base rate everywhere.
- B1: Conditional climatology:
      Bust rate by (region_id, lead_day, month) from training data with hierarchical
      backoff: (region, lead, month) -> (region, month) -> (lead, month) -> global base rate.
- B2: Raw ensemble spread:
      The naive version of the operational standard answer. Uses ens_spread directly
      as a risk ranking score, normalized to [0, 1].
- B3: Calibrated logistic regression on ensemble spread:
      logit(p) = w0 + w1 * ens_spread (+ optional lead_day, forecast_value).
      The fair version of the field's standard answer. THIS IS THE REAL BAR THAT
      THE ANALOGUE RETRIEVAL BET MUST BEAT ON DAY 5.
- B4: EMOS-style post-processing (Gneiting et al. 2005) scaffold.
- B5: Tabular LightGBM on all atmospheric/forecast features scaffold.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from pratyaya.models.fusion import PUBLICATION_FUSION_FEATURES

logger = logging.getLogger(__name__)

DEFAULT_CANONICAL_FCST_DIR = Path("data/canonical/forecasts")


def ensure_features(
    df: pd.DataFrame,
    forecasts_dir: Path = DEFAULT_CANONICAL_FCST_DIR,
) -> pd.DataFrame:
    """Ensure DataFrame has required columns like 'ens_spread' by joining forecasts if needed.

    Args:
        df: Input DataFrame (e.g. canonical labels).
        forecasts_dir: Directory containing partitioned canonical forecasts.

    Returns:
        pd.DataFrame: DataFrame enriched with forecast columns if available.
    """
    df_out = df.copy()

    # If ens_spread is missing but forecasts_dir exists, attempt merge
    if "ens_spread" not in df_out.columns:
        if forecasts_dir.exists() and "init_date" in df_out.columns and "region_id" in df_out.columns:
            years = sorted(pd.to_datetime(df_out["init_date"]).dt.year.unique())
            fcst_dfs = []
            for y in years:
                year_dir = forecasts_dir / f"year={y}"
                if year_dir.exists():
                    for fpath in year_dir.glob("*.parquet"):
                        try:
                            fcst_dfs.append(pd.read_parquet(fpath))
                        except Exception:
                            pass
            if fcst_dfs:
                all_fcsts = pd.concat(fcst_dfs, ignore_index=True)
                cols_to_merge = ["init_date", "region_id", "lead_day", "variable", "ens_spread"]
                existing_merge_cols = [c for c in cols_to_merge if c in all_fcsts.columns]
                join_keys = [c for c in ["init_date", "region_id", "lead_day", "variable"] if c in df_out.columns and c in all_fcsts.columns]

                if "ens_spread" in existing_merge_cols and join_keys:
                    all_fcsts["init_date"] = pd.to_datetime(all_fcsts["init_date"])
                    df_out["init_date"] = pd.to_datetime(df_out["init_date"])
                    df_out["region_id"] = df_out["region_id"].astype(str)
                    all_fcsts["region_id"] = all_fcsts["region_id"].astype(str)

                    df_out = pd.merge(
                        df_out,
                        all_fcsts[join_keys + ["ens_spread"]].drop_duplicates(subset=join_keys),
                        on=join_keys,
                        how="left",
                    )
                    df_out["ens_spread"] = df_out["ens_spread"].fillna(0.0)
                    logger.info("Successfully merged 'ens_spread' from %s into evaluation DataFrame.", forecasts_dir)

        # Fallback: create ens_spread = 0.0 if not available
        if "ens_spread" not in df_out.columns:
            df_out["ens_spread"] = 0.0

    # Phase 2 NCMRWF-informed features (D-013)
    # Architectural stub disclosure: acc_trajectory_slope is zero-variance (0.0) across all rows
    # because real 3D circulation grids (Z500/U850/V850) were not ingested; reserved for future ingest.
    if "acc_trajectory_slope" not in df_out.columns:
        df_out["acc_trajectory_slope"] = 0.0

    if "delta_spread" not in df_out.columns:
        if "lead_day" in df_out.columns and "ens_spread" in df_out.columns:
            if "init_date" in df_out.columns and "region_id" in df_out.columns:
                diff_series = (
                    df_out.sort_values(["init_date", "region_id", "lead_day"])
                    .groupby(["init_date", "region_id"])["ens_spread"]
                    .diff()
                    .fillna(0.0)
                )
                df_out["delta_spread"] = diff_series.reindex(df_out.index).fillna(0.0)
            else:
                diff_series = (
                    df_out.sort_values("lead_day")["ens_spread"]
                    .diff()
                    .fillna(0.0)
                )
                df_out["delta_spread"] = diff_series.reindex(df_out.index).fillna(0.0)
        else:
            df_out["delta_spread"] = 0.0

    # Architectural stub disclosure: standardized_acc is zero-variance (0.0) across all rows;
    # reserved for future ingest of 3D circulation fields.
    if "standardized_acc" not in df_out.columns:
        df_out["standardized_acc"] = 0.0

    if "spread_error_ratio" not in df_out.columns:
        if "forecast_value" in df_out.columns and "ens_spread" in df_out.columns:
            denom = df_out["forecast_value"].abs() + 1.0
            ratio = df_out["ens_spread"] / denom
            df_out["spread_error_ratio"] = ratio.replace([np.inf, -np.inf], 1.0).fillna(1.0)
        else:
            df_out["spread_error_ratio"] = 1.0

    return df_out


class BaseBustModel(ABC):
    """Abstract base class for all bust prediction models and baselines."""

    @abstractmethod
    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "BaseBustModel":
        """Fit model parameters on training dataset."""
        pass

    @abstractmethod
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of bust for each record in df. Returns 1D array in [0.0, 1.0]."""
        pass

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Generate binary bust predictions using decision threshold."""
        probas = self.predict_proba(df)
        return (probas >= threshold).astype(int)


class ConstantBaseRateBaseline(BaseBustModel):
    """B0: Constant base rate baseline.

    Predicts a constant probability equal to the empirical training-period bust rate.
    Any model scoring below this is broken.
    """

    def __init__(self, default_rate: float = 0.043) -> None:
        self.default_rate = default_rate
        self.fitted_rate_: float = default_rate

    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "ConstantBaseRateBaseline":
        if df.empty or y_col not in df.columns:
            self.fitted_rate_ = self.default_rate
        else:
            self.fitted_rate_ = float(df[y_col].mean())
        logger.info("B0 (Constant Base Rate) fitted: rate = %.4f", self.fitted_rate_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        n = len(df)
        return np.full(n, self.fitted_rate_, dtype=float)


class ConditionalClimatologyBaseline(BaseBustModel):
    """B1: Conditional monthly and regional climatology baseline.

    Bust rate conditional on (region_id, lead_day, month) from training years.
    Applies hierarchical fallback:
    1. (region_id, lead_day, month)
    2. (region_id, month)
    3. (lead_day, month)
    4. Global base rate
    """

    def __init__(self, fallback_rate: float = 0.043) -> None:
        self.fallback_rate = fallback_rate
        self.lookup_rlm_: Dict[Tuple[str, int, int], float] = {}
        self.lookup_rm_: Dict[Tuple[str, int], float] = {}
        self.lookup_lm_: Dict[Tuple[int, int], float] = {}
        self.global_rate_: float = fallback_rate

    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "ConditionalClimatologyBaseline":
        if df.empty or y_col not in df.columns:
            return self

        df_work = df.copy()
        df_work["_dt"] = pd.to_datetime(df_work["init_date"])
        df_work["_m"] = df_work["_dt"].dt.month
        df_work["_r"] = df_work["region_id"].astype(str)
        df_work["_l"] = df_work["lead_day"].astype(int)
        y = df_work[y_col].astype(float)

        self.global_rate_ = float(y.mean())

        # 1. (region, lead, month)
        rlm_grp = df_work.groupby(["_r", "_l", "_m"])[y_col].mean()
        self.lookup_rlm_ = {
            (str(k[0]), int(k[1]), int(k[2])): float(v)
            for k, v in rlm_grp.items()
        }

        # 2. (region, month)
        rm_grp = df_work.groupby(["_r", "_m"])[y_col].mean()
        self.lookup_rm_ = {
            (str(k[0]), int(k[1])): float(v)
            for k, v in rm_grp.items()
        }

        # 3. (lead, month)
        lm_grp = df_work.groupby(["_l", "_m"])[y_col].mean()
        self.lookup_lm_ = {
            (int(k[0]), int(k[1])): float(v)
            for k, v in lm_grp.items()
        }

        logger.info(
            "B1 (Conditional Climatology) fitted on %d records: %d (r,l,m) cells, %d (r,m) cells.",
            len(df),
            len(self.lookup_rlm_),
            len(self.lookup_rm_),
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=float)

        df_work = df.copy()
        df_work["_dt"] = pd.to_datetime(df_work["init_date"])
        months = df_work["_dt"].dt.month.values
        regions = df_work["region_id"].astype(str).values
        leads = df_work["lead_day"].astype(int).values

        n = len(df)
        probas = np.empty(n, dtype=float)

        for i in range(n):
            r = regions[i]
            l = leads[i]
            m = months[i]

            key3 = (r, l, m)
            if key3 in self.lookup_rlm_:
                probas[i] = self.lookup_rlm_[key3]
            elif (r, m) in self.lookup_rm_:
                probas[i] = self.lookup_rm_[(r, m)]
            elif (l, m) in self.lookup_lm_:
                probas[i] = self.lookup_lm_[(l, m)]
            else:
                probas[i] = self.global_rate_

        return np.clip(probas, 0.0, 1.0)


class RawEnsembleSpreadBaseline(BaseBustModel):
    """B2: Raw ensemble spread ranking baseline.

    Uses raw ensemble standard deviation ('ens_spread') directly as a risk score.
    Applies min-max normalization based on training dispersion.
    """

    def __init__(self, spread_col: str = "ens_spread") -> None:
        self.spread_col = spread_col
        self.min_spread_: float = 0.0
        self.max_spread_: float = 1.0

    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "RawEnsembleSpreadBaseline":
        df_work = ensure_features(df)
        s = df_work[self.spread_col].astype(float)
        self.min_spread_ = float(s.min()) if len(s) > 0 else 0.0
        self.max_spread_ = float(s.max()) if len(s) > 0 else 1.0
        if self.max_spread_ <= self.min_spread_:
            self.max_spread_ = self.min_spread_ + 1.0
        logger.info("B2 (Raw Spread) fitted: spread range [%.2f, %.2f]", self.min_spread_, self.max_spread_)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        df_work = ensure_features(df)
        s = df_work[self.spread_col].astype(float).values
        denom = self.max_spread_ - self.min_spread_
        norm_s = (s - self.min_spread_) / denom
        return np.clip(norm_s, 0.0, 1.0)


class CalibratedLogisticSpreadBaseline(BaseBustModel):
    """B3: Calibrated logistic regression on ensemble spread.

    logit(p) = w0 + w1 * ens_spread (+ optional lead_day, forecast_value)

    Per docs/05 §2:
    The fair version of the field's standard answer. THIS IS THE REAL BENCHMARK BAR.
    Beating B2 and losing to B3 is a common failure mode.
    """

    def __init__(
        self,
        features: Sequence[str] = ("ens_spread",),
        spread_col: str = "ens_spread",
        c_reg: float = 1.0,
        class_weight: Optional[Any] = None,
    ) -> None:
        self.features = list(features)
        self.spread_col = spread_col
        self.c_reg = c_reg
        self.class_weight = class_weight
        self.model_: Optional[LogisticRegression] = None
        self.fallback_rate_: float = 0.043

    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "CalibratedLogisticSpreadBaseline":
        df_work = ensure_features(df)
        y = df_work[y_col].astype(int).values
        self.fallback_rate_ = float(y.mean()) if len(y) > 0 else 0.043

        # Prepare feature matrix
        x_cols = [c for c in self.features if c in df_work.columns]
        if not x_cols:
            x_cols = [self.spread_col]

        X = df_work[x_cols].astype(float).values

        # Check if spread feature has variance and positive examples exist
        if len(np.unique(y)) < 2 or np.all(np.std(X, axis=0) < 1e-9):
            logger.warning("Insufficient feature variance or single class; B3 falling back to intercept-only.")
            self.model_ = None
            return self

        clf = LogisticRegression(
            C=self.c_reg,
            class_weight=self.class_weight,
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(X, y)
        self.model_ = clf
        self.features_used_ = x_cols
        logger.info(
            "B3 (Calibrated Logistic Spread) fitted: features = %s, intercept = %.4f, coef = %s",
            x_cols,
            float(clf.intercept_[0]),
            clf.coef_[0].tolist(),
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=float)

        if self.model_ is None:
            return np.full(len(df), self.fallback_rate_, dtype=float)

        df_work = ensure_features(df)
        X = df_work[self.features_used_].astype(float).values
        probas = self.model_.predict_proba(X)[:, 1]
        return probas


class EMOSBaseline(BaseBustModel):
    """B4: Ensemble Model Output Statistics (EMOS) baseline (Gneiting et al. 2005).

    Estimates predictive distribution parameters:
        mu = a + b * forecast_value
        sigma^2 = max(1e-4, c + d * (ens_spread ** 2))
    Fitted via minimum Continuous Ranked Probability Score (CRPS) against observed rainfall
    (or Log-Loss / NLL against bust labels).
    Bust probability is evaluated as tail exceedance over the empirical 95th percentile threshold:
        P(bust) = 1 - Phi((threshold - mu) / sigma) = Phi((mu - threshold) / sigma)
    """

    def __init__(
        self,
        default_threshold: float = 15.0,
        default_fallback_std: float = 5.0,
    ) -> None:
        self.default_threshold = default_threshold
        self.default_fallback_std = default_fallback_std
        self.a_: float = 0.0
        self.b_: float = 1.0
        self.c_: float = 0.1
        self.d_: float = 1.0
        self.thresholds_map_: Dict[Tuple[str, int], float] = {}
        self.global_threshold_: float = default_threshold
        self.regional_std_map_: Dict[Tuple[str, int], float] = {}
        self.global_fallback_std_: float = default_fallback_std
        self.is_fitted: bool = False

    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "EMOSBaseline":
        if df.empty:
            return self

        df_work = ensure_features(df)
        f_val = df_work["forecast_value"].astype(float).values if "forecast_value" in df_work.columns else np.zeros(len(df_work))
        raw_spread = df_work["ens_spread"].astype(float).values if "ens_spread" in df_work.columns else np.zeros(len(df_work))

        # Compute regional error std fallback and 95th percentile bust threshold
        if "error_signed" in df_work.columns:
            abs_err = df_work["error_signed"].abs().values
            self.global_threshold_ = float(np.percentile(abs_err, 95)) if len(abs_err) > 0 else self.default_threshold
            if self.global_threshold_ < 1e-3:
                self.global_threshold_ = self.default_threshold

            std_err = float(df_work["error_signed"].std())
            self.global_fallback_std_ = std_err if (not np.isnan(std_err) and std_err > 1e-4) else self.default_fallback_std

            # Per (region_id, lead_day) lookups
            if "region_id" in df_work.columns and "lead_day" in df_work.columns:
                grp_thresh = df_work.groupby(["region_id", "lead_day"])["error_signed"].apply(
                    lambda s: float(np.percentile(np.abs(s), 95)) if len(s) > 0 else self.global_threshold_
                )
                self.thresholds_map_ = {
                    (str(k[0]), int(k[1])): float(v) for k, v in grp_thresh.items()
                }

                grp_std = df_work.groupby(["region_id", "lead_day"])["error_signed"].std()
                self.regional_std_map_ = {
                    (str(k[0]), int(k[1])): float(v) if (not np.isnan(v) and v > 1e-4) else self.global_fallback_std_
                    for k, v in grp_std.items()
                }
        else:
            self.global_threshold_ = self.default_threshold
            self.global_fallback_std_ = self.default_fallback_std

        # Build spread array with historical regional error std fallback for zero or missing spread
        spread = np.zeros(len(df_work), dtype=float)
        reg_ids = df_work["region_id"].astype(str).values if "region_id" in df_work.columns else ["1"] * len(df_work)
        lead_days = df_work["lead_day"].astype(int).values if "lead_day" in df_work.columns else [1] * len(df_work)

        for i in range(len(df_work)):
            s_val = raw_spread[i]
            if np.isnan(s_val) or s_val <= 1e-4:
                spread[i] = self.regional_std_map_.get((reg_ids[i], lead_days[i]), self.global_fallback_std_)
            else:
                spread[i] = s_val

        # Target for CRPS vs NLL
        y_obs = None
        if "observed_value" in df_work.columns:
            y_obs = df_work["observed_value"].astype(float).values
        elif "error_signed" in df_work.columns and "forecast_value" in df_work.columns:
            y_obs = (df_work["forecast_value"] - df_work["error_signed"]).astype(float).values

        if y_obs is not None and len(y_obs) > 0 and not np.all(np.isnan(y_obs)):
            def crps_objective(params: Sequence[float]) -> float:
                a, b, c, d = params
                mu = a + b * f_val
                s2 = np.maximum(1e-4, c + d * (spread ** 2))
                sigma = np.sqrt(s2)
                z = (y_obs - mu) / sigma
                # Closed-form Gaussian CRPS: sigma * [z * (2*Phi(z) - 1) + 2*phi(z) - 1/sqrt(pi)]
                crps = sigma * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - (1.0 / np.sqrt(np.pi)))
                return float(np.mean(crps))

            initial_guess = [0.0, 1.0, 0.1, 1.0]
            bounds = [(None, None), (None, None), (1e-4, None), (0.0, None)]
            try:
                res = minimize(crps_objective, initial_guess, method="L-BFGS-B", bounds=bounds)
                if res.success or res.x is not None:
                    self.a_, self.b_, self.c_, self.d_ = float(res.x[0]), float(res.x[1]), max(1e-4, float(res.x[2])), max(0.0, float(res.x[3]))
            except Exception as exc:
                logger.warning("EMOS CRPS optimization encountered error: %s; keeping initial parameters.", exc)
        elif y_col in df_work.columns:
            y_labels = df_work[y_col].astype(float).values
            thresh_arr = np.array([
                self.thresholds_map_.get((reg_ids[i], lead_days[i]), self.global_threshold_)
                for i in range(len(df_work))
            ])

            def nll_objective(params: Sequence[float]) -> float:
                a, b, c, d = params
                mu = a + b * f_val
                s2 = np.maximum(1e-4, c + d * (spread ** 2))
                sigma = np.sqrt(s2)
                p = np.clip(1.0 - norm.cdf((thresh_arr - mu) / sigma), 1e-7, 1.0 - 1e-7)
                nll = -np.mean(y_labels * np.log(p) + (1.0 - y_labels) * np.log(1.0 - p))
                return float(nll)

            initial_guess = [0.0, 1.0, 0.1, 1.0]
            bounds = [(None, None), (None, None), (1e-4, None), (0.0, None)]
            try:
                res = minimize(nll_objective, initial_guess, method="L-BFGS-B", bounds=bounds)
                if res.success or res.x is not None:
                    self.a_, self.b_, self.c_, self.d_ = float(res.x[0]), float(res.x[1]), max(1e-4, float(res.x[2])), max(0.0, float(res.x[3]))
            except Exception as exc:
                logger.warning("EMOS NLL optimization encountered error: %s; keeping initial parameters.", exc)

        self.is_fitted = True
        logger.info(
            "B4 (EMOS) fitted: a=%.4f, b=%.4f, c=%.4f, d=%.4f (global threshold=%.2f, %d cells)",
            self.a_, self.b_, self.c_, self.d_, self.global_threshold_, len(self.thresholds_map_)
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=float)

        df_work = ensure_features(df)
        f_val = df_work["forecast_value"].astype(float).values if "forecast_value" in df_work.columns else np.zeros(len(df_work))
        raw_spread = df_work["ens_spread"].astype(float).values if "ens_spread" in df_work.columns else np.zeros(len(df_work))

        reg_ids = df_work["region_id"].astype(str).values if "region_id" in df_work.columns else ["1"] * len(df_work)
        lead_days = df_work["lead_day"].astype(int).values if "lead_day" in df_work.columns else [1] * len(df_work)

        spread = np.zeros(len(df_work), dtype=float)
        thresh_arr = np.zeros(len(df_work), dtype=float)

        for i in range(len(df_work)):
            s_val = raw_spread[i]
            key = (reg_ids[i], lead_days[i])
            if np.isnan(s_val) or s_val <= 1e-4:
                spread[i] = self.regional_std_map_.get(key, self.global_fallback_std_)
            else:
                spread[i] = s_val
            thresh_arr[i] = self.thresholds_map_.get(key, self.global_threshold_)

        mu = self.a_ + self.b_ * f_val
        s2 = np.maximum(1e-4, self.c_ + self.d_ * (spread ** 2))
        sigma = np.sqrt(s2)

        # Tail exceedance probability: 1 - Phi((threshold - mu) / sigma) = Phi((mu - threshold) / sigma)
        probas = 1.0 - norm.cdf((thresh_arr - mu) / sigma)
        return np.clip(probas, 0.0, 1.0)


class TabularLightGBMBaseline(BaseBustModel):
    """B5: Tabular LightGBM baseline on atmospheric/NWP features.

    Full GBDT baseline trained on tabular NWP features (excluding analogue-derived features)
    matching AnalogueNWPFusionModel hyperparameters.
    """

    def __init__(
        self,
        features: Optional[List[str]] = None,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        scale_pos_weight: Optional[float] = 1.0,
        random_state: int = 42,
        **model_params: Any,
    ) -> None:
        self.features = features if features is not None else [
            f for f in PUBLICATION_FUSION_FEATURES
            if f not in {"p_analogue", "d_k", "signed_expect", "analogue_dispersion"}
        ]
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_samples = min_child_samples
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.model_params = model_params

        self.model: Optional[Any] = None
        self.features_used_: List[str] = []
        self.fallback_rate_: float = 0.043
        self.is_fitted: bool = False

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        df_work = ensure_features(df)
        avail_features = [f for f in self.features if f in df_work.columns]
        if not avail_features:
            avail_features = [c for c in ["ens_spread", "forecast_value", "lead_day"] if c in df_work.columns]

        X_cols = []
        for col in avail_features:
            if col == "region_id":
                s = pd.to_numeric(df_work[col], errors="coerce").fillna(0).astype(float)
                X_cols.append(s.values)
            else:
                X_cols.append(pd.to_numeric(df_work[col], errors="coerce").fillna(0.0).values)

        X = np.column_stack(X_cols) if X_cols else np.zeros((len(df_work), 1), dtype=float)
        return np.asarray(X, dtype=np.float64), avail_features

    def fit(self, df: pd.DataFrame, y_col: str = "is_bust") -> "TabularLightGBMBaseline":
        if df.empty or y_col not in df.columns:
            return self

        try:
            import lightgbm as lgb
        except ImportError as err:
            raise ImportError("lightgbm is required for TabularLightGBMBaseline. Install lightgbm.") from err

        X, used_feats = self._prepare_features(df)
        y = df[y_col].astype(int).values
        self.fallback_rate_ = float(np.mean(y)) if len(y) > 0 else 0.043
        self.features_used_ = used_feats

        pos_count = int(np.sum(y == 1))
        neg_count = int(np.sum(y == 0))
        actual_scale_pos = self.scale_pos_weight if self.scale_pos_weight is not None else (neg_count / max(1, pos_count))

        n_samples = len(y)
        # Adapt min_child_samples and min_data_in_bin for small datasets (e.g. test fixtures)
        actual_min_child = min(self.min_child_samples, max(1, n_samples // 2))
        actual_min_bin = min(20, max(1, n_samples // 2))

        params: Dict[str, Any] = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": 15,
            "max_depth": self.max_depth,
            "min_child_samples": actual_min_child,
            "min_data_in_bin": actual_min_bin,
            "scale_pos_weight": actual_scale_pos,
            "random_state": self.random_state,
            "verbose": -1,
        }
        params.update(self.model_params)

        if len(np.unique(y)) < 2:
            logger.warning("Single class target in TabularLightGBMBaseline; falling back to constant.")
            self.model = None
            self.is_fitted = True
            return self

        clf = lgb.LGBMClassifier(**params)
        clf.fit(X, y)
        self.model = clf
        self.is_fitted = True
        logger.info(
            "B5 (Tabular LightGBM) fitted on %d samples with %d features: %s",
            n_samples, len(used_feats), used_feats
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        if df.empty:
            return np.array([], dtype=float)

        if self.model is None:
            return np.full(len(df), self.fallback_rate_, dtype=float)

        X, _ = self._prepare_features(df)
        probas = self.model.predict_proba(X)[:, 1]
        return np.clip(probas, 0.0, 1.0)
