"""Analogue Retrieval Engine v1: Euclidean k-NN in EOF Space (Nodes G1, G2, G6, docs/03 §6.3).

Retrieves historical meteorological analogues by Euclidean distance in 100-EOF space.
Enforces the mandatory temporal exclusion window: candidate analogues within +/- 15 calendar days
of the target query date MUST be masked out to prevent autocorrelation persistence artifacts.
Computes Gaussian kernel weights, top-8 historical analogues, bust probabilities, and OOD flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pratyaya.analogue.state import check_heldout_date_protection

logger = logging.getLogger(__name__)

# Mandatory exclusion window per docs/03 §6.3
TEMPORAL_EXCLUSION_DAYS: int = 15
DEFAULT_K_ANALOGUES: int = 8
DEFAULT_OOD_PERCENTILE: float = 99.0


@dataclass
class AnalogueMatch:
    """Individual historical analogue match and its attribution metadata."""
    candidate_date: pd.Timestamp
    distance: float
    weight: float
    rank: int
    candidate_idx: int


@dataclass
class RetrievalResult:
    """Result of an analogue retrieval query for an initialization cycle."""
    query_date: pd.Timestamp
    matches: List[AnalogueMatch]
    is_ood: bool
    d_k: float
    ood_threshold: Optional[float]

    @property
    def candidate_dates(self) -> List[pd.Timestamp]:
        """List of retrieved analogue dates ordered by proximity."""
        return [m.candidate_date for m in self.matches]

    @property
    def weights(self) -> List[float]:
        """List of normalized kernel weights ordered by proximity."""
        return [m.weight for m in self.matches]


class AnalogueRetriever:
    """Euclidean k-NN retrieval engine in 100-EOF space with temporal exclusion.

    Features:
    - Candidate pool populated strictly from training years (e.g. 2000–2001).
    - Mandatory +/- 15-day temporal exclusion window around target query date.
    - Gaussian kernel weighting: w_i = exp(-d_i^2 / (2 * sigma^2)), sigma = median(d).
    - Native OOD detection (Node G6): d_k > P99 triggers UNPRECEDENTED status.
    """

    def __init__(
        self,
        k_analogues: int = DEFAULT_K_ANALOGUES,
        exclusion_days: int = TEMPORAL_EXCLUSION_DAYS,
        ood_percentile: float = DEFAULT_OOD_PERCENTILE,
        metric: Optional[Any] = None,
    ) -> None:
        self.k_analogues = k_analogues
        self.exclusion_days = exclusion_days
        self.ood_percentile = ood_percentile
        self.metric = metric

        self.library_embeddings_: Optional[np.ndarray] = None  # (n_candidates, n_components)
        self.library_dates_: Optional[pd.DatetimeIndex] = None
        self.outcomes_df_: Optional[pd.DataFrame] = None
        self.ood_threshold_: Optional[float] = None
        self.is_fitted: bool = False
        self._labeled_date_set: set = set()
        self.last_query_unlabeled_fraction_: float = 0.0

    @property
    def labeled_date_set(self) -> set:
        if not hasattr(self, "_labeled_date_set") or self._labeled_date_set is None:
            if self.library_dates_ is not None:
                self._labeled_date_set = set(self.library_dates_.date)
            else:
                self._labeled_date_set = set()
        return self._labeled_date_set

    @property
    def last_query_unlabeled_fraction(self) -> float:
        return getattr(self, "last_query_unlabeled_fraction_", 0.0)

    def __getattr__(self, name: str) -> Any:
        if name == "last_query_unlabeled_fraction_":
            return 0.0
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def fit(
        self,
        embeddings: np.ndarray,
        dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
        outcomes_df: Optional[pd.DataFrame] = None,
        extra_library_embeddings: Optional[np.ndarray] = None,
        extra_library_dates: Optional[Sequence] = None,
    ) -> AnalogueRetriever:
        """Register the candidate analogue library strictly from training data.

        Args:
            embeddings: Array of shape (n_candidates, n_components) in EOF space.
            dates: Sequence of candidate initialization dates.
            outcomes_df: Optional canonical bust labels dataframe for outcome queries.
            extra_library_embeddings: Optional array of shape (n_extra, n_components) in EOF space.
            extra_library_dates: Optional sequence of extra candidate dates.

        Raises:
            PermissionError: If any date is in held-out years (2018, 2019).
            ValueError: If input dimensions do not match.
        """
        check_heldout_date_protection(dates)
        if extra_library_dates is not None:
            check_heldout_date_protection(extra_library_dates)

        self._labeled_date_set = set(pd.to_datetime(list(dates)).date)

        if extra_library_embeddings is not None:
            if extra_library_dates is None:
                raise ValueError("extra_library_dates must be provided if extra_library_embeddings is provided.")
            all_embs = np.vstack([np.asarray(embeddings, dtype=np.float64), np.asarray(extra_library_embeddings, dtype=np.float64)])
            all_dates = list(dates) + list(extra_library_dates)
        else:
            all_embs = np.asarray(embeddings, dtype=np.float64)
            all_dates = list(dates)

        dt_index = pd.to_datetime(list(all_dates))

        if len(dt_index) != len(all_embs):
            raise ValueError(
                f"Dates count ({len(dt_index)}) does not match embeddings rows ({len(all_embs)})."
            )

        if self.metric is not None:
            all_embs = self.metric.transform(all_embs)

        self.library_embeddings_ = all_embs
        self.library_dates_ = dt_index
        self.outcomes_df_ = outcomes_df

        # Calibrate native OOD threshold (Node G6) using internal leave-one-out distances
        self._calibrate_ood_threshold()

        self.is_fitted = True
        logger.info(
            "Fitted AnalogueRetriever with %d candidate dates (labeled: %d, extra: %d, years: %s), OOD P%.1f threshold: %.4f",
            len(dt_index),
            len(self.labeled_date_set),
            len(dt_index) - len(dates),
            sorted(dt_index.year.unique().tolist()),
            self.ood_percentile,
            self.ood_threshold_ if self.ood_threshold_ is not None else 0.0,
        )
        return self

    def _calibrate_ood_threshold(self) -> None:
        """Calibrate the 99th percentile d_k distance threshold on training fold."""
        if self.library_embeddings_ is None or len(self.library_embeddings_) < self.k_analogues + 1:
            self.ood_threshold_ = None
            return

        n_samples = len(self.library_embeddings_)
        d_k_list = []

        # Subsample for efficiency if pool is large, or compute exact for standard folds
        eval_indices = np.arange(n_samples)
        if n_samples > 500:
            np.random.seed(42)
            eval_indices = np.random.choice(n_samples, size=500, replace=False)

        for idx in eval_indices:
            q_emb = self.library_embeddings_[idx]
            q_date = self.library_dates_[idx]

            # Compute Euclidean distances
            diff = self.library_embeddings_ - q_emb
            dists = np.sqrt(np.sum(diff ** 2, axis=1))

            # Apply temporal exclusion (+/- exclusion_days)
            delta_days = np.abs((self.library_dates_ - q_date).days)
            valid_mask = delta_days > self.exclusion_days
            valid_dists = dists[valid_mask]

            if len(valid_dists) >= self.k_analogues:
                sorted_dists = np.sort(valid_dists)
                d_k_list.append(sorted_dists[self.k_analogues - 1])

        if d_k_list:
            self.ood_threshold_ = float(np.percentile(d_k_list, self.ood_percentile))
        else:
            self.ood_threshold_ = None

    def retrieve(
        self,
        query_embedding: np.ndarray,
        query_date: Union[date, datetime, pd.Timestamp, str],
        k: Optional[int] = None,
    ) -> RetrievalResult:
        """Retrieve top k analogues for a query date enforcing +/- 15-day exclusion.

        Args:
            query_embedding: 1D array of shape (n_components,) representing state in EOF space.
            query_date: Target initialization cycle timestamp.
            k: Number of analogues to return (defaults to self.k_analogues = 8).

        Returns:
            RetrievalResult: Retrieved analogue matches, kernel weights, and OOD status.
        """
        if not self.is_fitted or self.library_embeddings_ is None or self.library_dates_ is None:
            raise RuntimeError("AnalogueRetriever must be fitted before calling retrieve.")

        k_val = k if k is not None else self.k_analogues
        q_date = pd.Timestamp(query_date)
        q_vec = np.asarray(query_embedding, dtype=np.float64).reshape(1, -1)
        if self.metric is not None:
            q_vec = self.metric.transform(q_vec).reshape(1, -1)

        # 1. Compute Euclidean distances in EOF space
        diff = self.library_embeddings_ - q_vec
        dists = np.sqrt(np.sum(diff ** 2, axis=1))

        # 2. Enforce MANDATORY +/- 15-day temporal exclusion window
        # Any candidate date within [q_date - 15 days, q_date + 15 days] is masked out
        delta_days = np.abs((self.library_dates_ - q_date).days)
        valid_mask = (delta_days > self.exclusion_days)

        if not np.any(valid_mask):
            raise RuntimeError(
                f"No candidate analogues available for query date {q_date.strftime('%Y-%m-%d')} "
                f"after applying +/- {self.exclusion_days}-day temporal exclusion window."
            )

        valid_indices = np.where(valid_mask)[0]
        valid_dists = dists[valid_mask]

        effective_k = min(k_val, len(valid_dists))

        # 3. Sort by Euclidean distance to select top k
        top_k_sort_idx = np.argsort(valid_dists)[:effective_k]
        top_k_indices = valid_indices[top_k_sort_idx]
        top_k_dists = valid_dists[top_k_sort_idx]

        # 4. Compute Gaussian kernel weights: w_i ∝ exp(-d_i^2 / (2 * sigma^2))
        sigma = float(np.median(top_k_dists))
        if sigma <= 1e-6:
            sigma = 1.0  # Avoid zero-division if distances are identical

        kernel_logits = - (top_k_dists ** 2) / (2.0 * (sigma ** 2) + 1e-6)
        # Shift for numerical stability before exp
        exp_logits = np.exp(kernel_logits - np.max(kernel_logits))
        norm_weights = exp_logits / np.sum(exp_logits)

        # 5. Build match objects
        matches = []
        for rank, (idx, dist, weight) in enumerate(zip(top_k_indices, top_k_dists, norm_weights), start=1):
            cand_date = self.library_dates_[idx]
            # Safety assertion: candidate date MUST NEVER fall inside [q_date - 15, q_date + 15]
            assert abs((cand_date - q_date).days) > self.exclusion_days, (
                f"CRITICAL LEAKAGE: Candidate {cand_date} within {self.exclusion_days} days of query {q_date}!"
            )
            matches.append(
                AnalogueMatch(
                    candidate_date=cand_date,
                    distance=float(dist),
                    weight=float(weight),
                    rank=rank,
                    candidate_idx=int(idx),
                )
            )

        # 6. Native OOD detection (Node G6)
        d_k = float(top_k_dists[-1]) if len(top_k_dists) > 0 else 0.0
        is_ood = False
        if self.ood_threshold_ is not None and d_k > self.ood_threshold_:
            is_ood = True

        # Unlabeled neighbor fraction
        if len(matches) > 0:
            unlabeled_count = sum(1 for m in matches if m.candidate_date.date() not in self.labeled_date_set)
            self.last_query_unlabeled_fraction_ = float(unlabeled_count / len(matches))
        else:
            self.last_query_unlabeled_fraction_ = 0.0

        return RetrievalResult(
            query_date=q_date,
            matches=matches,
            is_ood=is_ood,
            d_k=d_k,
            ood_threshold=self.ood_threshold_,
        )

    def predict_bust_probabilities(
        self,
        query_embedding: np.ndarray,
        query_date: Union[date, datetime, pd.Timestamp, str],
        outcomes_df: Optional[pd.DataFrame] = None,
        k: Optional[int] = None,
    ) -> pd.DataFrame:
        """Compute weighted bust probabilities and error expectations per (region, lead).

        Per docs/03 §6.3:
        - p_analogue = sum(w_i * is_bust_i) / sum(w_i)
        - signed_expect = sum(w_i * error_signed_i) / sum(w_i)
        - analogue_dispersion = sqrt(sum(w_i * (error_signed_i - signed_expect)^2) / sum(w_i))

        For unlabeled extra library dates:
        - If analogue_date.date() in self.labeled_date_set: lookup probability from outcomes_df_
        - Else: assign base rate: float(outcomes_df_["is_bust"].mean())

        Returns:
            pd.DataFrame: Schema with [region_id, lead_day, p_analogue, signed_expect, analogue_dispersion, is_ood].
        """
        active_outcomes = outcomes_df if outcomes_df is not None else self.outcomes_df_
        if active_outcomes is None:
            raise ValueError("No outcomes dataframe provided to compute bust probabilities.")

        res = self.retrieve(query_embedding, query_date, k=k)
        if len(res.matches) == 0:
            return pd.DataFrame(columns=["region_id", "lead_day", "p_analogue", "signed_expect", "analogue_dispersion", "is_ood"])

        base_rate = float(active_outcomes["is_bust"].mean())

        # Map candidate dates to their weights
        match_map = {m.candidate_date: m.weight for m in res.matches}
        target_dates = list(match_map.keys())

        # Filter outcomes to matching analogue dates
        dt_col = pd.to_datetime(active_outcomes["init_date"])
        sub_df = active_outcomes[dt_col.isin(target_dates)].copy()
        sub_df["analogue_weight"] = sub_df["init_date"].apply(lambda d: match_map.get(pd.Timestamp(d), 0.0))

        # Weight of unlabeled analogues (matches not in labeled_date_set)
        unlabeled_weight = sum(
            m.weight for m in res.matches if m.candidate_date.date() not in self.labeled_date_set
        )

        region_lead_pairs = active_outcomes[["region_id", "lead_day"]].drop_duplicates().values
        results = []
        if len(sub_df) > 0:
            grouped = {k: grp for k, grp in sub_df.groupby(["region_id", "lead_day"])}
        else:
            grouped = {}

        for reg_id, lead in region_lead_pairs:
            grp = grouped.get((reg_id, lead))
            if grp is not None:
                w_lab = grp["analogue_weight"].values
                w_lab_sum = np.sum(w_lab)
                total_w = w_lab_sum + unlabeled_weight

                if total_w <= 0:
                    continue

                is_bust_lab = grp["is_bust"].astype(float).values
                bust_weighted_sum = np.sum(w_lab * is_bust_lab) + (unlabeled_weight * base_rate)
                p_analogue = float(bust_weighted_sum / total_w)

                err_lab = grp["error_signed"].values if "error_signed" in grp.columns else np.zeros_like(w_lab)
                err_weighted_sum = np.sum(w_lab * err_lab) + (unlabeled_weight * 0.0)
                signed_expect = float(err_weighted_sum / total_w)

                var_lab = np.sum(w_lab * (err_lab - signed_expect) ** 2)
                var_unlab = unlabeled_weight * (0.0 - signed_expect) ** 2
                dispersion = float(np.sqrt((var_lab + var_unlab) / total_w))
            else:
                p_analogue = base_rate
                signed_expect = 0.0
                dispersion = 0.0

            results.append({
                "region_id": str(reg_id),
                "lead_day": int(lead),
                "p_analogue": p_analogue,
                "signed_expect": signed_expect,
                "analogue_dispersion": dispersion,
                "is_ood": res.is_ood,
            })

        return pd.DataFrame(results)
