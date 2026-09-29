"""Linear Low-Rank Mahalanobis Metric Learning (Node G2, docs/03 §6.2).

Learns a low-rank Mahalanobis metric M = W^T W, W in R^(32 x 100), minimizing a margin-based
pairwise triplet ranking loss over atmospheric states in 100-EOF space.

Triplets (anchor a, positive p, negative n) are sampled such that the outcome distance in signed error
space satisfies ||y(a) - y(p)|| < ||y(a) - y(n)||, subject to the mandatory temporal exclusion constraint:
anchor a and positive p must be separated by >= 15 calendar days to prevent autocorrelation leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from pratyaya.analogue.retrieve import TEMPORAL_EXCLUSION_DAYS
from pratyaya.analogue.state import check_heldout_date_protection

logger = logging.getLogger(__name__)

DEFAULT_METRIC_COMPONENTS: int = 32
DEFAULT_EOF_DIM: int = 100
DEFAULT_MARGIN: float = 1.0
DEFAULT_LR: float = 0.005
DEFAULT_N_EPOCHS: int = 100
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_L2_REG: float = 1e-4
DEFAULT_N_TRIPLETS: int = 3000


@dataclass
class Triplet:
    """Sampled triplet index representation."""
    anchor_idx: int
    positive_idx: int
    negative_idx: int
    outcome_dist_pos: float
    outcome_dist_neg: float


def build_outcome_vectors(
    outcomes_df: pd.DataFrame,
    dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
    n_leads: int = 10,
    standardize: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Construct standardized outcome vectors y(t) in R^(36 x n_leads) for training dates.

    Each outcome vector captures the spatial and lead-time distribution of signed forecast errors:
        e(t, r, L) = forecast_value - observed_value (or from 'error_signed' column).
    Standardized per (region, lead) across training dates:
        y_std(t, r, L) = (e(t, r, L) - mu(r, L)) / (sigma(r, L) + eps).

    Args:
        outcomes_df: Canonical labels/forecasts DataFrame containing ['init_date', 'region_id', 'lead_day'].
        dates: Sequence of training initialization dates.
        n_leads: Number of lead days (default: 10, leads 1..10).
        standardize: Whether to standardize columns by mean and standard deviation.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]:
            - Outcome vectors array of shape (len(dates), 36 * n_leads).
            - Metadata dictionary with region list, lead list, means, and stds.

    Raises:
        PermissionError: If any date is in held-out years (2018, 2019).
        ValueError: If dates sequence is empty or required columns missing.
    """
    check_heldout_date_protection(dates)
    dt_index = pd.to_datetime(list(dates))
    if len(dt_index) == 0:
        raise ValueError("Cannot build outcome vectors from empty dates sequence.")

    df = outcomes_df.copy()
    if "init_date" not in df.columns or "region_id" not in df.columns or "lead_day" not in df.columns:
        raise ValueError("outcomes_df must contain ['init_date', 'region_id', 'lead_day'] columns.")

    df["init_date_dt"] = pd.to_datetime(df["init_date"])
    df["region_id_str"] = df["region_id"].astype(str)

    # Determine signed error column
    if "error_signed" in df.columns:
        err_col = "error_signed"
    elif "forecast_value" in df.columns and "observation_value" in df.columns:
        df["error_signed"] = df["forecast_value"] - df["observation_value"]
        err_col = "error_signed"
    else:
        raise ValueError("outcomes_df must have 'error_signed' or both 'forecast_value' and 'observation_value'.")

    unique_regions = sorted(df["region_id_str"].unique().tolist())
    if len(unique_regions) == 0:
        unique_regions = [str(r) for r in range(1, 37)]
    unique_leads = list(range(1, n_leads + 1))

    n_regions = len(unique_regions)
    dim = n_regions * n_leads

    # Index mapping
    feat_index = {}
    idx = 0
    for r in unique_regions:
        for lead in unique_leads:
            feat_index[(r, lead)] = idx
            idx += 1

    outcome_matrix = np.zeros((len(dt_index), dim), dtype=np.float64)

    # Filter dataframe to target dates
    sub_df = df[df["init_date_dt"].isin(dt_index)].copy()
    grouped = sub_df.groupby(["init_date_dt", "region_id_str", "lead_day"])[err_col].mean()

    for date_idx, dt_val in enumerate(dt_index):
        for (r, lead), col_idx in feat_index.items():
            key = (dt_val, r, lead)
            if key in grouped.index:
                outcome_matrix[date_idx, col_idx] = float(grouped.loc[key])

    means = np.mean(outcome_matrix, axis=0)
    stds = np.std(outcome_matrix, axis=0)
    stds[stds < 1e-6] = 1.0  # Avoid zero-division

    if standardize and len(dt_index) > 1:
        outcome_matrix = (outcome_matrix - means) / stds

    stats = {
        "regions": unique_regions,
        "leads": unique_leads,
        "dim": dim,
        "means": means,
        "stds": stds,
        "n_samples": len(dt_index),
    }
    logger.info("Built outcome vectors: shape %s across %d dates", outcome_matrix.shape, len(dt_index))
    return outcome_matrix, stats


def sample_triplets(
    outcome_vectors: np.ndarray,
    dates: Sequence[Union[date, datetime, pd.Timestamp, str]],
    n_triplets: int = DEFAULT_N_TRIPLETS,
    min_separation_days: int = TEMPORAL_EXCLUSION_DAYS,
    seed: int = 42,
) -> List[Tuple[int, int, int]]:
    """Sample (anchor, positive, negative) triplets with mandatory temporal exclusion.

    Constraint:
        1. ||y(a) - y(p)|| < ||y(a) - y(n)|| in outcome space.
        2. |date(a) - date(p)| >= min_separation_days (mandatory >= 15 calendar days).
        3. All indices correspond strictly to training dates (zero held-out access).

    Args:
        outcome_vectors: Array of shape (N, outcome_dim) representing standardized signed errors.
        dates: Sequence of N initialization dates.
        n_triplets: Total number of triplets to generate.
        min_separation_days: Minimum days separation between anchor and positive (default: 15).
        seed: Random seed for reproducibility.

    Returns:
        List[Tuple[int, int, int]]: List of (anchor_idx, positive_idx, negative_idx) tuples.

    Raises:
        PermissionError: If any date is in held-out years.
        ValueError: If pool has insufficient samples to form valid triplets.
    """
    check_heldout_date_protection(dates)
    dt_index = pd.to_datetime(list(dates))
    n_samples = len(dt_index)

    if n_samples < 3:
        raise ValueError(f"Cannot sample triplets with fewer than 3 samples (got {n_samples}).")

    rng = np.random.default_rng(seed)

    # Compute pairwise outcome distance matrix
    # dist[i, j] = ||y(i) - y(j)||_2
    diffs = outcome_vectors[:, np.newaxis, :] - outcome_vectors[np.newaxis, :, :]
    outcome_dists = np.sqrt(np.sum(diffs ** 2, axis=-1))

    # Compute pairwise day delta matrix
    day_deltas = np.abs((dt_index.values[:, np.newaxis] - dt_index.values[np.newaxis, :]) / np.timedelta64(1, "D"))

    triplets: List[Tuple[int, int, int]] = []
    max_attempts = n_triplets * 20
    attempts = 0

    while len(triplets) < n_triplets and attempts < max_attempts:
        attempts += 1
        a_idx = int(rng.integers(0, n_samples))

        # Valid positive candidates must satisfy temporal separation >= min_separation_days
        valid_pos_mask = (day_deltas[a_idx] >= min_separation_days) & (np.arange(n_samples) != a_idx)
        valid_pos_indices = np.where(valid_pos_mask)[0]

        if len(valid_pos_indices) == 0:
            continue

        # Sort valid candidates by outcome distance to anchor
        pos_dists = outcome_dists[a_idx, valid_pos_indices]
        sort_order = np.argsort(pos_dists)
        sorted_pos_indices = valid_pos_indices[sort_order]
        sorted_pos_dists = pos_dists[sort_order]

        # Pick positive from the closer half/candidates
        max_pos_rank = max(1, len(sorted_pos_indices) // 2)
        p_sub_idx = rng.integers(0, max_pos_rank)
        p_idx = int(sorted_pos_indices[p_sub_idx])
        d_ap = float(sorted_pos_dists[p_sub_idx])

        # Candidate negatives must have outcome distance strictly greater than positive: ||y(a) - y(n)|| > ||y(a) - y(p)||
        valid_neg_mask = (outcome_dists[a_idx] > d_ap) & (np.arange(n_samples) != a_idx) & (np.arange(n_samples) != p_idx)
        valid_neg_indices = np.where(valid_neg_mask)[0]

        if len(valid_neg_indices) == 0:
            continue

        n_idx = int(rng.choice(valid_neg_indices))

        # Enforce safety assertions
        assert abs((dt_index[a_idx] - dt_index[p_idx]).days) >= min_separation_days, (
            f"LEAKAGE VIOLATION: anchor {dt_index[a_idx]} and positive {dt_index[p_idx]} "
            f"separated by only {abs((dt_index[a_idx] - dt_index[p_idx]).days)} days!"
        )
        assert outcome_dists[a_idx, p_idx] < outcome_dists[a_idx, n_idx], (
            f"OUTCOME INVERSION: d(a, p)={outcome_dists[a_idx, p_idx]:.4f} >= d(a, n)={outcome_dists[a_idx, n_idx]:.4f}"
        )

        triplets.append((a_idx, p_idx, n_idx))

    logger.info(
        "Sampled %d valid triplets (target %d, attempts %d, min_separation=%dd)",
        len(triplets),
        n_triplets,
        attempts,
        min_separation_days,
    )
    return triplets


class LowRankMahalanobisMetric:
    """Linear Low-Rank Mahalanobis Metric Learning (Node G2, docs/03 §6.2).

    Parameterizes metric M = W^T W, W in R^(n_components x input_dim) (typically 32 x 100).
    Under this metric, the Mahalanobis distance between two EOF state vectors x1 and x2 is:
        d_M(x1, x2) = sqrt((x1 - x2)^T W^T W (x1 - x2)) = ||W(x1 - x2)||_2 = ||z1 - z2||_2
    where z = W x is the 32-dimensional linear embedding.

    Trained via margin-based triplet ranking loss using mini-batch Adam optimization in pure NumPy.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_METRIC_COMPONENTS,
        input_dim: int = DEFAULT_EOF_DIM,
        margin: float = DEFAULT_MARGIN,
        learning_rate: float = DEFAULT_LR,
        n_epochs: int = DEFAULT_N_EPOCHS,
        batch_size: int = DEFAULT_BATCH_SIZE,
        l2_reg: float = DEFAULT_L2_REG,
        seed: int = 42,
    ) -> None:
        self.n_components = n_components
        self.input_dim = input_dim
        self.margin = margin
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.l2_reg = l2_reg
        self.seed = seed

        self.W_: Optional[np.ndarray] = None  # Shape (n_components, input_dim) = (32, 100)
        self.loss_history_: List[float] = []
        self.is_fitted: bool = False

    @property
    def metric_matrix(self) -> np.ndarray:
        """Full Mahalanobis metric matrix M = W^T W in R^(100 x 100)."""
        if self.W_ is None:
            raise RuntimeError("Metric not fitted yet.")
        return self.W_.T @ self.W_

    def fit(
        self,
        X: np.ndarray,
        triplets: List[Tuple[int, int, int]],
    ) -> LowRankMahalanobisMetric:
        """Train projection matrix W linearly minimizing triplet margin ranking loss.

        Loss function:
            L(W) = (1 / |B|) * sum_{(a, p, n)} max(0, margin + ||W(x_a - x_p)|| - ||W(x_a - x_n)||)
                   + l2_reg * ||W||_F^2

        Args:
            X: Array of shape (N, input_dim) containing EOF state representations.
            triplets: List of (anchor_idx, positive_idx, negative_idx) integer tuples.

        Returns:
            LowRankMahalanobisMetric: Self instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        n_samples, actual_dim = X_arr.shape

        if actual_dim != self.input_dim:
            self.input_dim = actual_dim

        if len(triplets) == 0:
            raise ValueError("Cannot fit metric with empty triplets list.")

        rng = np.random.default_rng(self.seed)

        # Initialize W: truncated identity scaled by 1/sqrt(n_components) plus small noise
        W_init = np.zeros((self.n_components, self.input_dim), dtype=np.float64)
        for i in range(min(self.n_components, self.input_dim)):
            W_init[i, i] = 1.0 / np.sqrt(self.n_components)
        W_init += rng.normal(0.0, 0.01 / np.sqrt(self.input_dim), size=W_init.shape)
        W = W_init.copy()

        # Adam optimizer parameters
        m = np.zeros_like(W)
        v = np.zeros_like(W)
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8
        t_step = 0

        n_triplets = len(triplets)
        triplet_arr = np.array(triplets, dtype=np.int64)

        self.loss_history_ = []

        for epoch in range(1, self.n_epochs + 1):
            perm = rng.permutation(n_triplets)
            shuffled_triplets = triplet_arr[perm]
            epoch_loss = 0.0
            n_batches = int(np.ceil(n_triplets / self.batch_size))

            for b in range(n_batches):
                batch_trip = shuffled_triplets[b * self.batch_size : (b + 1) * self.batch_size]
                b_size = len(batch_trip)
                if b_size == 0:
                    continue

                a_idx = batch_trip[:, 0]
                p_idx = batch_trip[:, 1]
                n_idx = batch_trip[:, 2]

                # D_ap = X[a] - X[p], D_an = X[a] - X[n] (shape: b_size, input_dim)
                D_ap = X_arr[a_idx] - X_arr[p_idx]
                D_an = X_arr[a_idx] - X_arr[n_idx]

                # Projected differences: Z_ap = D_ap @ W^T, Z_an = D_an @ W^T (shape: b_size, n_components)
                Z_ap = D_ap @ W.T
                Z_an = D_an @ W.T

                # Norms in projected space
                d_ap = np.sqrt(np.sum(Z_ap ** 2, axis=1) + 1e-12)
                d_an = np.sqrt(np.sum(Z_an ** 2, axis=1) + 1e-12)

                # Margin ranking loss
                raw_loss = self.margin + d_ap - d_an
                active_mask = (raw_loss > 0).astype(np.float64)
                batch_loss = np.sum(raw_loss * active_mask) / b_size

                epoch_loss += float(batch_loss)

                # Vectorized gradient calculation
                # For active triplets: grad_ap = (Z_ap / d_ap)^T @ D_ap, grad_an = (Z_an / d_an)^T @ D_an
                U_ap = (Z_ap / d_ap[:, np.newaxis]) * active_mask[:, np.newaxis]  # (b_size, n_components)
                U_an = (Z_an / d_an[:, np.newaxis]) * active_mask[:, np.newaxis]  # (b_size, n_components)

                grad_W = (U_ap.T @ D_ap - U_an.T @ D_an) / b_size + 2.0 * self.l2_reg * W

                # Adam update
                t_step += 1
                m = beta1 * m + (1.0 - beta1) * grad_W
                v = beta2 * v + (1.0 - beta2) * (grad_W ** 2)

                m_hat = m / (1.0 - beta1 ** t_step)
                v_hat = v / (1.0 - beta2 ** t_step)

                W -= self.learning_rate * m_hat / (np.sqrt(v_hat) + eps)

            avg_epoch_loss = epoch_loss / max(1, n_batches)
            self.loss_history_.append(avg_epoch_loss)

            if epoch % 25 == 0 or epoch == self.n_epochs:
                logger.debug("Epoch %d/%d - Triplet Loss: %.4f", epoch, self.n_epochs, avg_epoch_loss)

        self.W_ = W
        self.is_fitted = True
        logger.info(
            "Fitted LowRankMahalanobisMetric: W shape (%d, %d), final loss: %.4f",
            self.W_.shape[0],
            self.W_.shape[1],
            self.loss_history_[-1] if self.loss_history_ else 0.0,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project atmospheric state from 100-EOF space to 32-dim Mahalanobis space: z = W x.

        Args:
            X: Array of shape (N, input_dim) or (input_dim,).

        Returns:
            np.ndarray: Projected representations of shape (N, n_components) or (n_components,).
        """
        if not self.is_fitted or self.W_ is None:
            raise RuntimeError("LowRankMahalanobisMetric must be fitted before calling transform.")

        X_arr = np.asarray(X, dtype=np.float64)
        is_1d = (X_arr.ndim == 1)
        if is_1d:
            X_arr = X_arr.reshape(1, -1)

        Z = X_arr @ self.W_.T
        return Z[0] if is_1d else Z

    def mahalanobis_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Mahalanobis distance between two 100-EOF state vectors: ||W(x1 - x2)||_2."""
        z1 = self.transform(x1)
        z2 = self.transform(x2)
        diff = z1 - z2
        return float(np.sqrt(np.sum(diff ** 2)))

    def pairwise_distances(self, X: np.ndarray, Y: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute all pairwise Mahalanobis distances between X and Y."""
        ZX = self.transform(X)
        ZY = self.transform(Y) if Y is not None else ZX
        diffs = ZX[:, np.newaxis, :] - ZY[np.newaxis, :, :]
        return np.sqrt(np.sum(diffs ** 2, axis=-1))

    def save(self, filepath: Union[str, Path]) -> None:
        """Persist metric weights W and configuration to disk."""
        if not self.is_fitted or self.W_ is None:
            raise RuntimeError("Cannot save unfitted metric.")
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W=self.W_,
            n_components=self.n_components,
            input_dim=self.input_dim,
            margin=self.margin,
            loss_history=np.array(self.loss_history_),
        )
        logger.info("Saved LowRankMahalanobisMetric to %s", path)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> LowRankMahalanobisMetric:
        """Load persisted metric weights from disk."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Metric file not found: {path}")
        data = np.load(path)
        inst = cls(
            n_components=int(data["n_components"]),
            input_dim=int(data["input_dim"]),
            margin=float(data["margin"]),
        )
        inst.W_ = data["W"]
        inst.loss_history_ = data["loss_history"].tolist()
        inst.is_fitted = True
        logger.info("Loaded LowRankMahalanobisMetric from %s (W shape: %s)", path, inst.W_.shape)
        return inst
