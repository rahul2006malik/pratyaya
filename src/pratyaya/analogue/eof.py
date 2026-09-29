"""Empirical Orthogonal Functions (EOF) / PCA Dimensionality Reduction (Node G1, docs/03 §6.2).

Compresses the wide-domain synoptic state vector (~11,590 features) into the top 100
principal components (EOFs).
Fitted strictly on training-fold dates (2000–2001). Zero leakage into validation or held-out folds.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from pratyaya.analogue.state import check_heldout_date_protection

logger = logging.getLogger(__name__)

DEFAULT_N_COMPONENTS: int = 100


class EOFReducer:
    """PCA / EOF reduction module retaining 100 principal components.

    Per docs/03 §6.2: Output x ∈ R^100 per init date. Euclidean analogue search
    in EOF space provides the classical, unlearned baseline fallback.
    """

    def __init__(
        self,
        n_components: int = DEFAULT_N_COMPONENTS,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.random_state = random_state
        self.pca_: Optional[PCA] = None
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.cumulative_variance_: float = 0.0
        self.mean_: Optional[np.ndarray] = None
        self.fitted_dates_: Optional[pd.DatetimeIndex] = None
        self.is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        dates: Optional[Sequence[Union[date, datetime, pd.Timestamp, str]]] = None,
    ) -> EOFReducer:
        """Fit EOF / PCA basis strictly on training-fold state vectors.

        Args:
            X: Array of shape (n_samples, n_features) representing standardized synoptic states.
            dates: Sequence of timestamps for the state vectors.

        Raises:
            PermissionError: If any date is in held-out years (2018, 2019).
            ValueError: If X has invalid dimensions or zero samples.
        """
        if dates is not None:
            check_heldout_date_protection(dates)
            self.fitted_dates_ = pd.to_datetime(list(dates))

        if X.ndim != 2:
            raise ValueError(f"Expected 2D array (n_samples, n_features), got shape {X.shape}")
        n_samples, n_features = X.shape
        if n_samples == 0:
            raise ValueError("Cannot fit EOF basis on 0 samples.")

        # Adjust components if sample size or features is smaller than n_components
        effective_components = min(self.n_components, n_samples, n_features)
        if effective_components < self.n_components:
            logger.warning(
                "Requested n_components=%d exceeds min(n_samples=%d, n_features=%d). Using %d components.",
                self.n_components,
                n_samples,
                n_features,
                effective_components,
            )

        self.pca_ = PCA(
            n_components=effective_components,
            random_state=self.random_state,
        )
        self.pca_.fit(X)

        self.components_ = self.pca_.components_
        self.explained_variance_ = self.pca_.explained_variance_
        self.explained_variance_ratio_ = self.pca_.explained_variance_ratio_
        self.cumulative_variance_ = float(np.sum(self.explained_variance_ratio_))
        self.mean_ = self.pca_.mean_
        self.is_fitted = True

        logger.info(
            "Fitted EOF basis with %d components on %d samples (cumulative explained variance: %.2f%%)",
            effective_components,
            n_samples,
            self.cumulative_variance_ * 100.0,
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project synoptic state vectors into EOF / principal component space.

        Args:
            X: Array of shape (n_samples, n_features) or (n_features,).

        Returns:
            np.ndarray: Array of shape (n_samples, n_components) or (n_components,).
        """
        if not self.is_fitted or self.pca_ is None:
            raise RuntimeError("EOFReducer must be fitted before calling transform.")

        is_single = (X.ndim == 1)
        if is_single:
            X_2d = X.reshape(1, -1)
        else:
            X_2d = X

        transformed = self.pca_.transform(X_2d)

        if is_single:
            return transformed[0]
        return transformed

    def fit_transform(
        self,
        X: np.ndarray,
        dates: Optional[Sequence[Union[date, datetime, pd.Timestamp, str]]] = None,
    ) -> np.ndarray:
        """Fit EOF basis on training fold and return transformed components."""
        return self.fit(X, dates).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct original state feature space from EOF components.

        Args:
            Z: Array of shape (n_samples, n_components) or (n_components,).

        Returns:
            np.ndarray: Reconstructed states in original feature space.
        """
        if not self.is_fitted or self.pca_ is None:
            raise RuntimeError("EOFReducer must be fitted before calling inverse_transform.")

        is_single = (Z.ndim == 1)
        if is_single:
            Z_2d = Z.reshape(1, -1)
        else:
            Z_2d = Z

        reconstructed = self.pca_.inverse_transform(Z_2d)

        if is_single:
            return reconstructed[0]
        return reconstructed
