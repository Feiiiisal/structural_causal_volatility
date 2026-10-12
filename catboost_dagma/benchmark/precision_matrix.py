"""Graphical LASSO benchmark."""
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.covariance import GraphicalLasso, GraphicalLassoCV, LedoitWolf

from catboost_dagma.dagma.solver import get_eigenvector_centrality


class PrecisionMatrixEstimator:
    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        cv_folds: int = 3,
        threshold: float = 0.10,
    ):
        self.alphas = alphas or [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        self.cv_folds = cv_folds
        self.threshold = threshold

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        std = X.std(axis=0, ddof=1)
        std[std < 1e-10] = 1.0
        return (X - X.mean(axis=0)) / std

    def fit_precision_matrix(self, X_window: np.ndarray) -> Tuple[np.ndarray, float]:
        X = self._standardize(X_window)
        if not np.isfinite(X).all():
            raise ValueError("Non-finite values in precision-matrix window.")

        try:
            model = GraphicalLassoCV(
                alphas=self.alphas,
                cv=self.cv_folds,
                max_iter=500,
            ).fit(X)
            return model.precision_.copy(), float(model.alpha_)
        except Exception:
            pass

        try:
            model = GraphicalLasso(alpha=0.01, max_iter=500).fit(X)
            return model.precision_.copy(), 0.01
        except Exception:
            cov = LedoitWolf().fit(X).covariance_
            return np.linalg.pinv(cov), 0.0

    @staticmethod
    def precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
        diag = np.clip(np.diag(precision), 1e-12, None)
        scale = np.sqrt(np.outer(diag, diag))
        partial = -precision / scale
        np.fill_diagonal(partial, 0.0)
        return partial

    def extract_topologies(self, precision_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        partial = self.precision_to_partial_correlation(precision_matrix)
        adj = np.where(np.abs(partial) > self.threshold, np.abs(partial), 0.0)

        return {
            "precision_degree": (adj > 0).sum(axis=1).astype(float),
            "precision_eigen": get_eigenvector_centrality(adj, self.threshold),
            "precision_matrix": precision_matrix,
            "partial_correlation": partial,
        }
