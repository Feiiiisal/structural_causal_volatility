"""Graphical LASSO benchmark for conditional-dependence networks."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso, LedoitWolf
from catboost_dagma.dagma.solver import get_eigenvector_centrality


class PrecisionMatrixEstimator:
    def __init__(self, alphas: Optional[List[float]] = None, cv_folds: int = 3, threshold: float = 0.10):
        self.alphas = alphas or [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        self.cv_folds = cv_folds
        self.threshold = threshold

    @staticmethod
    def _standardize(X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        mu = np.nanmean(X, axis=0, keepdims=True)
        sd = np.nanstd(X, axis=0, ddof=1, keepdims=True)
        sd = np.where(sd < 1e-10, 1.0, sd)
        return (X - mu) / sd

    def fit_precision_matrix(self, X_window: np.ndarray) -> Tuple[np.ndarray, float]:
        X = self._standardize(X_window)
        if not np.isfinite(X).all():
            raise ValueError("Precision benchmark received non-finite values.")
        try:
            model = GraphicalLassoCV(alphas=self.alphas, cv=self.cv_folds, max_iter=500).fit(X)
            precision = model.precision_.copy()
            alpha = float(model.alpha_)
        except Exception:
            try:
                model = GraphicalLasso(alpha=0.01, max_iter=500).fit(X)
                precision = model.precision_.copy()
                alpha = 0.01
            except Exception:
                cov = LedoitWolf().fit(X).covariance_
                precision = np.linalg.pinv(cov)
                alpha = 0.0
        return precision, alpha

    @staticmethod
    def precision_to_partial_correlation(precision: np.ndarray) -> np.ndarray:
        precision = np.asarray(precision, dtype=float)
        diag = np.clip(np.diag(precision), 1e-12, None)
        denom = np.sqrt(np.outer(diag, diag))
        partial = -precision / denom
        np.fill_diagonal(partial, 0.0)
        return partial

    def extract_topologies(self, precision_matrix: np.ndarray) -> Dict[str, np.ndarray]:
        partial = self.precision_to_partial_correlation(precision_matrix)
        adj = np.where(np.abs(partial) > self.threshold, np.abs(partial), 0.0)
        degree = np.sum(adj > 0, axis=1).astype(float)
        eigen = get_eigenvector_centrality(adj, threshold=self.threshold)
        return {
            "precision_degree": degree,
            "precision_eigen": eigen,
            "precision_matrix": precision_matrix,
            "partial_correlation": partial,
        }
