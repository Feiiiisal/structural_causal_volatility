"""
Precision Matrix Benchmark Engine via Graphical LASSO and Ledoit-Wolf Shrinkage.
Addresses Capstone Instructor Feedback (b):
Benchmarks non-linear DAGMA causal graphs against conditional independence structures
derived from the inverse covariance (precision) matrix Theta = Sigma^{-1}.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso, LedoitWolf
from catboost_dagma.dagma.solver import get_eigenvector_centrality


class PrecisionMatrixEstimator:
    """
    Estimates the regularized precision matrix Theta = Sigma^{-1}.
    Under Gaussian assumptions, Theta_{ij} = 0 if and only if variable i
    and variable j are conditionally independent given all other variables.
    """

    def __init__(
        self,
        alphas: Optional[List[float]] = None,
        cv_folds: int = 3,
        threshold: float = 0.05,
    ):
        self.alphas = alphas or [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
        self.cv_folds = cv_folds
        self.threshold = threshold

    def fit_precision_matrix(
        self,
        X_window: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """
        Estimates the sparse precision matrix using Graphical LASSO.
        Falls back to Ledoit-Wolf analytical shrinkage if Graphical LASSO encounters convergence difficulties.
        
        Args:
            X_window: Standardized or raw asset returns/residuals of shape (T, N)
        Returns:
            precision_matrix: Symmetric matrix of shape (N, N)
            best_alpha: Regularization parameter used
        """
        T, N = X_window.shape
        # Center the data
        X_centered = X_window - np.mean(X_window, axis=0, keepdims=True)

        try:
            # Graphical LASSO with cross-validation
            gl = GraphicalLassoCV(alphas=self.alphas, cv=self.cv_folds, max_iter=200)
            gl.fit(X_centered)
            precision = gl.precision_.copy()
            best_alpha = float(gl.alpha_)
        except Exception:
            try:
                # Fixed alpha graphical lasso fallback
                gl_fixed = GraphicalLasso(alpha=0.01, max_iter=200)
                gl_fixed.fit(X_centered)
                precision = gl_fixed.precision_.copy()
                best_alpha = 0.01
            except Exception:
                # Ledoit-Wolf analytical shrinkage fallback
                lw = LedoitWolf().fit(X_centered)
                cov = lw.covariance_
                precision = np.linalg.pinv(cov)
                best_alpha = 0.0

        # Zero out diagonal for topological graph metrics
        np.fill_diagonal(precision, 0.0)
        return precision, best_alpha

    def extract_topologies(
        self,
        precision_matrix: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """
        Extracts topological graph metrics from the precision matrix.
        - Precision degree centrality (number of conditionally dependent neighbors)
        - Precision eigenvector centrality (systemic centrality in conditional independence graph)
        """
        adj = np.where(np.abs(precision_matrix) > self.threshold, np.abs(precision_matrix), 0.0)
        degrees = np.sum(adj > 0, axis=1).astype(float)
        eigen_centrality = get_eigenvector_centrality(adj, threshold=self.threshold)

        return {
            "precision_degree": degrees,
            "precision_eigen": eigen_centrality,
            "precision_matrix": precision_matrix,
        }
