"""
Unit tests for Precision Matrix Estimation and Empirical Orthogonality Audit.
"""

import numpy as np
import pandas as pd
import pytest
from catboost_dagma.benchmark.precision_matrix import PrecisionMatrixEstimator
from catboost_dagma.benchmark.orthogonality import EmpiricalOrthogonalityAuditor


def test_precision_matrix_estimator():
    np.random.seed(42)
    n, d = 60, 4
    X = np.random.randn(n, d)
    estimator = PrecisionMatrixEstimator(threshold=0.01)
    p_mat, alpha = estimator.fit_precision_matrix(X)

    assert p_mat.shape == (d, d)
    assert np.allclose(p_mat, p_mat.T)  # Must be symmetric
    assert np.all(np.diag(p_mat) == 0.0)  # Diagonal zeroed

    topos = estimator.extract_topologies(p_mat)
    assert len(topos["precision_degree"]) == d
    assert len(topos["precision_eigen"]) == d


def test_empirical_orthogonality_auditor():
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        "causal_in_deg": np.random.uniform(0, 5, n),
        "causal_out_deg": np.random.uniform(0, 5, n),
        "causal_eigen": np.random.uniform(0, 1, n),
        "corr_deg": np.random.uniform(0, 5, n),
        "corr_eigen": np.random.uniform(0, 1, n),
        "precision_deg": np.random.uniform(0, 5, n),
        "precision_eigen": np.random.uniform(0, 1, n),
        "causal_drift": np.random.uniform(0, 2, n),
    })

    auditor = EmpiricalOrthogonalityAuditor()
    ortho_df = auditor.audit_topological_correlations(df)
    assert len(ortho_df) > 0
    assert "Pearson_r" in ortho_df.columns
    assert "Empirical_Verdict" in ortho_df.columns

    vif_df = auditor.audit_variance_inflation_factors(df)
    assert len(vif_df) > 0
    assert "VIF" in vif_df.columns
    assert "Multicollinearity_Status" in vif_df.columns

    # Directed asymmetry index on asymmetric matrix
    W_asym = np.array([[0, 1.5, 0], [0.2, 0, 0], [0, 0.8, 0]])
    asym_val = auditor.compute_directed_asymmetry_index(W_asym)
    assert asym_val > 0.0  # Must be strictly positive for directed graph
