import numpy as np
import pandas as pd

from catboost_dagma.benchmark.orthogonality import NetworkFeatureAudit
from catboost_dagma.benchmark.precision_matrix import PrecisionMatrixEstimator


def test_precision_matrix_and_partial_correlations():
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 4))

    estimator = PrecisionMatrixEstimator(threshold=0.05)
    precision, _ = estimator.fit_precision_matrix(X)
    partial = estimator.precision_to_partial_correlation(precision)
    topology = estimator.extract_topologies(precision)

    assert precision.shape == (4, 4)
    assert np.allclose(precision, precision.T)
    assert np.allclose(np.diag(partial), 0.0)
    assert len(topology["precision_degree"]) == 4
    assert len(topology["precision_eigen"]) == 4


def test_network_feature_audit():
    rng = np.random.default_rng(4)
    n = 100
    df = pd.DataFrame({
        "causal_in_deg": rng.uniform(0, 5, n),
        "causal_out_deg": rng.uniform(0, 5, n),
        "causal_eigen": rng.uniform(0, 1, n),
        "corr_deg": rng.uniform(0, 5, n),
        "corr_eigen": rng.uniform(0, 1, n),
        "precision_deg": rng.uniform(0, 5, n),
        "precision_eigen": rng.uniform(0, 1, n),
        "causal_drift": rng.uniform(0, 2, n),
    })

    audit = NetworkFeatureAudit()
    correlations = audit.audit_topological_correlations(df)
    vif = audit.audit_variance_inflation_factors(df)

    assert not correlations.empty
    assert "Pearson_r" in correlations
    assert not vif.empty
    assert "VIF" in vif

    W = np.array([
        [0.0, 1.5, 0.0],
        [0.2, 0.0, 0.0],
        [0.0, 0.8, 0.0],
    ])
    assert audit.compute_directed_asymmetry_index(W) > 0
