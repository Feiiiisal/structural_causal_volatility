"""
Unit tests for DeepDynotearsMLP with Student-t log-likelihood and log-det acyclicity.
"""

import torch
import numpy as np
import pytest
from catboost_dagma.dagma.model import (
    DeepDynotearsMLP,
    compute_student_t_loss,
    compute_gaussian_loss,
    tensor_to_numpy,
)
from catboost_dagma.dagma.solver import train_dagma_dynotears, get_eigenvector_centrality


def test_deep_dynotears_mlp_forward():
    d = 4
    p = 1
    n = 30
    model = DeepDynotearsMLP(d_vars=d, p_orders=p, hidden_dim=8)

    X = torch.randn(n, d, dtype=torch.float64)
    Xlags = torch.randn(n, p * d, dtype=torch.float64)

    X_hat = model(X, Xlags)
    assert X_hat.shape == (n, d)

    # Check that diagonal weights are zero
    W = model.get_W_adj()
    assert W.shape == (d, d)
    for i in range(d):
        assert torch.isclose(W[i, i], torch.tensor(0.0, dtype=torch.float64), atol=1e-5)


def test_student_t_loss_and_degrees_of_freedom():
    d = 3
    p = 1
    n = 25
    model = DeepDynotearsMLP(d_vars=d, p_orders=p, hidden_dim=8, init_nu=4.0)

    X = torch.randn(n, d, dtype=torch.float64)
    Xlags = torch.randn(n, p * d, dtype=torch.float64)

    loss_t = compute_student_t_loss(model, X, Xlags, learnable_nu=True)
    assert loss_t.item() > 0.0
    assert torch.isfinite(loss_t)

    nu = model.get_degrees_of_freedom()
    assert nu >= 2.0


def test_dagma_h_func_logdet():
    d = 3
    p = 1
    model = DeepDynotearsMLP(d_vars=d, p_orders=p, hidden_dim=8)
    h_val = model.h_func(s=1.0)
    assert torch.isfinite(h_val)


def test_train_dagma_dynotears_fast():
    np.random.seed(42)
    n, d = 40, 3
    X_curr = np.random.randn(n, d)
    X_lags = np.random.randn(n, d)

    model = train_dagma_dynotears(
        X_curr, X_lags,
        loss_type="student-t",
        T=2,
        warm_iter=15,
        max_iter=25,
        lambda1=0.01,
        verbose=False,
    )
    W = tensor_to_numpy(model.get_W_adj())
    assert W.shape == (d, d)
    assert np.all(W >= 0.0)

    # Eigenvector centrality check
    eigen = get_eigenvector_centrality(W, threshold=0.01)
    assert len(eigen) == d
