"""
DAGMA-DYNOTEARS Central-Path Optimizer and Parallel Rolling-Window Solver.
Implements the continuous unconstrained optimization framework with domain-safety rollback
and ProcessPoolExecutor CPU scaling with thread oversubscription prevention.
"""

import copy
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import networkx as nx
import torch

from catboost_dagma.dagma.model import (
    DeepDynotearsMLP,
    compute_student_t_loss,
    compute_gaussian_loss,
    tensor_to_numpy,
)
from catboost_dagma.config import DAGMA_CONFIG


def get_eigenvector_centrality(
    W: np.ndarray,
    threshold: float = 0.05,
) -> np.ndarray:
    """
    Computes eigenvector centrality on directed graph adjacency matrix W.
    Guards against non-convergence by falling back to degree centrality.
    """
    W_thresh = np.where(np.abs(W) > threshold, np.abs(W), 0.0)
    G = nx.from_numpy_array(W_thresh, create_using=nx.DiGraph)
    try:
        centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
        return np.array([centrality.get(i, 0.0) for i in range(len(W))])
    except Exception:
        # Fallback to normalized degree centrality
        degrees = np.sum(W_thresh, axis=1)
        norm = np.linalg.norm(degrees)
        return degrees / norm if norm > 0 else np.zeros(len(W))


def _minimize_stage(
    model: DeepDynotearsMLP,
    X: torch.Tensor,
    Xlags: torch.Tensor,
    max_iter: int,
    lr: float,
    lambda1: float,
    mu: float,
    s: float,
    loss_type: str = "student-t",
    learnable_nu: bool = True,
    tol: float = 1e-6,
    checkpoint: int = 50,
    verbose: bool = False,
) -> bool:
    """
    One central-path stage: minimizes mu * (loss + lambda1 * L1) + h^s(W).
    Returns True on convergence, False if domain violation (h < 0) occurs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))
    obj_prev = 1e16

    for i in range(max_iter):
        optimizer.zero_grad()
        h_val = model.h_func(s)

        # Domain violation check: iterate must stay in positive definite cone
        if h_val.item() < 0.0:
            if verbose:
                print(f"      [Domain Violation] h={h_val.item():.6f} at inner iter {i}")
            return False

        if loss_type == "student-t":
            score = compute_student_t_loss(model, X, Xlags, learnable_nu=learnable_nu)
        else:
            score = compute_gaussian_loss(model, X, Xlags)

        l1_penalty = lambda1 * model.fc1_l1_reg()
        obj = mu * (score + l1_penalty) + h_val

        if torch.isnan(obj) or torch.isinf(obj):
            return False

        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i % checkpoint == 0 or i == max_iter - 1:
            obj_new = obj.item()
            if abs((obj_prev - obj_new) / max(abs(obj_prev), 1e-12)) <= tol:
                break
            obj_prev = obj_new

    return True


def train_dagma_dynotears(
    X_np: np.ndarray,
    Xlags_np: np.ndarray,
    loss_type: str = "student-t",
    learnable_nu: bool = True,
    lambda1: float = 0.02,
    T: int = 4,
    mu_init: float = 0.1,
    mu_factor: float = 0.1,
    s: float = 1.0,
    warm_iter: int = 100,
    max_iter: int = 200,
    lr: float = 0.005,
    tol: float = 1e-6,
    checkpoint: int = 50,
    device: Optional[str] = None,
    verbose: bool = False,
) -> DeepDynotearsMLP:
    """
    Executes faithful central-path DAGMA optimization with geometric mu decay
    and automated rollback-and-retry upon encountering infeasible points.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float64

    X = torch.tensor(X_np, dtype=dtype, device=device)
    Xlags = torch.tensor(Xlags_np, dtype=dtype, device=device)
    n, d = X.shape
    p = Xlags.shape[1] // d

    model = DeepDynotearsMLP(d, p, dtype=dtype).to(device)

    mu = mu_init
    for t in range(T):
        inner_iter = max_iter if t == T - 1 else warm_iter
        s_cur = s
        lr_cur = lr
        checkpoint_state = copy.deepcopy(model.state_dict())
        success = False
        attempt = 0

        while not success and attempt < 5:
            attempt += 1
            success = _minimize_stage(
                model=model,
                X=X,
                Xlags=Xlags,
                max_iter=inner_iter,
                lr=lr_cur,
                lambda1=lambda1,
                mu=mu,
                s=s_cur,
                loss_type=loss_type,
                learnable_nu=learnable_nu,
                tol=tol,
                checkpoint=checkpoint,
                verbose=verbose,
            )
            if not success:
                # Rollback to stage start, halve learning rate, reset s=1.0
                model.load_state_dict(copy.deepcopy(checkpoint_state))
                lr_cur *= 0.5
                s_cur = 1.0
                if lr_cur < 1e-8:
                    break

        mu *= mu_factor

    return model


def _fit_single_window_worker(args: Tuple[int, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Worker function executed in separate process with strict thread isolation
    to prevent thread oversubscription on multi-core systems.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    window_idx, X_window, kwargs = args
    X_curr = X_window[1:]
    X_lags = X_window[:-1]

    kwargs_worker = kwargs.copy()
    kwargs_worker["device"] = "cpu"

    # Run DAGMA optimization
    model = train_dagma_dynotears(X_curr, X_lags, **kwargs_worker)
    W = tensor_to_numpy(model.get_W_adj())
    A = tensor_to_numpy(model.get_A_adj())
    nu = model.get_degrees_of_freedom()

    # Extract topological features with adaptive threshold
    w_abs = np.abs(W)
    thresh = float(np.percentile(w_abs, 80)) if np.max(w_abs) > 0 else 0.001
    thresh = max(thresh, 1e-4)

    in_degree_causal = (w_abs > thresh).sum(axis=0).astype(float)
    out_degree_causal = (w_abs > thresh).sum(axis=1).astype(float)
    eigen_causal = get_eigenvector_centrality(W, threshold=thresh)

    return {
        "window_idx": window_idx,
        "W": W,
        "A": A,
        "nu": nu,
        "in_degree_causal": in_degree_causal,
        "out_degree_causal": out_degree_causal,
        "eigen_causal": eigen_causal,
    }


def run_parallel_rolling_dagma(
    X_data: np.ndarray,
    dates: List[Any],
    window_size: int = 60,
    step_size: int = 5,
    dagma_kwargs: Optional[Dict[str, Any]] = None,
    n_workers: int = 8,
) -> List[Dict[str, Any]]:
    """
    Rolls DAGMA over time series with embarrassingly parallel ProcessPoolExecutor execution.
    """
    T_samples, N_assets = X_data.shape
    timesteps = list(range(window_size, T_samples, step_size))
    kwargs = dagma_kwargs or {
        "loss_type": "student-t",
        "learnable_nu": True,
        "warm_iter": 80,
        "max_iter": 150,
        "lambda1": 0.02,
        "verbose": False,
    }

    jobs = []
    for i, end_idx in enumerate(timesteps):
        X_window = X_data[end_idx - window_size:end_idx]
        jobs.append((i, X_window, kwargs))

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as executor:
        futures = {executor.submit(_fit_single_window_worker, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            idx = res["window_idx"]
            results[idx] = res
            results[idx]["end_idx"] = timesteps[idx]
            results[idx]["date"] = dates[timesteps[idx]]

    return results
