"""Rolling DAGMA fit."""
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import torch

from catboost_dagma.dagma.model import (
    DeepDynotearsMLP,
    compute_gaussian_loss,
    compute_student_t_loss,
    tensor_to_numpy,
)


def get_eigenvector_centrality(W: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    adj = np.where(np.abs(W) > threshold, np.abs(W), 0.0)
    graph = nx.from_numpy_array(adj, create_using=nx.DiGraph)

    try:
        values = nx.eigenvector_centrality_numpy(graph, weight="weight")
        return np.array([values.get(i, 0.0) for i in range(len(W))])
    except Exception:
        degree = adj.sum(axis=1)
        scale = np.linalg.norm(degree)
        return degree / scale if scale else np.zeros(len(W))


def _fit_stage(
    model,
    X,
    Xlags,
    max_iter,
    lr,
    lambda1,
    mu,
    s,
    loss_type,
    learnable_nu,
    tol,
    checkpoint,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))
    last = np.inf

    for i in range(max_iter):
        optimizer.zero_grad()
        h = model.h_func(s)
        if h.item() < 0:
            return False

        if loss_type == "student-t":
            loss = compute_student_t_loss(model, X, Xlags, learnable_nu)
        else:
            loss = compute_gaussian_loss(model, X, Xlags)

        objective = mu * (loss + lambda1 * model.fc1_l1_reg()) + h
        if not torch.isfinite(objective):
            return False

        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if i % checkpoint == 0 or i == max_iter - 1:
            value = objective.item()
            change = abs((last - value) / max(abs(last), 1e-12))
            if change <= tol:
                break
            last = value

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
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64

    X = torch.tensor(X_np, dtype=dtype, device=device)
    Xlags = torch.tensor(Xlags_np, dtype=dtype, device=device)
    d = X.shape[1]
    p = Xlags.shape[1] // d

    model = DeepDynotearsMLP(d, p, dtype=dtype).to(device)
    mu = mu_init

    for stage in range(T):
        iterations = max_iter if stage == T - 1 else warm_iter
        saved = copy.deepcopy(model.state_dict())
        step_lr = lr

        for _ in range(5):
            ok = _fit_stage(
                model, X, Xlags, iterations, step_lr, lambda1, mu, s,
                loss_type, learnable_nu, tol, checkpoint,
            )
            if ok:
                break
            model.load_state_dict(copy.deepcopy(saved))
            step_lr *= 0.5

        mu *= mu_factor

    return model


def _standardize(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    std = X.std(axis=0, ddof=1)
    std[std < 1e-10] = 1.0
    Z = (X - X.mean(axis=0)) / std

    if not np.isfinite(Z).all():
        raise ValueError("Non-finite values in DAGMA window.")

    return Z


def _fit_window(args: Tuple[int, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    idx, window, kwargs = args
    window = _standardize(window)
    X, Xlags = window[1:], window[:-1]

    params = kwargs.copy()
    params["device"] = "cpu"
    model = train_dagma_dynotears(X, Xlags, **params)

    W = tensor_to_numpy(model.get_W_adj())
    A = tensor_to_numpy(model.get_A_adj())

    mask = ~np.eye(W.shape[0], dtype=bool)
    weights = np.abs(W[mask])
    threshold = max(float(np.percentile(weights, 80)), 1e-4) if weights.size else 1e-4

    return {
        "window_idx": idx,
        "W": W,
        "A": A,
        "nu": model.get_degrees_of_freedom(),
        "edge_threshold": threshold,
        "in_degree_causal": (np.abs(W) > threshold).sum(axis=0).astype(float),
        "out_degree_causal": (np.abs(W) > threshold).sum(axis=1).astype(float),
        "eigen_causal": get_eigenvector_centrality(W, threshold),
    }


def run_parallel_rolling_dagma(
    X_data: np.ndarray,
    dates: List[Any],
    window_size: int = 252,
    step_size: int = 21,
    dagma_kwargs: Optional[Dict[str, Any]] = None,
    n_workers: int = 8,
) -> List[Dict[str, Any]]:
    X_data = np.asarray(X_data, dtype=float)
    rows, assets = X_data.shape

    if window_size <= assets:
        raise ValueError("window_size must be larger than the number of assets.")
    if len(dates) != rows:
        raise ValueError("dates and X_data must have the same number of rows.")

    ends = list(range(window_size, rows, step_size))
    if not ends:
        return []

    params = dagma_kwargs or {
        "loss_type": "student-t",
        "learnable_nu": True,
        "warm_iter": 80,
        "max_iter": 150,
        "lambda1": 0.02,
        "verbose": False,
    }
    jobs = [(i, X_data[end - window_size:end], params) for i, end in enumerate(ends)]

    import multiprocessing as mp

    results = [None] * len(jobs)
    ctx = mp.get_context("spawn")

    with ProcessPoolExecutor(
        max_workers=max(1, min(n_workers, len(jobs))),
        mp_context=ctx,
    ) as pool:
        futures = {pool.submit(_fit_window, job): job[0] for job in jobs}

        for future in as_completed(futures):
            result = future.result()
            i = result["window_idx"]
            end = ends[i]
            result["end_idx"] = end
            result["date"] = dates[end - 1]
            result["target_start_idx"] = end
            results[i] = result

    return results
