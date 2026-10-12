"""DAGMA/DYNOTEARS optimizer and rolling-window estimator."""
import copy
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import networkx as nx
import torch

from catboost_dagma.dagma.model import DeepDynotearsMLP, compute_student_t_loss, compute_gaussian_loss, tensor_to_numpy


def get_eigenvector_centrality(W: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    W_thresh = np.where(np.abs(W) > threshold, np.abs(W), 0.0)
    G = nx.from_numpy_array(W_thresh, create_using=nx.DiGraph)
    try:
        centrality = nx.eigenvector_centrality_numpy(G, weight="weight")
        return np.array([centrality.get(i, 0.0) for i in range(len(W))])
    except Exception:
        degrees = np.sum(W_thresh, axis=1)
        norm = np.linalg.norm(degrees)
        return degrees / norm if norm > 0 else np.zeros(len(W))


def _minimize_stage(model, X, Xlags, max_iter, lr, lambda1, mu, s, loss_type="student-t", learnable_nu=True, tol=1e-6, checkpoint=50, verbose=False):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.99, 0.999))
    obj_prev = 1e16
    for i in range(max_iter):
        optimizer.zero_grad()
        h_val = model.h_func(s)
        if h_val.item() < 0.0:
            return False
        score = compute_student_t_loss(model, X, Xlags, learnable_nu=learnable_nu) if loss_type == "student-t" else compute_gaussian_loss(model, X, Xlags)
        obj = mu * (score + lambda1 * model.fc1_l1_reg()) + h_val
        if torch.isnan(obj) or torch.isinf(obj):
            return False
        obj.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if i % checkpoint == 0 or i == max_iter - 1:
            val = obj.item()
            if abs((obj_prev - val) / max(abs(obj_prev), 1e-12)) <= tol:
                break
            obj_prev = val
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
        inner_iter = max_iter if stage == T - 1 else warm_iter
        state = copy.deepcopy(model.state_dict())
        success, attempt, lr_cur, s_cur = False, 0, lr, s
        while not success and attempt < 5:
            attempt += 1
            success = _minimize_stage(model, X, Xlags, inner_iter, lr_cur, lambda1, mu, s_cur, loss_type, learnable_nu, tol, checkpoint, verbose)
            if not success:
                model.load_state_dict(copy.deepcopy(state))
                lr_cur *= 0.5
                s_cur = 1.0
        mu *= mu_factor
    return model


def _standardize_window(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, ddof=1, keepdims=True)
    sd = np.where(sd < 1e-10, 1.0, sd)
    Z = (X - mu) / sd
    if not np.isfinite(Z).all():
        raise ValueError("DAGMA window contains non-finite values after standardization.")
    return Z


def _fit_single_window_worker(args: Tuple[int, np.ndarray, Dict[str, Any]]) -> Dict[str, Any]:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    window_idx, X_window, kwargs = args
    X_window = _standardize_window(X_window)
    X_curr, X_lags = X_window[1:], X_window[:-1]
    local_kwargs = kwargs.copy()
    local_kwargs["device"] = "cpu"
    model = train_dagma_dynotears(X_curr, X_lags, **local_kwargs)
    W = tensor_to_numpy(model.get_W_adj())
    A = tensor_to_numpy(model.get_A_adj())
    nu = model.get_degrees_of_freedom()
    w_abs = np.abs(W)
    off_diag = w_abs[~np.eye(w_abs.shape[0], dtype=bool)]
    thresh = max(float(np.percentile(off_diag, 80)) if off_diag.size and off_diag.max() > 0 else 0.001, 1e-4)
    return {
        "window_idx": window_idx,
        "W": W,
        "A": A,
        "nu": nu,
        "edge_threshold": thresh,
        "in_degree_causal": (w_abs > thresh).sum(axis=0).astype(float),
        "out_degree_causal": (w_abs > thresh).sum(axis=1).astype(float),
        "eigen_causal": get_eigenvector_centrality(W, threshold=thresh),
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
    T_samples, N_assets = X_data.shape
    if window_size <= N_assets:
        raise ValueError(f"window_size={window_size} must exceed asset count={N_assets} for a defensible high-dimensional fit.")
    if len(dates) != T_samples:
        raise ValueError("dates length must match X_data rows")
    timesteps = list(range(window_size, T_samples, step_size))
    kwargs = dagma_kwargs or {"loss_type": "student-t", "learnable_nu": True, "warm_iter": 80, "max_iter": 150, "lambda1": 0.02, "verbose": False}
    jobs = [(i, X_data[end-window_size:end], kwargs) for i, end in enumerate(timesteps)]
    if not jobs:
        return []

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    results = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, min(n_workers, len(jobs))), mp_context=ctx) as executor:
        futures = {executor.submit(_fit_single_window_worker, job): job[0] for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            idx = res["window_idx"]
            end_idx = timesteps[idx]
            # The graph only uses rows [end-window, end); its as-of date is therefore end-1.
            res["end_idx"] = end_idx
            res["date"] = dates[end_idx - 1]
            res["target_start_idx"] = end_idx
            results[idx] = res
    return results
