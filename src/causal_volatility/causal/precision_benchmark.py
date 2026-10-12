"""Precision Matrix Benchmark via Graphical LASSO / Sparse Inverse Covariance.

Estimates a sparse precision (inverse covariance) matrix using
Graphical LASSO (when scikit-learn is available) or regularized pseudo-inverse.
Compares conditional independence edges against Granger causal DAG edges
to identify high-confidence, doubly-validated causal channels.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from sklearn.covariance import GraphicalLassoCV
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


def estimate_precision_matrix(
    df: pd.DataFrame,
    target_var: str = "Vol_Innovations",
    alpha_range: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Estimate sparse precision matrix via GraphicalLassoCV or regularized pseudo-inverse.
    
    Args:
        df: DataFrame of stationary causal variables.
        target_var: Target variable name.
        alpha_range: Optional array of regularization alphas to cross-validate.
    
    Returns:
        Dictionary containing:
            - 'precision_matrix': ndarray of precision entries
            - 'covariance_matrix': ndarray of estimated covariance  
            - 'partial_corr_matrix': normalized partial correlations
            - 'var_names': list of variable names
            - 'optimal_alpha': selected regularization parameter
            - 'significant_edges': list of (source, target, partial_corr) tuples
    """
    clean_df = df.dropna()
    var_names = list(clean_df.columns)
    
    # Standardize for numerical stability
    X = clean_df.values
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    
    if HAS_SKLEARN:
        if alpha_range is None:
            alpha_range = np.logspace(-3, 0, 30)
        try:
            glasso = GraphicalLassoCV(alphas=alpha_range, cv=3, max_iter=300)
            glasso.fit(X_std)
            precision = glasso.precision_
            covariance = glasso.covariance_
            optimal_alpha = float(glasso.alpha_)
        except Exception:
            covariance = np.cov(X_std, rowvar=False)
            precision = np.linalg.pinv(covariance + 1e-4 * np.eye(len(var_names)))
            optimal_alpha = 1e-4
    else:
        # Robust fallback: regularized empirical precision
        covariance = np.cov(X_std, rowvar=False)
        reg_id = 1e-3 * np.eye(len(var_names))
        precision = np.linalg.pinv(covariance + reg_id)
        # Soft threshold small off-diagonal elements for sparsity
        diag_p = np.diag(precision)
        mask = np.abs(precision) < 0.05 * np.sqrt(np.outer(diag_p, diag_p))
        precision[mask] = 0.0
        np.fill_diagonal(precision, diag_p)
        optimal_alpha = 1e-3
    
    # Convert precision matrix to partial correlations:
    # rho_ij = -P_ij / sqrt(P_ii * P_jj)
    d = np.sqrt(np.abs(np.diag(precision)))
    d[d == 0] = 1.0
    partial_corr = -precision / np.outer(d, d)
    np.fill_diagonal(partial_corr, 1.0)
    
    # Extract significant edges (non-zero entries in precision matrix)
    target_idx = var_names.index(target_var) if target_var in var_names else 0
    significant_edges = []
    for i, name in enumerate(var_names):
        if i == target_idx:
            continue
        pc = partial_corr[i, target_idx]
        if abs(precision[i, target_idx]) > 1e-6:
            significant_edges.append((name, target_var, float(pc)))
    
    return {
        "precision_matrix": precision,
        "covariance_matrix": covariance,
        "partial_corr_matrix": partial_corr,
        "var_names": var_names,
        "optimal_alpha": optimal_alpha,
        "significant_edges": significant_edges,
    }


def cross_validate_edges(
    granger_edges: List[Dict[str, Any]],
    precision_result: Dict[str, Any],
    target_var: str = "Vol_Innovations",
) -> Dict[str, Any]:
    """Cross-validate Granger causal edges against precision matrix edges.
    
    Edges present in BOTH the Granger DAG and the precision matrix
    represent high-confidence structural channels.
    
    Args:
        granger_edges: List of significant Granger edges from pathways analysis.
        precision_result: Output from estimate_precision_matrix().
        target_var: Target variable name.
    
    Returns:
        Dictionary containing:
            - 'granger_only': edges found only in Granger DAG
            - 'precision_only': edges found only in precision matrix
            - 'doubly_validated': edges found in BOTH (high confidence)
            - 'summary_df': DataFrame comparing all edges
    """
    granger_sources = set(e["source"] for e in granger_edges)
    precision_sources = set(e[0] for e in precision_result["significant_edges"])
    
    doubly_validated = granger_sources & precision_sources
    granger_only = granger_sources - precision_sources
    precision_only = precision_sources - granger_sources
    
    all_sources = granger_sources | precision_sources
    rows = []
    for src in sorted(all_sources):
        in_granger = src in granger_sources
        in_precision = src in precision_sources
        
        granger_beta = 0.0
        granger_p = 1.0
        for e in granger_edges:
            if e["source"] == src:
                if e["p_value"] < granger_p:
                    granger_beta = e["beta"]
                    granger_p = e["p_value"]
        
        partial_corr = 0.0
        for e in precision_result["significant_edges"]:
            if e[0] == src:
                partial_corr = e[2]
                break
        
        rows.append({
            "Source": src,
            "In_Granger_DAG": in_granger,
            "In_Precision_Matrix": in_precision,
            "Doubly_Validated": src in doubly_validated,
            "Granger_Beta": granger_beta,
            "Granger_P": granger_p,
            "Partial_Correlation": partial_corr,
        })
    
    summary_df = pd.DataFrame(rows)
    
    return {
        "granger_only": list(granger_only),
        "precision_only": list(precision_only),
        "doubly_validated": list(doubly_validated),
        "summary_df": summary_df,
    }
