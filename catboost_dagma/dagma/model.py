"""
Deep DYNOTEARS MLP Architecture with Student-t Log-Likelihood Score Function.
Implements non-linear contemporaneous and lagged causal discovery with heavy-tailed loss.
"""

import math
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Safely converts a PyTorch tensor to a NumPy array across environments."""
    try:
        return tensor.detach().cpu().numpy()
    except RuntimeError:
        return np.array(tensor.detach().cpu().tolist())


class DeepDynotearsMLP(nn.Module):
    """
    Non-linear Multi-Layer Perceptron for Time-Series Causal Discovery (DYNOTEARS style).
    - fc1_W: contemporaneous (intra-slice) connections, strictly acyclic via log-det penalty.
    - fc1_A: lagged (inter-slice) connections, acyclic by temporal ordering.
    """

    def __init__(
        self,
        d_vars: int,
        p_orders: int = 1,
        hidden_dim: int = 16,
        init_nu: float = 3.0,
        dtype=torch.float64,
    ):
        super().__init__()
        self.d = d_vars
        self.p = p_orders
        self.hidden_dim = hidden_dim
        self.dtype = dtype

        # First layer weights
        self.fc1_W = nn.Parameter(torch.zeros(d_vars, d_vars, hidden_dim, dtype=dtype))
        self.fc1_A = nn.Parameter(torch.zeros(d_vars, p_orders * d_vars, hidden_dim, dtype=dtype))
        self.bias1 = nn.Parameter(torch.zeros(d_vars, hidden_dim, dtype=dtype))

        # Second layer weights
        self.fc2 = nn.Parameter(torch.zeros(d_vars, hidden_dim, 1, dtype=dtype))
        self.bias2 = nn.Parameter(torch.zeros(d_vars, 1, dtype=dtype))

        # Learnable Student-t degrees of freedom nu (re-parameterized as nu = exp(log_nu) + 2.0 to ensure finite variance)
        self.log_nu = nn.Parameter(torch.tensor(math.log(max(init_nu - 2.0, 1e-4)), dtype=dtype))

        self.reset_parameters()

    def reset_parameters(self):
        """Initializes weights with Xavier uniform, zeros diagonal of fc1_W (no self-loops)."""
        nn.init.xavier_uniform_(self.fc1_W)
        nn.init.xavier_uniform_(self.fc1_A)
        nn.init.xavier_uniform_(self.fc2)
        with torch.no_grad():
            for i in range(self.d):
                self.fc1_W[i, i, :] = 0.0

    def forward(self, X: torch.Tensor, Xlags: torch.Tensor) -> torch.Tensor:
        """
        Forward pass predicting X_{t} given contemporaneous X_{t} and lagged X_{t-1}.
        Args:
            X: Contemporaneous matrix of shape (n_samples, d_vars)
            Xlags: Lagged matrix of shape (n_samples, p_orders * d_vars)
        Returns:
            X_hat: Predicted contemporaneous matrix of shape (n_samples, d_vars)
        """
        # Zero out diagonal of fc1_W so variable i cannot predict itself contemporaneously
        mask = 1.0 - torch.eye(self.d, device=X.device, dtype=self.dtype).unsqueeze(-1)
        fc1_W_masked = self.fc1_W * mask

        # Non-linear activations
        h1_W = torch.einsum("ni,ijh->njh", X, fc1_W_masked)
        h1_A = torch.einsum("nl,ljh->njh", Xlags, self.fc1_A)
        h1 = torch.relu(h1_W + h1_A + self.bias1.unsqueeze(0))

        # Output projection
        X_hat = torch.einsum("njh,jho->njo", h1, self.fc2).squeeze(-1)
        X_hat = X_hat + self.bias2.squeeze(-1).unsqueeze(0)
        return X_hat

    def get_W_adj(self) -> torch.Tensor:
        """
        Extracts directed contemporaneous adjacency matrix W.
        W_{ij} = L2-norm of the hidden unit weights connecting node i to node j.
        """
        mask = 1.0 - torch.eye(self.d, device=self.fc1_W.device, dtype=self.dtype).unsqueeze(-1)
        fc1_W_masked = self.fc1_W * mask
        return torch.sqrt(torch.sum(fc1_W_masked ** 2, dim=2) + 1e-12)

    def get_A_adj(self) -> torch.Tensor:
        """
        Extracts directed lagged adjacency matrix A.
        """
        return torch.sqrt(torch.sum(self.fc1_A ** 2, dim=2) + 1e-12)

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        """
        DAGMA exact log-det acyclicity constraint for contemporaneous block W:
        h^s(W) = -logdet(sI - W o W) + d*log(s)
        
        Evaluated on M = W o W (element-wise squared weights).
        If iterate leaves positive-definite cone, slogdet sign becomes <= 0 or h becomes negative.
        """
        W_adj = self.get_W_adj()
        M = W_adj * W_adj
        I = torch.eye(self.d, device=M.device, dtype=self.dtype)
        diff = s * I - M
        sign, logabsdet = torch.linalg.slogdet(diff)
        if sign.item() <= 0:
            # Not positive definite - return negative barrier value to signal domain violation
            return torch.tensor(-1.0, dtype=self.dtype, device=M.device)
        h = -logabsdet + self.d * math.log(s)
        return h

    def fc1_l1_reg(self) -> torch.Tensor:
        """L1 group sparsity penalty across contemporaneous and lagged weights."""
        return torch.sum(self.get_W_adj()) + torch.sum(self.get_A_adj())

    def get_degrees_of_freedom(self) -> float:
        """Returns the current estimated Student-t degrees of freedom nu."""
        with torch.no_grad():
            return float(torch.exp(self.log_nu).item() + 2.0)


def compute_student_t_loss(
    model: DeepDynotearsMLP,
    X: torch.Tensor,
    Xlags: torch.Tensor,
    learnable_nu: bool = True,
) -> torch.Tensor:
    """
    Computes Student-t negative log-likelihood loss for heavy-tailed return distributions:
    L(X, X_hat; nu) = ((nu + 1) / 2n) * sum_{i,j} log(1 + (X_{ij} - X_hat_{ij})^2 / nu)
    Protects against fat-tailed outlier distortion common in financial return shocks.
    """
    X_hat = model(X, Xlags)
    n = X.shape[0]
    if learnable_nu:
        nu = torch.exp(model.log_nu) + 2.0
    else:
        nu = torch.tensor(3.0, dtype=X.dtype, device=X.device)

    residuals_sq = (X - X_hat) ** 2
    loss = ((nu + 1.0) / (2.0 * n)) * torch.sum(torch.log(1.0 + residuals_sq / nu))
    return loss


def compute_gaussian_loss(
    model: DeepDynotearsMLP,
    X: torch.Tensor,
    Xlags: torch.Tensor,
) -> torch.Tensor:
    """Standard Gaussian MSE loss (vanilla DAGMA baseline for ablation)."""
    X_hat = model(X, Xlags)
    n = X.shape[0]
    return (0.5 / n) * torch.sum((X - X_hat) ** 2)
