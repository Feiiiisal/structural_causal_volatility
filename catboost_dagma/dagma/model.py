"""DAGMA model used by the rolling network fit."""
import math
import numpy as np
import torch
import torch.nn as nn


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


class DeepDynotearsMLP(nn.Module):
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

        self.fc1_W = nn.Parameter(torch.zeros(d_vars, d_vars, hidden_dim, dtype=dtype))
        self.fc1_A = nn.Parameter(torch.zeros(d_vars, p_orders * d_vars, hidden_dim, dtype=dtype))
        self.bias1 = nn.Parameter(torch.zeros(d_vars, hidden_dim, dtype=dtype))
        self.fc2 = nn.Parameter(torch.zeros(d_vars, hidden_dim, 1, dtype=dtype))
        self.bias2 = nn.Parameter(torch.zeros(d_vars, 1, dtype=dtype))

        start_nu = max(init_nu - 2.0, 1e-4)
        self.log_nu = nn.Parameter(torch.tensor(math.log(start_nu), dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc1_W)
        nn.init.xavier_uniform_(self.fc1_A)
        nn.init.xavier_uniform_(self.fc2)

        with torch.no_grad():
            for i in range(self.d):
                self.fc1_W[i, i, :] = 0.0

    def _masked_W(self):
        mask = 1.0 - torch.eye(
            self.d, device=self.fc1_W.device, dtype=self.dtype
        ).unsqueeze(-1)
        return self.fc1_W * mask

    def forward(self, X: torch.Tensor, Xlags: torch.Tensor) -> torch.Tensor:
        h_now = torch.einsum("ni,ijh->njh", X, self._masked_W())
        h_lag = torch.einsum("nl,ljh->njh", Xlags, self.fc1_A)
        hidden = torch.relu(h_now + h_lag + self.bias1.unsqueeze(0))
        out = torch.einsum("njh,jho->njo", hidden, self.fc2).squeeze(-1)
        return out + self.bias2.squeeze(-1).unsqueeze(0)

    def get_W_adj(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self._masked_W() ** 2, dim=2) + 1e-12)

    def get_A_adj(self) -> torch.Tensor:
        return torch.sqrt(torch.sum(self.fc1_A ** 2, dim=2) + 1e-12)

    def h_func(self, s: float = 1.0) -> torch.Tensor:
        W = self.get_W_adj()
        M = W * W
        eye = torch.eye(self.d, device=M.device, dtype=self.dtype)
        sign, logdet = torch.linalg.slogdet(s * eye - M)

        if sign.item() <= 0:
            return torch.tensor(-1.0, dtype=self.dtype, device=M.device)

        return -logdet + self.d * math.log(s)

    def fc1_l1_reg(self) -> torch.Tensor:
        return self.get_W_adj().sum() + self.get_A_adj().sum()

    def get_degrees_of_freedom(self) -> float:
        with torch.no_grad():
            return float(torch.exp(self.log_nu).item() + 2.0)


def compute_student_t_loss(model, X, Xlags, learnable_nu=True):
    pred = model(X, Xlags)
    nu = torch.exp(model.log_nu) + 2.0 if learnable_nu else torch.tensor(
        3.0, dtype=X.dtype, device=X.device
    )
    error2 = (X - pred) ** 2
    return ((nu + 1.0) / (2.0 * X.shape[0])) * torch.log1p(error2 / nu).sum()


def compute_gaussian_loss(model, X, Xlags):
    pred = model(X, Xlags)
    return 0.5 * torch.square(X - pred).sum() / X.shape[0]
