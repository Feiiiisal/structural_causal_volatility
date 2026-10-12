"""
Causal Pathways & Graph Extraction Layer.

Extracts statistically significant directed causal edges (p < alpha) from
the output tensors of the structural causal discovery engine, and formats
the discovered pathways for macroeconomic interpretation and multiplier weighting.
"""

from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional
import numpy as np


@dataclass(frozen=True)
class CausalEdge:
    """Represents an active directed causal edge in the structural DAG."""

    source: str
    target: str
    lag: int
    beta: float
    p_value: float

    @property
    def p_val(self) -> float:
        """Alias for p_value for backward compatibility."""
        return self.p_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert edge to dictionary representation."""
        return {
            "source": self.source,
            "target": self.target,
            "lag": self.lag,
            "beta": self.beta,
            "p_value": self.p_value,
            "p_val": self.p_value,
        }


def analyze_causal_pathways(
    outputs: Dict[str, Any],
    target_var: str = "Vol_Innovations",
    alpha_thresh: float = 0.05,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    """
    Extract active directed causal edges where p < alpha_thresh.

    Args:
        outputs: Result dictionary returned by execute_structural_causal_discovery.
        target_var: Name of the target node (default: 'Vol_Innovations').
        alpha_thresh: Significance threshold for edge retention (default: 0.05).
        verbose: Whether to print formatted discovery table to stdout.

    Returns:
        List of active edge dictionaries containing:
            {'source', 'target', 'lag', 'beta', 'p_value', 'p_val'}
    """
    var_names: List[str] = outputs["var_names"]
    p_matrix: np.ndarray = outputs["p_matrix"]
    val_matrix: np.ndarray = outputs["val_matrix"]
    tau_max: int = outputs.get("tau_max", 5)

    if target_var in var_names:
        target_idx = var_names.index(target_var)
    else:
        # Fall back to target specified in discovery output or index 0
        recorded_target = outputs.get("target_var", var_names[0])
        target_idx = var_names.index(recorded_target) if recorded_target in var_names else 0
        target_var = var_names[target_idx]

    edges: List[Dict[str, Any]] = []

    for source_idx, source_name in enumerate(var_names):
        # Exclude self-autoregressive edge from external causal pathway extraction
        if source_name == target_var:
            continue

        for tau in range(1, tau_max + 1):
            p_val = float(p_matrix[source_idx, target_idx, tau])
            beta = float(val_matrix[source_idx, target_idx, tau])

            if p_val < alpha_thresh:
                edge = CausalEdge(
                    source=source_name,
                    target=target_var,
                    lag=tau,
                    beta=beta,
                    p_value=p_val,
                )
                edges.append(edge.to_dict())

    if verbose:
        print_pathways_table(edges, target_var)

    return edges


def print_pathways_table(edges: List[Dict[str, Any]], target_var: str = "Vol_Innovations") -> None:
    """Format and print causal parents summary matching baseline output."""
    print(f"\n🔍 CAUSAL PARENTS DISCOVERED FOR {target_var.upper()}:")
    print("=" * 85)
    print(f"{'Causal Parent Node':<25} | {'Time Lag (Tau)':<15} | {'Path Coefficient':<20} | {'P-Value'}")
    print("=" * 85)

    if edges:
        for e in edges:
            lag_str = f"t - {e['lag']} days"
            print(f"{e['source']:<25} | {lag_str:<15} | {e['beta']:<20.5f} | {e['p_value']:.5e}")
    else:
        print("⚠️ No significant causal pathways found at specified alpha threshold.")

    print("=" * 85)


class CausalGraph:
    """
    Container representing the structural causal DAG and its active channels.
    """

    def __init__(
        self,
        edges: List[Dict[str, Any]],
        var_names: List[str],
        target_var: str = "Vol_Innovations",
        alpha_thresh: float = 0.05,
    ):
        self.edges = edges
        self.var_names = var_names
        self.target_var = target_var
        self.alpha_thresh = alpha_thresh

    @classmethod
    def from_discovery(
        cls,
        outputs: Dict[str, Any],
        target_var: str = "Vol_Innovations",
        alpha_thresh: float = 0.05,
        verbose: bool = False,
    ) -> "CausalGraph":
        """Construct CausalGraph directly from discovery engine output."""
        edges = analyze_causal_pathways(
            outputs=outputs,
            target_var=target_var,
            alpha_thresh=alpha_thresh,
            verbose=verbose,
        )
        return cls(
            edges=edges,
            var_names=outputs.get("var_names", []),
            target_var=target_var,
            alpha_thresh=alpha_thresh,
        )

    @property
    def active_channels(self) -> int:
        """Total number of statistically significant causal edges."""
        return len(self.edges)

    @property
    def active_drivers(self) -> List[str]:
        """Unique list of macroeconomic driver variable names with active edges."""
        return sorted(list({e["source"] for e in self.edges}))

    def summary(self) -> str:
        """Generate human-readable summary string of the causal graph."""
        lines = [
            f"CausalGraph (target={self.target_var}, alpha={self.alpha_thresh})",
            f"  Active channels: {self.active_channels}",
            f"  Active drivers: {', '.join(self.active_drivers) if self.active_drivers else 'None'}",
        ]
        for e in self.edges:
            lines.append(f"  - {e['source']} (lag {e['lag']}) -> {e['target']}: beta={e['beta']:.5f}, p={e['p_value']:.3e}")
        return "\n".join(lines)

    def print_table(self) -> None:
        """Print formatted discovery table to stdout."""
        print_pathways_table(self.edges, self.target_var)

    def __len__(self) -> int:
        return len(self.edges)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.edges[idx]

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        return iter(self.edges)

    def __repr__(self) -> str:
        return f"<CausalGraph: {self.active_channels} channels, drivers={self.active_drivers}>"
