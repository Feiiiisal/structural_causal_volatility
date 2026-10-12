"""
Causal Discovery & Dynamic Risk Multiplier Module.

Provides bivariate Granger structural causal discovery, causal graph pathway extraction,
and dynamic adaptive volatility multipliers.
"""

from causal_volatility.causal.discovery import (
    StructuralCausalDiscovery,
    execute_structural_causal_discovery,
)
from causal_volatility.causal.multiplier import (
    CausalMultiplier,
    construct_causal_multiplier,
)
from causal_volatility.causal.pathways import (
    CausalEdge,
    CausalGraph,
    analyze_causal_pathways,
)

__all__ = [
    "execute_structural_causal_discovery",
    "StructuralCausalDiscovery",
    "analyze_causal_pathways",
    "CausalGraph",
    "CausalEdge",
    "construct_causal_multiplier",
    "CausalMultiplier",
]
