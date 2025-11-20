"""
Probe model definitions as PyTorch nn.Module classes.
Each probe is a standard class with __init__ and forward methods.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn


class LinearProbe(nn.Module):
    """
    Linear probe: input_dim → 1 (no hidden layer)
    Direct mapping from embeddings to binary output.
    """
    
    def __init__(self, input_dim: int, **kwargs):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLPProbe_1x200(nn.Module):
    """
    One-hidden-layer MLP probe: input_dim → 200 → 1
    ReLU activation, no dropout.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 200, **kwargs):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.relu(x)
        x = self.output(x)
        return x


# Registry mapping probe names to their classes
PROBE_REGISTRY: Dict[str, type[nn.Module]] = {
    "linear": LinearProbe,
    "mlp_1x200": MLPProbe_1x200,
}


def get_probe_class(name: str) -> type[nn.Module]:
    """Retrieve a probe class by name."""
    if name not in PROBE_REGISTRY:
        available = ", ".join(PROBE_REGISTRY.keys())
        raise ValueError(f"Unknown probe '{name}'. Available: {available}")
    return PROBE_REGISTRY[name]


def list_available_probes() -> list[str]:
    """Return all registered probe names."""
    return sorted(PROBE_REGISTRY.keys())


def register_probe(name: str, probe_class: type[nn.Module]) -> None:
    """Register a custom probe class."""
    PROBE_REGISTRY[name] = probe_class


def build_probe(name: str, input_dim: int, **kwargs) -> nn.Module:
    """
    Build a probe instance by name.
    
    Args:
        name: Probe architecture name
        input_dim: Input feature dimension
        **kwargs: Additional arguments passed to probe constructor
    
    Returns:
        Initialized probe module
    """
    probe_class = get_probe_class(name)
    return probe_class(input_dim=input_dim, **kwargs)
