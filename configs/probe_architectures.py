"""
Probe architecture registry and helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Union

import torch.nn as nn


ActivationLike = Union[str, Callable[[], nn.Module]]
DropoutLike = Union[float, Sequence[float]]
HiddenDims = Sequence[int]


def _activation_from_name(name: str) -> nn.Module:
    """Map string names to activation modules."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "gelu":
        return nn.GELU()
    if name == "sigmoid":
        return nn.Sigmoid()
    if name == "selu":
        return nn.SELU()
    raise ValueError(f"Unsupported activation: {name}")


def _ensure_sequence(value: Union[ActivationLike, Sequence[ActivationLike]], n: int) -> List[ActivationLike]:
    """Broadcast a scalar-like value to a list of length n."""
    if isinstance(value, (list, tuple)):
        if len(value) != n:
            raise ValueError(f"Expected {n} values, received {len(value)}")
        return list(value)
    return [value for _ in range(n)]


@dataclass(frozen=True)
class ProbeArchitectureSpec:
    """Dataclass describing a simple feed-forward probe."""

    name: str
    hidden_dims: HiddenDims
    activation: ActivationLike = "relu"
    dropout: DropoutLike = 0.0
    bias: bool = True
    layer_norm: bool = False
    batch_norm: bool = False

    def build(self, input_dim: int, output_dim: int = 1) -> nn.Module:
        """Construct an nn.Module based on the spec."""
        layers: List[nn.Module] = []
        prev_dim = input_dim

        act_fns = _ensure_sequence(self.activation, len(self.hidden_dims)) if self.hidden_dims else []
        drop_values = _ensure_sequence(self.dropout, len(self.hidden_dims)) if self.hidden_dims else []

        for idx, hidden_dim in enumerate(self.hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim, bias=self.bias))
            if self.batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if self.layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))

            activation = act_fns[idx]
            layers.append(activation() if callable(activation) else _activation_from_name(str(activation)))

            drop_rate = drop_values[idx]
            if drop_rate:
                layers.append(nn.Dropout(drop_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim, bias=self.bias))
        return nn.Sequential(*layers)


PROBE_ARCHITECTURES: Dict[str, ProbeArchitectureSpec] = {
    "mlp_1x200": ProbeArchitectureSpec(
        name="mlp_1x200",
        hidden_dims=(200,),
        activation="relu",
        dropout=0.0,
    ),
    "mlp_2x512": ProbeArchitectureSpec(
        name="mlp_2x512",
        hidden_dims=(512, 512),
        activation="gelu",
        dropout=(0.1, 0.1),
    ),
    "linear": ProbeArchitectureSpec(
        name="linear",
        hidden_dims=(),
        activation="relu",
        dropout=0.0,
    ),
}


def list_probe_architectures() -> List[str]:
    """Return all registered probe architecture identifiers."""
    return sorted(PROBE_ARCHITECTURES.keys())


def get_probe_architecture(name: str) -> ProbeArchitectureSpec:
    """Fetch a registered probe architecture."""
    if name not in PROBE_ARCHITECTURES:
        choices = ", ".join(list_probe_architectures())
        raise ValueError(f"Unknown probe architecture '{name}'. Available: {choices}")
    return PROBE_ARCHITECTURES[name]


def register_custom_architecture(spec: ProbeArchitectureSpec) -> None:
    """Register a custom architecture at runtime."""
    PROBE_ARCHITECTURES[spec.name] = spec

```