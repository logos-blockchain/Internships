"""
Anonymisation layers: the defence stage inserted between the system (consensus +
network) and the attacker. Every layer shares the signature
apply(slots, nodes, *, params, latency_oracle, rng) -> Trace and is selected by
name through the `LAYERS` registry (mirroring adversary.ATTACKS).
"""

from .trace import BROADCAST, ENTRY, EXIT, Trace, make_trace, passthrough
from .dummies import DummyParams, inject_dummies
from .single_path_mix import (
    SinglePathMixParams,
    apply as single_path_mix_apply,
    delay_moments,
    gamma_sum_pdf,
    random_delay_pdf,
    residual_delay_pdf,
    residual_moments,
)
from .mixnet import (
    MixnetParams,
    apply as mixnet_apply,
)

LAYERS = {
    "none": passthrough,
    "single_path_mix": single_path_mix_apply,
    "mixnet": mixnet_apply,
}

__all__ = [
    "Trace",
    "make_trace",
    "ENTRY",
    "EXIT",
    "BROADCAST",
    "passthrough",
    "LAYERS",
    "DummyParams",
    "inject_dummies",
    "SinglePathMixParams",
    "MixnetParams",
    "delay_moments",
    "random_delay_pdf",
    "residual_delay_pdf",
    "residual_moments",
    "gamma_sum_pdf",
]
