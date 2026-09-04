"""
Adversary package: the global passive adversary (GPA), its observation views and
the attacks it runs. `ATTACKS` is the name -> AttackSpec registry used by the
experiments framework; the measures that grade a guess live in `metrics`.
"""

from pipeline_contract import AttackSpec, POSTERIOR, SCALAR

from .gpa import observe_broadcasts, observe_sender_sets
from .stake_inference import (
    SetStakeInferenceParams,
    estimate_stake_from_sets,
    run_set_stake_inference,
)
from .bayes_attribution import (
    BayesAttributionParams,
    run as run_bayes_attribution,
)
from .mixnet_attribution import (
    MixnetAttributionParams,
    run as run_mixnet_attribution,
)
from .mixnet_attribution_oracle import (
    run as run_mixnet_attribution_oracle,
)

ATTACKS = {
    "set_stake_inference": AttackSpec(run=run_set_stake_inference, produces=SCALAR,
                                      params_cls=SetStakeInferenceParams,
                                      knows=("f", "p_s", "T", "N")),
    "bayes_attribution": AttackSpec(run=run_bayes_attribution, produces=POSTERIOR,
                                    params_cls=BayesAttributionParams,
                                    knows=("hops", "mix_scale", "sender_scale",
                                           "receiver_delays", "latency_profile")),
    "mixnet_attribution": AttackSpec(run=run_mixnet_attribution, produces=POSTERIOR,
                                     params_cls=MixnetAttributionParams,
                                     knows=("hops", "mix_scale", "sender_scale",
                                            "receiver_delays", "latency_profile")),
    "mixnet_attribution_oracle": AttackSpec(run=run_mixnet_attribution_oracle, produces=POSTERIOR,
                                            params_cls=MixnetAttributionParams,
                                            knows=("hops", "mix_scale", "sender_scale",
                                                   "receiver_delays", "latency_profile")),
}

__all__ = [
    "observe_sender_sets",
    "observe_broadcasts",
    "estimate_stake_from_sets",
    "run_set_stake_inference",
    "SetStakeInferenceParams",
    "run_bayes_attribution",
    "BayesAttributionParams",
    "run_mixnet_attribution",
    "run_mixnet_attribution_oracle",
    "MixnetAttributionParams",
    "ATTACKS",
]
