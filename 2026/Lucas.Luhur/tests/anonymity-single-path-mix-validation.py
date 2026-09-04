"""
Validation of the single_path_mix layer (shared-path routing + k-hop mixing delay).

Checks, on real consensus traffic with injected dummies:
  1. Schema -- M emissions -> 2M rows, M ENTRY + M EXIT, broadcast_id = batch group.
  2. Shared-path structure -- entry at the source, random destination, M reals batched by slot.
  3. Intentional delay -- exit - entry = Z = X_S + mixing > 0, mean sender_scale + k*mix_scale.
  4. Anonymity unlocked -- none gives certain links; single_path_mix hides each real in D+1.
  5. Registry contract; 6. entry = slot, X_S folded into Z, receiver_delays -> k+1 stages.
  7. mu wiring -- the latency profile adds mu = d_i^S + D_M + d_r^R at the exit.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from consensus import (  # noqa: E402
    DEFAULT_F,
    sample_relative_stakes,
    simulate_events,
)
from anonymity import (  # noqa: E402
    LAYERS,
    DummyParams,
    Trace,
    SinglePathMixParams,
    inject_dummies,
    passthrough,
)
from anonymity.single_path_mix import apply as single_path_mix_apply  # noqa: E402
from network import LatencyProfileParams, sample_latency_profile  # noqa: E402


def _check(name, ok, detail=""):
    """Print a PASS/FAIL line for one check and return its boolean result."""
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""))
    return bool(ok)


def main():
    """Run the single_path_mix validation checks; return 0 if all pass."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--N", type=int, default=1000)
    ap.add_argument("--T", type=int, default=40_000)
    ap.add_argument("--shape", type=float, default=1.33)
    ap.add_argument("--f", type=float, default=DEFAULT_F)
    ap.add_argument("--D", type=int, default=2, help="dummies per real winner")
    ap.add_argument("--hops", type=int, default=3)
    ap.add_argument("--mix-scale", type=float, default=1.0)
    ap.add_argument("--window", type=int, default=3, help="dummy slot jitter (spreads entries)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    alpha = sample_relative_stakes(args.N, args.shape, rng=rng)
    slots, nodes = simulate_events(alpha, f=args.f, T=args.T, rng=rng)
    dp = DummyParams(count=args.D, window=args.window)
    s, n, is_dummy, grp = inject_dummies(slots, nodes, args.N, params=dp, T=args.T, rng=rng)
    m = s.size
    M = slots.size
    print(f"consensus: {M} winners; +dummies D={args.D} -> {m} emissions; "
          f"k={args.hops}, mix_scale={args.mix_scale}\n")

    params = SinglePathMixParams(hops=args.hops, mix_scale=args.mix_scale, n_nodes=args.N)
    trace = single_path_mix_apply(s, n, is_dummy, grp, params=params, rng=rng)
    ok = True

    en, ex = trace.is_entry, trace.is_exit
    dtypes_ok = (trace.broadcast_id.dtype == np.int64 and trace.obs_time.dtype == np.float64
                 and trace.kind.dtype == np.int8 and trace.is_dummy.dtype == bool)
    bid_ok = (np.array_equal(trace.broadcast_id[en], grp)
              and np.array_equal(trace.broadcast_id[ex], grp))
    ok &= _check("schema well-formedness",
                 len(trace) == 2 * m and int(en.sum()) == m and int(ex.sum()) == m
                 and dtypes_ok and bid_ok,
                 f"2M={2 * m} rows, broadcast_id = batch group")

    entry_at_source = np.array_equal(trace.obs_node[en], trace.true_source[en])
    n_unique_exits = int(np.unique(trace.obs_node[ex]).size)
    exit_random = n_unique_exits > 0.5 * min(m, args.N)
    real_entries = int((~trace.is_dummy[en]).sum())
    reals_ok = real_entries == M and np.array_equal(
        np.sort(trace.broadcast_id[en][~trace.is_dummy[en]]), np.sort(slots))
    ok &= _check("shared-path structure (entry=source, random destination/msg, M reals batched by slot)",
                 entry_at_source and exit_random and reals_ok,
                 f"{n_unique_exits} distinct destinations over {m} msgs; {real_entries} reals")

    delay = trace.obs_time[ex] - trace.obs_time[en]
    k, sc, ss = args.hops, args.mix_scale, params.sender_scale
    mean_delay, theo_mean = float(delay.mean()), ss + k * sc
    se = np.sqrt((ss * ss + k * sc * sc) / m)
    z = abs(mean_delay - theo_mean) / se
    d2 = single_path_mix_apply(s, n, is_dummy, grp, params=SinglePathMixParams(2, sc, args.N), rng=rng)
    d5 = single_path_mix_apply(s, n, is_dummy, grp, params=SinglePathMixParams(5, sc, args.N), rng=rng)
    md2 = float((d2.obs_time[d2.is_exit] - d2.obs_time[d2.is_entry]).mean())
    md5 = float((d5.obs_time[d5.is_exit] - d5.obs_time[d5.is_entry]).mean())
    ok &= _check("intentional delay Z (exit-entry>0, mean = sender_scale + k*mix_scale, grows with k)",
                 bool(np.all(delay > 0)) and z < 4.0 and md5 > md2,
                 f"mean={mean_delay:.3f} vs sender+k*scale={theo_mean:.3f} (z={z:.2f}); "
                 f"k=2 -> {md2:.2f}, k=5 -> {md5:.2f}")

    none_trace = passthrough(s, n, is_dummy, grp)
    none_certain = np.array_equal(none_trace.obs_node[none_trace.is_exit],
                                  none_trace.true_source[none_trace.is_exit])
    tor_exit_at_source = float(np.mean(trace.obs_node[ex] == trace.true_source[ex]))
    batch_size = np.bincount(trace.broadcast_id[ex])
    real_sets = batch_size[trace.broadcast_id[ex][~trace.is_dummy[ex]]]
    candidate_set = int(np.median(real_sets))
    sets_ok = bool(real_sets.min() >= args.D + 1)
    ok &= _check("anonymity unlocked (none: set=1 certain; single_path_mix: real set>=count+1)",
                 none_certain and tor_exit_at_source < 0.05 and sets_ok,
                 f"none certain-link=100%; single_path_mix exit@source={tor_exit_at_source:.1%}, "
                 f"real candidate set={candidate_set}")

    layer = LAYERS.get("single_path_mix")
    reg = layer(s, n, is_dummy, grp, params=params, latency_oracle=None, rng=rng)
    registry_ok = (layer is single_path_mix_apply and isinstance(reg, Trace) and len(reg) == 2 * m)
    ok &= _check("registry contract (LAYERS['single_path_mix'] -> Trace)", registry_ok)

    hp, sc2 = args.hops, args.mix_scale
    ss = 2.0
    t_hold = single_path_mix_apply(s, n, is_dummy, grp,
                            params=SinglePathMixParams(hp, sc2, args.N, sender_scale=ss),
                            rng=np.random.default_rng(args.seed))
    entry_is_slot = np.array_equal(t_hold.obs_time[t_hold.is_entry], s.astype(np.float64))
    t_no = single_path_mix_apply(s, n, is_dummy, grp,
                            params=SinglePathMixParams(hp, sc2, args.N, sender_scale=0.0),
                            rng=np.random.default_rng(args.seed))
    d_hold = float((t_hold.obs_time[t_hold.is_exit] - t_hold.obs_time[t_hold.is_entry]).mean())
    d_no = float((t_no.obs_time[t_no.is_exit] - t_no.obs_time[t_no.is_entry]).mean())
    hold_ok = entry_is_slot and abs((d_hold - d_no) - ss) < 0.1 * ss
    ss_def = SinglePathMixParams().sender_scale
    t_rx = single_path_mix_apply(s, n, is_dummy, grp,
                          params=SinglePathMixParams(hp, sc2, args.N, receiver_delays=True),
                          rng=np.random.default_rng(args.seed))
    rxd = t_rx.obs_time[t_rx.is_exit] - t_rx.obs_time[t_rx.is_entry]
    rx_theo = ss_def + (hp + 1) * sc2
    rx_z = abs(rxd.mean() - rx_theo) / np.sqrt((ss_def ** 2 + (hp + 1) * sc2 * sc2) / m)
    ok &= _check("delay-model semantics (entry=slot; X_S in Z; receiver_delays -> k+1 mix)",
                 hold_ok and rx_z < 4.0,
                 f"entry==slot={entry_is_slot}; hold delta={d_hold - d_no:.3f} (~{ss}); "
                 f"rx Z mean={rxd.mean():.3f} vs {rx_theo:.2f} (z={rx_z:.2f})")

    D = 0.3
    prof_h = sample_latency_profile(args.N, hp, D, rng=np.random.default_rng(args.seed))
    t_mu = single_path_mix_apply(s, n, is_dummy, grp,
                            params=SinglePathMixParams(hp, sc2, args.N, sender_scale=0.0),
                            latency_oracle=prof_h, rng=np.random.default_rng(args.seed))
    t_nomu = single_path_mix_apply(s, n, is_dummy, grp,
                            params=SinglePathMixParams(hp, sc2, args.N, sender_scale=0.0),
                            latency_oracle=None, rng=np.random.default_rng(args.seed))
    dmu = float((t_mu.obs_time[t_mu.is_exit] - t_mu.obs_time[t_mu.is_entry]).mean())
    dnomu = float((t_nomu.obs_time[t_nomu.is_exit] - t_nomu.obs_time[t_nomu.is_entry]).mean())
    homog_ok = abs((dmu - dnomu) - (hp + 1) * D) < 1e-9
    prof_het = sample_latency_profile(args.N, hp, D,
                                      LatencyProfileParams(sender_low=0.1, sender_high=1.0),
                                      rng=np.random.default_rng(args.seed + 1))
    mu_a = float(prof_het.mu(np.array([0]), np.array([0]))[0])
    mu_b = float(prof_het.mu(np.array([1]), np.array([0]))[0])
    het_ok = mu_a != mu_b
    ok &= _check("mu wiring (homogeneous profile adds (k+1)d; heterogeneous d_i^S varies per sender)",
                 homog_ok and het_ok,
                 f"homog delta={dmu - dnomu:.4f} vs (k+1)d={(hp + 1) * D:.4f}; "
                 f"het mu[sender0]={mu_a:.3f} != mu[sender1]={mu_b:.3f}")

    print("\n" + ("All checks passed." if ok else "SOME CHECKS FAILED."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
