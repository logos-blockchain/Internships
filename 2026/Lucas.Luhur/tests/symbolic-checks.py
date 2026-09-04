"""
Symbolic (SymPy) re-derivation of the consensus and network closed forms.

Consensus checks 1-9: the lottery survival (1-f)^a, E[n] = sum phi, the epoch total
T * sum phi, the variance phi(1-phi), occupancy P(n>=1) = f, T*f filled slots, the fork
probability P(n>=2) and its f^2 order, and the Pareto Gini G = 1/(2k-1). Network checks
N1-N3: the edge-latency transform, the deterministic limit alpha = log(C-1)/d of the
propagation exponent, and the Lambert-W broadcast minimiser u*.
"""

from __future__ import annotations

import sympy as sp

f = sp.symbols("f", positive=True)
a, b, c = sp.symbols("a b c", positive=True)
p = sp.symbols("p")
p1, p2 = sp.symbols("p1 p2", positive=True)
k = sp.symbols("k", positive=True)
u = sp.symbols("u", positive=True)
t = sp.symbols("t", integer=True)
T = sp.symbols("T", positive=True, integer=True)

C = sp.symbols("C", positive=True)
d = sp.symbols("d", positive=True)
lam = sp.symbols("lambda", positive=True)
alpha = sp.symbols("alpha", positive=True)
y = sp.symbols("y", positive=True)
w = sp.symbols("w")


def phi(stake):
    """The leader-election lottery phi(alpha) = 1 - (1-f)^alpha."""
    return 1 - (1 - f) ** stake


def is_zero(expr) -> bool:
    """Robustly decide whether a symbolic expression is identically zero.

    Forced powsimp is needed because the survivals carry symbolic exponents
    (e.g. (1-f)^a * (1-f)^(1-a) only collapses to (1-f) under force=True).
    """
    forms = [expr]
    try:
        forms.append(sp.powsimp(expr, force=True))
        forms.append(sp.powsimp(sp.together(expr), force=True))
    except Exception:
        pass
    for e in forms:
        s = sp.simplify(e)
        if s == 0 or s.is_zero:
            return True
    eq = sp.powsimp(sp.expand(expr), force=True).equals(0)
    return bool(eq) if eq is not None else False


_results: list[bool] = []


def check(name, ok):
    """Record and print the PASS/FAIL result of one named check."""
    _results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    return ok


def main():
    """Run every symbolic check; return 0 if all pass."""
    print("=" * 72)
    print("Symbolic checks -- consensus & network closed forms re-derived in SymPy")
    print("=" * 72 + "\n")
    print("-- consensus closed forms --")

    check("1. lottery", is_zero((1 - phi(a)) - (1 - f) ** a))

    e_n = sum(
        (p1 ** s1 * (1 - p1) ** (1 - s1)) * (p2 ** s2 * (1 - p2) ** (1 - s2)) * (s1 + s2)
        for s1 in (0, 1)
        for s2 in (0, 1)
    )
    check("2. expected_winners_per_slot", is_zero(sp.expand(e_n) - (p1 + p2)))

    mu = p1 + p2
    epoch_total = sp.Sum(mu, (t, 1, T)).doit()
    check("3. total messages", is_zero(epoch_total - T * (p1 + p2)))

    check("4. var_winners_per_slot", is_zero(p * (1 - p) - (p - p ** 2)))

    surv_prod = (1 - f) ** a * (1 - f) ** b * (1 - f) ** c
    collapsed = sp.powsimp(surv_prod, force=True)
    p_atleast1 = 1 - collapsed.subs(c, 1 - a - b)
    check("5. prob_at_least_one", is_zero(p_atleast1 - f))

    filled = sp.Sum(f, (t, 1, T)).doit()
    check("6. filled slots", is_zero(filled - T * f))

    # survivals s_i = (1-f)^{a_i}, s1*s2 = 1-f: keeps P(n>=2) rational for SymPy
    s1, s2 = sp.symbols("s1 s2", positive=True)
    ph1, ph2 = 1 - s1, 1 - s2
    F = 1 - s1 * s2
    p2_struct = 1 - s1 * s2 - (ph1 * s2 + ph2 * s1)
    p2_doc = F - (1 - F) * (ph1 / s1 + ph2 / s2)
    check("7. prob_at_least_two", is_zero(p2_struct - p2_doc))

    phi1, phi2 = phi(a), phi(1 - a)
    pn2 = 1 - (1 - phi1) * (1 - phi2) - (phi1 * (1 - phi2) + phi2 * (1 - phi1))
    ser = sp.series(pn2, f, 0, 3).removeO()
    coeff2 = sp.expand(ser).coeff(f, 2)
    expected = sp.Rational(1, 2) * (1 - a ** 2 - (1 - a) ** 2)
    check(
        "8. prob_at_least_two [f^2]",
        is_zero(coeff2 - expected) and ser.coeff(f, 0) == 0 and ser.coeff(f, 1) == 0,
    )

    lorenz = 1 - (1 - u) ** (1 - sp.Rational(1) / k)
    gini_raw = 1 - 2 * sp.integrate(lorenz, (u, 0, 1), conds="none")
    # the 0**(2-1/k) boundary term is 0 for k > 1
    gini = sp.simplify(
        gini_raw.replace(lambda e: e.is_Pow and e.base == 0, lambda e: sp.Integer(0))
    )
    check("9. shape_from_gini", is_zero(sp.simplify(gini - 1 / (2 * k - 1))))

    print("\n-- network closed forms --")
    q = C - 1

    mgf_edge = sp.integrate(sp.exp(-alpha * (d + y)) * lam * sp.exp(-lam * y),
                            (y, 0, sp.oo))
    check("N1. edge-latency transform",
          is_zero(mgf_edge - sp.exp(-alpha * d) * lam / (lam + alpha)))

    balance = q * sp.exp(-alpha * d) * lam / (lam + alpha)
    det_limit = sp.limit(balance, lam, sp.oo)
    alpha_det = sp.log(q) / d
    check("N2. propagation exponent det. limit",
          is_zero(det_limit - q * sp.exp(-alpha * d))
          and is_zero((q * sp.exp(-alpha * d)).subs(alpha, alpha_det) - 1))

    z = -1 / (sp.E * q)
    u_star = 1 + 1 / w
    lhs = (1 - u_star) * sp.exp(u_star / (1 - u_star))
    reduces = is_zero(sp.simplify(lhs - (-sp.exp(-1) / (w * sp.exp(w)))))
    forces_root = is_zero(sp.simplify((-sp.exp(-1) / z) - q))
    check("N3. broadcast Lambert-W minimiser", reduces and forces_root)

    ok = all(_results)
    print("\n" + "=" * 72)
    print("RESULT:", "ALL SYMBOLIC CHECKS PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
