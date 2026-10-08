"""Estimators for a non-randomised rollout, written in numpy/scipy.

    naive_cross_section   treated vs untreated in the same weeks (ignores selection)
    naive_pre_post        treated units before vs after (ignores time effects)
    twfe                  two-way fixed-effects difference-in-differences
    callaway_santanna     group-time ATTs against never-treated units (event study + overall)
    synthetic_control     convex-weight counterfactual built from untreated donor units

Y is a balanced (units x weeks) matrix; g holds each unit's launch week or NEVER (-1).
"""
import numpy as np
from scipy.optimize import minimize

NEVER = -1


# ------------------------------------------------------------------ naive baselines
def _cohorts(g):
    return sorted(set(g[g != NEVER]))


def naive_cross_section(Y, g):
    T = Y.shape[1]
    ctrl = Y[g == NEVER]
    num = den = 0.0
    for c in _cohorts(g):
        tr = Y[g == c]
        w = tr.shape[0] * (T - c)
        num += w * (tr[:, c:].mean() - ctrl[:, c:].mean())
        den += w
    return num / den


def naive_pre_post(Y, g):
    T = Y.shape[1]
    num = den = 0.0
    for c in _cohorts(g):
        tr = Y[g == c]
        w = tr.shape[0] * (T - c)
        num += w * (tr[:, c:].mean() - tr[:, :c].mean())
        den += w
    return num / den


# ------------------------------------------------------------------ two-way fixed effects
def twfe(Y, g):
    """Static TWFE DiD: unit and week fixed effects plus one treatment dummy.
    Returns (estimate, cluster-robust standard error)."""
    n, T = Y.shape
    D = ((g[:, None] != NEVER) & (np.arange(T)[None, :] >= g[:, None])).astype(float)

    def demean(M):
        return M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + M.mean()

    Dt, Yt = demean(D), demean(Y)
    denom = (Dt**2).sum()
    delta = (Dt * Yt).sum() / denom
    resid = Yt - delta * Dt
    score = (Dt * resid).sum(1)
    se = np.sqrt(n / (n - 1) * (score**2).sum()) / denom
    return float(delta), float(se)


# ------------------------------------------------------------------ Callaway-Sant'Anna style
def _cs_from_group_means(means, sizes, cohorts, e_lo, e_hi):
    """means: dict group -> mean outcome path (T,). Base period is launch week - 1."""
    ctrl = means[NEVER]
    att = {c: (means[c] - means[c][c - 1]) - (ctrl - ctrl[c - 1]) for c in cohorts}
    total = sum(sizes[c] for c in cohorts)
    es_range = np.arange(e_lo, e_hi + 1)
    es = np.array([sum(sizes[c] * att[c][c + e] for c in cohorts) / total for e in es_range])
    num = sum(sizes[c] * att[c][c:].sum() for c in cohorts)
    den = sum(sizes[c] * len(att[c][c:]) for c in cohorts)
    pre = (es_range >= e_lo) & (es_range <= -2)
    slope = np.polyfit(es_range[pre], es[pre], 1)[0]
    return num / den, es, slope


def callaway_santanna(Y, g, n_boot=300, seed=0, e_lo=-20):
    """Group-time ATTs against never-treated units, aggregated to event time and overall.
    Confidence intervals come from a unit-level bootstrap within groups.
    Returns dict: overall, event_time, es, slope (pre-trend), and bootstrap draws."""
    rng = np.random.default_rng(seed)
    T = Y.shape[1]
    cohorts = _cohorts(g)
    e_hi = min(T - c for c in cohorts) - 1
    groups = cohorts + [NEVER]
    idx = {k: np.where(g == k)[0] for k in groups}
    sizes = {k: len(v) for k, v in idx.items()}
    means = {k: Y[idx[k]].mean(0) for k in groups}
    overall, es, slope = _cs_from_group_means(means, sizes, cohorts, e_lo, e_hi)
    ov_b, es_b, sl_b = [], [], []
    for _ in range(n_boot):
        mb = {k: Y[rng.choice(idx[k], sizes[k], replace=True)].mean(0) for k in groups}
        o, e, s = _cs_from_group_means(mb, sizes, cohorts, e_lo, e_hi)
        ov_b.append(o), es_b.append(e), sl_b.append(s)
    return {
        "overall": float(overall),
        "event_time": np.arange(e_lo, e_hi + 1),
        "es": es,
        "slope": float(slope),
        "overall_boot": np.array(ov_b),
        "es_boot": np.array(es_b),
        "slope_boot": np.array(sl_b),
    }


def ci(draws, level=0.95):
    a = (1 - level) / 2
    return np.quantile(draws, a, axis=0), np.quantile(draws, 1 - a, axis=0)


# ------------------------------------------------------------------ synthetic control
def sc_weights(y_pre, donors_pre):
    """Non-negative weights summing to one that best reproduce the treated pre-period path."""
    J = donors_pre.shape[1]

    def loss(w):
        r = y_pre - donors_pre @ w
        return float(r @ r)

    def grad(w):
        return -2 * donors_pre.T @ (y_pre - donors_pre @ w)

    res = minimize(
        loss,
        np.full(J, 1.0 / J),
        jac=grad,
        bounds=[(0, 1)] * J,
        constraints=({"type": "eq", "fun": lambda w: w.sum() - 1, "jac": lambda w: np.ones(J)},),
        method="SLSQP",
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def synthetic_control(y_treat, donors, launch):
    """y_treat: (T,) treated path; donors: (T, J); launch: first treated week.
    Returns weights, gap path, mean post gap (effect) and pre/post RMSPE."""
    w = sc_weights(y_treat[:launch], donors[:launch])
    synth = donors @ w
    gap = y_treat - synth
    pre_rmspe = float(np.sqrt(np.mean(gap[:launch] ** 2)))
    post_rmspe = float(np.sqrt(np.mean(gap[launch:] ** 2)))
    return {
        "weights": w,
        "synthetic": synth,
        "gap": gap,
        "effect": float(gap[launch:].mean()),
        "pre_rmspe": pre_rmspe,
        "post_rmspe": post_rmspe,
    }


def placebo_in_space(Y_donors, n_treated, launch, actual_ratio, n_placebo=200, seed=0):
    """Pretend random groups of untreated units were launched at `launch`, re-run synthetic
    control on the remaining donors and compare post/pre RMSPE ratios.
    Caveat: random groups do not share the selection that real launches had."""
    rng = np.random.default_rng(seed)
    J = Y_donors.shape[1]
    ratios, effects = [], []
    for _ in range(n_placebo):
        pick = rng.choice(J, n_treated, replace=False)
        rest = np.setdiff1d(np.arange(J), pick)
        r = synthetic_control(Y_donors[:, pick].mean(1), Y_donors[:, rest], launch)
        ratios.append(r["post_rmspe"] / max(r["pre_rmspe"], 1e-9))
        effects.append(r["effect"])
    ratios = np.array(ratios)
    p = (1 + (ratios >= actual_ratio).sum()) / (1 + len(ratios))
    return {"ratios": ratios, "effects": np.array(effects), "p_value": float(p)}


# ------------------------------------------------------------------ comparative interrupted time series
def _cits_effect(treated_path, control_path, launch):
    """Fit a linear trend to the pre-launch gap (treated minus untreated), extrapolate it,
    and measure how far the post-launch gap departs from that extrapolation.
    Assumes the pre-launch gap would have kept its linear trend without the launch."""
    gap = treated_path - control_path
    t = np.arange(len(gap))
    b, a = np.polyfit(t[:launch], gap[:launch], 1)
    counterfactual = a + b * t
    return float((gap[launch:] - counterfactual[launch:]).mean()), counterfactual, gap


def cits(Y, g, n_boot=300, seed=0):
    """Comparative ITS for a single launch wave, with a unit-level bootstrap CI."""
    rng = np.random.default_rng(seed)
    launch = int(_cohorts(g)[0])
    tr_idx, ct_idx = np.where(g != NEVER)[0], np.where(g == NEVER)[0]
    est, counterfactual, gap = _cits_effect(Y[tr_idx].mean(0), Y[ct_idx].mean(0), launch)
    boots = np.array([
        _cits_effect(Y[rng.choice(tr_idx, len(tr_idx))].mean(0), Y[rng.choice(ct_idx, len(ct_idx))].mean(0), launch)[0]
        for _ in range(n_boot)
    ])
    return {"effect": est, "boot": boots, "gap": gap, "counterfactual_gap": counterfactual, "launch": launch}
