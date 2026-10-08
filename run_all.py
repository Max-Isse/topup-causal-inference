"""Reproduce every number and figure in the README:  python run_all.py"""
import json

import numpy as np
import matplotlib.pyplot as plt

from src import estimators as E
from src import simulate, style

SEED = 1
LAUNCH_B = 40


def f(x):
    return float(x)


# ---------------------------------------------------------------- single worlds
def scenario_a():
    P = simulate.simulate_panel("A", seed=SEED)
    Y, g = P.Y, P.g
    tw, tw_se = E.twfe(Y, g)
    cs = E.callaway_santanna(Y, g, n_boot=500, seed=SEED)
    lo, hi = E.ci(cs["overall_boot"])
    slo, shi = E.ci(cs["slope_boot"])
    rows = [
        ("Naive: treated vs untreated\nafter launch", E.naive_cross_section(Y, g), None, None),
        ("Naive: treated before vs after", E.naive_pre_post(Y, g), None, None),
        ("Two-way fixed effects DiD", tw, tw - 1.96 * tw_se, tw + 1.96 * tw_se),
        ("Group-time DiD (never-treated\ncontrols)", cs["overall"], lo, hi),
    ]
    res = {
        "truth": P.truth_att,
        "naive_cross_section": f(rows[0][1]),
        "naive_pre_post": f(rows[1][1]),
        "twfe": f(tw), "twfe_ci": [f(rows[2][2]), f(rows[2][3])],
        "cs": f(cs["overall"]), "cs_ci": [f(lo), f(hi)],
        "pretrend_slope": f(cs["slope"]), "pretrend_slope_ci": [f(slo), f(shi)],
    }
    return P, cs, rows, res


def scenario_b():
    P = simulate.simulate_panel("B", seed=SEED)
    Y, g = P.Y, P.g
    tw, tw_se = E.twfe(Y, g)
    cs = E.callaway_santanna(Y, g, n_boot=500, seed=SEED)
    lo, hi = E.ci(cs["overall_boot"])
    slo, shi = E.ci(cs["slope_boot"])
    treated_path, donors = Y[g != simulate.NEVER].mean(0), Y[g == simulate.NEVER].T
    sc = E.synthetic_control(treated_path, donors, LAUNCH_B)
    ratio = sc["post_rmspe"] / sc["pre_rmspe"]
    pl = E.placebo_in_space(donors, int((g != simulate.NEVER).sum()), LAUNCH_B, ratio, n_placebo=200, seed=SEED)
    ci_t = E.cits(Y, g, n_boot=500, seed=SEED)
    clo, chi = E.ci(ci_t["boot"])
    rows = [
        ("Two-way fixed effects DiD", tw, tw - 1.96 * tw_se, tw + 1.96 * tw_se),
        ("Group-time DiD (never-treated\ncontrols)", cs["overall"], lo, hi),
        ("Synthetic control", sc["effect"], None, None),
        ("Comparative interrupted\ntime series (trend-adjusted)", ci_t["effect"], clo, chi),
    ]
    res = {
        "truth": P.truth_att,
        "twfe": f(tw), "twfe_ci": [f(rows[0][2]), f(rows[0][3])],
        "cs": f(cs["overall"]), "cs_ci": [f(lo), f(hi)],
        "sc": f(sc["effect"]), "sc_pre_rmspe": sc["pre_rmspe"], "sc_post_rmspe": sc["post_rmspe"],
        "sc_rmspe_ratio": f(ratio), "sc_placebo_p": pl["p_value"],
        "max_donor_weight": f(sc["weights"].max()), "donors_with_weight_gt_5pct": int((sc["weights"] > 0.05).sum()),
        "cits": f(ci_t["effect"]), "cits_ci": [f(clo), f(chi)],
        "pretrend_slope": f(cs["slope"]), "pretrend_slope_ci": [f(slo), f(shi)],
    }
    return P, cs, sc, pl, rows, res


# ---------------------------------------------------------------- Monte Carlo
def monte_carlo(scenario, n=200, selection_strength=None):
    kw = {} if selection_strength is None else {"selection_strength": selection_strength}
    keep = {}

    def add(k, v):
        keep.setdefault(k, []).append(v)

    for s in range(n):
        P = simulate.simulate_panel(scenario, seed=1000 + s, **kw)
        Y, g, truth = P.Y, P.g, P.truth_att
        add("truth", truth)
        if scenario == "A":
            add("naive_cross_section", E.naive_cross_section(Y, g))
            add("naive_pre_post", E.naive_pre_post(Y, g))
        d, se = E.twfe(Y, g)
        add("twfe", d)
        add("twfe_covered", abs(d - truth) < 1.96 * se)
        cs = E.callaway_santanna(Y, g, n_boot=150, seed=s)
        lo, hi = E.ci(cs["overall_boot"])
        add("cs", cs["overall"])
        add("cs_covered", lo <= truth <= hi)
        slo, shi = E.ci(cs["slope_boot"])
        add("pretrend_rejects", not (slo <= 0 <= shi))
        if scenario == "B":
            r = E.synthetic_control(Y[g != simulate.NEVER].mean(0), Y[g == simulate.NEVER].T, LAUNCH_B)
            add("sc", r["effect"])
            c = E.cits(Y, g, n_boot=150, seed=s)
            clo, chi = E.ci(c["boot"])
            add("cits", c["effect"])
            add("cits_covered", clo <= truth <= chi)
    truth = np.array(keep.pop("truth"))
    out = {"n_datasets": n, "mean_true_effect": f(truth.mean())}
    for k, v in keep.items():
        v = np.array(v, float)
        if k.endswith("_covered"):
            out[k.replace("_covered", "_ci_coverage")] = f(v.mean())
        elif k == "pretrend_rejects":
            out["pretrend_test_rejection_rate"] = f(v.mean())
        else:
            err = v - truth
            out[k] = {"bias": f(err.mean()), "rmse": f(np.sqrt((err**2).mean()))}
    return out


# ---------------------------------------------------------------- figures
def estimates_figure(rows, truth, fname, title):
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = np.arange(len(rows))[::-1]
    for yi, (label, est, lo, hi) in zip(y, rows):
        if lo is not None:
            ax.hlines(yi, lo, hi, color=style.BLUE, linewidth=2)
        ax.plot(est, yi, "o", color=style.BLUE, markersize=8, markeredgecolor=style.SURFACE, markeredgewidth=1.5)
        txt = f"{est:.2f}" + (f"  ({lo:.2f} to {hi:.2f})" if lo is not None else "")
        ax.text(max(est, hi if hi is not None else est) + 0.12, yi, txt, va="center", fontsize=8.5, color=style.INK2)
    ax.axvline(truth, color=style.INK, linestyle="--", linewidth=1.2)
    ax.text(truth, len(rows) - 0.35, f"True effect {truth:.2f}", ha="center", fontsize=8.5, color=style.INK,
            bbox={"facecolor": style.SURFACE, "edgecolor": "none", "pad": 1.5})
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_ylim(-0.6, len(rows) - 0.1)
    ax.set_xlim(left=0)
    ax.set_xlim(right=max(max(r[1], r[3] if r[3] is not None else 0) for r in rows) * 1.35)
    ax.set_xlabel("Estimated average effect on weekly top-up borrowing per eligible customer (£)")
    ax.set_title(title)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def event_study_figure(cs, P, fname, title):
    e = cs["event_time"]
    lo, hi = E.ci(cs["es_boot"])
    truth = np.array([P.truth_event[k] for k in e])
    truth = np.where(e < 0, 0.0, truth)
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.fill_between(e, lo, hi, color=style.BLUE, alpha=0.18, linewidth=0)
    ax.plot(e, cs["es"], color=style.BLUE, linewidth=2, label="Estimate (95% bootstrap band)")
    ax.plot(e, truth, color=style.INK, linestyle="--", linewidth=1.3, label="True effect")
    ax.axhline(0, color=style.AXIS, linewidth=1)
    ax.axvline(-0.5, color=style.MUTED, linewidth=1, linestyle=":")
    ax.set_xlabel("Weeks since launch")
    ax.set_ylabel("Effect (£ per eligible customer per week)")
    ax.set_title(title)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def sc_figure(P, sc, pl, fname):
    Y, g = P.Y, P.g
    t = np.arange(Y.shape[1])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    a1.plot(t, Y[g != simulate.NEVER].mean(0), color=style.BLUE, label="Launched units (average)")
    a1.plot(t, sc["synthetic"], color=style.ORANGE, label="Synthetic control")
    a1.plot(t, Y[g == simulate.NEVER].mean(0), color=style.AQUA, label="All untreated units (average)")
    a1.axvline(LAUNCH_B - 0.5, color=style.MUTED, linestyle=":", linewidth=1)
    a1.set_xlabel("Week")
    a1.set_ylabel("Weekly top-up borrowing per customer (£)")
    a1.set_title("Synthetic control fits before launch but overstates the effect")
    a1.legend(loc="upper left")
    a2.hist(pl["ratios"], bins=18, color=style.BLUE, alpha=0.85, edgecolor=style.SURFACE)
    ratio = sc["post_rmspe"] / sc["pre_rmspe"]
    a2.axvline(ratio, color=style.ORANGE, linewidth=2)
    a2.text(ratio, a2.get_ylim()[1] * 0.92, f" launched units: {ratio:.1f}\n placebo p = {pl['p_value']:.3f}", fontsize=8.5, color=style.INK2, va="top")
    a2.set_xlabel("Post / pre RMSPE ratio")
    a2.set_ylabel("Placebo runs")
    a2.set_title("Placebo test (200 random groups)")
    fig.tight_layout()
    fig.savefig(fname)
    plt.close(fig)


def main():
    style.apply()
    PA, csA, rowsA, resA = scenario_a()
    PB, csB, sc, pl, rowsB, resB = scenario_b()
    estimates_figure(rowsA, PA.truth_att, "figures/A_estimates.png", "Scenario A: staggered rollout favouring high-level units")
    event_study_figure(csA, PA, "figures/A_event_study.png", "Scenario A: event study (pre-launch estimates sit near zero)")
    estimates_figure(rowsB, PB.truth_att, "figures/B_estimates.png", "Scenario B: rollout favouring fast-growing units")
    event_study_figure(csB, PB, "figures/B_event_study.png", "Scenario B: event study (mild pre-launch drift, easy to miss)")
    sc_figure(PB, sc, pl, "figures/B_synthetic_control.png")
    mc = {
        "A_selection_on_levels": monte_carlo("A"),
        "B_selection_on_trends": monte_carlo("B"),
        "B0_launch_unrelated_to_trend": monte_carlo("B", selection_strength=0.0),
    }
    out = {"scenario_A_single_dataset": resA, "scenario_B_single_dataset": resB, "monte_carlo": mc}
    with open("results/results.json", "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
