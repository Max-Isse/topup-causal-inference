"""Checks against known answers. Run:  python tests/test_core.py   (or pytest, if installed)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import estimators as E  # noqa: E402
from src import simulate  # noqa: E402


def test_twfe_equals_classic_two_by_two_did():
    rng = np.random.default_rng(0)
    Y = rng.normal(10, 2, (8, 2))
    g = np.array([1, 1, 1, 1, -1, -1, -1, -1])  # first four launched in week 1
    classic = (Y[:4, 1].mean() - Y[:4, 0].mean()) - (Y[4:, 1].mean() - Y[4:, 0].mean())
    est, se = E.twfe(Y, g)
    assert abs(est - classic) < 1e-9 and se > 0


def test_group_time_did_is_exact_without_noise_and_with_parallel_trends():
    P = simulate.simulate_panel("A", seed=3, noise_sd=0.0, shock_sd=0.0, trend_sd=0.0)
    cs = E.callaway_santanna(P.Y, P.g, n_boot=5)
    assert abs(cs["overall"] - P.truth_att) < 1e-9
    assert abs(cs["slope"]) < 1e-9  # no pre-trend by construction


def test_synthetic_control_recovers_known_weights_and_effect():
    rng = np.random.default_rng(1)
    T, J, launch = 60, 8, 40
    donors = rng.normal(0, 1, (T, J)).cumsum(0)
    w_true = np.array([0.4, 0.3, 0.2, 0.1, 0, 0, 0, 0])
    y = donors @ w_true
    y[launch:] += 2.0
    r = E.synthetic_control(y, donors, launch)
    assert np.allclose(r["weights"], w_true, atol=1e-3)
    assert abs(r["effect"] - 2.0) < 1e-3


def test_synthetic_control_weights_are_a_valid_mixture():
    rng = np.random.default_rng(2)
    donors = rng.normal(0, 1, (50, 12)).cumsum(0)
    w = E.sc_weights(donors[:, :3].mean(1) + 0.5, donors)
    assert abs(w.sum() - 1) < 1e-9 and (w >= 0).all()


def test_trend_adjusted_estimator_is_exact_for_linear_gap():
    t = np.arange(60)
    control = np.sin(t / 5.0)
    treated = control + 1.0 + 0.1 * t
    treated[40:] += 3.0
    effect, _, _ = E._cits_effect(treated, control, 40)
    assert abs(effect - 3.0) < 1e-9


def test_naive_comparison_is_badly_biased_when_rollout_selects_on_level():
    P = simulate.simulate_panel("A", seed=1)
    assert E.naive_cross_section(P.Y, P.g) > P.truth_att + 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"{len(tests)} tests passed")
