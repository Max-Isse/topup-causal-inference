"""Simulate a non-randomised rollout of "you can borrow more" top-up notifications.

Unit  = a partner-segment (a group of eligible customers reached through one partner).
Outcome = weekly top-up borrowing per eligible customer (GBP), a stand-in for
          value-per-customer. Real customer lifetime value needs a longer window
          and careful handling of censoring; this proxy keeps the focus on causal design.

    y_it = alpha_i + beta_i * t + common_t + eps_it + tau_it * D_it

Every unit has its own level (alpha) and trend (beta); all units share seasonality,
a drift and common shocks; eps is AR(1) noise. The treatment effect tau builds up
over weeks since launch, so "event time" matters. The true effect is known.

Scenarios
    A  Staggered rollout (two launch waves + never-treated). Rollout favoured
       high-level units (selection on LEVELS). Parallel trends hold, effects
       differ by wave and grow over time.
    B  One launch wave. Rollout favoured units that were already growing faster
       (selection on TRENDS). Parallel trends FAIL.
"""
from dataclasses import dataclass

import numpy as np

NEVER = -1


@dataclass
class Panel:
    Y: np.ndarray  # (units, weeks) observed outcome
    g: np.ndarray  # launch week per unit, or NEVER (-1)
    tau: np.ndarray  # (units, weeks) true effect (noise-free)
    truth_att: float  # mean true effect over treated post unit-weeks
    truth_event: dict  # event time -> mean true effect across treated units
    scenario: str


def _ar1(rng, size, rho, sd):
    """AR(1) noise with stationary standard deviation sd; time on the last axis."""
    out = np.zeros(size)
    out[..., 0] = rng.normal(0, sd, size[:-1])
    for t in range(1, size[-1]):
        out[..., t] = rho * out[..., t - 1] + rng.normal(0, sd * np.sqrt(1 - rho**2), size[:-1])
    return out


def _build(alpha, beta, g, wave_effect, T, rng, noise_sd, shock_sd, truth_event_range):
    n = len(alpha)
    t = np.arange(T)
    common = 0.02 * t + 1.2 * np.sin(2 * np.pi * t / 52) + _ar1(rng, (T,), 0.5, shock_sd)
    eps = _ar1(rng, (n, T), 0.5, noise_sd)
    tau = np.zeros((n, T))
    for i in range(n):
        if g[i] != NEVER:
            k = t - g[i]
            hetero = 1 + rng.normal(0, 0.2)
            tau[i] = np.where(k >= 0, wave_effect[g[i]] * hetero * (1 - np.exp(-(k + 1) / 6.0)), 0.0)
    Y = alpha[:, None] + beta[:, None] * t[None, :] + common[None, :] + eps + tau
    treated = g != NEVER
    post = (t[None, :] >= g[:, None]) & treated[:, None]
    truth_att = float(tau[post].mean())
    truth_event = {}
    for e in truth_event_range:
        vals = [tau[i, g[i] + e] for i in np.where(treated)[0] if 0 <= g[i] + e < T]
        truth_event[e] = float(np.mean(vals)) if vals else np.nan
    return Y, tau, truth_att, truth_event


def simulate_panel(scenario="A", seed=0, n_units=60, n_weeks=78, noise_sd=0.8, shock_sd=0.4,
                   trend_sd=None, selection_strength=1.5):
    rng = np.random.default_rng(seed)
    alpha = rng.normal(10, 2.0, n_units)
    if scenario == "A":
        beta = rng.normal(0.02, 0.012 if trend_sd is None else trend_sd, n_units)
        z = (alpha - alpha.mean()) / alpha.std()
        p = np.exp(selection_strength * z)
        treated = rng.choice(n_units, 24, replace=False, p=p / p.sum())
        early, late = treated[:12], treated[12:]
        g = np.full(n_units, NEVER)
        g[early], g[late] = 40, 52
        wave_effect = {40: 1.5, 52: 0.8}
        e_range = range(-20, 26)
    elif scenario == "B":
        beta = rng.normal(0.02, 0.02 if trend_sd is None else trend_sd, n_units)
        z = (beta - beta.mean()) / beta.std()
        p = np.exp(selection_strength * z)
        treated = rng.choice(n_units, 15, replace=False, p=p / p.sum())
        g = np.full(n_units, NEVER)
        g[treated] = 40
        wave_effect = {40: 1.2}
        e_range = range(-20, 38)
    else:
        raise ValueError("scenario must be 'A' or 'B'")
    Y, tau, truth_att, truth_event = _build(alpha, beta, g, wave_effect, n_weeks, rng, noise_sd, shock_sd, e_range)
    return Panel(Y, g, tau, truth_att, truth_event, scenario)
