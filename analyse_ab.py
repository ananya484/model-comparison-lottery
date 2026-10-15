"""Derived quantities from Experiments A and B.

Required test-set size. In every configuration of Experiment A the empirical
standard deviation of the estimated AUC difference scales as kappa / sqrt(n)
(checked below). With kappa estimated per (AUC, prevalence, correlation, gap),
the test-set size at which
  - the naive winner is the better model with probability 0.90 is
        n90 = (z_0.90 * kappa / gap)^2
  - a two-sided 5% DeLong test has 80% power is
        n80 = ((z_0.975 + z_0.80) * kappa / gap)^2
These are normal approximations; they are checked against the simulated
correct-winner probabilities where the grid allows.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

OUT = Path(__file__).parent / "results"

if __name__ == "__main__":
    a = pd.read_csv(OUT / "a_fresh_test.csv")
    a["kappa"] = a["sd_d"] * np.sqrt(a["n_test"])
    # check the 1/sqrt(n) scaling: spread of kappa across n within a cell
    keys = ["auc_best", "gap", "rho", "prev"]
    big = a[a["n_test"] >= 500]
    spread = big.groupby(keys)["kappa"].agg(lambda k: k.std() / k.mean())
    print(f"kappa coefficient of variation across n>=500: median {spread.median():.3f}, max {spread.max():.3f}")
    kap = big.groupby(keys)["kappa"].mean().reset_index()
    kap = kap[kap["gap"] > 0]
    kap["n_for_p90_correct"] = (norm.ppf(0.90) * kap["kappa"] / kap["gap"]) ** 2
    kap["n_for_80pct_power"] = ((norm.ppf(0.975) + norm.ppf(0.80)) * kap["kappa"] / kap["gap"]) ** 2
    kap.to_csv(OUT / "a_required_n.csv", index=False)

    # check the approximation against simulation: predicted P(correct) = Phi(gap*sqrt(n)/kappa)
    m = a[a["gap"] > 0].merge(kap[keys + ["kappa"]], on=keys, suffixes=("", "_cell"))
    m["p_pred"] = norm.cdf(m["gap"] * np.sqrt(m["n_test"]) / m["kappa_cell"])
    err = (m["p_pred"] - m["p_correct"]).abs()
    print(f"normal approximation vs simulation, |error| in P(correct): median {err.median():.4f}, "
          f"95th pct {err.quantile(.95):.4f}, max {err.max():.4f} (n>=250: max "
          f"{err[m['n_test'] >= 250].max():.4f})")
    with pd.option_context("display.width", 200, "display.max_rows", 200):
        print(kap[kap["gap"].isin([0.01, 0.02])].round(0).to_string(index=False))
