"""Summaries and figure for Experiment E (GUSTO-I)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analyse_c import summarise

R = Path(__file__).parent / "results"
F = Path(__file__).parent / "figures"
Z = 1.959963984540054
PAIRS = {"gbm": "Gradient boosting minus ridge regression",
         "lr": "Unpenalised minus ridge logistic regression"}


def e1_summary(e1):
    rows = []
    for tag in PAIRS:
        d, se, dp = e1[f"d_test_{tag}"], e1[f"se_{tag}"], e1[f"d_pop_{tag}"]
        s = len(d)
        p = np.mean(d > 0) + 0.5 * np.mean(d == 0)
        lo, hi = d - Z * se, d + Z * se
        d30 = d.iloc[:30]
        n_tr, n_te = e1.n_train.iloc[0], e1.n_test.iloc[0]
        t_nb = d30.mean() / (d30.std(ddof=1) * np.sqrt(1 / 30 + n_te / n_tr))
        t_nv = d30.mean() / (d30.std(ddof=1) / np.sqrt(30))
        rows.append(dict(
            pair=tag, seeds=s, pop_gap_mean=dp.mean(), pop_gap_sd=dp.std(ddof=1),
            test_gap_mean=d.mean(), test_gap_sd=d.std(ddof=1),
            test_gap_p2_5=d.quantile(0.025), test_gap_p97_5=d.quantile(0.975),
            p_first_model_wins=p, two_analyst_disagree=2 * p * (1 - p),
            p_sig_first_better=np.mean(lo > 0), p_sig_ridge_better=np.mean(hi < 0),
            p_conditional_right=np.mean(np.sign(d) == np.sign(dp)),
            training_share=dp.var(ddof=1) / d.var(ddof=1),
            nb_p_30seeds=2 * stats.t.sf(abs(t_nb), 29), naive_p_30seeds=2 * stats.t.sf(abs(t_nv), 29),
            auc_ridge_pop=e1.auc_ridge_pop.mean(), auc_other_pop=e1[f"auc_{tag}_pop"].mean(),
            events_test_median=e1.events_test.median(),
        ))
    return pd.DataFrame(rows)


def e2_summary(e2):
    parts = []
    for tag in PAIRS:
        df = pd.DataFrame(dict(
            family=tag, level=0, param=0.0, n_pool=e2.n_pool, pool=e2.pool, split=e2.split,
            n_train=e2.n_train, n_test=e2.n_test, d_test=e2[f"d_test_{tag}"],
            se_delong=e2[f"se_{tag}"], auc_a_pop=e2[f"auc_{tag}_pop"], auc_b_pop=e2.auc_ridge_pop,
            d_pop=e2[f"d_pop_{tag}"]))
        parts.append(df)
    return summarise(pd.concat(parts, ignore_index=True))


def figure(e1, s1):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
    for i, tag in enumerate(PAIRS):
        d = e1[f"d_test_{tag}"]
        row = s1[s1.pair == tag].iloc[0]
        lim = max(abs(d.quantile(0.001)), abs(d.quantile(0.999))) * 1.05
        bins = np.linspace(-lim, lim, 41)
        ax[i].hist(d[d <= 0], bins=bins, color="#2171b5", label="ridge regression wins")
        ax[i].hist(d[d > 0], bins=bins, color="#d94801",
                   label=("gradient boosting wins" if tag == "gbm" else "unpenalised LR wins"))
        ax[i].axvline(row.pop_gap_mean, color="black", lw=1.2, ls="--", label="true (population) difference")
        ax[i].axvline(0, color="#7f7f7f", lw=0.6)
        ax[i].set_xlabel("Test-set AUC difference")
        ax[i].set_title(f"({'ab'[i]}) {PAIRS[tag].lower()}", fontsize=8, loc="left")
        other = "boosting" if tag == "gbm" else "unpenalised LR"
        ax[i].text(0.99, 0.97, f"blue: ridge wins {(1 - row.p_first_model_wins) * 100:.0f}% of seeds\n"
                   f"orange: {other} wins {row.p_first_model_wins * 100:.0f}%",
                   transform=ax[i].transAxes, ha="right", va="top", fontsize=7)
        ax[i].set_ylim(0, ax[i].get_ylim()[1] * 1.3)
        ax[i].text(0.01, 0.97, "dashed: true difference", transform=ax[i].transAxes, ha="left",
                   va="top", fontsize=7)
    ax[0].set_ylabel("Number of random seeds (of 500)")
    fig.tight_layout()
    for ext in ("eps", "png"):
        fig.savefig(F / f"Fig4_gusto.{ext}", **({"dpi": 300} if ext == "png" else {}))


if __name__ == "__main__":
    e1 = pd.read_csv(R / "e1_gusto_500_seeds.csv")
    s1 = e1_summary(e1)
    s1.to_csv(R / "e1_summary.csv", index=False)
    e2 = pd.read_csv(R / "e2_gusto_splits.csv.gz")
    s2 = e2_summary(e2)
    s2.to_csv(R / "e2_summary.csv", index=False)
    figure(e1, s1)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(s1.round(4).T)
        print(s2.round(4).T)
