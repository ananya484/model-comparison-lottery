"""Figures 1-4 for the manuscript (EPS for submission, PNG for checking)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(__file__).parent / "results"
F = Path(__file__).parent / "figures"
F.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.2})
GAP_STYLE = {0.005: ("#9e9e9e", ":"), 0.01: ("#6a51a3", "-"), 0.02: ("#2171b5", "--"),
             0.05: ("#000000", "-.")}


def save(fig, name):
    fig.savefig(F / f"{name}.eps", format="eps")
    fig.savefig(F / f"{name}.png", dpi=300)
    plt.close(fig)


def fig1():
    a = pd.read_csv(R / "a_fresh_test.csv")
    a = a[(a.auc_best == 0.80) & (a.prev == 0.30)]
    fig, ax = plt.subplots(2, 3, figsize=(7.0, 4.4), sharey="row", sharex=True)
    rows = [("p_correct", "P(better model has\nhigher test AUC)", 0.45),
            ("p_sig_right", "P(DeLong 95% CI excludes 0,\ncorrect direction)", 0.0)]
    for r, (col, lab, ymin) in enumerate(rows):
        for i, rho in enumerate([0.3, 0.6, 0.9]):
            for gap, (c, ls) in GAP_STYLE.items():
                d = a[(a.rho == rho) & (a.gap == gap)].sort_values("n_test")
                ax[r, i].plot(d.n_test, d[col], color=c, ls=ls, marker="o", ms=2.5,
                              label=f"true gap {gap:g}")
            ax[r, i].set_xscale("log")
            ax[r, i].set_xticks([100, 250, 500, 1000, 2500, 5000])
            ax[r, i].set_xticklabels(["100", "250", "500", "1000", "2500", "5000"])
            ax[r, i].minorticks_off()
            ax[r, i].set_ylim(ymin, 1.01)
            if r == 0:
                ax[r, i].axhline(0.5, color="#bdbdbd", lw=0.5)
                ax[r, i].set_title(f"error correlation {rho}", fontsize=8)
            else:
                ax[r, i].set_xlabel("Test-set size")
            ax[r, i].text(-0.02, 1.06, "abcdef"[3 * r + i], transform=ax[r, i].transAxes,
                          fontweight="bold", va="top")
        ax[r, 0].set_ylabel(lab)
    ax[1, 0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.06, 1.0))
    fig.tight_layout()
    save(fig, "Fig1")


def fig2():
    b = pd.read_csv(R / "b_split_lottery.csv")
    b = b[(b.prev == 0.30) & (b.test_fraction == 0.3) & (b.rho == 0.6)]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    for gap, (c, ls) in {0.0: ("#d94801", "-"), **GAP_STYLE}.items():
        d = b[b.gap == gap].sort_values("n_pool")
        ax[0].plot(d.n_pool, d.disagree_mean, color=c, ls=ls, marker="o", ms=2.5,
                   label=f"true gap {gap:g}")
    ax[0].set_xscale("log")
    ax[0].set_xticks([500, 1000, 2500, 5000])
    ax[0].set_xticklabels(["500", "1000", "2500", "5000"])
    ax[0].set_xlabel("Dataset size (30% held out)")
    ax[0].set_ylabel("P(two analysts name different winners)")
    ax[0].set_ylim(0, 0.62)
    ax[0].legend(fontsize=6.5, loc="upper center", ncol=3)
    ax[0].set_title("(a) split-seed disagreement", fontsize=8, loc="left")
    d = b[(b.n_pool == 1000) & (b.gap > 0)].sort_values("gap")
    x = np.arange(len(d))
    ax[1].bar(x - 0.18, d.p_correct_single_split, 0.36, color="#9e9e9e", label="one 30% test split")
    ax[1].bar(x + 0.18, d.p_correct_whole_pool, 0.36, color="#2171b5", label="all 1000 patients")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels([f"{g:g}" for g in d.gap])
    ax[1].set_xlabel("True AUC gap")
    ax[1].set_ylabel("P(better model named)")
    ax[1].set_ylim(0.4, 1.02)
    ax[1].legend(fontsize=6.5, loc="upper left")
    ax[1].set_title("(b) fixed models: splitting discards information", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "Fig2")


def group_labels(ax, c):
    n1 = int((c.family == "C1").sum())
    ax.text((n1 - 1) / 2, -0.25, "two logistic regressions", ha="center", fontsize=6.5,
            transform=ax.get_xaxis_transform())
    ax.text(n1 + (len(c) - n1 - 1) / 2, -0.25, "boosting vs ridge regression", ha="center",
            fontsize=6.5, transform=ax.get_xaxis_transform())
    ax.axvline(n1 - 0.5, color="#bdbdbd", lw=0.5)
    ax.set_xlabel("True strategy-level AUC gap (A minus B)", labelpad=20)


def fig3():
    c = pd.read_csv(R / "c_summary.csv")
    c = c[c.n_pool == 1000].copy()
    c["label"] = [f"{g:+.3f}" for g in c.strategy_gap]
    c = c.sort_values(["family", "strategy_gap"])
    x = np.arange(len(c))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.0))
    tr = c.var_training / (c.var_training + c.var_testsampling)
    ax[0].bar(x, tr, color="#2171b5", label="training (refitting)")
    ax[0].bar(x, 1 - tr, bottom=tr, color="#d9d9d9", label="test-set sampling")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(c.label, fontsize=6, rotation=45)
    group_labels(ax[0], c)
    ax[0].set_ylabel("Share of split-to-split variance\nin the estimated AUC difference")
    ax[0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0.0, 0.93))
    ax[0].set_title("(a) where the instability comes from", fontsize=8, loc="left")
    ax[1].bar(x - 0.2, c.disagree, 0.4, color="#9e9e9e", label="single split each")
    ax[1].bar(x + 0.2, c.two_averaged_analysts_disagree, 0.4, color="#2171b5",
              label="average of 15 splits each")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(c.label, fontsize=6, rotation=45)
    group_labels(ax[1], c)
    ax[1].set_ylabel("P(two analysts name different winners)")
    ax[1].set_ylim(0, 0.55)
    ax[1].legend(fontsize=6.5, loc="upper right")
    ax[1].set_title("(b) refitted models, 1000 patients", fontsize=8, loc="left")
    fig.tight_layout()
    save(fig, "Fig3")


def fig4():
    d = pd.read_csv(R / "d_many_models.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for gap, (c, ls) in {g: GAP_STYLE[g] for g in (0.01, 0.02, 0.05)}.items():
        e = d[(d.gap == gap) & (d.n_test == 500)].sort_values("k")
        ax[0].plot(e.k, e.p_best_selected, color=c, ls=ls, marker="o", ms=2.5, label=f"gap {gap:g}")
        ax[1].plot(e.k, e.selected_optimism, color=c, ls=ls, marker="o", ms=2.5, label=f"gap {gap:g}")
    ax[0].set_ylabel("P(truly best model selected)")
    ax[1].set_ylabel("Optimism of selected model's AUC")
    for i in range(2):
        ax[i].set_xlabel("Number of candidate models")
        ax[i].set_xticks([2, 3, 5, 10])
        ax[i].set_title(f"({'ab'[i]}) 500 test patients", fontsize=8, loc="left")
    ax[0].legend(fontsize=6.5)
    fig.tight_layout()
    save(fig, "Fig4")


if __name__ == "__main__":
    for f in (fig1, fig2, fig3, fig4):
        try:
            f()
            print("ok", f.__name__)
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
