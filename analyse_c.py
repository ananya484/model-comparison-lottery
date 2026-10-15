"""Summaries of Experiment C (refitting inside each split) from results/c_splits.csv.gz.

Estimands
  strategy gap      E[population AUC(A) - population AUC(B)] over development
                    samples: which modelling strategy is better at this n.
  conditional gap   population AUC difference of the two models actually fitted
                    in a given split.
All Monte Carlo standard errors treat the pool (dataset) as the independent unit.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

OUT = Path(__file__).parent / "results"
Z = 1.959963984540054


def per_pool(g: pd.DataFrame) -> pd.Series:
    d = g["d_test"].to_numpy()
    dp = g["d_pop"].to_numpy()
    s = len(d)
    wins = np.sum(d > 0) + 0.5 * np.sum(d == 0)
    n_tr, n_te = g["n_train"].iloc[0], g["n_test"].iloc[0]
    sd = d.std(ddof=1)
    tcrit = stats.t.ppf(0.975, s - 1)
    se_naive = sd / np.sqrt(s)
    se_nb = sd * np.sqrt(1.0 / s + n_te / n_tr)          # Nadeau-Bengio correction
    h1, h2 = d[: s // 2].mean(), d[s // 2:].mean()
    return pd.Series(dict(
        splits=s,
        disagree=2.0 * wins * (s - wins) / (s * (s - 1)),
        var_test=d.var(ddof=1), var_pop=dp.var(ddof=1), var_err=(d - dp).var(ddof=1),
        mean_d=d.mean(), se_naive=se_naive, se_nb=se_nb, tcrit=tcrit,
        halves_disagree=float(np.sign(h1) != np.sign(h2)),
    ))


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (fam, lvl, n_pool), g in df.groupby(["family", "level", "n_pool"]):
        pool_mean_pop = g.groupby("pool")["d_pop"].mean()
        strat = pool_mean_pop.mean()
        strat_mcse = pool_mean_pop.std(ddof=1) / np.sqrt(len(pool_mean_pop))
        sign = np.sign(strat) if abs(strat) > 2 * strat_mcse else 0.0
        d, dp, se = g["d_test"], g["d_pop"], g["se_delong"]
        lo, hi = d - Z * se, d + Z * se
        g = g.assign(
            right_strat=(np.sign(d) == sign).astype(float),
            right_cond=(np.sign(d) == np.sign(dp)).astype(float),
            cov_cond=((lo <= dp) & (dp <= hi)).astype(float),
            cov_strat=((lo <= strat) & (strat <= hi)).astype(float),
            sig_any=((lo > 0) | (hi < 0)).astype(float),
            sig_wrong=(((lo > 0) & (sign < 0)) | ((hi < 0) & (sign > 0))).astype(float),
            margin=np.abs(d),
        )
        by_pool = g.groupby("pool")
        pm = lambda c: by_pool[c].mean()
        mc = lambda c: pm(c).std(ddof=1) / np.sqrt(by_pool.ngroups)
        pp = by_pool.apply(per_pool, include_groups=False)
        t_naive = pp["mean_d"] / pp["se_naive"]
        t_nb = pp["mean_d"] / pp["se_nb"]
        rej_naive = (np.abs(t_naive) > pp["tcrit"]).astype(float)
        rej_nb = (np.abs(t_nb) > pp["tcrit"]).astype(float)
        nb_cov = ((pp["mean_d"] - pp["tcrit"] * pp["se_nb"] <= strat)
                  & (strat <= pp["mean_d"] + pp["tcrit"] * pp["se_nb"])).astype(float)
        rows.append(dict(
            family=fam, level=lvl, param=g["param"].iloc[0], n_pool=n_pool,
            n_train=int(g["n_train"].iloc[0]), n_test=int(g["n_test"].iloc[0]),
            pools=by_pool.ngroups, splits_total=len(g),
            auc_a_pop=g["auc_a_pop"].mean(), auc_b_pop=g["auc_b_pop"].mean(),
            strategy_gap=strat, strategy_gap_mcse=strat_mcse, better=int(sign),
            p_right_strategy=pm("right_strat").mean() if sign != 0 else np.nan,
            p_right_strategy_mcse=mc("right_strat") if sign != 0 else np.nan,
            p_right_conditional=pm("right_cond").mean(), p_right_conditional_mcse=mc("right_cond"),
            winner_margin=pm("margin").mean(), winner_margin_mcse=mc("margin"),
            margin_bias=pm("margin").mean() - abs(strat),
            disagree=pp["disagree"].mean(), disagree_mcse=pp["disagree"].std(ddof=1) / np.sqrt(len(pp)),
            disagree_p90=pp["disagree"].quantile(0.9), frac_near_coin=(pp["disagree"] > 0.4).mean(),
            var_test=pp["var_test"].mean(), var_training=pp["var_pop"].mean(),
            var_testsampling=pp["var_err"].mean(),
            training_share=pp["var_pop"].mean() / pp["var_test"].mean(),
            # single fresh study: variance over all datasets and splits
            fresh_training_share=dp.var() / d.var(),
            delong_cov_conditional=pm("cov_cond").mean(), delong_cov_conditional_mcse=mc("cov_cond"),
            delong_cov_strategy=pm("cov_strat").mean(),
            delong_sig_any=pm("sig_any").mean(), delong_sig_any_mcse=mc("sig_any"),
            delong_sig_wrong=pm("sig_wrong").mean() if sign != 0 else np.nan,
            rep_naive_reject=rej_naive.mean(), rep_naive_reject_mcse=rej_naive.std(ddof=1) / np.sqrt(len(pp)),
            rep_nb_reject=rej_nb.mean(), rep_nb_reject_mcse=rej_nb.std(ddof=1) / np.sqrt(len(pp)),
            rep_nb_coverage=nb_cov.mean(),
            two_averaged_analysts_disagree=pp["halves_disagree"].mean(),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_csv(OUT / "c_splits.csv.gz")
    s = summarise(df)
    s.to_csv(OUT / "c_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(s.round(4).T)
