"""Experiments A, B and D: score-level comparisons with known true AUCs.

A  fresh test set each replication. Naive rule (higher AUC wins) against a paired
   rule (declare a winner only if the 95% DeLong interval for the difference
   excludes zero). Records correct-winner probability, the bias of the winner's
   reported margin, how often that margin falls below the truth (it is not an
   upper bound), DeLong type I error, wrong-direction significance and coverage.
B  one fixed pool and two fixed scoring rules; only the split seed varies.
   Unbiased two-analyst disagreement per pool, its distribution over pools, and
   the probability of naming the better model from one split versus from the
   whole pool.
D  K > 2 candidate models: probability of selecting the best and optimism of the
   selected model's AUC.

Usage: python exp_ab_scores.py A|B|D
"""
from __future__ import annotations

import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import (N_JOBS, auc_fast, binormal_scores, child_rng, mcse_mean, mcse_prop,
                    paired_delong)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)
Z = 1.959963984540054

# ------------------------------------------------------------------ A
A_AUC = [0.70, 0.80, 0.90]
A_GAP = [0.0, 0.005, 0.01, 0.02, 0.05]
A_RHO = [0.3, 0.6, 0.9]
A_PREV = [0.05, 0.10, 0.30, 0.50]
A_N = [100, 250, 500, 1000, 2500, 5000]
A_REP = 4000


def a_config(key, auc_best, gap, rho, prev, n):
    rng = child_rng(1, *key)
    d, se, ok = [], [], 0
    for _ in range(A_REP):
        y, s = binormal_scores(rng, n, [auc_best, auc_best - gap], rho, prev)
        r = paired_delong(y, s[0], s[1])
        if r is None:
            continue
        d.append(r[2])
        se.append(r[3])
    d, se = np.array(d), np.array(se)
    r = len(d)
    lo, hi = d - Z * se, d + Z * se
    correct = np.mean((d > 0) + 0.5 * (d == 0))
    margin = np.abs(d)
    sig_right, sig_wrong = np.mean(lo > 0), np.mean(hi < 0)
    return dict(
        auc_best=auc_best, gap=gap, rho=rho, prev=prev, n_test=n, reps=r,
        reps_dropped=A_REP - r,
        p_correct=correct, p_correct_mcse=mcse_prop(correct, r),
        mean_d=d.mean(), sd_d=d.std(ddof=1), mean_se_delong=se.mean(),
        winner_margin=margin.mean(), winner_margin_mcse=mcse_mean(margin),
        margin_bias=margin.mean() - gap,
        p_margin_below_truth=np.mean(margin < gap) if gap > 0 else np.nan,
        p_sig_right=sig_right, p_sig_wrong=sig_wrong,
        p_sig_any=sig_right + sig_wrong, p_sig_any_mcse=mcse_prop(sig_right + sig_wrong, r),
        coverage=np.mean((lo <= gap) & (gap <= hi)),
        coverage_mcse=mcse_prop(np.mean((lo <= gap) & (gap <= hi)), r),
        # among significant declarations, the margin reported
        margin_if_sig=margin[(lo > 0) | (hi < 0)].mean() if ((lo > 0) | (hi < 0)).any() else np.nan,
    )


def run_a():
    grid = list(itertools.product(range(len(A_AUC)), range(len(A_GAP)), range(len(A_RHO)),
                                  range(len(A_PREV)), range(len(A_N))))
    out = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(a_config)(k, A_AUC[k[0]], A_GAP[k[1]], A_RHO[k[2]], A_PREV[k[3]], A_N[k[4]])
        for k in grid)
    pd.DataFrame(out).to_csv(OUT / "a_fresh_test.csv", index=False)


# ------------------------------------------------------------------ B
B_AUC = 0.80
B_GAP = [0.0, 0.005, 0.01, 0.02, 0.05]
B_RHO = [0.3, 0.6, 0.9]
B_PREV = [0.10, 0.30]
B_POOL = [500, 1000, 2500, 5000]
B_FRAC = [0.2, 0.3, 0.5]
B_POOLS = 200
B_SPLITS = 100


def b_config(key, gap, rho, prev, n_pool, frac):
    rng = child_rng(2, *key)
    n_te = int(round(n_pool * frac))
    dis, p_split, full_right, sd_split = [], [], [], []
    for _ in range(B_POOLS):
        y, s = binormal_scores(rng, n_pool, [B_AUC, B_AUC - gap], rho, prev)
        d_full = auc_fast(y, s[0]) - auc_fast(y, s[1])
        wins, valid, ds = 0.0, 0, []
        for _ in range(B_SPLITS):
            idx = rng.choice(n_pool, n_te, replace=False)
            yy = y[idx]
            if yy.sum() < 2 or (n_te - yy.sum()) < 2:
                continue
            dd = auc_fast(yy, s[0, idx]) - auc_fast(yy, s[1, idx])
            ds.append(dd)
            wins += 1.0 if dd > 0 else (0.5 if dd == 0 else 0.0)
            valid += 1
        if valid < 2:
            continue
        # unbiased estimator of 2p(1-p): chance two analysts with independent seeds disagree
        dis.append(2.0 * wins * (valid - wins) / (valid * (valid - 1)))
        p_split.append(wins / valid)
        full_right.append(1.0 if d_full > 0 else (0.5 if d_full == 0 else 0.0))
        sd_split.append(np.std(ds, ddof=1))
    dis, p_split = np.array(dis), np.array(p_split)
    return dict(
        auc_best=B_AUC, gap=gap, rho=rho, prev=prev, n_pool=n_pool, test_fraction=frac,
        n_test=n_te, pools=len(dis), splits=B_SPLITS,
        disagree_mean=dis.mean(), disagree_mcse=mcse_mean(dis),
        disagree_median=np.median(dis), disagree_p90=np.quantile(dis, 0.9),
        frac_pools_near_coin=np.mean(dis > 0.40),
        p_correct_single_split=p_split.mean() if gap > 0 else np.nan,
        p_correct_single_split_mcse=mcse_mean(p_split) if gap > 0 else np.nan,
        p_correct_whole_pool=np.mean(full_right) if gap > 0 else np.nan,
        p_correct_whole_pool_mcse=mcse_prop(np.mean(full_right), len(full_right)) if gap > 0 else np.nan,
        sd_d_across_splits=float(np.mean(sd_split)),
    )


def run_b():
    grid = list(itertools.product(range(len(B_GAP)), range(len(B_RHO)), range(len(B_PREV)),
                                  range(len(B_POOL)), range(len(B_FRAC))))
    out = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(b_config)(k, B_GAP[k[0]], B_RHO[k[1]], B_PREV[k[2]], B_POOL[k[3]], B_FRAC[k[4]])
        for k in grid)
    pd.DataFrame(out).to_csv(OUT / "b_split_lottery.csv", index=False)


# ------------------------------------------------------------------ D
D_K = [2, 3, 5, 10]
D_GAP = [0.01, 0.02, 0.05]
D_N = [250, 500, 1000, 2500]
D_RHO = 0.6
D_PREV = 0.30
D_AUC = 0.80
D_REP = 4000


def d_config(key, k, gap, n):
    rng = child_rng(4, *key)
    true = np.array([D_AUC] + [D_AUC - gap] * (k - 1))
    best, opt, margin = [], [], []
    for _ in range(D_REP):
        y, s = binormal_scores(rng, n, list(true), D_RHO, D_PREV)
        if y.sum() < 2 or (n - y.sum()) < 2:
            continue
        a = np.array([auc_fast(y, s[i]) for i in range(k)])
        j = int(np.argmax(a))
        best.append(j == 0)
        opt.append(a[j] - true[j])
        srt = np.sort(a)
        margin.append(srt[-1] - srt[-2])
    r = len(best)
    pb = float(np.mean(best))
    return dict(k=k, gap=gap, n_test=n, rho=D_RHO, prev=D_PREV, reps=r,
                p_best_selected=pb, p_best_selected_mcse=mcse_prop(pb, r),
                selected_optimism=float(np.mean(opt)), selected_optimism_mcse=mcse_mean(opt),
                reported_margin=float(np.mean(margin)), true_margin=gap)


def run_d():
    grid = list(itertools.product(range(len(D_K)), range(len(D_GAP)), range(len(D_N))))
    out = Parallel(n_jobs=N_JOBS, verbose=2)(
        delayed(d_config)(g, D_K[g[0]], D_GAP[g[1]], D_N[g[2]]) for g in grid)
    pd.DataFrame(out).to_csv(OUT / "d_many_models.csv", index=False)


if __name__ == "__main__":
    t0 = time.time()
    {"A": run_a, "B": run_b, "D": run_d}[sys.argv[1]]()
    print(f"{sys.argv[1]} done in {(time.time() - t0) / 60:.1f} min")
