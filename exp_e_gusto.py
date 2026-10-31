"""Experiment E: the split lottery in real clinical data (GUSTO-I, 40 830 patients,
30-day mortality 7.0%).

GUSTO-I is treated as the population. An analyst's study is a random sample of N
patients; inside it, ridge logistic regression and tuned gradient boosting (the same
procedures as Experiment C) are compared on a random 70/30 split. Each fitted model's
population AUC is computed on every GUSTO-I patient outside the study sample.

E1  one study sample of N = 5000, split with 500 different seeds.
E2  100 study samples each of N = 2500 and N = 5000, each split 30 times
    (same summaries as Experiment C).

Input: gusto.csv exported from the GUSTO-I data distributed with the R package
predtools 0.0.3 (CRAN). Usage: python exp_e_gusto.py PATH/TO/gusto.csv
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from common import N_JOBS, child_rng, paired_delong
from exp_c_refit import fit_gbm, fit_ridge

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
TEST_FRACTION = 0.30


def load(path):
    g = pd.read_csv(path)
    x = pd.DataFrame({
        "age": g.age, "female": (g.sex == "female").astype(int),
        "killip2": (g.Killip == "II").astype(int),
        "killip34": g.Killip.isin(["III", "IV"]).astype(int),
        "sysbp": g.sysbp, "pulse": g.pulse, "shock": g.sho, "diabetes": g.dia,
        "prev_mi": (g.pmi == "yes").astype(int), "anterior": (g.miloc == "Anterior").astype(int),
        "st_leads": g.ste, "relief_gt1h": g.ttr, "hypertension": (g.htn == 2).astype(int),
        "smoker": (g.smk == "current").astype(int), "height": g.height, "weight": g.weight,
    })
    return x.to_numpy(float), g.day30.to_numpy(int), list(x.columns)


def one_split(x, y, sample, rest, rng):
    n = len(sample)
    n_te = int(round(n * TEST_FRACTION))
    idx = rng.permutation(sample)
    te, tr = idx[:n_te], idx[n_te:]
    seed = int(rng.integers(2**31 - 1))
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    fa, fb = fit_gbm(x[tr], y[tr], seed), fit_ridge(x[tr], y[tr], seed)
    lr = make_pipeline(StandardScaler(), LogisticRegression(penalty=None, max_iter=2000)).fit(x[tr], y[tr])
    fc = lambda z: lr.decision_function(z)
    out = dict(n_train=len(tr), n_test=n_te, events_test=int(y[te].sum()))
    # pair 1: gradient boosting minus ridge; pair 2: unpenalised LR minus ridge
    for tag, f in (("gbm", fa), ("lr", fc)):
        a_te, b_te, d_te, se = paired_delong(y[te], f(x[te]), fb(x[te]))
        a_pop, b_pop = roc_auc_score(y[rest], f(x[rest])), roc_auc_score(y[rest], fb(x[rest]))
        out.update({f"auc_{tag}_test": a_te, f"auc_ridge_test": b_te, f"d_test_{tag}": d_te,
                    f"se_{tag}": se, f"auc_{tag}_pop": a_pop, "auc_ridge_pop": b_pop,
                    f"d_pop_{tag}": a_pop - b_pop})
    return out


def study(x, y, n, key, n_splits):
    rng = child_rng(500, n, key)
    perm = rng.permutation(len(y))
    sample, rest = perm[:n], perm[n:]
    rows = []
    for s in range(n_splits):
        r = one_split(x, y, sample, rest, child_rng(501, n, key, s))
        r.update(n_pool=n, pool=key, split=s, events_pool=int(y[sample].sum()))
        rows.append(r)
    return rows


def seed_job(x, y, sample, rest, s):
    r = one_split(x, y, sample, rest, child_rng(600, s))
    r.update(split=s)
    return r


if __name__ == "__main__":
    x, y, names = load(sys.argv[1])
    print(f"GUSTO-I: {len(y)} patients, {y.sum()} deaths ({y.mean():.4f}); predictors: {names}", flush=True)
    t0 = time.time()
    # E1: one study of 5000, 500 seeds
    rng = child_rng(599)
    perm = rng.permutation(len(y))
    sample, rest = perm[:5000], perm[5000:]
    e1 = Parallel(n_jobs=N_JOBS)(delayed(seed_job)(x, y, sample, rest, s) for s in range(500))
    pd.DataFrame(e1).to_csv(OUT / "e1_gusto_500_seeds.csv", index=False)
    print(f"E1 done ({(time.time() - t0) / 60:.1f} min)", flush=True)
    # E2: 100 studies x 30 splits at N = 2500 and 5000
    tasks = [(n, k) for n in (2500, 5000) for k in range(100)]
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(study)(x, y, n, k, 30) for n, k in tasks)
    pd.DataFrame([r for rows in out for r in rows]).to_csv(OUT / "e2_gusto_splits.csv.gz", index=False)
    print(f"E2 done ({(time.time() - t0) / 60:.1f} min)", flush=True)
