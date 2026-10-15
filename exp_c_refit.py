"""Experiment C: genuine predictor-level models refitted (and tuned) inside every split.

Two comparison families on the same data-generating mechanism (10 correlated
predictors, binary outcome, prevalence 0.20):

  C1  two non-nested, unpenalised logistic regressions that share four
      predictors and differ in a fifth (x5 versus x6). The true gap is set by
      the coefficient of x6; at a gap of zero the two models are exactly
      exchangeable by symmetry.
  C2  ridge logistic regression (penalty tuned by 5-fold CV) versus gradient
      boosting (tree depth tuned by 3-fold CV, iterations by early stopping),
      both on all 10 predictors. The true gap is set by the strength gamma of a
      nonlinear term that the regression cannot represent.

For each pool (one development dataset) the pool is split 70/30 S times; both
models are refitted inside every training part, evaluated on the test part
(paired DeLong), and also evaluated on a large independent validation sample,
which gives each fitted model's population AUC. That separates the two sources
of split-to-split variation: training (through the population AUCs) and test
sampling (test AUC minus population AUC).

Usage:
  python exp_c_refit.py calibrate    # writes results/c_calibration.json
  python exp_c_refit.py run          # writes results/c_splits.csv.gz
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import brentq

from common import N_JOBS, child_rng, paired_delong

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

P_PRED = 10
R_X = 0.3              # exchangeable correlation between predictors
PREV = 0.20
TEST_FRACTION = 0.30
N_POOLS = 200
N_SPLITS = 30
N_VALID = 50_000
POOL_SIZES = [1000, 2500]

C1_GAPS = [0.00, 0.01, 0.02, 0.05]
C2_TARGETS = [0.00, 0.01, 0.02, 0.05]   # gradient boosting minus ridge, at n_train 700
BETA_SHARED = 0.3
BETA5 = 1.0
BETA_C2 = 0.35


# ------------------------------------------------------------------ data
def draw_x(rng, n):
    z0 = rng.standard_normal((n, 1))
    return np.sqrt(R_X) * z0 + np.sqrt(1 - R_X) * rng.standard_normal((n, P_PRED))


def lin_c1(x, beta6):
    return BETA_SHARED * x[:, :4].sum(1) + BETA5 * x[:, 4] + beta6 * x[:, 5]


def nonlin(x):
    return (x[:, 0] ** 2 - 1.0) + x[:, 1] * x[:, 2]


def lin_c2(x, gamma):
    return BETA_C2 * x[:, :6].sum(1) + gamma * nonlin(x)


def intercept_for(eta, prev=PREV):
    return brentq(lambda b0: (1 / (1 + np.exp(-(b0 + eta)))).mean() - prev, -15, 15)


def draw_y(rng, eta, b0):
    return (rng.random(len(eta)) < 1 / (1 + np.exp(-(b0 + eta)))).astype(int)


# ------------------------------------------------------------------ models
def fit_c1(x, y):
    from sklearn.linear_model import LogisticRegression
    ma = LogisticRegression(penalty=None, max_iter=1000).fit(x[:, [0, 1, 2, 3, 4]], y)
    mb = LogisticRegression(penalty=None, max_iter=1000).fit(x[:, [0, 1, 2, 3, 5]], y)
    return (lambda z: ma.decision_function(z[:, [0, 1, 2, 3, 4]]),
            lambda z: mb.decision_function(z[:, [0, 1, 2, 3, 5]]))


def fit_ridge(x, y, seed):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    m = make_pipeline(StandardScaler(),
                      LogisticRegressionCV(Cs=10, cv=cv, scoring="neg_log_loss",
                                           max_iter=2000)).fit(x, y)
    return lambda z: m.decision_function(z)


def _gbm(depth, seed):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        max_depth=depth, learning_rate=0.05, max_iter=500, early_stopping=True,
        validation_fraction=0.2, n_iter_no_change=20, min_samples_leaf=20,
        l2_regularization=1.0, random_state=seed)


def fit_gbm(x, y, seed):
    from sklearn.metrics import log_loss
    from sklearn.model_selection import StratifiedKFold
    best = None
    for depth in (1, 2, 3):
        ll = []
        for tr, te in StratifiedKFold(3, shuffle=True, random_state=seed).split(x, y):
            m = _gbm(depth, seed).fit(x[tr], y[tr])
            ll.append(log_loss(y[te], m.predict_proba(x[te])[:, 1], labels=[0, 1]))
        if best is None or np.mean(ll) < best[0]:
            best = (float(np.mean(ll)), depth)
    m = _gbm(best[1], seed).fit(x, y)
    return lambda z: m.predict_proba(z)[:, 1]


def fit_c2(x, y, seed):
    # model A = gradient boosting, model B = ridge logistic regression
    return fit_gbm(x, y, seed), fit_ridge(x, y, seed)


# ------------------------------------------------------------------ calibration
def c1_asymptotic_gap(beta6, x, u, xv, uv):
    from sklearn.metrics import roc_auc_score
    eta, etav = lin_c1(x, beta6), lin_c1(xv, beta6)
    b0 = intercept_for(eta)
    y = (u < 1 / (1 + np.exp(-(b0 + eta)))).astype(int)
    yv = (uv < 1 / (1 + np.exp(-(b0 + etav)))).astype(int)
    fa, fb = fit_c1(x, y)
    return roc_auc_score(yv, fa(xv)) - roc_auc_score(yv, fb(xv)), roc_auc_score(yv, fa(xv))


def c2_expected_gap(gamma, n_train=700, reps=120, seed_key=0):
    """Mean over development samples of population AUC(GBM) - AUC(ridge)."""
    rng = child_rng(90, seed_key, int(round(gamma * 1000)))
    xv = draw_x(rng, N_VALID)
    b0 = intercept_for(lin_c2(draw_x(rng, 200_000), gamma))
    yv = draw_y(rng, lin_c2(xv, gamma), b0)

    def one(r):
        from sklearn.metrics import roc_auc_score
        g = child_rng(91, seed_key, int(round(gamma * 1000)), r)
        x = draw_x(g, n_train)
        y = draw_y(g, lin_c2(x, gamma), b0)
        fa, fb = fit_c2(x, y, r)
        return roc_auc_score(yv, fa(xv)), roc_auc_score(yv, fb(xv))

    res = np.array(Parallel(n_jobs=N_JOBS)(delayed(one)(r) for r in range(reps)))
    d = res[:, 0] - res[:, 1]
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(reps)), float(res[:, 0].mean()), float(res[:, 1].mean())


def calibrate():
    rng = child_rng(80)
    n_big = 400_000
    x, u = draw_x(rng, n_big), rng.random(n_big)
    xv, uv = draw_x(rng, n_big), rng.random(n_big)
    c1 = {}
    for gap in C1_GAPS:
        if gap == 0:
            b6 = BETA5
        else:
            b6 = brentq(lambda b: c1_asymptotic_gap(b, x, u, xv, uv)[0] - gap, 0.0, BETA5)
        g, auc_a = c1_asymptotic_gap(b6, x, u, xv, uv)
        c1[f"{gap:.2f}"] = dict(beta6=b6, asymptotic_gap=g, auc_a=auc_a)
        print(f"C1 gap {gap:.2f}: beta6 {b6:.4f}  asymptotic gap {g:.4f}  AUC_A {auc_a:.3f}", flush=True)

    # C2: coarse grid, then interpolate gamma for each target, then verify
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9]
    curve = []
    for gm in grid:
        m, se, aa, ab = c2_expected_gap(gm)
        curve.append((gm, m, se, aa, ab))
        print(f"C2 gamma {gm:.2f}: E[AUC_gbm - AUC_ridge] = {m:+.4f} (MCSE {se:.4f})  "
              f"gbm {aa:.3f} ridge {ab:.3f}", flush=True)
    g_arr, m_arr = np.array([c[0] for c in curve]), np.array([c[1] for c in curve])
    c2 = {"curve": curve}
    for t in C2_TARGETS:
        idx = np.where((m_arr[:-1] - t) * (m_arr[1:] - t) <= 0)[0]
        if len(idx) == 0:
            c2[f"{t:.2f}"] = None
            print(f"C2 target {t:.2f}: not reached on grid", flush=True)
            continue
        i = idx[0]
        gm = float(g_arr[i] + (t - m_arr[i]) * (g_arr[i + 1] - g_arr[i]) / (m_arr[i + 1] - m_arr[i]))
        m, se, aa, ab = c2_expected_gap(gm, reps=240, seed_key=1)
        c2[f"{t:.2f}"] = dict(gamma=gm, expected_gap=m, mcse=se, auc_gbm=aa, auc_ridge=ab)
        print(f"C2 target {t:.2f}: gamma {gm:.3f} -> verified gap {m:+.4f} (MCSE {se:.4f})", flush=True)
    (OUT / "c_calibration.json").write_text(json.dumps(dict(c1=c1, c2=c2), indent=2))


# ------------------------------------------------------------------ main experiment
def one_pool(family, level_key, param, n_pool, pool):
    rng = child_rng(100, 1 if family == "C1" else 2, level_key, n_pool, pool)
    lin = (lambda z: lin_c1(z, param)) if family == "C1" else (lambda z: lin_c2(z, param))
    b0 = intercept_for(lin(draw_x(child_rng(70, level_key, 1 if family == "C1" else 2), 200_000)))
    x = draw_x(rng, n_pool)
    y = draw_y(rng, lin(x), b0)
    xv = draw_x(rng, N_VALID)
    yv = draw_y(rng, lin(xv), b0)
    n_te = int(round(n_pool * TEST_FRACTION))
    rows = []
    from sklearn.metrics import roc_auc_score
    for s in range(N_SPLITS):
        idx = rng.permutation(n_pool)
        te, tr = idx[:n_te], idx[n_te:]
        if y[tr].sum() < 10 or y[te].sum() < 2:
            continue
        seed = int(rng.integers(2**31 - 1))
        fa, fb = fit_c1(x[tr], y[tr]) if family == "C1" else fit_c2(x[tr], y[tr], seed)
        res = paired_delong(y[te], fa(x[te]), fb(x[te]))
        if res is None:
            continue
        a_te, b_te, d_te, se = res
        a_pop, b_pop = roc_auc_score(yv, fa(xv)), roc_auc_score(yv, fb(xv))
        rows.append((family, level_key, param, n_pool, pool, s, len(tr), n_te, int(y[te].sum()),
                     a_te, b_te, d_te, se, a_pop, b_pop, a_pop - b_pop))
    return rows


COLS = ["family", "level", "param", "n_pool", "pool", "split", "n_train", "n_test", "events_test",
        "auc_a_test", "auc_b_test", "d_test", "se_delong", "auc_a_pop", "auc_b_pop", "d_pop"]


def run():
    cal = json.loads((OUT / "c_calibration.json").read_text())
    tasks = []
    for i, gap in enumerate(C1_GAPS):
        b6 = cal["c1"][f"{gap:.2f}"]["beta6"]
        for n_pool in POOL_SIZES:
            for p in range(N_POOLS):
                tasks.append(("C1", i, b6, n_pool, p))
    for i, t in enumerate(C2_TARGETS):
        c = cal["c2"].get(f"{t:.2f}")
        if not c:
            continue
        for n_pool in POOL_SIZES:
            for p in range(N_POOLS):
                tasks.append(("C2", i, c["gamma"], n_pool, p))
    # purely linear truth: ridge regression is the better strategy
    for n_pool in POOL_SIZES:
        for p in range(N_POOLS):
            tasks.append(("C2", len(C2_TARGETS), 0.0, n_pool, p))
    t0 = time.time()
    print(f"{len(tasks)} pool tasks on {N_JOBS} workers", flush=True)
    out = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(one_pool)(*t) for t in tasks)
    df = pd.DataFrame([r for rows in out for r in rows], columns=COLS)
    df.to_csv(OUT / "c_splits.csv.gz", index=False)
    print(f"wrote {len(df)} split records in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    {"calibrate": calibrate, "run": run}[sys.argv[1]]()
