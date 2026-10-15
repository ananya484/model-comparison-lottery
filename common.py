"""Shared helpers: paired DeLong inference, binormal score generation, seeding."""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy.stats import norm, rankdata

SEED = 20261015
N_JOBS = max(1, (os.cpu_count() or 2) - 2)


def child_rng(*key: int) -> np.random.Generator:
    """Independent stream per configuration, so results do not depend on scheduling."""
    return np.random.default_rng(np.random.SeedSequence([SEED, *key]))


def delta_for_auc(auc: float) -> float:
    """Case-control mean separation (unit variances) giving a target binormal AUC."""
    return float(norm.ppf(auc) * np.sqrt(2.0))


def delong(y: np.ndarray, scores: np.ndarray):
    """Paired DeLong estimates for K models scored on the same patients.

    scores has shape (K, n). Returns (aucs[K], covariance[K, K]); None if fewer
    than two events or two non-events.
    """
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m < 2 or n < 2:
        return None
    scores = np.atleast_2d(scores)
    k = scores.shape[0]
    v01 = np.empty((k, m))
    v10 = np.empty((k, n))
    aucs = np.empty(k)
    for i in range(k):
        sp, sn = scores[i, pos], scores[i, neg]
        tz = rankdata(np.concatenate([sp, sn]))
        tx, ty = rankdata(sp), rankdata(sn)
        aucs[i] = (tz[:m].sum() - m * (m + 1) / 2.0) / (m * n)
        v01[i] = (tz[:m] - tx) / n
        v10[i] = 1.0 - (tz[m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return aucs, cov


def paired_delong(y, sa, sb):
    """AUC difference A-B with its DeLong standard error; None if degenerate."""
    out = delong(y, np.vstack([sa, sb]))
    if out is None:
        return None
    a, c = out
    var = c[0, 0] + c[1, 1] - 2 * c[0, 1]
    return a[0], a[1], a[0] - a[1], float(np.sqrt(max(var, 0.0)))


def auc_fast(y, s):
    """Mann-Whitney AUC without the variance machinery."""
    pos = y == 1
    m = int(pos.sum())
    n = len(y) - m
    if m == 0 or n == 0:
        return np.nan
    r = rankdata(s)
    return (r[pos].sum() - m * (m + 1) / 2.0) / (m * n)


def correlated_errors(rng, k, n, rho):
    """k standard-normal error vectors with exchangeable correlation rho."""
    z0 = rng.standard_normal(n)
    z = rng.standard_normal((k, n))
    return np.sqrt(rho) * z0 + np.sqrt(1.0 - rho) * z


def binormal_scores(rng, n, aucs, rho, prev):
    """Scores for len(aucs) models on the same n patients (binormal, equal variance)."""
    y = (rng.random(n) < prev).astype(int)
    d = np.array([delta_for_auc(a) for a in aucs])[:, None]
    e = correlated_errors(rng, len(aucs), n, rho)
    return y, d * y + e


def mcse_prop(p, r):
    return float(np.sqrt(max(p * (1 - p), 0.0) / r)) if r > 0 else np.nan


def mcse_mean(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    return float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
