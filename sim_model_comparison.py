"""How often is the reported winner of a prediction-model comparison the better model?

A study that compares two clinical prediction models evaluates both on one test
set and reports the one with the higher area under the curve. This simulation
asks three things about that report: how often the declared winner is genuinely
the better model, how much the reported margin overstates the true one, and how
often two analysts holding the same data and the same models would disagree
because they split it with different random seeds.

Experiment 3 repeats the comparison with the models retrained inside each split,
so that training variability is included rather than assumed away.

Deterministic given SEED. Writes results_model_comparison.csv,
results_model_split_lottery.csv and results_model_retrained.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata

SEED = 20260823
N_REP = 4000              # replications for experiments 1 and 2
N_REP_RETRAIN = 1000      # replications for the slower retraining experiment

AUC_BEST = 0.80           # true area under the curve of the better model
GAPS = [0.00, 0.01, 0.02, 0.05]
CORRELATIONS = [0.5, 0.9]  # between the two models' errors on the same patient
TEST_SIZES = [100, 200, 500, 1000, 2000]
PREVALENCE = 0.30

POOL_SIZE = 1000          # experiment 2: one fixed dataset, resplit repeatedly
TEST_FRACTION = 0.30


def delta_for_auc(auc: float) -> float:
    """Case-control mean separation giving a target binormal AUC."""
    return float(norm.ppf(auc) * np.sqrt(2.0))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """Mann-Whitney estimate of the area under the ROC curve."""
    pos, neg = score[y == 1], score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0)
                 / (len(pos) * len(neg)))


def draw_scores(rng, n, d_a, d_b, rho, prev):
    """Two models' scores on the same n patients.

    Both models see the same patients, so their errors are correlated; rho is
    that correlation. Model A is the better one when d_a > d_b.
    """
    y = (rng.random(n) < prev).astype(int)
    z = rng.standard_normal((2, n))
    e_a = z[0]
    e_b = rho * z[0] + np.sqrt(1.0 - rho ** 2) * z[1]
    return y, d_a * y + e_a, d_b * y + e_b


# --------------------------------------------------------------- experiment 1
def experiment_fresh_test_sets(rng) -> pd.DataFrame:
    """Fresh test set each replication: how often is the declared winner right,
    and by how much does the winner's reported margin overstate the truth?"""
    rows = []
    d_a = delta_for_auc(AUC_BEST)
    for gap in GAPS:
        d_b = delta_for_auc(AUC_BEST - gap)
        for rho in CORRELATIONS:
            for n in TEST_SIZES:
                correct, margins, gaps_obs = 0, [], []
                valid = 0
                for _ in range(N_REP):
                    y, sa, sb = draw_scores(rng, n, d_a, d_b, rho, PREVALENCE)
                    if y.sum() < 2 or (n - y.sum()) < 2:
                        continue
                    a, b = auc(sa, y), auc(sb, y)
                    valid += 1
                    gaps_obs.append(a - b)
                    if a > b:
                        correct += 1
                        margins.append(a - b)
                    elif a == b:
                        correct += 0.5
                rows.append(dict(
                    true_gap=gap, correlation=rho, n_test=n,
                    p_correct_winner=correct / valid,
                    observed_gap_mean=float(np.mean(gaps_obs)),
                    winner_margin_mean=float(np.mean(margins)) if margins else np.nan,
                    margin_inflation=(float(np.mean(margins)) - gap) if margins else np.nan,
                    n_reps=valid,
                ))
                print(f"  gap {gap:.2f}  rho {rho}  n={n:<5} "
                      f"P(correct winner) {correct / valid:.3f}   "
                      f"winner's reported margin {np.mean(margins):.4f} "
                      f"(true {gap:.2f})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- experiment 2
def experiment_split_lottery(rng) -> pd.DataFrame:
    """One fixed dataset and two fixed models; only the split seed changes.

    Reports the chance that two analysts drawing different random test sets from
    the same pool declare different winners.
    """
    rows = []
    d_a = delta_for_auc(AUC_BEST)
    n_test = int(POOL_SIZE * TEST_FRACTION)
    for gap in GAPS:
        d_b = delta_for_auc(AUC_BEST - gap)
        for rho in CORRELATIONS:
            flip_rates = []
            for _ in range(200):                       # 200 independent pools
                y, sa, sb = draw_scores(rng, POOL_SIZE, d_a, d_b, rho, PREVALENCE)
                wins_a = 0
                for _ in range(200):                   # 200 splits of that pool
                    idx = rng.choice(POOL_SIZE, n_test, replace=False)
                    yy = y[idx]
                    if yy.sum() < 2 or (n_test - yy.sum()) < 2:
                        continue
                    wins_a += auc(sa[idx], yy) > auc(sb[idx], yy)
                p = wins_a / 200.0
                # two analysts with independent seeds disagree with probability
                # 2p(1-p); this is the split lottery, holding data and models fixed
                flip_rates.append(2 * p * (1 - p))
            rows.append(dict(
                true_gap=gap, correlation=rho, pool_size=POOL_SIZE, n_test=n_test,
                p_two_analysts_disagree=float(np.mean(flip_rates)),
                worst_pool_disagreement=float(np.max(flip_rates)),
            ))
            print(f"  gap {gap:.2f}  rho {rho}  "
                  f"two analysts disagree {np.mean(flip_rates):.3f} "
                  f"(worst pool {np.max(flip_rates):.3f})")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- experiment 3
def experiment_retrained(rng) -> pd.DataFrame:
    """As experiment 2 but the models are refitted inside each split, so the
    reported winner also absorbs training variability."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

    rows = []
    d_a = delta_for_auc(AUC_BEST)
    n_test = int(POOL_SIZE * TEST_FRACTION)
    for gap in GAPS:
        d_b = delta_for_auc(AUC_BEST - gap)
        for rho in [0.9]:
            flips = []
            for _ in range(N_REP_RETRAIN // 20):
                y, sa, sb = draw_scores(rng, POOL_SIZE, d_a, d_b, rho, PREVALENCE)
                x = np.column_stack([sa, sb])          # two features
                wins_a, n_ok = 0, 0
                for _ in range(20):
                    idx = rng.permutation(POOL_SIZE)
                    te, tr = idx[:n_test], idx[n_test:]
                    if y[tr].sum() < 5 or y[te].sum() < 2:
                        continue
                    m1 = LogisticRegression().fit(x[tr, :1], y[tr])
                    m2 = LinearDiscriminantAnalysis().fit(x[tr, 1:], y[tr])
                    a = auc(m1.predict_proba(x[te, :1])[:, 1], y[te])
                    b = auc(m2.predict_proba(x[te, 1:])[:, 1], y[te])
                    wins_a += a > b
                    n_ok += 1
                if n_ok:
                    p = wins_a / n_ok
                    flips.append(2 * p * (1 - p))
            rows.append(dict(
                true_gap=gap, correlation=rho, pool_size=POOL_SIZE, n_test=n_test,
                p_two_analysts_disagree=float(np.mean(flips)),
            ))
            print(f"  gap {gap:.2f}  retrained each split: "
                  f"two analysts disagree {np.mean(flips):.3f}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    print(f"seed {SEED}, better model true AUC {AUC_BEST}, prevalence {PREVALENCE}\n")
    print("A. fresh test set each replication")
    experiment_fresh_test_sets(rng).to_csv("results_model_comparison.csv", index=False)
    print("\nB. split lottery: one fixed dataset, fixed models, only the seed changes")
    experiment_split_lottery(rng).to_csv("results_model_split_lottery.csv", index=False)
    print("\nC. split lottery with the models retrained inside each split")
    experiment_retrained(rng).to_csv("results_model_retrained.csv", index=False)
    print("\nwrote results_model_comparison.csv, results_model_split_lottery.csv, "
          "results_model_retrained.csv")
