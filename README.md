# The declared winner of a clinical prediction model comparison is often decided by the random split

Simulation code and output for the preprint of the same name.

A model comparison evaluates two models on one test set and names the higher area under the
curve as better. This code measures how often that declaration is correct, how much the
winner's reported margin overstates the truth, and how often two analysts holding the same
data and the same models would announce different winners because they split it with
different random seeds.

## Reproducing

```bash
pip install -r requirements.txt
python sim_model_comparison.py
```

Runtime is a few minutes. The script is deterministic given the seed `20260823` and rewrites
all three CSV files byte for byte, so a successful run reproduces every number in the paper's
two tables exactly.

## Files

| file | contents |
|---|---|
| `sim_model_comparison.py` | all three experiments; no other code is needed |
| `results_model_comparison.csv` | Experiment A - fresh test set each replication, 40 configurations, 4000 replications each |
| `results_model_split_lottery.csv` | Experiment B - one fixed pool of 1000 patients resplit 200 times, over 200 pools |
| `results_model_retrained.csv` | Experiment C - as B, with both models refitted inside every split |

## What the columns mean

- `true_gap` — how far below the better model (true AUC 0.80) the second model sits
- `correlation` — correlation between the two models' errors on the same patient. High
  values mean two models that make similar mistakes, which makes them *easier* to rank
  because the shared error cancels.
- `p_correct_winner` — proportion of replications in which the declared winner was the
  genuinely better model. At a true gap of zero this must be 0.5, and is.
- `winner_margin_mean` — mean margin announced by whichever model won
- `margin_inflation` — that margin minus the true gap. Positive everywhere; at a true gap
  of zero the whole margin is spurious.
- `p_two_analysts_disagree` — 2p(1-p) where p is the proportion of splits favouring the
  better model on a fixed pool. This is the split lottery.
- `worst_pool_disagreement` — the maximum of that quantity across pools. On some datasets
  it reaches 0.5, meaning the reported winner is a fair coin toss.

## Licence

MIT.
