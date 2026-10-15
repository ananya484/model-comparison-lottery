# Which prediction model is better? A single random split often decides

Simulation code and complete output for the article "Which prediction model is better? A single
random split often decides: a simulation study with refitted models and calibrated inference"
(Ananya Sharma). The study follows the ADEMP framework (Morris, White and Crowther, Stat Med 2019).

## Experiments

| script | experiment | output |
|---|---|---|
| `exp_ab_scores.py A` | A - binormal scores, fresh test set each replication; naive rule vs paired DeLong; 1080 configurations x 4000 replications | `results/a_fresh_test.csv` |
| `exp_ab_scores.py B` | B - split lottery: one fixed dataset and fixed models, only the split seed varies; 360 configurations x 200 datasets x 100 splits | `results/b_split_lottery.csv` |
| `exp_c_refit.py calibrate` | calibration of the predictor-level mechanisms (coefficients giving the target AUC gaps) | `results/c_calibration.json` |
| `exp_c_refit.py run` | C - two logistic regressions, and tuned gradient boosting vs ridge regression, refitted inside each of 30 splits of 200 datasets per condition; population AUCs from 50 000-patient validation samples | `results/c_splits.csv.gz` (108 000 split records) |
| `exp_ab_scores.py D` | D - K = 2 to 10 candidate models | `results/d_many_models.csv` |
| `analyse_c.py` | summaries of C: strategy-level and conditional estimands, variance decomposition, DeLong coverage, naive vs Nadeau-Bengio repeated-split tests | `results/c_summary.csv` |
| `analyse_ab.py` | required test-set sizes from A | `results/a_required_n.csv` |
| `make_figures.py` | Figures 1-4 | `figures/` |

`common.py` holds the paired DeLong implementation (Sun and Xu's midrank algorithm), the binormal
generator and the seeding. Every configuration draws from its own stream derived from the master
seed 20261015, so results do not depend on parallel scheduling.

## Reproducing

```bash
pip install -r requirements.txt
python exp_c_refit.py calibrate
python exp_c_refit.py run
python exp_ab_scores.py A
python exp_ab_scores.py B
python exp_ab_scores.py D
python analyse_c.py
python analyse_ab.py
python make_figures.py
```

Experiment C fits about 216 000 models with internal tuning. On 22 cores it took about two hours;
A, B and D together take under 30 minutes.

## Earlier version

`v1-preprint/` holds the code and output of the earlier preprint (Research Square,
https://doi.org/10.21203/rs.3.rs-10794678/v1), kept unchanged for the record. Its refitting
experiment fitted single-feature models, which are monotone transformations of the scores and so
cannot change the AUC; the current Experiment C replaces it with genuine predictor-level models.

## Licence

MIT (see `LICENSE`).
