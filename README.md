# Earthquake Damage Prediction

Predicting the damage grade of buildings after the 2015 Gorkha earthquake in Nepal (ordinal, 3 classes). Built as Phase 1 of the Data Science Lab 2026 at KIT.

**Final model:** two-stage Random Forest + LightGBM, tuned with Optuna
**Result (5-fold CV):** Accuracy **0.7551** (micro-F1), MCC **0.5424**. Nested CV gives a fair estimate of 0.7535.

This page is the short overview. All experiments, tables and confusion matrices are in [docs/FULL_DOCUMENTATION.md](docs/FULL_DOCUMENTATION.md).

## Task

- Competition: [Richter's Predictor: Modeling Earthquake Damage](https://www.drivendata.org/competitions/57/nepal-earthquake/) (DrivenData)
- Data: 260,601 training and 86,868 test buildings, 38 features, no missing values
- Target `damage_grade`: 1 (low), 2 (medium), 3 (near-complete destruction)
- Metric: micro-averaged F1, which equals accuracy for single-label multi-class problems. MCC as secondary metric.

## Project Structure

```text
.
├── data/                    # not tracked, see data/README.md
├── docs/
│   └── FULL_DOCUMENTATION.md  # complete results of all weeks
├── notebooks/
│   └── week01_exploration.ipynb
├── outputs/                 # generated results (not tracked)
├── src/
│   └── pipeline_sklearn.py  # full pipeline, all experiments via CLI flags
├── requirements.txt
├── setup.sh
└── run.sh
```

## Setup

```bash
git clone https://github.com/Bescher-Kilani/earthquake-damage-prediction.git
cd earthquake-damage-prediction
bash setup.sh
```

Download `train_values.csv`, `train_labels.csv` and `test_values.csv` from the [competition page](https://www.drivendata.org/competitions/57/nepal-earthquake/data/) into `data/` (details in [data/README.md](data/README.md)).

## Usage

```bash
bash run.sh                         # final model, writes outputs/Team2_prediction.csv
bash run.sh --compare               # Week 4: model comparison (5-fold CV)
bash run.sh --compare --no-cv       # same, single split (faster)
bash run.sh --analyze               # Week 4: error analysis
bash run.sh --sampling              # Week 5: 8 sampling strategies
bash run.sh --threshold-tuning      # Week 5: threshold tuning
bash run.sh --boundary-features     # Week 5: boundary features
bash run.sh --two-stage             # Week 5: two-stage classifier
bash run.sh --tune-random           # Week 6: random search
bash run.sh --tune-optuna           # Week 6: Optuna
bash run.sh --tune-nested-cv        # Week 6: nested CV
bash run.sh --compare --sample 20000  # any experiment on a subsample for quick tests
```

Run `python3 src/pipeline_sklearn.py --help` for all options.

## Progress

| Week | Topic | Accuracy | MCC |
|---|---|---:|---:|
| 2 | Cleaning & pipeline | 0.6881 | 0.4077 |
| 3 | Feature engineering | 0.7327 | 0.5023 |
| 4 | Prediction models (single split) | 0.7437 | 0.5239 |
| 5 | Imbalanced data, two-stage (5-fold CV) | 0.7423 | 0.5211 |
| 6 | Hyperparameter tuning (5-fold CV) | 0.7551 | 0.5424 |

## Week 1: Exploration

Notebook: [`notebooks/week01_exploration.ipynb`](notebooks/week01_exploration.ipynb)

- Target is imbalanced: grade 1 = 9.6%, grade 2 = 56.9%, grade 3 = 33.5%.
- `age` is right-skewed with outliers up to 995 years (likely placeholder values).
- Many binary `has_secondary_use_*` and `has_superstructure_*` columns are >95% one class.
- Highest mutual information with the target: `geo_level_3_id` (0.30), `geo_level_2_id` (0.23), `geo_level_1_id` (0.13), `foundation_type` (0.09).

## Week 2: Cleaning & Pipeline

- Split before cleaning; all statistics are fit on the training part only (no leakage).
- `age = 0` treated as artificial missing value, with an `age_was_missing` indicator. Median/mode imputation.
- Dropped constant columns and numeric columns with correlation > 0.9.
- Label encoding for low-cardinality, target encoding for high-cardinality categoricals.

## Week 3: Feature Engineering

**Worked**
- Missing-value indicator (`age_was_missing`)
- Recombining `has_superstructure_*` / `has_secondary_use_*` into single categoricals
- Target encoding of `geo_level_1/2/3_id` and `superstructure_type`

**Did not work**
- Many feature combinations at once (overfitting)
- Merging rare categories alone (target encoding already handles them)
- Binning continuous variables such as `age` (trees find splits themselves)
- One-hot encoding of high-cardinality geo IDs

## Week 4: Prediction Models

Compared Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost, kNN, Logistic Regression, Naive Bayes, Linear SVM, Voting and Stacking.

| Model | Accuracy | MCC | Train time |
|---|---:|---:|---:|
| Stacking (RF + LGBM + CatBoost, LR meta-model) | 0.7437 | 0.5239 | 127.0 s |
| Random Forest | 0.7426 | 0.5185 | 10.3 s |
| Voting (soft) | 0.7424 | 0.5198 | 27.5 s |
| LightGBM | 0.7418 | 0.5209 | 3.7 s |
| XGBoost | 0.7392 | 0.5146 | 32.6 s |
| CatBoost | 0.7366 | 0.5087 | 10.4 s |

- Geographic target-encoded features are the most important across all models (`geo_level_3_id` ~85% in the decision tree).
- Main error for every model: **grade 3 predicted as grade 2** (34.8% of true grade 3 for Stacking).
- 22% of validation samples are misclassified by all four ensembles at once.

## Week 5: Imbalanced Multi-Class Data

8 sampling strategies, evaluated with 5-fold CV on all 260k samples:

| Strategy | Accuracy | MCC | Grade 1 recall |
|---|---:|---:|---:|
| **Baseline (no sampling)** | **0.7388** | **0.5130** | 0.497 |
| SMOTE + Under | 0.7179 | 0.5037 | 0.691 |
| SMOTE | 0.7178 | 0.5033 | 0.688 |
| class_weight=balanced | 0.7008 | 0.4952 | 0.740 |
| ROSE | 0.6858 | 0.4881 | 0.827 |
| RandomUnderSampler | 0.6768 | 0.4795 | 0.842 |
| EasyEnsemble | 0.6587 | 0.4517 | 0.793 |
| ClusterCentroids | 0.4450 | 0.3241 | 0.919 |

- Sampling does not help: every gain in minority recall costs overall accuracy.
- Threshold tuning lowered accuracy (RF −0.021, LightGBM −0.016).
- Boundary features (5 interactions) gave +0.0002, so the grade 2/3 confusion is inherent ambiguity, not a feature gap.
- **Two-stage classifier** (RF on all samples, LightGBM on uncertain samples with max probability < 0.65): +0.0012 accuracy and +0.0059 MCC over the RF baseline in all 5 folds.

## Week 6: Hyperparameter Tuning

Random search (20 iterations), Optuna/TPE (30 trials) and nested CV (3 outer × 3 inner folds). RF tuned with 5-fold CV on all data, LightGBM with 3-fold CV on the ~75k uncertain samples.

| Method | RF accuracy (CV) |
|---|---:|
| Default | 0.7423 |
| Random search | 0.7554 |
| Optuna | 0.7551 |
| Nested CV (unbiased) | 0.7535 |

- Optimistic bias of normal CV is only 0.0016, so the tuned parameters generalize.
- Grid search was skipped: 256 combinations × 5 folds would take ~9 h per model.
- Run tuning jobs sequentially. In parallel, LightGBM spawned too many threads and crashed the server.

**Final parameters**

```python
# Stage 1: Random Forest
n_estimators=500, max_depth=16, min_samples_leaf=6, max_features=0.5
# Stage 2: LightGBM (uncertain samples only)
n_estimators=500, learning_rate=0.05, num_leaves=76, min_child_samples=21
```

## Authors

Team 2, Data Science Lab 2026, Karlsruhe Institute of Technology.