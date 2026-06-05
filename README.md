# Earthquake Damage Prediction Pipeline

This project builds a machine-learning pipeline to predict building earthquake damage grades. The pipeline loads the data, cleans missing and noisy values, engineers structural and geographic features, compares multiple prediction models, performs error analysis, and generates a submission file.

---

## Project Structure

```text
TASK-1/
├── data/
│   ├── train_values.csv
│   ├── train_labels.csv
│   └── test_values.csv
│
├── notebooks/
│   └── week01_exploration.ipynb
│
├── outputs/
│   ├── Team2_prediction.csv
│   ├── hard_cases.csv
│   ├── model_comparison.csv
│   ├── feature_importance_comparison.csv
│   └── stacking_weights.csv
│
├── src/
│   └── pipeline_sklearn.py
│
├── .gitignore
├── README.md
├── requirements.txt
├── run.sh
└── setup.sh
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/Bescher-Kilani/earthquake-damage-prediction.git Earthquake
cd Earthquake

# Run setup script once — creates venv, installs dependencies, creates folders
bash setup.sh

# Copy data files into data/
cp train_values.csv train_labels.csv test_values.csv data/
```

Required input files:

```text
data/train_values.csv
data/train_labels.csv
data/test_values.csv
```

---

## Run Pipeline

Use `run.sh` — it activates the virtual environment and runs the pipeline in one command:

```bash
# Generate submission file (default — Two-Stage RF+LGBM with tuned params)
bash run.sh

# Model comparison with 5-Fold CV
bash run.sh --compare

# Model comparison with single split (faster)
bash run.sh --compare --no-cv

# Compare selected models only
bash run.sh --compare --models "Random Forest,LightGBM,CatBoost,Voting (soft),Stacking"

# Soft vs hard voting
bash run.sh --compare --voting-mode hard

# Deep error analysis
bash run.sh --analyze

# Subsample for faster testing
bash run.sh --compare --sample 20000
```

Output:

```text
outputs/Team2_prediction.csv   — submission file
outputs/hard_cases.csv         — uncertain training samples
```

---

## Pipeline Steps

1. **Load data** from the `data/` folder.
2. **Split data** into training and validation sets before cleaning to avoid data leakage.
3. **Clean missing values:**
   - Treat `age = 0` as artificial missing data.
   - Replace artificial missing values with `NaN`.
   - Optionally create missing-value indicators such as `age_was_missing`.
   - Impute numeric values with the median.
   - Impute categorical values with the mode.
4. **Remove uninformative features:**
   - Drop constant columns.
   - Drop highly correlated numeric columns using a correlation threshold of `0.9`.
5. **Encode categorical features:**
   - Use Label Encoding for low-cardinality categorical features.
   - Use Target Encoding for high-cardinality geographic and engineered categorical features.
6. **Engineer structural features:**
   - `height_per_floor`, `slenderness`, `age_group`, `ground_height_binned`
7. **Recombine binary features:**
   - Combine `has_superstructure_*` columns into `superstructure_type`.
   - Combine `has_secondary_use_*` columns into `secondary_use_type`.
8. **Group rare categories** — replace infrequent categories with `OTHER`.
9. **Train and compare prediction models.**
10. **Analyze model performance and feature importance.**
11. **Predict on the test set.**
12. **Save predictions to `outputs/Team2_prediction.csv`.**

---

## Week 4: Prediction Models

### Usage — Week 4 Commands

```bash
# 5-Fold CV model comparison on full training data
bash run.sh --compare

# Single split comparison (faster)
bash run.sh --compare --no-cv

# Compare selected models only
bash run.sh --compare --models "Random Forest,LightGBM,XGBoost,CatBoost,Voting (soft),Stacking"

# Soft vs hard voting
bash run.sh --compare --voting-mode soft
bash run.sh --compare --voting-mode hard

# Custom voting weights (order: RF, LGBM, CatBoost)
bash run.sh --compare --voting-weights "1,5,3"

# Deep error analysis
bash run.sh --analyze

# Subsample for faster testing
bash run.sh --compare --sample 20000
```

---

In Week 4, we extended the pipeline from a single classifier to a full model-comparison framework. Primary metric: Accuracy. Secondary metric: MCC.

Models compared: Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost, k-Nearest Neighbors, Logistic Regression, Naive Bayes, Linear SVM, Voting Classifier, Stacking Classifier.

### Voting Classifier

The Voting model combines predictions from Random Forest, LightGBM, and CatBoost. Supports soft voting (average probabilities) and hard voting (majority vote).

### Stacking Classifier

Uses Random Forest, LightGBM, and CatBoost as base models with Logistic Regression as the final meta-model.

Stacking contribution summary:

| Base model | Normalized contribution |
|---|---:|
| LightGBM | 0.57 |
| CatBoost | 0.32 |
| Random Forest | 0.11 |

### Week 4 Results — Single Split

| Model | Accuracy | MCC | Training time |
|---|---:|---:|---:|
| Stacking | 0.7437 | 0.5239 | 127.0s |
| Random Forest | 0.7426 | 0.5185 | 10.3s |
| Voting soft | 0.7424 | 0.5198 | 27.5s |
| LightGBM | 0.7418 | 0.5209 | 3.7s |
| XGBoost | 0.7392 | 0.5146 | 32.6s |
| CatBoost | 0.7366 | 0.5087 | 10.4s |

### Error Analysis

Biggest confusion for all models: **Grade 3 predicted as Grade 2**.

| Model | Biggest confusion | Count | % of true Grade 3 | % of validation set |
|---|---|---:|---:|---:|
| Random Forest | Grade 3 → Grade 2 | 6,381 | 36.58% | 12.24% |
| LightGBM | Grade 3 → Grade 2 | 6,184 | 35.45% | 11.86% |
| XGBoost | Grade 3 → Grade 2 | 6,269 | 35.94% | 12.03% |
| CatBoost | Grade 3 → Grade 2 | 6,369 | 36.51% | 12.22% |
| Voting soft | Grade 3 → Grade 2 | 6,332 | 36.30% | 12.15% |
| Stacking | Grade 3 → Grade 2 | 6,063 | 34.76% | 11.63% |

### Stacking Confusion Matrix

| True grade \ Predicted grade | Predicted Grade 1 | Predicted Grade 2 | Predicted Grade 3 | Total |
|---|---:|---:|---:|---:|
| True Grade 1 | 2,642 | 2,325 | 58 | 5,025 |
| True Grade 2 | 1,119 | 24,815 | 3,718 | 29,652 |
| True Grade 3 | 75 | 6,063 | 11,306 | 17,444 |

### Stacking Classification Metrics

| Grade | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| Grade 1 | 0.6887 | 0.5258 | 0.5963 | 5,025 |
| Grade 2 | 0.7474 | 0.8369 | 0.7896 | 29,652 |
| Grade 3 | 0.7496 | 0.6481 | 0.6952 | 17,444 |
| Accuracy | | | 0.7437 | 52,121 |
| Macro avg | 0.7286 | 0.6703 | 0.6937 | 52,121 |
| Weighted avg | 0.7425 | 0.7437 | 0.7394 | 52,121 |

22% of validation samples are wrong for all four ensemble models simultaneously — a hard ceiling on easy gains.

---

## What Worked

### 1. Missing Value Indicators

```python
age_was_missing = age.isnull()
```

Missing-value indicators preserve useful information about whether a value was originally missing. Especially useful for the `age` feature.

### 2. Binary Feature Recombination

```python
has_superstructure_* → superstructure_type
has_secondary_use_* → secondary_use_type
```

Reduced many binary columns into more meaningful categorical features and allowed effective Target Encoding.

### 3. Target Encoding

```python
geo_level_1_id → geo_level_1_id_target_enc
geo_level_2_id → geo_level_2_id_target_enc
geo_level_3_id → geo_level_3_id_target_enc
superstructure_type → superstructure_type_target_enc
```

Target Encoding replaced arbitrary IDs with meaningful damage-risk values. Geographic target-encoded features were consistently the most important across all models.

### 4. Ensemble Models

Tree-based ensemble models performed best overall. Random Forest, LightGBM, Voting, and Stacking all achieved similar accuracy, with Stacking producing the best single-split result.

### 5. Stacking

Stacking improved the final result by combining LightGBM, CatBoost, and Random Forest. The meta-model relied most heavily on LightGBM, followed by CatBoost and Random Forest.

### 6. Two-Stage Classifier (Week 5)

RF handles confident predictions, LGBM handles uncertain ones. Consistent improvement across all 5 folds (+0.0012 Acc, +0.0059 MCC over RF baseline).

### 7. Hyperparameter Tuning (Week 6)

Optuna improved Acc from 0.7423 to 0.7551 (+0.0128) by finding better RF and LGBM parameters.

---

## What Did Not Work

### Feature Combinations

Adding too many feature combinations at the same time caused overfitting and reduced performance.

### Infrequent Category Merging Alone

Merging rare categories alone did not significantly improve performance because Target Encoding already handled rare categories reasonably well.

### Binning Continuous Variables

Binning continuous variables such as age reduced performance for tree-based models. Tree models can already learn effective split points directly from continuous features.

### One-Hot Encoding

One-Hot Encoding caused a large performance drop because high-cardinality geographic features were not handled effectively. Target Encoding worked better for these features.

### Sampling Strategies (Week 5)

None improved Accuracy or MCC for large tree ensembles on 260k samples. Classic trade-off: better minority recall always costs overall accuracy.

### Threshold Tuning (Week 5)

Lowered Accuracy. Models already use near-optimal thresholds internally.

### Boundary Feature Engineering (Week 5)

Negligible improvement (+0.0002). Grade 2/3 confusion is inherent ambiguity in the data, not a feature gap.

---

## Feature Importance

Across Random Forest, LightGBM, XGBoost, and CatBoost, the most consistently important features were:

```text
geo_level_3_id_target_enc       — strongest predictor (~85% importance in DT)
geo_level_2_id_target_enc
geo_level_1_id_target_enc
superstructure_type_target_enc
ground_height_binned_target_enc
foundation_type
roof_type
age
age_was_missing
count_families
```

Location-based risk, structure type, foundation type, roof type, and age-related features are the strongest predictors of earthquake damage.

---

## Key Learnings

### Target Encoding

- Effective for high-cardinality features.
- Especially useful for geographic IDs.
- Smoothing helps reduce overfitting.

### Feature Engineering

- Recombined structural features improved model usability.
- Continuous numeric features should generally be kept as continuous values for tree-based models.
- Rare category grouping can help stabilize categorical features.

### Model Selection

- Stacking achieved the best single-split result.
- Voting was competitive but did not outperform Stacking.
- Random Forest remained strong and much faster than Stacking.
- LightGBM provided strong MCC with very fast training time.

### Error Analysis

- The main weakness is confusing severe damage with medium damage.
- The most common error is Grade 3 predicted as Grade 2.
- Grade 2/3 confusion is inherent ambiguity — not addressable through feature engineering alone.

---

## Output Format

The final submission file follows this format:

```csv
building_id,damage_grade
300051,3
99355,2
...
```

Saved to:

```text
outputs/Team2_prediction.csv
```

---

## Week 5: Imbalanced Multi-Class Data

### Usage — Week 5 Commands

```bash
# Generate submission file (Two-Stage RF+LGBM, threshold=0.65)
bash run.sh

# Compare all 8 sampling strategies with 5-Fold CV (robust, recommended)
bash run.sh --sampling

# Compare with single 80/20 split (faster)
bash run.sh --sampling --sampling-no-cv

# Compare specific strategies only
bash run.sh --sampling --strategies "Baseline,SMOTE,class_weight"

# Quick test with subsampling
bash run.sh --sampling --sampling-no-cv --sample 20000

# Experiment 1: Threshold Tuning
bash run.sh --threshold-tuning

# Experiment 2: Boundary Feature Engineering
bash run.sh --boundary-features

# Experiment 3: Two-Stage Classifier (tests thresholds 0.50–0.70)
bash run.sh --two-stage

# Two-Stage with specific threshold
bash run.sh --two-stage --uncertainty-threshold 0.65
```

Available strategies for `--strategies`: `Baseline`, `class_weight`, `SMOTE`, `RandomUnderSampler`, `SMOTE+Under`, `EasyEnsemble`, `ClusterCentroids`, `ROSE`

Week 5 output files:

```text
outputs/Team2_prediction.csv        — final submission (Two-Stage RF+LGBM)
outputs/hard_cases.csv              — uncertain training samples (Stage 2 training set)
outputs/sampling_comparison.csv     — sampling strategy results
outputs/threshold_tuning.csv        — threshold tuning results
outputs/boundary_features.csv       — boundary feature experiment results
outputs/two_stage_classifier.csv    — two-stage results per threshold
```

---

In Week 5, we systematically compared 8 sampling strategies to address the class imbalance problem (Grade 1 = 9.6% of training data) and ran three additional experiments targeting the Grade 2/3 boundary confusion.

All final results use **5-Fold CV on 260k samples**.

---

### Part 1: Sampling Strategy Comparison

8 strategies tested on RF, LGBM, XGB, CatBoost. EasyEnsemble evaluated as standalone classifier.

#### Results — 5-Fold CV

| Strategy | Avg Acc | Avg MCC | Avg G1-Recall | Avg G3-Recall |
|---|---:|---:|---:|---:|
| Baseline | **0.7388** | **0.5130** | 0.497 | 0.633 |
| SMOTE+Under (Mix) | 0.7179 | 0.5037 | 0.691 | 0.687 |
| SMOTE | 0.7178 | 0.5033 | 0.688 | 0.689 |
| class_weight=balanced | 0.7008 | 0.4952 | 0.740 | 0.709 |
| ROSE | 0.6858 | 0.4881 | 0.827 | 0.736 |
| RandomUnderSampler | 0.6768 | 0.4795 | 0.842 | 0.735 |
| EasyEnsemble | 0.6587 | 0.4517 | 0.793 | 0.735 |
| ClusterCentroids | 0.4450 | 0.3241 | 0.919 | 0.810 |

Best per model — all 4 models best with Baseline (no sampling):

| Model | Accuracy | MCC | G1-Recall |
|---|---:|---:|---:|
| RF | 0.7411 | 0.5153 | 0.469 |
| LGBM | 0.7408 | 0.5184 | 0.526 |
| XGB | 0.7383 | 0.5128 | 0.509 |
| CatBoost | 0.7350 | 0.5054 | 0.485 |

#### Key Findings — Sampling

Sampling does not improve Accuracy or MCC. Baseline wins on both metrics across all models. Classic imbalanced data trade-off: every strategy that improves G1-Recall costs Accuracy proportionally.

```
Baseline:           Acc=0.74   G1-Recall=0.50
SMOTE:              Acc=0.72   G1-Recall=0.69   (−0.02 Acc, +0.19 Recall)
class_weight:       Acc=0.70   G1-Recall=0.74   (−0.04 Acc, +0.24 Recall)
RandomUnderSampler: Acc=0.68   G1-Recall=0.84   (−0.06 Acc, +0.34 Recall)
ClusterCentroids:   Acc=0.44   G1-Recall=0.92   (−0.30 Acc  — unusable)
```

Best compromise if G1-Recall matters: **SMOTE + LGBM** (Acc=0.7342, G1-Recall=0.614, only −0.007 Acc loss).

---

### Part 2: Additional Experiments

#### Experiment 1 — Threshold Tuning (5-Fold CV)

Optimised probability thresholds t12 (Grade1/2) and t23 (Grade2/3) to maximise Accuracy.

| Model | Method | Acc | MCC | G1-Recall | G3-Recall |
|---|---|---:|---:|---:|---:|
| RF | Default (argmax) | 0.7411 | 0.5153 | 0.469 | 0.629 |
| RF | Tuned (t12=0.35, t23=0.37) | 0.7199 | 0.5072 | 0.621 | 0.736 |
| LGBM | Default (argmax) | 0.7408 | 0.5184 | 0.526 | 0.638 |
| LGBM | Tuned (t12=0.40, t23=0.41) | 0.7246 | 0.5106 | 0.604 | 0.708 |

**Verdict:** Threshold tuning lowers Accuracy (RF: −0.021, LGBM: −0.016). Not useful when Accuracy is the primary metric.

---

#### Experiment 2 — Boundary Feature Engineering (5-Fold CV)

Added 5 new interaction features: `superstructure_x_foundation`, `superstructure_x_roof`, `age_x_floors`, `floors_x_area`, `damage_risk_score`

| Model | Variant | Acc | MCC | G3-Recall |
|---|---|---:|---:|---:|
| RF | Baseline | 0.7411 | 0.5153 | 0.629 |
| RF | + BoundaryFeatures | 0.7413 | 0.5159 | 0.629 |
| LGBM | Baseline | 0.7408 | 0.5184 | 0.638 |
| LGBM | + BoundaryFeatures | 0.7409 | 0.5185 | 0.638 |

**Verdict:** Negligible improvement. Confirms Grade 2/3 confusion is inherent ambiguity, not a feature gap.

---

#### Experiment 3 — Two-Stage Classifier (5-Fold CV)

Stage 1: RF on all samples. Stage 2: LGBM on uncertain samples. Five thresholds tested.

| Threshold | Acc | Delta Acc | MCC | G1-Recall | Uncertain/fold |
|---|---:|---:|---:|---:|---:|
| 0.50 | 0.7413 | +0.0002 | 0.5162 | 0.478 | 981 |
| 0.55 | 0.7418 | +0.0007 | 0.5187 | 0.504 | 5,867 |
| 0.60 | 0.7419 | +0.0008 | 0.5200 | 0.519 | 10,724 |
| **0.65** | **0.7423** | **+0.0012** | **0.5211** | **0.527** | **15,670** |
| 0.70 | 0.7421 | +0.0010 | 0.5206 | 0.526 | 20,605 |

**Best result of Week 5.** Threshold=0.65 consistently improves both Accuracy (+0.0012) and MCC (+0.0059) over 5 folds.

### Week 5 — Final Decision

```
Submission model:     Two-Stage RF + LGBM (threshold=0.65)
Accuracy (5-Fold CV): 0.7423   (+0.0012 over RF Baseline)
MCC (5-Fold CV):      0.5211   (+0.0059 over RF Baseline)
```

### What Worked in Week 5

**Two-Stage Classifier** — RF handles confident predictions, LGBM handles uncertain ones. Consistent small gains on both Accuracy and MCC across all 5 folds.

### What Did Not Work in Week 5

**Sampling strategies** — none improved Accuracy or MCC over baseline for large tree ensemble models on 260k samples.

**Threshold Tuning** — optimising thresholds for Accuracy worsens Accuracy; the model already uses near-optimal thresholds internally.

**Boundary Feature Engineering** — new interaction features bring negligible improvement, confirming the Grade 2/3 confusion is inherent ambiguity not a feature gap.

---

## Week 6: Hyperparameter Tuning

### Usage — Week 6 Commands

```bash
# Random Search (20 iterations, 5-Fold CV on full 260k)
bash run.sh --tune-random

# Optuna Bayesian Optimization (30 trials, 5-Fold CV on full 260k)
bash run.sh --tune-optuna

# Nested CV for fair performance estimate (3 outer × 3 inner folds, 30 trials)
bash run.sh --tune-nested-cv

# Custom number of trials
bash run.sh --tune-optuna --tune-trials 50
bash run.sh --tune-random --tune-n-iter 30

# With subsampling for faster testing
bash run.sh --tune-random --sample 80000
bash run.sh --tune-optuna --tune-trials 20 --sample 80000
bash run.sh --tune-nested-cv --tune-trials 10 --sample 80000
```

---

In Week 6, we tuned the hyperparameters of the Two-Stage RF+LGBM classifier using three methods: Random Search, Optuna (Bayesian Optimization), and Nested CV for fair evaluation.

RF (Stage 1) tuned with 5-Fold CV on full 260k. LGBM (Stage 2) tuned with 3-Fold CV on ~75k hard cases subset. All jobs run **sequentially** (not in parallel) to avoid server crashes.

### Tuning Methods

**Random Search** (`--tune-random`) — tests 20 random combinations via `RandomizedSearchCV`. Fast baseline, no learning between iterations.

**Optuna** (`--tune-optuna`) — Bayesian Optimization with TPE sampler. Learns from previous trials and focuses on promising regions. 30 trials per model.

**Nested CV** (`--tune-nested-cv`) — fair performance evaluation to quantify optimistic bias of normal CV. Outer loop: 3 folds (evaluation). Inner loop: 3 folds + Optuna (tuning). Outer test folds are never seen during parameter selection.

### Why No Grid Search

Grid Search tests all combinations in a fixed raster. With 4 parameters × 4 values each = 256 combinations × 5-Fold CV × ~25s per RF fit = ~9h on the full 260k dataset. Random Search achieves comparable results with 20 iterations (~30 min) because only 1-2 parameters typically dominate the performance. Optuna is even more efficient by learning which regions to search.

### Runtimes

| Method | RF Stage 1 | LGBM Stage 2 | Total |
|---|---:|---:|---:|
| Random Search (20 iter) | ~7 min | ~3 min | ~10 min |
| Optuna (30 trials) | ~15 min | ~4 min | ~19 min |
| Nested CV (3×3, 30 trials) | ~45 min | — (default params) | ~45 min |

All jobs run **sequentially** on the full 260k dataset. Parallel execution caused server crashes due to LGBM spawning too many threads across jobs.

**RF (Stage 1):**

| Parameter | Search space | Meaning |
|---|---|---|
| `n_estimators` | [100, 200, 300, 500] | Number of trees |
| `max_depth` | 8–25 | Tree depth (complexity) |
| `min_samples_leaf` | 1–20 | Min samples per leaf (regularization) |
| `max_features` | sqrt, log2, 0.3, 0.5 | Features per split (diversity) |

**LGBM (Stage 2):**

| Parameter | Search space | Meaning |
|---|---|---|
| `n_estimators` | [200, 300, 500] | Boosting rounds |
| `learning_rate` | [0.01, 0.05, 0.1] | Step size per round |
| `num_leaves` | 31–127 | Tree complexity |
| `min_child_samples` | 10–50 | Min samples per leaf (regularization) |

### Results

#### Random Search (20 iterations, full 260k)

| Model | Best params | Acc (CV) |
|---|---|---:|
| RF | depth=18, max_f=0.3, leaf=4, n=500 | 0.7554 |
| LGBM | lr=0.1, leaves=65, child=17, n=500 | 0.5677* |

#### Optuna (30 trials, full 260k)

| Model | Best params | Acc (CV) |
|---|---|---:|
| RF | n=500, depth=16, leaf=6, max_f=0.5 | 0.7551 |
| LGBM | n=500, lr=0.05, leaves=76, child=21 | 0.5717* |

*LGBM Acc measured on hard cases subset only (28.9% of data). Lower value is expected — hard cases are the most difficult samples by definition.

#### Nested CV (3 outer × 3 inner folds, 30 trials, full 260k)

| Outer Fold | Best RF Params | Inner Acc | Outer Acc | Outer MCC |
|---|---|---:|---:|---:|
| 1 | depth=14, leaf=3, n=200, max_f=0.5 | 0.7526 | 0.7520 | 0.5396 |
| 2 | depth=18, leaf=5, n=200, max_f=0.3 | 0.7516 | 0.7537 | 0.5426 |
| 3 | depth=19, leaf=13, n=100, max_f=0.5 | 0.7511 | 0.7547 | 0.5449 |
| **Mean** | | **0.7518** | **0.7535** | **0.5424** |

#### Normal CV vs Nested CV

| Method | Acc | Notes |
|---|---:|---|
| Normal CV (Optuna RF) | 0.7551 | Used for parameter selection |
| Nested CV | 0.7535 | Fair unbiased estimate |
| Optimistic bias | 0.0016 | Very small — tuning is stable |

Bias of 0.0016 confirms the tuned parameters generalize well and are not overfit to the CV folds.

### Final Model — Week 6

Optuna params used for final submission:

```python
# Stage 1 — RF (tuned with Optuna, 30 trials, 5-Fold CV on 260k)
n_estimators=500, max_depth=16, min_samples_leaf=6, max_features=0.5

# Stage 2 — LGBM (tuned with Optuna, 30 trials, 3-Fold CV on ~75k hard cases)
n_estimators=500, learning_rate=0.05, num_leaves=76, min_child_samples=21
```

### What Worked in Week 6

**Optuna** — converged after ~15 trials. Both Random Search and Optuna agreed on `n_estimators=500` and `max_features=0.5`, confirming stable results.

**Small optimistic bias (0.0016)** — Nested CV confirms tuned params generalize well.

### What Did Not Work in Week 6

**Parallel tuning jobs** — running all three jobs simultaneously caused server crashes due to LGBM spawning too many threads. Solution: run sequentially.

**LGBM tuning with 5-Fold CV** — ~2.5h per trial on 75k hard cases. Fixed by switching to 3-Fold CV for LGBM tuning.

---

## Overall Performance

| Metric | Week 2 | Week 3 | Week 4 | Week 5 (CV) | Week 6 (CV) |
|---|---:|---:|---:|---:|---:|
| MCC | 0.4077 | 0.5023 | 0.5239 | 0.5211 | 0.5424 |
| Accuracy | 0.6881 | 0.7327 | 0.7437 | 0.7423 | 0.7551 |

Week 4: single split. Week 5–6: 5-Fold CV mean. Week 6 Nested CV Acc = 0.7535 (fair estimate).

---

## Open Questions / Next Steps

- Final presentation (09.06.)
- Report submission (15.06.)