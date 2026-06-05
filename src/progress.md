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
│   ├── model_comparison.csv
│   ├── hard_cases.csv
│   ├── feature_importance_comparison.csv
│   └── stacking_weights.csv
│
├── src/
│   └── pipeline.py
│
├── .gitignore
├── pipeline_readme.md
├── README.md
└── requirements.txt
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

The setup script creates a virtual environment, installs all dependencies from `requirements.txt`, and creates the `outputs/` and `data/` folders.

Required input files:

```text
data/train_values.csv
data/train_labels.csv
data/test_values.csv
```

---

## Run Pipeline

For convenience, use `run.sh` — it activates the virtual environment and runs the pipeline in one command:

```bash
# Run default pipeline (Two-Stage RF+LGBM → submission file)
bash run.sh

# Pass any pipeline argument
bash run.sh --compare
bash run.sh --sampling
bash run.sh --two-stage
bash run.sh --compare --no-cv --models "Random Forest,LightGBM"
```

Or activate the virtual environment manually and run directly:

```bash
source venv/bin/activate
python3 src/pipeline_sklearn.py
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
   - `height_per_floor`
   - `slenderness`
   - `age_group`
   - `ground_height_binned`
7. **Recombine binary features:**
   - Combine `has_superstructure_*` columns into `superstructure_type`.
   - Combine `has_secondary_use_*` columns into `secondary_use_type`.
8. **Group rare categories:**
   - Replace infrequent categories with `OTHER`.
9. **Train and compare prediction models.**
10. **Analyze model performance and feature importance.**
11. **Predict on the test set.**
12. **Save predictions to `outputs/Team2_prediction.csv`.**

---

## Week 4: Prediction Models

In Week 4, we extended the pipeline from a single classifier to a model-comparison framework. The preprocessing and feature-engineering steps remain the same, but we now compare multiple prediction models using Accuracy as the primary metric and MCC as the secondary metric.

The compared models include:

- Decision Tree
- Random Forest
- LightGBM
- XGBoost
- CatBoost
- k-Nearest Neighbors
- Logistic Regression
- Naive Bayes
- Linear SVM
- Voting Classifier
- Stacking Classifier

### Voting Classifier

The Voting model combines predictions from Random Forest, LightGBM, and CatBoost. It supports both:

- **Soft voting:** averages predicted probabilities.
- **Hard voting:** uses majority vote.

The voting mode can be selected from the command line using:

```bash
--voting-mode soft
```

or:

```bash
--voting-mode hard
```

Custom voting weights can also be tested:

```bash
--voting-weights "1,5,3"
```

The voting weights follow this order:

```text
Random Forest, LightGBM, CatBoost
```

### Stacking Classifier

The Stacking model uses Random Forest, LightGBM, and CatBoost as base models, with Logistic Regression as the final meta-model.

The stacking model also reports a contribution summary. These values are not simple voting weights. Instead, they are derived from the final Logistic Regression coefficients and show the relative influence of each base model.

Latest stacking contribution summary:

| Base model | Normalized contribution |
|---|---:|
| LightGBM | 0.57 |
| CatBoost | 0.32 |
| Random Forest | 0.11 |

---

## Results

### Final Performance

| Metric | Week 2 | Week 3 | Week 4 |
|---|---:|---:|---:|
| MCC | 0.4077 | 0.5023 | 0.5239 |
| Accuracy | 0.6881 | 0.7327 | 0.7437 |

Features after preprocessing and feature engineering:

```text
24 features
```

### Week 3 vs. Week 2

| Metric | Change |
|---|---:|
| MCC | +23.2% |
| Accuracy | +6.5% |

### Week 4 vs. Week 3

| Metric | Change |
|---|---:|
| MCC | +4.3% |
| Accuracy | +1.5% |

### Week 4 Model Comparison

Latest single-split analysis results:

| Model | Accuracy | MCC | Training time |
|---|---:|---:|---:|
| Stacking | 0.7437 | 0.5239 | 127.0s |
| Random Forest | 0.7426 | 0.5185 | 10.3s |
| Voting soft | 0.7424 | 0.5198 | 27.5s |
| LightGBM | 0.7418 | 0.5209 | 3.7s |
| XGBoost | 0.7392 | 0.5146 | 32.6s |
| CatBoost | 0.7366 | 0.5087 | 10.4s |

The best model in this analysis was **Stacking**, with:

```text
Accuracy = 0.7437
MCC      = 0.5239
```

---

## Error Analysis

Deep error analysis was added to compare the main ensemble models:

- Random Forest
- LightGBM
- XGBoost
- CatBoost
- Voting
- Stacking

The analysis includes:

- Confusion matrices
- Per-class precision, recall, and F1-score
- Hard-case analysis
- Feature profile comparison
- Feature importance comparison
- Stacking contribution summary

### Biggest Confusion per Model

The biggest confusion for all models was:

```text
True Grade 3 predicted as Grade 2
```

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
| Accuracy |  |  | 0.7437 | 52,121 |
| Macro avg | 0.7286 | 0.6703 | 0.6937 | 52,121 |
| Weighted avg | 0.7425 | 0.7437 | 0.7394 | 52,121 |

---

## What Worked

### 1. Missing Value Indicators

```python
age_was_missing = age.isnull()
```

Missing-value indicators preserve useful information about whether a value was originally missing. This was especially useful for the `age` feature.

### 2. Binary Feature Recombination

```python
has_superstructure_* → superstructure_type
has_secondary_use_* → secondary_use_type
```

This reduced many binary columns into more meaningful categorical features and allowed effective Target Encoding.

### 3. Target Encoding

```python
geo_level_1_id → geo_level_1_id_target_enc
geo_level_2_id → geo_level_2_id_target_enc
geo_level_3_id → geo_level_3_id_target_enc
superstructure_type → superstructure_type_target_enc
```

Target Encoding replaced arbitrary IDs with meaningful damage-risk values. The geographic target-encoded features were consistently among the most important features across models.

### 4. Ensemble Models

Tree-based ensemble models performed best overall. Random Forest, LightGBM, Voting, and Stacking all achieved similar accuracy, with Stacking producing the best single-split result.

### 5. Stacking

Stacking improved the final result by combining the strengths of LightGBM, CatBoost, and Random Forest. The meta-model relied most heavily on LightGBM, followed by CatBoost and Random Forest.

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

---

## Feature Importance

Across Random Forest, LightGBM, XGBoost, and CatBoost, the most consistently important features were:

```text
geo_level_3_id_target_enc
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

These results show that location-based risk, structure type, foundation type, roof type, and age-related features are the strongest predictors of earthquake damage.

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
- Future improvements should focus on separating Grade 2 and Grade 3 more clearly.

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
## Week 6: Hyperparameter Tuning

### Usage — Week 6 Commands

```bash
# Random Search (RandomizedSearchCV, 20 iterations, 5-Fold CV)
python3 src/pipeline_sklearn.py --tune-random

# Optuna (Bayesian Optimization, 30 trials, 5-Fold CV)
python3 src/pipeline_sklearn.py --tune-optuna

# Nested CV (fair performance estimate, 3 outer × 3 inner folds, 30 trials)
python3 src/pipeline_sklearn.py --tune-nested-cv

# With subsampling for faster testing
python3 src/pipeline_sklearn.py --tune-random --sample 80000
python3 src/pipeline_sklearn.py --tune-optuna --tune-trials 20 --sample 80000
python3 src/pipeline_sklearn.py --tune-nested-cv --tune-trials 10 --sample 80000
```

---

In Week 6, we tuned the hyperparameters of the Two-Stage RF+LGBM classifier using three methods: Random Search, Optuna (Bayesian Optimization), and Nested CV for fair evaluation.

All tuning runs on the full 260k training set. RF (Stage 1) tuned with 5-Fold CV. LGBM (Stage 2) tuned with 3-Fold CV on the ~75k hard cases subset.

---

### Tuning Methods

**Random Search** — tests 20 random hyperparameter combinations via RandomizedSearchCV. Fast baseline, no learning between iterations.

**Optuna** — Bayesian Optimization using TPE sampler. Learns from previous trials and focuses search on promising parameter regions. 30 trials per model.

**Nested CV** — fair performance evaluation. Outer loop: 3 folds (test). Inner loop: 3 folds + Optuna (tune). Outer test folds never seen during parameter selection. Quantifies optimistic bias of normal CV.

---

### Hyperparameters Tuned

**RF (Stage 1):**

| Parameter | Search space | Meaning |
|---|---|---|
| `n_estimators` | [100, 200, 300, 500] | Number of trees |
| `max_depth` | 8–25 | Tree depth (complexity) |
| `min_samples_leaf` | 1–20 | Minimum samples per leaf (regularization) |
| `max_features` | sqrt, log2, 0.3, 0.5 | Features per split (tree diversity) |

**LGBM (Stage 2):**

| Parameter | Search space | Meaning |
|---|---|---|
| `n_estimators` | [200, 300, 500] | Boosting rounds |
| `learning_rate` | [0.01, 0.05, 0.1] | Step size per round |
| `num_leaves` | 31–127 | Tree complexity |
| `min_child_samples` | 10–50 | Minimum samples per leaf (regularization) |

---

### Results

#### Random Search (20 iterations, full 260k)

```
RF:   max_depth=18, max_features=0.3, min_samples_leaf=4, n_estimators=500
      Acc (CV) = 0.7554

LGBM: learning_rate=0.1, min_child_samples=17, n_estimators=500, num_leaves=65
      Acc (CV) = 0.5677  (on hard cases subset only)
```

#### Optuna (30 trials, full 260k)

```
RF:   n_estimators=500, max_depth=16, min_samples_leaf=6, max_features=0.5
      Acc (CV) = 0.7551

LGBM: n_estimators=500, learning_rate=0.05, num_leaves=76, min_child_samples=21
      Acc (CV) = 0.5717  (on hard cases subset only)
```

#### Nested CV (3 outer × 3 inner folds, 30 trials, full 260k)

| Outer Fold | Best Params | Inner Acc | Outer Acc | Outer MCC |
|---|---|---:|---:|---:|
| 1 | depth=14, leaf=3, n=200, max_f=0.5 | 0.7526 | 0.7520 | 0.5396 |
| 2 | depth=18, leaf=5, n=200, max_f=0.3 | 0.7516 | 0.7537 | 0.5426 |
| 3 | depth=19, leaf=13, n=100, max_f=0.5 | 0.7511 | 0.7547 | 0.5449 |
| **Mean** | | **0.7518** | **0.7535** | **0.5424** |

#### Normal CV vs Nested CV

| Method | Acc | Notes |
|---|---:|---|
| Normal CV (Optuna) | 0.7551 | Used for parameter selection |
| Nested CV | 0.7535 | Fair unbiased estimate |
| Optimistic bias | 0.0016 | Very small — tuning is stable |

The bias of 0.0016 confirms the tuned parameters generalize well and are not overfit to the CV folds.

---

### Final Model — Week 6

Optuna params used for final submission:

```python
# Stage 1 — RF
n_estimators=500, max_depth=16, min_samples_leaf=6, max_features=0.5

# Stage 2 — LGBM
n_estimators=500, learning_rate=0.05, num_leaves=76, min_child_samples=21
```

### Updated Overall Performance

| Metric | Week 2 | Week 3 | Week 4 | Week 5 (CV) | Week 6 (CV) |
|---|---:|---:|---:|---:|---:|
| MCC | 0.4077 | 0.5023 | 0.5239 | 0.5211 | 0.5424 |
| Accuracy | 0.6881 | 0.7327 | 0.7437 | 0.7423 | 0.7551 |

Note: Week 6 reports Nested CV mean scores (fair estimate). Normal CV Acc = 0.7551.

---

### What Worked in Week 6

**Optuna** — converged after ~15 trials, confirming 30 trials is sufficient. Both Random Search and Optuna agreed on `n_estimators=500` and `max_features=0.5`.

**Small optimistic bias (0.0016)** — Nested CV confirms tuned params are robust.

---

### What Did Not Work in Week 6

**Parallel tuning jobs** — running all three jobs simultaneously caused crashes due to LGBM spawning too many threads. Solution: run sequentially.

**LGBM tuning with 5-Fold CV** — ~2.5h per trial on 75k hard cases. Fixed by switching to 3-Fold CV for LGBM.

---

## Open Questions / Next Steps

- Final presentation (09.06.)
- Report submission (15.06.)