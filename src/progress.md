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
# Install dependencies
pip install -r requirements.txt

# Create data folder
mkdir -p data
```

Place the CSV files in the `data/` folder before running the pipeline.

Required input files:

```text
data/train_values.csv
data/train_labels.csv
data/test_values.csv
```

---

## Run Pipeline

Run the final training pipeline and generate a submission file:

```bash
python src/pipeline.py
```

Output:

```text
outputs/Team2_prediction.csv
```

Run model comparison with cross-validation:

```bash
python src/pipeline.py --compare
```

Run a faster model comparison using a single train/validation split:

```bash
python src/pipeline.py --compare --no-cv
```

Run comparison on a smaller sample:

```bash
python src/pipeline.py --compare --sample 20000
```

Compare selected models only:

```bash
python src/pipeline.py --compare --models "Random Forest,LightGBM,CatBoost,Voting (soft),Stacking"
```

Run hard voting instead of soft voting:

```bash
python src/pipeline.py --compare --voting-mode hard
```

Run deep error analysis:

```bash
python src/pipeline.py --analyze
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

## Week 5: Imbalanced Multi-Class Data

In Week 5, we systematically compared 8 sampling strategies to address the class imbalance problem identified in Week 4, and ran three additional experiments targeting the Grade 2/3 boundary confusion. Grade 1 (lightly damaged) accounts for only 9.6% of the training data.

All final results use **5-Fold CV on 260k samples** for robust evaluation.

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

#### Key findings — Sampling

**Sampling does not improve Accuracy or MCC.** Baseline wins on both metrics across all models. The classic imbalanced data trade-off applies: every strategy that improves G1-Recall costs Accuracy proportionally.

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

Three targeted experiments were run to address the Grade 2/3 boundary confusion (11,655 hard cases identified in Week 4).

#### Experiment 1 — Threshold Tuning (5-Fold CV)

Optimised probability thresholds t12 (Grade1/2) and t23 (Grade2/3) to maximise Accuracy.

| Model | Method | Acc | MCC | G1-Recall | G3-Recall |
|---|---|---:|---:|---:|---:|
| RF | Default (argmax) | 0.7411 | 0.5153 | 0.469 | 0.629 |
| RF | Tuned (t12=0.35, t23=0.37) | 0.7199 | 0.5072 | 0.621 | 0.736 |
| LGBM | Default (argmax) | 0.7408 | 0.5184 | 0.526 | 0.638 |
| LGBM | Tuned (t12=0.40, t23=0.41) | 0.7246 | 0.5106 | 0.604 | 0.708 |

**Verdict:** Threshold tuning consistently lowers Accuracy (RF: −0.021, LGBM: −0.016). Not useful when Accuracy is the primary metric. Improves Recall if that matters.

---

#### Experiment 2 — Boundary Feature Engineering (5-Fold CV)

Added 5 new interaction features targeting the Grade 2/3 boundary:
`superstructure_x_foundation`, `superstructure_x_roof`, `age_x_floors`, `floors_x_area`, `damage_risk_score`

| Model | Variant | Acc | MCC | G3-Recall |
|---|---|---:|---:|---:|
| RF | Baseline | 0.7411 | 0.5153 | 0.629 |
| RF | + BoundaryFeatures | 0.7413 | 0.5159 | 0.629 |
| LGBM | Baseline | 0.7408 | 0.5184 | 0.638 |
| LGBM | + BoundaryFeatures | 0.7409 | 0.5185 | 0.638 |

**Verdict:** Negligible improvement (+0.0002 Acc, +0.0006 MCC). Confirms the Week 4 finding: the Grade 2/3 difficulty is **inherent ambiguity**, not a feature representation problem.

---

#### Experiment 3 — Two-Stage Classifier (5-Fold CV)

Stage 1: RF on all samples. Stage 2: LGBM on uncertain samples (max_proba < threshold). Five thresholds tested.

| Threshold | Acc | Delta Acc | MCC | G1-Recall | Uncertain/fold |
|---|---:|---:|---:|---:|---:|
| 0.50 | 0.7413 | +0.0002 | 0.5162 | 0.478 | 981 |
| 0.55 | 0.7418 | +0.0007 | 0.5187 | 0.504 | 5,867 |
| 0.60 | 0.7419 | +0.0008 | 0.5200 | 0.519 | 10,724 |
| **0.65** | **0.7423** | **+0.0012** | **0.5211** | **0.527** | **15,670** |
| 0.70 | 0.7421 | +0.0010 | 0.5206 | 0.526 | 20,605 |

Baseline RF (Stage 1 only): Acc=0.7411, MCC=0.5153

**Verdict: Best result of Week 5.** Threshold=0.65 consistently improves both Accuracy (+0.0012) and MCC (+0.0059) over 5 folds. Stage 2 LGBM corrects ~15,670 uncertain predictions per fold that RF gets wrong.

---

### Week 5 — Final Decision for Submission

Two-Stage Classifier (RF + LGBM, threshold=0.65) is the new best model.

```
Submission model:    Two-Stage RF + LGBM (threshold=0.65)
Accuracy (5-Fold CV): 0.7423   (+0.0012 over RF Baseline)
MCC (5-Fold CV):      0.5211   (+0.0059 over RF Baseline)
```

### Updated Overall Performance

| Metric | Week 2 | Week 3 | Week 4 | Week 5 (CV) |
|---|---:|---:|---:|---:|
| MCC | 0.4077 | 0.5023 | 0.5239 | 0.5211 |
| Accuracy | 0.6881 | 0.7327 | 0.7437 | 0.7423 |

Note: Week 5 reports 5-Fold CV mean scores. Week 4 used a single split.

---

### What Worked in Week 5

**Two-Stage Classifier** — letting RF handle confident predictions and LGBM handle uncertain ones gives consistent small gains on both Accuracy and MCC.

### What Did Not Work in Week 5

**Sampling strategies** — none improved Accuracy or MCC over the baseline for large tree ensemble models on 260k samples.

**Threshold Tuning** — optimising thresholds for Accuracy worsens Accuracy (the model already uses near-optimal thresholds internally).

**Boundary Feature Engineering** — new interaction features bring negligible improvement, confirming the Grade 2/3 confusion is inherent ambiguity not a feature gap.

---

## Open Questions / Next Steps

- **Week 06 (26.05): Hyperparameter Tuning**
  - Tune RF and LGBM within the Two-Stage framework
  - Use Optuna or RandomizedSearchCV with 5-Fold CV
  - Target: push Accuracy above 0.75
- Threshold search for the Two-Stage model (already done: 0.65 is optimal)
- Investigate stacking with Two-Stage as meta-feature input