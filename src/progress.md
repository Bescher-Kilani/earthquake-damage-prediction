# Earthquake Damage Prediction — Team 2

## Usage

```bash
# Generate submission file (no validation, full training data → submission CSV)
python src/pipeline_sklearn.py

# Model comparison with cross-validation (all models)
python src/pipeline_sklearn.py --compare

# Model comparison without cross-validation (single train/val split, faster) (all models)
python src/pipeline_sklearn.py --compare --no-cv

# Compare specific models only — with CV
python src/pipeline_sklearn.py --compare --models "Decision Tree,Random Forest,LightGBM"

# Compare specific models only — without CV (faster)
python src/pipeline_sklearn.py --compare --no-cv --models "Decision Tree,Random Forest,LightGBM"

# Use only N samples (quick test)
python src/pipeline_sklearn.py --compare --sample 20000

# Deep error analysis — fixed to the 4 ensemble models (Random Forest, LightGBM, XGBoost, CatBoost)
# These are our focus models going forward
python src/pipeline_sklearn.py --analyze

# Deep error analysis with subsampling (faster, for quick testing)
python src/pipeline_sklearn.py --analyze --sample 30000
```

Available models: `Decision Tree`, `Random Forest`, `LightGBM`, `XGBoost`, `CatBoost`, `k-Nearest Neighbors`, `Logistic Regression`, `Naive Bayes`, `SVM (Linear)`

## Results

### Week 03 — Baseline
Single train/val split (80/20), Decision Tree.

| Metric   | Score  |
|----------|--------|
| Accuracy | 0.7327 |
| MCC      | 0.5023 |

### Week 04 — Model Comparison & Error Analysis

**Step 1: All 9 models, single split (no CV)**

| Rank | Model               | Accuracy | MCC    | Time  |
|------|---------------------|----------|--------|-------|
| 1    | Random Forest       | 0.7426   | 0.5185 | 15.4s |
| 2    | LightGBM            | 0.7418   | 0.5209 | 10.8s |
| 3    | XGBoost             | 0.7394   | 0.5150 | 8.1s  |
| 4    | CatBoost            | 0.7366   | 0.5087 | 11.4s |
| 5    | Decision Tree       | 0.7327   | 0.5023 | 0.9s  |
| 6    | Logistic Regression | 0.7295   | 0.4918 | 2.9s  |
| 7    | SVM (Linear)        | 0.7154   | 0.4569 | 3.2s  |
| 8    | k-Nearest Neighbors | 0.7142   | 0.4675 | 0.0s  |
| 9    | Naive Bayes         | 0.5799   | 0.3587 | 0.1s  |

**Step 2: Baseline CV — original 5 models, 5-Fold CV (260k)**

| Rank | Model               | Acc (mean) | Acc (std) | MCC (mean) | MCC (std) | Time |
|------|---------------------|------------|-----------|------------|-----------|------|
| 1    | Decision Tree       | 0.7293     | 0.0032    | 0.4951     | 0.0042    | 116s |
| 2    | Logistic Regression | 0.7272     | 0.0016    | 0.4870     | 0.0027    | 132s |
| 3    | SVM (Linear)        | 0.7147     | 0.0022    | 0.4557     | 0.0030    | 131s |
| 4    | k-Nearest Neighbors | 0.7146     | 0.0018    | 0.4676     | 0.0026    | 187s |
| 5    | Naive Bayes         | 0.5790     | 0.0025    | 0.3543     | 0.0029    | 112s |

**Step 3: Top 5 ensemble models, 5-Fold CV (260k)**

| Rank | Model         | Acc (mean) | Acc (std) | MCC (mean) | MCC (std) | Time |
|------|---------------|------------|-----------|------------|-----------|------|
| 1    | Random Forest | 0.7411     | 0.0017    | 0.5153     | 0.0025    | 192s |
| 2    | LightGBM      | 0.7408     | 0.0019    | 0.5184     | 0.0027    | 152s |
| 3    | XGBoost       | 0.7383     | 0.0021    | 0.5128     | 0.0031    | 159s |
| 4    | CatBoost      | 0.7350     | 0.0014    | 0.5054     | 0.0016    | 176s |
| 5    | Decision Tree | 0.7293     | 0.0032    | 0.4951     | 0.0042    | 111s |

Decision: **Random Forest** best accuracy (0.7411), **LightGBM** best MCC (0.5184).
Focus: Random Forest, LightGBM, XGBoost, CatBoost for deeper analysis.

**Step 4: Error Analysis — Confusion Matrix**

80/20 single split. All 4 ensemble models show the same pattern.

| Model         | Accuracy | MCC    | Biggest confusion          |
|---------------|----------|--------|----------------------------|
| Random Forest | 0.7426   | 0.5185 | Grade 3 → predicted Grade 2 (6,381) |
| LightGBM      | 0.7418   | 0.5209 | Grade 3 → predicted Grade 2 (6,184) |
| XGBoost       | 0.7394   | 0.5150 | Grade 3 → predicted Grade 2 (6,253) |
| CatBoost      | 0.7366   | 0.5087 | Grade 3 → predicted Grade 2 (6,369) |

Per-class breakdown (Random Forest, representative for all):

| Grade   | Precision | Recall | F1     | Support |
|---------|-----------|--------|--------|---------|
| Grade 1 | 0.7048    | 0.4776 | 0.5694 | 5,025   |
| Grade 2 | 0.7388    | 0.8536 | 0.7921 | 29,652  |
| Grade 3 | 0.7604    | 0.6301 | 0.6891 | 17,444  |

Key findings:
- **Grade 1** has the worst recall (~0.48–0.54): more than half of all lightly damaged buildings are missed. Root cause: class imbalance (only 9.6% of samples).
- **Grade 2 ↔ Grade 3 boundary** is the main error source: ~6,200–6,400 Grade 3 buildings are predicted as Grade 2 by every model.
- Grade 2 is predicted best overall (recall ~0.84) because it dominates the training set (56.9%).

**Step 5: Disagreement Analysis — Hard Cases**

Samples wrong for **all 4 models simultaneously** (80/20 split, 52,121 val samples):

| Group                        | Count  | Share  |
|------------------------------|--------|--------|
| Wrong by ALL 4 models        | 11,655 | 22.4%  |
| Correct by ALL 4 models      | 36,580 | 70.2%  |

Grade distribution in hard cases vs. overall val set:

| Grade | Hard cases | All val |
|-------|-----------|---------|
| 1     | 18.3%     | 9.6%    |
| 2     | 32.0%     | 56.9%   |
| 3     | 49.7%     | 33.5%   |

Grade 1 is twice as prevalent in hard cases as in the full dataset. All 4 models predict ~67% of hard cases as Grade 2 regardless of true label — they fall back to the majority class under uncertainty.

Feature profile of hard cases vs. correctly predicted samples: numeric features (age, height, area, floors) show negligible differences (Δ < 0.31). The difficulty is not explainable by raw feature values — it reflects inherent ambiguity at the Grade 2/3 boundary.

**Step 6: Feature Importance — all 4 ensemble models**

10 features appear in the Top 15 for **all 4 models**:

| Feature                        | Notes                        |
|--------------------------------|------------------------------|
| `geo_level_3_id_target_enc`    | #1 by far in every model     |
| `geo_level_2_id_target_enc`    | #2 in RF, LGBM, CatBoost     |
| `geo_level_1_id_target_enc`    | consistent top-5             |
| `superstructure_type_target_enc` | strong across all models   |
| `ground_height_binned_target_enc` |                           |
| `foundation_type`              |                              |
| `roof_type`                    |                              |
| `age`                          |                              |
| `age_was_missing`              | missing-value indicator useful |
| `count_families`               |                              |

Geography (geo_level target encodings) dominates — the 3 geo features together account for ~69% of importance in Random Forest. Model-specific findings: `slenderness` appears in LGBM Top 5 but not in XGBoost/CatBoost Top 15; `other_floor_type` and `has_secondary_use` are XGBoost-specific.

## Open Questions / Next Steps

- **Week 05 (19.05): Imbalanced Multi-Class Data**
  - Try `class_weight='balanced'` on RF/LGBM/XGB/CatBoost → improve Grade 1 recall
  - Evaluate SMOTE oversampling for Grade 1
  - Monitor MCC trade-off: better recall on Grade 1 may lower overall accuracy
- Threshold tuning for Grade 2/3 boundary
- Investigate stacking: use ensemble outputs as meta-features for a second-level classifier