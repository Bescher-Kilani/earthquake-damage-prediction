# Week 03: Feature Engineering

---

## Project Structure

```
week3-feature-engineering/
├── pipeline.py                  ← Main pipeline (final version)
├── README.md                    ← This file
├── requirements.txt
├── data/
│   ├── train_values.csv
│   ├── train_labels.csv
│   └── test_values.csv
└── Team2_prediction.csv         ← Generated submission
```

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create data folder
mkdir -p data
```

**Note:** Place CSV files in `data/` folder before running.

---

## Run Pipeline

```bash
python pipeline.py
```

**Output:** `Team2_prediction.csv`

---

## Pipeline Steps

1. **Load data** from `data/` folder
2. **Clean & engineer features:**
   - Handle missing values (age=0 → NaN → median)
   - **Create missing indicators** (age_was_missing)
   - **Recombine binary features** (11 binary → 1 categorical)
   - Remove uninformative features
   - **Target Encoding** for high-cardinality (geo_levels, superstructure)
   - **Label Encoding** for low-cardinality (foundation_type, roof_type, etc.)
   - Remove highly correlated features (threshold=0.9)
3. **Train** Decision Tree classifier (max_depth=10)
4. **Analyze** feature importance
5. **Predict** on test set
6. **Save** to `Team2_prediction.csv`

---

## Results

### **Final Performance (Week 3)**
- **MCC:** 0.5270
- **Accuracy:** 0.7453
- **Features:** 20

### **Improvement vs. Week 2 Start**
- **MCC:** +29.3% (0.4077 → 0.5270)
- **Accuracy:** +8.3% (0.6881 → 0.7453)

---

## What Worked

### **1. Missing Value Indicators** (+6.03% MCC)
```python
age_was_missing = age.isnull()
```
- Preserves information: "missingness" is predictive
- Became 3rd most important feature (1.49%)

### **2. Binary Feature Recombination** (+2.43% MCC)
```python
# 11 binary → 1 categorical with 505 combinations
has_superstructure_* → superstructure_type
```
- Reduced dimensionality
- Enabled effective Target Encoding

### **3. Target Encoding** (+2.80% MCC combined)
```python
# For geo_levels + recombined features
geo_level_1_id → geo_level_1_id_target_enc (mean damage)
superstructure_type → superstructure_type_target_enc
```
- Replaced arbitrary IDs with meaningful risk values
- geo_level_3: 85.21% feature importance!

### **4. Correlation Threshold = 0.9** (+0.67% MCC)
- Preserved hierarchical geo_level features

---

## What Did NOT Work

### **Experiment 1: Feature Combinations (3x)** → -0.18% ❌
- Too many features simultaneously → Overfitting

### **Experiment 2: Merge Infrequent Categories** → -0.01% ❌
- Target Encoding already handles rare categories

### **Experiment 3: Binning** → -0.13% ❌
```
age_binned: 0.0026    vs    age: 0.0056 (original better!)
```
- Decision Trees don't need binning
- **Action:** Binning removed from pipeline

### **Experiment 4: One-Hot Encoding** → -22.5%! ❌❌
- Bug: geo_levels (int64) not detected as categorical
- Catastrophic performance loss

---

## Feature Importance (Top 10)

```
1.  geo_level_3_id_target_enc      85.21%  ← Dominant!
2.  superstructure_type_target_enc  6.54%
3.  age_was_missing                 1.49%
4.  geo_level_2_id_target_enc       1.37%
5.  geo_level_1_id_target_enc       1.30%
6.  foundation_type                 1.14%
7.  roof_type                       0.59%
8.  area_percentage                 0.59%
9.  age                             0.56%
10. count_families                  0.36%

Zero-importance: plan_configuration
```

---

## Key Learnings

**Target Encoding:**
- ✅ Effective for high-cardinality features (>10 categories)
- ✅ Smoothing (smoothing=10) prevents overfitting

**Binning:**
- ❌ Harmful for Decision Trees (they create optimal splits)
- ✅ Keep continuous features as-is

**Feature Combinations:**
- ❌ Don't add multiple features simultaneously
- ✅ Test one at a time

**Best Practices:**
- ✅ Check Feature Importance after every change
- ✅ Verify new features are actually used by model
- ✅ Prevent data leakage (fit on train only)

---

## Output Format

```csv
building_id,damage_grade
300051,3
99355,2
...
```

---

