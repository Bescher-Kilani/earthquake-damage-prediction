
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
│   └── Team2_prediction.csv
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

**Note:** Place CSV files in `data/` folder before running.

---

## Run Pipeline

```bash
python src/pipeline.py
```

**Output:** `outputs/Team2_prediction.csv`

---

## Pipeline Steps

1. **Load data** from `data/` folder
2. **Clean & engineer features:**
   * Handle missing values (age=0 → NaN → median)
3. **Feature Engineering:**
   * **Create missing indicators** (age_was_missing)
   * **Recombine binary features** (11 binary → 1 categorical)
   * Remove uninformative features
   * **Target Encoding** for high-cardinality (geo_levels, superstructure)
   * **Label Encoding** for low-cardinality (foundation_type, roof_type, etc.)
   * Remove highly correlated features (threshold=0.9)
   * **Height Per Floor Feature:** = height_percentage / count_floors_pre_eq
   * **Slenderness Feature** = height_percentage / area_percentage
   * **Age Group:** age is bucketed into 
      * New: [0, 10]
      * Medium: [10, 30]
      * Old: [30, 60]
      * Very old:  [60, 1000].
   * **Ground Height Feature:** = Ground Type + ‘_’ + height_percentage
   * **Rare Category Grouping:** Replacing categories with count below RARE_CATEGORY_THRESHOLD with OTHER.
3. **Train** Decision Tree classifier (max_depth=10)
4. **Analyze** feature importance
5. **Predict** on test set
6. **Save** to `outputs/Team2_prediction.csv`

---

## Results

### **Final Performance (Week 3)**

| Metric   | Week 2 | Week 3 |
| -------- | ------ | ------ |
| MCC      | 0.4077 | 0.5023 |
| Accuracy | 0.6881 | 0.7327 |

* **Features:** 24

### **Improvement vs. Week 2 Start**

* **MCC:** +23.2% (0.4077 → 0.5023)
* **Accuracy:** +6.5% (0.6881 → 0.7327)

---

## What Worked

### **1. Missing Value Indicators** 

```python
age_was_missing = age.isnull()
```

* Preserves information: "missingness" is predictive
* Became 3rd most important feature (1.49%)

### **2. Binary Feature Recombination** 

```python
# 11 binary → 1 categorical with 505 combinations
has_superstructure_* → superstructure_type
```

* Reduced dimensionality
* Enabled effective Target Encoding

### **3. Target Encoding**

```python
# For geo_levels + recombined features
geo_level_1_id → geo_level_1_id_target_enc (mean damage)
superstructure_type → superstructure_type_target_enc
```

* Replaced arbitrary IDs with meaningful risk values
* geo_level_3: 85.16% feature importance!

### **5. Additional Engineered Features**

| Feature                      | MCC Improvement |
| ---------------------------- | --------------- |
| `USE_HEIGHT_PER_FLOOR`       | +0.10%          |
| `USE_SLENDERNESS`            | +0.08%          |
| `USE_AGE_GROUP`              | +0.00%          |
| `USE_GROUND_HEIGHT_BINNED`   | +0.10%          |
| `USE_RARE_CATEGORY_GROUPING` | +0.28%          |

---

## What Did NOT Work

### **Experiment 1: Feature Combinations (3x)** → -0.18% ❌

* Too many features simultaneously → Overfitting

### **Experiment 2: Merge Infrequent Categories** → -0.01% ❌

* Target Encoding already handles rare categories

### **Experiment 3: Binning** → -0.13% ❌

```python
age_binned: 0.0026    vs    age: 0.0056 (original better!)
```

* Decision Trees don't need binning
* **Action:** Binning removed from pipeline

### **Experiment 4: One-Hot Encoding** → -22.5%! ❌❌

* Bug: geo_levels (int64) not detected as categorical
* Catastrophic performance loss

---

## Feature Importance (Top 15)

```text
1.  geo_level_3_id_target_enc.................... 0.851577  <-- Dominant
2.  superstructure_type_target_enc............... 0.064094
3.  age_was_missing.............................. 0.014762
4.  geo_level_2_id_target_enc.................... 0.013551
5.  geo_level_1_id_target_enc.................... 0.013113
6.  foundation_type.............................. 0.010691
7.  age.......................................... 0.005350
8.  area_percentage.............................. 0.005328
9.  roof_type.................................... 0.004923
10. ground_height_binned_target_enc.............. 0.003683
11. count_families............................... 0.003564
12. other_floor_type............................. 0.002779
13. slenderness.................................. 0.001886
14. count_floors_pre_eq.......................... 0.000942
15. height_per_floor............................. 0.000858
```

---

## Key Learnings

**Target Encoding:**

* ✅ Effective for high-cardinality features (>10 categories)
* ✅ Smoothing (smoothing=10) prevents overfitting

**Binning:**

* ❌ Harmful for Decision Trees (they create optimal splits)
* ✅ Keep continuous features as-is

**Feature Combinations:**

* ❌ Don't add multiple features simultaneously
* ✅ Test one at a time

**Best Practices:**

* ✅ Check Feature Importance after every change
* ✅ Verify new features are actually used by model
* ✅ Prevent data leakage (fit on train only)

---

## Output Format

```csv
building_id,damage_grade
300051,3
99355,2
...
```

---