# Week 02: Cleaning & Pipeline

---

## Project Structure

```
week2-cleaning-pipeline/
├── pipeline.py
├── README.md
├── requirements.txt
├── data/
│   ├── train_values.csv
│   ├── train_labels.csv
│   └── test_values.csv
└── Team2_prediction.csv (generated)
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Create data folder
mkdir -p data
```

**Note:** Place CSV files in `data/` folder before running.

## Run Pipeline

```bash
python pipeline.py
```

**Output:** `Team2_prediction.csv`

---

## Pipeline Steps

1. **Load data** from `data/` folder
2. **Clean data:**
   - Handle missing values (age=0 → median)
   - Remove uninformative features
   - Encode categorical features
   - Remove highly correlated features (>0.7)
3. **Train** Decision Tree classifier
4. **Predict** on test set
5. **Save** to `Team2_prediction.csv`

---

## Results

**Validation Set:**
- MCC: 0.4067
- Accuracy: 0.6883

**Features:** 38 → 36 (after cleaning)

---

## Output Format

```csv
building_id,damage_grade
300051,3
99355,2
...
```

---

## Next Steps

- Feature engineering
- Try other models (Random Forest, XGBoost)
- Hyperparameter tuning
- Cross-validation