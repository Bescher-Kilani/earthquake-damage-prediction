# Week 02: Cleaning & Pipeline

Earthquake Damage Prediction - Simple baseline pipeline.

## Setup

```bash
pip install -r requirements.txt
```

## Data

Place these files in `data/`:
- `train_values.csv`
- `train_labels.csv`
- `test_values.csv`

## Run

```bash
python pipeline.py
```

This will create `Team2_prediction.csv` with the predictions.

## Pipeline Steps

1. Load data from `data/`
2. Clean data:
   - Remove uninformative features (only one unique value)
   - Encode categorical features (strings → numbers)
3. Train Decision Tree classifier
4. Make predictions on test set
5. Save to `Team2_prediction.csv`

## Output Format

```csv
building_id,damage_grade
300051,3
99355,2
...
```
