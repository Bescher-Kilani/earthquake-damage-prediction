"""
Earthquake Damage Prediction Pipeline
Week 02: Cleaning & Pipeline

Simple pipeline that:
1. Loads data
2. Cleans data (removes uninformative features, encodes categorical variables)
3. Trains Decision Tree
4. Makes predictions
5. Saves submission
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import matthews_corrcoef, accuracy_score


def load_data():
    """Load training and test data."""
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    train_values = pd.read_csv('data/train_values.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    test_values = pd.read_csv('data/test_values.csv')
    
    # Extract IDs
    test_ids = test_values['building_id'].values
    
    # Set index
    X_train = train_values.set_index('building_id')
    y_train = train_labels.set_index('building_id')['damage_grade']
    X_test = test_values.set_index('building_id')
    
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print("="*60 + "\n")
    
    return X_train, y_train, X_test, test_ids


def clean_data(X_train, X_test):
    """
    Clean the data:
    - Remove uninformative features (only one unique value)
    - Encode categorical features
    
    Decision is made on training data only!
    """
    print("="*60)
    print("CLEANING DATA")
    print("="*60)
    
    # 1. Remove uninformative features
    uninformative = [col for col in X_train.columns if X_train[col].nunique() == 1]
    
    if uninformative:
        print(f"Removing {len(uninformative)} uninformative features")
        X_train = X_train.drop(columns=uninformative)
        X_test = X_test.drop(columns=uninformative)
    else:
        print("No uninformative features found")
    
    # 2. Encode categorical features
    categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
    
    if categorical_cols:
        print(f"Encoding {len(categorical_cols)} categorical features")
        encoders = {}
        
        for col in categorical_cols:
            encoder = LabelEncoder()
            # Fit on train
            encoder.fit(X_train[col].astype(str))
            encoders[col] = encoder
            
            # Transform both train and test
            X_train[col] = encoder.transform(X_train[col].astype(str))
            # Handle unseen categories in test
            test_col_str = X_test[col].astype(str)
            class_to_int = {cls: i for i, cls in enumerate(encoder.classes_)}
            X_test[col] = test_col_str.map(lambda x: class_to_int.get(x, -1))
    
    print("="*60 + "\n")
    
    return X_train, X_test


def train_model(X_train, y_train):
    """Train Decision Tree classifier."""
    print("="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50
    )
    
    model.fit(X_train, y_train)
    
    print("Training complete!")
    print("="*60 + "\n")
    
    return model


def evaluate_model(X_train, y_train):
    """
    Evaluate model performance using train/validation split.
    Returns MCC and Accuracy scores.
    """
    print("="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Split data for evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model on training subset
    eval_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50
    )
    eval_model.fit(X_tr, y_tr)
    
    # Make predictions on validation set
    y_pred = eval_model.predict(X_val)
    
    # Calculate metrics
    mcc = matthews_corrcoef(y_val, y_pred)
    acc = accuracy_score(y_val, y_pred)
    
    print(f"Validation Set Performance:")
    print(f"  MCC (Matthews Correlation Coefficient): {mcc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print("="*60 + "\n")
    
    return mcc, acc


def make_predictions(model, X_test):
    """Make predictions on test set."""
    print("="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    predictions = model.predict(X_test)
    
    print(f"Predictions: {len(predictions):,}")
    print(f"Distribution:")
    for grade in [1, 2, 3]:
        count = (predictions == grade).sum()
        print(f"  Grade {grade}: {count:,}")
    
    print("="*60 + "\n")
    
    return predictions


def save_submission(predictions, building_ids):
    """Save predictions to CSV file."""
    submission = pd.DataFrame({
        'building_id': building_ids,
        'damage_grade': predictions
    })
    
    submission.to_csv('Team2_prediction.csv', index=False)
    
    print("="*60)
    print("SUBMISSION SAVED")
    print("="*60)
    print("File: Team2_prediction.csv")
    print(f"Rows: {len(submission):,}")
    print("="*60 + "\n")


def main():
    """Main pipeline."""
    print("\n" + "="*60)
    print("EARTHQUAKE DAMAGE PREDICTION PIPELINE")
    print("Week 02: Cleaning & Pipeline")
    print("="*60 + "\n")
    
    # Set random seed
    np.random.seed(42)
    
    # 1. Load data
    X_train, y_train, X_test, test_ids = load_data()
    
    # 2. Clean data
    X_train, X_test = clean_data(X_train, X_test)
    
    # 3. Evaluate model performance
    mcc, acc = evaluate_model(X_train, y_train)
    
    # 4. Train model on full training set
    model = train_model(X_train, y_train)
    
    # 5. Make predictions
    predictions = make_predictions(model, X_test)
    
    # 6. Save submission
    save_submission(predictions, test_ids)
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Estimated Performance (on validation set):")
    print(f"  MCC: {mcc:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"\nSubmission file: Team2_prediction.csv")
    print("="*60 + "\n")
    
    print("PIPELINE COMPLETE!\n")


if __name__ == '__main__':
    main()