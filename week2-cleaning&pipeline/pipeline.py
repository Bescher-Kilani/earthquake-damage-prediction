"""
Earthquake Damage Prediction Pipeline
Week 02: Cleaning & Pipeline

Pipeline:
1. Loads data
2. Cleans data:
   - Handles missing values (artificial defaults & imputation)
   - Removes uninformative features
   - Encodes categorical variables
   - Removes highly correlated features
3. Trains Decision Tree classifier
4. Makes predictions
5. Saves submission file
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
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


def clean_data(X_train, X_test, correlation_threshold=0.7, handle_missing=True, 
               artificial_missing_rules=None, add_missing_indicators=False):
    """
    Clean the data:
    - Handle missing values (both explicit NaN and artificial defaults)
    - Remove uninformative features (only one unique value)
    - Encode categorical features
    - Remove highly correlated features
    
    Decision is made on training data only!
    
    Args:
        X_train: Training features
        X_test: Test features
        correlation_threshold: Threshold for removing correlated features (default 0.7)
        handle_missing: Whether to handle missing values (default True)
        artificial_missing_rules: Dict mapping feature names to values that should be treated as missing
                                  Example: {'age': [0], 'geo_level_1_id': [0]}
                                  Default: {'age': [0]}
        add_missing_indicators: Whether to add binary features indicating missingness (default False)
    """
    print("="*60)
    print("CLEANING DATA")
    print("="*60)
    
    # Default artificial missing rules
    if artificial_missing_rules is None:
        artificial_missing_rules = {'age': [0]}
    
    # 1. Handle Missing Values
    if handle_missing:
        print("\n--- Step 1: Missing Value Handling ---")
        
        # 1a. Replace artificial missing values with NaN
        if artificial_missing_rules:
            print(f"Replacing artificial missing values:")
            for feature, values in artificial_missing_rules.items():
                if feature in X_train.columns:
                    for val in values:
                        train_count = (X_train[feature] == val).sum()
                        test_count = (X_test[feature] == val).sum()
                        
                        if train_count > 0 or test_count > 0:
                            print(f"  {feature} = {val} → NaN (Train: {train_count}, Test: {test_count})")
                            X_train[feature] = X_train[feature].replace(val, np.nan)
                            X_test[feature] = X_test[feature].replace(val, np.nan)
        
        # 1b. Add missing indicators if requested
        if add_missing_indicators:
            print("\nAdding missing indicators:")
            for col in X_train.columns:
                if X_train[col].isnull().sum() > 0:
                    indicator_name = f"{col}_was_missing"
                    X_train[indicator_name] = X_train[col].isnull().astype(int)
                    X_test[indicator_name] = X_test[col].isnull().astype(int)
                    print(f"  Added: {indicator_name}")
        
        # 1c. Impute missing values
        # Separate numeric and categorical columns
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
        
        # Impute numeric features with median
        if numeric_cols:
            missing_numeric = [col for col in numeric_cols if X_train[col].isnull().sum() > 0]
            if missing_numeric:
                print(f"\nImputing {len(missing_numeric)} numeric features with median (fitted on train):")
                for col in missing_numeric:
                    train_missing = X_train[col].isnull().sum()
                    test_missing = X_test[col].isnull().sum()
                    
                    # Calculate median on non-missing training values
                    median_value = X_train[col].median()
                    
                    # Fill both train and test (use assignment instead of inplace)
                    X_train[col] = X_train[col].fillna(median_value)
                    X_test[col] = X_test[col].fillna(median_value)
                    
                    print(f"  {col}: median={median_value:.2f} (Train NaN: {train_missing}, Test NaN: {test_missing})")
        
        # Impute categorical features with mode
        if categorical_cols:
            missing_categorical = [col for col in categorical_cols if X_train[col].isnull().sum() > 0]
            if missing_categorical:
                print(f"\nImputing {len(missing_categorical)} categorical features with mode (fitted on train):")
                for col in missing_categorical:
                    train_missing = X_train[col].isnull().sum()
                    test_missing = X_test[col].isnull().sum()
                    
                    # Calculate mode on non-missing training values
                    mode_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'unknown'
                    
                    # Fill both train and test (use assignment instead of inplace)
                    X_train[col] = X_train[col].fillna(mode_value)
                    X_test[col] = X_test[col].fillna(mode_value)
                    
                    print(f"  {col}: mode={mode_value} (Train NaN: {train_missing}, Test NaN: {test_missing})")
        
        print("\n✓ Missing value handling complete")
    
    # 2. Remove uninformative features
    print("\n--- Step 2: Uninformative Feature Removal ---")
    uninformative = [col for col in X_train.columns if X_train[col].nunique() == 1]
    
    if uninformative:
        print(f"Removing {len(uninformative)} uninformative features")
        X_train = X_train.drop(columns=uninformative)
        X_test = X_test.drop(columns=uninformative)
    else:
        print("No uninformative features found")
    
    # 3. Encode categorical features
    print("\n--- Step 3: Categorical Encoding ---")
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
    
    # 4. Remove highly correlated features
    print("\n--- Step 4: Highly Correlated Feature Removal ---")
    # Calculate correlation matrix on TRAINING data only (Pearson correlation)
    corr_matrix = X_train.corr().abs()
    
    # Get upper triangle of correlation matrix (avoid duplicates)
    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    
    # Find features with correlation > threshold
    to_drop = [
        column for column in upper_tri.columns 
        if any(upper_tri[column] > correlation_threshold)
    ]
    
    if to_drop:
        print(f"Removing {len(to_drop)} highly correlated features (threshold={correlation_threshold}):")
        # Show which pairs were correlated
        for col in to_drop:
            correlations = upper_tri[col][upper_tri[col] > correlation_threshold]
            for corr_col, corr_val in correlations.items():
                print(f"  - {col} <-> {corr_col}: {corr_val:.3f}")
        
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    else:
        print(f"No highly correlated features found (threshold={correlation_threshold})")
    
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


def evaluate_model(X_train, y_train, metrics=None):
    """
    Evaluate model performance using train/validation split.
    
    Args:
        X_train: Training features
        y_train: Training labels
        metrics: List of metrics to compute. Options: 'mcc', 'accuracy'
                 Default: ['mcc', 'accuracy']
    
    Returns:
        Dictionary with computed metrics
    """
    if metrics is None:
        metrics = ['mcc', 'accuracy']
    
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
    results = {}
    
    print(f"Validation Set Performance:")
    
    if 'mcc' in metrics:
        mcc = matthews_corrcoef(y_val, y_pred)
        results['mcc'] = mcc
        print(f"  MCC (Matthews Correlation Coefficient): {mcc:.4f}")
    
    if 'accuracy' in metrics:
        acc = accuracy_score(y_val, y_pred)
        results['accuracy'] = acc
        print(f"  Accuracy: {acc:.4f}")
    
    print("="*60 + "\n")
    
    return results


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
    X_train, X_test = clean_data(
        X_train, X_test, 
        correlation_threshold=0.7,
        handle_missing=True,
        artificial_missing_rules={'age': [0]},
        add_missing_indicators=False
    )
    
    # 3. Evaluate model performance
    results = evaluate_model(X_train, y_train, metrics=['mcc', 'accuracy'])
    
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
    print(f"  MCC: {results['mcc']:.4f}")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"\nSubmission file: Team2_prediction.csv")
    print("="*60 + "\n")
    
    print("PIPELINE COMPLETE!\n")


if __name__ == '__main__':
    main()