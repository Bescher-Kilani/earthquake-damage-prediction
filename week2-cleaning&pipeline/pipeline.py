"""
Earthquake Damage Prediction Pipeline
Week 03: Feature Engineering

Pipeline Steps:
1. Load data from CSV files
2. Clean and prepare data:
   - Handle missing values (identify artificial defaults and impute)
   - Recombine binary features into categorical features
   - Remove uninformative features (constant values)
   - Encode categorical variables (Target Encoding for high-cardinality features)
   - Remove highly correlated features to reduce multicollinearity
3. Train Decision Tree classifier
4. Generate predictions on test set
5. Save submission file in competition format
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
    """
    Load training and test datasets from CSV files.
    
    Returns:
        X_train: Training features (DataFrame)
        y_train: Training labels (Series)
        X_test: Test features (DataFrame)
        test_ids: Test building IDs for submission (numpy array)
    """
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)
    
    # Read CSV files
    train_values = pd.read_csv('data/train_values.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    test_values = pd.read_csv('data/test_values.csv')
    
    # Extract building IDs before setting as index
    test_ids = test_values['building_id'].values
    
    # Set building_id as index for easier data manipulation
    X_train = train_values.set_index('building_id')
    y_train = train_labels.set_index('building_id')['damage_grade']
    X_test = test_values.set_index('building_id')
    
    # Display dataset information
    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")
    print(f"Features: {X_train.shape[1]}")
    print("="*60 + "\n")
    
    return X_train, y_train, X_test, test_ids


def recombine_binary_features(X_train, X_test):
    """
    Recombine binary one-hot encoded features into categorical features.
    
    Problem: Binary features like has_superstructure_adobe_mud, has_superstructure_timber
    are essentially one-hot encodings of a single categorical variable.
    
    Solution: Create a single categorical feature that captures material combinations.
    
    Example:
        Before: has_superstructure_adobe_mud=1, has_superstructure_timber=1
        After:  superstructure_type="adobe_mud_timber"
    
    Benefits:
        - Reduces dimensionality (11 binary → 1 categorical)
        - Captures material combinations explicitly
        - Enables better encoding methods (Target Encoding)
    
    Args:
        X_train: Training features
        X_test: Test features
        
    Returns:
        X_train, X_test: DataFrames with recombined features
    """
    print("\n--- Recombining Binary Features ---")
    
    # 1. Process Superstructure Features
    # Find all columns starting with 'has_superstructure_'
    superstructure_cols = [col for col in X_train.columns if col.startswith('has_superstructure_')]
    
    if superstructure_cols:
        print(f"Recombining {len(superstructure_cols)} superstructure features")
        
        # For each row: collect all active superstructure types
        # axis=1 means apply function row-wise
        X_train['superstructure_type'] = X_train[superstructure_cols].apply(
            lambda row: '_'.join([
                col.replace('has_superstructure_', '')  # Remove prefix
                for col in superstructure_cols 
                if row[col] == 1  # Only include columns with value=1
            ]) or 'none',  # Use 'none' if no materials are active
            axis=1
        )
        
        # Apply same transformation to test set
        X_test['superstructure_type'] = X_test[superstructure_cols].apply(
            lambda row: '_'.join([
                col.replace('has_superstructure_', '') 
                for col in superstructure_cols 
                if row[col] == 1
            ]) or 'none',
            axis=1
        )
        
        # Remove original binary columns (no longer needed)
        X_train = X_train.drop(columns=superstructure_cols)
        X_test = X_test.drop(columns=superstructure_cols)
        
        print(f"  Created 'superstructure_type' with {X_train['superstructure_type'].nunique()} unique values")
    
    # 2. Process Secondary Use Features
    # Same logic as above for secondary use features
    secondary_use_cols = [col for col in X_train.columns if col.startswith('has_secondary_use_')]
    
    if secondary_use_cols:
        print(f"Recombining {len(secondary_use_cols)} secondary_use features")
        
        X_train['secondary_use_type'] = X_train[secondary_use_cols].apply(
            lambda row: '_'.join([
                col.replace('has_secondary_use_', '') 
                for col in secondary_use_cols 
                if row[col] == 1
            ]) or 'none',
            axis=1
        )
        
        X_test['secondary_use_type'] = X_test[secondary_use_cols].apply(
            lambda row: '_'.join([
                col.replace('has_secondary_use_', '') 
                for col in secondary_use_cols 
                if row[col] == 1
            ]) or 'none',
            axis=1
        )
        
        # Remove original binary columns
        X_train = X_train.drop(columns=secondary_use_cols)
        X_test = X_test.drop(columns=secondary_use_cols)
        
        print(f"  Created 'secondary_use_type' with {X_train['secondary_use_type'].nunique()} unique values")
    
    print("✓ Binary feature recombination complete")
    
    return X_train, X_test


def target_encode_features(X_train, y_train, X_test, features, smoothing=10):
    """
    Apply Target Encoding to categorical features.
    
    Target Encoding replaces each category with the average target value for that category,
    making the encoded values directly meaningful for prediction.
    
    Why Target Encoding?
    - Label Encoding assigns arbitrary integers (0, 1, 2, ...) which imply false ordering
    - Target Encoding uses mean(damage_grade) which has direct predictive meaning
    
    Example:
        Category "adobe_mud":
            - Appears 1000 times in training with average damage_grade = 2.5
            - Target Encoding: "adobe_mud" → 2.5
            - Interpretation: Buildings with adobe_mud have average damage of 2.5
    
    Smoothing:
        Prevents overfitting for rare categories by blending category mean with global mean.
        Formula: smoothed_mean = (count × mean + smoothing × global_mean) / (count + smoothing)
        
        Effect:
            - Many samples: smoothed_mean ≈ category mean (trust the data)
            - Few samples: smoothed_mean ≈ global mean (be conservative)
    
    Args:
        X_train: Training features
        y_train: Training labels (needed to compute means)
        X_test: Test features
        features: List of feature names to target-encode
        smoothing: Regularization parameter (default 10)
        
    Returns:
        X_train, X_test: DataFrames with target-encoded features
    """
    print("\n--- Target Encoding ---")
    
    # Calculate global mean damage grade (fallback for unknown categories)
    global_mean = y_train.mean()
    print(f"Global mean damage_grade: {global_mean:.4f}")
    
    # Process each feature
    for col in features:
        if col in X_train.columns:
            # Create DataFrame with target and category columns for grouping
            agg_data = pd.DataFrame({
                'target': y_train,
                'category': X_train[col]
            }).groupby('category')['target'].agg(['mean', 'count'])
            
            # Apply smoothing to prevent overfitting on rare categories
            # Smoothing formula blends category mean with global mean based on sample count
            agg_data['smoothed_mean'] = (
                (agg_data['mean'] * agg_data['count'] + smoothing * global_mean) 
                / (agg_data['count'] + smoothing)
            )
            
            # Map categories to their smoothed means in training set
            X_train[col + '_target_enc'] = X_train[col].map(agg_data['smoothed_mean'])
            
            # Map categories in test set
            # fillna(global_mean) handles categories that don't appear in training
            X_test[col + '_target_enc'] = X_test[col].map(agg_data['smoothed_mean']).fillna(global_mean)
            
            # Remove original categorical column (now replaced with numeric encoding)
            X_train = X_train.drop(columns=[col])
            X_test = X_test.drop(columns=[col])
            
            print(f"  {col}: {len(agg_data)} categories → {col}_target_enc")
    
    print("✓ Target encoding complete")
    
    return X_train, X_test


def clean_data(X_train, y_train, X_test, correlation_threshold=0.9, handle_missing=True, 
               artificial_missing_rules=None, add_missing_indicators=False):
	"""
	Comprehensive data cleaning pipeline.
	
	IMPORTANT: All decisions are made based on TRAINING DATA ONLY to prevent data leakage!
	
	Steps:
		1. Handle missing values:
		   - Identify artificial missing values (e.g., age=0)
		   - Optionally create missing indicators (preserves information about missingness)
		   - Impute with median (numeric) or mode (categorical)
		
		2. Recombine binary features into categorical features
		
		3. Remove uninformative features (only one unique value)
		
		4. Encode categorical features:
		   - Target Encoding: For high-cardinality features (>10-15 categories)
		   - Label Encoding: For low-cardinality features (<10 categories)
		
		5. Remove highly correlated features (reduces multicollinearity)
	
	Args:
		X_train: Training features
		y_train: Training labels (required for Target Encoding)
		X_test: Test features
		correlation_threshold: Threshold for removing correlated features (default 0.9)
		handle_missing: Whether to handle missing values (default True)
		artificial_missing_rules: Dict mapping features to values that represent missing
		                          Example: {'age': [0]} treats age=0 as missing
		add_missing_indicators: Whether to create binary features indicating missingness
		
	Returns:
		X_train, X_test: Cleaned DataFrames ready for model training
	"""
	print("="*60)
	print("CLEANING DATA")
	print("="*60)
	
	# Set default artificial missing rules if none provided
	if artificial_missing_rules is None:
		artificial_missing_rules = {'age': [0]}
	
	# ========== STEP 1: Missing Value Handling ==========
	if handle_missing:
		print("\n--- Step 1: Missing Value Handling ---")
		
		# 1a. Replace artificial missing values with NaN
		# Some features use special values (like 0) to represent missing data
		if artificial_missing_rules:
			print(f"Replacing artificial missing values:")
			for feature, values in artificial_missing_rules.items():
				if feature in X_train.columns:
					for val in values:
						train_count = (X_train[feature] == val).sum()
						test_count = (X_test[feature] == val).sum()
						
						if train_count > 0 or test_count > 0:
							print(f"  {feature} = {val} → NaN (Train: {train_count}, Test: {test_count})")
							# Replace artificial missing value with actual NaN
							X_train[feature] = X_train[feature].replace(val, np.nan)
							X_test[feature] = X_test[feature].replace(val, np.nan)
		
		# 1b. Add missing indicators if requested
		# Creates binary features indicating whether a value was originally missing
		# This preserves information: "missing" itself can be predictive!
		if add_missing_indicators:
			print("\nAdding missing indicators:")
			for col in X_train.columns:
				if X_train[col].isnull().sum() > 0:
					indicator_name = f"{col}_was_missing"
					# Create binary indicator: 1 if missing, 0 if present
					X_train[indicator_name] = X_train[col].isnull().astype(int)
					X_test[indicator_name] = X_test[col].isnull().astype(int)
					print(f"  Added: {indicator_name}")
		
		# 1c. Impute missing values
		# Separate numeric and categorical columns for different imputation strategies
		numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
		categorical_cols = X_train.select_dtypes(include=['object', 'string']).columns.tolist()
		
		# Impute numeric features with median (more robust than mean)
		if numeric_cols:
			missing_numeric = [col for col in numeric_cols if X_train[col].isnull().sum() > 0]
			if missing_numeric:
				print(f"\nImputing {len(missing_numeric)} numeric features with median (fitted on train):")
				for col in missing_numeric:
					train_missing = X_train[col].isnull().sum()
					test_missing = X_test[col].isnull().sum()
					
					# Calculate median ONLY on training data (prevent data leakage)
					median_value = X_train[col].median()
					
					# Fill missing values in both train and test with training median
					X_train[col] = X_train[col].fillna(median_value)
					X_test[col] = X_test[col].fillna(median_value)
					
					print(f"  {col}: median={median_value:.2f} (Train NaN: {train_missing}, Test NaN: {test_missing})")
		
		# Impute categorical features with mode (most frequent category)
		if categorical_cols:
			missing_categorical = [col for col in categorical_cols if X_train[col].isnull().sum() > 0]
			if missing_categorical:
				print(f"\nImputing {len(missing_categorical)} categorical features with mode (fitted on train):")
				for col in missing_categorical:
					train_missing = X_train[col].isnull().sum()
					test_missing = X_test[col].isnull().sum()
					
					# Calculate mode ONLY on training data
					mode_value = X_train[col].mode()[0] if len(X_train[col].mode()) > 0 else 'unknown'
					
					# Fill missing values with training mode
					X_train[col] = X_train[col].fillna(mode_value)
					X_test[col] = X_test[col].fillna(mode_value)
					
					print(f"  {col}: mode={mode_value} (Train NaN: {train_missing}, Test NaN: {test_missing})")
		
		print("\n✓ Missing value handling complete")
	
	# ========== STEP 1.5: Recombine Binary Features ==========
	# Convert binary one-hot features back to categorical
	X_train, X_test = recombine_binary_features(X_train, X_test)

	
	# ========== STEP 2: Remove Uninformative Features ==========
	print("\n--- Step 2: Uninformative Feature Removal ---")
	# Features with only one unique value provide no information for prediction
	uninformative = [col for col in X_train.columns if X_train[col].nunique() == 1]
	
	if uninformative:
		print(f"Removing {len(uninformative)} uninformative features")
		X_train = X_train.drop(columns=uninformative)
		X_test = X_test.drop(columns=uninformative)
	else:
		print("No uninformative features found")
	
	# ========== STEP 3: Encode Categorical Features ==========
	print("\n--- Step 3: Categorical Encoding ---")
	# Identify all categorical columns (including binned features with category dtype!)
	categorical_cols = X_train.select_dtypes(include=['object', 'string', 'category']).columns.tolist()

	# Identify features for Target Encoding (high-cardinality features)
	# Target Encoding is better for features with many categories because:
	# 1. It creates comparable scales across features (all in "damage risk" units)
	# 2. It avoids arbitrary orderings (Label 0 < Label 504 means nothing)
	# 3. It provides direct predictive information (higher value = higher risk)
	
	geo_features = [col for col in X_train.columns if col.startswith('geo_level_')]
	target_encode_features_list = geo_features.copy()

	# Add recombined features to Target Encoding list
	# These have many categories (505 and 18 respectively)
	if 'superstructure_type' in categorical_cols:
		target_encode_features_list.append('superstructure_type')
	if 'secondary_use_type' in categorical_cols:
		target_encode_features_list.append('secondary_use_type')

	# Remaining categorical features use Label Encoding
	# These typically have fewer categories (<10), making Label Encoding acceptable
	other_categorical = [col for col in categorical_cols 
	                     if col not in target_encode_features_list]

	print(f"Features for Target Encoding ({len(target_encode_features_list)}): {target_encode_features_list}")
	print(f"Features for Label Encoding ({len(other_categorical)}): {other_categorical}")
	
	# Apply Label Encoding to low-cardinality features
	if other_categorical:
		print(f"Label Encoding {len(other_categorical)} categorical features")
		encoders = {}
		
		for col in other_categorical:
			encoder = LabelEncoder()
			# Fit encoder on training data only
			encoder.fit(X_train[col].astype(str))
			encoders[col] = encoder
			
			# Transform training data
			X_train[col] = encoder.transform(X_train[col].astype(str))
			
			# Handle unseen categories in test set
			# Map known categories, assign -1 to unknown ones
			test_col_str = X_test[col].astype(str)
			class_to_int = {cls: i for i, cls in enumerate(encoder.classes_)}
			X_test[col] = test_col_str.map(lambda x: class_to_int.get(x, -1))
	
	# Apply Target Encoding to high-cardinality features
	if target_encode_features_list:
		X_train, X_test = target_encode_features(X_train, y_train, X_test, 
		                                          target_encode_features_list, smoothing=10)
	
	# ========== STEP 4: Remove Highly Correlated Features ==========
	print("\n--- Step 4: Highly Correlated Feature Removal ---")
	# Calculate Pearson correlation matrix on TRAINING data only
	# NOW all features are numeric (encoded), so correlation works!
	corr_matrix = X_train.corr().abs()
	
	# Get upper triangle of correlation matrix (avoid checking both A-B and B-A)
	upper_tri = corr_matrix.where(
		np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
	)
	
	# Find features with correlation > threshold
	# High correlation indicates redundancy - one feature can be removed
	to_drop = [
		column for column in upper_tri.columns 
		if any(upper_tri[column] > correlation_threshold)
	]
	
	if to_drop:
		print(f"Removing {len(to_drop)} highly correlated features (threshold={correlation_threshold}):")
		# Show which pairs were correlated for transparency
		for col in to_drop:
			correlations = upper_tri[col][upper_tri[col] > correlation_threshold]
			for corr_col, corr_val in correlations.items():
				print(f"  - {col} <-> {corr_col}: {corr_val:.3f}")
		
		# Remove correlated features
		X_train = X_train.drop(columns=to_drop)
		X_test = X_test.drop(columns=to_drop)
	else:
		print(f"No highly correlated features found (threshold={correlation_threshold})")
	
	print("="*60 + "\n")
	
	return X_train, X_test

def train_model(X_train, y_train):
    """
    Train a Decision Tree classifier on the cleaned data.
    
    Hyperparameters chosen to prevent overfitting:
        - max_depth=10: Limits tree depth to avoid memorizing training data
        - min_samples_split=100: Requires at least 100 samples to split a node
        - min_samples_leaf=50: Each leaf must contain at least 50 samples
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        model: Trained DecisionTreeClassifier
    """
    print("="*60)
    print("TRAINING MODEL")
    print("="*60)
    
    # Initialize Decision Tree with regularization to prevent overfitting
    model = DecisionTreeClassifier(
        random_state=42,        # For reproducibility
        max_depth=10,           # Limit tree depth
        min_samples_split=100,  # Min samples required to split internal node
        min_samples_leaf=50     # Min samples required at leaf node
    )
    
    # Fit model on training data
    model.fit(X_train, y_train)
    
    print("Training complete!")
    print("="*60 + "\n")
    
    return model


def evaluate_model(X_train, y_train, metrics=None):
    """
    Evaluate model performance using a train/validation split.
    
    This provides an estimate of how well the model will perform on unseen data.
    
    Metrics:
        - MCC (Matthews Correlation Coefficient): Balanced metric for multi-class classification
          Range: -1 to +1, where +1 is perfect prediction
        - Accuracy: Simple percentage of correct predictions
    
    Args:
        X_train: Training features
        y_train: Training labels
        metrics: List of metrics to compute (default: ['mcc', 'accuracy'])
        
    Returns:
        results: Dictionary with computed metric values
    """
    if metrics is None:
        metrics = ['mcc', 'accuracy']
    
    print("="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    # Split data: 80% for training, 20% for validation
    # stratify=y_train ensures each class is proportionally represented in both splits
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Train model on training subset (80%)
    eval_model = DecisionTreeClassifier(
        random_state=42,
        max_depth=10,
        min_samples_split=100,
        min_samples_leaf=50
    )
    eval_model.fit(X_tr, y_tr)
    
    # Make predictions on validation subset (20%)
    y_pred = eval_model.predict(X_val)
    
    # Calculate requested metrics
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
    """
    Generate predictions on the test set.
    
    Args:
        model: Trained model
        X_test: Test features
        
    Returns:
        predictions: Array of predicted damage grades (1, 2, or 3)
    """
    print("="*60)
    print("MAKING PREDICTIONS")
    print("="*60)
    
    # Generate predictions
    predictions = model.predict(X_test)
    
    # Display prediction statistics
    print(f"Predictions: {len(predictions):,}")
    print(f"Distribution:")
    for grade in [1, 2, 3]:
        count = (predictions == grade).sum()
        print(f"  Grade {grade}: {count:,}")
    
    print("="*60 + "\n")
    
    return predictions


def save_submission(predictions, building_ids):
    """
    Save predictions to CSV file in competition submission format.
    
    Format:
        building_id,damage_grade
        300051,3
        99355,2
        ...
    
    Args:
        predictions: Array of predicted damage grades
        building_ids: Array of building IDs corresponding to predictions
    """
    # Create submission DataFrame
    submission = pd.DataFrame({
        'building_id': building_ids,
        'damage_grade': predictions
    })
    
    # Save to CSV (no index column)
    submission.to_csv('Team2_prediction.csv', index=False)
    
    print("="*60)
    print("SUBMISSION SAVED")
    print("="*60)
    print("File: Team2_prediction.csv")
    print(f"Rows: {len(submission):,}")
    print("="*60 + "\n")

def analyze_feature_importance(model, feature_names, top_n=15):
    """
    Analyze and display feature importance from trained model.
    
    Helps identify:
    - Most predictive features
    - Zero-importance features (candidates for removal)
    - Which engineered features are useful
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importances
    importances = model.feature_importances_
    
    # Create DataFrame for easier sorting
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Display top N features
    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for idx, row in feature_importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:.<45} {row['importance']:.6f}")
    
    # Find zero-importance features
    zero_importance = feature_importance_df[feature_importance_df['importance'] == 0]
    if len(zero_importance) > 0:
        print(f"\n⚠ Found {len(zero_importance)} zero-importance features:")
        for feat in zero_importance['feature'].values[:10]:  # Show max 10
            print(f"  - {feat}")
        if len(zero_importance) > 10:
            print(f"  ... and {len(zero_importance) - 10} more")
    else:
        print("\n✓ No zero-importance features found")
    
    # Summary statistics
    print(f"\nSummary:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Non-zero importance: {(importances > 0).sum()}")
    print(f"  Top 5 cumulative importance: {feature_importance_df.head(5)['importance'].sum():.2%}")
    print(f"  Top 10 cumulative importance: {feature_importance_df.head(10)['importance'].sum():.2%}")
    
    print("="*60 + "\n")
    
    return feature_importance_df

def main():
    """
    Main pipeline orchestrator.
    
    Executes the complete ML pipeline:
    1. Load data
    2. Clean and prepare data
    3. Evaluate model performance
    4. Train final model
    5. Generate predictions
    6. Save submission file
    """
    print("\n" + "="*60)
    print("EARTHQUAKE DAMAGE PREDICTION PIPELINE")
    print("Week 03: Feature Engineering")
    print("="*60 + "\n")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # 1. Load data from CSV files
    X_train, y_train, X_test, test_ids = load_data()
    
    # 2. Clean and prepare data
    # NOTE: y_train is passed for Target Encoding!
    X_train, X_test = clean_data(
        X_train, y_train, X_test,
        correlation_threshold=0.9,           # Higher threshold to preserve hierarchical features
        handle_missing=True,
        artificial_missing_rules={'age': [0]},  # Treat age=0 as missing
        add_missing_indicators=True          # Create binary missing indicators
    )
    
    # 3. Evaluate model performance on validation set
    results = evaluate_model(X_train, y_train, metrics=['mcc', 'accuracy'])
    
    # 4. Train final model on full training set
    model = train_model(X_train, y_train)
    
    # NEW: Analyze feature importance
    feature_importance_df = analyze_feature_importance(model, X_train.columns.tolist(), top_n=15)
    # 5. Generate predictions on test set
    predictions = make_predictions(model, X_test)
    
    # 6. Save submission file
    save_submission(predictions, test_ids)
    
    # Display summary
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