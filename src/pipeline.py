"""
Earthquake Damage Prediction Pipeline
Week 03: Feature Engineering

Pipeline Steps:
1. Load data from CSV files
2. Split data into train/validation BEFORE any cleaning (prevents data leakage)
3. Clean and prepare data:
   - Handle missing values (identify artificial defaults and impute)
   - Remove uninformative features (constant values)
   - Encode low-cardinality categorical variables (Label Encoding)
   - Remove highly correlated features to reduce multicollinearity
4. Feature Engineering:
   - Recombine binary features into categorical features
   - Target Encoding for high-cardinality features
5. Train Decision Tree classifier
6. Generate predictions on test set
7. Save submission file in competition format
"""

from sklearn.model_selection import KFold
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


def split_data(X_train, y_train, test_size=0.2, random_state=42):
    """
    Split training data into train and validation subsets.

    IMPORTANT: This must be called BEFORE clean_data and feature_engineer
    to prevent data leakage. Cleaning and encoding decisions must be made
    only on X_tr (the training fold), then applied to X_val separately.

    Args:
        X_train: Full training features
        y_train: Full training labels
        test_size: Fraction of data to use for validation (default: 0.2)
        random_state: Random seed for reproducibility (default: 42)

    Returns:
        X_tr: Training fold features
        X_val: Validation fold features
        y_tr: Training fold labels
        y_val: Validation fold labels
    """
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)

    # stratify=y_train ensures each class is proportionally represented in both splits
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train
    )

    print(f"Training fold:   {len(X_tr):,} samples ({1 - test_size:.0%})")
    print(f"Validation fold: {len(X_val):,} samples ({test_size:.0%})")
    print("="*60 + "\n")

    return X_tr, X_val, y_tr, y_val


# ── Feature Engineering helpers ───────────────────────────────────────────────

def recombine_binary_features(X_train, X_test=None, X_val=None):
    """
    Recombine binary one-hot encoded features into categorical features.

    This is feature engineering, not cleaning. The mapping is learned from the
    training columns, then applied consistently to validation/test data when
    those datasets are provided.

    Args:
        X_train: Training features.
        X_test: Test features (optional).
        X_val: Validation features (optional).

    Returns:
        X_train, X_test, X_val: DataFrames with recombined features.
    """
    print("\n--- Recombining Binary Features ---")

    def combine_columns(X, cols, prefix):
        available_cols = [col for col in cols if col in X.columns]
        if not available_cols:
            return pd.Series("none", index=X.index)

        return X[available_cols].apply(
            lambda row: '_'.join([
                col.replace(prefix, '')
                for col in available_cols
                if row[col] == 1
            ]) or 'none',
            axis=1
        )

    def apply_recombine(feature_cols, prefix, new_col):
        nonlocal X_train, X_test, X_val

        if not feature_cols:
            return

        print(f"Recombining {len(feature_cols)} {new_col} source features")

        for dataset_name, X in (("Train", X_train), ("Test", X_test), ("Validation", X_val)):
            if X is None:
                continue

            X[new_col] = combine_columns(X, feature_cols, prefix)
            X.drop(columns=feature_cols, inplace=True, errors="ignore")

            if dataset_name == "Train":
                print(
                    f"  Created '{new_col}' with "
                    f"{X[new_col].nunique()} unique values"
                )

    superstructure_cols = [
        col for col in X_train.columns
        if col.startswith('has_superstructure_')
    ]
    apply_recombine(
        feature_cols=superstructure_cols,
        prefix='has_superstructure_',
        new_col='superstructure_type'
    )

    secondary_use_cols = [
        col for col in X_train.columns
        if col.startswith('has_secondary_use_')
    ]
    apply_recombine(
        feature_cols=secondary_use_cols,
        prefix='has_secondary_use_',
        new_col='secondary_use_type'
    )

    print("Binary feature recombination complete")

    return X_train, X_test, X_val


def target_encode_features(X_train, y_train, X_test=None, features=None,
                           smoothing=10, X_val=None):
    """
    Apply target encoding to selected features.

    Encodings are fitted only on X_train/y_train, then applied to any provided
    validation/test data. This keeps validation/test transformations separate
    from cleaning and avoids using validation/test targets.

    Args:
        X_train: Training features.
        y_train: Training labels used to compute category means.
        X_test: Test features (optional).
        features: Feature names to target-encode.
        smoothing: Regularization parameter.
        X_val: Validation features (optional).

    Returns:
        X_train, X_test, X_val: Encoded DataFrames.
    """
    print("\n--- Target Encoding ---")

    if features is None:
        features = []

    global_mean = y_train.mean()
    print(f"Global mean damage_grade: {global_mean:.4f}")

    for col in features:
        if col not in X_train.columns:
            continue

        agg_data = pd.DataFrame({
            'target': y_train,
            'category': X_train[col]
        }).groupby('category', observed=False)['target'].agg(['mean', 'count'])

        agg_data['smoothed_mean'] = (
            (agg_data['mean'] * agg_data['count'] + smoothing * global_mean)
            / (agg_data['count'] + smoothing)
        )

        encoded_col = col + '_target_enc'
        X_train[encoded_col] = (
            X_train[col]
            .map(agg_data['smoothed_mean'])
            .astype(float)
            .fillna(global_mean)
        )

        for X in (X_test, X_val):
            if X is None or col not in X.columns:
                continue
            X[encoded_col] = (
                X[col]
                .map(agg_data['smoothed_mean'])
                .astype(float)
                .fillna(global_mean)
            )

        X_train = X_train.drop(columns=[col])
        if X_test is not None:
            X_test = X_test.drop(columns=[col], errors="ignore")
        if X_val is not None:
            X_val = X_val.drop(columns=[col], errors="ignore")

        print(f"  {col}: {len(agg_data)} categories -> {encoded_col}")

    print("Target encoding complete")

    return X_train, X_test, X_val


def clean_data(X_train, y_train, X_test, correlation_threshold=0.9,
               handle_missing=True, artificial_missing_rules=None,
               add_missing_indicators=False):
    """
    Comprehensive data cleaning pipeline.

    IMPORTANT: All decisions are made based on TRAINING DATA ONLY to prevent
    data leakage. X_train here should already be the training fold (after
    split_data), not the full dataset.

    Steps:
        1. Handle missing values:
           - Identify artificial missing values (e.g., age=0)
           - Optionally create missing indicators (preserves missingness info)
           - Impute with median (numeric) or mode (categorical)
        2. Remove uninformative features (only one unique value)
        3. Encode low-cardinality categorical features with Label Encoding
        4. Remove highly correlated features (reduces multicollinearity)

    Note: High-cardinality encoding (Target Encoding) and binary feature
    recombination are handled separately in feature_engineer().

    Args:
        X_train: Training fold features (NOT the full dataset)
        y_train: Training fold labels
        X_test: Test / validation features to transform consistently
        correlation_threshold: Threshold for removing correlated features (default 0.9)
        handle_missing: Whether to handle missing values (default True)
        artificial_missing_rules: Dict mapping features to values that represent missing
                                  Example: {'age': [0]} treats age=0 as missing
        add_missing_indicators: Whether to create binary features indicating missingness

    Returns:
        X_train, X_test: Cleaned DataFrames ready for feature engineering
    """
    print("="*60)
    print("CLEANING DATA")
    print("="*60)

    if artificial_missing_rules is None:
        artificial_missing_rules = {'age': [0]}

    # ========== STEP 1: Missing Value Handling ==========
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
                            print(
                                f"  {feature} = {val} → NaN "
                                f"(Train: {train_count}, Test: {test_count})")
                            X_train[feature] = X_train[feature].replace(
                                val, np.nan)
                            X_test[feature] = X_test[feature].replace(
                                val, np.nan)

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
        numeric_cols = X_train.select_dtypes(
            include=[np.number]).columns.tolist()
        categorical_cols = X_train.select_dtypes(
            include=['object', 'string']).columns.tolist()

        if numeric_cols:
            missing_numeric = [
                col for col in numeric_cols if X_train[col].isnull().sum() > 0]
            if missing_numeric:
                print(
                    f"\nImputing {len(missing_numeric)} numeric features with "
                    f"median (fitted on train):")
                for col in missing_numeric:
                    train_missing = X_train[col].isnull().sum()
                    test_missing = X_test[col].isnull().sum()
                    median_value = X_train[col].median()
                    X_train[col] = X_train[col].fillna(median_value)
                    X_test[col] = X_test[col].fillna(median_value)
                    print(
                        f"  {col}: median={median_value:.2f} "
                        f"(Train NaN: {train_missing}, Test NaN: {test_missing})")

        if categorical_cols:
            missing_categorical = [
                col for col in categorical_cols if X_train[col].isnull().sum() > 0]
            if missing_categorical:
                print(
                    f"\nImputing {len(missing_categorical)} categorical features with "
                    f"mode (fitted on train):")
                for col in missing_categorical:
                    train_missing = X_train[col].isnull().sum()
                    test_missing = X_test[col].isnull().sum()
                    mode_value = (X_train[col].mode()[0]
                                  if len(X_train[col].mode()) > 0 else 'unknown')
                    X_train[col] = X_train[col].fillna(mode_value)
                    X_test[col] = X_test[col].fillna(mode_value)
                    print(
                        f"  {col}: mode={mode_value} "
                        f"(Train NaN: {train_missing}, Test NaN: {test_missing})")

        print("\n✓ Missing value handling complete")

    # ========== STEP 2: Remove Uninformative Features ==========
    print("\n--- Step 2: Uninformative Feature Removal ---")
    uninformative = [
        col for col in X_train.columns if X_train[col].nunique() == 1]

    if uninformative:
        print(f"Removing {len(uninformative)} uninformative features")
        X_train = X_train.drop(columns=uninformative)
        X_test = X_test.drop(columns=uninformative)
    else:
        print("No uninformative features found")

    # ========== STEP 3: Label Encode Low-Cardinality Categoricals ==========
    print("\n--- Step 3: Label Encoding (low-cardinality categoricals) ---")

    # Exclude features that will be target-encoded in feature_engineer()
    geo_features = [
        col for col in X_train.columns if col.startswith('geo_level_')]
    # superstructure/secondary_use don't exist yet
    high_cardinality = set(geo_features)

    categorical_cols = X_train.select_dtypes(
        include=['object', 'string', 'category']).columns.tolist()
    label_encode_cols = [col for col in categorical_cols
                         if col not in high_cardinality]

    if label_encode_cols:
        print(
            f"Label Encoding {len(label_encode_cols)} features: {label_encode_cols}")
        for col in label_encode_cols:
            encoder = LabelEncoder()
            encoder.fit(X_train[col].astype(str))
            X_train[col] = encoder.transform(X_train[col].astype(str))
            class_to_int = {cls: i for i, cls in enumerate(encoder.classes_)}
            X_test[col] = X_test[col].astype(str).map(
                lambda x, m=class_to_int: m.get(x, -1))
    else:
        print("No low-cardinality categorical features to label-encode")

    # ========== STEP 4: Remove Highly Correlated Features ==========
    print("\n--- Step 4: Highly Correlated Feature Removal ---")

    # Only numeric columns can be correlated at this point
    numeric_only = X_train.select_dtypes(include=[np.number])
    corr_matrix = numeric_only.corr().abs()

    upper_tri = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column for column in upper_tri.columns
        if any(upper_tri[column] > correlation_threshold)
    ]

    if to_drop:
        print(
            f"Removing {len(to_drop)} highly correlated features "
            f"(threshold={correlation_threshold}):")
        for col in to_drop:
            correlations = upper_tri[col][upper_tri[col]
                                          > correlation_threshold]
            for corr_col, corr_val in correlations.items():
                print(f"  - {col} <-> {corr_col}: {corr_val:.3f}")
        X_train = X_train.drop(columns=to_drop)
        X_test = X_test.drop(columns=to_drop)
    else:
        print(
            f"No highly correlated features found (threshold={correlation_threshold})")

    print("="*60 + "\n")

    return X_train, X_test


def feature_engineer(
    X_train,
    y_train,
    X_test=None,
    X_val=None,
    smoothing=10,
    n_splits=5,
    random_state=42,

    # ======================================
    # FEATURE ENGINEERING CONFIG
    # ======================================

    # Recombine binary one-hot columns
    USE_RECOMBINE_BINARY=True,

    # Structural features
    USE_HEIGHT_PER_FLOOR=True,
    USE_SLENDERNESS=True,

    # Age features
    USE_AGE_GROUP=True,

    # Ground interactions
    USE_GROUND_HEIGHT_RAW=True,
    USE_GROUND_HEIGHT_BINNED=True,

    # Geographic aggregations
    USE_GEO2_AVG_AGE=True,

    # OOF target mean features
    USE_GEO2_TARGET_MEAN=True,
    USE_GEO3_TARGET_MEAN=True,

    # Rare category grouping
    USE_RARE_CATEGORY_GROUPING=True,
    RARE_CATEGORY_THRESHOLD=20,

    # Target encoding
    USE_TARGET_ENCODING=True,

    # Cleanup
    DROP_ORIGINAL_FEATURES=True
):
    """
    Comprehensive feature engineering pipeline.

    Pipeline:
        1. Recombine binary one-hot features
        2. Create engineered numerical features
        3. Create interaction features
        4. Create aggregation features
        5. Create OOF target mean features
        6. Handle rare categories
        7. Apply target encoding

    IMPORTANT:
        - All train statistics are computed using TRAIN ONLY
        - OOF encoding is used for target-based aggregation features
        - Validation/test data are transformed with mappings learned on train
    """

    print("=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)

    # =====================================================
    # COPY DATA
    # =====================================================

    X_train = X_train.copy()

    if X_test is not None:
        X_test = X_test.copy()

    if X_val is not None:
        X_val = X_val.copy()

    def active_datasets():
        return [X for X in (X_train, X_test, X_val) if X is not None]

    # =====================================================
    # STEP 1 - RECOMBINE BINARY FEATURES
    # =====================================================

    if USE_RECOMBINE_BINARY:

        print("\n[INFO] Recombining binary features...")

        X_train, X_test, X_val = recombine_binary_features(
            X_train=X_train,
            X_test=X_test,
            X_val=X_val
        )

    # =====================================================
    # STEP 2 - STRUCTURAL FEATURES
    # =====================================================

    print("\n[INFO] Creating structural features...")

    for X in active_datasets():

        # -----------------------------------------
        # Height per floor
        # -----------------------------------------

        if USE_HEIGHT_PER_FLOOR:
            X["height_per_floor"] = (
                X["height_percentage"] /
                X["count_floors_pre_eq"].replace(0, 1)
            )

        # -----------------------------------------
        # Slenderness
        # -----------------------------------------

        if USE_SLENDERNESS:
            X["slenderness"] = (
                X["height_percentage"] /
                (X["area_percentage"] + 1)
            )

        # -----------------------------------------
        # Age Group
        # -----------------------------------------

        if USE_AGE_GROUP:
            X["age_group"] = pd.cut(
                X["age"],
                bins=[0, 10, 30, 60, 1000],
                labels=["new", "medium", "old", "very_old"],
                include_lowest=True
            )

    # =====================================================
    # STEP 3 - INTERACTION FEATURES
    # =====================================================

    print("\n[INFO] Creating interaction features...")

    for X in active_datasets():

        # -----------------------------------------
        # RAW Ground Height Interaction
        # -----------------------------------------

        if USE_GROUND_HEIGHT_RAW:
            X["ground_height_raw"] = (
                X["ground_floor_type"].astype(str)
                + "_"
                + X["height_percentage"].astype(str)
            )

        # -----------------------------------------
        # BINNED Ground Height Interaction
        # -----------------------------------------

        if USE_GROUND_HEIGHT_BINNED:
            height_bins = pd.cut(
                X["height_percentage"],
                bins=[-np.inf, 3, 6, 10, np.inf],
                labels=["low", "medium", "high", "very_high"]
            )

            X["ground_height_binned"] = (
                X["ground_floor_type"].astype(str)
                + "_"
                + height_bins.astype(str)
            )

    # =====================================================
    # STEP 4 - GEO AGGREGATION FEATURES
    # =====================================================

    print("\n[INFO] Creating aggregation features...")

    # -----------------------------------------
    # GEO2 AVG AGE
    # Leakage-safe predictor aggregation
    # -----------------------------------------

    if USE_GEO2_AVG_AGE:
        geo2_avg_age = X_train.groupby("geo_level_2_id")["age"].mean()
        global_avg_age = X_train["age"].mean()

        for X in active_datasets():
            X["geo2_avg_age"] = (
                X["geo_level_2_id"]
                .map(geo2_avg_age)
                .fillna(global_avg_age)
            )

    # =====================================================
    # STEP 5 - OOF TARGET MEAN FEATURES
    # =====================================================

    # -----------------------------------------
    # GEO2 TARGET MEAN
    # -----------------------------------------

    if USE_GEO2_TARGET_MEAN:

        print("\n[INFO] Creating OOF geo2 target mean...")

        X_train["geo2_target_mean"] = create_oof_target_mean(
            X=X_train,
            y=y_train,
            group_col="geo_level_2_id",
            n_splits=n_splits,
            random_state=random_state
        )

        global_mean = y_train.mean()

        geo2_mapping = (
            pd.concat([X_train["geo_level_2_id"], y_train], axis=1)
            .groupby("geo_level_2_id")[y_train.name]
            .mean()
        )

        for X in (X_test, X_val):
            if X is None:
                continue
            X["geo2_target_mean"] = (
                X["geo_level_2_id"]
                .map(geo2_mapping)
                .fillna(global_mean)
            )

    # -----------------------------------------
    # GEO3 TARGET MEAN
    # -----------------------------------------

    if USE_GEO3_TARGET_MEAN:

        print("\n[INFO] Creating OOF geo3 target mean...")

        X_train["geo3_target_mean"] = create_oof_target_mean(
            X=X_train,
            y=y_train,
            group_col="geo_level_3_id",
            n_splits=n_splits,
            random_state=random_state
        )

        global_mean = y_train.mean()

        geo3_mapping = (
            pd.concat([X_train["geo_level_3_id"], y_train], axis=1)
            .groupby("geo_level_3_id")[y_train.name]
            .mean()
        )

        for X in (X_test, X_val):
            if X is None:
                continue
            X["geo3_target_mean"] = (
                X["geo_level_3_id"]
                .map(geo3_mapping)
                .fillna(global_mean)
            )

    # =====================================================
    # STEP 6 - DROP ORIGINAL FEATURES
    # =====================================================

    if DROP_ORIGINAL_FEATURES:

        print("\n[INFO] Dropping original features...")

        cols_to_drop = [
            "height_percentage",
            "count_floors_pre_eq",
            "area_percentage",
            "age",
            "ground_floor_type"
        ]

        for X in active_datasets():
            X.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    # =====================================================
    # STEP 7 - HANDLE RARE CATEGORIES
    # =====================================================

    if USE_RARE_CATEGORY_GROUPING:

        print("\n[INFO] Handling rare categories...")

        categorical_cols = X_train.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        for col in categorical_cols:
            value_counts = X_train[col].value_counts()
            rare_categories = value_counts[
                value_counts < RARE_CATEGORY_THRESHOLD
            ].index

            if len(rare_categories) == 0:
                continue

            X_train[col] = X_train[col].astype("object").replace(
                rare_categories,
                "OTHER"
            )

            for X in (X_test, X_val):
                if X is None or col not in X.columns:
                    continue
                X[col] = X[col].astype("object").replace(
                    rare_categories,
                    "OTHER"
                )

    # =====================================================
    # STEP 8 - TARGET ENCODING
    # =====================================================

    if USE_TARGET_ENCODING:

        print("\n[INFO] Applying target encoding...")

        categorical_cols = X_train.select_dtypes(
            include=["object", "string", "category"]
        ).columns.tolist()

        geo_features = [
            col for col in X_train.columns
            if col.startswith("geo_level_")
        ]

        target_encode_cols = geo_features.copy()

        additional_categorical_features = [
            "superstructure_type",
            "secondary_use_type",
            "age_group",
            "ground_height_raw",
            "ground_height_binned"
        ]

        for col in additional_categorical_features:
            if col in categorical_cols:
                target_encode_cols.append(col)

        target_encode_cols = list(dict.fromkeys(target_encode_cols))

        print(f"\nTarget Encoding Features ({len(target_encode_cols)}):")
        print(target_encode_cols)

        if target_encode_cols:
            X_train, X_test, X_val = target_encode_features(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                features=target_encode_cols,
                smoothing=smoothing,
                X_val=X_val
            )

    # Keep validation/test column order exactly aligned to train.
    train_columns = X_train.columns.tolist()

    if X_test is not None:
        X_test = X_test.reindex(columns=train_columns, fill_value=0)

    if X_val is not None:
        X_val = X_val.reindex(columns=train_columns, fill_value=0)

    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 60)

    return X_train, X_test, X_val


def create_oof_target_mean(
    X,
    y,
    group_col,
    n_splits=5,
    random_state=42
):
    """
    Create Out-of-Fold target mean feature.

    Prevents leakage by ensuring each row only sees
    target statistics from OTHER folds.
    """

    kf = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    oof_feature = pd.Series(
        index=X.index,
        dtype=float
    )

    global_mean = y.mean()

    for train_idx, val_idx in kf.split(X):

        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]

        X_fold_val = X.iloc[val_idx]

        mapping = (
            pd.concat(
                [X_fold_train[group_col], y_fold_train],
                axis=1
            )
            .groupby(group_col)[y.name]
            .mean()
        )

        oof_feature.iloc[val_idx] = (
            X_fold_val[group_col]
            .map(mapping)
            .fillna(global_mean)
        )

    return oof_feature


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


def evaluate_model(model, X_val, y_val, metrics=None):
    """
    Evaluate a trained model on a pre-split validation set.

    The validation set must have been split (and cleaned/engineered) separately
    from the training data to ensure there is no data leakage.

    Metrics:
        - MCC (Matthews Correlation Coefficient): Balanced metric for multi-class
          classification. Range: -1 to +1, where +1 is perfect prediction.
        - Accuracy: Simple percentage of correct predictions.

    Args:
        model: Trained model (already fitted on X_tr / y_tr)
        X_val: Validation fold features
        y_val: Validation fold labels
        metrics: List of metrics to compute (default: ['mcc', 'accuracy'])

    Returns:
        results: Dictionary with computed metric values
    """
    if metrics is None:
        metrics = ['mcc', 'accuracy']

    print("="*60)
    print("EVALUATING MODEL")
    print("="*60)

    y_pred = model.predict(X_val)

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

    predictions = model.predict(X_test)

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
    submission = pd.DataFrame({
        'building_id': building_ids,
        'damage_grade': predictions
    })

    submission.to_csv('outputs//Team2_prediction.csv', index=False)

    print("="*60)
    print("SUBMISSION SAVED")
    print("="*60)
    print("File: outputs//Team2_prediction.csv")
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

    importances = model.feature_importances_

    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for idx, row in feature_importance_df.head(top_n).iterrows():
        print(f"  {row['feature']:.<45} {row['importance']:.6f}")

    zero_importance = feature_importance_df[feature_importance_df['importance'] == 0]
    if len(zero_importance) > 0:
        print(f"\n⚠ Found {len(zero_importance)} zero-importance features:")
        for feat in zero_importance['feature'].values[:10]:
            print(f"  - {feat}")
        if len(zero_importance) > 10:
            print(f"  ... and {len(zero_importance) - 10} more")
    else:
        print("\n✓ No zero-importance features found")

    print(f"\nSummary:")
    print(f"  Total features: {len(feature_names)}")
    print(f"  Non-zero importance: {(importances > 0).sum()}")
    print(
        f"  Top 5 cumulative importance: "
        f"{feature_importance_df.head(5)['importance'].sum():.2%}")
    print(
        f"  Top 10 cumulative importance: "
        f"{feature_importance_df.head(10)['importance'].sum():.2%}")

    print("="*60 + "\n")

    return feature_importance_df


def main():
    """
    Main pipeline orchestrator.

    Executes the complete ML pipeline:
    1. Load data
    2. Split into train / validation folds (BEFORE any cleaning)
    3. Clean training fold → apply same transforms to validation fold
    4. Feature engineering on training fold → apply to validation fold
    5. Train evaluation model on training fold, score on validation fold
    6. Clean full training data → apply same transforms to test set
    7. Feature engineering on full training data → apply to test set
    8. Train final model on full training data
    9. Analyze feature importance
    10. Generate predictions on test set
    11. Save submission file
    """

    print("\n" + "=" * 60)
    print("EARTHQUAKE DAMAGE PREDICTION PIPELINE")
    print("=" * 60 + "\n")

    np.random.seed(42)

    # Load raw data
    X_train_full, y_train_full, X_test, test_ids = load_data()

    # Split BEFORE preprocessing to avoid leakage
    X_tr, X_val, y_tr, y_val = split_data(
        X_train_full,
        y_train_full,
        test_size=0.2,
        random_state=42
    )

    clean_args = dict(
        correlation_threshold=0.9,
        handle_missing=True,
        artificial_missing_rules={'age': [0]},
        add_missing_indicators=True
    )

    feature_args = dict(
        USE_RECOMBINE_BINARY=True,

        USE_HEIGHT_PER_FLOOR=True,
        USE_SLENDERNESS=True,

        USE_AGE_GROUP=True,

        USE_GROUND_HEIGHT_RAW=False,
        USE_GROUND_HEIGHT_BINNED=True,

        USE_GEO2_AVG_AGE=False,

        USE_GEO2_TARGET_MEAN=False,
        USE_GEO3_TARGET_MEAN=False,

        USE_RARE_CATEGORY_GROUPING=True,
        RARE_CATEGORY_THRESHOLD=50,

        USE_TARGET_ENCODING=True,

        DROP_ORIGINAL_FEATURES=False,

        smoothing=10,
        n_splits=5,
        random_state=42
    )

    print(">>> Preparing validation folds")

    # Clean validation folds using train statistics only
    X_tr_clean, X_val_clean = clean_data(
        X_tr,
        y_tr,
        X_val,
        **clean_args
    )

    # Feature engineering using train statistics only
    X_tr_eng, _, X_val_eng = feature_engineer(
        X_train=X_tr_clean,
        y_train=y_tr,
        X_test=None,
        X_val=X_val_clean,
        **feature_args
    )

    print("\n>>> Training evaluation model")

    eval_model = train_model(
        X_tr_eng,
        y_tr
    )

    results = evaluate_model(
        eval_model,
        X_val_eng,
        y_val,
        metrics=['mcc', 'accuracy']
    )

    print("\n>>> Preparing full training data")

    # Prepare full train + test for final model
    X_train_clean, X_test_clean = clean_data(
        X_train_full,
        y_train_full,
        X_test,
        **clean_args
    )

    X_train_eng, X_test_eng, _ = feature_engineer(
        X_train=X_train_clean,
        y_train=y_train_full,
        X_test=X_test_clean,
        X_val=None,
        **feature_args
    )

    print("\n>>> Training final model")

    final_model = train_model(
        X_train_eng,
        y_train_full
    )

    print("\n>>> Analyzing feature importance")

    feature_importance_df = analyze_feature_importance(
        final_model,
        X_train_eng.columns.tolist(),
        top_n=15
    )

    print("\n>>> Generating predictions")

    predictions = make_predictions(
        final_model,
        X_test_eng
    )

    save_submission(
        predictions,
        test_ids
    )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"MCC: {results['mcc']:.4f}")
    print(f"Accuracy: {results['accuracy']:.4f}")

    print("\nSubmission file: Team2_prediction.csv")

    print("=" * 60)
    print("PIPELINE COMPLETE!\n")


if __name__ == '__main__':
    main()
