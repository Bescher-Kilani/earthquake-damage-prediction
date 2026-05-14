"""
Earthquake Damage Prediction Pipeline
Week 04: Feature Engineering — sklearn Pipeline Edition + Model Comparison

All logic from pipeline.py is preserved; it is now wrapped in
sklearn-compatible transformers (BaseEstimator + TransformerMixin).

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

NEW Week 04:
   - compare_models(): vergleicht Decision Tree, kNN, Logistic Regression,
     Naive Bayes, SVM auf dem gleichen preprocessed Dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import matthews_corrcoef, accuracy_score

# ── NEU Week 04: Modelle für Vergleich ──────────────────────────────────────
import time
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Helper: OOF Target Mean (unchanged from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def create_oof_target_mean(X, y, group_col, n_splits=5, random_state=42):
    """
    Create Out-of-Fold target mean feature.
    Prevents leakage by ensuring each row only sees
    target statistics from OTHER folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_feature = pd.Series(index=X.index, dtype=float)
    global_mean = y.mean()

    for train_idx, val_idx in kf.split(X):
        X_fold_train = X.iloc[train_idx]
        y_fold_train = y.iloc[train_idx]
        X_fold_val   = X.iloc[val_idx]

        mapping = (
            pd.concat([X_fold_train[group_col], y_fold_train], axis=1)
            .groupby(group_col)[y.name]
            .mean()
        )

        oof_feature.iloc[val_idx] = (
            X_fold_val[group_col].map(mapping).fillna(global_mean)
        )

    return oof_feature


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 1 – ArtificialNanReplacer
# Replaces artificial sentinel values (e.g. age=0) with NaN
# ─────────────────────────────────────────────────────────────────────────────

class ArtificialNanReplacer(BaseEstimator, TransformerMixin):
    """
    Replaces defined sentinel values with NaN.
    Corresponds to Step 1a in clean_data().
    """

    def __init__(self, rules=None):
        # rules: Dict[str, list]  e.g. {'age': [0]}
        self.rules = rules if rules is not None else {'age': [0]}

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        for feature, values in self.rules.items():
            if feature in X.columns:
                for val in values:
                    X[feature] = X[feature].replace(val, np.nan)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 2 – MissingIndicatorAdder
# Adds binary indicator columns for missing values
# ─────────────────────────────────────────────────────────────────────────────

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    """
    Creates <col>_was_missing columns.
    Corresponds to Step 1b in clean_data().
    add_indicators=False → No-op (easy toggle).
    """

    def __init__(self, add_indicators=True):
        self.add_indicators = add_indicators

    def fit(self, X, y=None):
        if self.add_indicators:
            self.missing_cols_ = [
                col for col in X.columns if X[col].isnull().sum() > 0
            ]
        else:
            self.missing_cols_ = []
        return self

    def transform(self, X):
        if not self.add_indicators:
            return X
        X = X.copy()
        for col in self.missing_cols_:
            X[f"{col}_was_missing"] = X[col].isnull().astype(int)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 3 – MedianModeImputer
# Median for numeric, mode for categorical — fit on training data only
# ─────────────────────────────────────────────────────────────────────────────

class MedianModeImputer(BaseEstimator, TransformerMixin):
    """
    Imputes missing values:
      - Numeric     → Median (from training data)
      - Categorical → Mode   (from training data)
    Corresponds to Step 1c in clean_data().
    """

    def fit(self, X, y=None):
        self.fill_values_ = {}
        for col in X.columns:
            if X[col].isnull().sum() == 0:
                continue
            if pd.api.types.is_numeric_dtype(X[col]):
                self.fill_values_[col] = X[col].median()
            else:
                mode = X[col].mode()
                self.fill_values_[col] = mode[0] if len(mode) > 0 else 'unknown'
        return self

    def transform(self, X):
        X = X.copy()
        for col, val in self.fill_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(val)
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 4 – ConstantFeatureDropper
# Removes columns with only one unique value
# ─────────────────────────────────────────────────────────────────────────────

class ConstantFeatureDropper(BaseEstimator, TransformerMixin):
    """
    Removes uninformative features (only 1 unique value).
    Corresponds to Step 2 in clean_data().
    """

    def fit(self, X, y=None):
        self.drop_cols_ = [
            col for col in X.columns if X[col].nunique() == 1
        ]
        if self.drop_cols_:
            print(f"[ConstantFeatureDropper] Dropping {len(self.drop_cols_)} uninformative features")
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols_, errors='ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 5 – LowCardinalityLabelEncoder
# Label encoding for low-cardinality categorical features
# ─────────────────────────────────────────────────────────────────────────────

class LowCardinalityLabelEncoder(BaseEstimator, TransformerMixin):
    """
    Label encoding for categorical columns that are NOT geo_level_*
    (those are handled later via target encoding).
    Corresponds to Step 3 in clean_data().
    Unknown categories in the test set are encoded as -1.
    """

    def fit(self, X, y=None):
        geo_features = {col for col in X.columns if col.startswith('geo_level_')}
        categorical_cols = X.select_dtypes(
            include=['object', 'string', 'category']
        ).columns.tolist()
        self.encode_cols_ = [c for c in categorical_cols if c not in geo_features]

        self.encoders_ = {}
        for col in self.encode_cols_:
            enc = LabelEncoder()
            enc.fit(X[col].astype(str))
            self.encoders_[col] = {cls: i for i, cls in enumerate(enc.classes_)}
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.encoders_.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(lambda x, m=mapping: m.get(x, -1))
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 6 – HighCorrelationDropper
# Removes highly correlated numeric features
# ─────────────────────────────────────────────────────────────────────────────

class HighCorrelationDropper(BaseEstimator, TransformerMixin):
    """
    Removes features that correlate above the threshold with another feature
    (upper triangle of the correlation matrix).
    Corresponds to Step 4 in clean_data().
    """

    def __init__(self, threshold=0.9):
        self.threshold = threshold

    def fit(self, X, y=None):
        numeric_only = X.select_dtypes(include=[np.number])
        corr_matrix = numeric_only.corr().abs()
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.drop_cols_ = [
            col for col in upper_tri.columns
            if any(upper_tri[col] > self.threshold)
        ]
        if self.drop_cols_:
            print(f"[HighCorrelationDropper] Dropping {len(self.drop_cols_)} highly correlated features (threshold={self.threshold})")
        return self

    def transform(self, X):
        return X.drop(columns=self.drop_cols_, errors='ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 7 – BinaryFeatureRecombiner
# Combines binary one-hot columns into a single categorical column
# ─────────────────────────────────────────────────────────────────────────────

class BinaryFeatureRecombiner(BaseEstimator, TransformerMixin):
    """
    Combines binary has_superstructure_* / has_secondary_use_* columns into
    'superstructure_type' and 'secondary_use_type'.
    Corresponds to recombine_binary_features() in pipeline.py.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    @staticmethod
    def _combine(X, cols, prefix):
        available = [c for c in cols if c in X.columns]
        if not available:
            return pd.Series("none", index=X.index)
        return X[available].apply(
            lambda row: '_'.join([
                c.replace(prefix, '') for c in available if row[c] == 1
            ]) or 'none',
            axis=1
        )

    def fit(self, X, y=None):
        if self.enabled:
            self.superstructure_cols_ = [c for c in X.columns if c.startswith('has_superstructure_')]
            self.secondary_use_cols_  = [c for c in X.columns if c.startswith('has_secondary_use_')]
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()
        if self.superstructure_cols_:
            X['superstructure_type'] = self._combine(X, self.superstructure_cols_, 'has_superstructure_')
            X.drop(columns=self.superstructure_cols_, inplace=True, errors='ignore')
        if self.secondary_use_cols_:
            X['secondary_use_type'] = self._combine(X, self.secondary_use_cols_, 'has_secondary_use_')
            X.drop(columns=self.secondary_use_cols_, inplace=True, errors='ignore')
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 8 – StructuralFeatureCreator
# Creates height_per_floor, slenderness, age_group
# ─────────────────────────────────────────────────────────────────────────────

class StructuralFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates derived structural features.
    Corresponds to Step 2 in feature_engineer().
    """

    def __init__(self,
                 use_height_per_floor=True,
                 use_slenderness=True,
                 use_age_group=True):
        self.use_height_per_floor = use_height_per_floor
        self.use_slenderness = use_slenderness
        self.use_age_group = use_age_group

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.use_height_per_floor:
            X['height_per_floor'] = (
                X['height_percentage'] / X['count_floors_pre_eq'].replace(0, 1)
            )
        if self.use_slenderness:
            X['slenderness'] = X['height_percentage'] / (X['area_percentage'] + 1)
        if self.use_age_group:
            X['age_group'] = pd.cut(
                X['age'],
                bins=[0, 10, 30, 60, 1000],
                labels=['new', 'medium', 'old', 'very_old'],
                include_lowest=True
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 9 – InteractionFeatureCreator
# Creates ground_height_raw and ground_height_binned
# ─────────────────────────────────────────────────────────────────────────────

class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates interaction features from ground_floor_type × height_percentage.
    Corresponds to Step 3 in feature_engineer().
    """

    def __init__(self, use_raw=True, use_binned=True):
        self.use_raw = use_raw
        self.use_binned = use_binned

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        if self.use_raw:
            X['ground_height_raw'] = (
                X['ground_floor_type'].astype(str)
                + '_'
                + X['height_percentage'].astype(str)
            )
        if self.use_binned:
            height_bins = pd.cut(
                X['height_percentage'],
                bins=[-np.inf, 3, 6, 10, np.inf],
                labels=['low', 'medium', 'high', 'very_high']
            )
            X['ground_height_binned'] = (
                X['ground_floor_type'].astype(str) + '_' + height_bins.astype(str)
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 10 – GeoAggregationFeatureCreator
# Creates geo2_avg_age (leakage-safe: fit on training data only)
# ─────────────────────────────────────────────────────────────────────────────

class GeoAggregationFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates geographic aggregation features.
    Corresponds to Step 4 in feature_engineer().
    Mapping is learned on training data only.
    """

    def __init__(self, use_geo2_avg_age=True):
        self.use_geo2_avg_age = use_geo2_avg_age

    def fit(self, X, y=None):
        if self.use_geo2_avg_age:
            self.geo2_avg_age_   = X.groupby('geo_level_2_id')['age'].mean()
            self.global_avg_age_ = X['age'].mean()
        return self

    def transform(self, X):
        X = X.copy()
        if self.use_geo2_avg_age:
            X['geo2_avg_age'] = (
                X['geo_level_2_id']
                .map(self.geo2_avg_age_)
                .fillna(self.global_avg_age_)
            )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 11 – OOFTargetMeanEncoder
# Creates OOF target mean features for geo2 / geo3
# ─────────────────────────────────────────────────────────────────────────────

class OOFTargetMeanEncoder(BaseEstimator, TransformerMixin):
    """
    Creates Out-of-Fold target mean features.
    During transform() (test/val), uses the mapping computed on the full
    training set — identical to pipeline.py.
    Corresponds to Step 5 in feature_engineer().
    """

    def __init__(self,
                 use_geo2=True,
                 use_geo3=True,
                 n_splits=5,
                 random_state=42):
        self.use_geo2 = use_geo2
        self.use_geo3 = use_geo3
        self.n_splits = n_splits
        self.random_state = random_state

    def fit(self, X, y=None):
        self.global_mean_ = y.mean()
        if self.use_geo2:
            self.geo2_mapping_ = (
                pd.concat([X['geo_level_2_id'], y], axis=1)
                .groupby('geo_level_2_id')[y.name]
                .mean()
            )
        if self.use_geo3:
            self.geo3_mapping_ = (
                pd.concat([X['geo_level_3_id'], y], axis=1)
                .groupby('geo_level_3_id')[y.name]
                .mean()
            )
        # Store OOF features for training data
        self._X_train_index = X.index
        if self.use_geo2:
            self._oof_geo2 = create_oof_target_mean(
                X, y, 'geo_level_2_id', self.n_splits, self.random_state
            )
        if self.use_geo3:
            self._oof_geo3 = create_oof_target_mean(
                X, y, 'geo_level_3_id', self.n_splits, self.random_state
            )
        return self

    def transform(self, X):
        X = X.copy()
        is_train = X.index.equals(self._X_train_index)

        if self.use_geo2:
            if is_train:
                X['geo2_target_mean'] = self._oof_geo2
            else:
                X['geo2_target_mean'] = (
                    X['geo_level_2_id'].map(self.geo2_mapping_).fillna(self.global_mean_)
                )

        if self.use_geo3:
            if is_train:
                X['geo3_target_mean'] = self._oof_geo3
            else:
                X['geo3_target_mean'] = (
                    X['geo_level_3_id'].map(self.geo3_mapping_).fillna(self.global_mean_)
                )
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 12 – OriginalFeatureDropper
# Drops the original raw columns that have been replaced by engineered features
# ─────────────────────────────────────────────────────────────────────────────

class OriginalFeatureDropper(BaseEstimator, TransformerMixin):
    """
    Drops raw columns replaced during feature engineering.
    Corresponds to Step 6 in feature_engineer().
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.cols_to_drop = [
            'height_percentage',
            'count_floors_pre_eq',
            'area_percentage',
            'age',
            'ground_floor_type'
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        return X.drop(columns=self.cols_to_drop, errors='ignore')


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 13 – RareCategoryGrouper
# Groups rare categories into "OTHER"
# ─────────────────────────────────────────────────────────────────────────────

class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """
    Replaces categories with fewer than `threshold` occurrences (in training set)
    with "OTHER".
    Corresponds to Step 7 in feature_engineer().
    """

    def __init__(self, enabled=True, threshold=50):
        self.enabled = enabled
        self.threshold = threshold

    def fit(self, X, y=None):
        if not self.enabled:
            self.rare_map_ = {}
            return self

        categorical_cols = X.select_dtypes(
            include=['object', 'string', 'category']
        ).columns.tolist()

        self.rare_map_ = {}
        for col in categorical_cols:
            counts = X[col].value_counts()
            rare = counts[counts < self.threshold].index.tolist()
            if rare:
                self.rare_map_[col] = rare
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()
        for col, rare_categories in self.rare_map_.items():
            if col in X.columns:
                X[col] = X[col].astype('object').replace(rare_categories, 'OTHER')
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 14 – TargetEncoder
# Smoothed target encoding for categorical features
# ─────────────────────────────────────────────────────────────────────────────

class TargetEncoder(BaseEstimator, TransformerMixin):
    """
    Smoothed target encoding.
    Fit mapping is computed on training data only.
    Corresponds to target_encode_features() in pipeline.py.
    """

    def __init__(self, enabled=True, smoothing=10):
        self.enabled = enabled
        self.smoothing = smoothing

    def _get_target_encode_cols(self, X):
        categorical_cols = set(
            X.select_dtypes(include=['object', 'string', 'category']).columns
        )
        geo_features = [c for c in X.columns if c.startswith('geo_level_')]
        additional = [
            'superstructure_type', 'secondary_use_type',
            'age_group', 'ground_height_raw', 'ground_height_binned'
        ]
        cols = geo_features + [c for c in additional if c in categorical_cols]
        return list(dict.fromkeys(cols))  # deduplicated, order preserved

    def fit(self, X, y=None):
        if not self.enabled:
            self.encodings_ = {}
            self.global_mean_ = None
            return self

        self.global_mean_ = float(y.mean())
        cols = self._get_target_encode_cols(X)
        self.encodings_ = {}

        for col in cols:
            if col not in X.columns:
                continue
            agg = pd.DataFrame({
                'target': y,
                'category': X[col]
            }).groupby('category', observed=False)['target'].agg(['mean', 'count'])

            agg['smoothed_mean'] = (
                (agg['mean'] * agg['count'] + self.smoothing * self.global_mean_)
                / (agg['count'] + self.smoothing)
            )
            self.encodings_[col] = agg['smoothed_mean'].to_dict()

        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()
        for col, mapping in self.encodings_.items():
            if col not in X.columns:
                continue
            encoded_col = col + '_target_enc'
            X[encoded_col] = (
                X[col].map(mapping).astype(float).fillna(self.global_mean_)
            )
            X.drop(columns=[col], inplace=True, errors='ignore')
        return X


# ─────────────────────────────────────────────────────────────────────────────
# Transformer 15 – ColumnAligner
# Ensures test/val columns exactly match training columns
# ─────────────────────────────────────────────────────────────────────────────

class ColumnAligner(BaseEstimator, TransformerMixin):
    """
    Aligns test/val columns to training columns via reindex.
    Corresponds to the final reindex in feature_engineer().
    """

    def fit(self, X, y=None):
        self.train_columns_ = X.columns.tolist()
        return self

    def transform(self, X):
        return X.reindex(columns=self.train_columns_, fill_value=0)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline Builder
# ─────────────────────────────────────────────────────────────────────────────

def build_pipeline(
    # clean_data args
    correlation_threshold=0.9,
    handle_missing=True,
    artificial_missing_rules=None,
    add_missing_indicators=False,

    # feature_engineer args
    USE_RECOMBINE_BINARY=True,
    USE_HEIGHT_PER_FLOOR=True,
    USE_SLENDERNESS=True,
    USE_AGE_GROUP=True,
    USE_GROUND_HEIGHT_RAW=True,
    USE_GROUND_HEIGHT_BINNED=True,
    USE_GEO2_AVG_AGE=True,
    USE_GEO2_TARGET_MEAN=True,
    USE_GEO3_TARGET_MEAN=True,
    USE_RARE_CATEGORY_GROUPING=True,
    RARE_CATEGORY_THRESHOLD=20,
    USE_TARGET_ENCODING=True,
    DROP_ORIGINAL_FEATURES=True,
    smoothing=10,
    n_splits=5,
    random_state=42,

    # NEW Week 04: welcher Classifier soll genutzt werden
    classifier=None,
):
    """
    Builds the full sklearn Pipeline from preprocessing +
    feature engineering + classifier.

    Note: OOFTargetMeanEncoder requires y at fit() time — since sklearn
    pipelines pass y through to every step, this works transparently.

    Parameters
    ----------
    classifier : sklearn estimator, optional
        Wenn None → wird DecisionTreeClassifier mit Default-Params genutzt.
        Kann mit jedem sklearn-kompatiblen Classifier überschrieben werden.
    """

    if artificial_missing_rules is None:
        artificial_missing_rules = {'age': [0]}

    if classifier is None:
        classifier = DecisionTreeClassifier(
            random_state=42,
            max_depth=10,
            min_samples_split=100,
            min_samples_leaf=50
        )

    steps = [
        # ── Cleaning ─────────────────────────────────────────────────────────
        ('artificial_nan',    ArtificialNanReplacer(rules=artificial_missing_rules)),
        ('missing_indicator', MissingIndicatorAdder(add_indicators=add_missing_indicators if handle_missing else False)),
        ('imputer',           MedianModeImputer()),
        ('constant_drop',     ConstantFeatureDropper()),
        ('label_enc',         LowCardinalityLabelEncoder()),
        ('corr_drop',         HighCorrelationDropper(threshold=correlation_threshold)),

        # ── Feature Engineering ──────────────────────────────────────────────
        ('binary_recombine',  BinaryFeatureRecombiner(enabled=USE_RECOMBINE_BINARY)),
        ('structural',        StructuralFeatureCreator(
                                  use_height_per_floor=USE_HEIGHT_PER_FLOOR,
                                  use_slenderness=USE_SLENDERNESS,
                                  use_age_group=USE_AGE_GROUP)),
        ('interaction',       InteractionFeatureCreator(
                                  use_raw=USE_GROUND_HEIGHT_RAW,
                                  use_binned=USE_GROUND_HEIGHT_BINNED)),
        ('geo_agg',           GeoAggregationFeatureCreator(use_geo2_avg_age=USE_GEO2_AVG_AGE)),
        ('oof_target_mean',   OOFTargetMeanEncoder(
                                  use_geo2=USE_GEO2_TARGET_MEAN,
                                  use_geo3=USE_GEO3_TARGET_MEAN,
                                  n_splits=n_splits,
                                  random_state=random_state)),
        ('drop_originals',    OriginalFeatureDropper(enabled=DROP_ORIGINAL_FEATURES)),
        ('rare_category',     RareCategoryGrouper(
                                  enabled=USE_RARE_CATEGORY_GROUPING,
                                  threshold=RARE_CATEGORY_THRESHOLD)),
        ('target_enc',        TargetEncoder(enabled=USE_TARGET_ENCODING, smoothing=smoothing)),
        ('col_aligner',       ColumnAligner()),

        # ── Classifier ───────────────────────────────────────────────────────
        ('classifier',        classifier),
    ]

    return Pipeline(steps)


# ─────────────────────────────────────────────────────────────────────────────
# NEW Week 04: compare_models()
# Vergleicht Decision Tree, kNN, Logistic Regression, Naive Bayes, SVM
# auf dem gleichen preprocessed Dataset
# ─────────────────────────────────────────────────────────────────────────────

def get_models_for_comparison():
    """
    Gibt alle Vergleichsmodelle zurück.
    Tuple: (classifier, needs_scaling)
    needs_scaling=True → StandardScaler wird vor dem Classifier geschaltet.
    """
    return {
        'Decision Tree': (
            DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=100,
                min_samples_leaf=50
            ),
            False   # Bäume brauchen kein Scaling
        ),
        'k-Nearest Neighbors': (
            KNeighborsClassifier(
                n_neighbors=10,
                metric='euclidean',
                n_jobs=-1
            ),
            True    # kNN ist sehr sensitiv auf Feature-Skalen
        ),
        'Logistic Regression': (
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                C=1.0
            ),
            True    # braucht skalierte Features
        ),
        'Naive Bayes': (
            GaussianNB(),
            False   # GaussianNB ist scale-invariant
        ),
        'SVM (Linear)': (
            LinearSVC(
                max_iter=2000,
                random_state=42,
                C=1.0
            ),
            True    # SVM braucht zwingend Scaling
        ),
    }


def compare_models(X_train, y_train, X_val, y_val,
                   pipeline_kwargs=None, sample_size=None):
    """
    Preprocesst die Daten einmal mit der bestehenden Pipeline,
    dann werden alle Modelle auf den gleichen Features verglichen.

    Parameters
    ----------
    X_train, y_train : Trainingsdaten (roh, vor Preprocessing)
    X_val, y_val     : Validierungsdaten (roh, vor Preprocessing)
    pipeline_kwargs  : dict, optional — Pipeline-Konfiguration (gleiche Keys
                       wie build_pipeline). Wenn None → Default-Config.
    sample_size      : int, optional — nur N Trainingsamples nutzen
                       (z.B. 20000 für schnellen Test)

    Returns
    -------
    pd.DataFrame mit MCC, Accuracy, Trainingszeit pro Modell
    """

    if pipeline_kwargs is None:
        pipeline_kwargs = dict(
            correlation_threshold=0.9,
            handle_missing=True,
            artificial_missing_rules={'age': [0]},
            add_missing_indicators=True,
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
            random_state=42,
        )

    # Optionale Stichprobe
    if sample_size is not None:
        idx = np.random.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
        X_train_s = X_train.iloc[idx]
        y_train_s = y_train.iloc[idx]
        print(f"  (Stichprobe: {len(X_train_s):,} von {len(X_train):,} Trainingssamples)")
    else:
        X_train_s = X_train
        y_train_s = y_train

    # ── Preprocessing einmal fitten (kein Classifier in der Pipe) ────────────
    print("\n" + "="*60)
    print("PREPROCESSING (einmalig für alle Modelle)")
    print("="*60)

    # build_pipeline ohne Classifier → wir ersetzen ihn mit einem Dummy
    from sklearn.dummy import DummyClassifier
    preproc_kwargs = {k: v for k, v in pipeline_kwargs.items()}
    preproc_pipeline = build_pipeline(**preproc_kwargs, classifier=DummyClassifier())

    # Nur die Preprocessing-Steps fitten (bis col_aligner)
    print("  Fitting preprocessor auf Trainingsdaten...")
    t0 = time.time()

    # Preprocessing-Steps einzeln ausführen um X_train_proc zu bekommen
    preproc_steps = Pipeline(preproc_pipeline.steps[:-1])  # alle außer classifier
    X_train_proc = preproc_steps.fit_transform(X_train_s, y_train_s)
    X_val_proc   = preproc_steps.transform(X_val)

    print(f"  Fertig in {time.time()-t0:.1f}s | Features nach Preprocessing: {X_train_proc.shape[1]}")

    # Sicherstellen dass alles numerisch ist
    X_train_proc = np.nan_to_num(X_train_proc.astype(float))
    X_val_proc   = np.nan_to_num(X_val_proc.astype(float))

    # StandardScaler einmal fitten (für Modelle die Scaling brauchen)
    scaler       = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_train_proc)
    X_val_scaled = scaler.transform(X_val_proc)

    # ── Modelle evaluieren ───────────────────────────────────────────────────
    print("\n" + "="*60)
    print("MODELLVERGLEICH")
    print("="*60)

    models  = get_models_for_comparison()
    results = []

    for name, (clf, needs_scaling) in models.items():
        X_tr_  = X_tr_scaled  if needs_scaling else X_train_proc
        X_val_ = X_val_scaled if needs_scaling else X_val_proc

        print(f"\n  Training: {name} ...", end=' ', flush=True)
        t0 = time.time()

        try:
            clf.fit(X_tr_, y_train_s)
            train_time = time.time() - t0

            y_pred = clf.predict(X_val_)
            mcc    = matthews_corrcoef(y_val, y_pred)
            acc    = accuracy_score(y_val, y_pred)

            results.append({
                'Modell':        name,
                'MCC':           round(mcc, 4),
                'Accuracy':      round(acc, 4),
                'Trainingszeit': f"{train_time:.1f}s",
                'Scaling':       'Ja' if needs_scaling else 'Nein',
                'Status':        '✓'
            })
            print(f"MCC={mcc:.4f}  Acc={acc:.4f}  ({train_time:.1f}s)")

        except Exception as e:
            train_time = time.time() - t0
            results.append({
                'Modell':        name,
                'MCC':           None,
                'Accuracy':      None,
                'Trainingszeit': f"{train_time:.1f}s",
                'Scaling':       'Ja' if needs_scaling else 'Nein',
                'Status':        f'FEHLER: {str(e)[:50]}'
            })
            print(f"FEHLER: {e}")

    # ── Ergebnistabelle ──────────────────────────────────────────────────────
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('MCC', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted.index += 1  # Ranking ab 1

    print("\n\n" + "="*60)
    print("ERGEBNISSE (sortiert nach MCC)")
    print("="*60)
    print(df_sorted.to_string())
    print("="*60)
    print(f"\nBestes Modell: {df_sorted.iloc[0]['Modell']}  "
          f"(MCC={df_sorted.iloc[0]['MCC']})")
    print("="*60 + "\n")

    return df_sorted


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (unchanged from pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    print("\n" + "="*60)
    print("LOADING DATA")
    print("="*60)

    train_values = pd.read_csv('data/train_values.csv')
    train_labels = pd.read_csv('data/train_labels.csv')
    test_values  = pd.read_csv('data/test_values.csv')

    test_ids = test_values['building_id'].values

    X_train = train_values.set_index('building_id')
    y_train = train_labels.set_index('building_id')['damage_grade']
    X_test  = test_values.set_index('building_id')

    print(f"Training samples: {len(X_train):,}")
    print(f"Test samples:     {len(X_test):,}")
    print(f"Features:         {X_train.shape[1]}")
    print("="*60 + "\n")

    return X_train, y_train, X_test, test_ids


def split_data(X_train, y_train, test_size=0.2, random_state=42):
    print("\n" + "="*60)
    print("SPLITTING DATA")
    print("="*60)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=test_size,
        random_state=random_state,
        stratify=y_train
    )

    print(f"Training fold:   {len(X_tr):,} samples ({1-test_size:.0%})")
    print(f"Validation fold: {len(X_val):,} samples ({test_size:.0%})")
    print("="*60 + "\n")

    return X_tr, X_val, y_tr, y_val


def evaluate_model(pipeline, X_val, y_val, metrics=None):
    if metrics is None:
        metrics = ['mcc', 'accuracy']

    print("="*60)
    print("EVALUATING MODEL")
    print("="*60)

    y_pred = pipeline.predict(X_val)
    results = {}

    print("Validation Set Performance:")

    if 'mcc' in metrics:
        mcc = matthews_corrcoef(y_val, y_pred)
        results['mcc'] = mcc
        print(f"  MCC:      {mcc:.4f}")

    if 'accuracy' in metrics:
        acc = accuracy_score(y_val, y_pred)
        results['accuracy'] = acc
        print(f"  Accuracy: {acc:.4f}")

    print("="*60 + "\n")
    return results


def analyze_feature_importance(pipeline, top_n=15):
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)

    classifier   = pipeline.named_steps['classifier']
    col_aligner  = pipeline.named_steps['col_aligner']
    feature_names = col_aligner.train_columns_
    importances   = classifier.feature_importances_

    df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\nTop {top_n} Most Important Features:")
    print("-" * 60)
    for _, row in df.head(top_n).iterrows():
        print(f"  {row['feature']:.<45} {row['importance']:.6f}")

    zero = df[df['importance'] == 0]
    if len(zero) > 0:
        print(f"\n⚠ {len(zero)} Zero-Importance Features gefunden")
    else:
        print("\n✓ No zero-importance features")

    print(f"\nTotal features:          {len(feature_names)}")
    print(f"Non-zero importance:     {(importances > 0).sum()}")
    print(f"Top-5 kum. Importance:   {df.head(5)['importance'].sum():.2%}")
    print(f"Top-10 kum. Importance:  {df.head(10)['importance'].sum():.2%}")
    print("="*60 + "\n")

    return df


def save_submission(predictions, building_ids, path='outputs/Team2_prediction.csv'):
    submission = pd.DataFrame({
        'building_id': building_ids,
        'damage_grade': predictions
    })
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(path, index=False)

    print("="*60)
    print("SUBMISSION SAVED")
    print("="*60)
    print(f"File: {path}")
    print(f"Rows: {len(submission):,}")
    print("="*60 + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Earthquake Pipeline Week 04')
    parser.add_argument('--compare', action='store_true',
                        help='Modellvergleich ausführen (Week 04)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Nur N Samples für Modellvergleich (schneller Test)')
    parser.add_argument('--no_val', action='store_true',
                        help='Kein Validation-Split — direkt auf vollem Trainingsset trainieren')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("EARTHQUAKE DAMAGE PREDICTION PIPELINE (sklearn)")
    print("="*60 + "\n")

    np.random.seed(42)

    # ── Load data ────────────────────────────────────────────────────────────
    X_train_full, y_train_full, X_test, test_ids = load_data()

    # ── Split BEFORE preprocessing (no leakage) ───────────────────────────────
    X_tr, X_val, y_tr, y_val = split_data(
        X_train_full, y_train_full, test_size=0.2, random_state=42
    )

    # ── Pipeline configuration ──────────────────────────────────────────────
    pipeline_kwargs = dict(
        # clean_data
        correlation_threshold=0.9,
        handle_missing=True,
        artificial_missing_rules={'age': [0]},
        add_missing_indicators=True,

        # feature_engineer
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
        random_state=42,
    )

    # ── NEU Week 04: Modellvergleich ─────────────────────────────────────────
    if args.compare:
        print(">>> Modellvergleich (Week 04)")
        comparison_results = compare_models(
            X_tr, y_tr,
            X_val, y_val,
            pipeline_kwargs=pipeline_kwargs,
            sample_size=args.sample
        )
        out_path = Path('outputs/model_comparison.csv')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_results.to_csv(out_path, index=True)
        print(f"Vergleich gespeichert: {out_path}")
        return  # nur Vergleich, kein Submission-File

    # ── Evaluation pipeline (train fold → val fold) ────────────────────────
    if not args.no_val:
        print(">>> Preparing validation pipeline")
        eval_pipeline = build_pipeline(**pipeline_kwargs)
        eval_pipeline.fit(X_tr, y_tr)
        results = evaluate_model(eval_pipeline, X_val, y_val, metrics=['mcc', 'accuracy'])

    # ── Final pipeline (full train → test) ────────────────────────────────
    print(">>> Training final pipeline on full training data")
    final_pipeline = build_pipeline(**pipeline_kwargs)
    final_pipeline.fit(X_train_full, y_train_full)

    analyze_feature_importance(final_pipeline, top_n=15)

    predictions = final_pipeline.predict(X_test)

    # Vorhersage-Verteilung ausgeben
    print("="*60)
    print("PREDICTIONS")
    print("="*60)
    print(f"Total: {len(predictions):,}")
    for grade in [1, 2, 3]:
        count = (predictions == grade).sum()
        print(f"  Grade {grade}: {count:,}")
    print("="*60 + "\n")

    save_submission(predictions, test_ids)

    if not args.no_val:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"MCC:      {results['mcc']:.4f}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print("\nSubmission file: outputs/Team2_prediction.csv")
        print("="*60)
    print("PIPELINE COMPLETE!\n")


if __name__ == '__main__':
    main()