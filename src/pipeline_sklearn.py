"""
Earthquake Damage Prediction Pipeline
Week 04: Prediction Models — sklearn Pipeline Edition + Model Comparison

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
5. Train classifier
6. Generate predictions on test set
7. Save submission file in competition format

NEW Week 04:
   - compare_models(): compares Decision Tree, kNN, Logistic Regression,
     Naive Bayes, SVM using cross-validation on X_trainval
   - Primary metric: Accuracy (competition); secondary: MCC (course)
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

# ── NEW Week 04: models for comparison ──────────────────────────────────────
import time
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

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
        # Default: Random Forest — best Accuracy (0.7411) in Week 04 model comparison
        classifier = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            random_state=42,
            n_jobs=-1
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
# Compares Decision Tree, kNN, Logistic Regression, Naive Bayes, SVM
# using cross-validation on X_trainval.
# Primary metric: Accuracy (competition); secondary: MCC (course)
# ─────────────────────────────────────────────────────────────────────────────

def get_models_for_comparison():
    """
    Returns all comparison models as a dict.
    Tuple: (classifier, needs_scaling)
    needs_scaling=True → StandardScaler is applied before the classifier.
    """
    return {
        'Decision Tree': (
            DecisionTreeClassifier(
                random_state=42,
                max_depth=10,
                min_samples_split=100,
                min_samples_leaf=50
            ),
            False
        ),
        'Random Forest': (
            RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            False   # tree-based, no scaling needed
        ),
        'LightGBM': (
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            ),
            False   # handles raw features natively
        ),
        'XGBoost': (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=6,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss',
                verbosity=0
            ),
            False
        ),
        'CatBoost': (
            CatBoostClassifier(
                iterations=300,
                learning_rate=0.05,
                depth=6,
                random_seed=42,
                verbose=0
            ),
            False   # CatBoost handles categoricals natively
        ),
        'k-Nearest Neighbors': (
            KNeighborsClassifier(
                n_neighbors=10,
                metric='euclidean',
                n_jobs=-1
            ),
            True
        ),
        'Logistic Regression': (
            LogisticRegression(
                max_iter=1000,
                random_state=42,
                n_jobs=-1,
                C=1.0
            ),
            True
        ),
        'Naive Bayes': (
            GaussianNB(),
            False
        ),
        'SVM (Linear)': (
            LinearSVC(
                max_iter=2000,
                random_state=42,
                C=1.0
            ),
            True
        ),
    }


def _build_preproc_steps(pipeline_kwargs):
    """
    Builds the preprocessing-only pipeline (no classifier).
    Used inside each CV fold to avoid leakage.
    """
    from sklearn.dummy import DummyClassifier
    full_pipeline = build_pipeline(**pipeline_kwargs, classifier=DummyClassifier())
    return Pipeline(full_pipeline.steps[:-1])  # drop the classifier step


def compare_models_single_split(X_train, y_train, X_val, y_val,
                                pipeline_kwargs=None, sample_size=None,
                                selected_models=None):
    """
    Compares models with a single train/val split (no CV).
    Faster than CV but less reliable — use for quick exploration.

    Parameters
    ----------
    X_train, y_train : training data (raw)
    X_val, y_val     : validation data (raw)
    pipeline_kwargs  : pipeline configuration dict
    sample_size      : optional — use only N samples from X_train

    Returns
    -------
    pd.DataFrame with Accuracy, MCC, and training time per model
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    if sample_size is not None:
        idx = np.random.choice(len(X_train), size=min(sample_size, len(X_train)), replace=False)
        X_train = X_train.iloc[idx]
        y_train = y_train.iloc[idx]
        print(f"  (Sample: {len(X_train):,} training samples)")

    # Preprocess once for all models
    print("\n" + "="*60)
    print("PREPROCESSING (once for all models)")
    print("="*60)
    preproc = _build_preproc_steps(pipeline_kwargs)
    t0 = time.time()
    X_train_proc = preproc.fit_transform(X_train, y_train)
    X_val_proc   = preproc.transform(X_val)
    print(f"  Done in {time.time()-t0:.1f}s | Features: {X_train_proc.shape[1]}")

    X_train_proc = np.nan_to_num(X_train_proc.astype(float))
    X_val_proc   = np.nan_to_num(X_val_proc.astype(float))

    scaler       = StandardScaler()
    X_tr_scaled  = scaler.fit_transform(X_train_proc)
    X_val_scaled = scaler.transform(X_val_proc)

    models  = get_models_for_comparison()
    if selected_models:
        models = {k: v for k, v in models.items() if k in selected_models}
    results = []

    print("\n" + "="*60)
    print("MODEL COMPARISON — single split")
    print("Primary metric: Accuracy | Secondary: MCC")
    print("="*60)

    for name, (clf, needs_scaling) in models.items():
        X_tr_  = X_tr_scaled  if needs_scaling else X_train_proc
        X_val_ = X_val_scaled if needs_scaling else X_val_proc

        print(f"\n  Training: {name} ...", end=' ', flush=True)
        t0 = time.time()
        try:
            # XGBoost requires labels 0,1,2 — shift damage_grade 1,2,3 → 0,1,2
            y_train_fit = y_train - 1 if isinstance(clf, XGBClassifier) else y_train
            clf.fit(X_tr_, y_train_fit)
            train_time = time.time() - t0
            y_pred = clf.predict(X_val_)
            # shift predictions back for XGBoost
            if isinstance(clf, XGBClassifier):
                y_pred = y_pred + 1
            acc = accuracy_score(y_val, y_pred)
            mcc = matthews_corrcoef(y_val, y_pred)
            results.append({
                'Model':   name,
                'Acc':     round(acc, 4),
                'MCC':     round(mcc, 4),
                'Time':    f"{train_time:.1f}s",
                'Status':  '✓'
            })
            print(f"Acc={acc:.4f}  MCC={mcc:.4f}  ({train_time:.1f}s)")
        except Exception as e:
            results.append({
                'Model':  name,
                'Acc':    None,
                'MCC':    None,
                'Time':   f"{time.time()-t0:.1f}s",
                'Status': f'ERROR: {str(e)[:50]}'
            })
            print(f"ERROR: {e}")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Acc', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted.index += 1

    print("\n\n" + "="*60)
    print("RESULTS — sorted by Accuracy")
    print("="*60)
    print(df_sorted.to_string())
    print("="*60)
    print(f"\nBest model: {df_sorted.iloc[0]['Model']}  "
          f"(Acc={df_sorted.iloc[0]['Acc']}  MCC={df_sorted.iloc[0]['MCC']})")
    print("="*60 + "\n")

    return df_sorted


def compare_models(X_trainval, y_trainval,
                   pipeline_kwargs=None, n_cv_splits=5, sample_size=None,
                   selected_models=None):
    """
    Runs cross-validation on the full training data for each model.

    Data flow:
        X_trainval (260k) → 5-Fold CV
        Each fold: 208k train, 52k val → Score
        → mean Accuracy and MCC across 5 folds

    Parameters
    ----------
    X_trainval, y_trainval : full training data (raw, before preprocessing)
    pipeline_kwargs        : pipeline configuration dict
    n_cv_splits            : number of CV folds (default: 5)
    sample_size            : optional — use only N samples (faster testing)

    Returns
    -------
    pd.DataFrame with mean Accuracy, mean MCC, std, and training time per model
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

    # Optional: subsample X_trainval for faster testing
    if sample_size is not None:
        idx = np.random.choice(len(X_trainval), size=min(sample_size, len(X_trainval)), replace=False)
        X_trainval = X_trainval.iloc[idx]
        y_trainval = y_trainval.iloc[idx]
        print(f"  (Sample: {len(X_trainval):,} of total training samples)")

    models  = get_models_for_comparison()
    if selected_models:
        models = {k: v for k, v in models.items() if k in selected_models}
    results = []

    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

    print("\n" + "="*60)
    print(f"CROSS-VALIDATION ({n_cv_splits} folds on X_trainval)")
    print("Primary metric: Accuracy | Secondary: MCC")
    print("="*60)

    for name, (clf, needs_scaling) in models.items():
        print(f"\n  Model: {name}")
        fold_acc  = []
        fold_mcc  = []
        t0 = time.time()

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
                # Split into fold train / fold val
                X_fold_train = X_trainval.iloc[train_idx]
                y_fold_train = y_trainval.iloc[train_idx]
                X_fold_val   = X_trainval.iloc[val_idx]
                y_fold_val   = y_trainval.iloc[val_idx]

                # Preprocessing: fit ONLY on fold train (no leakage)
                preproc = _build_preproc_steps(pipeline_kwargs)
                X_fold_train_proc = preproc.fit_transform(X_fold_train, y_fold_train)
                X_fold_val_proc   = preproc.transform(X_fold_val)

                # Ensure numeric
                X_fold_train_proc = np.nan_to_num(X_fold_train_proc.astype(float))
                X_fold_val_proc   = np.nan_to_num(X_fold_val_proc.astype(float))

                # Apply scaling if needed
                if needs_scaling:
                    scaler = StandardScaler()
                    X_fold_train_proc = scaler.fit_transform(X_fold_train_proc)
                    X_fold_val_proc   = scaler.transform(X_fold_val_proc)

                # Train and evaluate
                clf_fold = type(clf)(**clf.get_params())  # fresh copy per fold
                # XGBoost requires labels 0,1,2 — shift damage_grade 1,2,3 → 0,1,2
                y_fold_train_fit = y_fold_train - 1 if isinstance(clf_fold, XGBClassifier) else y_fold_train
                clf_fold.fit(X_fold_train_proc, y_fold_train_fit)
                y_pred = clf_fold.predict(X_fold_val_proc)
                if isinstance(clf_fold, XGBClassifier):
                    y_pred = y_pred + 1

                fold_acc.append(accuracy_score(y_fold_val, y_pred))
                fold_mcc.append(matthews_corrcoef(y_fold_val, y_pred))

                print(f"    Fold {fold_idx+1}/{n_cv_splits}  "
                      f"Acc={fold_acc[-1]:.4f}  MCC={fold_mcc[-1]:.4f}")

            train_time = time.time() - t0
            mean_acc = np.mean(fold_acc)
            std_acc  = np.std(fold_acc)
            mean_mcc = np.mean(fold_mcc)
            std_mcc  = np.std(fold_mcc)

            print(f"  → Mean  Acc={mean_acc:.4f} ± {std_acc:.4f}  "
                  f"MCC={mean_mcc:.4f} ± {std_mcc:.4f}  ({train_time:.1f}s total)")

            results.append({
                'Model':         name,
                'Acc (mean)':    round(mean_acc, 4),
                'Acc (std)':     round(std_acc, 4),
                'MCC (mean)':    round(mean_mcc, 4),
                'MCC (std)':     round(std_mcc, 4),
                'Time':          f"{train_time:.1f}s",
                'Status':        '✓'
            })

        except Exception as e:
            train_time = time.time() - t0
            results.append({
                'Model':         name,
                'Acc (mean)':    None,
                'Acc (std)':     None,
                'MCC (mean)':    None,
                'MCC (std)':     None,
                'Time':          f"{train_time:.1f}s",
                'Status':        f'ERROR: {str(e)[:50]}'
            })
            print(f"  ERROR: {e}")

    # Results table — sorted by Accuracy (primary metric)
    df = pd.DataFrame(results)
    df_sorted = df.sort_values('Acc (mean)', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted.index += 1  # rank from 1

    print("\n\n" + "="*60)
    print("RESULTS — sorted by Accuracy (primary metric)")
    print("="*60)
    print(df_sorted.to_string())
    print("="*60)
    print(f"\nBest model: {df_sorted.iloc[0]['Model']}  "
          f"(Acc={df_sorted.iloc[0]['Acc (mean)']}  MCC={df_sorted.iloc[0]['MCC (mean)']})")
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
        print(f"\n⚠ {len(zero)} zero-importance features found")
    else:
        print("\n✓ No zero-importance features")

    print(f"\nTotal features:          {len(feature_names)}")
    print(f"Non-zero importance:     {(importances > 0).sum()}")
    print(f"Top-5 cumulative importance:    {df.head(5)['importance'].sum():.2%}")
    print(f"Top-10 cumulative importance:  {df.head(10)['importance'].sum():.2%}")
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
# analyze_errors()
# Deep error analysis for the 4 ensemble models (RF, LGBM, XGB, CatBoost)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_errors(X_train_full, y_train_full, pipeline_kwargs, sample_size=None):
    from sklearn.metrics import confusion_matrix, classification_report

    ENSEMBLE_MODELS = ['Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']

    if sample_size is not None:
        idx = np.random.choice(len(X_train_full), size=min(sample_size, len(X_train_full)), replace=False)
        X_train_full = X_train_full.iloc[idx]
        y_train_full = y_train_full.iloc[idx]
        print(f"  (Sample mode: {len(X_train_full):,} samples)")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    print(f"  Train: {len(X_tr):,}  |  Val: {len(X_val):,}\n")

    print("PREPROCESSING (once for all models)")
    preproc = _build_preproc_steps(pipeline_kwargs)
    t0 = time.time()
    X_tr_proc  = preproc.fit_transform(X_tr, y_tr)
    X_val_proc = preproc.transform(X_val)
    print(f"  Done in {time.time()-t0:.1f}s | Features: {X_tr_proc.shape[1]}")

    col_aligner   = preproc.named_steps.get('col_aligner')
    feature_names = (col_aligner.train_columns_ if col_aligner is not None
                     else [f"f{i}" for i in range(X_tr_proc.shape[1])])

    X_tr_proc  = np.nan_to_num(X_tr_proc.astype(float))
    X_val_proc = np.nan_to_num(X_val_proc.astype(float))

    all_models    = get_models_for_comparison()
    ensemble_clfs = {k: v[0] for k, v in all_models.items() if k in ENSEMBLE_MODELS}
    predictions   = {}
    importances   = {}

    print("\nTRAINING ENSEMBLE MODELS")
    for name, clf in ensemble_clfs.items():
        print(f"  {name} ...", end=' ', flush=True)
        t0 = time.time()
        y_tr_fit = y_tr - 1 if isinstance(clf, XGBClassifier) else y_tr
        clf.fit(X_tr_proc, y_tr_fit)
        y_pred = clf.predict(X_val_proc)
        if isinstance(clf, XGBClassifier):
            y_pred = y_pred + 1
        y_pred = np.asarray(y_pred).ravel()
        acc = accuracy_score(y_val, y_pred)
        mcc = matthews_corrcoef(y_val, y_pred)
        print(f"Acc={acc:.4f}  MCC={mcc:.4f}  ({time.time()-t0:.1f}s)")
        predictions[name] = y_pred
        if hasattr(clf, 'feature_importances_'):
            importances[name] = clf.feature_importances_

    # STEP 1: Confusion Matrix + Per-Class Metrics
    print("\nSTEP 1: CONFUSION MATRIX + PER-CLASS METRICS")
    for name, y_pred in predictions.items():
        print(f"\n  {name}")
        cm = confusion_matrix(y_val, y_pred, labels=[1, 2, 3])
        print("  Confusion Matrix (rows=true, cols=predicted):")
        print(f"  {'':12} Pred 1   Pred 2   Pred 3")
        for i, g in enumerate([1, 2, 3]):
            row = "  ".join(f"{cm[i, j]:>7,}" for j in range(3))
            print(f"  True {g}    {row}")
        print(f"\n{classification_report(y_val, y_pred, labels=[1,2,3], target_names=['Grade 1','Grade 2','Grade 3'], digits=4)}")
        cm_nodiag = cm.copy()
        np.fill_diagonal(cm_nodiag, 0)
        wt, wp = np.unravel_index(cm_nodiag.argmax(), cm_nodiag.shape)
        print(f"  Biggest confusion: Grade {wt+1} predicted as Grade {wp+1} ({cm_nodiag[wt, wp]:,} samples)")

    # STEP 2: Hard Cases
    print("\nSTEP 2: DISAGREEMENT ANALYSIS — HARD CASES")
    y_val_arr   = y_val.values
    wrong_masks = {name: (pred != y_val_arr) for name, pred in predictions.items()}
    print("\n  Per-model error rate:")
    for name, mask in wrong_masks.items():
        print(f"    {name:<20} {mask.mean():.2%}  ({mask.sum():,} wrong)")

    hard_mask = np.ones(len(y_val_arr), dtype=bool)
    for mask in wrong_masks.values():
        hard_mask = hard_mask & mask
    n_hard = hard_mask.sum()

    any_wrong = np.zeros(len(y_val_arr), dtype=bool)
    for mask in wrong_masks.values():
        any_wrong = any_wrong | mask
    easy_mask = ~any_wrong

    print(f"\n  Wrong by ALL 4 models:   {n_hard:,} ({n_hard/len(y_val_arr):.2%})")
    print(f"  Correct by ALL 4 models: {easy_mask.sum():,} ({easy_mask.mean():.2%})")
    print("\n  Grade distribution — hard cases vs. all val:")
    for g in [1, 2, 3]:
        hp = (y_val_arr[hard_mask] == g).mean() if n_hard > 0 else 0
        ap = (y_val_arr == g).mean()
        print(f"    Grade {g}:  hard={hp:.1%}  all={ap:.1%}")

    hard_cases_df = X_val.iloc[hard_mask].copy()
    hard_cases_df['true_grade'] = y_val_arr[hard_mask]
    for name, pred in predictions.items():
        hard_cases_df[f'pred_{name.replace(" ", "_").lower()}'] = pred[hard_mask]

    # STEP 3: Feature Profile
    print("\nSTEP 3: FEATURE PROFILE — HARD vs. CORRECT")
    numeric_features = [c for c in ['age', 'height_percentage', 'area_percentage',
                                     'count_floors_pre_eq', 'count_families'] if c in X_val.columns]
    print(f"  {'Feature':<30}  {'Hard':>10}  {'Correct':>10}  {'Diff':>8}")
    for feat in numeric_features:
        hm = X_val.iloc[hard_mask][feat].mean()
        em = X_val.iloc[easy_mask][feat].mean()
        print(f"  {feat:<30}  {hm:>10.2f}  {em:>10.2f}  {hm-em:>+8.2f}")

    # STEP 4: Feature Importance
    print("\nSTEP 4: FEATURE IMPORTANCE — TOP 15 PER MODEL")
    importance_df = pd.DataFrame({'feature': feature_names})
    for name, imp in importances.items():
        importance_df[name] = imp
    importance_df = importance_df.set_index('feature')

    top15 = {name: set(importance_df[name].sort_values(ascending=False).head(15).index)
             for name in importances}
    consensus = set.intersection(*top15.values()) if top15 else set()
    print(f"\n  In Top 15 for ALL 4 models ({len(consensus)} features):")
    for f in sorted(consensus):
        print(f"    ✓  {f}")

    for name in importances:
        print(f"\n  {name} — Top 15:")
        ranked = importance_df[name].sort_values(ascending=False).head(15)
        for rank, (feat, imp) in enumerate(ranked.items(), 1):
            marker = '★' if feat in consensus else ' '
            print(f"    {rank:>2}. {feat:<40} {imp:>10.6f} {marker}")

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_cases_df.to_csv(out_dir / 'hard_cases.csv')
    importance_df.reset_index().to_csv(out_dir / 'feature_importance_comparison.csv', index=False)
    print(f"\n  Saved: outputs/hard_cases.csv ({n_hard:,} rows)")
    print(f"  Saved: outputs/feature_importance_comparison.csv")

    return hard_cases_df, importance_df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Earthquake Pipeline Week 04')
    parser.add_argument('--compare', action='store_true',
                        help='Run model comparison with cross-validation on full training data')
    parser.add_argument('--no-cv', action='store_true',
                        help='Compare models with a single train/val split instead of CV (faster)')
    parser.add_argument('--sample', type=int, default=None,
                        help='Use only N samples for CV (faster testing, e.g. --sample 20000)')
    parser.add_argument('--models', type=str, default=None,
                        help='Comma-separated list of models to compare, e.g. --models "Decision Tree,Random Forest,LightGBM"'
                             ' | Available: Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost,'
                             ' k-Nearest Neighbors, Logistic Regression, Naive Bayes, SVM (Linear)')
    parser.add_argument('--analyze', action='store_true',
                        help='Deep error analysis — fixed to the 4 ensemble models (RF, LGBM, XGB, CatBoost)')
    args = parser.parse_args()

    print("\n" + "="*60)
    print("EARTHQUAKE DAMAGE PREDICTION PIPELINE (sklearn)")
    print("="*60 + "\n")

    np.random.seed(42)

    # ── Load data ─────────────────────────────────────────────────────────
    X_train_full, y_train_full, X_test, test_ids = load_data()

    # ── Pipeline configuration ─────────────────────────────────────────────
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

    # ── --analyze: deep error analysis ───────────────────────────────────
    if args.analyze:
        print(">>> Deep Error Analysis — 4 Ensemble Models (RF, LGBM, XGB, CatBoost)")
        analyze_errors(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            sample_size=getattr(args, 'sample', None),
        )
        return

    # ── --compare: model comparison ───────────────────────────────────────
    if args.compare:
        # Parse --models filter
        selected_models = None
        if args.models:
            selected_models = [m.strip() for m in args.models.split(',')]
            print(f">>> Selected models: {selected_models}")

        no_cv = getattr(args, 'no_cv', False)
        if no_cv:
            print(">>> Model Comparison — single train/val split (80/20)")
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train_full, y_train_full,
                test_size=0.2, random_state=42, stratify=y_train_full
            )
            comparison_results = compare_models_single_split(
                X_tr, y_tr, X_val, y_val,
                pipeline_kwargs=pipeline_kwargs,
                sample_size=args.sample,
                selected_models=selected_models
            )
        else:
            print(">>> Model Comparison — 5-Fold CV on full training data (260k)")
            comparison_results = compare_models(
                X_train_full, y_train_full,
                pipeline_kwargs=pipeline_kwargs,
                n_cv_splits=5,
                sample_size=args.sample,
                selected_models=selected_models
            )
        out_path = Path('outputs/model_comparison.csv')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_results.to_csv(out_path, index=True)
        print(f"Results saved: {out_path}")
        return

    # ── normal run: train on full 260k → submission file ──────────────────
    print(">>> Training final pipeline on full training data (260k)")
    print("    Classifier: Random Forest (best Accuracy in Week 04 comparison)")
    final_pipeline = build_pipeline(**pipeline_kwargs)
    final_pipeline.fit(X_train_full, y_train_full)

    analyze_feature_importance(final_pipeline, top_n=15)

    predictions = final_pipeline.predict(X_test)

    print("="*60)
    print("PREDICTIONS")
    print("="*60)
    print(f"Total: {len(predictions):,}")
    for grade in [1, 2, 3]:
        count = (predictions == grade).sum()
        print(f"  Grade {grade}: {count:,}")
    print("="*60 + "\n")

    save_submission(predictions, test_ids)
    print("PIPELINE COMPLETE!\n")


if __name__ == '__main__':
    main()