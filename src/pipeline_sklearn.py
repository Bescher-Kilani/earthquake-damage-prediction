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
     Naive Bayes, SVM, tree ensembles, Voting, and Stacking using
     cross-validation on X_trainval
   - Voting supports --voting-mode soft or hard
   - Stacking prints/saves the final-estimator coefficient contribution
     for each base model
   - Primary metric: Accuracy (competition); secondary: MCC (course)
"""

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.base import BaseEstimator, TransformerMixin, clone
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
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
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
        X_fold_val = X.iloc[val_idx]

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
                self.fill_values_[col] = mode[0] if len(
                    mode) > 0 else 'unknown'
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
            print(
                f"[ConstantFeatureDropper] Dropping {len(self.drop_cols_)} uninformative features")
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
        geo_features = {
            col for col in X.columns if col.startswith('geo_level_')}
        categorical_cols = X.select_dtypes(
            include=['object', 'string', 'category']
        ).columns.tolist()
        self.encode_cols_ = [
            c for c in categorical_cols if c not in geo_features]

        self.encoders_ = {}
        for col in self.encode_cols_:
            enc = LabelEncoder()
            enc.fit(X[col].astype(str))
            self.encoders_[col] = {cls: i for i,
                                   cls in enumerate(enc.classes_)}
        return self

    def transform(self, X):
        X = X.copy()
        for col, mapping in self.encoders_.items():
            if col in X.columns:
                X[col] = X[col].astype(str).map(
                    lambda x, m=mapping: m.get(x, -1))
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
            print(
                f"[HighCorrelationDropper] Dropping {len(self.drop_cols_)} highly correlated features (threshold={self.threshold})")
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
            self.superstructure_cols_ = [
                c for c in X.columns if c.startswith('has_superstructure_')]
            self.secondary_use_cols_ = [
                c for c in X.columns if c.startswith('has_secondary_use_')]
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()
        if self.superstructure_cols_:
            X['superstructure_type'] = self._combine(
                X, self.superstructure_cols_, 'has_superstructure_')
            X.drop(columns=self.superstructure_cols_,
                   inplace=True, errors='ignore')
        if self.secondary_use_cols_:
            X['secondary_use_type'] = self._combine(
                X, self.secondary_use_cols_, 'has_secondary_use_')
            X.drop(columns=self.secondary_use_cols_,
                   inplace=True, errors='ignore')
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
            X['slenderness'] = X['height_percentage'] / \
                (X['area_percentage'] + 1)
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
                X['ground_floor_type'].astype(
                    str) + '_' + height_bins.astype(str)
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
            self.geo2_avg_age_ = X.groupby('geo_level_2_id')['age'].mean()
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
                    X['geo_level_2_id'].map(
                        self.geo2_mapping_).fillna(self.global_mean_)
                )

        if self.use_geo3:
            if is_train:
                X['geo3_target_mean'] = self._oof_geo3
            else:
                X['geo3_target_mean'] = (
                    X['geo_level_3_id'].map(
                        self.geo3_mapping_).fillna(self.global_mean_)
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
                X[col] = X[col].astype('object').replace(
                    rare_categories, 'OTHER')
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
            'age_group', 'ground_height_raw', 'ground_height_binned',
            # boundary interaction features — present only when BoundaryFeatureCreator is active
            'superstructure_x_foundation', 'superstructure_x_roof',
        ]
        # Include any remaining object-typed columns not yet covered above
        covered = set(geo_features) | set(additional)
        extra = [c for c in categorical_cols if c not in covered]
        cols = geo_features + [c for c in additional if c in categorical_cols] + extra
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
                (agg['mean'] * agg['count'] +
                 self.smoothing * self.global_mean_)
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
        ('artificial_nan',    ArtificialNanReplacer(
            rules=artificial_missing_rules)),
        ('missing_indicator', MissingIndicatorAdder(
            add_indicators=add_missing_indicators if handle_missing else False)),
        ('imputer',           MedianModeImputer()),
        ('constant_drop',     ConstantFeatureDropper()),
        ('label_enc',         LowCardinalityLabelEncoder()),
        ('corr_drop',         HighCorrelationDropper(
            threshold=correlation_threshold)),

        # ── Feature Engineering ──────────────────────────────────────────────
        ('binary_recombine',  BinaryFeatureRecombiner(enabled=USE_RECOMBINE_BINARY)),
        ('structural',        StructuralFeatureCreator(
            use_height_per_floor=USE_HEIGHT_PER_FLOOR,
            use_slenderness=USE_SLENDERNESS,
            use_age_group=USE_AGE_GROUP)),
        ('interaction',       InteractionFeatureCreator(
            use_raw=USE_GROUND_HEIGHT_RAW,
            use_binned=USE_GROUND_HEIGHT_BINNED)),
        ('geo_agg',           GeoAggregationFeatureCreator(
            use_geo2_avg_age=USE_GEO2_AVG_AGE)),
        ('oof_target_mean',   OOFTargetMeanEncoder(
            use_geo2=USE_GEO2_TARGET_MEAN,
            use_geo3=USE_GEO3_TARGET_MEAN,
            n_splits=n_splits,
            random_state=random_state)),
        ('drop_originals',    OriginalFeatureDropper(
            enabled=DROP_ORIGINAL_FEATURES)),
        ('rare_category',     RareCategoryGrouper(
            enabled=USE_RARE_CATEGORY_GROUPING,
            threshold=RARE_CATEGORY_THRESHOLD)),
        ('target_enc',        TargetEncoder(
            enabled=USE_TARGET_ENCODING, smoothing=smoothing)),
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

def _make_random_forest():
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )


def _make_lightgbm():
    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )


def _make_xgboost():
    return XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        n_jobs=-1,
        eval_metric='mlogloss',
        verbosity=0
    )


def _make_catboost():
    return CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        random_seed=42,
        verbose=0
    )


def _make_logistic_regression(max_iter=1000):
    return LogisticRegression(
        max_iter=max_iter,
        random_state=42,
        n_jobs=-1,
        C=1.0
    )


def get_meta_ensemble_base_estimators():
    """
    Base estimators used by Voting and Stacking.

    XGBoost is intentionally not included here because this script's target
    labels are 1, 2, 3, while the XGBoost sklearn wrapper commonly expects
    multiclass labels to start at 0. Standalone XGBoost is still compared with
    the existing label-shift logic in compare_models().
    """
    return [
        ('rf', _make_random_forest()),
        ('lgbm', _make_lightgbm()),
        ('catboost', _make_catboost()),
    ]


def build_voting_classifier(voting_mode='soft', weights=None):
    """
    Builds a VotingClassifier.

    Parameters
    ----------
    voting_mode : {'soft', 'hard'}
        soft → average predicted probabilities; hard → majority vote.
    weights : list[float] or None
        Optional voting weights in this exact order: rf, lgbm, catboost.
    """
    voting_mode = voting_mode.lower().strip()
    if voting_mode not in {'soft', 'hard'}:
        raise ValueError("voting_mode must be either 'soft' or 'hard'")

    if weights is not None and len(weights) != len(get_meta_ensemble_base_estimators()):
        raise ValueError("Voting weights must have 3 values: rf,lgbm,catboost")

    return VotingClassifier(
        estimators=get_meta_ensemble_base_estimators(),
        voting=voting_mode,
        weights=weights,
        n_jobs=-1
    )


def build_stacking_classifier(cv=5, passthrough=False):
    """
    Builds a StackingClassifier.

    The final estimator is multinomial Logistic Regression. The stacking
    contribution printed later is based on the final estimator's coefficients
    over each base model's predicted probabilities.
    """
    return StackingClassifier(
        estimators=get_meta_ensemble_base_estimators(),
        final_estimator=_make_logistic_regression(max_iter=2000),
        stack_method='predict_proba',
        cv=cv,
        passthrough=passthrough,
        n_jobs=-1
    )


def get_stacking_model_weights(stacking_clf):
    """
    Extracts an interpretable contribution table from a fitted StackingClassifier.

    Note: stacking does not learn one simple scalar weight per base model.
    It trains a final estimator on base-model predictions. This function groups
    the absolute final-estimator coefficients by base model and normalizes them
    so they are easy to read as relative contribution scores.
    """
    if not isinstance(stacking_clf, StackingClassifier):
        return pd.DataFrame()

    final_estimator = getattr(stacking_clf, 'final_estimator_', None)
    coef = getattr(final_estimator, 'coef_', None)
    if coef is None:
        return pd.DataFrame()

    coef = np.asarray(coef)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)

    estimator_names = [name for name, _ in stacking_clf.estimators]
    n_estimators = len(estimator_names)
    if n_estimators == 0:
        return pd.DataFrame()

    # If passthrough=True, original features are appended after stack features.
    # We only summarize the columns created by base model predictions.
    total_columns = coef.shape[1]
    n_stack_columns = total_columns
    if getattr(stacking_clf, 'passthrough', False) and hasattr(stacking_clf, 'n_features_in_'):
        n_stack_columns = max(total_columns - stacking_clf.n_features_in_, 0)

    if n_stack_columns == 0:
        return pd.DataFrame()

    columns_per_estimator = n_stack_columns // n_estimators
    rows = []
    for i, name in enumerate(estimator_names):
        start = i * columns_per_estimator
        end = start + columns_per_estimator
        block = coef[:, start:end]
        score = float(np.mean(np.abs(block))) if block.size else 0.0
        rows.append({
            'Model': name,
            'MeanAbsFinalCoef': score
        })

    df = pd.DataFrame(rows)
    total = df['MeanAbsFinalCoef'].sum()
    df['NormalizedWeight'] = df['MeanAbsFinalCoef'] / total if total > 0 else 0.0
    return df.sort_values('NormalizedWeight', ascending=False).reset_index(drop=True)


def format_stacking_weights(weight_df):
    """Formats a stacking contribution table into a compact printable string."""
    if weight_df is None or weight_df.empty:
        return ''
    return ', '.join(
        f"{row.Model}={row.NormalizedWeight:.2f}"
        for row in weight_df.itertuples(index=False)
    )


def average_stacking_weights(weight_dfs):
    """Averages stacking contribution tables collected across CV folds."""
    weight_dfs = [df for df in weight_dfs if df is not None and not df.empty]
    if not weight_dfs:
        return pd.DataFrame()
    combined = pd.concat(weight_dfs, ignore_index=True)
    avg = (
        combined
        .groupby('Model', as_index=False)
        .agg({
            'MeanAbsFinalCoef': 'mean',
            'NormalizedWeight': 'mean'
        })
        .sort_values('NormalizedWeight', ascending=False)
        .reset_index(drop=True)
    )
    return avg


def parse_voting_weights(weights_arg):
    """Parses --voting-weights rf,lgbm,catboost into a list of floats."""
    if weights_arg is None:
        return None
    weights = [float(x.strip()) for x in weights_arg.split(',') if x.strip()]
    if len(weights) != 3:
        raise ValueError(
            "--voting-weights must contain exactly 3 numbers: rf,lgbm,catboost")
    return weights


def _filter_selected_models(models, selected_models):
    """
    Filters model dictionary by exact names or friendly aliases.

    Examples accepted:
      --models "Voting"
      --models "Stacking"
      --models "Voting (hard),Stacking"
    """
    if not selected_models:
        return models

    selected = {m.strip().lower() for m in selected_models}
    filtered = {}

    for name, value in models.items():
        lower = name.lower()
        aliases = {lower}
        if lower.startswith('voting'):
            aliases.add('voting')
            aliases.add('voting ensemble')
        if lower.startswith('stacking'):
            aliases.add('stacking')
            aliases.add('stacking ensemble')
        if any(alias in selected for alias in aliases):
            filtered[name] = value

    return filtered


def get_models_for_comparison(voting_mode='soft', voting_weights=None,
                              include_voting=True, include_stacking=True,
                              stacking_cv=5, stacking_passthrough=False):
    """
    Returns all comparison models as a dict.
    Tuple: (classifier, needs_scaling)
    needs_scaling=True → StandardScaler is applied before the classifier.
    """
    models = {
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
            _make_random_forest(),
            False   # tree-based, no scaling needed
        ),
        'LightGBM': (
            _make_lightgbm(),
            False   # tree-based, no scaling needed
        ),
        'XGBoost': (
            _make_xgboost(),
            False
        ),
        'CatBoost': (
            _make_catboost(),
            False   # preprocessing already converts categoricals to numeric
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
            _make_logistic_regression(),
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

    if include_voting:
        voting_name = f"Voting ({voting_mode.lower()})"
        models[voting_name] = (
            build_voting_classifier(
                voting_mode=voting_mode, weights=voting_weights),
            False
        )

    if include_stacking:
        models['Stacking'] = (
            build_stacking_classifier(
                cv=stacking_cv, passthrough=stacking_passthrough),
            False
        )

    return models


def _build_preproc_steps(pipeline_kwargs):
    """
    Builds the preprocessing-only pipeline (no classifier).
    Used inside each CV fold to avoid leakage.
    """
    from sklearn.dummy import DummyClassifier
    full_pipeline = build_pipeline(
        **pipeline_kwargs, classifier=DummyClassifier())
    return Pipeline(full_pipeline.steps[:-1])  # drop the classifier step


def compare_models_single_split(X_train, y_train, X_val, y_val,
                                pipeline_kwargs=None, sample_size=None,
                                selected_models=None, voting_mode='soft',
                                voting_weights=None, include_voting=True,
                                include_stacking=True, stacking_cv=5,
                                stacking_passthrough=False):
    """
    Compares models with a single train/val split (no CV).
    Faster than CV but less reliable - use for quick exploration.

    Parameters
    ----------
    X_train, y_train : training data (raw)
    X_val, y_val     : validation data (raw)
    pipeline_kwargs  : pipeline configuration dict
    sample_size      : optional - use only N samples from X_train
    voting_mode      : {'soft', 'hard'} - mode for VotingClassifier
    voting_weights   : optional list of 3 floats for rf,lgbm,catboost

    Returns
    -------
    pd.DataFrame with Accuracy, MCC, training time, and stacking weights.
    """
    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    if sample_size is not None:
        idx = np.random.choice(len(X_train), size=min(
            sample_size, len(X_train)), replace=False)
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
    X_val_proc = preproc.transform(X_val)
    print(
        f"  Done in {time.time()-t0:.1f}s | Features: {X_train_proc.shape[1]}")

    X_train_proc = np.nan_to_num(X_train_proc.astype(float))
    X_val_proc = np.nan_to_num(X_val_proc.astype(float))

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_train_proc)
    X_val_scaled = scaler.transform(X_val_proc)

    models = get_models_for_comparison(
        voting_mode=voting_mode,
        voting_weights=voting_weights,
        include_voting=include_voting,
        include_stacking=include_stacking,
        stacking_cv=stacking_cv,
        stacking_passthrough=stacking_passthrough,
    )
    models = _filter_selected_models(models, selected_models)
    if not models:
        raise ValueError("No models selected. Check --models spelling.")

    results = []

    print("\n" + "="*60)
    print("MODEL COMPARISON - single split")
    print("Primary metric: Accuracy | Secondary: MCC")
    print("="*60)

    for name, (clf, needs_scaling) in models.items():
        X_tr_ = X_tr_scaled if needs_scaling else X_train_proc
        X_val_ = X_val_scaled if needs_scaling else X_val_proc

        print(f"\n  Training: {name} ...", end=' ', flush=True)
        t0 = time.time()
        try:
            model = clone(clf)

            # XGBoost requires labels 0,1,2 - shift damage_grade 1,2,3 -> 0,1,2.
            y_train_fit = y_train - \
                1 if isinstance(model, XGBClassifier) else y_train
            model.fit(X_tr_, y_train_fit)
            train_time = time.time() - t0
            y_pred = model.predict(X_val_)

            # Shift predictions back for standalone XGBoost.
            if isinstance(model, XGBClassifier):
                y_pred = y_pred + 1

            acc = accuracy_score(y_val, y_pred)
            mcc = matthews_corrcoef(y_val, y_pred)

            stacking_weight_summary = ''
            if isinstance(model, StackingClassifier):
                weight_df = get_stacking_model_weights(model)
                stacking_weight_summary = format_stacking_weights(weight_df)

            results.append({
                'Model':            name,
                'Acc':              round(acc, 4),
                'MCC':              round(mcc, 4),
                'Time':             f"{train_time:.1f}s",
                'Stacking weights': stacking_weight_summary,
                'Status':           'OK'
            })

            extra = f" | Stack weights: {stacking_weight_summary}" if stacking_weight_summary else ''
            print(f"Acc={acc:.4f}  MCC={mcc:.4f}  ({train_time:.1f}s){extra}")

        except Exception as e:
            results.append({
                'Model':            name,
                'Acc':              None,
                'MCC':              None,
                'Time':             f"{time.time()-t0:.1f}s",
                'Stacking weights': '',
                'Status':           f'ERROR: {str(e)[:80]}'
            })
            print(f"ERROR: {e}")

    df = pd.DataFrame(results)
    df_sorted = df.sort_values(
        'Acc', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted.index += 1

    print("\n\n" + "="*60)
    print("RESULTS - sorted by Accuracy")
    print("="*60)
    print(df_sorted.to_string())
    print("="*60)
    print(f"\nBest model: {df_sorted.iloc[0]['Model']}  "
          f"(Acc={df_sorted.iloc[0]['Acc']}  MCC={df_sorted.iloc[0]['MCC']})")
    print("="*60 + "\n")

    return df_sorted


def compare_models(X_trainval, y_trainval,
                   pipeline_kwargs=None, n_cv_splits=5, sample_size=None,
                   selected_models=None, voting_mode='soft',
                   voting_weights=None, include_voting=True,
                   include_stacking=True, stacking_cv=5,
                   stacking_passthrough=False):
    """
    Runs cross-validation on the full training data for each model.

    Data flow:
        X_trainval (260k) -> K-Fold CV
        Each fold: train fold -> validation fold -> score
        -> mean Accuracy and MCC across folds

    Parameters
    ----------
    X_trainval, y_trainval : full training data (raw, before preprocessing)
    pipeline_kwargs        : pipeline configuration dict
    n_cv_splits            : number of CV folds (default: 5)
    sample_size            : optional - use only N samples (faster testing)
    voting_mode            : {'soft', 'hard'} - mode for VotingClassifier
    voting_weights         : optional list of 3 floats for rf,lgbm,catboost

    Returns
    -------
    pd.DataFrame with mean Accuracy, mean MCC, std, training time, and
    average stacking weights across folds.
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
        idx = np.random.choice(len(X_trainval), size=min(
            sample_size, len(X_trainval)), replace=False)
        X_trainval = X_trainval.iloc[idx]
        y_trainval = y_trainval.iloc[idx]
        print(f"  (Sample: {len(X_trainval):,} of total training samples)")

    models = get_models_for_comparison(
        voting_mode=voting_mode,
        voting_weights=voting_weights,
        include_voting=include_voting,
        include_stacking=include_stacking,
        stacking_cv=stacking_cv,
        stacking_passthrough=stacking_passthrough,
    )
    models = _filter_selected_models(models, selected_models)
    if not models:
        raise ValueError("No models selected. Check --models spelling.")

    results = []

    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=42)

    print("\n" + "="*60)
    print(f"CROSS-VALIDATION ({n_cv_splits} folds on X_trainval)")
    print("Primary metric: Accuracy | Secondary: MCC")
    print("="*60)

    for name, (clf, needs_scaling) in models.items():
        print(f"\n  Model: {name}")
        fold_acc = []
        fold_mcc = []
        fold_stacking_weights = []
        t0 = time.time()

        try:
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_trainval)):
                # Split into fold train / fold val
                X_fold_train = X_trainval.iloc[train_idx]
                y_fold_train = y_trainval.iloc[train_idx]
                X_fold_val = X_trainval.iloc[val_idx]
                y_fold_val = y_trainval.iloc[val_idx]

                # Preprocessing: fit ONLY on fold train (no leakage)
                preproc = _build_preproc_steps(pipeline_kwargs)
                X_fold_train_proc = preproc.fit_transform(
                    X_fold_train, y_fold_train)
                X_fold_val_proc = preproc.transform(X_fold_val)

                # Ensure numeric
                X_fold_train_proc = np.nan_to_num(
                    X_fold_train_proc.astype(float))
                X_fold_val_proc = np.nan_to_num(X_fold_val_proc.astype(float))

                # Apply scaling if needed
                if needs_scaling:
                    scaler = StandardScaler()
                    X_fold_train_proc = scaler.fit_transform(X_fold_train_proc)
                    X_fold_val_proc = scaler.transform(X_fold_val_proc)

                # Train and evaluate. clone() supports meta-estimators such as
                # VotingClassifier and StackingClassifier more reliably than
                # type(clf)(**clf.get_params()).
                clf_fold = clone(clf)

                # XGBoost requires labels 0,1,2 - shift damage_grade 1,2,3 -> 0,1,2.
                y_fold_train_fit = y_fold_train - \
                    1 if isinstance(clf_fold, XGBClassifier) else y_fold_train
                clf_fold.fit(X_fold_train_proc, y_fold_train_fit)
                y_pred = clf_fold.predict(X_fold_val_proc)
                if isinstance(clf_fold, XGBClassifier):
                    y_pred = y_pred + 1

                if isinstance(clf_fold, StackingClassifier):
                    weight_df = get_stacking_model_weights(clf_fold)
                    fold_stacking_weights.append(weight_df)

                fold_acc.append(accuracy_score(y_fold_val, y_pred))
                fold_mcc.append(matthews_corrcoef(y_fold_val, y_pred))

                print(f"    Fold {fold_idx+1}/{n_cv_splits}  "
                      f"Acc={fold_acc[-1]:.4f}  MCC={fold_mcc[-1]:.4f}")

            train_time = time.time() - t0
            mean_acc = np.mean(fold_acc)
            std_acc = np.std(fold_acc)
            mean_mcc = np.mean(fold_mcc)
            std_mcc = np.std(fold_mcc)
            avg_stacking_weights = average_stacking_weights(
                fold_stacking_weights)
            stacking_weight_summary = format_stacking_weights(
                avg_stacking_weights)

            print(f"  Mean  Acc={mean_acc:.4f} +/- {std_acc:.4f}  "
                  f"MCC={mean_mcc:.4f} +/- {std_mcc:.4f}  ({train_time:.1f}s total)")
            if stacking_weight_summary:
                print(
                    f"  Average stacking contribution: {stacking_weight_summary}")

            results.append({
                'Model':            name,
                'Acc (mean)':       round(mean_acc, 4),
                'Acc (std)':        round(std_acc, 4),
                'MCC (mean)':       round(mean_mcc, 4),
                'MCC (std)':        round(std_mcc, 4),
                'Time':             f"{train_time:.1f}s",
                'Stacking weights': stacking_weight_summary,
                'Status':           'OK'
            })

        except Exception as e:
            train_time = time.time() - t0
            results.append({
                'Model':            name,
                'Acc (mean)':       None,
                'Acc (std)':        None,
                'MCC (mean)':       None,
                'MCC (std)':        None,
                'Time':             f"{train_time:.1f}s",
                'Stacking weights': '',
                'Status':           f'ERROR: {str(e)[:80]}'
            })
            print(f"  ERROR: {e}")

    # Results table - sorted by Accuracy (primary metric)
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(
        'Acc (mean)', ascending=False, na_position='last').reset_index(drop=True)
    df_sorted.index += 1  # rank from 1

    print("\n\n" + "="*60)
    print("RESULTS - sorted by Accuracy (primary metric)")
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
    test_values = pd.read_csv('data/test_values.csv')

    test_ids = test_values['building_id'].values

    X_train = train_values.set_index('building_id')
    y_train = train_labels.set_index('building_id')['damage_grade']
    X_test = test_values.set_index('building_id')

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

    classifier = pipeline.named_steps['classifier']
    col_aligner = pipeline.named_steps['col_aligner']
    feature_names = col_aligner.train_columns_
    importances = classifier.feature_importances_

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
    print(
        f"Top-5 cumulative importance:    {df.head(5)['importance'].sum():.2%}")
    print(
        f"Top-10 cumulative importance:  {df.head(10)['importance'].sum():.2%}")
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

def analyze_errors(X_train_full, y_train_full, pipeline_kwargs, sample_size=None,
                   voting_mode='soft', voting_weights=None, include_voting=True,
                   include_stacking=True, stacking_cv=5,
                   stacking_passthrough=False):
    from sklearn.metrics import confusion_matrix, classification_report

    analysis_model_names = ['Random Forest', 'LightGBM', 'XGBoost', 'CatBoost']
    if include_voting:
        analysis_model_names.append(f"Voting ({voting_mode.lower()})")
    if include_stacking:
        analysis_model_names.append('Stacking')

    if sample_size is not None:
        idx = np.random.choice(len(X_train_full), size=min(
            sample_size, len(X_train_full)), replace=False)
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
    X_tr_proc = preproc.fit_transform(X_tr, y_tr)
    X_val_proc = preproc.transform(X_val)
    print(f"  Done in {time.time()-t0:.1f}s | Features: {X_tr_proc.shape[1]}")

    col_aligner = preproc.named_steps.get('col_aligner')
    feature_names = (col_aligner.train_columns_ if col_aligner is not None
                     else [f"f{i}" for i in range(X_tr_proc.shape[1])])

    X_tr_proc = np.nan_to_num(X_tr_proc.astype(float))
    X_val_proc = np.nan_to_num(X_val_proc.astype(float))

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_proc)
    X_val_scaled = scaler.transform(X_val_proc)

    all_models = get_models_for_comparison(
        voting_mode=voting_mode,
        voting_weights=voting_weights,
        include_voting=include_voting,
        include_stacking=include_stacking,
        stacking_cv=stacking_cv,
        stacking_passthrough=stacking_passthrough,
    )
    analysis_models = {k: v for k,
                       v in all_models.items() if k in analysis_model_names}

    predictions = {}
    importances = {}
    stacking_weight_tables = []

    print("\nTRAINING ANALYSIS MODELS")
    for name, (clf, needs_scaling) in analysis_models.items():
        X_tr_model = X_tr_scaled if needs_scaling else X_tr_proc
        X_val_model = X_val_scaled if needs_scaling else X_val_proc

        print(f"  {name} ...", end=' ', flush=True)
        t0 = time.time()
        model = clone(clf)
        y_tr_fit = y_tr - 1 if isinstance(model, XGBClassifier) else y_tr
        model.fit(X_tr_model, y_tr_fit)
        y_pred = model.predict(X_val_model)
        if isinstance(model, XGBClassifier):
            y_pred = y_pred + 1
        y_pred = np.asarray(y_pred).ravel()
        acc = accuracy_score(y_val, y_pred)
        mcc = matthews_corrcoef(y_val, y_pred)

        if isinstance(model, StackingClassifier):
            weight_df = get_stacking_model_weights(model)
            if not weight_df.empty:
                weight_df.insert(0, 'Context', 'single_split_analysis')
                stacking_weight_tables.append(weight_df)
                print(
                    f"Acc={acc:.4f}  MCC={mcc:.4f}  ({time.time()-t0:.1f}s) | Stack weights: {format_stacking_weights(weight_df)}")
            else:
                print(f"Acc={acc:.4f}  MCC={mcc:.4f}  ({time.time()-t0:.1f}s)")
        else:
            print(f"Acc={acc:.4f}  MCC={mcc:.4f}  ({time.time()-t0:.1f}s)")

        predictions[name] = y_pred
        if hasattr(model, 'feature_importances_'):
            importances[name] = model.feature_importances_

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
        print(
            f"\n{classification_report(y_val, y_pred, labels=[1,2,3], target_names=['Grade 1','Grade 2','Grade 3'], digits=4)}")
        cm_nodiag = cm.copy()
        np.fill_diagonal(cm_nodiag, 0)
        wt, wp = np.unravel_index(cm_nodiag.argmax(), cm_nodiag.shape)
        print(
            f"  Biggest confusion: Grade {wt+1} predicted as Grade {wp+1} ({cm_nodiag[wt, wp]:,} samples)")

    # STEP 2: Hard Cases
    print("\nSTEP 2: DISAGREEMENT ANALYSIS - HARD CASES")
    y_val_arr = y_val.values
    wrong_masks = {name: (pred != y_val_arr)
                   for name, pred in predictions.items()}
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

    n_models = len(predictions)
    print(
        f"\n  Wrong by ALL {n_models} models:   {n_hard:,} ({n_hard/len(y_val_arr):.2%})")
    print(
        f"  Correct by ALL {n_models} models: {easy_mask.sum():,} ({easy_mask.mean():.2%})")
    print("\n  Grade distribution - hard cases vs. all val:")
    for g in [1, 2, 3]:
        hp = (y_val_arr[hard_mask] == g).mean() if n_hard > 0 else 0
        ap = (y_val_arr == g).mean()
        print(f"    Grade {g}:  hard={hp:.1%}  all={ap:.1%}")

    hard_cases_df = X_val.iloc[hard_mask].copy()
    hard_cases_df['true_grade'] = y_val_arr[hard_mask]
    for name, pred in predictions.items():
        safe_name = (
            name.replace(' ', '_')
                .replace('(', '')
                .replace(')', '')
                .replace('-', '_')
                .lower()
        )
        hard_cases_df[f'pred_{safe_name}'] = pred[hard_mask]

    # STEP 3: Feature Profile
    print("\nSTEP 3: FEATURE PROFILE - HARD vs. CORRECT")
    numeric_features = [c for c in ['age', 'height_percentage', 'area_percentage',
                                    'count_floors_pre_eq', 'count_families'] if c in X_val.columns]
    print(f"  {'Feature':<30}  {'Hard':>10}  {'Correct':>10}  {'Diff':>8}")
    for feat in numeric_features:
        hm = X_val.iloc[hard_mask][feat].mean()
        em = X_val.iloc[easy_mask][feat].mean()
        print(f"  {feat:<30}  {hm:>10.2f}  {em:>10.2f}  {hm-em:>+8.2f}")

    # STEP 4: Feature Importance
    print("\nSTEP 4: FEATURE IMPORTANCE - TOP 15 PER MODEL")
    if importances:
        importance_df = pd.DataFrame({'feature': feature_names})
        for name, imp in importances.items():
            importance_df[name] = imp
        importance_df = importance_df.set_index('feature')

        top15 = {name: set(importance_df[name].sort_values(ascending=False).head(15).index)
                 for name in importances}
        consensus = set.intersection(*top15.values()) if top15 else set()
        print(
            f"\n  In Top 15 for ALL feature-importance models ({len(consensus)} features):")
        for f in sorted(consensus):
            print(f"    {f}")

        for name in importances:
            print(f"\n  {name} - Top 15:")
            ranked = importance_df[name].sort_values(ascending=False).head(15)
            for rank, (feat, imp) in enumerate(ranked.items(), 1):
                marker = '*' if feat in consensus else ' '
                print(f"    {rank:>2}. {feat:<40} {imp:>10.6f} {marker}")
    else:
        importance_df = pd.DataFrame()
        print("  No native feature_importances_ available for selected models.")

    # STEP 5: Stacking weights
    stacking_weights_df = pd.concat(
        stacking_weight_tables, ignore_index=True) if stacking_weight_tables else pd.DataFrame()
    if not stacking_weights_df.empty:
        print("\nSTEP 5: STACKING MODEL CONTRIBUTION")
        print(stacking_weights_df.to_string(index=False))

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_cases_df.to_csv(out_dir / 'hard_cases.csv')
    print(f"\n  Saved: outputs/hard_cases.csv ({n_hard:,} rows)")

    if not importance_df.empty:
        importance_df.reset_index().to_csv(
            out_dir / 'feature_importance_comparison.csv', index=False)
        print("  Saved: outputs/feature_importance_comparison.csv")

    if not stacking_weights_df.empty:
        stacking_weights_df.to_csv(
            out_dir / 'stacking_weights.csv', index=False)
        print("  Saved: outputs/stacking_weights.csv")

    return hard_cases_df, importance_df, stacking_weights_df


# ─────────────────────────────────────────────────────────────────────────────
# Week 05: compare_sampling_strategies()
# Compares all sampling strategies on RF, LGBM, XGB, CatBoost
# Metrics: Accuracy (primary), MCC (secondary), Grade-1 Recall
# ─────────────────────────────────────────────────────────────────────────────

def compare_sampling_strategies(X_train_full, y_train_full,
                                 pipeline_kwargs=None,
                                 sample_size=None,
                                 use_cv=True,
                                 n_cv_splits=5,
                                 random_state=42,
                                 selected_strategies=None):
    """
    Compares all sampling strategies for Week 05.

    Strategies:
      1. Baseline            — no sampling
      2. class_weight        — balanced class weights inside the model
      3. SMOTE               — synthetic oversampling
      4. RandomUnderSampler  — undersampling of the majority class
      5. SMOTE + Under       — mix of over- and undersampling
      6. EasyEnsemble        — ensemble-based undersampling
      7. ClusterCentroids    — clustering-based undersampling
      8. ROSE                — random oversampling examples

    Models: RF, LGBM, XGB, CatBoost
    Evaluation: 80/20 single split (faster) or n_cv_splits-Fold CV (robust)
    """
    from sklearn.metrics import classification_report
    from scipy.optimize import minimize_scalar

    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler, ClusterCentroids
        from imblearn.combine import SMOTETomek
        from imblearn.ensemble import EasyEnsembleClassifier, RUSBoostClassifier
        from imblearn.over_sampling import RandomOverSampler
    except ImportError:
        print("ERROR: imbalanced-learn is not installed!")
        print("  pip install imbalanced-learn")
        return pd.DataFrame()

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    np.random.seed(random_state)

    # ── Optionally subsample training data for faster testing ────────────────
    if sample_size is not None:
        idx = np.random.choice(len(X_train_full),
                               size=min(sample_size, len(X_train_full)),
                               replace=False)
        X_train_full = X_train_full.iloc[idx]
        y_train_full = y_train_full.iloc[idx]
        print(f"  (Subsampled to {len(X_train_full):,} training samples)")

    print("\n" + "="*60)
    print("WEEK 05 — SAMPLING STRATEGY COMPARISON")
    print("Primary: Accuracy | Secondary: MCC | Also: Grade-1 Recall")
    print("="*60)

    # ── For single-split mode: preprocess once outside the strategy loop ─────
    # For CV mode: preprocessing happens inside each fold to prevent leakage
    print("\n[1/2] Preparing data...")
    preproc = _build_preproc_steps(pipeline_kwargs)

    if use_cv:
        # Preprocessing will be done per fold inside the loop
        pass
    else:
        # Single split: fit preprocessor on train, apply to both splits
        X_tr_raw, X_val_raw, y_tr_raw, y_val_raw = train_test_split(
            X_train_full, y_train_full,
            test_size=0.2, random_state=random_state, stratify=y_train_full
        )
        preproc.fit(X_tr_raw, y_tr_raw)
        # nan_to_num replaces any remaining NaNs with 0 (safe fallback)
        X_tr_proc = np.nan_to_num(preproc.transform(X_tr_raw).astype(float))
        X_val_proc = np.nan_to_num(preproc.transform(X_val_raw).astype(float))
        # Convert pandas Series to numpy arrays (required by imbalanced-learn)
        y_tr = y_tr_raw.values
        y_val = y_val_raw.values

    print("[2/2] Starting strategy comparison...\n")

    # ── Classifier factory: creates fresh instances for each strategy/fold ───
    def make_classifiers(class_weight_mode=None):
        """Returns RF, LGBM, XGB, CatBoost — optionally with class_weight."""
        cw = 'balanced' if class_weight_mode == 'balanced' else None
        clfs = {
            'RF':       RandomForestClassifier(
                            n_estimators=200, max_depth=15,
                            random_state=random_state, n_jobs=-1,
                            class_weight=cw),
            'LGBM':     LGBMClassifier(
                            n_estimators=300, learning_rate=0.05,
                            num_leaves=63, random_state=random_state,
                            n_jobs=-1, verbose=-1,
                            class_weight=cw),
            'XGB':      XGBClassifier(
                            n_estimators=300, learning_rate=0.05,
                            max_depth=6, random_state=random_state,
                            n_jobs=-1, eval_metric='mlogloss',
                            verbosity=0),
            'CatBoost': CatBoostClassifier(
                            iterations=300, learning_rate=0.05,
                            depth=6, random_seed=random_state,
                            verbose=0,
                            auto_class_weights='Balanced' if cw else None),
        }
        return clfs

    def eval_single_split(X_tr_p, y_tr_p, X_val_p, y_val_p, clfs):
        """Trains and evaluates all 4 models on one train/val split."""
        rows = []
        for clf_name, clf in clfs.items():
            t0 = time.time()
            try:
                # XGBoost expects labels 0, 1, 2 — shift damage_grade 1,2,3 -> 0,1,2
                if isinstance(clf, XGBClassifier):
                    clf.fit(X_tr_p, y_tr_p - 1)
                    y_pred = clf.predict(X_val_p) + 1
                else:
                    clf.fit(X_tr_p, y_tr_p)
                    y_pred = clf.predict(X_val_p)

                acc  = accuracy_score(y_val_p, y_pred)
                mcc  = matthews_corrcoef(y_val_p, y_pred)
                rep  = classification_report(y_val_p, y_pred,
                                             labels=[1, 2, 3],
                                             output_dict=True,
                                             zero_division=0)
                r1 = rep.get('1', rep.get(1, {})).get('recall', 0.0)
                r3 = rep.get('3', rep.get(3, {})).get('recall', 0.0)
                elapsed = time.time() - t0
                rows.append({'model': clf_name, 'acc': acc, 'mcc': mcc,
                             'grade1_recall': r1, 'grade3_recall': r3,
                             'time': elapsed})
            except Exception as e:
                print(f"    ERROR in {clf_name}: {e}")
                rows.append({'model': clf_name, 'acc': 0.0, 'mcc': 0.0,
                             'grade1_recall': 0.0, 'grade3_recall': 0.0,
                             'time': 0.0})
        return rows

    # ── Define all sampling strategies ───────────────────────────────────────
    # Format: (strategy_name, sampler_or_None, use_class_weight,
    #          is_ensemble_method, ensemble_factory_or_None)
    strategies = []

    # 1. Baseline — no sampling, no class weighting
    strategies.append(('1. Baseline', None, False, False, None))

    # 2. class_weight='balanced' — model adjusts loss weights internally
    strategies.append(('2. class_weight=balanced', None, True, False, None))

    # 3. SMOTE — creates synthetic minority samples as linear interpolations
    strategies.append(('3. SMOTE', SMOTE(random_state=random_state), False, False, None))

    # 4. RandomUnderSampler — randomly removes majority class samples
    strategies.append(('4. RandomUnderSampler',
                        RandomUnderSampler(random_state=random_state),
                        False, False, None))

    # 5. SMOTETomek — SMOTE oversampling + Tomek links undersampling combined
    strategies.append(('5. SMOTE+Under (Mix)',
                        SMOTETomek(random_state=random_state),
                        False, False, None))

    # 6. EasyEnsemble — trains multiple balanced subsets and combines predictions
    def make_easy_ensemble():
        return EasyEnsembleClassifier(
            n_estimators=10, random_state=random_state, n_jobs=-1)
    strategies.append(('6. EasyEnsemble', None, False, True, make_easy_ensemble))

    # 7. ClusterCentroids — replaces majority class samples with cluster centroids
    strategies.append(('7. ClusterCentroids',
                        ClusterCentroids(random_state=random_state),
                        False, False, None))

    # 8. ROSE — random oversampling by sampling in the neighborhood of existing points
    strategies.append(('8. ROSE (RandomOverSampler)',
                        RandomOverSampler(random_state=random_state),
                        False, False, None))

    # ── Filter strategies if --strategies was provided ───────────────────────
    if selected_strategies is not None:
        # Normalize to lowercase for case-insensitive matching
        selected_lower = {s.lower().strip() for s in selected_strategies}
        # Map friendly names to strategy name prefixes
        strategies = [
            s for s in strategies
            if any(alias in s[0].lower() for alias in selected_lower)
        ]
        if not strategies:
            print(f"  WARNING: No strategies matched {selected_strategies}.")
            print(f"  Available: Baseline, class_weight, SMOTE, RandomUnderSampler,")
            print(f"             SMOTE+Under, EasyEnsemble, ClusterCentroids, ROSE")
            return pd.DataFrame()
        print(f"  Running {len(strategies)} selected strategies.")

    # ── Run all strategies and collect results ───────────────────────────────
    all_results = []

    for strat_name, sampler, use_cw, is_ensemble, ensemble_factory in strategies:
        print(f"\n{'─'*55}")
        print(f"  Strategy: {strat_name}")
        print(f"{'─'*55}")

        t_strat = time.time()

        if use_cv:
            kf = KFold(n_splits=n_cv_splits, shuffle=True,
                       random_state=random_state)
            fold_rows = []

            for fold_i, (tr_idx, val_idx) in enumerate(
                    kf.split(X_train_full), 1):
                X_tr_f = X_train_full.iloc[tr_idx]
                y_tr_f = y_train_full.iloc[tr_idx]
                X_val_f = X_train_full.iloc[val_idx]
                y_val_f = y_train_full.iloc[val_idx]

                # Preprocess inside each fold to prevent data leakage
                preproc_f = _build_preproc_steps(pipeline_kwargs)
                X_tr_p = np.nan_to_num(
                    preproc_f.fit_transform(X_tr_f, y_tr_f).astype(float))
                X_val_p = np.nan_to_num(
                    preproc_f.transform(X_val_f).astype(float))
                y_tr_p = y_tr_f.values
                y_val_p = y_val_f.values

                # Apply sampler only to training data — never to validation!
                if sampler is not None and not is_ensemble:
                    try:
                        X_tr_p, y_tr_p = sampler.fit_resample(X_tr_p, y_tr_p)
                    except Exception as e:
                        print(f"    Sampling error fold {fold_i}: {e}")

                if is_ensemble and ensemble_factory is not None:
                    # EasyEnsemble is its own classifier — train and evaluate directly
                    t0 = time.time()
                    clf = ensemble_factory()
                    try:
                        clf.fit(X_tr_p, y_tr_p)
                        y_pred = clf.predict(X_val_p)
                        from sklearn.metrics import classification_report as _cr
                        acc  = accuracy_score(y_val_p, y_pred)
                        mcc  = matthews_corrcoef(y_val_p, y_pred)
                        rep  = _cr(y_val_p, y_pred, labels=[1, 2, 3],
                                   output_dict=True, zero_division=0)
                        r1 = rep.get('1', rep.get(1, {})).get('recall', 0.0)
                        r3 = rep.get('3', rep.get(3, {})).get('recall', 0.0)
                        fold_rows.append({
                            'model': 'EasyEnsemble',
                            'acc': acc, 'mcc': mcc,
                            'grade1_recall': r1, 'grade3_recall': r3,
                            'time': time.time() - t0
                        })
                    except Exception as e:
                        print(f"    EasyEnsemble error fold {fold_i}: {e}")
                else:
                    # Fresh classifiers per fold to avoid any state leakage
                    clfs = make_classifiers('balanced' if use_cw else None)
                    rows = eval_single_split(X_tr_p, y_tr_p,
                                             X_val_p, y_val_p, clfs)
                    fold_rows.extend(rows)

            # Average metrics across all folds
            fold_df = pd.DataFrame(fold_rows)
            agg = fold_df.groupby('model').agg(
                acc=('acc', 'mean'),
                mcc=('mcc', 'mean'),
                grade1_recall=('grade1_recall', 'mean'),
                grade3_recall=('grade3_recall', 'mean'),
                time=('time', 'mean')
            ).reset_index()

            for _, row in agg.iterrows():
                print(f"    {row['model']:<10} Acc={row['acc']:.4f}  "
                      f"MCC={row['mcc']:.4f}  "
                      f"G1-Recall={row['grade1_recall']:.3f}  "
                      f"G3-Recall={row['grade3_recall']:.3f}  "
                      f"({row['time']:.1f}s/fold)")
                all_results.append({
                    'strategy': strat_name,
                    'model': row['model'],
                    'accuracy': round(row['acc'], 4),
                    'mcc': round(row['mcc'], 4),
                    'grade1_recall': round(row['grade1_recall'], 4),
                    'grade3_recall': round(row['grade3_recall'], 4),
                    'time_s': round(row['time'], 1),
                    'eval_mode': f'{n_cv_splits}-Fold CV'
                })

        else:
            # Single split path
            if is_ensemble and ensemble_factory is not None:
                # EasyEnsemble has its own internal sampling — no separate sampler needed
                t0 = time.time()
                clf = ensemble_factory()
                try:
                    clf.fit(X_tr_proc, y_tr)
                    y_pred = clf.predict(X_val_proc)
                    acc  = accuracy_score(y_val, y_pred)
                    mcc  = matthews_corrcoef(y_val, y_pred)
                    rep  = classification_report(y_val, y_pred,
                                                 labels=[1, 2, 3],
                                                 output_dict=True,
                                                 zero_division=0)
                    r1 = rep.get('1', rep.get(1, {})).get('recall', 0.0)
                    r3 = rep.get('3', rep.get(3, {})).get('recall', 0.0)
                    elapsed = time.time() - t0
                    print(f"    EasyEnsemble   Acc={acc:.4f}  MCC={mcc:.4f}  "
                          f"G1-Recall={r1:.3f}  G3-Recall={r3:.3f}  ({elapsed:.1f}s)")
                    all_results.append({
                        'strategy': strat_name,
                        'model': 'EasyEnsemble',
                        'accuracy': round(acc, 4),
                        'mcc': round(mcc, 4),
                        'grade1_recall': round(r1, 4),
                        'grade3_recall': round(r3, 4),
                        'time_s': round(elapsed, 1),
                        'eval_mode': 'single split'
                    })
                except Exception as e:
                    print(f"    EasyEnsemble error: {e}")
            else:
                X_tr_use = X_tr_proc.copy()
                y_tr_use = y_tr.copy()

                if sampler is not None:
                    try:
                        X_tr_use, y_tr_use = sampler.fit_resample(
                            X_tr_use, y_tr_use)
                        unique, counts = np.unique(y_tr_use, return_counts=True)
                        class_dist = {int(k): int(v) for k, v in zip(unique, counts)}
                        print(f"    After sampling: {len(y_tr_use):,} samples | "
                              f"classes: {class_dist}")
                    except Exception as e:
                        print(f"    Sampling error: {e}")

                clfs = make_classifiers('balanced' if use_cw else None)
                rows = eval_single_split(X_tr_use, y_tr_use,
                                         X_val_proc, y_val, clfs)
                for row in rows:
                    print(f"    {row['model']:<10} Acc={row['acc']:.4f}  "
                          f"MCC={row['mcc']:.4f}  "
                          f"G1-Recall={row['grade1_recall']:.3f}  "
                          f"G3-Recall={row['grade3_recall']:.3f}  "
                          f"({row['time']:.1f}s)")
                    all_results.append({
                        'strategy': strat_name,
                        'model': row['model'],
                        'accuracy': round(row['acc'], 4),
                        'mcc': round(row['mcc'], 4),
                        'grade1_recall': round(row['grade1_recall'], 4),
                        'grade3_recall': round(row['grade3_recall'], 4),
                        'time_s': round(row['time'], 1),
                        'eval_mode': 'single split'
                    })

        print(f"  -> Strategy done in {time.time()-t_strat:.1f}s")

    # ── Print results summary ────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print("\n\n" + "="*80)
    print("WEEK 05 — RESULTS SUMMARY")
    print("Sorted by: Accuracy (primary), MCC (secondary)")
    print("="*80)

    # Average over all 4 models per strategy for a quick overview
    summary = results_df.groupby('strategy').agg(
        avg_accuracy=('accuracy', 'mean'),
        avg_mcc=('mcc', 'mean'),
        avg_grade1_recall=('grade1_recall', 'mean'),
        avg_grade3_recall=('grade3_recall', 'mean'),
    ).reset_index().sort_values(
        ['avg_accuracy', 'avg_mcc'], ascending=False
    )

    print(f"\n{'Strategy':<30} {'Avg Acc':>8} {'Avg MCC':>8} "
          f"{'Avg G1-Rec':>12} {'Avg G3-Rec':>12}")
    print("-" * 75)
    for _, row in summary.iterrows():
        print(f"{row['strategy']:<30} {row['avg_accuracy']:>8.4f} "
              f"{row['avg_mcc']:>8.4f} "
              f"{row['avg_grade1_recall']:>12.4f} "
              f"{row['avg_grade3_recall']:>12.4f}")

    print("\n" + "="*80)
    print("DETAIL — Best strategy per model (by Accuracy)")
    print("="*80)
    for model_name in ['RF', 'LGBM', 'XGB', 'CatBoost']:
        model_df = results_df[results_df['model'] == model_name]
        if model_df.empty:
            continue
        best = model_df.loc[model_df['accuracy'].idxmax()]
        baseline = model_df[model_df['strategy'].str.contains('Baseline')]
        baseline_acc = baseline['accuracy'].values[0] if len(baseline) > 0 else 0
        delta = best['accuracy'] - baseline_acc
        print(f"\n  {model_name}:")
        print(f"    Best strategy   : {best['strategy']}")
        print(f"    Accuracy        : {best['accuracy']:.4f}  "
              f"(Baseline: {baseline_acc:.4f}, delta={delta:+.4f})")
        print(f"    MCC             : {best['mcc']:.4f}")
        print(f"    Grade-1 Recall  : {best['grade1_recall']:.4f}")

    print("\n" + "="*80)
    print("FULL TABLE (all strategies x all models)")
    print("Sorted by Accuracy")
    print("="*80)
    full_table = results_df.sort_values(
        ['accuracy', 'mcc'], ascending=False
    ).reset_index(drop=True)
    print(full_table.to_string(index=False))

    # Save results to CSV
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'sampling_comparison.csv'
    results_df.to_csv(out_path, index=False)
    print(f"\n  Saved: {out_path}")

    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Week 05 Experiment 1 — Threshold Tuning
# Finds optimal per-class probability thresholds to maximise ACCURACY
# Works on top of any classifier that supports predict_proba()
# ─────────────────────────────────────────────────────────────────────────────

def threshold_tuning(X_train_full, y_train_full,
                     pipeline_kwargs=None,
                     random_state=42):
    """
    Trains RF and LGBM with default settings, then searches for the best
    probability threshold pair (t12, t23) that separates:
      Grade 1 vs Grade 2  →  threshold t12
      Grade 2 vs Grade 3  →  threshold t23

    Decision rule given predicted probabilities [p1, p2, p3]:
      if p1 >= t12        → predict Grade 1
      elif p3 >= t23      → predict Grade 3
      else                → predict Grade 2

    t12 and t23 are optimised via grid search to maximise ACCURACY
    (primary competition metric). MCC is reported alongside for reference.

    Outputs a comparison table: Default threshold vs. Tuned threshold.
    """
    from sklearn.metrics import classification_report

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    print("\n" + "="*60)
    print("WEEK 05 — THRESHOLD TUNING")
    print("Optimise Grade1/2 and Grade2/3 thresholds to maximise ACCURACY")
    print("Evaluation: 5-Fold CV")
    print("="*60)

    n_cv_splits = 5
    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)

    models_to_test = {
        'RF':   RandomForestClassifier(n_estimators=200, max_depth=15,
                                        random_state=random_state, n_jobs=-1),
        'LGBM': LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                num_leaves=63, random_state=random_state,
                                n_jobs=-1, verbose=-1),
    }

    results = []

    for model_name, clf_template in models_to_test.items():
        print(f"\n{'─'*55}")
        print(f"  Model: {model_name}")
        print(f"{'─'*55}")

        fold_default = []
        fold_tuned   = []
        best_t12_folds = []
        best_t23_folds = []

        for fold_i, (tr_idx, val_idx) in enumerate(
                kf.split(X_train_full), 1):

            X_tr_f = X_train_full.iloc[tr_idx]
            y_tr_f = y_train_full.iloc[tr_idx]
            X_val_f = X_train_full.iloc[val_idx]
            y_val_f = y_train_full.iloc[val_idx]

            preproc_f = _build_preproc_steps(pipeline_kwargs)
            X_tr_p  = np.nan_to_num(
                preproc_f.fit_transform(X_tr_f, y_tr_f).astype(float))
            X_val_p = np.nan_to_num(
                preproc_f.transform(X_val_f).astype(float))
            y_tr_arr  = y_tr_f.values
            y_val_arr = y_val_f.values

            clf = clone(clf_template)
            clf.fit(X_tr_p, y_tr_arr)
            proba = clf.predict_proba(X_val_p)

            # Default (argmax)
            y_pred_default = np.argmax(proba, axis=1) + 1
            acc_d = accuracy_score(y_val_arr, y_pred_default)
            mcc_d = matthews_corrcoef(y_val_arr, y_pred_default)
            rep_d = classification_report(y_val_arr, y_pred_default,
                                          labels=[1,2,3], output_dict=True,
                                          zero_division=0)
            fold_default.append({
                'acc': acc_d, 'mcc': mcc_d,
                'r1': rep_d.get('1', rep_d.get(1, {})).get('recall', 0.0),
                'r3': rep_d.get('3', rep_d.get(3, {})).get('recall', 0.0),
            })

            # Threshold search — maximise ACCURACY
            def apply_thresholds(t12, t23):
                preds = np.full(len(proba), 2, dtype=int)
                preds[proba[:, 0] >= t12] = 1
                mask_not1 = preds != 1
                preds[mask_not1 & (proba[:, 2] >= t23)] = 3
                return preds

            best_acc = acc_d
            best_t12 = 1/3
            best_t23 = 1/3

            for t12 in np.arange(0.05, 0.81, 0.05):
                for t23 in np.arange(0.05, 0.81, 0.05):
                    acc = accuracy_score(y_val_arr,
                                         apply_thresholds(t12, t23))
                    if acc > best_acc:
                        best_acc = acc
                        best_t12 = t12
                        best_t23 = t23

            # Fine-tune
            for t12 in np.arange(max(0.01, best_t12-0.05),
                                   min(0.95, best_t12+0.06), 0.01):
                for t23 in np.arange(max(0.01, best_t23-0.05),
                                      min(0.95, best_t23+0.06), 0.01):
                    acc = accuracy_score(y_val_arr,
                                         apply_thresholds(t12, t23))
                    if acc > best_acc:
                        best_acc = acc
                        best_t12 = t12
                        best_t23 = t23

            y_pred_tuned = apply_thresholds(best_t12, best_t23)
            acc_t = accuracy_score(y_val_arr, y_pred_tuned)
            mcc_t = matthews_corrcoef(y_val_arr, y_pred_tuned)
            rep_t = classification_report(y_val_arr, y_pred_tuned,
                                          labels=[1,2,3], output_dict=True,
                                          zero_division=0)
            fold_tuned.append({
                'acc': acc_t, 'mcc': mcc_t,
                'r1': rep_t.get('1', rep_t.get(1, {})).get('recall', 0.0),
                'r3': rep_t.get('3', rep_t.get(3, {})).get('recall', 0.0),
            })
            best_t12_folds.append(best_t12)
            best_t23_folds.append(best_t23)

        # Average over folds
        d_df = pd.DataFrame(fold_default)
        t_df = pd.DataFrame(fold_tuned)
        d_mean = d_df.mean()
        t_mean = t_df.mean()
        avg_t12 = float(np.mean(best_t12_folds))
        avg_t23 = float(np.mean(best_t23_folds))

        print(f"  Default (argmax):         "
              f"Acc={d_mean['acc']:.4f}  MCC={d_mean['mcc']:.4f}  "
              f"G1-Recall={d_mean['r1']:.3f}  G3-Recall={d_mean['r3']:.3f}")
        print(f"  Tuned (avg t12={avg_t12:.2f}, t23={avg_t23:.2f}):  "
              f"Acc={t_mean['acc']:.4f}  MCC={t_mean['mcc']:.4f}  "
              f"G1-Recall={t_mean['r1']:.3f}  G3-Recall={t_mean['r3']:.3f}")
        print(f"  Delta:                    "
              f"Acc={t_mean['acc']-d_mean['acc']:+.4f}  "
              f"MCC={t_mean['mcc']-d_mean['mcc']:+.4f}  "
              f"G1={t_mean['r1']-d_mean['r1']:+.3f}  "
              f"G3={t_mean['r3']-d_mean['r3']:+.3f}")

        results.append({
            'model': model_name, 'threshold': 'default (argmax)',
            'acc': round(float(d_mean['acc']), 4),
            'mcc': round(float(d_mean['mcc']), 4),
            'g1_recall': round(float(d_mean['r1']), 4),
            'g3_recall': round(float(d_mean['r3']), 4),
            't12': round(1/3, 3), 't23': round(1/3, 3),
            'eval_mode': f'{n_cv_splits}-Fold CV'
        })
        results.append({
            'model': model_name,
            'threshold': f'tuned (avg t12={avg_t12:.2f}, t23={avg_t23:.2f})',
            'acc': round(float(t_mean['acc']), 4),
            'mcc': round(float(t_mean['mcc']), 4),
            'g1_recall': round(float(t_mean['r1']), 4),
            'g3_recall': round(float(t_mean['r3']), 4),
            't12': round(avg_t12, 3), 't23': round(avg_t23, 3),
            'eval_mode': f'{n_cv_splits}-Fold CV'
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("THRESHOLD TUNING — SUMMARY (5-Fold CV)")
    print("="*80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'threshold_tuning.csv', index=False)
    print(f"\n  Saved: outputs/threshold_tuning.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Week 05 Experiment 2 — New Features for Grade 2/3 Boundary
# Adds interaction features targeting the Grade2/3 confusion
# ─────────────────────────────────────────────────────────────────────────────

class BoundaryFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Creates interaction features specifically designed to help separate
    the Grade 2/3 boundary — the main error source identified in Week 4.

    String features (superstructure_x_foundation, superstructure_x_roof)
    are picked up automatically by the TargetEncoder downstream because
    they are object-typed categorical columns.

    Numeric features (age_x_floors, floors_x_area, damage_risk_score)
    pass through directly as floats.
    """

    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not self.enabled:
            return X
        X = X.copy()

        # 1. superstructure × foundation — combined structural vulnerability
        # Kept as string → picked up by TargetEncoder downstream
        if 'superstructure_type' in X.columns and 'foundation_type' in X.columns:
            X['superstructure_x_foundation'] = (
                X['superstructure_type'].astype(str)
                + '_'
                + X['foundation_type'].astype(str)
            )

        # 2. superstructure × roof — combined material risk
        # Kept as string → picked up by TargetEncoder downstream
        if 'superstructure_type' in X.columns and 'roof_type' in X.columns:
            X['superstructure_x_roof'] = (
                X['superstructure_type'].astype(str)
                + '_'
                + X['roof_type'].astype(str)
            )

        # 3. age × floors — older and taller buildings sustain more damage
        if 'age' in X.columns and 'count_floors_pre_eq' in X.columns:
            X['age_x_floors'] = (
                X['age'].fillna(0).astype(float)
                * X['count_floors_pre_eq'].fillna(1).astype(float)
            )

        # 4. floors × area — building volume proxy
        if 'count_floors_pre_eq' in X.columns and 'area_percentage' in X.columns:
            X['floors_x_area'] = (
                X['count_floors_pre_eq'].fillna(1).astype(float)
                * X['area_percentage'].fillna(0).astype(float)
            )

        # 5. Numeric damage risk score — weighted combination of key risk factors
        if all(c in X.columns for c in
               ['height_percentage', 'count_floors_pre_eq', 'age']):
            X['damage_risk_score'] = (
                0.4 * X['height_percentage'].fillna(0).astype(float)
                + 0.4 * X['count_floors_pre_eq'].fillna(0).astype(float)
                + 0.2 * (X['age'].fillna(0).astype(float) / 100.0)
            )

        return X


def boundary_feature_experiment(X_train_full, y_train_full,
                                  pipeline_kwargs=None,
                                  n_cv_splits=5,
                                  random_state=42):
    """
    Compares baseline pipeline vs. pipeline + BoundaryFeatureCreator.
    Uses 5-Fold CV for robust evaluation (same as sampling comparison).
    Models: RF and LGBM.
    """
    from sklearn.metrics import classification_report

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    print("\n" + "="*60)
    print("WEEK 05 — BOUNDARY FEATURE EXPERIMENT")
    print("New interaction features targeting Grade 2/3 confusion")
    print(f"Evaluation: {n_cv_splits}-Fold CV")
    print("="*60)

    results = []

    for variant_name, use_boundary in [('Baseline', False),
                                        ('+ BoundaryFeatures', True)]:
        print(f"\n{'─'*55}")
        print(f"  Variant: {variant_name}")
        print(f"{'─'*55}")

        kf = KFold(n_splits=n_cv_splits, shuffle=True,
                   random_state=random_state)
        fold_rows = []

        for fold_i, (tr_idx, val_idx) in enumerate(
                kf.split(X_train_full), 1):
            X_tr_f = X_train_full.iloc[tr_idx]
            y_tr_f = y_train_full.iloc[tr_idx]
            X_val_f = X_train_full.iloc[val_idx]
            y_val_f = y_train_full.iloc[val_idx]

            # Build preprocessing pipeline per fold — inject BoundaryFeatureCreator
            from sklearn.dummy import DummyClassifier
            base_pipe = build_pipeline(**pipeline_kwargs,
                                       classifier=DummyClassifier())
            steps = base_pipe.steps[:-1]  # drop classifier step

            if use_boundary:
                # Insert BoundaryFeatureCreator after the structural step
                new_steps = []
                for name, step in steps:
                    new_steps.append((name, step))
                    if name == 'structural':
                        new_steps.append(('boundary_features',
                                          BoundaryFeatureCreator(enabled=True)))
                steps = new_steps

            preproc = Pipeline(steps)
            X_tr_p  = np.nan_to_num(
                preproc.fit_transform(X_tr_f, y_tr_f).astype(float))
            X_val_p = np.nan_to_num(
                preproc.transform(X_val_f).astype(float))

            for clf_name, clf in [
                ('RF',   RandomForestClassifier(
                             n_estimators=200, max_depth=15,
                             random_state=random_state, n_jobs=-1)),
                ('LGBM', LGBMClassifier(
                             n_estimators=300, learning_rate=0.05,
                             num_leaves=63, random_state=random_state,
                             n_jobs=-1, verbose=-1)),
            ]:
                t0 = time.time()
                clf.fit(X_tr_p, y_tr_f.values)
                y_pred = clf.predict(X_val_p)

                acc = accuracy_score(y_val_f.values, y_pred)
                mcc = matthews_corrcoef(y_val_f.values, y_pred)
                rep = classification_report(y_val_f.values, y_pred,
                                            labels=[1, 2, 3],
                                            output_dict=True,
                                            zero_division=0)
                r1 = rep.get('1', rep.get(1, {})).get('recall', 0.0)
                r3 = rep.get('3', rep.get(3, {})).get('recall', 0.0)
                fold_rows.append({
                    'variant': variant_name, 'model': clf_name,
                    'acc': acc, 'mcc': mcc,
                    'g1_recall': r1, 'g3_recall': r3,
                    'time': time.time() - t0
                })

        # Average over folds
        fold_df = pd.DataFrame(fold_rows)
        agg = fold_df.groupby(['variant', 'model']).agg(
            acc=('acc', 'mean'), mcc=('mcc', 'mean'),
            g1_recall=('g1_recall', 'mean'),
            g3_recall=('g3_recall', 'mean'),
            time=('time', 'mean')
        ).reset_index()

        for _, row in agg.iterrows():
            print(f"  {row['model']:<6} Acc={row['acc']:.4f}  "
                  f"MCC={row['mcc']:.4f}  "
                  f"G1-Recall={row['g1_recall']:.3f}  "
                  f"G3-Recall={row['g3_recall']:.3f}  "
                  f"({row['time']:.1f}s/fold)")
            results.append({
                'variant': variant_name,
                'model': row['model'],
                'accuracy': round(row['acc'], 4),
                'mcc': round(row['mcc'], 4),
                'g1_recall': round(row['g1_recall'], 4),
                'g3_recall': round(row['g3_recall'], 4),
                'eval_mode': f'{n_cv_splits}-Fold CV'
            })

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*80)
    print("BOUNDARY FEATURE EXPERIMENT — SUMMARY")
    print("="*80)
    df = pd.DataFrame(results)

    for model_name in ['RF', 'LGBM']:
        base = df[(df['model'] == model_name) &
                  (df['variant'] == 'Baseline')].iloc[0]
        new  = df[(df['model'] == model_name) &
                  (df['variant'] == '+ BoundaryFeatures')].iloc[0]
        print(f"\n  {model_name}:")
        print(f"    Baseline:          Acc={base['accuracy']:.4f}  "
              f"MCC={base['mcc']:.4f}  G3-Recall={base['g3_recall']:.3f}")
        print(f"    BoundaryFeatures:  Acc={new['accuracy']:.4f}  "
              f"MCC={new['mcc']:.4f}  G3-Recall={new['g3_recall']:.3f}")
        print(f"    Delta:             Acc={new['accuracy']-base['accuracy']:+.4f}  "
              f"MCC={new['mcc']-base['mcc']:+.4f}  "
              f"G3-Recall={new['g3_recall']-base['g3_recall']:+.3f}")

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / 'boundary_features.csv', index=False)
    print(f"\n  Saved: outputs/boundary_features.csv")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Week 05 Experiment 3 — Two-Stage Classifier on Hard Cases
# Stage 1: RF on all samples
# Stage 2: LGBM trained only on uncertain/hard cases from Stage 1
# Hard cases = samples where Stage 1 max probability < uncertainty_threshold
# ─────────────────────────────────────────────────────────────────────────────

def two_stage_classifier(X_train_full, y_train_full,
                          pipeline_kwargs=None,
                          uncertainty_threshold=0.50,
                          random_state=42):
    """
    Two-stage classification to target the hard cases identified in Week 4.

    Stage 1 — RF trained on all 260k samples.
               Samples where max(proba) < uncertainty_threshold are
               considered 'uncertain' and passed to Stage 2.

    Stage 2 — LGBM trained only on the hard/uncertain cases from Stage 1.
               Using LGBM (not RF) for Stage 2 adds diversity: the two
               models use different learning mechanisms, so Stage 2 can
               correct errors Stage 1 consistently makes.

    Evaluation: 5-Fold CV for robust results.
    Also tests multiple thresholds [0.50, 0.55, 0.60, 0.65, 0.70] to find
    the best operating point.

    Outputs comparison: Stage 1 alone vs. Two-stage combined per threshold.
    """
    from sklearn.metrics import classification_report

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    # Test multiple thresholds to find best operating point
    thresholds_to_test = [0.50, 0.55, 0.60, 0.65, 0.70]

    print("\n" + "="*60)
    print("WEEK 05 — TWO-STAGE CLASSIFIER")
    print("Stage 1: RF on all samples")
    print("Stage 2: LGBM on uncertain samples")
    print(f"Thresholds tested: {thresholds_to_test}")
    print(f"Evaluation: 5-Fold CV")
    print("="*60)

    all_results = []
    n_cv_splits = 5
    kf = KFold(n_splits=n_cv_splits, shuffle=True, random_state=random_state)

    for threshold in thresholds_to_test:
        print(f"\n{'─'*55}")
        print(f"  Threshold: {threshold}")
        print(f"{'─'*55}")

        fold_s1_rows = []
        fold_combined_rows = []

        for fold_i, (tr_idx, val_idx) in enumerate(
                kf.split(X_train_full), 1):

            X_tr_f = X_train_full.iloc[tr_idx]
            y_tr_f = y_train_full.iloc[tr_idx]
            X_val_f = X_train_full.iloc[val_idx]
            y_val_f = y_train_full.iloc[val_idx]

            # Preprocess inside fold — no leakage
            preproc_f = _build_preproc_steps(pipeline_kwargs)
            X_tr_p  = np.nan_to_num(
                preproc_f.fit_transform(X_tr_f, y_tr_f).astype(float))
            X_val_p = np.nan_to_num(
                preproc_f.transform(X_val_f).astype(float))
            y_tr_arr  = y_tr_f.values
            y_val_arr = y_val_f.values

            # ── Stage 1: RF on full fold training data ───────────────────────
            stage1 = RandomForestClassifier(
                n_estimators=200, max_depth=15,
                random_state=random_state, n_jobs=-1)
            stage1.fit(X_tr_p, y_tr_arr)

            proba_val  = stage1.predict_proba(X_val_p)
            y_pred_s1  = np.argmax(proba_val, axis=1) + 1
            max_proba  = proba_val.max(axis=1)

            acc_s1 = accuracy_score(y_val_arr, y_pred_s1)
            mcc_s1 = matthews_corrcoef(y_val_arr, y_pred_s1)
            rep_s1 = classification_report(y_val_arr, y_pred_s1,
                                            labels=[1, 2, 3],
                                            output_dict=True, zero_division=0)
            r1_s1 = rep_s1.get('1', rep_s1.get(1, {})).get('recall', 0.0)
            r3_s1 = rep_s1.get('3', rep_s1.get(3, {})).get('recall', 0.0)
            fold_s1_rows.append({'acc': acc_s1, 'mcc': mcc_s1,
                                  'r1': r1_s1, 'r3': r3_s1})

            # ── Stage 2: find hard cases via OOF on fold training data ───────
            # Use OOF so Stage 2 training data has no leakage
            inner_kf = KFold(n_splits=3, shuffle=True,
                             random_state=random_state)
            oof_proba = np.zeros((len(X_tr_p), 3))
            for in_tr, in_val in inner_kf.split(X_tr_p):
                clf_inner = RandomForestClassifier(
                    n_estimators=100, max_depth=15,
                    random_state=random_state, n_jobs=-1)
                clf_inner.fit(X_tr_p[in_tr], y_tr_arr[in_tr])
                oof_proba[in_val] = clf_inner.predict_proba(X_tr_p[in_val])

            hard_mask = oof_proba.max(axis=1) < threshold
            n_hard = hard_mask.sum()
            n_uncertain_val = (max_proba < threshold).sum()

            if n_hard < 50:
                # Too few hard cases — skip Stage 2 for this fold/threshold
                fold_combined_rows.append({'acc': acc_s1, 'mcc': mcc_s1,
                                            'r1': r1_s1, 'r3': r3_s1,
                                            'n_uncertain': 0})
                continue

            # Train Stage 2 LGBM on hard cases only
            stage2 = LGBMClassifier(
                n_estimators=300, learning_rate=0.05,
                num_leaves=63, random_state=random_state,
                n_jobs=-1, verbose=-1)
            stage2.fit(X_tr_p[hard_mask], y_tr_arr[hard_mask])

            # Override uncertain val predictions with Stage 2
            uncertain_mask = max_proba < threshold
            y_pred_combined = y_pred_s1.copy()
            if uncertain_mask.sum() > 0:
                y_pred_combined[uncertain_mask] = stage2.predict(
                    X_val_p[uncertain_mask])

            acc_c = accuracy_score(y_val_arr, y_pred_combined)
            mcc_c = matthews_corrcoef(y_val_arr, y_pred_combined)
            rep_c = classification_report(y_val_arr, y_pred_combined,
                                           labels=[1, 2, 3],
                                           output_dict=True, zero_division=0)
            r1_c = rep_c.get('1', rep_c.get(1, {})).get('recall', 0.0)
            r3_c = rep_c.get('3', rep_c.get(3, {})).get('recall', 0.0)
            fold_combined_rows.append({'acc': acc_c, 'mcc': mcc_c,
                                        'r1': r1_c, 'r3': r3_c,
                                        'n_uncertain': int(n_uncertain_val)})

        # Average over folds
        s1_df = pd.DataFrame(fold_s1_rows)
        s1_mean = s1_df.mean()
        c_df  = pd.DataFrame(fold_combined_rows)
        c_mean = c_df.mean()

        print(f"  Stage 1 only (RF):       "
              f"Acc={s1_mean['acc']:.4f}  MCC={s1_mean['mcc']:.4f}  "
              f"G1-Recall={s1_mean['r1']:.3f}  G3-Recall={s1_mean['r3']:.3f}")
        print(f"  Two-stage (RF+LGBM):     "
              f"Acc={c_mean['acc']:.4f}  MCC={c_mean['mcc']:.4f}  "
              f"G1-Recall={c_mean['r1']:.3f}  G3-Recall={c_mean['r3']:.3f}")
        print(f"  Delta:                   "
              f"Acc={c_mean['acc']-s1_mean['acc']:+.4f}  "
              f"MCC={c_mean['mcc']-s1_mean['mcc']:+.4f}  "
              f"G1={c_mean['r1']-s1_mean['r1']:+.3f}  "
              f"G3={c_mean['r3']-s1_mean['r3']:+.3f}")
        print(f"  Avg uncertain/fold:      {c_mean['n_uncertain']:.0f} samples")

        all_results.append({
            'threshold': threshold,
            'method': 'Stage 1 only (RF)',
            'accuracy': round(float(s1_mean['acc']), 4),
            'mcc': round(float(s1_mean['mcc']), 4),
            'g1_recall': round(float(s1_mean['r1']), 4),
            'g3_recall': round(float(s1_mean['r3']), 4),
            'uncertain_samples': 0,
            'eval_mode': f'{n_cv_splits}-Fold CV'
        })
        all_results.append({
            'threshold': threshold,
            'method': 'Two-stage (RF+LGBM)',
            'accuracy': round(float(c_mean['acc']), 4),
            'mcc': round(float(c_mean['mcc']), 4),
            'g1_recall': round(float(c_mean['r1']), 4),
            'g3_recall': round(float(c_mean['r3']), 4),
            'uncertain_samples': round(float(c_mean['n_uncertain'])),
            'eval_mode': f'{n_cv_splits}-Fold CV'
        })

    # ── Summary ──────────────────────────────────────────────────────────────
    results_df = pd.DataFrame(all_results)

    print("\n" + "="*80)
    print("TWO-STAGE CLASSIFIER — SUMMARY (5-Fold CV)")
    print("Sorted by Accuracy (Two-stage rows only)")
    print("="*80)
    two_stage_df = results_df[results_df['method'] == 'Two-stage (RF+LGBM)'] \
        .sort_values('accuracy', ascending=False)
    baseline_acc = results_df[results_df['method'] == 'Stage 1 only (RF)'] \
        ['accuracy'].iloc[0]

    print(f"\n  {'Threshold':>10} {'Acc':>8} {'Delta Acc':>10} "
          f"{'MCC':>8} {'G1-Recall':>10} {'G3-Recall':>10} "
          f"{'Uncertain':>10}")
    print("  " + "-"*72)
    for _, row in two_stage_df.iterrows():
        delta = row['accuracy'] - baseline_acc
        print(f"  {row['threshold']:>10.2f} {row['accuracy']:>8.4f} "
              f"{delta:>+10.4f} {row['mcc']:>8.4f} "
              f"{row['g1_recall']:>10.3f} {row['g3_recall']:>10.3f} "
              f"{row['uncertain_samples']:>10}")

    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / 'two_stage_classifier.csv', index=False)
    print(f"\n  Saved: outputs/two_stage_classifier.csv")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Week 06: tune_random_search()
# Random Search for RF (Stage 1) and LGBM (Stage 2)
# Uses RandomizedSearchCV with 5-Fold CV
# Primary metric: Accuracy
# ─────────────────────────────────────────────────────────────────────────────

def tune_random_search(X_train_full, y_train_full,
                       pipeline_kwargs=None,
                       n_iter=20,
                       sample_size=None,
                       random_state=42):
    """
    Hyperparameter tuning via Random Search.

    Tunes RF (Stage 1) and LGBM (Stage 2) independently.
    RF is tuned on all 260k samples with 5-Fold CV.
    LGBM is tuned on the hard cases identified by RF OOF probabilities.

    Parameters
    ----------
    n_iter : int
        Number of random parameter combinations to try per model.
    """
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    if sample_size is not None:
        idx = np.random.choice(len(X_train_full),
                               size=min(sample_size, len(X_train_full)),
                               replace=False)
        X_train_full = X_train_full.iloc[idx]
        y_train_full = y_train_full.iloc[idx]
        print(f"  (Subsampled to {len(X_train_full):,} samples)")

    print("\n" + "="*60)
    print("WEEK 06 — RANDOM SEARCH")
    print(f"n_iter={n_iter} per model | 5-Fold CV | metric=Accuracy")
    print("="*60)

    # ── Preprocess once on full data ──────────────────────────────────────
    print("\n[1/3] Preprocessing...")
    preproc = _build_preproc_steps(pipeline_kwargs)
    t0 = time.time()
    X_proc = np.nan_to_num(
        preproc.fit_transform(X_train_full, y_train_full).astype(float))
    y_arr = y_train_full.values
    print(f"  Done in {time.time()-t0:.1f}s | Features: {X_proc.shape[1]}")

    # ── Stage 1: Tune RF on all 260k ─────────────────────────────────────
    print("\n[2/3] Tuning RF (Stage 1)...")
    rf_param_dist = {
        'n_estimators':     [100, 200, 300, 500],
        'max_depth':        randint(8, 26),
        'min_samples_leaf': randint(1, 21),
        'max_features':     ['sqrt', 'log2', 0.3, 0.5],
    }

    rf_base = RandomForestClassifier(random_state=random_state, n_jobs=-1)
    rf_search = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=rf_param_dist,
        n_iter=n_iter,
        cv=5,
        scoring='accuracy',
        random_state=random_state,
        n_jobs=1,   # RF itself uses n_jobs=-1 — no nested parallelism
        verbose=1,
        refit=True,
    )
    t0 = time.time()
    rf_search.fit(X_proc, y_arr)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Best RF params:    {rf_search.best_params_}")
    print(f"  Best RF Acc (CV):  {rf_search.best_score_:.4f}")

    # ── Find hard cases via OOF with best RF ─────────────────────────────
    print("\n  Identifying hard cases with best RF (OOF)...")
    UNCERTAINTY_THRESHOLD = 0.65
    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_proba = np.zeros((len(X_proc), 3))
    for tr_idx, val_idx in kf.split(X_proc):
        clf_oof = RandomForestClassifier(
            **rf_search.best_params_,
            random_state=random_state, n_jobs=-1)
        clf_oof.fit(X_proc[tr_idx], y_arr[tr_idx])
        oof_proba[val_idx] = clf_oof.predict_proba(X_proc[val_idx])

    hard_mask = oof_proba.max(axis=1) < UNCERTAINTY_THRESHOLD
    n_hard = hard_mask.sum()
    print(f"  Hard cases: {n_hard:,} ({100*n_hard/len(y_arr):.1f}%)")

    # ── Stage 2: Tune LGBM on hard cases ─────────────────────────────────
    print("\n[3/3] Tuning LGBM (Stage 2) on hard cases...")
    lgbm_param_dist = {
        'n_estimators':      [200, 300, 500],
        'learning_rate':     [0.01, 0.05, 0.1],
        'num_leaves':        randint(31, 128),
        'min_child_samples': randint(10, 51),
    }

    lgbm_base = LGBMClassifier(
        random_state=random_state, n_jobs=-1, verbose=-1)
    lgbm_search = RandomizedSearchCV(
        estimator=lgbm_base,
        param_distributions=lgbm_param_dist,
        n_iter=n_iter,
        cv=3,       # 3-Fold instead of 5 — hard cases subset is large enough
        scoring='accuracy',
        random_state=random_state,
        n_jobs=1,   # LGBM itself uses n_jobs=-1 — no nested parallelism
        verbose=1,
        refit=True,
    )
    t0 = time.time()
    lgbm_search.fit(X_proc[hard_mask], y_arr[hard_mask])
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Best LGBM params:   {lgbm_search.best_params_}")
    print(f"  Best LGBM Acc (CV): {lgbm_search.best_score_:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    results = {
        'rf_best_params':   rf_search.best_params_,
        'rf_best_acc':      round(rf_search.best_score_, 4),
        'lgbm_best_params': lgbm_search.best_params_,
        'lgbm_best_acc':    round(lgbm_search.best_score_, 4),
    }

    print("\n" + "="*60)
    print("RANDOM SEARCH — SUMMARY")
    print("="*60)
    print(f"  RF   best Acc={results['rf_best_acc']}  "
          f"params={results['rf_best_params']}")
    print(f"  LGBM best Acc={results['lgbm_best_acc']}  "
          f"params={results['lgbm_best_params']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Week 06: tune_optuna()
# Bayesian Optimization via Optuna for RF (Stage 1) and LGBM (Stage 2)
# ─────────────────────────────────────────────────────────────────────────────

def tune_optuna(X_train_full, y_train_full,
                pipeline_kwargs=None,
                n_trials=30,
                sample_size=None,
                random_state=42):
    """
    Hyperparameter tuning via Optuna (Bayesian Optimization / TPE).

    Same structure as tune_random_search() but uses Optuna instead of
    RandomizedSearchCV. Optuna learns from previous trials and focuses
    search on promising parameter regions.

    Parameters
    ----------
    n_trials : int
        Number of Optuna trials per model (default: 30).
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        return {}

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    if sample_size is not None:
        idx = np.random.choice(len(X_train_full),
                               size=min(sample_size, len(X_train_full)),
                               replace=False)
        X_train_full = X_train_full.iloc[idx]
        y_train_full = y_train_full.iloc[idx]
        print(f"  (Subsampled to {len(X_train_full):,} samples)")

    print("\n" + "="*60)
    print("WEEK 06 — OPTUNA HYPERPARAMETER TUNING")
    print(f"n_trials={n_trials} per model | 5-Fold CV | metric=Accuracy")
    print("="*60)

    # ── Preprocess once ───────────────────────────────────────────────────
    print("\n[1/3] Preprocessing...")
    preproc = _build_preproc_steps(pipeline_kwargs)
    t0 = time.time()
    X_proc = np.nan_to_num(
        preproc.fit_transform(X_train_full, y_train_full).astype(float))
    y_arr = y_train_full.values
    print(f"  Done in {time.time()-t0:.1f}s | Features: {X_proc.shape[1]}")

    kf_tune = KFold(n_splits=5, shuffle=True, random_state=random_state)

    # ── Stage 1: Tune RF ──────────────────────────────────────────────────
    print("\n[2/3] Tuning RF (Stage 1) with Optuna...")

    def rf_objective(trial):
        params = {
            'n_estimators':     trial.suggest_categorical(
                                    'n_estimators', [100, 200, 300, 500]),
            'max_depth':        trial.suggest_int('max_depth', 8, 25),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features':     trial.suggest_categorical(
                                    'max_features', ['sqrt', 'log2', 0.3, 0.5]),
        }
        clf = RandomForestClassifier(
            **params, random_state=random_state, n_jobs=-1)
        fold_accs = []
        for tr_idx, val_idx in kf_tune.split(X_proc):
            clf_fold = clone(clf)
            clf_fold.fit(X_proc[tr_idx], y_arr[tr_idx])
            y_pred = clf_fold.predict(X_proc[val_idx])
            fold_accs.append(accuracy_score(y_arr[val_idx], y_pred))
        return np.mean(fold_accs)

    t0 = time.time()
    rf_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state))
    rf_study.optimize(rf_objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Best RF params:   {rf_study.best_params}")
    print(f"  Best RF Acc (CV): {rf_study.best_value:.4f}")

    # ── Find hard cases with best RF ──────────────────────────────────────
    print("\n  Identifying hard cases with best RF (OOF)...")
    UNCERTAINTY_THRESHOLD = 0.65
    kf_oof = KFold(n_splits=5, shuffle=True, random_state=random_state)
    oof_proba = np.zeros((len(X_proc), 3))
    for tr_idx, val_idx in kf_oof.split(X_proc):
        clf_oof = RandomForestClassifier(
            **rf_study.best_params,
            random_state=random_state, n_jobs=-1)
        clf_oof.fit(X_proc[tr_idx], y_arr[tr_idx])
        oof_proba[val_idx] = clf_oof.predict_proba(X_proc[val_idx])

    hard_mask = oof_proba.max(axis=1) < UNCERTAINTY_THRESHOLD
    n_hard = hard_mask.sum()
    print(f"  Hard cases: {n_hard:,} ({100*n_hard/len(y_arr):.1f}%)")

    # ── Stage 2: Tune LGBM on hard cases ─────────────────────────────────
    print("\n[3/3] Tuning LGBM (Stage 2) with Optuna...")
    X_hard = X_proc[hard_mask]
    y_hard = y_arr[hard_mask]
    kf_hard = KFold(n_splits=3, shuffle=True, random_state=random_state)  # 3-Fold — faster on hard cases subset

    def lgbm_objective(trial):
        params = {
            'n_estimators':      trial.suggest_categorical(
                                     'n_estimators', [200, 300, 500]),
            'learning_rate':     trial.suggest_categorical(
                                     'learning_rate', [0.01, 0.05, 0.1]),
            'num_leaves':        trial.suggest_int('num_leaves', 31, 127),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        }
        clf = LGBMClassifier(
            **params, random_state=random_state, n_jobs=-1, verbose=-1)
        fold_accs = []
        for tr_idx, val_idx in kf_hard.split(X_hard):
            clf_fold = clone(clf)
            clf_fold.fit(X_hard[tr_idx], y_hard[tr_idx])
            y_pred = clf_fold.predict(X_hard[val_idx])
            fold_accs.append(accuracy_score(y_hard[val_idx], y_pred))
        return np.mean(fold_accs)

    t0 = time.time()
    lgbm_study = optuna.create_study(
        direction='maximize',
        sampler=optuna.samplers.TPESampler(seed=random_state))
    lgbm_study.optimize(
        lgbm_objective, n_trials=n_trials, show_progress_bar=True)
    print(f"  Done in {time.time()-t0:.1f}s")
    print(f"  Best LGBM params:   {lgbm_study.best_params}")
    print(f"  Best LGBM Acc (CV): {lgbm_study.best_value:.4f}")

    # ── Summary ──────────────────────────────────────────────────────────
    results = {
        'rf_best_params':   rf_study.best_params,
        'rf_best_acc':      round(rf_study.best_value, 4),
        'lgbm_best_params': lgbm_study.best_params,
        'lgbm_best_acc':    round(lgbm_study.best_value, 4),
    }

    print("\n" + "="*60)
    print("OPTUNA — SUMMARY")
    print("="*60)
    print(f"  RF   best Acc={results['rf_best_acc']}  "
          f"params={results['rf_best_params']}")
    print(f"  LGBM best Acc={results['lgbm_best_acc']}  "
          f"params={results['lgbm_best_params']}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Week 06: tune_nested_cv()
# Nested Cross-Validation for fair performance estimation
# Outer: 3 folds | Inner: 3 folds + Optuna (20 trials)
# ─────────────────────────────────────────────────────────────────────────────

def tune_nested_cv(X_train_full, y_train_full,
                   pipeline_kwargs=None,
                   n_outer=3,
                   n_inner=3,
                   n_trials=20,
                   sample_size=None,
                   random_state=42):
    """
    Nested Cross-Validation for fair hyperparameter tuning evaluation.

    Outer loop: n_outer folds — each outer test fold is never seen during
                                parameter selection (fair evaluation).
    Inner loop: n_inner folds + Optuna — finds best RF params on inner folds
                                         for each outer fold independently.

    Reports:
      - Best params found per outer fold (may differ between folds)
      - Accuracy per outer fold on unseen outer test data
      - Mean nested CV accuracy = fair performance estimate

    Compare mean nested CV accuracy against normal CV accuracy in the README
    to quantify the optimistic bias of standard hyperparameter tuning.

    Parameters
    ----------
    n_outer  : int   outer CV folds (default: 3)
    n_inner  : int   inner CV folds for Optuna (default: 3)
    n_trials : int   Optuna trials per inner search (default: 20)
    """
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        print("ERROR: optuna not installed. Run: pip install optuna")
        return pd.DataFrame()

    if pipeline_kwargs is None:
        pipeline_kwargs = {}

    if sample_size is not None:
        idx = np.random.choice(len(X_train_full),
                               size=min(sample_size, len(X_train_full)),
                               replace=False)
        X_train_full = X_train_full.iloc[idx]
        y_train_full = y_train_full.iloc[idx]
        print(f"  (Subsampled to {len(X_train_full):,} samples)")

    print("\n" + "="*60)
    print("WEEK 06 — NESTED CROSS-VALIDATION")
    print(f"Outer={n_outer} folds | Inner={n_inner} folds | "
          f"Optuna trials={n_trials}")
    print("Tuning RF (Stage 1) | LGBM (Stage 2) uses default params")
    print("="*60)

    # ── Preprocess once on full data ──────────────────────────────────────
    print("\n[1/2] Preprocessing...")
    preproc = _build_preproc_steps(pipeline_kwargs)
    t0 = time.time()
    X_proc = np.nan_to_num(
        preproc.fit_transform(X_train_full, y_train_full).astype(float))
    y_arr = y_train_full.values
    print(f"  Done in {time.time()-t0:.1f}s | Features: {X_proc.shape[1]}")

    print("\n[2/2] Running Nested CV...")
    outer_kf = KFold(
        n_splits=n_outer, shuffle=True, random_state=random_state)

    outer_results = []
    UNCERTAINTY_THRESHOLD = 0.65

    for outer_fold, (outer_tr_idx, outer_test_idx) in enumerate(
            outer_kf.split(X_proc), 1):

        print(f"\n{'─'*55}")
        print(f"  Outer Fold {outer_fold}/{n_outer}  "
              f"(train={len(outer_tr_idx):,}  test={len(outer_test_idx):,})")
        print(f"{'─'*55}")

        X_outer_tr   = X_proc[outer_tr_idx]
        y_outer_tr   = y_arr[outer_tr_idx]
        X_outer_test = X_proc[outer_test_idx]
        y_outer_test = y_arr[outer_test_idx]

        # ── Inner loop: find best RF params via Optuna on outer_tr ───────
        inner_kf = KFold(
            n_splits=n_inner, shuffle=True, random_state=random_state)

        def rf_objective_inner(trial):
            params = {
                'n_estimators':     trial.suggest_categorical(
                                        'n_estimators', [100, 200, 300, 500]),
                'max_depth':        trial.suggest_int('max_depth', 8, 25),
                'min_samples_leaf': trial.suggest_int(
                                        'min_samples_leaf', 1, 20),
                'max_features':     trial.suggest_categorical(
                                        'max_features',
                                        ['sqrt', 'log2', 0.3, 0.5]),
            }
            clf = RandomForestClassifier(
                **params, random_state=random_state, n_jobs=-1)
            fold_accs = []
            for in_tr, in_val in inner_kf.split(X_outer_tr):
                clf_fold = clone(clf)
                clf_fold.fit(X_outer_tr[in_tr], y_outer_tr[in_tr])
                y_pred = clf_fold.predict(X_outer_tr[in_val])
                fold_accs.append(
                    accuracy_score(y_outer_tr[in_val], y_pred))
            return np.mean(fold_accs)

        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state))
        study.optimize(
            rf_objective_inner, n_trials=n_trials, show_progress_bar=False)

        best_params = study.best_params
        print(f"  Inner best params: {best_params}")
        print(f"  Inner best Acc:    {study.best_value:.4f}")

        # ── Train with best params on full outer_tr ───────────────────────
        rf_final = RandomForestClassifier(
            **best_params, random_state=random_state, n_jobs=-1)
        rf_final.fit(X_outer_tr, y_outer_tr)

        # Find hard cases via OOF on outer_tr (no leakage)
        kf_oof = KFold(
            n_splits=3, shuffle=True, random_state=random_state)
        oof_proba = np.zeros((len(X_outer_tr), 3))
        for in_tr, in_val in kf_oof.split(X_outer_tr):
            clf_oof = RandomForestClassifier(
                **best_params, random_state=random_state, n_jobs=-1)
            clf_oof.fit(X_outer_tr[in_tr], y_outer_tr[in_tr])
            oof_proba[in_val] = clf_oof.predict_proba(X_outer_tr[in_val])

        hard_mask = oof_proba.max(axis=1) < UNCERTAINTY_THRESHOLD
        n_hard = hard_mask.sum()

        # Stage 2: LGBM with default params on hard cases
        # (tuning LGBM here would double runtime without meaningful benefit
        #  for the nested CV evaluation purpose)
        lgbm_stage2 = LGBMClassifier(
            n_estimators=300, learning_rate=0.05, num_leaves=63,
            random_state=random_state, n_jobs=-1, verbose=-1)
        if n_hard >= 50:
            lgbm_stage2.fit(
                X_outer_tr[hard_mask], y_outer_tr[hard_mask])

        # ── Evaluate on outer test set (never seen during tuning) ─────────
        proba_test = rf_final.predict_proba(X_outer_test)
        y_pred = np.argmax(proba_test, axis=1) + 1
        max_proba = proba_test.max(axis=1)

        if n_hard >= 50:
            uncertain_mask = max_proba < UNCERTAINTY_THRESHOLD
            if uncertain_mask.sum() > 0:
                y_pred[uncertain_mask] = lgbm_stage2.predict(
                    X_outer_test[uncertain_mask])

        acc = accuracy_score(y_outer_test, y_pred)
        mcc = matthews_corrcoef(y_outer_test, y_pred)
        print(f"  Outer test Acc={acc:.4f}  MCC={mcc:.4f}  "
              f"hard cases={n_hard:,}")

        outer_results.append({
            'outer_fold':    outer_fold,
            'best_params':   str(best_params),
            'inner_acc':     round(study.best_value, 4),
            'outer_acc':     round(acc, 4),
            'outer_mcc':     round(mcc, 4),
            'n_hard':        n_hard,
        })

    # ── Summary ──────────────────────────────────────────────────────────
    df = pd.DataFrame(outer_results)
    mean_acc = df['outer_acc'].mean()
    mean_mcc = df['outer_mcc'].mean()
    std_acc  = df['outer_acc'].std()

    print("\n" + "="*60)
    print("NESTED CV — SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print(f"\n  Mean Nested CV Acc = {mean_acc:.4f} +/- {std_acc:.4f}")
    print(f"  Mean Nested CV MCC = {mean_mcc:.4f}")
    print(f"\n  → Use this value in the README to compare against")
    print(f"    normal CV Acc (optimistic bias = normal_CV - nested_CV)")

    return df


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
                        help='Comma-separated list of models to compare, e.g. --models "Random Forest,Voting,Stacking"'
                             ' | Available: Decision Tree, Random Forest, LightGBM, XGBoost, CatBoost,'
                             ' k-Nearest Neighbors, Logistic Regression, Naive Bayes, SVM (Linear), Voting, Stacking')
    parser.add_argument('--voting-mode', type=str, default='soft', choices=['soft', 'hard'],
                        help='VotingClassifier mode: soft averages predicted probabilities; hard uses majority vote')
    parser.add_argument('--voting-weights', type=str, default=None,
                        help='Optional voting weights in order rf,lgbm,catboost, e.g. "1,1,2"')
    parser.add_argument('--no-voting', action='store_true',
                        help='Exclude VotingClassifier from model comparison and analysis')
    parser.add_argument('--no-stacking', action='store_true',
                        help='Exclude StackingClassifier from model comparison and analysis')
    parser.add_argument('--stacking-cv', type=int, default=5,
                        help='Internal CV folds used by StackingClassifier')
    parser.add_argument('--stacking-passthrough', action='store_true',
                        help='Let the stacking final estimator also see original preprocessed features')
    parser.add_argument('--analyze', action='store_true',
                        help='Deep error analysis including RF, LGBM, XGB, CatBoost, Voting, and Stacking')
    # ── Week 05: Sampling ─────────────────────────────────────────────────
    parser.add_argument('--sampling', action='store_true',
                        help='Week 05: Compare all sampling strategies on RF, LGBM, XGB, CatBoost')
    parser.add_argument('--sampling-no-cv', action='store_true',
                        help='Sampling comparison with single 80/20 split instead of CV (faster)')
    parser.add_argument('--strategies', type=str, default=None,
                        help='Comma-separated strategies to run, e.g. --strategies "Baseline,SMOTE,class_weight"'
                             ' | Available: Baseline, class_weight, SMOTE, RandomUnderSampler,'
                             ' SMOTE+Under, EasyEnsemble, ClusterCentroids, ROSE')
    # ── Week 05 extra experiments ─────────────────────────────────────────
    parser.add_argument('--threshold-tuning', action='store_true',
                        help='Week 05 Exp 1: Optimise Grade1/2 and Grade2/3 thresholds on RF and LGBM')
    parser.add_argument('--boundary-features', action='store_true',
                        help='Week 05 Exp 2: Test new interaction features targeting Grade2/3 boundary')
    parser.add_argument('--two-stage', action='store_true',
                        help='Week 05 Exp 3: Two-stage classifier — RF on all, LGBM on hard cases')
    parser.add_argument('--uncertainty-threshold', type=float, default=0.50,
                        help='Max-probability threshold below which Stage 1 considers a sample uncertain (default: 0.50)')
    # ── Week 06: Hyperparameter Tuning ────────────────────────────────────
    parser.add_argument('--tune-random', action='store_true',
                        help='Week 06: Random Search for RF and LGBM hyperparameters (RandomizedSearchCV)')
    parser.add_argument('--tune-optuna', action='store_true',
                        help='Week 06: Bayesian Optimization via Optuna (TPE sampler)')
    parser.add_argument('--tune-nested-cv', action='store_true',
                        help='Week 06: Nested CV for fair performance estimation (compare with normal CV in README)')
    parser.add_argument('--tune-trials', type=int, default=30,
                        help='Number of trials for Optuna or Nested CV inner search (default: 30)')
    parser.add_argument('--tune-n-iter', type=int, default=20,
                        help='Number of iterations for Random Search (default: 20)')
    args = parser.parse_args()

    voting_weights = parse_voting_weights(args.voting_weights)
    include_voting = not args.no_voting
    include_stacking = not args.no_stacking

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

    # ── --sampling: Week 05 Sampling-Vergleich ───────────────────────────
    if args.sampling:
        use_cv = not args.sampling_no_cv
        mode_str = "single 80/20 split" if args.sampling_no_cv else "5-Fold CV"
        selected_strategies = (
            [s.strip() for s in args.strategies.split(',')]
            if args.strategies else None
        )
        print(f">>> Week 05 — Sampling Strategy Comparison ({mode_str})")
        print(f"    Models: RF, LGBM, XGB, CatBoost")
        if selected_strategies:
            print(f"    Strategies (filtered): {selected_strategies}")
        else:
            print(f"    Strategies: all 8")
        compare_sampling_strategies(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            sample_size=args.sample,
            use_cv=use_cv,
            n_cv_splits=5,
            random_state=42,
            selected_strategies=selected_strategies,
        )
        return

    # ── --threshold-tuning ────────────────────────────────────────────────
    if args.threshold_tuning:
        print(">>> Week 05 Exp 1 — Threshold Tuning (RF + LGBM)")
        threshold_tuning(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            random_state=42,
        )
        return

    # ── --boundary-features ───────────────────────────────────────────────
    if args.boundary_features:
        print(">>> Week 05 Exp 2 — Boundary Feature Experiment (RF + LGBM)")
        boundary_feature_experiment(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            random_state=42,
        )
        return

    # ── --two-stage ───────────────────────────────────────────────────────
    if args.two_stage:
        print(f">>> Week 05 Exp 3 — Two-Stage Classifier "
              f"(uncertainty threshold={args.uncertainty_threshold})")
        two_stage_classifier(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            uncertainty_threshold=args.uncertainty_threshold,
            random_state=42,
        )
        return

    # ── --tune-random: Week 06 Random Search ─────────────────────────────
    if args.tune_random:
        print(f">>> Week 06 — Random Search (n_iter={args.tune_n_iter})")
        tune_random_search(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            n_iter=args.tune_n_iter,
            sample_size=args.sample,
            random_state=42,
        )
        return

    # ── --tune-optuna: Week 06 Optuna ─────────────────────────────────────
    if args.tune_optuna:
        print(f">>> Week 06 — Optuna Tuning (n_trials={args.tune_trials})")
        tune_optuna(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            n_trials=args.tune_trials,
            sample_size=args.sample,
            random_state=42,
        )
        return

    # ── --tune-nested-cv: Week 06 Nested CV ───────────────────────────────
    if args.tune_nested_cv:
        print(f">>> Week 06 — Nested CV (n_trials={args.tune_trials})")
        tune_nested_cv(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            n_trials=args.tune_trials,
            sample_size=args.sample,
            random_state=42,
        )
        return

    # ── --analyze: deep error analysis ───────────────────────────────────
    if args.analyze:
        print(">>> Deep Error Analysis - RF, LGBM, XGB, CatBoost, Voting, and Stacking")
        analyze_errors(
            X_train_full, y_train_full,
            pipeline_kwargs=pipeline_kwargs,
            sample_size=getattr(args, 'sample', None),
            voting_mode=args.voting_mode,
            voting_weights=voting_weights,
            include_voting=include_voting,
            include_stacking=include_stacking,
            stacking_cv=args.stacking_cv,
            stacking_passthrough=args.stacking_passthrough,
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
                selected_models=selected_models,
                voting_mode=args.voting_mode,
                voting_weights=voting_weights,
                include_voting=include_voting,
                include_stacking=include_stacking,
                stacking_cv=args.stacking_cv,
                stacking_passthrough=args.stacking_passthrough,
            )
        else:
            print(">>> Model Comparison — 5-Fold CV on full training data (260k)")
            comparison_results = compare_models(
                X_train_full, y_train_full,
                pipeline_kwargs=pipeline_kwargs,
                n_cv_splits=5,
                sample_size=args.sample,
                selected_models=selected_models,
                voting_mode=args.voting_mode,
                voting_weights=voting_weights,
                include_voting=include_voting,
                include_stacking=include_stacking,
                stacking_cv=args.stacking_cv,
                stacking_passthrough=args.stacking_passthrough,
            )
        out_path = Path('outputs/model_comparison.csv')
        out_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_results.to_csv(out_path, index=True)
        print(f"Results saved: {out_path}")
        return

    # ── normal run: train on full data → submission file ─────────────────
    # Default model: Two-Stage Classifier (RF + LGBM, threshold=0.65)
    # Week 05 baseline:  Acc=0.7423, MCC=0.5211 (5-Fold CV)
    # Week 06 tuned params (Optuna, 30 trials, full 260k):
    #   RF:   n_estimators=500, max_depth=16, min_samples_leaf=6, max_features=0.5
    #   LGBM: n_estimators=500, learning_rate=0.05, num_leaves=76, min_child_samples=21
    print(">>> Training final Two-Stage pipeline on full training data (260k)")
    print("    Stage 1: RF   (Week 06 tuned: n_estimators=500, max_depth=16,")
    print("                   min_samples_leaf=6, max_features=0.5)")
    print("    Stage 2: LGBM (Week 06 tuned: n_estimators=500, learning_rate=0.05,")
    print("                   num_leaves=76, min_child_samples=21)")

    # ── Preprocess full training data ─────────────────────────────────────
    preproc_final = _build_preproc_steps(pipeline_kwargs)
    X_train_proc = np.nan_to_num(
        preproc_final.fit_transform(X_train_full, y_train_full).astype(float))
    X_test_proc  = np.nan_to_num(
        preproc_final.transform(X_test).astype(float))
    y_train_arr  = y_train_full.values

    # ── Stage 1: RF on all training data ──────────────────────────────────
    UNCERTAINTY_THRESHOLD = 0.65
    print(f"\n[Stage 1] Training RF on {len(X_train_proc):,} samples...")
    t0 = time.time()
    stage1_final = RandomForestClassifier(
        n_estimators=500,       # Week 06 tuned (Optuna)
        max_depth=16,           # Week 06 tuned (Optuna)
        min_samples_leaf=6,     # Week 06 tuned (Optuna)
        max_features=0.5,       # Week 06 tuned (Optuna)
        random_state=42, n_jobs=-1)
    stage1_final.fit(X_train_proc, y_train_arr)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Find hard cases via OOF on training data (no leakage) ─────────────
    print(f"\n[Stage 2] Identifying hard cases (OOF, threshold={UNCERTAINTY_THRESHOLD})...")
    kf_final = KFold(n_splits=5, shuffle=True, random_state=42)
    oof_proba_final = np.zeros((len(X_train_proc), 3))
    for tr_idx, val_idx in kf_final.split(X_train_proc):
        clf_oof = RandomForestClassifier(
            n_estimators=500, max_depth=16,
            min_samples_leaf=6, max_features=0.5,
            random_state=42, n_jobs=-1)
        clf_oof.fit(X_train_proc[tr_idx], y_train_arr[tr_idx])
        oof_proba_final[val_idx] = clf_oof.predict_proba(X_train_proc[val_idx])

    hard_mask_final = oof_proba_final.max(axis=1) < UNCERTAINTY_THRESHOLD
    n_hard_final = hard_mask_final.sum()
    print(f"  Hard cases: {n_hard_final:,} / {len(y_train_arr):,} "
          f"({100*n_hard_final/len(y_train_arr):.1f}%)")

    # Save hard cases to outputs/
    hard_cases_df = X_train_full[hard_mask_final].copy()
    hard_cases_df['damage_grade'] = y_train_arr[hard_mask_final]
    hard_cases_df['oof_max_proba'] = oof_proba_final[hard_mask_final].max(axis=1)
    out_dir = Path('outputs')
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_cases_df.to_csv(out_dir / 'hard_cases.csv', index=True)
    print(f"  Hard cases saved: outputs/hard_cases.csv")

    # ── Stage 2: LGBM on hard cases only ──────────────────────────────────
    print(f"\n  Training Stage 2 LGBM on {n_hard_final:,} hard cases...")
    t0 = time.time()
    stage2_final = LGBMClassifier(
        n_estimators=500,           # Week 06 tuned (Optuna)
        learning_rate=0.05,         # Week 06 tuned (Optuna)
        num_leaves=76,              # Week 06 tuned (Optuna)
        min_child_samples=21,       # Week 06 tuned (Optuna)
        random_state=42, n_jobs=-1, verbose=-1)
    stage2_final.fit(X_train_proc[hard_mask_final], y_train_arr[hard_mask_final])
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Predict on test set ────────────────────────────────────────────────
    print("\n[Predict] Applying Two-Stage to test set...")
    proba_test   = stage1_final.predict_proba(X_test_proc)
    predictions  = np.argmax(proba_test, axis=1) + 1
    max_proba_test = proba_test.max(axis=1)

    # Override uncertain test predictions with Stage 2
    uncertain_test_mask = max_proba_test < UNCERTAINTY_THRESHOLD
    n_uncertain_test = uncertain_test_mask.sum()
    if n_uncertain_test > 0:
        predictions[uncertain_test_mask] = stage2_final.predict(
            X_test_proc[uncertain_test_mask])
    print(f"  Uncertain test samples handled by Stage 2: "
          f"{n_uncertain_test:,} / {len(predictions):,} "
          f"({100*n_uncertain_test/len(predictions):.1f}%)")

    print("\n" + "="*60)
    print("PREDICTIONS")
    print("="*60)
    print(f"Total: {len(predictions):,}")
    for grade in [1, 2, 3]:
        count = (predictions == grade).sum()
        print(f"  Grade {grade}: {count:,}")
    print("="*60 + "\n")

    # ── Save submission — only Team2_prediction.csv ────────────────────────
    save_submission(predictions, test_ids)
    print("PIPELINE COMPLETE!\n")
    print(f"Output files:")
    print(f"  outputs/Team2_prediction.csv  — submission file")
    print(f"  outputs/hard_cases.csv        — hard cases from training set")


if __name__ == '__main__':
    main()