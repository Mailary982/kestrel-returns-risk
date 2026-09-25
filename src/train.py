from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"


# ============================================================
# CONFIG
# ============================================================

VALIDATION_FRACTION = 0.20
RANDOM_STATE = 42

MODEL_PATH = MODEL_DIR / "model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

BASELINE_ROC_AUC = 0.7492
BASELINE_PR_AUC = 0.3582


# ============================================================
# LOAD RAW DATA
# ============================================================

def load_data():
    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["order_placed_at"],
    )

    customers = pd.read_csv(
        CUSTOMERS_PATH,
        parse_dates=["signup_date"],
    )

    products = pd.read_csv(
        PRODUCTS_PATH,
        parse_dates=["launch_date"],
    )

    return train, customers, products


# ============================================================
# NORMALIZE OCTOBER PAYMENT ANOMALY
# ============================================================

def normalize_payment_values(train, products):
    train = train.copy()

    product_prices = (
        products[
            [
                "sku",
                "list_price_inr",
            ]
        ]
        .drop_duplicates("sku")
    )

    train = train.merge(
        product_prices,
        on="sku",
        how="left",
        suffixes=("", "_product"),
    )

    expected_value = (
        train["list_price_inr"]
        * train["qty"]
        * (1 - train["discount_pct"] / 100)
    )

    ratio = (
        train["order_value_inr"]
        / expected_value.replace(0, np.nan)
    )

    october_2025 = (
        (train["order_placed_at"].dt.year == 2025)
        & (train["order_placed_at"].dt.month == 10)
    )

    anomaly = (
        october_2025
        & ratio.between(
            90,
            110,
            inclusive="both",
        )
    )

    correction_count = int(anomaly.sum())

    train.loc[
        anomaly,
        "order_value_inr",
    ] = (
        train.loc[
            anomaly,
            "order_value_inr",
        ]
        / 100
    )

    print(
        f"October 2025 payment anomalies corrected: "
        f"{correction_count:,}"
    )

    train = train.drop(
        columns=["list_price_inr"]
    )

    return train


# ============================================================
# DEDUPLICATE TRAINING DATA
# ============================================================

def deduplicate_train(train):
    before = len(train)

    train = (
        train
        .sort_values("order_placed_at")
        .drop_duplicates(
            subset=["order_id"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    removed = before - len(train)

    print(
        f"Duplicate training rows removed: "
        f"{removed:,}"
    )

    return train


# ============================================================
# CUSTOMER HISTORY FEATURES
# ============================================================

def add_customer_history(train):
    train = train.copy()

    train = (
        train
        .sort_values(
            [
                "customer_id",
                "order_placed_at",
            ]
        )
        .reset_index(drop=True)
    )

    grouped = train.groupby(
        "customer_id",
        sort=False,
    )

    train["customer_prior_orders_temporal"] = (
        grouped.cumcount()
    )

    train["customer_prior_returns_temporal"] = (
        grouped["returned"]
        .cumsum()
        - train["returned"]
    )

    train["customer_prior_returns_temporal"] = (
        train["customer_prior_returns_temporal"]
        .clip(lower=0)
    )

    train["customer_prior_return_rate_temporal"] = np.where(
        train["customer_prior_orders_temporal"] > 0,
        train["customer_prior_returns_temporal"]
        / train["customer_prior_orders_temporal"],
        0.0,
    )

    return train


# ============================================================
# CUSTOMER + PRODUCT ENRICHMENT
# ============================================================

def add_customer_product_features(
    train,
    customers,
    products,
):
    train = train.copy()

    train = train.merge(
        customers,
        on="customer_id",
        how="left",
    )

    train = train.merge(
        products,
        on="sku",
        how="left",
        suffixes=("", "_product"),
    )

    # --------------------------------------------------------
    # Customer features
    # --------------------------------------------------------

    train["customer_tenure_days"] = (
        train["order_placed_at"]
        - train["signup_date"]
    ).dt.days

    train["has_customer_history"] = (
        train["customer_prior_orders"] > 0
    ).astype(int)

    train["has_prior_return"] = (
        train["customer_prior_returns"] > 0
    ).astype(int)

    train["customer_prior_return_rate"] = np.where(
        train["customer_prior_orders"] > 0,
        train["customer_prior_returns"]
        / train["customer_prior_orders"],
        0.0,
    )

    # --------------------------------------------------------
    # Product features
    # --------------------------------------------------------

    train["product_age_days"] = (
        train["order_placed_at"]
        - train["launch_date"]
    ).dt.days

    # --------------------------------------------------------
    # Date features
    # --------------------------------------------------------

    train["order_year"] = (
        train["order_placed_at"].dt.year
    )

    train["order_month"] = (
        train["order_placed_at"].dt.month
    )

    train["order_day"] = (
        train["order_placed_at"].dt.day
    )

    train["order_day_of_week"] = (
        train["order_placed_at"].dt.dayofweek
    )

    train["order_week_of_year"] = (
        train["order_placed_at"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    train["order_day_of_month"] = (
        train["order_placed_at"].dt.day
    )

    train["order_hour"] = (
        train["order_placed_at"].dt.hour
    )

    train["is_weekend"] = (
        train["order_day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # Cyclical date features
    # --------------------------------------------------------

    train["month_sin"] = np.sin(
        2 * np.pi * train["order_month"] / 12
    )

    train["month_cos"] = np.cos(
        2 * np.pi * train["order_month"] / 12
    )

    train["day_of_week_sin"] = np.sin(
        2 * np.pi * train["order_day_of_week"] / 7
    )

    train["day_of_week_cos"] = np.cos(
        2 * np.pi * train["order_day_of_week"] / 7
    )

    # --------------------------------------------------------
    # Pincode
    # --------------------------------------------------------

    train["is_default_pincode"] = (
        train["delivery_pincode"] == 0
    ).astype(int)

    # --------------------------------------------------------
    # Delivery note metadata
    # --------------------------------------------------------

    note = (
        train["delivery_note"]
        .fillna("")
        .astype(str)
    )

    train["delivery_note_missing"] = (
        train["delivery_note"]
        .isna()
        .astype(int)
    )

    train["delivery_note_length"] = (
        note.str.len()
    )

    train["delivery_note_word_count"] = (
        note.str.split()
        .str.len()
    )

    # --------------------------------------------------------
    # Order value features
    # --------------------------------------------------------

    train["order_value_per_unit"] = (
        train["order_value_inr"]
        / train["qty"].replace(0, np.nan)
    )

    train["multi_quantity_order"] = (
        train["qty"] > 1
    ).astype(int)

    return train


# ============================================================
# FEATURE PREPARATION
# ============================================================

def prepare_features(df):
    df = df.copy()

    # --------------------------------------------------------
    # Canonical product price
    # --------------------------------------------------------

    if "list_price_inr" in df.columns:
        df["list_price_inr_canonical"] = (
            df["list_price_inr"]
        )

    elif "list_price_inr_product" in df.columns:
        df["list_price_inr_canonical"] = (
            df["list_price_inr_product"]
        )

    else:
        raise ValueError(
            "Canonical product price unavailable."
        )

    df["list_price_inr"] = (
        df["list_price_inr_canonical"]
    )

    # --------------------------------------------------------
    # Columns excluded from the model
    # --------------------------------------------------------

    columns_to_drop = [
        "returned",
        "order_id",
        "customer_id",
        "order_placed_at",
        "delivery_note",
        "pickup_scheduled_at",
        "last_service_event_type",
        "delivery_pincode",
        "discount_rate",
        "order_day_of_month",
        "customer_prior_orders_temporal",
        "list_price_inr_x",
        "list_price_inr_y",
        "list_price_inr_canonical",
    ]

    df = df.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in df.columns
        ]
    )

    return df


# ============================================================
# FEATURE SELECTION
# ============================================================

def get_feature_columns(df):
    target = "returned"

    feature_columns = [
        column
        for column in df.columns
        if column != target
    ]

    return feature_columns


# ============================================================
# PREPROCESSOR
# ============================================================

def build_preprocessor(
    categorical_columns,
    numerical_columns,
):
    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_columns,
            ),
            (
                "categorical",
                categorical_pipeline,
                categorical_columns,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


# ============================================================
# XGBOOST
# ============================================================

def build_model():
    return XGBClassifier(
        objective="binary:logistic",
        n_estimators=500,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=2.0,
        gamma=0.0,
        eval_metric="aucpr",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def create_threshold_analysis(
    y_true,
    probabilities,
):
    thresholds = [
        0.05,
        0.075,
        0.10,
        0.125,
        0.15,
        0.175,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
        0.60,
        0.70,
    ]

    results = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        flagged = int(predictions.sum())

        results.append(
            {
                "threshold": threshold,
                "orders_flagged": flagged,
                "flag_rate": flagged / len(y_true),
                "accuracy": accuracy_score(
                    y_true,
                    predictions,
                ),
                "precision": precision_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    y_true,
                    predictions,
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(results)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def save_feature_importance(
    model,
    preprocessor,
):
    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = (
        model.feature_importances_
    )

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    output_path = (
        REPORT_DIR / "feature_importance.csv"
    )

    importance_df.to_csv(
        output_path,
        index=False,
    )

    print("\nTop 20 features:")
    print(
        importance_df
        .head(20)
        .to_string(index=False)
    )


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("KESTREL HOME — MODEL VALIDATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train, customers, products = load_data()

    print(
        f"\nRaw training rows: {len(train):,}"
    )

    # --------------------------------------------------------
    # Payment anomaly correction
    # --------------------------------------------------------

    train = normalize_payment_values(
        train,
        products,
    )

    # --------------------------------------------------------
    # Deduplicate
    # --------------------------------------------------------

    train = deduplicate_train(train)

    # --------------------------------------------------------
    # Customer temporal features
    # --------------------------------------------------------

    train = add_customer_history(train)

    # --------------------------------------------------------
    # Customer/product enrichment
    # --------------------------------------------------------

    train = add_customer_product_features(
        train,
        customers,
        products,
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    train_features = prepare_features(
        train
    )

    y = train["returned"].astype(int)

    feature_columns = get_feature_columns(
        train_features.assign(returned=y)
    )

    X = train_features[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    sort_index = (
        train["order_placed_at"]
        .sort_values()
        .index
    )

    X = X.loc[sort_index].reset_index(drop=True)
    y = y.loc[sort_index].reset_index(drop=True)
    train_sorted = (
        train.loc[sort_index]
        .reset_index(drop=True)
    )

    split_index = int(
        len(X)
        * (1 - VALIDATION_FRACTION)
    )

    X_train = X.iloc[:split_index].copy()
    X_valid = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_valid = y.iloc[split_index:].copy()

    valid_orders = train_sorted.iloc[
        split_index:
    ].copy()

    print("\nChronological split")
    print("-" * 70)

    print(
        f"Training rows:   {len(X_train):,}"
    )

    print(
        f"Validation rows: {len(X_valid):,}"
    )

    print(
        f"Training period: "
        f"{train_sorted['order_placed_at'].iloc[0]}"
        f" → "
        f"{train_sorted['order_placed_at'].iloc[split_index - 1]}"
    )

    print(
        f"Validation period: "
        f"{valid_orders['order_placed_at'].iloc[0]}"
        f" → "
        f"{valid_orders['order_placed_at'].iloc[-1]}"
    )

    print(
        f"Training return rate: "
        f"{y_train.mean():.4%}"
    )

    print(
        f"Validation return rate: "
        f"{y_valid.mean():.4%}"
    )

    # --------------------------------------------------------
    # Column types
    # --------------------------------------------------------

    categorical_columns = (
        X_train
        .select_dtypes(
            include=["object", "string"]
        )
        .columns
        .tolist()
    )

    numerical_columns = (
        X_train
        .select_dtypes(
            include=["number", "bool"]
        )
        .columns
        .tolist()
    )

    print(
        f"\nFeatures before encoding: "
        f"{len(feature_columns)}"
    )

    print(
        f"Numerical features: "
        f"{len(numerical_columns)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_columns)}"
    )

    # --------------------------------------------------------
    # Preprocessing
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        categorical_columns,
        numerical_columns,
    )

    X_train_encoded = (
        preprocessor.fit_transform(
            X_train
        )
    )

    X_valid_encoded = (
        preprocessor.transform(
            X_valid
        )
    )

    print(
        f"Encoded feature count: "
        f"{X_train_encoded.shape[1]}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = build_model()

    print("\nTraining XGBoost...")

    model.fit(
        X_train_encoded,
        y_train,
        verbose=False,
    )

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    probabilities = (
        model.predict_proba(
            X_valid_encoded
        )[:, 1]
    )

    default_threshold = 0.50

    predictions = (
        probabilities >= default_threshold
    ).astype(int)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    roc_auc = roc_auc_score(
        y_valid,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_valid,
        probabilities,
    )

    accuracy = accuracy_score(
        y_valid,
        predictions,
    )

    precision = precision_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_valid,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_valid,
        predictions,
    )

    print("\n" + "=" * 70)
    print("VALIDATION PERFORMANCE")
    print("=" * 70)

    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"PR-AUC:    {pr_auc:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1:        {f1:.4f}")

    print("\nConfusion matrix:")
    print(cm)

    # --------------------------------------------------------
    # Threshold analysis
    # --------------------------------------------------------

    threshold_df = create_threshold_analysis(
        y_valid,
        probabilities,
    )

    threshold_df.to_csv(
        REPORT_DIR / "threshold_analysis.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Validation predictions
    # --------------------------------------------------------

    validation_predictions = (
        valid_orders[
            [
                "order_id",
                "order_placed_at",
                "returned",
            ]
        ]
        .copy()
    )

    validation_predictions[
        "predicted_return_probability"
    ] = probabilities

    validation_predictions[
        "predicted_return_default_threshold"
    ] = predictions

    validation_predictions.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    # --------------------------------------------------------
    # Metrics report
    # --------------------------------------------------------

    metrics = {
        "model": "XGBClassifier",
        "validation_method": (
            "chronological 80/20 split"
        ),
        "total_clean_training_rows": int(
            len(train)
        ),
        "training_rows": int(
            len(X_train)
        ),
        "validation_rows": int(
            len(X_valid)
        ),
        "validation_return_rate": float(
            y_valid.mean()
        ),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "accuracy_at_0_50": float(
            accuracy
        ),
        "precision_at_0_50": float(
            precision
        ),
        "recall_at_0_50": float(
            recall
        ),
        "f1_at_0_50": float(f1),
        "confusion_matrix": cm.tolist(),
        "baseline_roc_auc": BASELINE_ROC_AUC,
        "baseline_pr_auc": BASELINE_PR_AUC,
        "roc_auc_delta": float(
            roc_auc - BASELINE_ROC_AUC
        ),
        "pr_auc_delta": float(
            pr_auc - BASELINE_PR_AUC
        ),
        "feature_count_before_encoding": int(
            len(feature_columns)
        ),
        "encoded_feature_count": int(
            X_train_encoded.shape[1]
        ),
    }

    with open(
        REPORT_DIR / "model_metrics.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metrics,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    save_feature_importance(
        model,
        preprocessor,
    )

    # --------------------------------------------------------
    # Save validation artifacts only
    #
    # The final production model will be created by test.py.
    # --------------------------------------------------------

    validation_metadata = {
        "purpose": "validation_model",
        "model": "XGBClassifier",
        "validation_method": (
            "chronological 80/20 split"
        ),
        "random_state": RANDOM_STATE,
        "feature_columns": feature_columns,
        "feature_count_before_encoding": len(
            feature_columns
        ),
        "encoded_feature_count": int(
            X_train_encoded.shape[1]
        ),
        "excluded_columns": [
            "order_id",
            "customer_id",
            "order_placed_at",
            "delivery_note",
            "pickup_scheduled_at",
            "last_service_event_type",
            "delivery_pincode",
            "discount_rate",
            "order_day_of_month",
            "customer_prior_orders_temporal",
            "list_price_inr_x",
            "list_price_inr_y",
        ],
    }

    with open(
        REPORT_DIR / "validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            validation_metadata,
            file,
            indent=2,
        )

    print("\n" + "=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print(
        f"Saved: "
        f"{OUTPUT_DIR / 'validation_predictions.csv'}"
    )

    print(
        f"Saved: "
        f"{REPORT_DIR / 'model_metrics.json'}"
    )

    print(
        f"Saved: "
        f"{REPORT_DIR / 'threshold_analysis.csv'}"
    )

    print(
        "\nNOTE: The final production model is created "
        "separately by src/test.py using ALL cleaned "
        "training rows."
    )


if __name__ == "__main__":
    main()