from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
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

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test_unlabelled.csv"
CUSTOMERS_PATH = DATA_DIR / "customers.csv"
PRODUCTS_PATH = DATA_DIR / "products.csv"

MODEL_PATH = MODEL_DIR / "model.joblib"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    train = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["order_placed_at"],
    )

    test = pd.read_csv(
        TEST_PATH,
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

    return (
        train,
        test,
        customers,
        products,
    )


# ============================================================
# PAYMENT ANOMALY CORRECTION
# ============================================================

def normalize_payment_values(
    train,
    test,
    products,
):
    train = train.copy()
    test = test.copy()

    product_prices = (
        products[
            [
                "sku",
                "list_price_inr",
            ]
        ]
        .drop_duplicates("sku")
    )

    def normalize(df, name):
        df = df.merge(
            product_prices,
            on="sku",
            how="left",
            suffixes=("", "_product"),
        )

        expected_value = (
            df["list_price_inr"]
            * df["qty"]
            * (1 - df["discount_pct"] / 100)
        )

        ratio = (
            df["order_value_inr"]
            / expected_value.replace(
                0,
                np.nan,
            )
        )

        october_2025 = (
            (df["order_placed_at"].dt.year == 2025)
            & (df["order_placed_at"].dt.month == 10)
        )

        anomaly = (
            october_2025
            & ratio.between(
                90,
                110,
                inclusive="both",
            )
        )

        count = int(anomaly.sum())

        df.loc[
            anomaly,
            "order_value_inr",
        ] = (
            df.loc[
                anomaly,
                "order_value_inr",
            ]
            / 100
        )

        print(
            f"{name} October 2025 payment anomalies "
            f"corrected: {count:,}"
        )

        df = df.drop(
            columns=["list_price_inr"]
        )

        return df

    train = normalize(
        train,
        "Train",
    )

    test = normalize(
        test,
        "Test",
    )

    return train, test


# ============================================================
# DEDUPLICATE TRAIN
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
        f"Training duplicate rows removed: "
        f"{removed:,}"
    )

    return train


# ============================================================
# CUSTOMER HISTORY
# ============================================================

def add_customer_history(
    train,
    test,
):
    train = train.copy()
    test = test.copy()

    train["_dataset_order"] = 0
    test["_dataset_order"] = 1

    train["_returned_for_history"] = (
        train["returned"]
    )

    test["_returned_for_history"] = np.nan

    combined = pd.concat(
        [
            train,
            test,
        ],
        ignore_index=True,
        sort=False,
    )

    combined = (
        combined
        .sort_values(
            [
                "customer_id",
                "order_placed_at",
                "_dataset_order",
            ]
        )
        .reset_index(drop=True)
    )

    grouped = combined.groupby(
        "customer_id",
        sort=False,
    )

    combined[
        "customer_prior_orders_temporal"
    ] = grouped.cumcount()

    combined[
        "customer_prior_returns_temporal"
    ] = (
        grouped[
            "_returned_for_history"
        ]
        .cumsum()
        .fillna(0)
        - combined[
            "_returned_for_history"
        ].fillna(0)
    )

    combined[
        "customer_prior_returns_temporal"
    ] = (
        combined[
            "customer_prior_returns_temporal"
        ]
        .astype(float)
    )

    combined[
        "customer_prior_return_rate_temporal"
    ] = np.where(
        combined[
            "customer_prior_orders_temporal"
        ] > 0,
        combined[
            "customer_prior_returns_temporal"
        ]
        / combined[
            "customer_prior_orders_temporal"
        ],
        0.0,
    )

    train_result = (
        combined[
            combined["_dataset_order"] == 0
        ]
        .copy()
    )

    test_result = (
        combined[
            combined["_dataset_order"] == 1
        ]
        .copy()
    )

    helper_columns = [
        "_dataset_order",
        "_returned_for_history",
    ]

    train_result = train_result.drop(
        columns=helper_columns
    )

    test_result = test_result.drop(
        columns=helper_columns
    )

    return (
        train_result,
        test_result,
    )


# ============================================================
# CUSTOMER + PRODUCT FEATURES
# ============================================================

def add_customer_product_features(
    train,
    test,
    customers,
    products,
):
    train = train.copy()
    test = test.copy()

    def enrich(df):
        df = df.merge(
            customers,
            on="customer_id",
            how="left",
        )

        df = df.merge(
            products,
            on="sku",
            how="left",
            suffixes=("", "_product"),
        )

        # Customer features

        df["customer_tenure_days"] = (
            df["order_placed_at"]
            - df["signup_date"]
        ).dt.days

        df["has_customer_history"] = (
            df["customer_prior_orders"] > 0
        ).astype(int)

        df["has_prior_return"] = (
            df["customer_prior_returns"] > 0
        ).astype(int)

        df["customer_prior_return_rate"] = np.where(
            df["customer_prior_orders"] > 0,
            df["customer_prior_returns"]
            / df["customer_prior_orders"],
            0.0,
        )

        # Product features

        df["product_age_days"] = (
            df["order_placed_at"]
            - df["launch_date"]
        ).dt.days

        # Date features

        df["order_year"] = (
            df["order_placed_at"].dt.year
        )

        df["order_month"] = (
            df["order_placed_at"].dt.month
        )

        df["order_day"] = (
            df["order_placed_at"].dt.day
        )

        df["order_day_of_week"] = (
            df["order_placed_at"].dt.dayofweek
        )

        df["order_week_of_year"] = (
            df["order_placed_at"]
            .dt.isocalendar()
            .week
            .astype(int)
        )

        df["order_day_of_month"] = (
            df["order_placed_at"].dt.day
        )

        df["order_hour"] = (
            df["order_placed_at"].dt.hour
        )

        df["is_weekend"] = (
            df["order_day_of_week"] >= 5
        ).astype(int)

        # Cyclical features

        df["month_sin"] = np.sin(
            2
            * np.pi
            * df["order_month"]
            / 12
        )

        df["month_cos"] = np.cos(
            2
            * np.pi
            * df["order_month"]
            / 12
        )

        df["day_of_week_sin"] = np.sin(
            2
            * np.pi
            * df["order_day_of_week"]
            / 7
        )

        df["day_of_week_cos"] = np.cos(
            2
            * np.pi
            * df["order_day_of_week"]
            / 7
        )

        # Pincode

        df["is_default_pincode"] = (
            df["delivery_pincode"] == 0
        ).astype(int)

        # Delivery note metadata

        note = (
            df["delivery_note"]
            .fillna("")
            .astype(str)
        )

        df["delivery_note_missing"] = (
            df["delivery_note"]
            .isna()
            .astype(int)
        )

        df["delivery_note_length"] = (
            note.str.len()
        )

        df["delivery_note_word_count"] = (
            note.str.split()
            .str.len()
        )

        # Order value

        df["order_value_per_unit"] = (
            df["order_value_inr"]
            / df["qty"].replace(
                0,
                np.nan,
            )
        )

        df["multi_quantity_order"] = (
            df["qty"] > 1
        ).astype(int)

        return df

    train = enrich(train)
    test = enrich(test)

    return train, test


# ============================================================
# PREPARE MODEL FEATURES
# ============================================================

def prepare_features(
    train,
    test,
):
    train = train.copy()
    test = test.copy()

    # Canonical product price

    if "list_price_inr" in train.columns:
        train["list_price_inr_canonical"] = (
            train["list_price_inr"]
        )

        test["list_price_inr_canonical"] = (
            test["list_price_inr"]
        )

    elif "list_price_inr_product" in train.columns:
        train["list_price_inr_canonical"] = (
            train["list_price_inr_product"]
        )

        test["list_price_inr_canonical"] = (
            test["list_price_inr_product"]
        )

    else:
        raise ValueError(
            "Canonical product price unavailable."
        )

    train["list_price_inr"] = (
        train["list_price_inr_canonical"]
    )

    test["list_price_inr"] = (
        test["list_price_inr_canonical"]
    )

    columns_to_drop = [
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
        "returned",
        "signup_date",
        "launch_date",
    ]

    train = train.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in train.columns
        ]
    )

    test = test.drop(
        columns=[
            column
            for column in columns_to_drop
            if column in test.columns
        ]
    )

    return train, test


# ============================================================
# BUILD PREPROCESSOR
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

    return ColumnTransformer(
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


# ============================================================
# XGBOOST MODEL
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
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("KESTREL HOME — FINAL FULL-TRAIN TEST PREDICTION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load raw data
    # --------------------------------------------------------

    (
        train,
        test,
        customers,
        products,
    ) = load_data()

    print(
        f"\nRaw train rows: {len(train):,}"
    )

    print(
        f"Raw test rows:  {len(test):,}"
    )

    # --------------------------------------------------------
    # Normalize payment anomaly
    # --------------------------------------------------------

    (
        train,
        test,
    ) = normalize_payment_values(
        train,
        test,
        products,
    )

    # --------------------------------------------------------
    # Deduplicate training data
    # --------------------------------------------------------

    train = deduplicate_train(
        train
    )

    print(
        f"Clean training rows: {len(train):,}"
    )

    # --------------------------------------------------------
    # Customer history
    # --------------------------------------------------------

    (
        train,
        test,
    ) = add_customer_history(
        train,
        test,
    )

    # --------------------------------------------------------
    # Customer/product enrichment
    # --------------------------------------------------------

    (
        train,
        test,
    ) = add_customer_product_features(
        train,
        test,
        customers,
        products,
    )

    # --------------------------------------------------------
    # Prepare model features
    # --------------------------------------------------------

    (
        train_features,
        test_features,
    ) = prepare_features(
        train,
        test,
    )

    feature_columns = list(
        train_features.columns
    )

    # Ensure exact same columns
    missing_test_features = [
        column
        for column in feature_columns
        if column not in test_features.columns
    ]

    if missing_test_features:
        raise ValueError(
            "Test is missing model features:\n"
            + "\n".join(
                missing_test_features
            )
        )

    extra_test_features = [
        column
        for column in test_features.columns
        if column not in feature_columns
    ]

    if extra_test_features:
        print("\nIgnoring extra test features:")
        for column in extra_test_features:
            print(f"  - {column}")

    test_features = test_features[
        feature_columns
    ].copy()

    y = train["returned"].astype(int)

    X = train_features[
        feature_columns
    ].copy()

    # --------------------------------------------------------
    # Column types
    # --------------------------------------------------------

    categorical_columns = (
        X
        .select_dtypes(
            include=["object", "string"]
        )
        .columns
        .tolist()
    )

    numerical_columns = (
        X
        .select_dtypes(
            include=["number", "bool"]
        )
        .columns
        .tolist()
    )

    print("\n" + "=" * 70)
    print("FINAL TRAINING DATA")
    print("=" * 70)

    print(
        f"Training rows: "
        f"{len(X):,}"
    )

    print(
        f"Return rows: "
        f"{int(y.sum()):,}"
    )

    print(
        f"Non-return rows: "
        f"{int((y == 0).sum()):,}"
    )

    print(
        f"Return rate: "
        f"{y.mean():.4%}"
    )

    print(
        f"Features before encoding: "
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
    # Fit preprocessor on ALL training data
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        categorical_columns,
        numerical_columns,
    )

    X_encoded = (
        preprocessor.fit_transform(
            X
        )
    )

    test_encoded = (
        preprocessor.transform(
            test_features
        )
    )

    print(
        f"Encoded features: "
        f"{X_encoded.shape[1]}"
    )

    # --------------------------------------------------------
    # Train final model on ALL training rows
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINAL MODEL ON ALL TRAINING DATA")
    print("=" * 70)

    model = build_model()

    model.fit(
        X_encoded,
        y,
        verbose=False,
    )

    print(
        f"Model trained on "
        f"{len(X):,} rows."
    )

    # --------------------------------------------------------
    # Predict ALL test rows
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("GENERATING TEST PREDICTIONS")
    print("=" * 70)

    scores = (
        model.predict_proba(
            test_encoded
        )[:, 1]
    )

    predictions = pd.DataFrame(
        {
            "order_id": test["order_id"].astype(str),
            "score": scores.astype(float),
        }
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(predictions) != len(test):
        raise ValueError(
            "Prediction count does not match test row count."
        )

    if predictions["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs found in predictions."
        )

    if predictions["score"].isna().any():
        raise ValueError(
            "NaN prediction scores found."
        )

    if not predictions["score"].between(
        0,
        1,
    ).all():
        raise ValueError(
            "Prediction scores outside [0, 1]."
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_path = (
        OUTPUT_DIR / "predictions.csv"
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save final model
    # --------------------------------------------------------

    joblib.dump(
        model,
        MODEL_PATH,
    )

    joblib.dump(
        preprocessor,
        PREPROCESSOR_PATH,
    )

    metadata = {
        "model": "XGBClassifier",
        "purpose": "final_production_test_prediction",
        "training_rows": int(len(X)),
        "test_rows": int(len(test)),
        "training_return_rate": float(
            y.mean()
        ),
        "random_state": RANDOM_STATE,
        "feature_columns": feature_columns,
        "feature_count_before_encoding": int(
            len(feature_columns)
        ),
        "encoded_feature_count": int(
            X_encoded.shape[1]
        ),
        "numerical_features": numerical_columns,
        "categorical_features": categorical_columns,
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
            "signup_date",
            "launch_date",
            "returned",
        ],
        "prediction_file": (
            "outputs/predictions.csv"
        ),
    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("FINAL PREDICTION SUMMARY")
    print("=" * 70)

    print(
        f"Predictions generated: "
        f"{len(predictions):,}"
    )

    print(
        f"Minimum score: "
        f"{predictions['score'].min():.6f}"
    )

    print(
        f"Maximum score: "
        f"{predictions['score'].max():.6f}"
    )

    print(
        f"Mean score: "
        f"{predictions['score'].mean():.6f}"
    )

    print(
        f"Median score: "
        f"{predictions['score'].median():.6f}"
    )

    print("\nThreshold counts:")

    for threshold in [
        0.10,
        0.15,
        0.20,
        0.25,
        0.30,
        0.35,
        0.40,
        0.50,
    ]:
        count = int(
            (
                predictions["score"]
                >= threshold
            ).sum()
        )

        percentage = (
            count
            / len(predictions)
            * 100
        )

        print(
            f"  >= {threshold:.2f}: "
            f"{count:,} orders "
            f"({percentage:.2f}%)"
        )

    print("\nFirst 10 predictions:")
    print(
        predictions
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(predictions_path)
    print(MODEL_PATH)
    print(PREPROCESSOR_PATH)
    print(METADATA_PATH)

    print("\n" + "=" * 70)
    print("FINAL TEST PREDICTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()