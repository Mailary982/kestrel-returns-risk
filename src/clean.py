from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

# Confirmed from the payment anomaly audit:
# Every October 2025 training row has order_value exactly 100x
# the expected value derived from product price, quantity and discount.
OCTOBER_ANOMALY_MULTIPLIER = 100.0

LEAKAGE_COLUMNS = [
    "pickup_scheduled_at",
    "last_service_event_type",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")

    return train, test, customers, products


# ============================================================
# BASIC CLEANING
# ============================================================

def parse_dates(df):
    date_columns = [
        "order_placed_at",
    ]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce"
            )

    return df


# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_train(train):
    before = len(train)

    train = (
        train
        .sort_values(
            ["order_id", "order_placed_at"]
        )
        .drop_duplicates(
            subset=["order_id"],
            keep="first"
        )
        .reset_index(drop=True)
    )

    after = len(train)

    print("\n" + "=" * 70)
    print("DEDUPLICATION")
    print("=" * 70)

    print(f"Rows before deduplication: {before:,}")
    print(f"Rows after deduplication:  {after:,}")
    print(f"Rows removed:              {before - after:,}")

    return train


# ============================================================
# PAYMENT VALUE NORMALIZATION
# ============================================================

def normalize_payment_value(df, products):
    """
    Correct the confirmed October 2025 payment-value anomaly.

    Expected value:
        list_price * qty * (1 - discount_pct / 100)

    The audit established that October 2025 values are exactly
    100x the expected value.

    We only correct rows where:
        - order is in October 2025
        - expected value is positive
        - observed / expected is approximately 100
    """

    df = df.copy()

    if "order_value_inr" not in df.columns:
        return df

    product_price = products[
        ["sku", "list_price_inr"]
    ].drop_duplicates(
        subset=["sku"]
    )

    df = df.merge(
        product_price,
        on="sku",
        how="left",
        validate="many_to_one"
    )

    df["expected_order_value_inr"] = (
        df["list_price_inr"]
        * df["qty"]
        * (1 - df["discount_pct"] / 100)
    )

    df["payment_value_ratio"] = np.where(
        df["expected_order_value_inr"] > 0,
        df["order_value_inr"]
        / df["expected_order_value_inr"],
        np.nan
    )

    october_mask = (
        (df["order_placed_at"].dt.year == 2025)
        &
        (df["order_placed_at"].dt.month == 10)
    )

    anomaly_mask = (
        october_mask
        &
        df["payment_value_ratio"].between(
            99.5,
            100.5
        )
    )

    anomaly_count = int(anomaly_mask.sum())

    print("\n" + "=" * 70)
    print("PAYMENT VALUE NORMALIZATION")
    print("=" * 70)

    print(
        f"Confirmed October 2025 anomalies corrected: "
        f"{anomaly_count:,}"
    )

    # Preserve original value for auditability.
    df["order_value_inr_raw"] = df["order_value_inr"]

    df.loc[
        anomaly_mask,
        "order_value_inr"
    ] = (
        df.loc[
            anomaly_mask,
            "order_value_inr"
        ]
        / OCTOBER_ANOMALY_MULTIPLIER
    )

    # Diagnostic output after correction.
    corrected_ratio = np.where(
        df["expected_order_value_inr"] > 0,
        df["order_value_inr"]
        / df["expected_order_value_inr"],
        np.nan
    )

    df["payment_value_ratio_after_cleaning"] = corrected_ratio

    print(
        "Maximum payment-value ratio after cleaning: "
        f"{np.nanmax(corrected_ratio):.4f}"
    )

    return df


# ============================================================
# CUSTOMER HISTORY FEATURES
# ============================================================

def add_temporal_customer_features(
    train,
    test
):
    """
    Create customer-history features using only information that
    existed BEFORE the current order.

    This avoids using the current/future return label to construct
    the current order's feature.

    For training:
        customer_prior_orders_temporal
        customer_prior_returns_temporal
        customer_prior_return_rate_temporal

    For test:
        all historical training orders occurring before the test
        order are used.
    """

    train = train.copy()
    test = test.copy()

    train["_dataset"] = "train"
    test["_dataset"] = "test"

    train["_row_order"] = np.arange(len(train))
    test["_row_order"] = np.arange(len(test))

    combined = pd.concat(
        [train, test],
        ignore_index=True,
        sort=False
    )

    combined = combined.sort_values(
        [
            "customer_id",
            "order_placed_at",
            "_dataset",
            "_row_order",
        ]
    ).reset_index(drop=True)

    # ---------------------------------------------------------
    # Previous orders for the customer
    # ---------------------------------------------------------

    combined["customer_prior_orders_temporal"] = (
        combined
        .groupby("customer_id")
        .cumcount()
    )

    # ---------------------------------------------------------
    # Previous returns
    #
    # For train rows, returned is known.
    # For test rows, returned is NaN.
    #
    # shift(1) ensures current row's label is never included.
    # ---------------------------------------------------------

    previous_return_labels = (
        combined
        .groupby("customer_id")["returned"]
        .transform(
            lambda s: s.shift(1).fillna(0).cumsum()
        )
    )

    combined["customer_prior_returns_temporal"] = (
        previous_return_labels
    )

    # ---------------------------------------------------------
    # Historical return rate
    # ---------------------------------------------------------

    combined["customer_prior_return_rate_temporal"] = np.where(
        combined["customer_prior_orders_temporal"] > 0,
        combined["customer_prior_returns_temporal"]
        / combined["customer_prior_orders_temporal"],
        0.0
    )

    # ---------------------------------------------------------
    # Whether customer has any previous order
    # ---------------------------------------------------------

    combined["has_customer_history"] = (
        combined["customer_prior_orders_temporal"] > 0
    ).astype(int)

    # ---------------------------------------------------------
    # Split back
    # ---------------------------------------------------------

    train_result = (
        combined[
            combined["_dataset"] == "train"
        ]
        .sort_values("_row_order")
        .drop(
            columns=[
                "_dataset",
                "_row_order",
            ]
        )
        .reset_index(drop=True)
    )

    test_result = (
        combined[
            combined["_dataset"] == "test"
        ]
        .sort_values("_row_order")
        .drop(
            columns=[
                "_dataset",
                "_row_order",
                "returned",
            ],
            errors="ignore"
        )
        .reset_index(drop=True)
    )

    return train_result, test_result


# ============================================================
# DATE FEATURES
# ============================================================

def add_date_features(df):
    df = df.copy()

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
        df["order_placed_at"].dt.isocalendar().week.astype(int)
    )

    df["order_day_of_month"] = (
        df["order_placed_at"].dt.day
    )

    df["order_hour"] = (
        df["order_placed_at"].dt.hour
    )

    df["is_weekend"] = (
        df["order_placed_at"].dt.dayofweek >= 5
    ).astype(int)

    # Cyclical calendar representation.
    df["month_sin"] = np.sin(
        2 * np.pi * df["order_month"] / 12
    )

    df["month_cos"] = np.cos(
        2 * np.pi * df["order_month"] / 12
    )

    df["day_of_week_sin"] = np.sin(
        2 * np.pi * df["order_day_of_week"] / 7
    )

    df["day_of_week_cos"] = np.cos(
        2 * np.pi * df["order_day_of_week"] / 7
    )

    return df


# ============================================================
# PINCODE FEATURES
# ============================================================

def add_pincode_features(df):
    df = df.copy()

    # 000000 is a documented system-default/no-address value.
    df["is_default_pincode"] = (
        df["delivery_pincode"]
        .fillna(0)
        .astype(str)
        .str.zfill(6)
        .eq("000000")
        .astype(int)
    )

    # Treat pincode as categorical rather than numeric geography.
    df["delivery_pincode"] = (
        df["delivery_pincode"]
        .fillna(0)
        .astype(str)
        .str.zfill(6)
    )

    return df


# ============================================================
# TEXT FEATURES
# ============================================================

def add_delivery_note_features(df):
    df = df.copy()

    df["delivery_note_missing"] = (
        df["delivery_note"]
        .isna()
        .astype(int)
    )

    df["delivery_note_length"] = (
        df["delivery_note"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    df["delivery_note_word_count"] = (
        df["delivery_note"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    return df


# ============================================================
# CUSTOMER FEATURES
# ============================================================

def add_customer_features(df, customers):
    customer_data = customers.copy()

    customer_data["signup_date"] = pd.to_datetime(
        customer_data["signup_date"],
        errors="coerce"
    )

    df = df.merge(
        customer_data,
        on="customer_id",
        how="left",
        validate="many_to_one"
    )

    # Customer tenure at order time.
    df["customer_tenure_days"] = (
        df["order_placed_at"]
        - df["signup_date"]
    ).dt.days

    # Prevent negative values caused by inconsistent dates.
    df["customer_tenure_days"] = (
        df["customer_tenure_days"]
        .clip(lower=0)
    )

    return df


# ============================================================
# PRODUCT FEATURES
# ============================================================

def add_product_features(df, products):
    product_data = products.copy()

    product_data["launch_date"] = pd.to_datetime(
        product_data["launch_date"],
        errors="coerce"
    )

    df = df.merge(
        product_data,
        on="sku",
        how="left",
        validate="many_to_one"
    )

    df["product_age_days"] = (
        df["order_placed_at"]
        - df["launch_date"]
    ).dt.days

    df["product_age_days"] = (
        df["product_age_days"]
        .clip(lower=0)
    )

    return df


# ============================================================
# DERIVED ORDER FEATURES
# ============================================================

def add_order_features(df):
    df = df.copy()

    # Discount as a decimal.
    df["discount_rate"] = (
        df["discount_pct"] / 100.0
    )

    # Order value per unit.
    df["order_value_per_unit"] = np.where(
        df["qty"] > 0,
        df["order_value_inr"] / df["qty"],
        df["order_value_inr"]
    )

    # Prior returns per prior order is already represented by
    # customer_prior_return_rate_temporal.

    # Whether the customer has previously returned anything.
    df["has_prior_return"] = (
        df["customer_prior_returns_temporal"] > 0
    ).astype(int)

    # High-level order size indicator.
    df["multi_quantity_order"] = (
        df["qty"] > 1
    ).astype(int)

    return df


# ============================================================
# LEAKAGE REMOVAL
# ============================================================

def remove_leakage_columns(df):
    df = df.copy()

    columns_to_remove = [
        column
        for column in LEAKAGE_COLUMNS
        if column in df.columns
    ]

    df = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    return df


# ============================================================
# RAW / HELPER COLUMN CLEANUP
# ============================================================

def cleanup_helper_columns(df):
    df = df.copy()

    helper_columns = [
        "expected_order_value_inr",
        "payment_value_ratio",
        "payment_value_ratio_after_cleaning",
        "order_value_inr_raw",
        "list_price_inr",
        "launch_date",
        "signup_date",
    ]

    df = df.drop(
        columns=[
            column
            for column in helper_columns
            if column in df.columns
        ],
        errors="ignore"
    )

    return df


# ============================================================
# FINAL COLUMN CLEANUP
# ============================================================

def clean_missing_values(df):
    df = df.copy()

    # Categorical/text columns.
    categorical_columns = [
        "sales_channel",
        "payment_mode",
        "is_gift",
        "delivery_note",
        "source",
        "family",
        "model_name",
        "shield_member",
        "city",
        "state",
        "sku",
        "customer_id",
        "delivery_pincode",
    ]

    for column in categorical_columns:
        if column in df.columns:
            df[column] = (
                df[column]
                .fillna("UNKNOWN")
                .astype(str)
            )

    # Numeric columns.
    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns

    for column in numeric_columns:
        if column == "returned":
            continue

        df[column] = (
            df[column]
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0)
        )

    return df


# ============================================================
# BUILD TRAIN / TEST
# ============================================================

def build_datasets(
    train,
    test,
    customers,
    products
):

    # ---------------------------------------------------------
    # Parse dates first.
    # ---------------------------------------------------------

    train = parse_dates(train)
    test = parse_dates(test)

    # ---------------------------------------------------------
    # Deduplicate historical orders.
    # ---------------------------------------------------------

    train = deduplicate_train(train)

    # ---------------------------------------------------------
    # Normalize payment values.
    #
    # Training data has the confirmed October anomaly.
    # Test has no observed anomaly according to the audit, but
    # we still pass it through the same function for consistency.
    # ---------------------------------------------------------

    train = normalize_payment_value(
        train,
        products
    )

    test = normalize_payment_value(
        test,
        products
    )

    # ---------------------------------------------------------
    # Temporal customer history.
    # ---------------------------------------------------------

    train, test = add_temporal_customer_features(
        train,
        test
    )

    # ---------------------------------------------------------
    # Customer information.
    # ---------------------------------------------------------

    train = add_customer_features(
        train,
        customers
    )

    test = add_customer_features(
        test,
        customers
    )

    # ---------------------------------------------------------
    # Product information.
    # ---------------------------------------------------------

    train = add_product_features(
        train,
        products
    )

    test = add_product_features(
        test,
        products
    )

    # ---------------------------------------------------------
    # Date features.
    # ---------------------------------------------------------

    train = add_date_features(train)
    test = add_date_features(test)

    # ---------------------------------------------------------
    # Pincode features.
    # ---------------------------------------------------------

    train = add_pincode_features(train)
    test = add_pincode_features(test)

    # ---------------------------------------------------------
    # Delivery-note features.
    # ---------------------------------------------------------

    train = add_delivery_note_features(train)
    test = add_delivery_note_features(test)

    # ---------------------------------------------------------
    # Order-level features.
    # ---------------------------------------------------------

    train = add_order_features(train)
    test = add_order_features(test)

    # ---------------------------------------------------------
    # Remove post-dispatch leakage.
    # ---------------------------------------------------------

    train = remove_leakage_columns(train)
    test = remove_leakage_columns(test)

    # ---------------------------------------------------------
    # Remove helper columns.
    # ---------------------------------------------------------

    train = cleanup_helper_columns(train)
    test = cleanup_helper_columns(test)

    # ---------------------------------------------------------
    # Clean remaining missing values.
    # ---------------------------------------------------------

    train = clean_missing_values(train)
    test = clean_missing_values(test)

    # ---------------------------------------------------------
    # Ensure train target exists.
    # ---------------------------------------------------------

    if "returned" not in train.columns:
        raise ValueError(
            "Training dataset does not contain 'returned' target."
        )

    # ---------------------------------------------------------
    # Ensure test does not contain target.
    # ---------------------------------------------------------

    test = test.drop(
        columns=["returned"],
        errors="ignore"
    )

    return train, test


# ============================================================
# VALIDATION
# ============================================================

def validate_datasets(train, test):

    print("\n" + "=" * 70)
    print("CLEANED DATASET VALIDATION")
    print("=" * 70)

    print(f"Clean train rows: {len(train):,}")
    print(f"Clean train columns: {len(train.columns):,}")

    print(f"Clean test rows: {len(test):,}")
    print(f"Clean test columns: {len(test.columns):,}")

    print(
        f"\nTrain return rate: "
        f"{train['returned'].mean():.4%}"
    )

    # Leakage check.
    remaining_leakage = [
        column
        for column in LEAKAGE_COLUMNS
        if column in train.columns
        or column in test.columns
    ]

    if remaining_leakage:
        print(
            "\nWARNING — leakage columns still present:"
        )
        print(remaining_leakage)
    else:
        print(
            "\nLeakage columns successfully removed."
        )

    # Missing values.
    train_missing = train.isna().sum()
    train_missing = train_missing[
        train_missing > 0
    ]

    test_missing = test.isna().sum()
    test_missing = test_missing[
        test_missing > 0
    ]

    print("\nRemaining train missing values:")

    if len(train_missing) == 0:
        print("  None")
    else:
        print(train_missing.to_string())

    print("\nRemaining test missing values:")

    if len(test_missing) == 0:
        print("  None")
    else:
        print(test_missing.to_string())

    # Confirm order IDs.
    print(
        f"\nUnique train order IDs: "
        f"{train['order_id'].nunique():,}"
    )

    print(
        f"Unique test order IDs: "
        f"{test['order_id'].nunique():,}"
    )

    # Confirm October correction.
    october_train = train[
        (train["order_year"] == 2025)
        &
        (train["order_month"] == 10)
    ]

    if len(october_train) > 0:

        print(
            f"\nOctober 2025 cleaned rows: "
            f"{len(october_train):,}"
        )

        print(
            "October 2025 median cleaned "
            f"order value: ₹"
            f"{october_train['order_value_inr'].median():,.2f}"
        )

        print(
            "October 2025 maximum cleaned "
            f"order value: ₹"
            f"{october_train['order_value_inr'].max():,.2f}"
        )


# ============================================================
# SAVE
# ============================================================

def save_datasets(train, test):

    train_path = OUTPUT_DIR / "clean_train.csv"
    test_path = OUTPUT_DIR / "clean_test.csv"

    train.to_csv(
        train_path,
        index=False
    )

    test.to_csv(
        test_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(train_path)
    print(test_path)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KESTREL HOME — DATA CLEANING PIPELINE")
    print("=" * 70)

    train, test, customers, products = load_data()

    train, test = build_datasets(
        train,
        test,
        customers,
        products
    )

    validate_datasets(
        train,
        test
    )

    save_datasets(
        train,
        test
    )

    print("\n" + "=" * 70)
    print("DATA CLEANING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()