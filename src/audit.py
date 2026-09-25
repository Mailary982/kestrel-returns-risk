from pathlib import Path
import json
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test_unlabelled.csv")
    customers = pd.read_csv(DATA_DIR / "customers.csv")
    products = pd.read_csv(DATA_DIR / "products.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")

    return train, test, customers, products, sample_submission


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

def basic_info(df, name):
    print("\n" + "=" * 70)
    print(f"{name.upper()} DATASET")
    print("=" * 70)

    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    for column in df.columns:
        print(f"  - {column}: {df[column].dtype}")

    print("\nMissing values:")
    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:
        print("  No missing values")
    else:
        for column, count in missing.items():
            percentage = count / len(df) * 100
            print(f"  - {column}: {count:,} ({percentage:.2f}%)")


# ============================================================
# DUPLICATE AUDIT
# ============================================================

def duplicate_audit(train):
    print("\n" + "=" * 70)
    print("DUPLICATE ORDER AUDIT")
    print("=" * 70)

    total_rows = len(train)
    unique_orders = train["order_id"].nunique()
    duplicate_rows = total_rows - unique_orders

    print(f"Total train rows:       {total_rows:,}")
    print(f"Unique order IDs:       {unique_orders:,}")
    print(f"Duplicate rows:         {duplicate_rows:,}")

    duplicate_ids = (
        train["order_id"]
        .value_counts()
        .loc[lambda x: x > 1]
    )

    print(f"Order IDs appearing >1 time: {len(duplicate_ids):,}")

    # Check whether duplicate IDs have conflicting labels.
    conflicting = (
        train.groupby("order_id")["returned"]
        .nunique()
        .loc[lambda x: x > 1]
    )

    print(f"Order IDs with conflicting return labels: {len(conflicting):,}")

    if len(conflicting) > 0:
        print("\nWARNING: Conflicting labels detected.")
        print(conflicting.head(20))
    else:
        print("No conflicting return labels found.")


# ============================================================
# RETURN RATE AUDIT
# ============================================================

def return_rate_audit(train):
    print("\n" + "=" * 70)
    print("RETURN RATE AUDIT")
    print("=" * 70)

    raw_rate = train["returned"].mean()

    deduplicated = train.drop_duplicates(
        subset=["order_id"],
        keep="first"
    )

    deduplicated_rate = deduplicated["returned"].mean()

    print(f"Raw row-level return rate:        {raw_rate:.4%}")
    print(f"Deduplicated order-level rate:    {deduplicated_rate:.4%}")

    print("\nReturn counts:")
    print(train["returned"].value_counts(dropna=False))

    print("\nDeduplicated return counts:")
    print(deduplicated["returned"].value_counts(dropna=False))


# ============================================================
# LEAKAGE AUDIT
# ============================================================

def leakage_audit(train, test):
    print("\n" + "=" * 70)
    print("PRE-DISPATCH LEAKAGE AUDIT")
    print("=" * 70)

    leakage_candidates = [
        "pickup_scheduled_at",
        "last_service_event_type"
    ]

    for column in leakage_candidates:

        if column not in train.columns:
            continue

        train_non_null = train[column].notna().sum()
        test_non_null = test[column].notna().sum()

        print(f"\n{column}")
        print(f"  Train non-null: {train_non_null:,}")
        print(f"  Test non-null:  {test_non_null:,}")

    if "last_service_event_type" in train.columns:

        print("\nlast_service_event_type distribution:")

        print(
            train["last_service_event_type"]
            .value_counts(dropna=False)
            .to_string()
        )

    if "pickup_scheduled_at" in train.columns:

        print("\npickup_scheduled_at return relationship:")

        pickup_present = train[
            train["pickup_scheduled_at"].notna()
        ]

        pickup_missing = train[
            train["pickup_scheduled_at"].isna()
        ]

        if len(pickup_present) > 0:
            print(
                f"  With pickup scheduled: "
                f"{pickup_present['returned'].mean():.4%}"
            )

        if len(pickup_missing) > 0:
            print(
                f"  Without pickup scheduled: "
                f"{pickup_missing['returned'].mean():.4%}"
            )

    reverse_pickup = train[
        train["last_service_event_type"]
        .astype(str)
        .str.upper()
        .eq("REVERSE_PICKUP")
    ]

    if len(reverse_pickup) > 0:
        print(
            f"\nREVERSE_PICKUP rows: {len(reverse_pickup):,}"
        )

        print(
            f"REVERSE_PICKUP return rate: "
            f"{reverse_pickup['returned'].mean():.4%}"
        )


# ============================================================
# PAYMENT / ORDER VALUE AUDIT
# ============================================================

def payment_value_audit(train, test):
    print("\n" + "=" * 70)
    print("ORDER VALUE / PAYMENT AUDIT")
    print("=" * 70)

    required_columns = [
        "order_value_inr",
        "qty",
        "discount_pct"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in train.columns
    ]

    if missing_columns:
        print(
            "Cannot perform order-value audit. "
            f"Missing columns: {missing_columns}"
        )
        return {}

    print(
        f"Train order value range: "
        f"₹{train['order_value_inr'].min():,.2f} - "
        f"₹{train['order_value_inr'].max():,.2f}"
    )

    print(
        f"Test order value range: "
        f"₹{test['order_value_inr'].min():,.2f} - "
        f"₹{test['order_value_inr'].max():,.2f}"
    )

    # Detect suspicious 100x relationship within repeated order-value
    # patterns rather than assuming a specific pricing formula.
    suspicious_100x = (
        train["order_value_inr"].notna()
        &
        (
            train["order_value_inr"] % 100 == 0
        )
    )

    print(
        f"\nTrain rows with order_value divisible by 100: "
        f"{suspicious_100x.sum():,}"
    )

    # October-specific inspection
    if "order_placed_at" in train.columns:

        dates = pd.to_datetime(
            train["order_placed_at"],
            errors="coerce"
        )

        october = train[
            (dates.dt.year == 2025)
            &
            (dates.dt.month == 10)
        ]

        print(
            f"October 2025 train rows: {len(october):,}"
        )

        if len(october) > 0:

            print(
                f"October 2025 median order value: "
                f"₹{october['order_value_inr'].median():,.2f}"
            )

            print(
                f"Overall train median order value: "
                f"₹{train['order_value_inr'].median():,.2f}"
            )

            print(
                f"October 2025 max order value: "
                f"₹{october['order_value_inr'].max():,.2f}"
            )

    return {
        "train_order_value_min": float(
            train["order_value_inr"].min()
        ),
        "train_order_value_max": float(
            train["order_value_inr"].max()
        ),
        "test_order_value_min": float(
            test["order_value_inr"].min()
        ),
        "test_order_value_max": float(
            test["order_value_inr"].max()
        ),
    }


# ============================================================
# SEGMENT AUDIT
# ============================================================

def segment_audit(train):
    print("\n" + "=" * 70)
    print("SEGMENT RETURN-RATE AUDIT")
    print("=" * 70)

    segments = [
        "sales_channel",
        "payment_mode",
        "source",
        "is_gift",
        "last_service_event_type"
    ]

    for column in segments:

        if column not in train.columns:
            continue

        print(f"\n--- {column} ---")

        result = (
            train.groupby(column)["returned"]
            .agg(
                orders="size",
                return_rate="mean"
            )
            .sort_values("return_rate", ascending=False)
        )

        result["return_rate"] = (
            result["return_rate"] * 100
        ).round(2)

        print(result.to_string())

    # Pincode default indicator
    if "delivery_pincode" in train.columns:

        print("\n--- delivery_pincode == 000000 ---")

        default_pin = train[
            train["delivery_pincode"].astype(str) == "0"
        ]

        default_pin_alt = train[
            train["delivery_pincode"].astype(str) == "000000"
        ]

        if len(default_pin) > 0:
            selected = default_pin
        else:
            selected = default_pin_alt

        if len(selected) > 0:
            print(f"Orders: {len(selected):,}")
            print(
                f"Return rate: "
                f"{selected['returned'].mean():.4%}"
            )
        else:
            print("No 000000 pincode rows detected.")


# ============================================================
# TRAIN / TEST COLUMN COMPARISON
# ============================================================

def schema_comparison(train, test):
    print("\n" + "=" * 70)
    print("TRAIN / TEST SCHEMA COMPARISON")
    print("=" * 70)

    train_columns = set(train.columns)
    test_columns = set(test.columns)

    only_train = sorted(train_columns - test_columns)
    only_test = sorted(test_columns - train_columns)

    print(f"Train-only columns: {only_train}")
    print(f"Test-only columns:  {only_test}")

    common = sorted(train_columns & test_columns)

    print(f"\nCommon columns: {len(common)}")

    for column in common:

        train_dtype = str(train[column].dtype)
        test_dtype = str(test[column].dtype)

        if train_dtype != test_dtype:

            print(
                f"WARNING dtype mismatch: "
                f"{column}: train={train_dtype}, "
                f"test={test_dtype}"
            )


# ============================================================
# DATA QUALITY SUMMARY
# ============================================================

def build_summary(train, test, customers, products):

    summary = {
        "train_rows": int(len(train)),
        "train_columns": int(len(train.columns)),
        "test_rows": int(len(test)),
        "test_columns": int(len(test.columns)),
        "customers_rows": int(len(customers)),
        "products_rows": int(len(products)),
        "unique_train_orders": int(
            train["order_id"].nunique()
        ),
        "duplicate_train_rows": int(
            len(train) - train["order_id"].nunique()
        ),
        "raw_return_rate": float(
            train["returned"].mean()
        ),
        "deduplicated_return_rate": float(
            train.drop_duplicates(
                subset=["order_id"],
                keep="first"
            )["returned"].mean()
        ),
        "train_missing_values": {
            column: int(count)
            for column, count in train.isna().sum().items()
            if count > 0
        },
        "test_missing_values": {
            column: int(count)
            for column, count in test.isna().sum().items()
            if count > 0
        }
    }

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KESTREL HOME — RETURNS DATA AUDIT")
    print("=" * 70)

    train, test, customers, products, sample_submission = load_data()

    basic_info(train, "Train")
    basic_info(test, "Test")
    basic_info(customers, "Customers")
    basic_info(products, "Products")

    duplicate_audit(train)

    return_rate_audit(train)

    leakage_audit(
        train,
        test
    )

    payment_value_audit(
        train,
        test
    )

    segment_audit(train)

    schema_comparison(
        train,
        test
    )

    summary = build_summary(
        train,
        test,
        customers,
        products
    )

    output_file = REPORTS_DIR / "data_audit_summary.json"

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    print("\n" + "=" * 70)
    print("AUDIT COMPLETE")
    print("=" * 70)

    print(
        f"\nSummary saved to:\n"
        f"{output_file}"
    )


if __name__ == "__main__":
    main()