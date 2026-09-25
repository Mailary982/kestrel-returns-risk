from pathlib import Path
import json

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)


TRAIN_PATH = OUTPUT_DIR / "clean_train.csv"


# ============================================================
# LOAD
# ============================================================

def load_data():

    df = pd.read_csv(
        TRAIN_PATH,
        parse_dates=["order_placed_at"]
    )

    return df


# ============================================================
# CUSTOMER FEATURE CONSISTENCY
# ============================================================

def audit_customer_history(df):

    print("\n" + "=" * 70)
    print("CUSTOMER HISTORY AUDIT")
    print("=" * 70)

    comparison = df[
        [
            "customer_prior_orders",
            "customer_prior_returns",
            "customer_prior_orders_temporal",
            "customer_prior_returns_temporal",
        ]
    ].copy()

    comparison["orders_difference"] = (
        comparison["customer_prior_orders"]
        - comparison["customer_prior_orders_temporal"]
    )

    comparison["returns_difference"] = (
        comparison["customer_prior_returns"]
        - comparison["customer_prior_returns_temporal"]
    )

    orders_match = (
        comparison["orders_difference"] == 0
    )

    returns_match = (
        comparison["returns_difference"] == 0
    )

    print(
        "Provided prior-order count vs reconstructed:"
    )

    print(
        f"  Exact match: "
        f"{orders_match.mean():.2%}"
    )

    print(
        f"  Different: "
        f"{(~orders_match).sum():,}"
    )

    print(
        "\nProvided prior-return count vs reconstructed:"
    )

    print(
        f"  Exact match: "
        f"{returns_match.mean():.2%}"
    )

    print(
        f"  Different: "
        f"{(~returns_match).sum():,}"
    )

    print("\nDifference statistics:")

    print(
        comparison[
            [
                "orders_difference",
                "returns_difference",
            ]
        ].describe().to_string()
    )

    return comparison


# ============================================================
# FEATURE DUPLICATION
# ============================================================

def audit_duplicate_features(df):

    print("\n" + "=" * 70)
    print("DUPLICATE / REDUNDANT FEATURE AUDIT")
    print("=" * 70)

    suspicious_pairs = [
        (
            "list_price_inr_x",
            "list_price_inr_y",
        ),
        (
            "discount_pct",
            "discount_rate",
        ),
        (
            "customer_prior_orders",
            "customer_prior_orders_temporal",
        ),
        (
            "customer_prior_returns",
            "customer_prior_returns_temporal",
        ),
    ]

    results = []

    for first, second in suspicious_pairs:

        if first not in df.columns:
            continue

        if second not in df.columns:
            continue

        a = df[first]
        b = df[second]

        same = (
            np.isclose(
                a.astype(float),
                b.astype(float),
                equal_nan=True
            )
        )

        correlation = a.corr(b)

        result = {
            "feature_a": first,
            "feature_b": second,
            "exact_or_numeric_match_rate": float(
                same.mean()
            ),
            "correlation": float(correlation),
        }

        results.append(result)

        print(
            f"\n{first} vs {second}"
        )

        print(
            f"  Match rate: {same.mean():.4%}"
        )

        print(
            f"  Correlation: {correlation:.6f}"
        )

    return results


# ============================================================
# PINCODE AUDIT
# ============================================================

def audit_pincode(df):

    print("\n" + "=" * 70)
    print("PINCODE AUDIT")
    print("=" * 70)

    print(
        f"Unique pincodes: "
        f"{df['delivery_pincode'].nunique():,}"
    )

    print(
        f"Default pincode orders: "
        f"{df['is_default_pincode'].sum():,}"
    )

    print(
        f"Default pincode rate: "
        f"{df['is_default_pincode'].mean():.2%}"
    )

    grouped = (
        df.groupby("is_default_pincode")["returned"]
        .agg(
            orders="count",
            returns="sum",
            return_rate="mean",
        )
        .reset_index()
    )

    print("\nReturn rate by default-pincode flag:")
    print(
        grouped.to_string(
            index=False
        )
    )

    return grouped


# ============================================================
# TRAIN / VALIDATION DISTRIBUTION AUDIT
# ============================================================

def audit_temporal_distribution(df):

    print("\n" + "=" * 70)
    print("TEMPORAL DISTRIBUTION AUDIT")
    print("=" * 70)

    df = df.sort_values(
        "order_placed_at"
    ).reset_index(drop=True)

    split_index = int(
        len(df) * 0.80
    )

    train = df.iloc[:split_index]
    valid = df.iloc[split_index:]

    categorical_columns = [
        "sales_channel",
        "payment_mode",
        "is_gift",
        "source",
        "city",
        "state",
        "shield_member",
        "family",
        "model_name",
    ]

    rows = []

    for column in categorical_columns:

        if column not in df.columns:
            continue

        train_dist = (
            train[column]
            .value_counts(
                normalize=True
            )
        )

        valid_dist = (
            valid[column]
            .value_counts(
                normalize=True
            )
        )

        categories = set(
            train_dist.index
        ).union(
            valid_dist.index
        )

        for category in categories:

            train_rate = train_dist.get(
                category,
                0.0
            )

            valid_rate = valid_dist.get(
                category,
                0.0
            )

            rows.append(
                {
                    "feature": column,
                    "category": category,
                    "train_rate": train_rate,
                    "validation_rate": valid_rate,
                    "absolute_difference": abs(
                        train_rate - valid_rate
                    ),
                }
            )

    distribution_df = pd.DataFrame(rows)

    if not distribution_df.empty:

        distribution_df = (
            distribution_df
            .sort_values(
                "absolute_difference",
                ascending=False
            )
            .reset_index(drop=True)
        )

        print(
            "\nLargest categorical distribution shifts:"
        )

        print(
            distribution_df.head(20).to_string(
                index=False
            )
        )

    return distribution_df


# ============================================================
# SEGMENT PERFORMANCE
# ============================================================

def audit_return_rates(df):

    print("\n" + "=" * 70)
    print("RETURN-RATE SEGMENT AUDIT")
    print("=" * 70)

    segment_columns = [
        "payment_mode",
        "sales_channel",
        "shield_member",
        "family",
        "source",
        "is_gift",
        "is_default_pincode",
    ]

    all_results = []

    for column in segment_columns:

        if column not in df.columns:
            continue

        grouped = (
            df.groupby(column)["returned"]
            .agg(
                orders="count",
                returns="sum",
                return_rate="mean",
            )
            .reset_index()
        )

        grouped["segment"] = column

        grouped = grouped.rename(
            columns={
                column: "value"
            }
        )

        all_results.append(
            grouped[
                [
                    "segment",
                    "value",
                    "orders",
                    "returns",
                    "return_rate",
                ]
            ]
        )

        print(f"\n--- {column} ---")

        print(
            grouped[
                [
                    "value",
                    "orders",
                    "returns",
                    "return_rate",
                ]
            ]
            .sort_values(
                "return_rate",
                ascending=False
            )
            .to_string(
                index=False
            )
        )

    return pd.concat(
        all_results,
        ignore_index=True
    )


# ============================================================
# NUMERIC SHIFT AUDIT
# ============================================================

def audit_numeric_shift(df):

    print("\n" + "=" * 70)
    print("NUMERIC DISTRIBUTION SHIFT AUDIT")
    print("=" * 70)

    df = df.sort_values(
        "order_placed_at"
    ).reset_index(drop=True)

    split_index = int(
        len(df) * 0.80
    )

    train = df.iloc[:split_index]
    valid = df.iloc[split_index:]

    excluded = [
        "returned",
        "delivery_pincode",
    ]

    numeric_columns = df.select_dtypes(
        include=["number"]
    ).columns.tolist()

    rows = []

    for column in numeric_columns:

        if column in excluded:
            continue

        train_median = train[column].median()
        valid_median = valid[column].median()

        train_mean = train[column].mean()
        valid_mean = valid[column].mean()

        if train_median != 0:

            median_ratio = (
                valid_median
                / train_median
            )

        else:

            median_ratio = np.nan

        rows.append(
            {
                "feature": column,
                "train_mean": train_mean,
                "validation_mean": valid_mean,
                "train_median": train_median,
                "validation_median": valid_median,
                "validation_to_train_median_ratio": median_ratio,
            }
        )

    result = pd.DataFrame(rows)

    result["median_ratio_distance"] = (
        np.abs(
            np.log(
                result[
                    "validation_to_train_median_ratio"
                ].replace(
                    0,
                    np.nan
                )
            )
        )
    )

    result = result.sort_values(
        "median_ratio_distance",
        ascending=False
    )

    print(
        result.head(20).to_string(
            index=False
        )
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("KESTREL HOME — MODEL QUALITY AUDIT")
    print("=" * 70)

    df = load_data()

    print(
        f"\nRows: {len(df):,}"
    )

    print(
        f"Columns: {len(df.columns):,}"
    )

    customer_comparison = (
        audit_customer_history(df)
    )

    duplicate_features = (
        audit_duplicate_features(df)
    )

    pincode_results = (
        audit_pincode(df)
    )

    distribution_results = (
        audit_temporal_distribution(df)
    )

    segment_results = (
        audit_return_rates(df)
    )

    numeric_results = (
        audit_numeric_shift(df)
    )

    # --------------------------------------------------------
    # Save reports
    # --------------------------------------------------------

    customer_comparison.to_csv(
        REPORT_DIR / "customer_history_comparison.csv",
        index=False
    )

    pincode_results.to_csv(
        REPORT_DIR / "pincode_audit.csv",
        index=False
    )

    distribution_results.to_csv(
        REPORT_DIR / "categorical_distribution_shift.csv",
        index=False
    )

    segment_results.to_csv(
        REPORT_DIR / "segment_return_rates.csv",
        index=False
    )

    numeric_results.to_csv(
        REPORT_DIR / "numeric_distribution_shift.csv",
        index=False
    )

    with open(
        REPORT_DIR / "duplicate_feature_audit.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            duplicate_features,
            f,
            indent=2
        )

    print("\n" + "=" * 70)
    print("MODEL QUALITY AUDIT COMPLETE")
    print("=" * 70)

    print(
        "\nReports saved under:"
    )

    print(
        "reports/customer_history_comparison.csv"
    )

    print(
        "reports/pincode_audit.csv"
    )

    print(
        "reports/categorical_distribution_shift.csv"
    )

    print(
        "reports/numeric_distribution_shift.csv"
    )

    print(
        "reports/segment_return_rates.csv"
    )


if __name__ == "__main__":
    main()