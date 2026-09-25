from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = PROJECT_ROOT / "reports"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


VALIDATION_PATH = OUTPUT_DIR / "validation_predictions.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "predictions.csv"

ECONOMICS_PATH = REPORT_DIR / "intervention_economics.csv"
SUMMARY_PATH = REPORT_DIR / "intervention_economics_summary.txt"


# ============================================================
# BUSINESS ASSUMPTIONS
# ============================================================

RETURN_COST = 1150.0
CONFIRMATION_CALL_COST = 45.0

# Historical spring pilot:
# approximately 35% of otherwise-occurring returns prevented.
HISTORICAL_PREVENTION_RATE = 0.35


THRESHOLDS = [
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


# ============================================================
# LOAD VALIDATION DATA
# ============================================================

def load_validation():

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{VALIDATION_PATH}"
        )

    validation = pd.read_csv(
        VALIDATION_PATH
    )

    required_columns = [
        "predicted_return_probability",
        "returned",
    ]

    missing = [
        column
        for column in required_columns
        if column not in validation.columns
    ]

    if missing:
        raise ValueError(
            "Validation file is missing columns:\n"
            + "\n".join(missing)
        )

    validation = validation.dropna(
        subset=required_columns
    ).copy()

    validation["returned"] = (
        validation["returned"]
        .astype(int)
    )

    validation["predicted_return_probability"] = (
        validation["predicted_return_probability"]
        .astype(float)
    )

    return validation


# ============================================================
# LOAD TEST PREDICTIONS
# ============================================================

def load_test_predictions():

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n{PREDICTIONS_PATH}"
        )

    predictions = pd.read_csv(
        PREDICTIONS_PATH
    )

    required_columns = [
        "order_id",
        "score",
    ]

    missing = [
        column
        for column in required_columns
        if column not in predictions.columns
    ]

    if missing:
        raise ValueError(
            "Prediction file is missing columns:\n"
            + "\n".join(missing)
        )

    if predictions["order_id"].duplicated().any():
        raise ValueError(
            "Duplicate order IDs found in predictions."
        )

    if predictions["score"].isna().any():
        raise ValueError(
            "Missing prediction scores found."
        )

    predictions["score"] = (
        predictions["score"]
        .astype(float)
    )

    return predictions


# ============================================================
# ECONOMIC ANALYSIS
# ============================================================

def calculate_economics(
    validation,
    test_predictions,
):

    rows = []

    validation_total = len(validation)

    validation_returns = int(
        validation["returned"].sum()
    )

    validation_return_rate = (
        validation_returns
        / validation_total
    )

    print("=" * 78)
    print("KESTREL HOME — INTERVENTION ECONOMICS")
    print("=" * 78)

    print(
        f"\nValidation orders: "
        f"{validation_total:,}"
    )

    print(
        f"Validation returns: "
        f"{validation_returns:,}"
    )

    print(
        f"Overall validation return rate: "
        f"{validation_return_rate:.2%}"
    )

    print(
        f"\nReturn processing cost: "
        f"₹{RETURN_COST:,.0f}"
    )

    print(
        f"Confirmation call cost: "
        f"₹{CONFIRMATION_CALL_COST:,.0f}"
    )

    print(
        f"Historical prevention assumption: "
        f"{HISTORICAL_PREVENTION_RATE:.0%}"
    )

    # --------------------------------------------------------
    # Pure scenario break-even probability
    # --------------------------------------------------------

    break_even_probability = (
        CONFIRMATION_CALL_COST
        / (
            HISTORICAL_PREVENTION_RATE
            * RETURN_COST
        )
    )

    print(
        f"\nScenario break-even return probability: "
        f"{break_even_probability:.2%}"
    )

    print(
        "\nCalculating threshold economics..."
    )

    for threshold in THRESHOLDS:

        # ----------------------------------------------------
        # Validation orders flagged at threshold
        # ----------------------------------------------------

        flagged_validation = validation[
            validation[
                "predicted_return_probability"
            ] >= threshold
        ]

        validation_flagged_count = len(
            flagged_validation
        )

        if validation_flagged_count == 0:
            continue

        observed_return_rate = (
            flagged_validation["returned"]
            .mean()
        )

        observed_returns = int(
            flagged_validation["returned"]
            .sum()
        )

        validation_flagged_pct = (
            validation_flagged_count
            / validation_total
        )

        # ----------------------------------------------------
        # Test orders flagged
        # ----------------------------------------------------

        flagged_test = test_predictions[
            test_predictions["score"] >= threshold
        ]

        test_flagged_count = len(
            flagged_test
        )

        test_flagged_pct = (
            test_flagged_count
            / len(test_predictions)
        )

        # ----------------------------------------------------
        # Expected returns in test population
        #
        # We use the observed return rate among similarly
        # scored validation orders as the scenario estimate.
        # ----------------------------------------------------

        expected_test_returns = (
            test_flagged_count
            * observed_return_rate
        )

        # ----------------------------------------------------
        # Expected prevented returns
        # ----------------------------------------------------

        expected_prevented_returns = (
            expected_test_returns
            * HISTORICAL_PREVENTION_RATE
        )

        # ----------------------------------------------------
        # Cost of confirmation calls
        # ----------------------------------------------------

        confirmation_call_cost = (
            test_flagged_count
            * CONFIRMATION_CALL_COST
        )

        # ----------------------------------------------------
        # Gross avoided return processing cost
        # ----------------------------------------------------

        gross_avoided_return_cost = (
            expected_prevented_returns
            * RETURN_COST
        )

        # ----------------------------------------------------
        # Net scenario value
        # ----------------------------------------------------

        net_scenario_value = (
            gross_avoided_return_cost
            - confirmation_call_cost
        )

        # ----------------------------------------------------
        # Benefit / cost ratio
        # ----------------------------------------------------

        if confirmation_call_cost > 0:

            benefit_cost_ratio = (
                gross_avoided_return_cost
                / confirmation_call_cost
            )

        else:

            benefit_cost_ratio = np.nan

        # ----------------------------------------------------
        # Prevention rate required to break even
        # for this observed score band
        # ----------------------------------------------------

        if (
            observed_return_rate > 0
            and RETURN_COST > 0
        ):

            break_even_prevention_rate = (
                CONFIRMATION_CALL_COST
                / (
                    observed_return_rate
                    * RETURN_COST
                )
            )

        else:

            break_even_prevention_rate = np.nan

        rows.append(
            {
                "threshold": threshold,

                "validation_flagged_orders": (
                    validation_flagged_count
                ),

                "validation_flagged_pct": (
                    validation_flagged_pct
                ),

                "validation_observed_return_rate": (
                    observed_return_rate
                ),

                "validation_observed_returns": (
                    observed_returns
                ),

                "test_flagged_orders": (
                    test_flagged_count
                ),

                "test_flagged_pct": (
                    test_flagged_pct
                ),

                "expected_test_returns": (
                    expected_test_returns
                ),

                "expected_prevented_returns": (
                    expected_prevented_returns
                ),

                "confirmation_call_cost_inr": (
                    confirmation_call_cost
                ),

                "gross_avoided_return_cost_inr": (
                    gross_avoided_return_cost
                ),

                "net_scenario_value_inr": (
                    net_scenario_value
                ),

                "benefit_cost_ratio": (
                    benefit_cost_ratio
                ),

                "break_even_prevention_rate": (
                    break_even_prevention_rate
                ),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# SUMMARY
# ============================================================

def create_summary(
    economics,
    test_predictions,
    validation,
):

    lines = []

    lines.append(
        "KESTREL HOME — INTERVENTION ECONOMICS SUMMARY"
    )

    lines.append("=" * 65)
    lines.append("")

    lines.append(
        f"Validation sample: "
        f"{len(validation):,} orders"
    )

    lines.append(
        f"Validation return rate: "
        f"{validation['returned'].mean():.2%}"
    )

    lines.append(
        f"Test prediction population: "
        f"{len(test_predictions):,} orders"
    )

    lines.append("")

    lines.append(
        "BUSINESS ASSUMPTIONS"
    )

    lines.append("-" * 65)

    lines.append(
        f"Return processing cost: "
        f"₹{RETURN_COST:,.0f}"
    )

    lines.append(
        f"Confirmation call cost: "
        f"₹{CONFIRMATION_CALL_COST:,.0f}"
    )

    lines.append(
        f"Historical prevention assumption: "
        f"{HISTORICAL_PREVENTION_RATE:.0%}"
    )

    break_even_probability = (
        CONFIRMATION_CALL_COST
        / (
            HISTORICAL_PREVENTION_RATE
            * RETURN_COST
        )
    )

    lines.append(
        f"Scenario break-even return probability: "
        f"{break_even_probability:.2%}"
    )

    lines.append("")

    lines.append(
        "THRESHOLD SCENARIOS"
    )

    lines.append("-" * 65)

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

        row = economics[
            economics["threshold"] == threshold
        ]

        if row.empty:
            continue

        row = row.iloc[0]

        lines.append(
            f"Threshold {threshold:.3f}: "
            f"{int(row['test_flagged_orders']):,} "
            f"test orders flagged "
            f"({row['test_flagged_pct']:.2%}); "
            f"validation return rate "
            f"{row['validation_observed_return_rate']:.2%}; "
            f"expected prevented returns "
            f"{row['expected_prevented_returns']:.1f}; "
            f"net scenario value "
            f"₹{row['net_scenario_value_inr']:,.0f}"
        )

    lines.append("")

    # --------------------------------------------------------
    # Highest modeled scenario value
    # --------------------------------------------------------

    if not economics.empty:

        max_value_row = economics.loc[
            economics[
                "net_scenario_value_inr"
            ].idxmax()
        ]

        lines.append(
            "HIGHEST MODELED SCENARIO VALUE"
        )

        lines.append("-" * 65)

        lines.append(
            f"Threshold: "
            f"{max_value_row['threshold']:.3f}"
        )

        lines.append(
            f"Test orders flagged: "
            f"{int(max_value_row['test_flagged_orders']):,}"
        )

        lines.append(
            f"Validation observed return rate: "
            f"{max_value_row['validation_observed_return_rate']:.2%}"
        )

        lines.append(
            f"Expected test returns: "
            f"{max_value_row['expected_test_returns']:.1f}"
        )

        lines.append(
            f"Expected prevented returns: "
            f"{max_value_row['expected_prevented_returns']:.1f}"
        )

        lines.append(
            f"Confirmation-call cost: "
            f"₹{max_value_row['confirmation_call_cost_inr']:,.0f}"
        )

        lines.append(
            f"Gross avoided return cost: "
            f"₹{max_value_row['gross_avoided_return_cost_inr']:,.0f}"
        )

        lines.append(
            f"Net scenario value: "
            f"₹{max_value_row['net_scenario_value_inr']:,.0f}"
        )

    lines.append("")

    lines.append(
        "INTERPRETATION"
    )

    lines.append("-" * 65)

    lines.append(
        "The economic values are scenario estimates, "
        "not guaranteed savings."
    )

    lines.append(
        "They combine out-of-time validation return rates "
        "with the historical 35% prevention assumption."
    )

    lines.append(
        "The model score is used primarily as a risk-ranking "
        "signal rather than being treated as a calibrated "
        "probability."
    )

    lines.append(
        "The operating threshold should therefore be validated "
        "through a controlled intervention pilot."
    )

    lines.append(
        "The analysis does not assign a cost to holding or "
        "cancelling an order because no such cost was provided."
    )

    lines.append("")

    lines.append(
        "IMPORTANT: Highest modeled scenario value is an "
        "analytical result under stated assumptions, not a "
        "claim that the threshold is universally optimal."
    )

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    validation = load_validation()

    test_predictions = load_test_predictions()

    # --------------------------------------------------------
    # Calculate
    # --------------------------------------------------------

    economics = calculate_economics(
        validation,
        test_predictions,
    )

    if economics.empty:
        raise ValueError(
            "No threshold produced any flagged validation orders."
        )

    # --------------------------------------------------------
    # Save detailed table
    # --------------------------------------------------------

    economics.to_csv(
        ECONOMICS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Create summary
    # --------------------------------------------------------

    summary = create_summary(
        economics,
        test_predictions,
        validation,
    )

    with open(
        SUMMARY_PATH,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(summary)

    # --------------------------------------------------------
    # Display detailed economics
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("THRESHOLD ECONOMICS")
    print("=" * 78)

    display = economics[
        [
            "threshold",
            "validation_observed_return_rate",
            "test_flagged_orders",
            "test_flagged_pct",
            "expected_test_returns",
            "expected_prevented_returns",
            "confirmation_call_cost_inr",
            "gross_avoided_return_cost_inr",
            "net_scenario_value_inr",
            "benefit_cost_ratio",
        ]
    ].copy()

    display["threshold"] = display[
        "threshold"
    ].map(
        lambda x: f"{x:.3f}"
    )

    display[
        "validation_observed_return_rate"
    ] = display[
        "validation_observed_return_rate"
    ].map(
        lambda x: f"{x:.2%}"
    )

    display["test_flagged_pct"] = display[
        "test_flagged_pct"
    ].map(
        lambda x: f"{x:.2%}"
    )

    display["expected_test_returns"] = display[
        "expected_test_returns"
    ].map(
        lambda x: f"{x:.1f}"
    )

    display["expected_prevented_returns"] = display[
        "expected_prevented_returns"
    ].map(
        lambda x: f"{x:.1f}"
    )

    for column in [
        "confirmation_call_cost_inr",
        "gross_avoided_return_cost_inr",
        "net_scenario_value_inr",
    ]:

        display[column] = display[column].map(
            lambda x: f"₹{x:,.0f}"
        )

    display["benefit_cost_ratio"] = display[
        "benefit_cost_ratio"
    ].map(
        lambda x: f"{x:.2f}x"
    )

    print(
        display.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)

    print(summary)

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    print("\n" + "=" * 78)
    print("FILES SAVED")
    print("=" * 78)

    print(
        ECONOMICS_PATH
    )

    print(
        SUMMARY_PATH
    )

    print("\n" + "=" * 78)
    print("INTERVENTION ECONOMICS COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()