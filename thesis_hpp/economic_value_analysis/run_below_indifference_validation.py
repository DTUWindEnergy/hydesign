"""Validate one fixed price-capacity pair below the indifference curve."""

from __future__ import annotations

import json

from common import RESULTS_DIR, base_summary, solve_fixed_capacity, write_summary


REFERENCE_LAMBDA_PPA = 80.0
TEST_LAMBDA_PPA = 55.0
OBJECTIVE_TOLERANCE_EUR = 1.0
OUTPUT_PATH = (
    "below_indifference_validation/"
    "reference_lambda_80_test_lambda_55/summary.json"
)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_curve_point(curve: list[dict]) -> dict:
    matches = [
        point
        for point in curve
        if abs(float(point["reference_lambda_ppa"]) - REFERENCE_LAMBDA_PPA)
        < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one indifference-curve point for reference "
            f"lambda={REFERENCE_LAMBDA_PPA:g}, found {len(matches)}."
        )
    return matches[0]


def main() -> None:
    existing_output = RESULTS_DIR / OUTPUT_PATH
    if existing_output.exists():
        print(f"Below-indifference validation already exists: {existing_output}")
        return

    curve = read_json(RESULTS_DIR / "indifference_curve_summary.json")
    merchant = read_json(RESULTS_DIR / "merchant" / "summary.json")
    point = select_curve_point(curve)

    capacity = float(point["s_ppa_fixed"])
    contracted_energy = float(point["contracted_energy_mwh"])
    indifference_price = float(
        point["risk_adjusted_indifference_price_eur_per_mwh"]
    )
    merchant_objective = float(merchant["objective_value"])

    if TEST_LAMBDA_PPA >= indifference_price:
        raise ValueError(
            f"Test price {TEST_LAMBDA_PPA:g} must be below the indifference "
            f"price {indifference_price:.10f}."
        )

    predicted_difference = (
        TEST_LAMBDA_PPA - indifference_price
    ) * contracted_energy
    predicted_objective = merchant_objective + predicted_difference

    # S_PPA is fixed here. Every stage-two and stage-three variable remains free.
    model, results, elapsed_seconds, cfg = solve_fixed_capacity(
        TEST_LAMBDA_PPA,
        capacity,
    )
    summary = base_summary(model, results, elapsed_seconds, cfg)
    observed_difference = summary["objective_value"] - merchant_objective
    prediction_error = summary["objective_value"] - predicted_objective

    summary.update(
        {
            "run_type": "fixed_capacity_below_indifference_validation",
            "reference_lambda_ppa": REFERENCE_LAMBDA_PPA,
            "test_lambda_ppa": TEST_LAMBDA_PPA,
            "risk_adjusted_indifference_price_eur_per_mwh": indifference_price,
            "price_difference_from_indifference_eur_per_mwh": (
                TEST_LAMBDA_PPA - indifference_price
            ),
            "merchant_objective": merchant_objective,
            "predicted_objective_value": predicted_objective,
            "predicted_objective_difference_from_merchant": predicted_difference,
            "observed_objective_difference_from_merchant": observed_difference,
            "objective_prediction_error": prediction_error,
            "only_s_ppa_fixed": True,
            "stage_2_and_3_decisions_reoptimized": True,
            "below_merchant_objective": observed_difference < 0.0,
            "translation_identity_validated": (
                abs(prediction_error) <= OBJECTIVE_TOLERANCE_EUR
            ),
            "objective_tolerance_eur": OBJECTIVE_TOLERANCE_EUR,
        }
    )
    summary["expected_profit_difference_from_merchant"] = (
        summary["expected_profit"] - float(merchant["expected_profit"])
    )
    summary["cvar_difference_from_merchant"] = (
        summary["cvar_profit"] - float(merchant["cvar_profit"])
    )

    output = write_summary(OUTPUT_PATH, summary)

    print("\n=== BELOW-INDIFFERENCE VALIDATION ===")
    print(f"Fixed S_PPA: {capacity:.10f} MW")
    print(f"Indifference price: {indifference_price:.10f} EUR/MWh")
    print(f"Test price: {TEST_LAMBDA_PPA:.10f} EUR/MWh")
    print(f"Merchant objective: {merchant_objective:.6f} EUR")
    print(f"Predicted objective: {predicted_objective:.6f} EUR")
    print(f"Observed objective: {summary['objective_value']:.6f} EUR")
    print(f"Observed difference from merchant: {observed_difference:.6f} EUR")
    print(f"Prediction error: {prediction_error:.10f} EUR")
    print(f"Summary: {output}")

    if observed_difference >= 0.0:
        raise ValueError(
            "Below-indifference validation failed: the optimized fixed contract "
            "did not fall below the merchant objective."
        )
    if abs(prediction_error) > OBJECTIVE_TOLERANCE_EUR:
        raise ValueError(
            "Below-indifference validation failed: the solved objective differs "
            f"from the analytical prediction by {prediction_error:.6f} EUR."
        )


if __name__ == "__main__":
    main()
