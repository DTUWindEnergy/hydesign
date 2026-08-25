from __future__ import annotations

import json

from collect_indifference_validations import main as collect_validation_summaries
from common import (
    RESULTS_DIR,
    base_summary,
    solve_fixed_capacity,
    write_summary,
)


def read_json(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validation_directory_name(reference_lambda: float) -> str:
    label = f"{reference_lambda:g}".replace(".", "p")
    return f"indifference_validations/reference_lambda_{label}"


def main() -> None:
    curve = read_json(RESULTS_DIR / "indifference_curve_summary.json")
    merchant = read_json(RESULTS_DIR / "merchant" / "summary.json")
    required_merchant_fields = {"objective_value", "expected_profit", "cvar_profit"}
    missing = required_merchant_fields.difference(merchant)
    if missing:
        raise ValueError(
            "The merchant summary predates component reporting and is missing "
            f"{sorted(missing)}. Rerun economic_value_analysis/run_merchant.py."
        )

    if len(curve) != 4:
        raise ValueError(
            f"Expected four indifference-curve points, found {len(curve)}."
        )

    for point in curve:
        reference_lambda = float(point["reference_lambda_ppa"])
        indifference_price = float(
            point["risk_adjusted_indifference_price_eur_per_mwh"]
        )
        capacity = float(point["s_ppa_fixed"])
        model, results, elapsed_seconds, cfg = solve_fixed_capacity(
            indifference_price, capacity
        )
        summary = base_summary(model, results, elapsed_seconds, cfg)
        summary.update(
            {
                "run_type": "fixed_capacity_risk_adjusted_indifference_validation",
                "reference_lambda_ppa": reference_lambda,
                "risk_adjusted_indifference_price_eur_per_mwh": indifference_price,
                "merchant_objective": float(merchant["objective_value"]),
                "merchant_expected_profit": float(merchant["expected_profit"]),
                "merchant_cvar_profit": float(merchant["cvar_profit"]),
            }
        )
        summary["objective_difference_from_merchant"] = (
            summary["objective_value"] - summary["merchant_objective"]
        )
        summary["expected_profit_difference_from_merchant"] = (
            summary["expected_profit"] - summary["merchant_expected_profit"]
        )
        summary["cvar_difference_from_merchant"] = (
            summary["cvar_profit"] - summary["merchant_cvar_profit"]
        )
        energy = float(summary["contracted_energy_mwh"])
        summary["expected_profit_indifference_price_eur_per_mwh"] = (
            indifference_price
            - summary["expected_profit_difference_from_merchant"] / energy
        )

        relative_path = (
            f"{validation_directory_name(reference_lambda)}/summary.json"
        )
        output_path = write_summary(relative_path, summary)
        print("\n=== INDIFFERENCE VALIDATION ===")
        print(f"Reference PPA price: {reference_lambda:.0f} EUR/MWh")
        print(f"Fixed S_PPA: {capacity:.10f} MW")
        print(f"Risk-adjusted indifference price: {indifference_price:.10f} EUR/MWh")
        print(
            "Objective difference from merchant: "
            f"{summary['objective_difference_from_merchant']:.10f} EUR"
        )
        print(
            "Expected-profit difference from merchant: "
            f"{summary['expected_profit_difference_from_merchant']:.10f} EUR"
        )
        print(f"Summary: {output_path}")

    collect_validation_summaries()


if __name__ == "__main__":
    main()
