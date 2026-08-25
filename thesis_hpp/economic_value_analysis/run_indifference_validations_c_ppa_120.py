from __future__ import annotations

import csv
import json
from pathlib import Path

from common import (
    ANALYSIS_DIR,
    RESULTS_DIR,
    base_summary,
    solve_fixed_capacity,
    write_summary,
)


CONFIG_PATH = ANALYSIS_DIR / "analysis_config_c_ppa_120.json"
OUTPUT_ROOT = RESULTS_DIR / "c_ppa_120"


def read_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def reference_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def calculate_curve(config: dict, merchant: dict) -> list[dict]:
    merchant_objective = float(merchant["objective_value"])
    rows = []
    for point in config["validation_points"]:
        reference_price = float(point["lambda_ppa_reference"])
        capacity = float(point["s_ppa_fixed"])
        energy = float(point["contracted_energy_mwh"])
        reference_objective = float(point["reference_objective"])
        indifference_price = reference_price + (
            merchant_objective - reference_objective
        ) / energy
        rows.append(
            {
                "c_ppa": float(config["c_ppa"]),
                "reference_lambda_ppa": reference_price,
                "s_ppa_fixed": capacity,
                "contracted_energy_mwh": energy,
                "reference_objective": reference_objective,
                "merchant_objective": merchant_objective,
                "risk_adjusted_indifference_price_eur_per_mwh": (
                    indifference_price
                ),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_curve(rows: list[dict]) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT_ROOT / "indifference_curve_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(rows, handle, indent=2)
        handle.write("\n")
    write_rows(OUTPUT_ROOT / "indifference_curve_summary.csv", rows)


def collect_summaries() -> Path:
    root = OUTPUT_ROOT / "indifference_validations"
    summaries = []
    for path in sorted(root.glob("reference_lambda_*/summary.json")):
        summaries.append(read_json(path))
    if not summaries:
        raise FileNotFoundError(f"No completed validations below {root}.")
    summaries.sort(key=lambda row: float(row["reference_lambda_ppa"]))
    fieldnames = []
    for summary in summaries:
        for key in summary:
            if key not in fieldnames:
                fieldnames.append(key)
    output = OUTPUT_ROOT / "indifference_validations_summary.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return output


def main() -> None:
    config = read_json(CONFIG_PATH)
    merchant = read_json(RESULTS_DIR / "merchant" / "summary.json")
    required = {"objective_value", "expected_profit", "cvar_profit"}
    missing = required.difference(merchant)
    if missing:
        raise ValueError(
            "Merchant summary is missing "
            f"{sorted(missing)}. Rerun run_merchant.py first."
        )

    curve = calculate_curve(config, merchant)
    write_curve(curve)
    c_ppa = float(config["c_ppa"])

    for point in curve:
        reference_price = float(point["reference_lambda_ppa"])
        indifference_price = float(
            point["risk_adjusted_indifference_price_eur_per_mwh"]
        )
        capacity = float(point["s_ppa_fixed"])
        relative = (
            "c_ppa_120/indifference_validations/"
            f"reference_lambda_{reference_label(reference_price)}/summary.json"
        )
        existing_output = RESULTS_DIR / relative
        if existing_output.exists():
            print(f"Skipping completed validation: {existing_output}")
            continue
        model, results, elapsed_seconds, cfg = solve_fixed_capacity(
            indifference_price,
            capacity,
            c_ppa=c_ppa,
        )
        summary = base_summary(model, results, elapsed_seconds, cfg)
        summary.update(
            {
                "run_type": "fixed_capacity_risk_adjusted_indifference_validation",
                "reference_lambda_ppa": reference_price,
                "risk_adjusted_indifference_price_eur_per_mwh": (
                    indifference_price
                ),
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
        summary["expected_profit_indifference_price_eur_per_mwh"] = (
            indifference_price
            - summary["expected_profit_difference_from_merchant"]
            / float(summary["contracted_energy_mwh"])
        )

        output = write_summary(relative, summary)
        print("\n=== C_PPA=120 INDIFFERENCE VALIDATION ===")
        print(f"Reference PPA price: {reference_price:.0f} EUR/MWh")
        print(f"Fixed S_PPA: {capacity:.10f} MW")
        print(f"Indifference price: {indifference_price:.10f} EUR/MWh")
        print(
            "Objective difference from merchant: "
            f"{summary['objective_difference_from_merchant']:.10f} EUR"
        )
        print(f"Summary: {output}")

    combined = collect_summaries()
    print(f"\nCombined validation results: {combined}")


if __name__ == "__main__":
    main()
