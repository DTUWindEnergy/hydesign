"""Directly validate one settlement-period indifference-price point."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ANALYSIS_DIR = HERE.parent
PROJECT_ROOT = ANALYSIS_DIR.parent
RESULTS_DIR = ANALYSIS_DIR / "results"
CURVE_PATH = (
    RESULTS_DIR
    / "settlement_period_comparison"
    / "indifference_prices_by_settlement_period.csv"
)
MERCHANT_PATH = RESULTS_DIR / "merchant" / "summary.json"

sys.path.insert(0, str(ANALYSIS_DIR))
from common import base_summary, solve_fixed_capacity  # noqa: E402

sys.path.insert(0, str(HERE))
from settings import (  # noqa: E402
    VALIDATION_C_PPA,
    VALIDATION_REFERENCE_PPA_PRICE,
    VALIDATION_SETTLEMENT_PERIOD,
)


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}.")
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    if not CURVE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {CURVE_PATH}. Run calculate_and_plot.py first."
        )

    curve = pd.read_csv(CURVE_PATH)
    selected = curve.loc[
        (curve["settlement_period"].str.lower() == VALIDATION_SETTLEMENT_PERIOD)
        & np.isclose(curve["c_ppa"], VALIDATION_C_PPA)
        & np.isclose(curve["lambda_PPA"], VALIDATION_REFERENCE_PPA_PRICE)
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one validation point, found {len(selected)}.")

    point = selected.iloc[0]
    capacity = float(point["S_PPA"])
    indifference_price = float(point["indifference_price_eur_per_mwh"])
    merchant = read_json(MERCHANT_PATH)

    model, results, elapsed_seconds, cfg = solve_fixed_capacity(
        indifference_price,
        capacity,
        c_ppa=VALIDATION_C_PPA,
        settlement_period=VALIDATION_SETTLEMENT_PERIOD,
    )
    summary = base_summary(model, results, elapsed_seconds, cfg)
    summary.update(
        {
            "run_type": "settlement_period_indifference_validation",
            "reference_lambda_ppa": VALIDATION_REFERENCE_PPA_PRICE,
            "indifference_price_eur_per_mwh": indifference_price,
            "merchant_objective": float(merchant["objective_value"]),
        }
    )
    summary["objective_difference_from_merchant"] = (
        summary["objective_value"] - summary["merchant_objective"]
    )

    output = (
        RESULTS_DIR
        / "settlement_period_comparison"
        / "validation"
        / f"{VALIDATION_SETTLEMENT_PERIOD}_c_ppa_{VALIDATION_C_PPA:.0f}"
        / f"reference_lambda_{VALIDATION_REFERENCE_PPA_PRICE:.0f}"
        / "summary.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    print(f"Fixed S_PPA: {capacity:.10f} MW")
    print(f"Indifference price: {indifference_price:.10f} EUR/MWh")
    print(f"Merchant objective: {summary['merchant_objective']:.10f} EUR")
    print(f"Validation objective: {summary['objective_value']:.10f} EUR")
    print(
        "Difference from merchant: "
        f"{summary['objective_difference_from_merchant']:.10f} EUR"
    )
    print(f"Summary: {output}")


if __name__ == "__main__":
    main()
