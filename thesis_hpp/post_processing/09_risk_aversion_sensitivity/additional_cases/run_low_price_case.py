"""Solve the low-price PPA case across the selected beta values."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pandas as pd


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[2]
INPUTS_ROOT = PROJECT_ROOT / "inputs"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from prepare_inputs_from_btc_2000 import prepare_inputs
from run_additional_cases import RESULTS_ROOT, beta_slug, solve_case, write_json
from settings import (
    DELIVERY_PROFILE,
    STAGE2_WIND_METHOD,
    WEATHER_YEARS,
    XI_WIND_FILES,
)


OUTPUT_ROOT = HERE / "outputs" / "low_price_case"
BETAS = (0.0, 0.1, 0.5, 0.8)
CASE = {
    "case_id": "low_price",
    "label": "Low-price PPA",
    "lambda_ppa": 40.0,
    "c_ppa": 120.0,
    "optimize_s_ppa": True,
    "fixed_s_ppa": None,
}


def result_path(beta: float) -> Path:
    return OUTPUT_ROOT / beta_slug(beta) / "summary.json"


def sync_scenario_profits(beta: float) -> Path:
    source = RESULTS_ROOT / CASE["case_id"] / beta_slug(beta) / "scenario_profits.csv"
    destination = OUTPUT_ROOT / beta_slug(beta) / "scenario_profits.csv"
    if not source.exists() or source.stat().st_size == 0:
        raise FileNotFoundError(f"Missing populated scenario-profit source: {source}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as source_handle, destination.open("wb") as output_handle:
        shutil.copyfileobj(source_handle, output_handle)

    if destination.stat().st_size != source.stat().st_size:
        raise IOError(
            f"Incomplete scenario-profit copy: {source} -> {destination}"
        )
    return destination


def collect_results() -> Path:
    rows = []
    for beta in BETAS:
        path = result_path(beta)
        if not path.exists():
            raise FileNotFoundError(f"Missing completed low-price result: {path}")
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    rows.sort(key=lambda row: float(row["beta"]))
    output = OUTPUT_ROOT / "low_price_beta_sensitivity.csv"
    pd.DataFrame(rows).to_csv(output, index=False, float_format="%.12g")
    return output


def main() -> None:
    missing_betas = [beta for beta in BETAS if not result_path(beta).exists()]
    if missing_betas:
        prepare_inputs(
            delivery_profile_name=DELIVERY_PROFILE,
            weather_years=list(WEATHER_YEARS),
            xi_wind_files=list(XI_WIND_FILES),
            stage2_wind_method=STAGE2_WIND_METHOD,
        )

    for beta in BETAS:
        output = result_path(beta)
        if output.exists():
            print(f"Skipping completed low-price case, beta={beta:g}: {output}")
            print(f"Scenario profits: {sync_scenario_profits(beta)}")
            continue
        print(f"\n=== LOW-PRICE PPA: BETA={beta:g} ===")
        row = solve_case(CASE, beta)
        write_json(output, row)
        scenario_output = sync_scenario_profits(beta)
        print(f"S_PPA: {row['S_PPA']:.10f} MW")
        print(f"Expected profit: {row['expected_profit']:.6f} EUR")
        print(f"CVaR: {row['cvar_profit']:.6f} EUR")
        print(f"Summary: {output}")
        print(f"Scenario profits: {scenario_output}")

    print(f"\nLow-price comparison: {collect_results()}")


if __name__ == "__main__":
    main()
