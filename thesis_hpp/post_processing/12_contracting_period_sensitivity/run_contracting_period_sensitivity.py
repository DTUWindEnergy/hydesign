"""Compare annual, seasonal, and monthly first-stage PPA capacities."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import pandas as pd
import pyomo.environ as pyo


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
INPUTS_ROOT = PROJECT_ROOT / "inputs"
GENERATED_INPUTS = INPUTS_ROOT / "Input_files_created"
OUTPUT_ROOT = HERE / "outputs"
RESULTS_ROOT = OUTPUT_ROOT / "results"
ANNUAL_BASELINE_PATH = (
    HERE.parent
    / "07_settlement_period_effect"
    / "outputs"
    / "settlement_period_analysis.csv"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from config import (  # noqa: E402
    CONTRACTING_PERIODS,
    C_PPA,
    DELIVERY_PROFILE,
    DEVIATION_SETTLEMENT_PERIOD,
    E0,
    E_MAX,
    LAMBDA_PPA,
    P_B_MAX,
    REUSE_EXISTING_ANNUAL_RESULT,
    SDP_LABEL,
    SKIP_EXISTING_RESULTS,
    SOLVER_NAME,
    SOLVER_THREADS,
    STAGE2_WIND_METHOD,
    WEATHER_YEARS,
    XI_WIND_FILES,
)
from prepare_inputs_from_btc_2000 import prepare_inputs  # noqa: E402
from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


# BTC's synthetic year runs from 2 January to 30 December (364 days).
MONTH_NAMES = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
MONTH_DAYS = (30, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 30)
MONTH_TO_SEASON = {
    1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring",
    5: "Spring", 6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn", 12: "Winter",
}
SEASON_ORDER = ("Winter", "Spring", "Summer", "Autumn")


def month_by_hour(t_values: list[int]) -> dict[int, int]:
    if len(t_values) != 8_736:
        raise ValueError("Contracting-period sensitivity requires 8,736 hours.")
    result = {}
    offset = 0
    for month, days in enumerate(MONTH_DAYS, start=1):
        end = offset + 24 * days
        for t in t_values[offset:end]:
            result[t] = month
        offset = end
    if offset != len(t_values):
        raise AssertionError("Month definitions do not cover the full horizon.")
    return result


def contracting_period_map(
    t_values: list[int],
    contracting_period: str,
) -> tuple[list[str], dict[int, str], dict[str, list[int]]]:
    """Return ordered labels, hour-to-period mapping, and months per period."""
    month_of_t = month_by_hour(t_values)
    if contracting_period == "annual":
        labels = ["Annual"]
        period_of_t = {t: "Annual" for t in t_values}
        months = {"Annual": list(range(1, 13))}
    elif contracting_period == "seasonal":
        labels = list(SEASON_ORDER)
        period_of_t = {
            t: MONTH_TO_SEASON[month_of_t[t]] for t in t_values
        }
        months = {
            label: [m for m in range(1, 13) if MONTH_TO_SEASON[m] == label]
            for label in labels
        }
    elif contracting_period == "monthly":
        labels = list(MONTH_NAMES)
        period_of_t = {
            t: MONTH_NAMES[month_of_t[t] - 1] for t in t_values
        }
        months = {MONTH_NAMES[m - 1]: [m] for m in range(1, 13)}
    else:
        raise ValueError(
            "contracting_period must be 'annual', 'seasonal', or 'monthly'."
        )
    return labels, period_of_t, months


def apply_contracting_period(model, contracting_period: str):
    """Replace the scalar annual capacity with period-indexed capacities."""
    t_values = list(model.T)
    labels, period_of_t, months = contracting_period_map(
        t_values, contracting_period
    )

    model.ppa_commitment.deactivate()
    model.S_PPA.fix(0.0)
    model.ContractPeriod = pyo.Set(initialize=labels, ordered=True)
    model.S_PPA_contract = pyo.Var(
        model.ContractPeriod, domain=pyo.NonNegativeReals
    )

    def period_commitment_rule(m, t):
        return (
            m.P_PPA[t]
            == m.sdp[t] * m.S_PPA_contract[period_of_t[t]]
        )

    model.ppa_contracting_period_commitment = pyo.Constraint(
        model.T, rule=period_commitment_rule
    )
    model._contracting_period_name = contracting_period
    model._contract_period_of_t = period_of_t
    model._contract_period_months = months
    return labels


def scenario_probabilities(model) -> dict[tuple[int, int], float]:
    return {
        (w, x): float(
            pyo.value(model.pi_omega[w])
            * pyo.value(model.pi_xi_given_omega[w, x])
        )
        for w in model.Omega
        for x in model.Xi
    }


def expected_components(model) -> dict[str, float]:
    probabilities = scenario_probabilities(model)
    delta_t = float(pyo.value(model.delta_t))
    expected_profit = sum(
        probabilities[w, x] * float(pyo.value(model.Phi[w, x]))
        for w in model.Omega
        for x in model.Xi
    )
    expected_shortfall = sum(
        probabilities[w, x] * float(pyo.value(model.S_short[sp, w, x]))
        for sp in model.SP
        for w in model.Omega
        for x in model.Xi
    )
    contracted_energy = sum(
        float(pyo.value(model.P_PPA[t])) * delta_t for t in model.T
    )
    eta = float(pyo.value(model.eta_cvar))
    alpha = float(pyo.value(model.alpha))
    cvar_profit = -(
        eta
        + sum(
            probabilities[w, x] * float(pyo.value(model.u[w, x]))
            for w in model.Omega
            for x in model.Xi
        )
        / (1.0 - alpha)
    )
    return {
        "expected_profit_eur": expected_profit,
        "cvar_profit_eur": cvar_profit,
        "objective_value_eur": float(pyo.value(model.OBJ)),
        "annual_contracted_energy_mwh": contracted_energy,
        "average_daily_contracted_energy_mwh": contracted_energy / 364.0,
        "expected_shortfall_mwh": expected_shortfall,
        "expected_shortfall_rate_pct": (
            100.0 * expected_shortfall / contracted_energy
            if contracted_energy > 0.0
            else 0.0
        ),
    }


def capacity_rows(model, contracting_period: str) -> list[dict]:
    rows = []
    for order, label in enumerate(model.ContractPeriod, start=1):
        months = model._contract_period_months[label]
        hours = [
            t for t in model.T
            if model._contract_period_of_t[t] == label
        ]
        contracted_energy = sum(
            float(pyo.value(model.P_PPA[t]))
            * float(pyo.value(model.delta_t))
            for t in hours
        )
        rows.append(
            {
                "contracting_period": contracting_period,
                "period_order": order,
                "period_label": label,
                "months": ", ".join(MONTH_NAMES[m - 1] for m in months),
                "number_of_hours": len(hours),
                "S_PPA_mw": float(pyo.value(model.S_PPA_contract[label])),
                "contracted_energy_mwh": contracted_energy,
            }
        )
    return rows


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)
        handle.write("\n")


def reuse_existing_annual_result() -> None:
    """Copy the completed annual benchmark into this analysis's output schema."""
    if not ANNUAL_BASELINE_PATH.exists():
        raise FileNotFoundError(
            f"Existing annual benchmark not found: {ANNUAL_BASELINE_PATH}"
        )
    frame = pd.read_csv(ANNUAL_BASELINE_PATH)
    selected = frame.loc[
        (frame["sdp_label"] == SDP_LABEL)
        & frame["settlement_period"].str.lower().eq(
            DEVIATION_SETTLEMENT_PERIOD
        )
        & frame["lambda_PPA"].astype(float).eq(LAMBDA_PPA)
        & frame["c_ppa"].astype(float).eq(C_PPA)
        & frame["P_b_max"].astype(float).eq(P_B_MAX)
        & frame["E_max"].astype(float).eq(E_MAX)
    ]
    if len(selected) != 1:
        raise ValueError(
            "Expected exactly one existing annual benchmark row, found "
            f"{len(selected)} in {ANNUAL_BASELINE_PATH}."
        )
    row = selected.iloc[0]
    case_dir = RESULTS_ROOT / "annual"
    case_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "contracting_period": "annual",
        "number_of_capacity_decisions": 1,
        "delivery_profile": DELIVERY_PROFILE,
        "sdp_label": SDP_LABEL,
        "deviation_settlement_period": DEVIATION_SETTLEMENT_PERIOD,
        "lambda_PPA": LAMBDA_PPA,
        "c_ppa": C_PPA,
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "E0": E0,
        "solver_status": "existing result",
        "termination_condition": "optimal",
        "elapsed_seconds": None,
        "expected_profit_eur": float(row["expected_profit"]),
        "cvar_profit_eur": float(row["cvar_profit"]),
        "objective_value_eur": float(row["reported_objective"]),
        "annual_contracted_energy_mwh": float(
            row["annual_contracted_energy_mwh"]
        ),
        "average_daily_contracted_energy_mwh": float(
            row["annual_contracted_energy_mwh"]
        ) / 364.0,
        "expected_shortfall_mwh": float(row["expected_shortfall_mwh"]),
        "expected_shortfall_rate_pct": float(
            row["expected_shortfall_rate_pct"]
        ),
    }
    capacities = [{
        "contracting_period": "annual",
        "period_order": 1,
        "period_label": "Annual",
        "months": ", ".join(MONTH_NAMES),
        "number_of_hours": 8_736,
        "S_PPA_mw": float(row["S_PPA"]),
        "contracted_energy_mwh": float(row["annual_contracted_energy_mwh"]),
    }]
    write_json(case_dir / "summary.json", summary)
    pd.DataFrame(capacities).to_csv(
        case_dir / "contract_capacities.csv",
        index=False,
        float_format="%.15g",
    )
    print(f"Reused existing annual benchmark from {ANNUAL_BASELINE_PATH}")


def solve_case(contracting_period: str) -> tuple[dict, list[dict]]:
    data = load_all_data(GENERATED_INPUTS)
    data["config"].update(
        {
            "lambda_PPA": LAMBDA_PPA,
            "c_ppa": C_PPA,
            "P_b_max": P_B_MAX,
            "E_max": E_MAX,
            "E0": E0,
            "settlement_period": DEVIATION_SETTLEMENT_PERIOD,
        }
    )
    model = build_model(data)
    apply_contracting_period(model, contracting_period)

    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_THREADS:
        solver.options["threads"] = SOLVER_THREADS
    started = time.perf_counter()
    results = solver.solve(model, tee=True)
    elapsed_seconds = time.perf_counter() - started
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            f"Solver failed for {contracting_period}: "
            f"status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )

    capacities = capacity_rows(model, contracting_period)
    summary = {
        "contracting_period": contracting_period,
        "number_of_capacity_decisions": len(capacities),
        "delivery_profile": DELIVERY_PROFILE,
        "sdp_label": SDP_LABEL,
        "deviation_settlement_period": DEVIATION_SETTLEMENT_PERIOD,
        "lambda_PPA": LAMBDA_PPA,
        "c_ppa": C_PPA,
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "E0": E0,
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "elapsed_seconds": elapsed_seconds,
        **expected_components(model),
    }
    return summary, capacities


def collect_outputs() -> None:
    summaries = []
    capacities = []
    for contracting_period in CONTRACTING_PERIODS:
        case_dir = RESULTS_ROOT / contracting_period
        summary_path = case_dir / "summary.json"
        capacity_path = case_dir / "contract_capacities.csv"
        if not summary_path.exists() or not capacity_path.exists():
            continue
        with summary_path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
        capacities.extend(pd.read_csv(capacity_path).to_dict("records"))

    order = {name: index for index, name in enumerate(CONTRACTING_PERIODS)}
    summaries.sort(key=lambda row: order[row["contracting_period"]])
    annual_rows = [
        row for row in summaries if row["contracting_period"] == "annual"
    ]
    if annual_rows:
        annual = annual_rows[0]
        comparison_metrics = (
            "expected_profit_eur",
            "objective_value_eur",
            "annual_contracted_energy_mwh",
            "expected_shortfall_mwh",
            "expected_shortfall_rate_pct",
        )
        for row in summaries:
            for metric in comparison_metrics:
                row[f"{metric}_change_vs_annual"] = (
                    float(row[metric]) - float(annual[metric])
                )
            row["expected_profit_pct_change_vs_annual"] = (
                100.0
                * (
                    float(row["expected_profit_eur"])
                    - float(annual["expected_profit_eur"])
                )
                / abs(float(annual["expected_profit_eur"]))
            )
    capacities.sort(
        key=lambda row: (
            order[row["contracting_period"]], int(row["period_order"])
        )
    )
    pd.DataFrame(summaries).to_csv(
        OUTPUT_ROOT / "contracting_period_summary.csv",
        index=False,
        float_format="%.15g",
    )
    pd.DataFrame(capacities).to_csv(
        OUTPUT_ROOT / "contracting_period_capacities.csv",
        index=False,
        float_format="%.15g",
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    if REUSE_EXISTING_ANNUAL_RESULT:
        reuse_existing_annual_result()
    prepare_inputs(
        delivery_profile_name=DELIVERY_PROFILE,
        weather_years=list(WEATHER_YEARS),
        xi_wind_files=list(XI_WIND_FILES),
        stage2_wind_method=STAGE2_WIND_METHOD,
    )

    for contracting_period in CONTRACTING_PERIODS:
        case_dir = RESULTS_ROOT / contracting_period
        summary_path = case_dir / "summary.json"
        capacity_path = case_dir / "contract_capacities.csv"
        if (
            SKIP_EXISTING_RESULTS
            and summary_path.exists()
            and capacity_path.exists()
        ):
            print(f"Skipping completed {contracting_period} result.")
            continue

        print(f"\n=== {contracting_period.upper()} CONTRACTING PERIOD ===")
        summary, capacities = solve_case(contracting_period)
        case_dir.mkdir(parents=True, exist_ok=True)
        write_json(summary_path, summary)
        pd.DataFrame(capacities).to_csv(
            capacity_path, index=False, float_format="%.15g"
        )
        print(f"Expected profit: {summary['expected_profit_eur']:,.2f} EUR/year")
        print(f"Objective: {summary['objective_value_eur']:,.2f} EUR/year")
        print(f"Capacities: {[row['S_PPA_mw'] for row in capacities]}")

    collect_outputs()
    print(f"\nCombined outputs saved to {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
