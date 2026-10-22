"""Solve six targeted battery and settlement-period interaction cases."""

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
OUTPUT_ROOT = HERE / "outputs" / "results"
COMBINED_OUTPUT = HERE / "outputs" / "battery_settlement_interaction_summary.csv"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from prepare_inputs_from_btc_2000 import prepare_inputs  # noqa: E402
from settings import (  # noqa: E402
    BATTERY_CASES_TO_SOLVE,
    C_PPA,
    DELIVERY_PROFILE,
    LAMBDA_PPA,
    SDP_LABEL,
    SETTLEMENT_PERIODS,
    SKIP_COMPLETED_CASES,
    SOLVER_NAME,
    SOLVER_THREADS,
    STAGE2_WIND_METHOD,
    WEATHER_YEARS,
    XI_WIND_FILES,
)
from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


def battery_slug(power: float, energy: float) -> str:
    return f"battery_p_{power:g}_e_{energy:g}"


def result_path(period: str, power: float, energy: float) -> Path:
    return OUTPUT_ROOT / f"settlement_{period}" / battery_slug(power, energy)


def scenario_probabilities(model) -> dict[tuple[int, int], float]:
    return {
        (w, x): float(
            pyo.value(model.pi_omega[w])
            * pyo.value(model.pi_xi_given_omega[w, x])
        )
        for w in model.Omega
        for x in model.Xi
    }


def expected_results(model) -> dict[str, float]:
    """Calculate expected profit components and physical indicators."""
    probabilities = scenario_probabilities(model)
    delta_t = float(pyo.value(model.delta_t))
    c_imb = float(pyo.value(model.c_imb))
    c_ppa = float(pyo.value(model.c_ppa))
    degradation_cost = float(pyo.value(model.degradation_cost))
    components = {
        name: 0.0
        for name in (
            "ppa_revenue",
            "spot_revenue",
            "imbalance_settlement",
            "imbalance_penalty",
            "mfrr_up",
            "mfrr_down",
            "ppa_shortfall_cost",
            "battery_degradation",
        )
    }
    expected_shortfall = 0.0
    expected_throughput = 0.0

    for w in model.Omega:
        for x in model.Xi:
            probability = probabilities[w, x]
            scenario = {name: 0.0 for name in components}
            for t in model.T:
                scenario["ppa_revenue"] += (
                    pyo.value(model.lambda_PPA)
                    * pyo.value(model.P_PPA[t])
                    * delta_t
                )
                scenario["spot_revenue"] += (
                    pyo.value(model.lambda_spot[t, w])
                    * pyo.value(model.P_DA[t, w])
                    * delta_t
                )
                imbalance = pyo.value(model.delta[t, w, x])
                scenario["imbalance_settlement"] += (
                    pyo.value(model.lambda_imb[t, w, x])
                    * imbalance
                    * delta_t
                )
                scenario["imbalance_penalty"] -= (
                    c_imb
                    * (
                        pyo.value(model.delta_pos[t, w, x])
                        + pyo.value(model.delta_neg[t, w, x])
                    )
                    * delta_t
                )
                scenario["mfrr_up"] += (
                    pyo.value(model.a_up[t, w, x])
                    * pyo.value(model.lambda_mfrr_up[t, w, x])
                    * delta_t
                )
                scenario["mfrr_down"] -= (
                    pyo.value(model.a_down[t, w, x])
                    * pyo.value(model.lambda_mfrr_down[t, w, x])
                    * delta_t
                )
                throughput = (
                    pyo.value(model.P_dis_rt[t, w, x])
                    + pyo.value(model.P_cha_rt[t, w, x])
                ) * delta_t
                scenario["battery_degradation"] -= degradation_cost * throughput
                expected_throughput += probability * throughput

            shortfall = sum(
                pyo.value(model.S_short[sp, w, x]) for sp in model.SP
            )
            scenario["ppa_shortfall_cost"] = -c_ppa * shortfall
            expected_shortfall += probability * shortfall
            for name, value in scenario.items():
                components[name] += probability * value

    expected_profit = sum(components.values())
    alpha = float(pyo.value(model.alpha))
    eta = float(pyo.value(model.eta_cvar))
    cvar_profit = -(
        eta
        + sum(
            probabilities[w, x] * pyo.value(model.u[w, x])
            for w in model.Omega
            for x in model.Xi
        )
        / (1.0 - alpha)
    )
    beta = float(pyo.value(model.beta))
    reconstructed_objective = (
        (1.0 - beta) * expected_profit + beta * cvar_profit
    )
    reported_objective = float(pyo.value(model.OBJ))
    contracted_energy = sum(
        pyo.value(model.P_PPA[t]) * delta_t for t in model.T
    )
    days = len(model.T) * delta_t / 24.0

    return {
        **components,
        "expected_profit": expected_profit,
        "cvar_profit": cvar_profit,
        "reconstructed_objective": reconstructed_objective,
        "reported_objective": reported_objective,
        "objective_audit_difference": reconstructed_objective - reported_objective,
        "annual_contracted_energy_mwh": contracted_energy,
        "contracted_energy_mwh_per_day": contracted_energy / days,
        "expected_shortfall_mwh": expected_shortfall,
        "expected_shortfall_rate_pct": (
            100.0 * expected_shortfall / contracted_energy
            if contracted_energy > 0.0
            else 0.0
        ),
        "expected_battery_throughput_mwh": expected_throughput,
    }


def solve_case(period: str, power: float, energy: float, initial_energy: float) -> dict:
    started = time.perf_counter()
    data = load_all_data(GENERATED_INPUTS)
    data["config"].update(
        {
            "lambda_PPA": LAMBDA_PPA,
            "c_ppa": C_PPA,
            "P_b_max": power,
            "E_max": energy,
            "E0": initial_energy,
            "settlement_period": period,
        }
    )
    model = build_model(data)
    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_THREADS:
        solver.options["threads"] = SOLVER_THREADS
    results = solver.solve(model, tee=True)
    elapsed = time.perf_counter() - started
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            f"Solve failed for {period}, {power:g} MW/{energy:g} MWh: "
            f"{results.solver.termination_condition}"
        )

    summary = {
        "run_type": "battery_settlement_interaction",
        "sdp_label": SDP_LABEL,
        "delivery_profile": DELIVERY_PROFILE,
        "lambda_PPA": LAMBDA_PPA,
        "c_ppa": C_PPA,
        "settlement_period": period,
        "P_b_max": power,
        "E_max": energy,
        "E0": initial_energy,
        "S_PPA": float(pyo.value(model.S_PPA)),
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "elapsed_seconds": elapsed,
        **expected_results(model),
    }
    output = result_path(period, power, energy)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(
        output / "summary.csv", index=False, float_format="%.15g"
    )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    pd.DataFrame(
        [
            {"component": name, "expected_value_eur_per_year": summary[name]}
            for name in (
                "ppa_revenue",
                "spot_revenue",
                "imbalance_settlement",
                "imbalance_penalty",
                "mfrr_up",
                "mfrr_down",
                "ppa_shortfall_cost",
                "battery_degradation",
                "expected_profit",
            )
        ]
    ).to_csv(output / "profit_components.csv", index=False, float_format="%.15g")
    return summary


def collect_summaries() -> Path:
    rows = []
    for path in OUTPUT_ROOT.glob("settlement_*/battery_*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    expected_cases = len(SETTLEMENT_PERIODS) * len(BATTERY_CASES_TO_SOLVE)
    if len(rows) != expected_cases:
        raise ValueError(f"Expected {expected_cases} completed cases, found {len(rows)}.")
    rows.sort(
        key=lambda row: (
            SETTLEMENT_PERIODS.index(row["settlement_period"]),
            row["P_b_max"],
        )
    )
    COMBINED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(COMBINED_OUTPUT, index=False, float_format="%.15g")
    return COMBINED_OUTPUT


def main() -> None:
    requested = [
        (period, power, energy, initial_energy)
        for period in SETTLEMENT_PERIODS
        for power, energy, initial_energy in BATTERY_CASES_TO_SOLVE
    ]
    missing = [
        case
        for case in requested
        if not (
            SKIP_COMPLETED_CASES
            and (result_path(case[0], case[1], case[2]) / "summary.json").exists()
        )
    ]
    if missing:
        prepare_inputs(
            delivery_profile_name=DELIVERY_PROFILE,
            weather_years=list(WEATHER_YEARS),
            xi_wind_files=list(XI_WIND_FILES),
            stage2_wind_method=STAGE2_WIND_METHOD,
        )
    for period, power, energy, initial_energy in requested:
        output = result_path(period, power, energy) / "summary.json"
        if SKIP_COMPLETED_CASES and output.exists():
            print(f"Skipping completed case: {output}")
            continue
        print(f"\n=== {period.upper()}, {power:g} MW/{energy:g} MWh ===")
        summary = solve_case(period, power, energy, initial_energy)
        print(f"S_PPA: {summary['S_PPA']:.6f} MW")
        print(f"Expected profit: {summary['expected_profit']:,.2f} EUR/year")
        print(f"Shortfall rate: {summary['expected_shortfall_rate_pct']:.6f}%")
        print(f"Saved: {output}")
    print(f"\nCombined summary: {collect_summaries()}")


if __name__ == "__main__":
    main()
