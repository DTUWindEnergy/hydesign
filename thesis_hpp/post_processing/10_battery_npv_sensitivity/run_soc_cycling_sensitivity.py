"""Solve one benchmark case with a 72-hour battery SoC cycling condition."""

from __future__ import annotations

from pathlib import Path
import json
import sys
import time

import pandas as pd
import pyomo.environ as pyo


HERE = Path(__file__).resolve().parent
THESIS_ROOT = HERE.parents[1]
INPUTS_ROOT = THESIS_ROOT / "inputs"
GENERATED_INPUTS = INPUTS_ROOT / "Input_files_created"
OUTPUT_ROOT = HERE / "outputs" / "three_day_run"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(THESIS_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from config import (  # noqa: E402
    C_PPA,
    DELIVERY_PROFILE,
    LAMBDA_PPA,
    SDP_LABEL,
    SETTLEMENT_PERIOD,
    SKIP_EXISTING_THREE_DAY_RESULTS,
    SOC_CYCLE_HOURS,
    SOLVER_NAME,
    SOLVER_THREADS,
    STAGE2_WIND_METHOD,
    THREE_DAY_BATTERY_CASES,
    WEATHER_YEARS,
    XI_WIND_FILES,
)
from prepare_inputs_from_btc_2000 import prepare_inputs  # noqa: E402
from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


def apply_soc_cycling_horizon(model: pyo.ConcreteModel, hours: int) -> list[int]:
    """Replace daily SoC cycling with fixed SoC at each block start."""
    if hours <= 0:
        raise ValueError("The SoC cycling horizon must be positive.")

    time_values = list(model.T)
    if not time_values:
        raise ValueError("The model has an empty time horizon.")

    cycle_starts = time_values[::hours]
    model.soc_cycling.deactivate()
    model.SOC_CYCLE_STARTS = pyo.Set(initialize=cycle_starts, ordered=True)

    def cycle_start_rule(m, t, w, x):
        return m.E_rt[t, w, x] == m.E0

    model.soc_cycling_sensitivity = pyo.Constraint(
        model.SOC_CYCLE_STARTS,
        model.Omega,
        model.Xi,
        rule=cycle_start_rule,
    )
    model._soc_cycle_hours = hours
    return cycle_starts


def scenario_probabilities(model: pyo.ConcreteModel) -> dict[tuple[int, int], float]:
    return {
        (w, x): float(
            pyo.value(model.pi_omega[w])
            * pyo.value(model.pi_xi_given_omega[w, x])
        )
        for w in model.Omega
        for x in model.Xi
    }


def expected_results(model: pyo.ConcreteModel) -> dict[str, float]:
    """Calculate expected profit components directly from the solved model."""
    probabilities = scenario_probabilities(model)
    delta_t = float(pyo.value(model.delta_t))
    c_imb = float(pyo.value(model.c_imb))
    c_ppa = float(pyo.value(model.c_ppa))
    degradation_cost = float(pyo.value(model.degradation_cost))

    component_names = (
        "ppa_revenue",
        "spot_revenue",
        "imbalance_settlement",
        "imbalance_penalty",
        "mfrr_up",
        "mfrr_down",
        "ppa_shortfall_cost",
        "battery_degradation",
    )
    expected = {name: 0.0 for name in component_names}
    expected_shortfall = 0.0
    expected_throughput = 0.0

    for w in model.Omega:
        for x in model.Xi:
            probability = probabilities[w, x]
            scenario = {name: 0.0 for name in component_names}

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

            for name in component_names:
                expected[name] += probability * scenario[name]

    expected_profit = sum(expected.values())
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

    contracted_energy = sum(
        pyo.value(model.P_PPA[t]) * delta_t for t in model.T
    )
    number_of_days = len(model.T) * delta_t / 24.0
    result = dict(expected)
    result.update(
        {
            "expected_profit": expected_profit,
            "cvar_profit": cvar_profit,
            "reconstructed_objective": reconstructed_objective,
            "reported_objective": float(pyo.value(model.OBJ)),
            "objective_audit_difference": (
                reconstructed_objective - float(pyo.value(model.OBJ))
            ),
            "annual_contracted_energy_mwh": contracted_energy,
            "contracted_energy_mwh_per_day": contracted_energy / number_of_days,
            "expected_shortfall_mwh": expected_shortfall,
            "expected_shortfall_rate_pct": (
                100.0 * expected_shortfall / contracted_energy
                if contracted_energy > 0.0
                else 0.0
            ),
            "expected_battery_throughput_mwh": expected_throughput,
        }
    )
    return result


def build_soc_audit(
    model: pyo.ConcreteModel,
    cycle_starts: list[int],
) -> pd.DataFrame:
    """Return one compact row per scenario and SoC cycle block."""
    time_values = list(model.T)
    position = {t: index for index, t in enumerate(time_values)}
    final_t = time_values[-1]
    rows = []

    for block, start_t in enumerate(cycle_starts, start=1):
        start_index = position[start_t]
        end_index = min(start_index + SOC_CYCLE_HOURS - 1, len(time_values) - 1)
        end_t = time_values[end_index]
        block_hours = end_index - start_index + 1
        for w in model.Omega:
            for x in model.Xi:
                start_soc = pyo.value(model.E_rt[start_t, w, x])
                end_soc = (
                    pyo.value(model.E_rt[end_t, w, x])
                    - pyo.value(model.P_dis_rt[end_t, w, x])
                    / pyo.value(model.eta_dis)
                    + pyo.value(model.P_cha_rt[end_t, w, x])
                    * pyo.value(model.eta_cha)
                )
                rows.append(
                    {
                        "block": block,
                        "start_t": start_t,
                        "end_t": end_t,
                        "number_of_hours": block_hours,
                        "omega": w,
                        "xi": x,
                        "start_soc_mwh": start_soc,
                        "end_soc_mwh": end_soc,
                        "start_residual_mwh": start_soc - pyo.value(model.E0),
                        "end_residual_mwh": end_soc - pyo.value(model.E0),
                        "end_fixed_by_year_end_constraint": end_t == final_t,
                    }
                )
    return pd.DataFrame(rows)


def battery_case_slug(power: float, energy: float) -> str:
    return f"battery_p_{power:g}_e_{energy:g}"


def completed_summary_path(power: float, energy: float) -> Path | None:
    """Return an existing matching summary, including the original 20 MW run."""
    case_summary = (
        OUTPUT_ROOT
        / battery_case_slug(power, energy)
        / "three_day_case_summary.csv"
    )
    candidates = [case_summary]
    if power == 20.0 and energy == 60.0:
        candidates.append(OUTPUT_ROOT / "three_day_case_summary.csv")

    for path in candidates:
        if not path.exists():
            continue
        existing = pd.read_csv(path)
        if (
            len(existing) == 1
            and float(existing.iloc[0]["P_b_max"]) == power
            and float(existing.iloc[0]["E_max"]) == energy
            and float(existing.iloc[0]["soc_cycle_hours"]) == SOC_CYCLE_HOURS
        ):
            return path
    return None


def solve_battery_case(power: float, energy: float, initial_energy: float) -> None:
    run_start = time.perf_counter()
    case_output = OUTPUT_ROOT / battery_case_slug(power, energy)
    case_output.mkdir(parents=True, exist_ok=True)

    data = load_all_data(GENERATED_INPUTS)
    data["config"].update(
        {
            "lambda_PPA": LAMBDA_PPA,
            "c_ppa": C_PPA,
            "P_b_max": power,
            "E_max": energy,
            "E0": initial_energy,
            "settlement_period": SETTLEMENT_PERIOD,
        }
    )
    model = build_model(data)
    cycle_starts = apply_soc_cycling_horizon(model, SOC_CYCLE_HOURS)

    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_THREADS:
        solver.options["threads"] = SOLVER_THREADS
    results = solver.solve(model, tee=True)
    elapsed_seconds = time.perf_counter() - run_start

    termination = str(results.solver.termination_condition)
    if termination.lower() not in {"optimal", "feasible"}:
        raise RuntimeError(
            f"Solver did not return a usable solution: {termination}"
        )

    summary = {
        "run_type": "soc_cycling_sensitivity",
        "sdp_label": SDP_LABEL,
        "delivery_profile": DELIVERY_PROFILE,
        "settlement_period": SETTLEMENT_PERIOD,
        "lambda_PPA": LAMBDA_PPA,
        "c_ppa": C_PPA,
        "P_b_max": power,
        "E_max": energy,
        "E0": initial_energy,
        "soc_cycle_hours": SOC_CYCLE_HOURS,
        "soc_cycle_blocks": len(cycle_starts),
        "S_PPA": float(pyo.value(model.S_PPA)),
        "solver_status": str(results.solver.status),
        "termination_condition": termination,
        "elapsed_seconds": elapsed_seconds,
        **expected_results(model),
    }
    pd.DataFrame([summary]).to_csv(
        case_output / "three_day_case_summary.csv",
        index=False,
        float_format="%.15g",
    )
    with (case_output / "three_day_case_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    component_columns = [
        "ppa_revenue",
        "spot_revenue",
        "imbalance_settlement",
        "imbalance_penalty",
        "mfrr_up",
        "mfrr_down",
        "ppa_shortfall_cost",
        "battery_degradation",
        "expected_profit",
    ]
    pd.DataFrame(
        [
            {"component": name, "expected_value_eur_per_year": summary[name]}
            for name in component_columns
        ]
    ).to_csv(
        case_output / "profit_components.csv",
        index=False,
        float_format="%.15g",
    )

    audit = build_soc_audit(model, cycle_starts)
    audit.to_csv(
        case_output / "soc_cycle_audit.csv",
        index=False,
        float_format="%.15g",
    )
    max_start_residual = audit["start_residual_mwh"].abs().max()
    max_end_residual = audit["end_residual_mwh"].abs().max()

    print("\n=== SOC CYCLING SENSITIVITY ===")
    print(f"Battery: {power:g} MW/{energy:g} MWh")
    print(f"S_PPA: {summary['S_PPA']:.6f} MW")
    print(f"Expected profit: {summary['expected_profit']:,.2f} EUR/year")
    print(f"Objective: {summary['reported_objective']:,.2f} EUR/year")
    print(f"Maximum block-start SoC residual: {max_start_residual:.3e} MWh")
    print(f"Maximum block-end SoC residual: {max_end_residual:.3e} MWh")
    print(f"Saved results to: {case_output}")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    missing_cases = []
    for power, energy, initial_energy in THREE_DAY_BATTERY_CASES:
        existing = completed_summary_path(power, energy)
        if SKIP_EXISTING_THREE_DAY_RESULTS and existing is not None:
            print(
                f"Reusing completed {power:g} MW/{energy:g} MWh result: "
                f"{existing}"
            )
        else:
            missing_cases.append((power, energy, initial_energy))

    if not missing_cases:
        print("All requested three-day battery cases are already complete.")
        return

    prepare_inputs(
        delivery_profile_name=DELIVERY_PROFILE,
        weather_years=list(WEATHER_YEARS),
        xi_wind_files=list(XI_WIND_FILES),
        stage2_wind_method=STAGE2_WIND_METHOD,
    )
    for battery_case in missing_cases:
        solve_battery_case(*battery_case)


if __name__ == "__main__":
    main()
