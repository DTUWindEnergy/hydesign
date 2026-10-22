"""Solve the high-contract and merchant risk-aversion comparisons."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import pandas as pd
import pyomo.environ as pyo


HERE = Path(__file__).resolve().parent
RISK_DIR = HERE.parent
POST_PROCESSING_ROOT = RISK_DIR.parent
PROJECT_ROOT = POST_PROCESSING_ROOT.parent
INPUTS_ROOT = PROJECT_ROOT / "inputs"
GENERATED_INPUTS = INPUTS_ROOT / "Input_files_created"
OUTPUT_ROOT = HERE / "outputs"
RESULTS_ROOT = OUTPUT_ROOT / "results"

BASELINE_DECOMPOSITION = (
    POST_PROCESSING_ROOT
    / "04_profit_decomposition"
    / "outputs"
    / "expected_profit_decomposition_split_imbalance.csv"
)
BASELINE_SETTLEMENT = (
    POST_PROCESSING_ROOT
    / "07_settlement_period_effect"
    / "outputs"
    / "settlement_period_analysis.csv"
)
MERCHANT_BASELINE = (
    PROJECT_ROOT / "economic_value_analysis" / "results" / "merchant" / "summary.json"
)

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from prepare_inputs_from_btc_2000 import prepare_inputs  # noqa: E402
from settings import (  # noqa: E402
    ALPHA,
    BASELINE_BETA,
    CASES,
    DELIVERY_PROFILE,
    E0,
    E_MAX,
    P_B_MAX,
    SETTLEMENT_PERIOD,
    SOLVER_NAME,
    SOLVER_THREADS,
    STAGE2_WIND_METHOD,
    WEATHER_YEARS,
    XI_WIND_FILES,
)
from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


SUMMARY_FIELDS = (
    "case_id",
    "case_label",
    "beta",
    "alpha",
    "delivery_profile",
    "lambda_PPA",
    "c_ppa",
    "P_b_max",
    "E_max",
    "settlement_period",
    "S_PPA",
    "annual_contracted_energy_mwh",
    "expected_shortfall_mwh",
    "expected_shortfall_rate_pct",
    "expected_profit",
    "cvar_profit",
    "downside_gap_eur",
    "normalized_downside_gap_pct",
    "objective_value",
    "objective_reconstructed",
    "solver_status",
    "termination_condition",
    "elapsed_seconds",
    "result_source",
)


def beta_slug(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p")


def scenario_rows(model, case: dict, beta: float) -> list[dict]:
    omega_year = pd.read_csv(
        GENERATED_INPUTS / "omega_probabilities.csv",
        usecols=["omega", "weather_year"],
    ).set_index("omega")["weather_year"].to_dict()
    xi_file = pd.read_csv(
        GENERATED_INPUTS / "xi_probabilities.csv",
        usecols=["omega", "xi", "wind_file"],
    ).set_index(["omega", "xi"])["wind_file"].to_dict()
    rows = []
    for w in model.Omega:
        for x in model.Xi:
            rows.append(
                {
                    "case_id": case["case_id"],
                    "beta": float(beta),
                    "omega": int(w),
                    "weather_year": int(omega_year[int(w)]),
                    "xi": int(x),
                    "wind_file": xi_file[(int(w), int(x))],
                    "scenario_probability": (
                        float(pyo.value(model.pi_omega[w]))
                        * float(pyo.value(model.pi_xi_given_omega[w, x]))
                    ),
                    "scenario_profit": float(pyo.value(model.Phi[w, x])),
                }
            )
    rows.sort(key=lambda row: row["scenario_profit"])
    for rank, row in enumerate(rows, start=1):
        row["profit_rank"] = rank
    tail_probability = 1.0 - ALPHA
    scenario_probability = rows[0]["scenario_probability"]
    tail_count = int(round(tail_probability / scenario_probability))
    var_profit = rows[tail_count - 1]["scenario_profit"]
    tolerance = max(1e-7 * abs(var_profit), 1e-6)
    strict_rows = [
        row for row in rows if row["scenario_profit"] < var_profit - tolerance
    ]
    boundary_rows = [
        row
        for row in rows
        if abs(row["scenario_profit"] - var_profit) <= tolerance
    ]
    remaining_probability = (
        tail_probability
        - sum(row["scenario_probability"] for row in strict_rows)
    )
    boundary_probability = remaining_probability / len(boundary_rows)
    for row in rows:
        if row in strict_rows:
            row["tail_status"] = "strict_tail"
            row["cvar_tail_probability"] = row["scenario_probability"]
        elif row in boundary_rows:
            row["tail_status"] = "var_boundary"
            row["cvar_tail_probability"] = boundary_probability
        else:
            row["tail_status"] = "outside_tail"
            row["cvar_tail_probability"] = 0.0
    return rows


def expected_profit(rows: list[dict]) -> float:
    return sum(
        row["scenario_probability"] * row["scenario_profit"] for row in rows
    )


def lower_tail_cvar(rows: list[dict], alpha: float) -> float:
    losses = [-row["scenario_profit"] for row in rows]
    candidates = set(losses)
    loss_cvar = min(
        eta
        + sum(
            row["scenario_probability"] * max(loss - eta, 0.0)
            for row, loss in zip(rows, losses)
        )
        / (1.0 - alpha)
        for eta in candidates
    )
    return -loss_cvar


def annual_contracted_energy(model) -> float:
    delta_t = float(pyo.value(model.delta_t))
    return sum(float(pyo.value(model.P_PPA[t])) * delta_t for t in model.T)


def expected_shortfall(model) -> float:
    return sum(
        float(pyo.value(model.pi_omega[w]))
        * float(pyo.value(model.pi_xi_given_omega[w, x]))
        * float(pyo.value(model.S_short[sp, w, x]))
        for sp in model.SP
        for w in model.Omega
        for x in model.Xi
    )


def add_gap_metrics(row: dict) -> dict:
    expected = float(row["expected_profit"])
    cvar = float(row["cvar_profit"])
    row["downside_gap_eur"] = expected - cvar
    row["normalized_downside_gap_pct"] = (
        100.0 * (expected - cvar) / expected if expected != 0.0 else 0.0
    )
    return row


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
        handle.write("\n")


def select_high_contract_baseline() -> dict:
    decomposition = pd.read_csv(BASELINE_DECOMPOSITION)
    settlement = pd.read_csv(BASELINE_SETTLEMENT)
    mask = (
        decomposition["sdp_label"].eq("Generic Industrial")
        & decomposition["settlement_period"].str.lower().eq(SETTLEMENT_PERIOD)
        & decomposition["lambda_PPA"].astype(float).eq(100.0)
        & decomposition["c_ppa"].astype(float).eq(120.0)
        & decomposition["P_b_max"].astype(float).eq(P_B_MAX)
        & decomposition["E_max"].astype(float).eq(E_MAX)
    )
    selected_decomposition = decomposition.loc[mask]
    settlement_mask = (
        settlement["sdp_label"].eq("Generic Industrial")
        & settlement["settlement_period"].str.lower().eq(SETTLEMENT_PERIOD)
        & settlement["lambda_PPA"].astype(float).eq(100.0)
        & settlement["c_ppa"].astype(float).eq(120.0)
        & settlement["P_b_max"].astype(float).eq(P_B_MAX)
        & settlement["E_max"].astype(float).eq(E_MAX)
    )
    selected_settlement = settlement.loc[settlement_mask]
    if len(selected_decomposition) != 1 or len(selected_settlement) != 1:
        raise ValueError("Could not identify one existing high-contract beta=0.2 case.")
    d = selected_decomposition.iloc[0]
    s = selected_settlement.iloc[0]
    row = {
        "case_id": "high_contract",
        "case_label": "High-contract PPA",
        "beta": BASELINE_BETA,
        "alpha": ALPHA,
        "delivery_profile": DELIVERY_PROFILE,
        "lambda_PPA": 100.0,
        "c_ppa": 120.0,
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "settlement_period": SETTLEMENT_PERIOD,
        "S_PPA": float(d["S_PPA"]),
        "annual_contracted_energy_mwh": float(s["annual_contracted_energy_mwh"]),
        "expected_shortfall_mwh": float(s["expected_shortfall_mwh"]),
        "expected_shortfall_rate_pct": float(s["expected_shortfall_rate_pct"]),
        "expected_profit": float(d["expected_profit"]),
        "cvar_profit": float(d["cvar_profit"]),
        "objective_value": float(d["reported_objective"]),
        "objective_reconstructed": float(d["reconstructed_objective"]),
        "solver_status": "existing result",
        "termination_condition": "optimal",
        "elapsed_seconds": None,
        "result_source": "existing beta=0.2 main-model result",
    }
    return add_gap_metrics(row)


def select_merchant_baseline() -> dict:
    with MERCHANT_BASELINE.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    row = {
        "case_id": "merchant",
        "case_label": "Merchant operation",
        "beta": BASELINE_BETA,
        "alpha": ALPHA,
        "delivery_profile": DELIVERY_PROFILE,
        "lambda_PPA": 0.0,
        "c_ppa": 160.0,
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "settlement_period": SETTLEMENT_PERIOD,
        "S_PPA": 0.0,
        "annual_contracted_energy_mwh": 0.0,
        "expected_shortfall_mwh": 0.0,
        "expected_shortfall_rate_pct": 0.0,
        "expected_profit": float(source["expected_profit"]),
        "cvar_profit": float(source["cvar_profit"]),
        "objective_value": float(source["objective_value"]),
        "objective_reconstructed": float(source["objective_reconstructed"]),
        "solver_status": "existing result",
        "termination_condition": "optimal",
        "elapsed_seconds": None,
        "result_source": "existing beta=0.2 merchant result",
    }
    return add_gap_metrics(row)


def write_existing_baselines() -> None:
    baselines = {
        "high_contract": select_high_contract_baseline(),
        "merchant": select_merchant_baseline(),
    }
    for case_id, row in baselines.items():
        output = RESULTS_ROOT / case_id / beta_slug(BASELINE_BETA) / "summary.json"
        write_json(output, row)


def solve_case(case: dict, beta: float) -> dict:
    data = load_all_data(GENERATED_INPUTS)
    cfg = data["config"]
    cfg["lambda_PPA"] = float(case["lambda_ppa"])
    cfg["c_ppa"] = float(case["c_ppa"])
    cfg["P_b_max"] = P_B_MAX
    cfg["E_max"] = E_MAX
    cfg["E0"] = E0
    cfg["settlement_period"] = SETTLEMENT_PERIOD
    cfg["alpha"] = ALPHA
    cfg["beta"] = float(beta)

    model = build_model(data)
    if not case["optimize_s_ppa"]:
        model.S_PPA.fix(float(case["fixed_s_ppa"]))

    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_NAME == "cplex":
        solver.options["threads"] = SOLVER_THREADS
    started = time.perf_counter()
    results = solver.solve(model, tee=True)
    elapsed = time.perf_counter() - started
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            f"Solver failed for {case['case_id']}, beta={beta:g}: "
            f"status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )

    scenarios = scenario_rows(model, case, beta)
    expected = expected_profit(scenarios)
    cvar = lower_tail_cvar(scenarios, ALPHA)
    objective = float(pyo.value(model.OBJ))
    reconstructed = (1.0 - beta) * expected + beta * cvar
    if abs(objective - reconstructed) > max(5.0, 1e-7 * abs(objective)):
        raise ValueError(
            f"Objective reconstruction failed for {case['case_id']}, beta={beta:g}."
        )

    annual_energy = annual_contracted_energy(model)
    shortfall = expected_shortfall(model)
    result_dir = RESULTS_ROOT / case["case_id"] / beta_slug(beta)
    result_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scenarios).to_csv(
        result_dir / "scenario_profits.csv",
        index=False,
        float_format="%.12g",
    )
    row = {
        "case_id": case["case_id"],
        "case_label": case["label"],
        "beta": float(beta),
        "alpha": ALPHA,
        "delivery_profile": DELIVERY_PROFILE,
        "lambda_PPA": float(case["lambda_ppa"]),
        "c_ppa": float(case["c_ppa"]),
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "settlement_period": SETTLEMENT_PERIOD,
        "S_PPA": float(pyo.value(model.S_PPA)),
        "annual_contracted_energy_mwh": annual_energy,
        "expected_shortfall_mwh": shortfall,
        "expected_shortfall_rate_pct": (
            100.0 * shortfall / annual_energy if annual_energy > 0.0 else 0.0
        ),
        "expected_profit": expected,
        "cvar_profit": cvar,
        "objective_value": objective,
        "objective_reconstructed": reconstructed,
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "elapsed_seconds": elapsed,
        "result_source": "new compact risk-aversion solve",
    }
    return add_gap_metrics(row)


def collect_results() -> Path:
    rows = []
    for path in RESULTS_ROOT.glob("*/beta_*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    expected_rows = sum(len(case["betas"]) for case in CASES)
    if len(rows) != expected_rows:
        raise ValueError(f"Expected {expected_rows} summaries, found {len(rows)}.")
    rows.sort(key=lambda row: (row["case_id"], float(row["beta"])))
    output = OUTPUT_ROOT / "additional_risk_aversion_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    prepare_inputs(
        delivery_profile_name=DELIVERY_PROFILE,
        weather_years=list(WEATHER_YEARS),
        xi_wind_files=list(XI_WIND_FILES),
        stage2_wind_method=STAGE2_WIND_METHOD,
    )
    write_existing_baselines()

    for case in CASES:
        for beta in case["betas"]:
            output = RESULTS_ROOT / case["case_id"] / beta_slug(beta) / "summary.json"
            if output.exists():
                print(
                    f"Skipping completed case={case['case_id']}, beta={beta:g}: {output}"
                )
                continue
            print(f"\n=== CASE={case['case_id']}, BETA={beta:g} ===")
            row = solve_case(case, beta)
            write_json(output, row)
            print(f"S_PPA: {row['S_PPA']:.10f} MW")
            print(f"Expected profit: {row['expected_profit']:.6f} EUR")
            print(f"CVaR: {row['cvar_profit']:.6f} EUR")
            print(
                "Normalized downside gap: "
                f"{row['normalized_downside_gap_pct']:.6f}%"
            )
            print(f"Summary: {output}")

    combined = collect_results()
    print(f"\nCombined additional-case results: {combined}")


if __name__ == "__main__":
    main()
