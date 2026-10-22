from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import time

import pandas as pd
import pyomo.environ as pyo


HERE = Path(__file__).resolve().parent
POST_PROCESSING_ROOT = HERE.parent
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

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(INPUTS_ROOT))

from config import (  # noqa: E402
    ALPHA,
    BASELINE_BETA,
    BETAS_TO_SOLVE,
    C_PPA,
    DELIVERY_PROFILE,
    E0,
    E_MAX,
    LAMBDA_PPA,
    P_B_MAX,
    PROFILE_LABEL,
    SETTLEMENT_PERIOD,
    SOLVER_NAME,
    SOLVER_THREADS,
    STAGE2_WIND_METHOD,
    WEATHER_YEARS,
    XI_WIND_FILES,
)
from prepare_inputs_from_btc_2000 import prepare_inputs  # noqa: E402
from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


def beta_slug(beta: float) -> str:
    return f"beta_{beta:g}".replace(".", "p")


def scenario_profit_distribution(model) -> list[tuple[float, float]]:
    return [
        (
            float(pyo.value(model.Phi[w, x])),
            float(pyo.value(model.pi_omega[w]))
            * float(pyo.value(model.pi_xi_given_omega[w, x])),
        )
        for w in model.Omega
        for x in model.Xi
    ]


def expected_profit(distribution: list[tuple[float, float]]) -> float:
    return sum(profit * probability for profit, probability in distribution)


def ex_post_cvar_profit(
    distribution: list[tuple[float, float]],
    alpha: float,
) -> float:
    """Evaluate lower-tail profit CVaR even when beta is zero."""
    losses = [-profit for profit, _probability in distribution]
    candidates = set(losses)
    loss_cvar = min(
        eta
        + sum(
            probability * max(loss - eta, 0.0)
            for loss, (_profit, probability) in zip(losses, distribution)
        )
        / (1.0 - alpha)
        for eta in candidates
    )
    return -loss_cvar


def contracted_energy(model) -> float:
    delta_t = float(pyo.value(model.delta_t))
    return sum(float(pyo.value(model.P_PPA[t])) * delta_t for t in model.T)


def daily_contracted_energy(model) -> float:
    delta_t = float(pyo.value(model.delta_t))
    first_day = list(model.T)[:24]
    return sum(float(pyo.value(model.P_PPA[t])) * delta_t for t in first_day)


def expected_shortfall(model) -> float:
    return sum(
        float(pyo.value(model.pi_omega[w]))
        * float(pyo.value(model.pi_xi_given_omega[w, x]))
        * float(pyo.value(model.S_short[sp, w, x]))
        for sp in model.SP
        for w in model.Omega
        for x in model.Xi
    )


def write_json(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2)
        handle.write("\n")


def select_one(frame: pd.DataFrame, source: Path) -> pd.Series:
    selected = frame.loc[
        (frame["sdp_label"] == PROFILE_LABEL)
        & frame["settlement_period"].str.lower().eq(SETTLEMENT_PERIOD)
        & frame["lambda_PPA"].astype(float).eq(LAMBDA_PPA)
        & frame["c_ppa"].astype(float).eq(C_PPA)
        & frame["P_b_max"].astype(float).eq(P_B_MAX)
        & frame["E_max"].astype(float).eq(E_MAX)
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one benchmark row in {source}, found {len(selected)}."
        )
    return selected.iloc[0]


def load_baseline() -> dict:
    for path in (BASELINE_DECOMPOSITION, BASELINE_SETTLEMENT):
        if not path.exists():
            raise FileNotFoundError(f"Missing baseline data: {path}")
    decomposition = select_one(
        pd.read_csv(BASELINE_DECOMPOSITION), BASELINE_DECOMPOSITION
    )
    settlement = select_one(
        pd.read_csv(BASELINE_SETTLEMENT), BASELINE_SETTLEMENT
    )
    return {
        "run_type": "existing_baseline",
        "beta": BASELINE_BETA,
        "alpha": ALPHA,
        "delivery_profile": DELIVERY_PROFILE,
        "lambda_PPA": LAMBDA_PPA,
        "c_ppa": C_PPA,
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "settlement_period": SETTLEMENT_PERIOD,
        "S_PPA": float(decomposition["S_PPA"]),
        "annual_contracted_energy_mwh": float(
            settlement["annual_contracted_energy_mwh"]
        ),
        "daily_contracted_energy_mwh": (
            float(settlement["annual_contracted_energy_mwh"]) / 364.0
        ),
        "expected_shortfall_mwh": float(settlement["expected_shortfall_mwh"]),
        "expected_shortfall_rate_pct": float(
            settlement["expected_shortfall_rate_pct"]
        ),
        "expected_profit": float(decomposition["expected_profit"]),
        "cvar_profit": float(decomposition["cvar_profit"]),
        "objective_value": float(decomposition["reported_objective"]),
        "objective_reconstructed": float(
            decomposition["reconstructed_objective"]
        ),
        "solver_status": "existing result",
        "termination_condition": "optimal",
        "elapsed_seconds": None,
    }


def solve_beta(beta: float) -> dict:
    data = load_all_data(GENERATED_INPUTS)
    cfg = data["config"]
    cfg["lambda_PPA"] = LAMBDA_PPA
    cfg["c_ppa"] = C_PPA
    cfg["P_b_max"] = P_B_MAX
    cfg["E_max"] = E_MAX
    cfg["E0"] = E0
    cfg["settlement_period"] = SETTLEMENT_PERIOD
    cfg["alpha"] = ALPHA
    cfg["beta"] = float(beta)

    model = build_model(data)
    solver = pyo.SolverFactory(SOLVER_NAME)
    if SOLVER_NAME == "cplex":
        solver.options["threads"] = SOLVER_THREADS
    started = time.perf_counter()
    results = solver.solve(model, tee=True)
    elapsed_seconds = time.perf_counter() - started
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            f"Solver failed for beta={beta:g}: status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )

    annual_energy = contracted_energy(model)
    shortfall = expected_shortfall(model)
    distribution = scenario_profit_distribution(model)
    alpha = float(pyo.value(model.alpha))
    expected = expected_profit(distribution)
    cvar = ex_post_cvar_profit(distribution, alpha)
    objective = float(pyo.value(model.OBJ))
    return {
        "run_type": "risk_aversion_sensitivity",
        "beta": float(beta),
        "alpha": alpha,
        "delivery_profile": DELIVERY_PROFILE,
        "lambda_PPA": LAMBDA_PPA,
        "c_ppa": C_PPA,
        "P_b_max": P_B_MAX,
        "E_max": E_MAX,
        "settlement_period": SETTLEMENT_PERIOD,
        "S_PPA": float(pyo.value(model.S_PPA)),
        "annual_contracted_energy_mwh": annual_energy,
        "daily_contracted_energy_mwh": daily_contracted_energy(model),
        "expected_shortfall_mwh": shortfall,
        "expected_shortfall_rate_pct": (
            100.0 * shortfall / annual_energy if annual_energy > 0.0 else 0.0
        ),
        "expected_profit": expected,
        "cvar_profit": cvar,
        "objective_value": objective,
        "objective_reconstructed": (1.0 - beta) * expected + beta * cvar,
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "elapsed_seconds": elapsed_seconds,
    }


def collect_results() -> Path:
    rows = []
    for path in sorted(RESULTS_ROOT.glob("beta_*/summary.json")):
        with path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        raise FileNotFoundError(f"No summaries found below {RESULTS_ROOT}.")
    rows.sort(key=lambda row: float(row["beta"]))
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output = OUTPUT_ROOT / "risk_aversion_summary.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output


def main() -> None:
    baseline = load_baseline()
    write_json(
        RESULTS_ROOT / beta_slug(BASELINE_BETA) / "summary.json",
        baseline,
    )

    prepare_inputs(
        delivery_profile_name=DELIVERY_PROFILE,
        weather_years=list(WEATHER_YEARS),
        xi_wind_files=list(XI_WIND_FILES),
        stage2_wind_method=STAGE2_WIND_METHOD,
    )

    for beta in BETAS_TO_SOLVE:
        output = RESULTS_ROOT / beta_slug(beta) / "summary.json"
        if output.exists():
            print(f"Skipping completed beta={beta:g} result: {output}")
            continue
        print(f"\n=== RISK-AVERSION RUN: beta={beta:g} ===")
        row = solve_beta(beta)
        write_json(output, row)
        print(f"Objective: {row['objective_value']:.6f} EUR")
        print(f"Expected profit: {row['expected_profit']:.6f} EUR")
        print(f"CVaR: {row['cvar_profit']:.6f} EUR")
        print(f"S_PPA: {row['S_PPA']:.10f} MW")
        print(f"Summary: {output}")

    combined = collect_results()
    print(f"\nCombined risk-aversion results: {combined}")


if __name__ == "__main__":
    main()
