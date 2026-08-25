from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyomo.environ as pyo


ANALYSIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ANALYSIS_DIR.parent
INPUTS_DIR = PROJECT_ROOT / "inputs" / "Input_files_created"
RESULTS_DIR = ANALYSIS_DIR / "results"
CONFIG_PATH = ANALYSIS_DIR / "analysis_config.json"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.thesis_hpp.data_loader import load_all_data  # noqa: E402
from src.thesis_hpp.model import build_model  # noqa: E402


def load_analysis_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_benchmark_data(
    lambda_ppa: float,
    c_ppa: float | None = None,
    settlement_period: str | None = None,
) -> tuple[dict, dict]:
    cfg = load_analysis_config()
    data = load_all_data(INPUTS_DIR)
    model_cfg = data["config"]
    model_cfg["lambda_PPA"] = float(lambda_ppa)
    model_cfg["c_ppa"] = float(cfg["c_ppa"] if c_ppa is None else c_ppa)
    model_cfg["P_b_max"] = float(cfg["p_b_max"])
    model_cfg["E_max"] = float(cfg["e_max"])
    model_cfg["E0"] = float(cfg["e0"])
    selected_period = (
        cfg["settlement_period"]
        if settlement_period is None
        else str(settlement_period).strip().lower()
    )
    model_cfg["settlement_period"] = selected_period
    cfg = {**cfg, "settlement_period": selected_period}
    return data, cfg


def solve_fixed_capacity(
    lambda_ppa: float,
    s_ppa: float,
    c_ppa: float | None = None,
    settlement_period: str | None = None,
):
    data, cfg = load_benchmark_data(
        lambda_ppa,
        c_ppa=c_ppa,
        settlement_period=settlement_period,
    )
    if c_ppa is not None:
        cfg = {**cfg, "c_ppa": float(c_ppa)}
    model = build_model(data)
    model.S_PPA.fix(float(s_ppa))
    solver = pyo.SolverFactory("cplex")
    started = time.perf_counter()
    results = solver.solve(model, tee=True)
    elapsed_seconds = time.perf_counter() - started
    if not pyo.check_optimal_termination(results):
        raise RuntimeError(
            "CPLEX did not return an optimal solution: "
            f"status={results.solver.status}, "
            f"termination={results.solver.termination_condition}"
        )
    return model, results, elapsed_seconds, cfg


def contracted_energy_mwh(model) -> float:
    delta_t = float(pyo.value(model.delta_t))
    return sum(float(pyo.value(model.P_PPA[t])) * delta_t for t in model.T)


def expected_profit(model) -> float:
    return sum(
        float(pyo.value(model.pi_omega[w]))
        * float(pyo.value(model.pi_xi_given_omega[w, x]))
        * float(pyo.value(model.Phi[w, x]))
        for w in model.Omega
        for x in model.Xi
    )


def cvar_profit(model) -> float:
    tail_loss = sum(
        float(pyo.value(model.pi_omega[w]))
        * float(pyo.value(model.pi_xi_given_omega[w, x]))
        * float(pyo.value(model.u[w, x]))
        for w in model.Omega
        for x in model.Xi
    )
    return -(
        float(pyo.value(model.eta_cvar))
        + tail_loss / (1.0 - float(pyo.value(model.alpha)))
    )


def write_summary(relative_path: str, summary: dict) -> Path:
    output_path = RESULTS_DIR / relative_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return output_path


def base_summary(model, results, elapsed_seconds: float, cfg: dict) -> dict:
    expected = expected_profit(model)
    cvar = cvar_profit(model)
    beta = float(pyo.value(model.beta))
    return {
        "delivery_profile": cfg["delivery_profile"],
        "c_ppa": float(cfg["c_ppa"]),
        "p_b_max": float(cfg["p_b_max"]),
        "e_max": float(cfg["e_max"]),
        "e0": float(cfg["e0"]),
        "settlement_period": cfg["settlement_period"],
        "lambda_ppa": float(pyo.value(model.lambda_PPA)),
        "s_ppa": float(pyo.value(model.S_PPA)),
        "contracted_energy_mwh": contracted_energy_mwh(model),
        "objective_value": float(pyo.value(model.OBJ)),
        "expected_profit": expected,
        "cvar_profit": cvar,
        "beta": beta,
        "objective_reconstructed": (1.0 - beta) * expected + beta * cvar,
        "solver_status": str(results.solver.status),
        "termination_condition": str(results.solver.termination_condition),
        "elapsed_seconds": elapsed_seconds,
    }
