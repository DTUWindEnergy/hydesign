# basic libraries
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import numpy as np

# from numpy import newaxis as na
import pandas as pd
import pyomo.environ as pyo
from docplex.mp.constants import ComparisonType

# import xarray as xr
from docplex.mp.model import Model
from ortools.linear_solver import pywraplp

from .ems_core_battery import EMSCoreBattery
from .ems_core_helpers import EMSCoreInterface, split_in_batch
from .ems_core_solar import EMSCoreSolar
from .ems_core_wind import EMSCoreWind


def ems_core(**kwargs):
    technology = kwargs.get("technology", {"wind", "solar", "battery"})
    ems_solver = kwargs.get("ems_solver", "cplex")

    # if ems_solver == "cplex":
    #     ems_batch = ems_cplex_parts
    # elif ems_solver == "pyomo":
    #     ems_batch = ems_Pyomo_parts
    # elif ems_solver == "ortools":
    #     ems_batch = ems_ortools_parts

    #
    out_ems = {}
    ems_core_defaults = EMSCoreDefaults(**kwargs)
    ems_core_wind = EMSCoreWind(**kwargs)
    ems_core_solar = EMSCoreSolar(**kwargs)
    ems_core_battery = EMSCoreBattery(**kwargs)

    batches = ems_core_defaults.batches

    ems_solver = "ortools"

    if ems_solver == "cplex":
        for ib, batch in enumerate(batches):
            ## cplex share
            mdl = Model(name="EMS")
            mdl.context.cplex_parameters.threads = 1
            mdl.context.cplex_parameters.emphasis.mip = 1

            ems_core_defaults.solver_init(mdl, batch, **kwargs)

            if "wind" in technology:
                ems_core_wind.solver_init(mdl, batch, **kwargs)

            if "solar" in technology:
                ems_core_solar.solver_init(mdl, batch, **kwargs)

            if "battery" in technology:
                ems_core_battery.solver_init(mdl, batch, **kwargs)

            ems_core_defaults.cplex_var_constraints(mdl, batch, **kwargs)
            if "wind" in technology:
                ems_core_wind.cplex_var_constraints(mdl, batch, **kwargs)

            if "solar" in technology:
                ems_core_solar.cplex_var_constraints(mdl, batch, **kwargs)

            if "battery" in technology:
                ems_core_battery.cplex_var_constraints(mdl, batch, **kwargs)

            # out_ems_batch.update(ems_core_wind.cplex(mdl, batch, **kwargs))
            # out_ems_batch.update()
            ems_core_defaults.cplex_opjective(mdl)
            if "wind" in technology:
                ems_core_wind.cplex_opjective(mdl)

            if "solar" in technology:
                ems_core_solar.cplex_opjective(mdl)

            if "battery" in technology:
                ems_core_battery.cplex_opjective(mdl)

            sol = mdl.solve(log_output=False)

            ems_core_defaults.cplex_outputs(sol)
            ems_core_wind.cplex_outputs(sol)
            ems_core_solar.cplex_outputs(sol)
            ems_core_battery.cplex_outputs(sol)

            # update ems outputs
            out_ems = ems_core_defaults.get_final_outputs()
            out_ems.update(ems_core_wind.get_final_outputs())
            out_ems.update(ems_core_solar.get_final_outputs())
            out_ems.update(ems_core_battery.get_final_outputs())

    elif ems_solver == "ortools":
        for ib, batch in enumerate(batches):
            # Create linear solver (GLOP)
            solver = pywraplp.Solver.CreateSolver("GLOP")
            if solver is None:
                raise RuntimeError(
                    "OR-Tools GLOP solver not available. Install ortools or choose another solver backend."
                )

            ems_core_defaults.solver_init(batch, **kwargs)
            if "wind" in technology:
                ems_core_wind.solver_init(batch, **kwargs)

            if "solar" in technology:
                ems_core_solar.solver_init(batch, **kwargs)

            if "battery" in technology:
                ems_core_battery.solver_init(batch, **kwargs)

                # ----------
            # vars and constraints

            ems_core_defaults.ortools_var_constraints(solver)
            if "wind" in technology:
                ems_core_wind.ortools_var_constraints(solver)

            if "solar" in technology:
                ems_core_solar.ortools_var_constraints(solver)

            if "battery" in technology:
                ems_core_battery.ortools_var_constraints(solver)

            ems_core_defaults.ortools_objective(solver)
            if "wind" in technology:
                ems_core_wind.ortools_objective(solver)

            if "solar" in technology:
                ems_core_solar.ortools_objective(solver)

            if "battery" in technology:
                ems_core_battery.ortools_objective(solver)

            # ------------
            # Solve
            result_status = solver.Solve()
            if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                raise RuntimeError(
                    f"Solver failed or returned infeasible (status {result_status})."
                )

            ems_core_defaults.ortools_outputs()

            # update ems outputs
            out_ems = ems_core_defaults.get_final_outputs()
            out_ems.update(ems_core_wind.get_final_outputs())
            out_ems.update(ems_core_solar.get_final_outputs())
            out_ems.update(ems_core_battery.get_final_outputs())

    # final process on outputs
    ems_core_defaults.final_process(**out_ems)
    if "wind" in technology:
        ems_core_wind.final_process(**out_ems)

    if "solar" in technology:
        ems_core_solar.final_process(**out_ems)

    if "battery" in technology:
        ems_core_battery.final_process(**out_ems)

    return out_ems


class EMSCoreDefaults(EMSCoreInterface):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        super().__init__(**kwargs)

        # allocate arrays

        # split in batches
        n_ts = self.n_ts
        batches_all = split_in_batch(list(range(n_ts)), self.batch_size)
        self.batches = batches_all

        n_ts = self.n_ts
        self.P_HPP_ts = np.zeros(n_ts)
        self.P_curtailment_ts = np.zeros(n_ts)
        self.penalty_ts = np.zeros(n_ts)

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        """ """
        self.price_ts_batch = self.price_ts.iloc[batch]
        self.batch = batch

        # Penalties
        self.N_t_batch = len(self.price_ts_batch.index)
        self.N_days_batch = self.N_t_batch / 24
        self.e_peak_day_expected = (
            self.n_full_power_hours_expected_per_day_at_peak_price
            * self.hpp_grid_connection
        )
        self.e_peak_period_expected_batch = self.e_peak_day_expected * self.N_days_batch
        self.price_peak_batch = np.quantile(
            self.price_ts_batch.values, self.peak_hr_quantile
        )
        self.peak_hours_index_batch = np.where(
            self.price_ts_batch.values >= self.price_peak_batch
        )[0]

        price_ts_to_max_batch = self.price_peak_batch - self.price_ts_batch
        price_ts_to_max_batch.loc[price_ts_to_max_batch < 0] = 0
        price_ts_to_max_batch.iloc[:-1] = (
            0.5 * price_ts_to_max_batch.iloc[:-1].values
            + 0.5 * price_ts_to_max_batch.iloc[1:].values
        )

        self.price_ts_to_max_batch = price_ts_to_max_batch

        self.time_batch = self.price_ts_batch.index

    def cplex_var_constraints(self, mdl, batch, **kwargs) -> Dict[str, Any]:
        """ """
        # Variables definition
        self.P_HPP_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, ub=self.hpp_grid_connection, name="HPP power output"
        )
        self.P_curtailment_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, name="Curtailment"
        )

        self.penalty_batch = mdl.continuous_var(name="penalty", lb=-1e12)
        self.e_penalty_batch = mdl.continuous_var(name="e_penalty", lb=-1e12)

        # Constraints
        mdl.add_constraint(
            self.e_penalty_batch
            == (
                self.e_peak_period_expected_batch
                - mdl.sum(
                    self.P_HPP_t_batch[self.time_batch[i]]
                    for i in self.peak_hours_index_batch
                )
            )
        )
        # Piecewise function for "only positive" function
        self.f1 = mdl.piecewise(0, [(0, 0)], 1)
        mdl.add_constraint(
            self.penalty_batch == self.price_peak_batch * self.f1(self.e_penalty_batch)
        )

        for t in self.time_batch:
            mdl.add_constraint(
                self.P_HPP_t_batch[t] == -self.P_curtailment_t_batch[t],
                ctname=f"P_HPP_t_balance{t}",
            )

    def cplex_opjective(self, mdl):
        # Piecewise function for "absolute value" function
        self.fabs = mdl.piecewise(-1, [(0, 0)], 1)

        mdl.maximize(0)

        electricity_income = mdl.sum(
            self.price_ts_batch[t] * self.P_HPP_t_batch[t] for t in self.time_batch
        )
        self.add_to_objective(mdl, addition_expr=electricity_income)

        penalty_grid = self.penalty_batch
        self.add_to_objective(mdl, addition_expr=penalty_grid)

    def cplex_outputs(self, sol):
        P_HPP_ts_batch_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_HPP_t_batch), orient="index"
        ).loc[:, 0]

        P_curtailment_ts_batch_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_curtailment_t_batch), orient="index"
        ).loc[:, 0]

        # make a time series like P_HPP with a constant penalty
        penalty_2 = sol.get_value(self.penalty_batch)
        penalty_ts_batch = np.ones(self.N_t_batch) * (penalty_2 / self.N_t_batch)

        # Cplex sometimes returns missing values :O
        P_HPP_ts_batch = P_HPP_ts_batch_df.reindex(self.time_batch, fill_value=0).values
        P_curtailment_ts_batch = P_curtailment_ts_batch_df.reindex(
            self.time_batch, fill_value=0
        ).values

        if len(P_HPP_ts_batch_df) < 1:  # len(self.wind_ts_batch):
            print("recomputing p_hpp")
            print("to do!")
            # P_HPP_ts_batch = (
            #     self.wind_ts_batch.values
            #     + self.solar_ts_batch.values
            #     - P_curtailment_ts_batch
            # )

        batch = self.batch

        self.P_HPP_ts[batch] = P_HPP_ts_batch
        self.P_curtailment_ts[batch] = P_curtailment_ts_batch
        self.penalty_ts[batch] = penalty_ts_batch

    def get_final_outputs(self):
        return {
            "P_HPP_ts": self.P_HPP_ts,
            "P_curtailment_ts": self.P_curtailment_ts,
            "penalty_ts": self.penalty_ts,
        }

    def ortools_var_constraints(self, solver):
        """
        EMS solver implemented with OR-Tools linear solver.
        Returns:
            P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
        """

        time_batch = self.time_batch
        N_t_batch = self.N_t_batch
        e_peak_period_expected_batch = self.e_peak_period_expected_batch
        peak_hours_index_batch = self.peak_hours_index_batch
        hpp_grid_connection = self.hpp_grid_connection

        # Variables
        # HPP power
        P_HPP_batch = {
            t: solver.NumVar(0.0, float(hpp_grid_connection), f"P_HPP_{i}")
            for i, t in enumerate(time_batch)
        }
        # Curtailment
        P_curtail_batch = {
            t: solver.NumVar(0.0, solver.infinity(), f"P_curt_{i}")
            for i, t in enumerate(time_batch)
        }

        # penalty variables
        e_penalty_batch = solver.NumVar(
            -solver.infinity(), solver.infinity(), "e_penalty"
        )
        p_nonneg_batch = solver.NumVar(
            0.0, solver.infinity(), "p_nonneg"
        )  # represents max(0, e_penalty)

        # delta variables for absolute battery power change cost
        delta = {}
        for i in range(N_t_batch - 1):
            t = time_batch[i]
            delta[t] = solver.NumVar(0.0, solver.infinity(), f"delta_{i}")

        # Constraints

        # e_penalty == e_peak_period_expected - sum(P_HPP over peak hours)
        if len(peak_hours_index_batch) > 0:
            expr = solver.Sum(
                [P_HPP_batch[time_batch[int(idx)]] for idx in peak_hours_index_batch]
            )
        else:
            expr = solver.Sum([])
        solver.Add(e_penalty_batch == float(e_peak_period_expected_batch) - expr)

        # p_nonneg >= e_penalty  (and p_nonneg >= 0 by var bound)
        solver.Add(p_nonneg_batch >= e_penalty_batch)

        # Per-time constraints
        for i, t in enumerate(time_batch):
            # power balance: P_HPP = wind + solar - curtail + (P_dis - P_ch)
            solver.Add(
                P_HPP_batch[t] == -P_curtail_batch[t], name=f"P_HPP_t_balance{t}"
            )

        self.P_HPP_batch = P_HPP_batch
        self.p_nonneg_batch = p_nonneg_batch
        self.delta = delta
        self.P_curtail_t_batch = P_curtail_batch

    def ortools_objective(self, solver):
        # ------------
        # Objective: maximize sum(price * P_HPP) - penalty - sum(delta * cost_ratio * price_ts_to_max)
        objective = solver.Objective()
        for t in self.time_batch:
            objective.SetCoefficient(
                self.P_HPP_batch[t], float(self.price_ts_batch.loc[t])
            )

        # penalty term (note penalty = price_peak * p_nonneg; objective subtracts it)
        objective.SetCoefficient(self.p_nonneg_batch, -float(self.price_peak_batch))

        objective.SetMaximization()

    def ortools_outputs(self):
        batch = self.batch
        time_batch = self.time_batch
        N_t_batch = self.N_t_batch
        price_peak_batch = self.price_peak_batch
        p_nonneg_batch = self.p_nonneg_batch
        P_HPP_batch = self.P_HPP_batch
        P_curtail_batch = self.P_curtail_t_batch

        # extract results
        P_HPP_ts_batch = np.array([P_HPP_batch[t].solution_value() for t in time_batch])
        P_curtail_ts_batch = np.array(
            [P_curtail_batch[t].solution_value() for t in time_batch]
        )

        penalty_val_batch = float(price_peak_batch * p_nonneg_batch.solution_value())
        penalty_ts_batch = np.ones(N_t_batch) * (penalty_val_batch / float(N_t_batch))

        self.P_HPP_ts[batch] = P_HPP_ts_batch
        self.P_curtailment_ts[batch] = P_curtail_ts_batch
        self.penalty_ts[batch] = penalty_ts_batch
