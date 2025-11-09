# basic libraries
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from docplex.mp.model import Model
from ortools.linear_solver import pywraplp

from hydesign.ems_modular.modules.ems_core_helpers import (
    EMSCoreInterface,
    series_from_indexed_var,
    split_in_batch,
)


class EMSCoreDefaults(EMSCoreInterface):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        """ """
        # --- defaults
        self.ems_solver = kwargs["ems_solver"]
        self.hpp_grid_connection = kwargs.get("hpp_grid_connection", 0)
        self.price_ts = kwargs["price_ts"]
        self.peak_hr_quantile = kwargs.get("peak_hr_quantile", 0.9)
        self.n_full_power_hours_expected_per_day_at_peak_price = kwargs.get(
            "n_full_power_hours_expected_per_day_at_peak_price", 3
        )
        self.batch_size = kwargs.get("batch_size", 110)
        n_ts = len(self.price_ts)
        self.n_ts = n_ts

        # split in batches
        batches_all = split_in_batch(list(range(n_ts)), self.batch_size)
        self.batches = batches_all

        self.P_HPP_ts = np.zeros(n_ts)
        self.P_curtailment_ts = np.zeros(n_ts)
        self.penalty_ts = np.zeros(n_ts)

    def solve(self, **kwargs):
        out_ems = {}
        ems_solver = self.ems_solver
        for ib, batch in enumerate(self.batches):
            self.batch_data(batch=batch)
            if ems_solver == "cplex":
                self.cplex()
            elif ems_solver == "pyomo":
                self.pyomo()
            elif ems_solver == "ortools":
                self.ortools()
            self.update_final_outputs(out_ems)
        return out_ems

    def batch_data(self, batch, **kwargs) -> Dict[str, Any]:
        """ """
        # --- defaults
        self.price_ts_batch = self.price_ts.iloc[batch]
        self.batch = batch
        self.time_batch = self.price_ts_batch.index
        # time set with an additional time slot for the last soc
        self.SOCtime_batch = self.time_batch.append(
            pd.Index([self.time_batch[-1] + pd.Timedelta("1hour")])
        )

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

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        # ------------
        # Variables
        # ------------
        mdl = Model(name="EMS")
        mdl.context.cplex_parameters.threads = 1
        mdl.context.cplex_parameters.emphasis.mip = 1
        # --- defaults
        self.P_HPP_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, ub=self.hpp_grid_connection, name="HPP power output"
        )
        self.P_curtailment_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, name="Curtailment"
        )

        self.penalty_el_batch = mdl.continuous_var(name="penalty", lb=-1e12)
        self.e_penalty_batch = mdl.continuous_var(name="e_penalty", lb=-1e12)

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        # ------------
        # Constraints
        # ------------
        # --- defaults
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
            self.penalty_el_batch
            == self.price_peak_batch * self.f1(self.e_penalty_batch),
            ctname="penalty_el",
        )

        for t in self.time_batch:
            mdl.add_constraint(
                self.P_HPP_t_batch[t] == -self.P_curtailment_t_batch[t],
                ctname=f"P_HPP_t_balance{t}",
            )

    def cplex_opjective(self, mdl):
        """ """
        # ------------
        # Objective
        # ------------
        # --- default
        self.fabs = mdl.piecewise(-1, [(0, 0)], 1)
        mdl.maximize(0)

        revenue_el = mdl.sum(
            self.price_ts_batch[t] * self.P_HPP_t_batch[t] for t in self.time_batch
        )
        self.add_to_objective(mdl, addition_expr=revenue_el)

        self.add_to_objective(mdl, addition_expr=-self.penalty_el_batch)

    def cplex_solve(self, mdl):
        sol = mdl.solve(log_output=False)

    def cplex_outputs(self, sol):
        """ """
        # ------------
        # Outputs
        # ------------
        # --- defaults
        P_HPP_ts_batch_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_HPP_t_batch), orient="index"
        ).loc[:, 0]

        P_curtailment_ts_batch_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_curtailment_t_batch), orient="index"
        ).loc[:, 0]

        # make a time series like P_HPP with a constant penalty
        penalty_el_2 = sol.get_value(self.penalty_el_batch)
        penalty_el_ts_batch = np.ones(self.N_t_batch) * (penalty_el_2 / self.N_t_batch)

        # Cplex sometimes returns missing values :O
        P_HPP_ts_batch = P_HPP_ts_batch_df.reindex(self.time_batch, fill_value=0).values
        P_curtailment_ts_batch = P_curtailment_ts_batch_df.reindex(
            self.time_batch, fill_value=0
        ).values

        if len(P_HPP_ts_batch_df) < 1:  # len(self.wind_ts_batch):
            print("recomputing p_hpp")
            print("to do!")
        #     P_HPP_ts_batch = (
        #         wind_ts_batch.values
        #         + solar_ts_batch.values
        #         + -P_curtailment_ts_batch
        #         + P_charge_discharge_ts_batch
        #     )

        self.P_HPP_ts[self.batch] = P_HPP_ts_batch
        self.P_curtailment_ts[self.batch] = P_curtailment_ts_batch
        self.penalty_ts[self.batch] = penalty_el_ts_batch

    def pyomo_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        # ------------
        # Variables
        # ------------
        mdl = pyo.ConcreteModel()
        # --- defaults
        time_batch = self.time_batch

        hpp_grid_connection = self.hpp_grid_connection

        mdl.IDX1 = range(len(time_batch))
        mdl.IDX2 = range(1)

        mdl.P_HPP_t = pyo.Var(
            mdl.IDX1, domain=pyo.NonNegativeReals, bounds=(0, hpp_grid_connection)
        )
        mdl.P_curtailment_t = pyo.Var(mdl.IDX1, domain=pyo.NonNegativeReals)
        mdl.penalty = pyo.Var(mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))
        mdl.e_penalty = pyo.Var(mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))

    def pyomo_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        # ------------
        # Constraints
        # ------------
        # --- defaults
        time_batch = self.time_batch

        e_peak_period_expected_batch = self.e_peak_period_expected_batch
        price_peak_batch = self.price_peak_batch

        mdl.curtailment_constraint = pyo.ConstraintList()
        mdl.power_constraint = pyo.ConstraintList()

        mdl.penalty_constraint = pyo.ConstraintList()

        mdl.penalty_constraint.add(
            mdl.e_penalty[0]
            == (e_peak_period_expected_batch - sum(mdl.P_HPP_t[t] for t in mdl.IDX3))
        )

        mdl.penalty_constraint_pw = pyo.Piecewise(
            mdl.IDX2,
            mdl.penalty,
            mdl.e_penalty,
            pw_pts=[-1e12, 0, 1e12],
            f_rule=[0, 0, price_peak_batch * 1e12],
            pw_constr_type="EQ",
            # pw_repn='SOS2',
            # pw_repn='CC',
            # pw_repn='DCC',
            pw_repn="DLOG",
            force_pw=False,
            warn_domain_coverage=False,
            # unbounded_domain_var=True
        )
        from pyomo.environ import Constraint, Set

        mdl.T = Set(initialize=range(len(time_batch)))

        # define indexed constraint once (rule takes index t)
        def power_rule(m, t):
            return m.P_HPP_t[t] == -m.P_curtailment_t[t]

        if hasattr(mdl, "power_constraint"):
            mdl.del_component("power_constraint")

        mdl.power_constraint = Constraint(mdl.T, rule=power_rule)

    def pyomo_opjective(self, mdl, **kwargs):
        """ """
        # ------------
        # Objective
        # ------------
        # --- defaults
        price_ts_batch = self.price_ts_batch

        mdl.OBJ = pyo.Objective(
            # revenues and OPEX
            expr=0,
            sense=pyo.maximize,
        )

        electricity_income = sum(
            price_ts_batch.iloc[t] * mdl.P_HPP_t[t] for t in mdl.IDX1
        )
        penalty = -mdl.penalty[0]

        self.add_to_objective_pyomo(mdl, addition_expr=electricity_income)
        self.add_to_objective_pyomo(mdl, addition_expr=penalty)

    def pyomo_solve(self, mdl):
        solver = pyo.SolverFactory("cplex")  # glpk
        result = solver.solve(mdl, tee=False)  # tee=True if you want solver log

    def pyomo_outputs(self, mdl, **kwargs):
        """ """
        # ------------
        # Outputs
        # ------------
        N_t = self.N_t_batch
        batch = self.batch
        time_batch = self.time_batch
        batch = self.batch

        # --- defaults
        P_HPP_ts_batch_df = series_from_indexed_var(mdl.P_HPP_t, time_batch)
        P_curtailment_ts_batch_df = series_from_indexed_var(
            mdl.P_curtailment_t, time_batch
        )

        # scalar penalty (assumes penalty is a scalar Var or Param)
        try:
            penalty_2 = pyo.value(mdl.penalty[0])
        except Exception:
            penalty_2 = 0.0

        # make a time_batch series like P_HPP with a constant penalty
        penalty_ts_batch = np.ones(N_t) * (penalty_2 / N_t)

        # CPLEX / Pyomo should give values for each index — but keep your reindex/fill behaviour
        P_HPP_ts_batch = P_HPP_ts_batch_df.reindex(time_batch, fill_value=0).values
        P_curtailment_ts_batch = P_curtailment_ts_batch_df.reindex(
            time_batch, fill_value=0
        ).values

        self.P_HPP_ts[batch] = P_HPP_ts_batch
        self.P_curtailment_ts[batch] = P_curtailment_ts_batch
        self.penalty_ts[batch] = penalty_ts_batch

    def ortools_variables(self, solver):
        """ """
        # ------------
        # Variables
        # ------------
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            raise RuntimeError(
                "OR-Tools GLOP solver not available. Install ortools or choose another solver backend."
            )
        time_batch = self.time_batch
        hpp_grid_connection = self.hpp_grid_connection

        # --- defaults
        P_HPP_batch = {
            t: solver.NumVar(0.0, float(hpp_grid_connection), f"P_HPP_{i}")
            for i, t in enumerate(time_batch)
        }

        P_curtail_batch = {
            t: solver.NumVar(0.0, solver.infinity(), f"P_curt_{i}")
            for i, t in enumerate(time_batch)
        }

        penalty_el_batch = solver.NumVar(
            -solver.infinity(), solver.infinity(), "penalty_el"
        )

        e_penalty_batch = solver.NumVar(
            -solver.infinity(), solver.infinity(), "e_penalty"
        )

        p_nonneg_batch = solver.NumVar(
            0.0, solver.infinity(), "p_nonneg"
        )  # represents max(0, e_penalty)

        self.P_HPP_batch = P_HPP_batch
        self.p_nonneg_batch = p_nonneg_batch
        self.P_curtail_batch = P_curtail_batch
        self.penalty_el_batch = penalty_el_batch
        self.e_penalty_batch = e_penalty_batch

    def ortools_constraints(self, solver):
        """ """
        # ------------
        # Constraints
        # ------------
        time_batch = self.time_batch
        N_t_batch = self.N_t_batch
        e_peak_period_expected_batch = self.e_peak_period_expected_batch
        peak_hours_index_batch = self.peak_hours_index_batch

        P_HPP_batch = self.P_HPP_batch
        p_nonneg_batch = self.p_nonneg_batch
        P_curtail_batch = self.P_curtail_batch
        e_penalty_batch = self.e_penalty_batch
        penalty_el_batch = self.penalty_el_batch

        # delta variables for absolute battery power change cost
        delta = {}
        for i in range(N_t_batch - 1):
            t = time_batch[i]
            delta[t] = solver.NumVar(0.0, solver.infinity(), f"delta_{i}")

        self.delta = delta

        # --- defaults
        # e_penalty == e_peak_period_expected - sum(P_HPP over peak hours)
        # Sum of P_HPP over peak hours
        expr = (
            solver.Sum(
                [P_HPP_batch[time_batch[int(idx)]] for idx in peak_hours_index_batch]
            )
            if len(peak_hours_index_batch) > 0
            else 0.0
        )

        # Define e_penalty

        solver.Add(e_penalty_batch == float(e_peak_period_expected_batch) - expr)

        # Linearize f_pos = max(0, e_penalty)
        f_pos = solver.NumVar(0.0, solver.infinity(), "f_pos")
        solver.Add(f_pos >= e_penalty_batch)

        # Penalty = price_peak * f_pos
        solver.Add(penalty_el_batch == float(self.price_peak_batch) * f_pos)

        # p_nonneg >= e_penalty  (and p_nonneg >= 0 by var bound)
        solver.Add(p_nonneg_batch >= e_penalty_batch)

        # Per-time constraints
        for i, t in enumerate(time_batch):
            # power balance: P_HPP = wind + solar - curtail + (P_dis - P_ch)
            solver.Add(
                P_HPP_batch[t] == -P_curtail_batch[t], name=f"P_HPP_t_balance{t}"
            )

    def ortools_objective(self, solver):
        """ """
        # ------------
        # Objective
        # ------------
        # --- defaults
        objective = solver.Objective()
        for t in self.time_batch:
            objective.SetCoefficient(
                self.P_HPP_batch[t], float(self.price_ts_batch.loc[t])
            )

        if self.penalty_el_batch is not None:
            objective.SetCoefficient(self.penalty_el_batch, -1.0)
        # penalty term (note penalty = price_peak * p_nonneg; objective subtracts it)
        # objective.SetCoefficient(self.p_nonneg_batch, -float(self.price_peak_batch))

        objective.SetMaximization()

    def ortools_solve(self, solver):
        result_status = solver.Solve()
        if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(
                f"Solver failed or returned infeasible (status {result_status})."
            )

    def ortools_outputs(self):
        """ """
        # ------------
        # Outputs
        # ------------
        batch = self.batch
        time_batch = self.time_batch
        N_t_batch = self.N_t_batch
        price_peak_batch = self.price_peak_batch
        p_nonneg_batch = self.p_nonneg_batch
        P_HPP_batch = self.P_HPP_batch
        P_curtail_batch = self.P_curtail_batch

        # --- defaults
        P_HPP_ts_batch = np.array([P_HPP_batch[t].solution_value() for t in time_batch])
        P_curtail_ts_batch = np.array(
            [P_curtail_batch[t].solution_value() for t in time_batch]
        )

        penalty_el_val_batch = float(price_peak_batch * p_nonneg_batch.solution_value())
        penalty_el_ts_batch = np.ones(N_t_batch) * (
            penalty_el_val_batch / float(N_t_batch)
        )

        penalty_ts_batch = penalty_el_ts_batch

        self.P_HPP_ts[batch] = P_HPP_ts_batch
        self.P_curtailment_ts[batch] = P_curtail_ts_batch
        self.penalty_ts[batch] = penalty_ts_batch

    def update_final_outputs(self, out):
        """ """
        # --- defaults
        out_defaults = {
            "P_HPP_ts": self.P_HPP_ts,
            "P_curtailment_ts": self.P_curtailment_ts,
            "penalty_ts": self.penalty_ts,
        }
        out.update(out_defaults)
