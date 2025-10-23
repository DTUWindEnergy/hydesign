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

    ems_solver = "ortools"  # "cplex"

    if ems_solver == "cplex":
        for ib, batch in enumerate(batches):
            ## cplex share
            mdl = Model(name="EMS")
            mdl.context.cplex_parameters.threads = 1
            mdl.context.cplex_parameters.emphasis.mip = 1

            # init
            ems_core_defaults.solver_init(batch, **kwargs)
            if "wind" in technology:
                ems_core_wind.solver_init(batch, **kwargs)
            if "solar" in technology:
                ems_core_solar.solver_init(batch, **kwargs)
            if "battery" in technology:
                ems_core_battery.solver_init(batch, **kwargs)

            # variables
            ems_core_defaults.cplex_variables(mdl)
            if "wind" in technology:
                ems_core_wind.cplex_variables(mdl)

            if "solar" in technology:
                ems_core_solar.cplex_variables(mdl)

            if "battery" in technology:
                ems_core_battery.cplex_variables(mdl)

            # objective
            ems_core_defaults.cplex_opjective(mdl)
            if "wind" in technology:
                ems_core_wind.cplex_opjective(mdl)
            if "solar" in technology:
                ems_core_solar.cplex_opjective(mdl)
            if "battery" in technology:
                ems_core_battery.cplex_opjective(mdl)

            # Constraints
            ems_core_defaults.cplex_constraints(mdl)
            if "wind" in technology:
                ems_core_wind.cplex_constraints(mdl)

            if "solar" in technology:
                ems_core_solar.cplex_constraints(mdl)

            if "battery" in technology:
                ems_core_battery.cplex_constraints(mdl)

            # Solve
            sol = mdl.solve(log_output=False)

            ems_core_defaults.cplex_outputs(sol)
            if "wind" in technology:
                ems_core_wind.cplex_outputs(sol)
            if "solar" in technology:
                ems_core_solar.cplex_outputs(sol)
            if "battery" in technology:
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

            ems_core_defaults.ortools_variables(solver)
            if "wind" in technology:
                ems_core_wind.ortools_variables(solver)

            if "solar" in technology:
                ems_core_solar.ortools_variables(solver)

            if "battery" in technology:
                ems_core_battery.ortools_variables(solver)

            ems_core_defaults.ortools_objective(solver)
            if "wind" in technology:
                ems_core_wind.ortools_objective(solver)

            if "solar" in technology:
                ems_core_solar.ortools_objective(solver)

            if "battery" in technology:
                ems_core_battery.ortools_objective(solver)

            ems_core_defaults.ortools_constraints(solver)
            if "wind" in technology:
                ems_core_wind.ortools_constraints(solver)

            if "solar" in technology:
                ems_core_solar.ortools_constraints(solver)

            if "battery" in technology:
                ems_core_battery.ortools_constraints(solver)

            # ------------
            # Solve
            result_status = solver.Solve()
            if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
                raise RuntimeError(
                    f"Solver failed or returned infeasible (status {result_status})."
                )

            ems_core_defaults.ortools_outputs()
            if "wind" in technology:
                ems_core_wind.ortools_outputs()
            if "solar" in technology:
                ems_core_solar.ortools_outputs()
            if "battery" in technology:
                ems_core_battery.ortools_outputs()

        # update ems outputs
        out_ems = ems_core_defaults.get_final_outputs()
        out_ems.update(ems_core_wind.get_final_outputs())
        out_ems.update(ems_core_solar.get_final_outputs())
        out_ems.update(ems_core_battery.get_final_outputs())

    elif ems_solver == "pyomo":
        for ib, batch in enumerate(batches):
            ems_core_defaults.solver_init(batch, **kwargs)
            out_pyomo = ems_core_defaults.ems_Pyomo_parts(**kwargs)
            ems_core_defaults.P_HPP_ts[batch] = out_pyomo["P_HPP_ts"]
            ems_core_defaults.P_curtailment_ts[batch] = out_pyomo["P_curtailment_ts"]
            ems_core_defaults.penalty_ts[batch] = out_pyomo["penalty_ts"]
            ems_core_battery.P_charge_discharge_ts[batch] = out_pyomo[
                "P_charge_discharge_ts"
            ]
            ems_core_battery.E_SOC_ts[batch] = out_pyomo["E_SOC_ts"][:-1]

            # print(out_pyomo["P_HPP_ts"])
            # print(ems_core_defaults.P_HPP_ts[batch])
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

    vals = 10 * np.ones(len(kwargs["price_ts"]))
    out_ems2 = {
        "P_HPP_ts": 0 * vals,
        "P_curtailment_ts": vals,
        "penalty_ts": 0 * vals,
        "P_charge_discharge_ts": vals,
        "E_SOC_ts": vals,
    }
    return out_ems2


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

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:
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

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
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

    def ortools_variables(self, solver):
        """
        EMS solver implemented with OR-Tools linear solver.
        Returns:
            P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
        """

        time_batch = self.time_batch
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

        self.P_HPP_batch = P_HPP_batch
        self.p_nonneg_batch = p_nonneg_batch
        self.P_curtail_batch = P_curtail_batch
        self.e_penalty_batch = e_penalty_batch

    def ortools_constraints(self, solver):
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

        P_HPP_batch = self.P_HPP_batch
        p_nonneg_batch = self.p_nonneg_batch
        P_curtail_batch = self.P_curtail_batch
        e_penalty_batch = self.e_penalty_batch

        # delta variables for absolute battery power change cost
        delta = {}
        for i in range(N_t_batch - 1):
            t = time_batch[i]
            delta[t] = solver.NumVar(0.0, solver.infinity(), f"delta_{i}")

        self.delta = delta
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
        P_curtail_batch = self.P_curtail_batch

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

    def ems_Pyomo_parts(self, **kwargs):
        """EMS solver implemented in Pyomo (Parts)

        Parameters
        ----------
        wind_ts : WPP power time series
        solar_ts : PVP power time series
        price_ts : price time series
        P_batt_MW : battery power
        E_batt_MWh_t : battery energy capacity time series
        hpp_grid_connection : grid connection
        battery_depth_of_discharge : battery depth of discharge
        charge_efficiency : battery charge efficiency
        peak_hr_quantile : quantile of price time series to define peak price hours
        cost_of_battery_P_fluct_in_peak_price_ratio : cost of battery power fluctuations computed as a peak price ratio
        n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached

        Returns
        -------
        P_HPP_ts: HPP power time series
        P_curtailment_ts: HPP curtailed power time series
        P_charge_discharge_ts: Battery charge/discharge power time series
        E_SOC_ts: Battery energy SOC time series
        penalty_ts: penalty time series for not reaching expected energy production at peak hours
        """
        batch = self.batch

        wind_ts = kwargs["wind_ts"].iloc[batch]

        solar_ts = 0 * kwargs["solar_ts"].iloc[batch]

        hpp_grid_connection = kwargs.get("hpp_grid_connection", 0)
        price_ts = self.price_ts_batch  # kwargs["price_ts"].iloc[batch]
        peak_hr_quantile = kwargs.get("peak_hr_quantile", 0.9)
        n_full_power_hours_expected_per_day_at_peak_price = (
            self.n_full_power_hours_expected_per_day_at_peak_price
        )

        P_batt_MW = kwargs["P_batt_MW"]
        # self.hpp_grid_connection = kwargs["hpp_grid_connection"]
        battery_depth_of_discharge = kwargs["battery_depth_of_discharge"]
        charge_efficiency = kwargs["charge_efficiency"]
        cost_of_battery_P_fluct_in_peak_price_ratio = kwargs.get(
            "cost_of_battery_P_fluct_in_peak_price_ratio", 0.5
        )

        E_batt_MWh_t = kwargs["E_batt_MWh_t"].iloc[batch]

        # extract parameters into the variable space
        # globals().update(self.__dict__)
        # Penalties
        N_t = self.N_t_batch  # len(price_ts.index)
        e_peak_period_expected = self.e_peak_period_expected_batch
        price_peak = self.price_peak_batch
        peak_hours_index = self.peak_hours_index_batch

        price_ts_to_max = self.price_ts_to_max_batch  # price_peak - price_ts

        time = price_ts.index

        # time set with an additional time slot for the last soc
        SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))

        mdl = pyo.ConcreteModel()

        ## Variables ##
        mdl.IDX1 = range(len(time))
        mdl.IDX2 = range(1)
        mdl.IDX3 = [t for t in range(len(time)) if t in peak_hours_index]
        mdl.IDX4 = range(len(SOCtime))
        mdl.IDX5 = range(len(time) - 1)

        mdl.P_HPP_t = pyo.Var(
            mdl.IDX1, domain=pyo.NonNegativeReals, bounds=(0, hpp_grid_connection)
        )
        mdl.P_curtailment_t = pyo.Var(mdl.IDX1, domain=pyo.NonNegativeReals)

        # Power charge/discharge from battery with Lower/Upper bounds
        mdl.P_charge_discharge_t = pyo.Var(
            mdl.IDX1,
            domain=pyo.Reals,
            bounds=(-2 * P_batt_MW, 2 * P_batt_MW),  # excess bounds
        )
        # Battery energy level
        mdl.E_SOC_t = pyo.Var(
            mdl.IDX4, domain=pyo.NonNegativeReals, bounds=(0, 1e12)
        )  # bounds=(0,E_batt_MWh))

        ## Constraints ##
        mdl.curtailment_constraint = pyo.ConstraintList()
        mdl.power_constraint = pyo.ConstraintList()
        mdl.charge_discharge_constraint = pyo.ConstraintList()
        mdl.battery_energy_constraint = pyo.ConstraintList()
        mdl.battery_energy_min_constraint = pyo.ConstraintList()
        mdl.battery_dynamics_constraint = pyo.ConstraintList()
        mdl.penalty_constraint = pyo.ConstraintList()

        mdl.SOC_initial_condition = pyo.Constraint(
            expr=mdl.E_SOC_t[0] == 0.5 * E_batt_MWh_t.iloc[0]
        )

        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        mdl.SOC_final = pyo.Constraint(
            expr=mdl.E_SOC_t[len(time) - 1] == 0.5 * E_batt_MWh_t.iloc[len(time) - 1]
        )

        # x-values of the piece-wise function
        f_piecewise_x_vals = [-P_batt_MW, 0, P_batt_MW]
        # y-values of the piece-wise function
        f_piecewise_y_vals = [
            -P_batt_MW / charge_efficiency,
            0,
            P_batt_MW * charge_efficiency,
        ]

        mdl.P_charge_discharge_with_eff_t = pyo.Var(
            mdl.IDX1,
            domain=pyo.Reals,
            bounds=(-P_batt_MW / charge_efficiency, P_batt_MW * charge_efficiency),
        )
        mdl.battery_eff_constraint = pyo.Piecewise(
            mdl.IDX1,
            mdl.P_charge_discharge_with_eff_t,
            mdl.P_charge_discharge_t,
            pw_pts=f_piecewise_x_vals,
            f_rule=f_piecewise_y_vals,
            pw_constr_type="EQ",
            # pw_repn='SOS2',
            # pw_repn='CC',
            # pw_repn='DCC',
            pw_repn="DLOG",
            force_pw=False,
            warn_domain_coverage=False,
            unbounded_domain_var=True,
        )

        mdl.penalty = pyo.Var(mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))
        mdl.e_penalty = pyo.Var(mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))

        mdl.penalty_constraint.add(
            mdl.e_penalty[0]
            == (e_peak_period_expected - sum(mdl.P_HPP_t[t] for t in mdl.IDX3))
        )

        mdl.penalty_constraint_pw = pyo.Piecewise(
            mdl.IDX2,
            mdl.penalty,
            mdl.e_penalty,
            pw_pts=[-1e12, 0, 1e12],
            f_rule=[0, 0, price_peak * 1e12],
            pw_constr_type="EQ",
            # pw_repn='SOS2',
            # pw_repn='CC',
            # pw_repn='DCC',
            pw_repn="DLOG",
            force_pw=False,
            warn_domain_coverage=False,
            # unbounded_domain_var=True
        )

        # Delta_t of 1 hour
        dt = 1

        for t in range(0, len(time)):
            # Constraining battery energy level to maximum battery level
            mdl.battery_energy_constraint.add(mdl.E_SOC_t[t] <= E_batt_MWh_t.iloc[t])
            # Constraining battery energy level to minimum battery level
            mdl.battery_energy_min_constraint.add(
                mdl.E_SOC_t[t]
                >= (1 - battery_depth_of_discharge) * E_batt_MWh_t.iloc[t]
            )
            # print(battery_depth_of_discharge)

            # Power constraint
            mdl.power_constraint.add(
                mdl.P_HPP_t[t]
                == wind_ts.iloc[t]
                + solar_ts.iloc[t]
                - mdl.P_curtailment_t[t]
                + mdl.P_charge_discharge_with_eff_t[t]
            )

        # Battery dynamics with efficiency charge =! discharge
        for t in range(1, len(time)):
            mdl.battery_dynamics_constraint.add(
                mdl.E_SOC_t[t]
                == mdl.E_SOC_t[t - 1] - mdl.P_charge_discharge_with_eff_t[t] * dt
            )

        # Battery delta for battery operation constraints
        mdl.P_battery_delta = pyo.Var(mdl.IDX5, domain=pyo.Reals, bounds=(-1e12, 1e12))

        mdl.P_battery_delta_abs_pw = pyo.Var(
            mdl.IDX5, domain=pyo.Reals, bounds=(0, 1e12)
        )

        for t in mdl.IDX5:
            mdl.power_constraint.add(
                mdl.P_battery_delta[t]
                == mdl.P_charge_discharge_with_eff_t[t + 1]
                - mdl.P_charge_discharge_with_eff_t[t]
            )

        mdl.P_battery_delta_abs_pw_constr = pyo.Piecewise(
            mdl.IDX5,
            mdl.P_battery_delta_abs_pw,
            mdl.P_battery_delta,
            pw_pts=[-1e12, 0, 1e12],
            f_rule=[1e12, 0, 1e12],
            pw_constr_type="EQ",
            # pw_repn='SOS2',
            # pw_repn='CC',
            # pw_repn='DCC',
            pw_repn="DLOG",
            force_pw=False,
            warn_domain_coverage=False,
            # unbounded_domain_var=True
        )

        # Objective Function ##
        mdl.OBJ = pyo.Objective(
            # revenues and OPEX
            expr=sum(price_ts.iloc[t] * mdl.P_HPP_t[t] for t in mdl.IDX1)
            - mdl.penalty[0]
            + -sum(
                mdl.P_battery_delta_abs_pw[t]
                * cost_of_battery_P_fluct_in_peak_price_ratio
                * price_ts_to_max.iloc[t]
                for t in mdl.IDX5
            ),
            sense=pyo.maximize,
        )

        # solve with CPLEX via Pyomo
        solver = pyo.SolverFactory("cplex")
        solver.options["threads"] = 6
        result = solver.solve(mdl, tee=False)  # tee=True if you want solver log

        # helper to safely read an indexed Pyomo Var into a pandas Series
        def series_from_indexed_var(var, idx_list):
            values = {}
            for idx in idx_list:
                try:
                    values[idx] = pyo.value(var[idx])
                except Exception:
                    # if index missing or value not available, fill with 0 (consistent with your original handling)
                    values[idx] = 0.0
            return pd.Series(values)

        # Extract time series as pandas Series (index = your time items)
        P_HPP_ts_df = series_from_indexed_var(mdl.P_HPP_t, time)
        P_curtailment_ts_df = series_from_indexed_var(mdl.P_curtailment_t, time)
        P_charge_discharge_ts_df = series_from_indexed_var(
            mdl.P_charge_discharge_t, time
        )
        E_SOC_ts_df = series_from_indexed_var(mdl.E_SOC_t, SOCtime)

        # scalar penalty (assumes penalty is a scalar Var or Param)
        try:
            penalty_2 = pyo.value(mdl.penalty[0])
        except Exception:
            penalty_2 = 0.0

        # make a time series like P_HPP with a constant penalty
        penalty_ts = np.ones(N_t) * (penalty_2 / N_t)

        # CPLEX / Pyomo should give values for each index — but keep your reindex/fill behaviour
        P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values
        P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values
        P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(
            time, fill_value=0
        ).values
        E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values

        # fallback recompute if docplex-like missing behaviour happened in your original code
        if len(P_HPP_ts_df) < len(wind_ts):
            P_HPP_ts = (
                wind_ts.values
                + solar_ts.values
                + -P_curtailment_ts
                + P_charge_discharge_ts
            )

        # (optionally) do any mdl cleanup if desired — Pyomo doesn't need mdl.end()
        # return results to caller
        return {
            "P_HPP_ts": P_HPP_ts,
            "P_curtailment_ts": P_curtailment_ts,
            "penalty_ts": penalty_ts,
            "P_charge_discharge_ts": P_charge_discharge_ts,
            "E_SOC_ts": E_SOC_ts,
        }
        # return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
