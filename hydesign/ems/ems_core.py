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

            ems_core_defaults.cplex_init(mdl, batch, **kwargs)

            if "wind" in technology:
                ems_core_wind.cplex_init(mdl, batch, **kwargs)

            if "solar" in technology:
                ems_core_solar.cplex_init(mdl, batch, **kwargs)

            if "battery" in technology:
                ems_core_battery.cplex_init(mdl, batch, **kwargs)

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
            # from ortools.sat.python import cp_model

            # mdl = cp_model.CpModel()
            # # --- define variables and constraints here ---

            # solver = cp_model.CpSolver()
            # # equivalent of threads
            # solver.parameters.num_search_workers = 1

            # # There is no direct "mip emphasis" option in CP-SAT.
            # # You can influence solver behaviour instead with parameters:
            # solver.parameters.max_time_in_seconds = 60.0          # stop after N seconds
            # #solver.parameters.maximize = False                   # example; not an "emphasis"
            # solver.parameters.log_search_progress = True

            ems_core_defaults.cplex_init(batch, **kwargs)
            if "wind" in technology:
                ems_core_wind.cplex_init(batch, **kwargs)

            if "solar" in technology:
                ems_core_solar.cplex_init(batch, **kwargs)

            if "battery" in technology:
                ems_core_battery.cplex_init(batch, **kwargs)

            ems_core_defaults.ems_ortools_parts(**kwargs)
            # if "wind" in technology:
            #     ems_core_wind.ortools_var_constraints(mdl, batch, **kwargs)

            # if "solar" in technology:
            #     ems_core_solar.ortools_var_constraints(mdl, batch, **kwargs)

            # if "battery" in technology:
            #     ems_core_battery.ortools_var_constraints(mdl, batch, **kwargs)

            # # out_ems_batch.update(ems_core_wind.ortools(mdl, batch, **kwargs))
            # # out_ems_batch.update()
            # ems_core_defaults.ortools_opjective(mdl)
            # if "wind" in technology:
            #     ems_core_wind.ortools_opjective(mdl)

            # if "solar" in technology:
            #     ems_core_solar.ortools_opjective(mdl)

            # if "battery" in technology:
            #     ems_core_battery.ortools_opjective(mdl)

            # sol = solver.Solve(mdl)

            # ems_core_defaults.ortools_outputs(sol)
            # ems_core_wind.ortools_outputs(sol)
            # ems_core_solar.ortools_outputs(sol)
            # ems_core_battery.ortools_outputs(sol)

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

        # battery added for ortools
        self.E_batt_MWh_t = kwargs["E_batt_MWh_t"]
        self.P_charge_discharge_ts = np.zeros(self.n_ts)
        self.E_SOC_ts = np.zeros(self.n_ts + 1)

    def cplex_init(self, batch, **kwargs) -> Dict[str, Any]:
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
        # time set with an additional time slot for the last soc
        self.SOCtime_batch = self.time_batch.append(
            pd.Index([self.time_batch[-1] + pd.Timedelta("1hour")])
        )

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

    def ems_ortools_parts(self, **kwargs):
        """
        EMS solver implemented with OR-Tools linear solver.
        Returns:
            P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
        """
        # at top of file
        from ortools.linear_solver import pywraplp

        batch = self.batch
        wind_ts = self.wind_ts.iloc[batch]
        solar_ts = self.solar_ts.iloc[batch]
        price_ts = self.price_ts.iloc[batch]

        P_batt_MW = kwargs["P_batt_MW"]
        # self.hpp_grid_connection = kwargs["hpp_grid_connection"]
        battery_depth_of_discharge = kwargs["battery_depth_of_discharge"]
        charge_efficiency = kwargs["charge_efficiency"]
        cost_of_battery_P_fluct_in_peak_price_ratio = kwargs[
            "cost_of_battery_P_fluct_in_peak_price_ratio"
        ]
        E_batt_MWh_t = kwargs["E_batt_MWh_t"]
        hpp_grid_connection = self.hpp_grid_connection
        peak_hr_quantile = self.peak_hr_quantile
        n_full_power_hours_expected_per_day_at_peak_price = (
            self.n_full_power_hours_expected_per_day_at_peak_price
        )

        # self.batch = batch

        # self.time_batch = self.price_ts_batch.index

        time = price_ts.index
        N_t = len(time)
        N_days = N_t / 24.0
        e_peak_day_expected = (
            n_full_power_hours_expected_per_day_at_peak_price * hpp_grid_connection
        )
        e_peak_period_expected = e_peak_day_expected * N_days

        price_peak = float(np.quantile(price_ts.values, peak_hr_quantile))
        peak_hours_index = np.where(price_ts.values >= price_peak)[0]

        # price_ts_to_max like in original (peak - price, clipped at 0, smoothed half-next)
        price_ts_to_max = pd.Series(price_peak - price_ts.values, index=time)
        price_ts_to_max[price_ts_to_max < 0] = 0.0
        if N_t >= 2:
            arr = price_ts_to_max.values.copy()
            arr[:-1] = 0.5 * arr[:-1] + 0.5 * arr[1:]
            price_ts_to_max = pd.Series(arr, index=time)

        # Create linear solver (GLOP)
        solver = pywraplp.Solver.CreateSolver("GLOP")
        if solver is None:
            raise RuntimeError(
                "OR-Tools GLOP solver not available. Install ortools or choose another solver backend."
            )

        # Variables
        # HPP power
        P_HPP = {
            t: solver.NumVar(0.0, float(hpp_grid_connection), f"P_HPP_{i}")
            for i, t in enumerate(time)
        }
        # Curtailment
        P_curtail = {
            t: solver.NumVar(0.0, solver.infinity(), f"P_curt_{i}")
            for i, t in enumerate(time)
        }

        # Decompose battery power into discharge and charge parts (both >=0)
        # Bounds chosen to reflect original asymmetric bounds:
        # original ub for P (positive) = P_batt_MW * charge_efficiency
        # original lb for P (negative) = - P_batt_MW / charge_efficiency
        P_dis = {
            t: solver.NumVar(0.0, float(P_batt_MW * charge_efficiency), f"P_dis_{i}")
            for i, t in enumerate(time)
        }
        P_ch = {
            t: solver.NumVar(0.0, float(P_batt_MW / charge_efficiency), f"P_ch_{i}")
            for i, t in enumerate(time)
        }

        # SOC variables for each hour plus final hour
        SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))
        E_SOC = {
            t: solver.NumVar(0.0, float(E_batt_MWh_t.max()), f"E_SOC_{i}")
            for i, t in enumerate(SOCtime)
        }

        # penalty variables
        e_penalty = solver.NumVar(-solver.infinity(), solver.infinity(), "e_penalty")
        p_nonneg = solver.NumVar(
            0.0, solver.infinity(), "p_nonneg"
        )  # represents max(0, e_penalty)

        # delta variables for absolute battery power change cost
        delta = {}
        for i in range(N_t - 1):
            t = time[i]
            delta[t] = solver.NumVar(0.0, solver.infinity(), f"delta_{i}")

        # Objective: maximize sum(price * P_HPP) - penalty - sum(delta * cost_ratio * price_ts_to_max)
        objective = solver.Objective()
        for t in time:
            objective.SetCoefficient(P_HPP[t], float(price_ts.loc[t]))

        # penalty term (note penalty = price_peak * p_nonneg; objective subtracts it)
        objective.SetCoefficient(p_nonneg, -float(price_peak))

        # battery fluctuation cost
        for i in range(N_t - 1):
            t = time[i]
            coef = -float(
                cost_of_battery_P_fluct_in_peak_price_ratio * price_ts_to_max.loc[t]
            )
            objective.SetCoefficient(delta[t], coef)

        objective.SetMaximization()

        # Constraints

        # e_penalty == e_peak_period_expected - sum(P_HPP over peak hours)
        if len(peak_hours_index) > 0:
            expr = solver.Sum([P_HPP[time[int(idx)]] for idx in peak_hours_index])
        else:
            expr = solver.Sum([])
        solver.Add(e_penalty == float(e_peak_period_expected) - expr)

        # p_nonneg >= e_penalty  (and p_nonneg >= 0 by var bound)
        solver.Add(p_nonneg >= e_penalty)

        # initial and final SOC equal to 0.5 * E_batt at first time (same as original)
        solver.Add(E_SOC[SOCtime[0]] == 0.5 * float(E_batt_MWh_t.loc[time[0]]))
        solver.Add(E_SOC[SOCtime[-1]] == 0.5 * float(E_batt_MWh_t.loc[time[0]]))

        # Per-time constraints
        for i, t in enumerate(time):
            tt = time[i] + pd.Timedelta("1hour")
            dt = 1.0

            # power balance: P_HPP = wind + solar - curtail + (P_dis - P_ch)
            solver.Add(
                P_HPP[t]
                == float(wind_ts.loc[t])
                + float(solar_ts.loc[t])
                - P_curtail[t]
                + (P_dis[t] - P_ch[t])
            )

            # SOC transition with asymmetric efficiency:
            # E_SOC_next == E_SOC - (1/eff)*P_dis*dt + eff*P_ch*dt
            solver.Add(
                E_SOC[tt]
                == E_SOC[t]
                - (1.0 / float(charge_efficiency)) * P_dis[t] * dt
                + float(charge_efficiency) * P_ch[t] * dt
            )

            # SOC bounds for time t
            min_SOC = (1.0 - float(battery_depth_of_discharge)) * float(
                E_batt_MWh_t.loc[t]
            )
            max_SOC = float(E_batt_MWh_t.loc[t])
            solver.Add(E_SOC[t] >= min_SOC)
            solver.Add(E_SOC[t] <= max_SOC)

            # delta constraints for absolute differences
            if i < N_t - 1:
                t_next = time[i + 1]
                diff_expr = (P_dis[t_next] - P_ch[t_next]) - (P_dis[t] - P_ch[t])
                solver.Add(delta[t] >= diff_expr)
                solver.Add(delta[t] >= -diff_expr)

        # SOC bounds for final SOC node (redundant but included to match original)
        last_t = SOCtime[-1]
        solver.Add(E_SOC[last_t] <= float(E_batt_MWh_t.loc[time[-1]]))
        solver.Add(
            E_SOC[last_t]
            >= (1.0 - float(battery_depth_of_discharge))
            * float(E_batt_MWh_t.loc[time[-1]])
        )

        # Solve
        result_status = solver.Solve()
        if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(
                f"Solver failed or returned infeasible (status {result_status})."
            )

        # extract results
        P_HPP_ts_batch = np.array([P_HPP[t].solution_value() for t in time])
        P_curtail_ts_batch = np.array([P_curtail[t].solution_value() for t in time])
        P_charge_discharge_ts_batch = np.array(
            [(P_dis[t].solution_value() - P_ch[t].solution_value()) for t in time]
        )
        E_SOC_ts_batch = np.array([E_SOC[t].solution_value() for t in SOCtime])

        penalty_val_batch = float(price_peak * p_nonneg.solution_value())
        penalty_ts_batch = np.ones(N_t) * (penalty_val_batch / float(N_t))

        self.P_HPP_ts[batch] = P_HPP_ts_batch
        self.P_curtailment_ts[batch] = P_curtail_ts_batch
        self.penalty_ts[batch] = penalty_ts_batch
        self.P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[batch] = E_SOC_ts_batch[:-1]

        # return P_HPP_ts, P_curtail_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
