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


class EMSCoreInterface(ABC):
    """
    Abstract interface for EMS core classes.
    Concrete implementations must provide:
      - inputs
      - inputs_batch
      - outputs_batch
      - final_process
    """

    def __init__(self, **kwargs):
        self.wind_ts = kwargs["wind_ts"]

        self.solar_ts = kwargs["solar_ts"]

        self.hpp_grid_connection = kwargs.get("hpp_grid_connection", 0)
        self.price_ts = kwargs["price_ts"]
        self.peak_hr_quantile = kwargs.get("peak_hr_quantile", 0.9)
        self.n_full_power_hours_expected_per_day_at_peak_price = kwargs.get(
            "n_full_power_hours_expected_per_day_at_peak_price", 1
        )
        self.batch_size = kwargs.get("batch_size", 110)

        n_ts = len(self.price_ts)
        self.n_ts = n_ts

    def inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare and return the initial inputs / allocated arrays and meta values."""
        return {}

    def cplex(self, mdl, **kwargs) -> Dict[str, Any]:
        """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
        raise NotImplementedError

    # def pyomo(self, batch: Iterable[int], **kwargs) -> Dict[str, Any]:
    #     """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
    #     raise NotImplementedError

    # def ortools(self, batch: Iterable[int], **kwargs) -> Dict[str, Any]:
    #     """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
    #     raise NotImplementedError

    def cplex_var_constraints(self, solver):
        pass

    def cplex_opjective(self, mdl):
        pass

    def cplex_outputs(self, sol):
        pass

    def ortools_var_constraints(self, solver):
        pass

    def ortools_objective(self, solver):
        pass

    def ortools_outputs(self, sol):
        pass

    def final_process(self, **kwargs) -> Dict[str, Any]:
        """
        Final processing step.
        """
        return kwargs

    def get_final_outputs(self):
        return {}

    def update_constraint_rhs(self, mdl, constraint_name, added_value):
        constr = mdl.get_constraint_by_name(constraint_name)
        if constr is None:
            raise ValueError(f"Constraint '{constraint_name}' not found in the model.")

        lhs_expr = constr.lhs
        rhs_value = constr.rhs
        ct_type = constr.sense

        # Remove old constraint
        mdl.remove_constraint(constr)

        # Add new constraint with updated RHS
        if ct_type == ComparisonType.EQ:
            mdl.add_constraint(
                lhs_expr == rhs_value + added_value, ctname=constraint_name
            )
        elif ct_type == ComparisonType.LE:
            mdl.add_constraint(
                lhs_expr <= rhs_value + added_value, ctname=constraint_name
            )
        elif ct_type == ComparisonType.GE:
            mdl.add_constraint(
                lhs_expr >= rhs_value + added_value, ctname=constraint_name
            )
        else:
            raise ValueError(f"Unknown constraint sense: {ct_type}")

    def add_to_objective(self, mdl, addition_expr):
        """
        Add a new expression to the current model objective.

        Parameters
        ----------
        mdl : docplex.mp.model.Model
            The CPLEX model.
        addition_expr : docplex expression
            Expression to add to the existing objective.

        Returns
        -------
        None (updates model in-place)
        """
        # Get the existing objective expression and sense
        old_obj = mdl.objective_expr
        sense = mdl.objective_sense

        # Remove old objective
        mdl.remove_objective()

        # Add new objective: old + addition
        if sense.name.lower() == "maximize":
            mdl.maximize(old_obj + addition_expr)
        else:
            mdl.minimize(old_obj + addition_expr)

    def update_constraint_rhs_ortools(
        self, solver, constraint_name, addition_expr: float, coeff: float = 1
    ):
        """
        Add a numeric 'addition_expr' to the RHS of the named equality constraint.
        This function shifts the constraint's bounds (lb and ub) by addition_expr.
        """
        target = None
        for c in solver.constraints():
            if c.name() == constraint_name:
                target = c
                break
        if target is None:
            raise KeyError("constraint not found")

        target.SetCoefficient(addition_expr, -coeff)


def ems_cplex_parts(**kwargs):
    """EMS solver implemented in cplex

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
    wind_ts = kwargs["wind_ts_sel"]
    solar_ts = kwargs["solar_ts_sel"]
    price_ts = kwargs["price_ts_sel"]
    P_batt_MW = kwargs["P_batt_MW"]
    E_batt_MWh_t = kwargs["E_batt_MWh_t_sel"]
    hpp_grid_connection = kwargs["hpp_grid_connection"]
    battery_depth_of_discharge = kwargs["battery_depth_of_discharge"]
    charge_efficiency = kwargs["charge_efficiency"]
    peak_hr_quantile = kwargs["peak_hr_quantile"]
    cost_of_battery_P_fluct_in_peak_price_ratio = kwargs[
        "cost_of_battery_P_fluct_in_peak_price_ratio"
    ]
    n_full_power_hours_expected_per_day_at_peak_price = kwargs[
        "n_full_power_hours_expected_per_day_at_peak_price"
    ]

    # Penalties
    N_t = len(price_ts.index)
    N_days = N_t / 24
    e_peak_day_expected = (
        n_full_power_hours_expected_per_day_at_peak_price * hpp_grid_connection
    )
    e_peak_period_expected = e_peak_day_expected * N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts.values >= price_peak)[0]

    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max < 0] = 0
    price_ts_to_max.iloc[:-1] = (
        0.5 * price_ts_to_max.iloc[:-1].values + 0.5 * price_ts_to_max.iloc[1:].values
    )

    mdl = Model(name="EMS")
    mdl.context.cplex_parameters.threads = 1
    # CPLEX parameter pg 87 Emphasize feasibility over optimality
    mdl.context.cplex_parameters.emphasis.mip = 1
    # mdl.context.cplex_parameters.timelimit = 1e-2
    # mdl.context.cplex_parameters.mip.limits.strongit = 3
    # mdl.context.cplex_parameters.mip.strategy.search = 1 #  branch and cut strategy; disable dynamic

    # cpx = mdl.get_cplex()
    # cpx.parameters.mip.tolerances.integrality.set(0)
    # cpx.parameters.simplex.tolerances.markowitz.set(0.999)
    # cpx.parameters.simplex.tolerances.optimality.set(1e-6)#1e-9)
    # cpx.parameters.simplex.tolerances.feasibility.set(1e-5)#1e-9)
    # cpx.parameters.mip.pool.intensity.set(2)
    # cpx.parameters.mip.pool.absgap.set(1e75)
    # cpx.parameters.mip.pool.relgap.set(1e75)
    # cpx.parameters.mip.limits.populate.set(50)

    time = price_ts.index
    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))

    # Variables definition
    P_HPP_t = mdl.continuous_var_dict(
        time, lb=0, ub=hpp_grid_connection, name="HPP power output"
    )
    P_curtailment_t = mdl.continuous_var_dict(time, lb=0, name="Curtailment")

    # Power charge/discharge from battery
    # Lower bound as large negative number in order to allow the variable to
    # have either positive or negative values
    P_charge_discharge = mdl.continuous_var_dict(
        time,
        lb=-P_batt_MW / charge_efficiency,
        ub=P_batt_MW * charge_efficiency,
        name="Battery power",
    )
    # Battery energy level, energy stored
    E_SOC_t = mdl.continuous_var_dict(
        SOCtime,
        lb=0,
        name="Energy level",  # ub=E_batt_MWh_t.max(),
    )

    penalty = mdl.continuous_var(name="penalty", lb=-1e12)
    e_penalty = mdl.continuous_var(name="e_penalty", lb=-1e12)

    # Piecewise function for "absolute value" function
    fabs = mdl.piecewise(-1, [(0, 0)], 1)

    mdl.maximize(
        # revenues and OPEX
        mdl.sum(price_ts[t] * P_HPP_t[t] for t in time)
        - penalty  # Add cost for rapid charge-discharge for limiting the battery life use
        - mdl.sum(
            fabs(P_charge_discharge[t + pd.Timedelta("1hour")] - P_charge_discharge[t])
            * cost_of_battery_P_fluct_in_peak_price_ratio
            * price_ts_to_max[t]
            for t in time[:-1]
        )
    )

    # Constraints
    mdl.add_constraint(
        e_penalty
        == (
            e_peak_period_expected - mdl.sum(P_HPP_t[time[i]] for i in peak_hours_index)
        )
    )
    # Piecewise function for "only positive" function
    f1 = mdl.piecewise(0, [(0, 0)], 1)
    mdl.add_constraint(penalty == price_peak * f1(e_penalty))

    # Intitial and end SOC
    mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]])

    # SOC at the end of the year has to be equal to SOC at the beginning of the year
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]])

    # pircewise linear representation of charge vs dischrage effciency
    f2 = mdl.piecewise(charge_efficiency, [(0, 0)], 1 / charge_efficiency)

    for t in time:
        # Time index for successive time step
        tt = t + pd.Timedelta("1hour")
        # Delta_t of 1 hour
        dt = 1

        # Only one variable for battery
        mdl.add_constraint(
            P_HPP_t[t]
            == wind_ts[t] + solar_ts[t] + -P_curtailment_t[t] + P_charge_discharge[t]
        )

        # charge/dischrage equation
        mdl.add_constraint(E_SOC_t[tt] == E_SOC_t[t] - f2(P_charge_discharge[t]) * dt)

        # Constraining battery energy level to minimum battery level
        mdl.add_constraint(
            E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t]
        )

        # Constraining battery energy level to maximum battery level
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])

        # Battery charge/discharge within its power rating
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW * charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW / charge_efficiency)

    # Solving the problem
    sol = mdl.solve(log_output=False)
    # log_output=True)

    # print(mdl.export_to_string())
    # sol.display()

    P_HPP_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_HPP_t), orient="index"
    ).loc[:, 0]

    P_curtailment_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_curtailment_t), orient="index"
    ).loc[:, 0]

    P_charge_discharge_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_charge_discharge), orient="index"
    ).loc[:, 0]

    E_SOC_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(E_SOC_t), orient="index"
    ).loc[:, 0]

    # make a time series like P_HPP with a constant penalty
    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2 / N_t)

    mdl.end()

    # Cplex sometimes returns missing values :O
    P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values
    P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values
    P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values
    E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values

    if len(P_HPP_ts_df) < len(wind_ts):
        # print('recomputing p_hpp')
        P_HPP_ts = (
            wind_ts.values + solar_ts.values - P_curtailment_ts + P_charge_discharge_ts
        )

    return {
        "P_HPP_ts_batch": P_HPP_ts,
        "P_curtailment_ts_batch": P_curtailment_ts,
        "P_charge_discharge_ts_batch": P_charge_discharge_ts,
        "E_SOC_ts_batch": E_SOC_ts,
        "penalty_batch": penalty_ts,
    }


def ems_Pyomo_parts(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile,
    cost_of_battery_P_fluct_in_peak_price_ratio,  # [0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price,
):
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

    # extract parameters into the variable space
    # globals().update(self.__dict__)
    # Penalties
    N_t = len(price_ts.index)
    N_days = N_t / 24
    e_peak_day_expected = (
        n_full_power_hours_expected_per_day_at_peak_price * hpp_grid_connection
    )
    e_peak_period_expected = e_peak_day_expected * N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts >= price_peak)[0]

    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max < 0] = 0
    price_ts_to_max.iloc[:-1] = (
        0.5 * price_ts_to_max.iloc[:-1].values + 0.5 * price_ts_to_max.iloc[1:].values
    )

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
            mdl.E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t.iloc[t]
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

    mdl.P_battery_delta_abs_pw = pyo.Var(mdl.IDX5, domain=pyo.Reals, bounds=(0, 1e12))

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
    P_charge_discharge_ts_df = series_from_indexed_var(mdl.P_charge_discharge_t, time)
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
    P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values
    E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values

    # fallback recompute if docplex-like missing behaviour happened in your original code
    if len(P_HPP_ts_df) < len(wind_ts):
        P_HPP_ts = (
            wind_ts.values + solar_ts.values + -P_curtailment_ts + P_charge_discharge_ts
        )

    # (optionally) do any mdl cleanup if desired — Pyomo doesn't need mdl.end()
    # return results to caller
    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts


import numpy as np
import pandas as pd

# OR-Tools rewrite of your EMS CPLEX function
# Requires: pip install ortools
from ortools.linear_solver import pywraplp


def ems_ortools_parts(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.5,
    n_full_power_hours_expected_per_day_at_peak_price=3,
):
    """
    EMS solver implemented with OR-Tools linear solver.
    Returns:
        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
    """

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
        min_SOC = (1.0 - float(battery_depth_of_discharge)) * float(E_batt_MWh_t.loc[t])
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
        >= (1.0 - float(battery_depth_of_discharge)) * float(E_batt_MWh_t.loc[time[-1]])
    )

    # Solve
    result_status = solver.Solve()
    if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        raise RuntimeError(
            f"Solver failed or returned infeasible (status {result_status})."
        )

    # extract results
    P_HPP_ts = np.array([P_HPP[t].solution_value() for t in time])
    P_curtail_ts = np.array([P_curtail[t].solution_value() for t in time])
    P_charge_discharge_ts = np.array(
        [(P_dis[t].solution_value() - P_ch[t].solution_value()) for t in time]
    )
    E_SOC_ts = np.array([E_SOC[t].solution_value() for t in SOCtime])

    penalty_val = float(price_peak * p_nonneg.solution_value())
    penalty_ts = np.ones(N_t) * (penalty_val / float(N_t))

    return P_HPP_ts, P_curtail_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts


def ems_cplex_constantoutput(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.5,  # [0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price=3,
    batch_size=2 * 24,
    load_min=3,
    load_min_penalty_factor=1e6,
):
    # split in batches, ussually a week
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2] + batches_all[-1]

    # allocate vars
    P_HPP_ts = np.zeros(len(wind_ts))
    P_curtailment_ts = np.zeros(len(wind_ts))
    P_charge_discharge_ts = np.zeros(len(wind_ts))
    E_SOC_ts = np.zeros(len(wind_ts) + 1)
    penalty_ts = np.zeros(len(wind_ts))

    for ib, batch in enumerate(batches):
        wind_ts_sel = wind_ts.iloc[batch]
        solar_ts_sel = solar_ts.iloc[batch]
        price_ts_sel = price_ts.iloc[batch]
        E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch]

        # print(f'batch {ib+1} out of {len(batches)}')
        (
            P_HPP_ts_batch,
            P_curtailment_ts_batch,
            P_charge_discharge_ts_batch,
            E_SOC_ts_batch,
            penalty_batch,
        ) = ems_cplex_parts_constantoutput(
            wind_ts=wind_ts_sel,
            solar_ts=solar_ts_sel,
            price_ts=price_ts_sel,
            P_batt_MW=P_batt_MW,
            E_batt_MWh_t=E_batt_MWh_t_sel,
            hpp_grid_connection=hpp_grid_connection,
            battery_depth_of_discharge=battery_depth_of_discharge,
            charge_efficiency=charge_efficiency,
            peak_hr_quantile=peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price,
            load_min=load_min,
            load_min_penalty_factor=load_min_penalty_factor,
        )

        # print()
        # print()
        # print()
        # print(ib, len(batch))
        # print()
        # print('len(wind_ts_sel)',len(wind_ts_sel))
        # print('len(P_HPP_ts_batch)',len(P_HPP_ts_batch))
        # print('len(P_curtailment_ts_batch)',len(P_curtailment_ts_batch))
        # print('len(P_charge_discharge_ts_batch)',len(P_charge_discharge_ts_batch))
        # print('len(E_SOC_ts_batch)',len(E_SOC_ts_batch))
        # print('len(penalty_batch)',len(penalty_batch))

        P_HPP_ts[batch] = P_HPP_ts_batch
        P_curtailment_ts[batch] = P_curtailment_ts_batch
        P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        E_SOC_ts[batch] = E_SOC_ts_batch[:-1]
        penalty_ts[batch] = penalty_batch

    E_SOC_ts[-1] = E_SOC_ts[0]

    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts


def ems_cplex_parts_constantoutput(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.5,  # [0, 0.8]. For higher values might cause errors
    n_full_power_hours_expected_per_day_at_peak_price=3,
    load_min=2,  # MW
    load_min_penalty_factor=1e6,
):
    """EMS solver implemented in cplex

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
    load_min: minimum electrical load to meet [MW]
    load_min_penalty_factor: penalty factor to scale the penalty when not meeting required load

    Returns
    -------
    P_HPP_ts: HPP power time series
    P_curtailment_ts: HPP curtailed power time series
    P_charge_discharge_ts: Battery charge/discharge power time series
    E_SOC_ts: Battery energy SOC time series
    penalty_ts: penalty time series for not reaching expected energy production at peak hours
    """

    # Penalties
    N_t = len(price_ts.index)
    N_days = N_t / 24
    # e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*hpp_grid_connection
    # e_peak_period_expected = e_peak_day_expected*N_days
    # price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    # peak_hours_index = np.where(price_ts>=price_peak)[0]

    # price_ts_to_max = price_peak - price_ts
    # price_ts_to_max.loc[price_ts_to_max<0] = 0
    # price_ts_to_max.iloc[:-1] = 0.5*price_ts_to_max.iloc[:-1].values + 0.5*price_ts_to_max.iloc[1:].values

    mdl = Model(name="EMS")
    mdl.context.cplex_parameters.threads = 1
    # CPLEX parameter pg 87 Emphasize feasibility over optimality
    mdl.context.cplex_parameters.emphasis.mip = 1
    # mdl.context.cplex_parameters.timelimit = 1e-2
    # mdl.context.cplex_parameters.mip.limits.strongit = 3
    # mdl.context.cplex_parameters.mip.strategy.search = 1 #  branch and cut strategy; disable dynamic

    # cpx = mdl.get_cplex()
    # cpx.parameters.mip.tolerances.integrality.set(0)
    # cpx.parameters.simplex.tolerances.markowitz.set(0.999)
    # cpx.parameters.simplex.tolerances.optimality.set(1e-6)#1e-9)
    # cpx.parameters.simplex.tolerances.feasibility.set(1e-5)#1e-9)
    # cpx.parameters.mip.pool.intensity.set(2)
    # cpx.parameters.mip.pool.absgap.set(1e75)
    # cpx.parameters.mip.pool.relgap.set(1e75)
    # cpx.parameters.mip.limits.populate.set(50)

    time = price_ts.index

    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))

    # Variables definition
    P_HPP_t = mdl.continuous_var_dict(
        time, lb=0, ub=hpp_grid_connection, name="HPP power output"
    )
    P_curtailment_t = mdl.continuous_var_dict(time, lb=0, name="Curtailment")

    # Power charge/discharge from battery
    # Lower bound as large negative number in order to allow the variable to
    # have either positive or negative values
    P_charge_discharge = mdl.continuous_var_dict(
        time,
        lb=-P_batt_MW / charge_efficiency,
        ub=P_batt_MW * charge_efficiency,
        name="Battery power",
    )
    # Battery energy level, energy stored
    E_SOC_t = mdl.continuous_var_dict(
        SOCtime,
        lb=0,
        name="Energy level",  # ub=E_batt_MWh_t.max(),
    )

    P_constant = mdl.continuous_var_dict(
        range(int(N_days)), lb=0, name="constant output"
    )

    penalty = mdl.continuous_var(name="penalty", lb=-1e12)
    # e_penalty = mdl.continuous_var(name='e_penalty', lb=-1e12)

    # Piecewise function for "absolute value" function
    # fabs = mdl.piecewise(-1, [(0,0)], 1)

    # Piecewise function for "only negative" function
    fneg = mdl.piecewise(-1, [(0, 0)], 0)
    mdl.maximize(
        # revenues and OPEX
        mdl.sum(price_ts[t] * P_HPP_t[t] for t in time)
        - penalty
        # Add cost for rapid charge-discharge for limiting the battery life use
        # - mdl.sum(
        #   fabs(P_charge_discharge[t + pd.Timedelta('1hour')] - \
        #        P_charge_discharge[t])*cost_of_battery_P_fluct_in_peak_price_ratio*price_ts_to_max[t]
        #   for t in time[:-1])
    )

    # Constraints
    # mdl.add_constraint(
    #   e_penalty == ( e_peak_period_expected - mdl.sum(P_HPP_t[time[i]] for i in peak_hours_index) )
    #   )
    # Piecewise function for "only positive" function
    # f1 = mdl.piecewise(0, [(0,0)], 1)
    mdl.add_constraint(
        penalty
        == mdl.sum(load_min_penalty_factor * fneg(P_HPP_t[t] - load_min) for t in time)
    )

    # Intitial and end SOC
    mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]])

    # SOC at the end of the year has to be equal to SOC at the beginning of the year
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]])

    # pircewise linear representation of charge vs dischrage effciency
    f2 = mdl.piecewise(charge_efficiency, [(0, 0)], 1 / charge_efficiency)

    for t in time:
        # Time index for successive time step
        tt = t + pd.Timedelta("1hour")
        # Delta_t of 1 hour
        dt = 1

        # Only one variable for battery
        mdl.add_constraint(
            P_HPP_t[t]
            == wind_ts[t] + solar_ts[t] + -P_curtailment_t[t] + P_charge_discharge[t]
        )

        # charge/dischrage equation
        mdl.add_constraint(E_SOC_t[tt] == E_SOC_t[t] - f2(P_charge_discharge[t]) * dt)

        # Constraining battery energy level to minimum battery level
        mdl.add_constraint(
            E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t]
        )

        # Constraining battery energy level to maximum battery level
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])

        # Battery charge/discharge within its power rating
        mdl.add_constraint(P_charge_discharge[t] <= P_batt_MW * charge_efficiency)
        mdl.add_constraint(P_charge_discharge[t] >= -P_batt_MW / charge_efficiency)

        mdl.add_constraint(P_HPP_t[t] == P_constant[int(time.get_loc(t) / 24)])

    # Solving the problem
    sol = mdl.solve(log_output=False)
    # log_output=True)
    aa = mdl.get_solve_details()
    # print(aa.status)
    # if not aa.status=='integer optimal solution':
    #   print(aa.status)
    #   print(wind_ts)
    #   print(solar_ts)
    # print(mdl.export_to_string())
    # sol.display()

    P_HPP_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_HPP_t), orient="index"
    ).loc[:, 0]

    P_curtailment_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_curtailment_t), orient="index"
    ).loc[:, 0]

    P_charge_discharge_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_charge_discharge), orient="index"
    ).loc[:, 0]

    E_SOC_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(E_SOC_t), orient="index"
    ).loc[:, 0]

    # make a time series like P_HPP with a constant penalty
    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2 / N_t)

    mdl.end()

    # Cplex sometimes returns missing values :O
    P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values
    P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values
    P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values
    E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values

    if len(P_HPP_ts_df) < len(wind_ts):
        # print('recomputing p_hpp')
        P_HPP_ts = (
            wind_ts.values + solar_ts.values + -P_curtailment_ts + P_charge_discharge_ts
        )

    return P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts


def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N : end]]
        counter += 1
    return batch
