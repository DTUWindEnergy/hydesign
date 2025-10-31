# basic libraries
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from hydesign.ems_modular.modules.ems_core_helpers import (
    EMSCoreInterface,
    series_from_indexed_var,
)


class EMSCoreBattery(EMSCoreInterface):
    """
    EMSCoreBattery
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        self.P_batt_MW = kwargs["P_batt_MW"]
        self.battery_depth_of_discharge = kwargs["battery_depth_of_discharge"]
        self.charge_efficiency = kwargs["charge_efficiency"]
        self.cost_of_battery_P_fluct_in_peak_price_ratio = kwargs.get(
            "cost_of_battery_P_fluct_in_peak_price_ratio", 0.5
        )
        self.E_batt_MWh_t = kwargs["E_batt_MWh_t"]
        self.P_charge_discharge_ts = np.zeros(self.n_ts)
        self.E_SOC_ts = np.zeros(self.n_ts + 1)

    def final_process(self, **kwargs) -> Dict[str, Any]:
        # set last SOC element equal to initial SOC.
        kwargs["E_SOC_ts"][-1] = kwargs["E_SOC_ts"][0]
        # return kwargs

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        self.E_batt_MWh_t_batch = self.E_batt_MWh_t.iloc[batch]

        # time set with an additional time slot for the last soc
        self.SOCtime_batch = self.time_batch.append(
            pd.Index([self.time_batch[-1] + pd.Timedelta("1hour")])
        )

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        # Power charge/discharge from battery
        # Lower bound as large negative number in order to allow the variable to
        # have either positive or negative values
        self.P_charge_discharge_batch = mdl.continuous_var_dict(
            self.time_batch,
            lb=-self.P_batt_MW / self.charge_efficiency,
            ub=self.P_batt_MW * self.charge_efficiency,
            name="Battery power",
        )
        # Battery energy level, energy stored
        self.E_SOC_t_batch = mdl.continuous_var_dict(
            self.SOCtime_batch,
            lb=0,
            name="Energy level",  # ub=E_batt_MWh_t.max(),
        )

    def ortools_variables(self, solver):
        time_batch = self.time_batch
        SOCtime_batch = self.SOCtime_batch

        # Decompose battery power into discharge and charge parts (both >=0)
        # Bounds chosen to reflect original asymmetric bounds:
        # original ub for P (positive) = P_batt_MW * self.charge_efficiency
        # original lb for P (negative) = - P_batt_MW / self.charge_efficiency
        P_dis_batch = {
            t: solver.NumVar(
                0.0, float(self.P_batt_MW * self.charge_efficiency), f"P_dis_{i}"
            )
            for i, t in enumerate(time_batch)
        }
        P_ch_batch = {
            t: solver.NumVar(
                0.0, float(self.P_batt_MW / self.charge_efficiency), f"P_ch_{i}"
            )
            for i, t in enumerate(time_batch)
        }

        # SOC variables for each hour plus final hour
        E_SOC_batch = {
            t: solver.NumVar(0.0, float(self.E_batt_MWh_t_batch.max()), f"E_SOC_{i}")
            for i, t in enumerate(SOCtime_batch)
        }

        delta = {}
        for i in range(self.N_t_batch - 1):
            t = time_batch[i]
            delta[t] = solver.NumVar(0.0, solver.infinity(), f"delta_{i}")

        self.delta = delta

        self.P_dis_batch = P_dis_batch
        self.P_ch_batch = P_ch_batch
        self.E_SOC_batch = E_SOC_batch

    def pyomo_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        time_batch = self.time_batch
        SOCtime_batch = self.SOCtime_batch
        P_batt_MW = self.P_batt_MW
        charge_efficiency = self.charge_efficiency

        peak_hours_index_batch = self.peak_hours_index_batch
        mdl.IDX3 = [t for t in range(len(time_batch)) if t in peak_hours_index_batch]
        mdl.IDX4 = range(len(SOCtime_batch))
        mdl.IDX5 = range(len(time_batch) - 1)
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

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        # Intitial and end SOC
        mdl.add_constraint(
            self.E_SOC_t_batch[self.SOCtime_batch[0]]
            == 0.5 * self.E_batt_MWh_t_batch[self.time_batch[0]]
        )

        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        mdl.add_constraint(
            self.E_SOC_t_batch[self.SOCtime_batch[-1]]
            == 0.5 * self.E_batt_MWh_t_batch[self.time_batch[0]]
        )

        # pircewise linear representation of charge vs dischrage effciency
        self.f2 = mdl.piecewise(
            self.charge_efficiency, [(0, 0)], 1 / self.charge_efficiency
        )

        for t in self.time_batch:
            # Time index for successive time step
            tt = t + pd.Timedelta("1hour")
            # Delta_t of 1 hour
            dt = 1

            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.P_charge_discharge_batch[t],
            )

            # charge/dischrage equation
            mdl.add_constraint(
                self.E_SOC_t_batch[tt]
                == self.E_SOC_t_batch[t]
                - self.f2(self.P_charge_discharge_batch[t]) * dt
            )

            # Constraining battery energy level to minimum battery level
            mdl.add_constraint(
                self.E_SOC_t_batch[t]
                >= (1 - self.battery_depth_of_discharge) * self.E_batt_MWh_t_batch[t]
            )

            # Constraining battery energy level to maximum battery level
            mdl.add_constraint(self.E_SOC_t_batch[t] <= self.E_batt_MWh_t_batch[t])

            # Battery charge/discharge within its power rating
            mdl.add_constraint(
                self.P_charge_discharge_batch[t]
                <= self.P_batt_MW * self.charge_efficiency
            )
            mdl.add_constraint(
                self.P_charge_discharge_batch[t]
                >= -self.P_batt_MW / self.charge_efficiency
            )

    def ortools_constraints(self, solver):
        time_batch = self.time_batch
        N_t_batch = self.N_t_batch
        SOCtime_batch = self.SOCtime_batch
        E_SOC_batch = self.E_SOC_batch
        P_dis_batch = self.P_dis_batch
        P_ch_batch = self.P_ch_batch

        # delta variables for absolute battery power change cost
        delta = self.delta

        # initial and final SOC equal to 0.5 * E_batt at first time (same as original)
        solver.Add(
            E_SOC_batch[SOCtime_batch[0]]
            == 0.5 * float(self.E_batt_MWh_t_batch.loc[time_batch[0]])
        )
        solver.Add(
            E_SOC_batch[SOCtime_batch[-1]]
            == 0.5 * float(self.E_batt_MWh_t_batch.loc[time_batch[0]])
        )

        # Per-time constraints
        for i, t in enumerate(time_batch):
            tt = time_batch[i] + pd.Timedelta("1hour")
            dt = 1.0

            self.update_constraint_rhs_ortools(
                solver,
                constraint_name=f"P_HPP_t_balance{t}",
                addition_expr=P_dis_batch[t],
            )  # - P_ch_batch[t])
            self.update_constraint_rhs_ortools(
                solver,
                constraint_name=f"P_HPP_t_balance{t}",
                addition_expr=P_ch_batch[t],
                coeff=-1,
            )

            # SOC transition with asymmetric efficiency:
            # E_SOC_next == E_SOC - (1/eff)*P_dis*dt + eff*P_ch*dt
            solver.Add(
                E_SOC_batch[tt]
                == E_SOC_batch[t]
                - (1.0 / float(self.charge_efficiency)) * P_dis_batch[t] * dt
                + float(self.charge_efficiency) * P_ch_batch[t] * dt
            )

            # SOC bounds for time t
            min_SOC_batch = (1.0 - float(self.battery_depth_of_discharge)) * float(
                self.E_batt_MWh_t_batch.loc[t]
            )
            max_SOC_batch = float(self.E_batt_MWh_t_batch.loc[t])
            solver.Add(E_SOC_batch[t] >= min_SOC_batch)
            solver.Add(E_SOC_batch[t] <= max_SOC_batch)

            # delta constraints for absolute differences
            if i < N_t_batch - 1:
                t_next = time_batch[i + 1]
                diff_expr_batch = (P_dis_batch[t_next] - P_ch_batch[t_next]) - (
                    P_dis_batch[t] - P_ch_batch[t]
                )
                solver.Add(delta[t] >= diff_expr_batch)
                solver.Add(delta[t] >= -diff_expr_batch)

        # SOC bounds for final SOC node (redundant but included to match original)
        last_t = SOCtime_batch[-1]
        solver.Add(
            E_SOC_batch[last_t] <= float(self.E_batt_MWh_t_batch.loc[time_batch[-1]])
        )
        solver.Add(
            E_SOC_batch[last_t]
            >= (1.0 - float(self.battery_depth_of_discharge))
            * float(self.E_batt_MWh_t_batch.loc[time_batch[-1]])
        )

    def pyomo_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        time_batch = self.time_batch
        SOCtime_batch = self.SOCtime_batch

        battery_depth_of_discharge = self.battery_depth_of_discharge
        E_batt_MWh_t_batch = self.E_batt_MWh_t_batch

        for t in range(0, len(time_batch)):
            # battery
            self.update_constraint_rhs_pyomo(
                mdl,
                "power_constraint",
                added_value=mdl.P_charge_discharge_with_eff_t[t],
                idx=t,
            )

        mdl.charge_discharge_constraint = pyo.ConstraintList()
        mdl.battery_energy_constraint = pyo.ConstraintList()
        mdl.battery_energy_min_constraint = pyo.ConstraintList()
        mdl.battery_dynamics_constraint = pyo.ConstraintList()

        mdl.SOC_initial_condition = pyo.Constraint(
            expr=mdl.E_SOC_t[0] == 0.5 * E_batt_MWh_t_batch.iloc[0]
        )

        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        mdl.SOC_final = pyo.Constraint(
            expr=mdl.E_SOC_t[len(time_batch) - 1]
            == 0.5 * E_batt_MWh_t_batch.iloc[len(time_batch) - 1]
        )

        # Delta_t of 1 hour
        dt = 1
        for t in range(0, len(time_batch)):
            # Constraining battery energy level to maximum battery level
            mdl.battery_energy_constraint.add(
                mdl.E_SOC_t[t] <= E_batt_MWh_t_batch.iloc[t]
            )
            # Constraining battery energy level to minimum battery level
            mdl.battery_energy_min_constraint.add(
                mdl.E_SOC_t[t]
                >= (1 - battery_depth_of_discharge) * E_batt_MWh_t_batch.iloc[t]
            )
            # print(battery_depth_of_discharge)

        # Battery dynamics with efficiency charge =! discharge
        for t in range(1, len(time_batch)):
            mdl.battery_dynamics_constraint.add(
                mdl.E_SOC_t[t]
                == mdl.E_SOC_t[t - 1] - mdl.P_charge_discharge_with_eff_t[t] * dt
            )

        # Battery delta for battery operation constraints
        mdl.P_battery_delta = pyo.Var(mdl.IDX5, domain=pyo.Reals, bounds=(-1e12, 1e12))

        mdl.P_battery_delta_abs_pw = pyo.Var(
            mdl.IDX5, domain=pyo.Reals, bounds=(0, 1e12)
        )

        # for t in mdl.IDX5:
        #     mdl.power_constraint.add(
        #         mdl.P_battery_delta[t]
        #         == mdl.P_charge_discharge_with_eff_t[t + 1]
        #         - mdl.P_charge_discharge_with_eff_t[t]
        #     )

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

    def cplex_opjective(self, mdl):
        # Piecewise function for "absolute value" function
        self.fabs = mdl.piecewise(-1, [(0, 0)], 1)
        battery_penalty = -mdl.sum(
            self.fabs(
                self.P_charge_discharge_batch[t + pd.Timedelta("1hour")]
                - self.P_charge_discharge_batch[t]
            )
            * self.cost_of_battery_P_fluct_in_peak_price_ratio
            * self.price_ts_to_max_batch[t]
            for t in self.time_batch[:-1]
        )

        self.add_to_objective(mdl, addition_expr=battery_penalty)

    def ortools_objective(self, solver):
        objective = solver.Objective()

        # battery fluctuation cost
        for i in range(self.N_t_batch - 1):
            t = self.time_batch[i]
            coef = -float(
                self.cost_of_battery_P_fluct_in_peak_price_ratio
                * self.price_ts_to_max_batch.loc[t]
            )
            objective.SetCoefficient(self.delta[t], coef)

        objective.SetMaximization()

    def pyomo_opjective(self, mdl, **kwargs):
        cost_of_battery_P_fluct_in_peak_price_ratio = (
            self.cost_of_battery_P_fluct_in_peak_price_ratio
        )

        price_ts_to_max = self.price_ts_to_max_batch

        # battery
        battery_cost = sum(
            mdl.P_battery_delta_abs_pw[t]
            * cost_of_battery_P_fluct_in_peak_price_ratio
            * price_ts_to_max.iloc[t]
            for t in mdl.IDX5
        )

        self.add_to_objective_pyomo(mdl, addition_expr=-battery_cost)

    def cplex_outputs(self, sol):
        P_charge_discharge_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_charge_discharge_batch), orient="index"
        ).loc[:, 0]

        E_SOC_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.E_SOC_t_batch), orient="index"
        ).loc[:, 0]

        # Cplex sometimes returns missing values :O
        P_charge_discharge_ts_batch = P_charge_discharge_ts_df.reindex(
            self.time_batch, fill_value=0
        ).values
        E_SOC_ts_batch = E_SOC_ts_df.reindex(self.SOCtime_batch, fill_value=0).values

        self.P_charge_discharge_ts[self.batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[self.batch] = E_SOC_ts_batch[:-1]

    def ortools_outputs(self):
        SOCtime_batch = self.SOCtime_batch
        P_dis_batch = self.P_dis_batch
        P_ch_batch = self.P_ch_batch
        E_SOC_batch = self.E_SOC_batch

        P_charge_discharge_ts_batch = np.array(
            [
                (P_dis_batch[t].solution_value() - P_ch_batch[t].solution_value())
                for t in self.time_batch
            ]
        )
        E_SOC_ts_batch = np.array(
            [E_SOC_batch[t].solution_value() for t in SOCtime_batch]
        )

        self.P_charge_discharge_ts[self.batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[self.batch] = E_SOC_ts_batch[:-1]

    def pyomo_outputs(self, mdl, **kwargs):
        batch = self.batch
        time_batch = self.time_batch
        batch = self.batch

        # Extract time series as pandas Series (index = your time_batch items)

        P_charge_discharge_ts_batch_df = series_from_indexed_var(
            mdl.P_charge_discharge_t, time_batch
        )
        E_SOC_ts_batch_df = series_from_indexed_var(mdl.E_SOC_t, self.SOCtime_batch)

        P_charge_discharge_ts_batch = P_charge_discharge_ts_batch_df.reindex(
            time_batch, fill_value=0
        ).values
        E_SOC_ts_batch = E_SOC_ts_batch_df.reindex(
            self.SOCtime_batch, fill_value=0
        ).values

        self.P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[batch] = E_SOC_ts_batch[:-1]

    def update_final_outputs(self, out):
        out_battery = {
            "P_charge_discharge_ts": self.P_charge_discharge_ts,
            "E_SOC_ts": self.E_SOC_ts,
        }
        out.update(out_battery)


def ems_cplex_parts_P2X(
    # wind_ts,
    # solar_ts,
    # price_ts,
    # P_batt_MW,
    # E_batt_MWh_t,
    # hpp_grid_connection,
    # battery_depth_of_discharge,
    # charge_efficiency,
    price_H2,
    ptg_MW,
    HSS_kg,
    storage_eff,
    hhv,
    m_H2_demand_ts,
    H2_storage_t,
    penalty_factor_H2,
    eff_curve,
    min_power_standby,
    load_min_penalty_factor=1e6,
    # peak_hr_quantile=0.9,
    # cost_of_battery_P_fluct_in_peak_price_ratio=0.5,  # [0, 0.8]. For higher values might cause errors
    # n_full_power_hours_expected_per_day_at_peak_price=3,
    electrolyzer_eff_curve_type="production",
):

    # --- variables

    # Hydrogen storgae level
    LoS_H2_t = mdl.continuous_var_dict(SOCtime, lb=0, name="Hydrogen storage level")

    # Power to gas plant power consumption, produced hydrogen, electrolyzer efficiency
    P_ptg_t = mdl.continuous_var_dict(
        time,
        lb=min_power_standby * ptg_MW,
        ub=ptg_MW,
        name="Power to gas plant consumption",
    )

    m_H2_t = mdl.continuous_var_dict(time, lb=0, name="Produced hydrogen")
    m_H2_offtake_t = mdl.continuous_var_dict(time, lb=0, name="Hydrogen offtake")

    # --- constraints

    # # Piecewise function for "only positive" function
    f1 = mdl.piecewise(0, [(0, 0)], 1)

    mdl.add_constraint(
        penalty
        == price_peak * f1(e_penalty)
        + mdl.sum(
            penalty_factor_H2 * (m_H2_demand_ts[t] - m_H2_offtake_t[t]) for t in time
        )
    )

    # Intitial and end LoS_H2
    mdl.add_constraint(LoS_H2_t[SOCtime[0]] == 0)
    mdl.add_constraint(LoS_H2_t[SOCtime[-1]] == 0)

    # piecewise linear representation of H2 storage vs offtake effciency
    f3 = mdl.piecewise(1 / storage_eff, [(0, 0)], storage_eff)

    # Caclulating electrolyzer efficiency as a function of load (piecewise linear approximation)
    if electrolyzer_eff_curve_type == "production":
        H2_curve_list = [(load * ptg_MW, H2 * ptg_MW) for load, H2 in eff_curve]
    elif electrolyzer_eff_curve_type == "efficiency":
        H2_curve_list = [
            (load * ptg_MW, load * ptg_MW * efficiency / hhv * 1000)
            for load, efficiency in eff_curve
        ]
    else:
        raise ValueError(
            f'electrolyzer_eff_curve_type is: "{electrolyzer_eff_curve_type}". Available options are ["production", "efficiency"]'
        )

    H2_prod = mdl.piecewise(0, H2_curve_list, 0)

    for t in time:
        # Time index for successive time step
        tt = t + pd.Timedelta("1hour")
        # Delta_t of 1 hour
        dt = 1

        # Only one variable for battery
        mdl.add_constraint(
            P_HPP_t[t]
            == wind_ts[t]
            + solar_ts[t]
            - P_curtailment_t[t]
            + P_charge_discharge[t]
            - P_ptg_t[t]
        )

        # Hydrogen storgae equation
        mdl.add_constraint(
            LoS_H2_t[tt] == LoS_H2_t[t] + f3(m_H2_t[t] - m_H2_offtake_t[t])
        )

        # Constraining hydrogen offtake as per the demand time series and level of storage
        mdl.add_constraint(LoS_H2_t[t] <= H2_storage_t[t])
        mdl.add_constraint(m_H2_offtake_t[t] <= m_H2_demand_ts[t])

        # when the H2 offtake is infinite, there is no storage, then H2_offtake is same as H2_produced
        if H2_storage_t[t] == 0:
            m_H2_offtake_t[t] = m_H2_t[t]

        # Calculating Hydrogen production with HPC
        mdl.add_constraint(m_H2_t[t] == H2_prod(P_ptg_t[t]))

    # --- objective
    mdl.maximize(
        # revenues and OPEX
        mdl.sum(price_ts[t] * P_HPP_t[t] + price_H2 * m_H2_offtake_t[t] for t in time)
        - penalty  # Add cost for rapid charge-discharge for limiting the battery life use
        - mdl.sum(
            fabs(P_charge_discharge[t + pd.Timedelta("1hour")] - P_charge_discharge[t])
            * cost_of_battery_P_fluct_in_peak_price_ratio
            * price_ts_to_max[t]
            for t in time[:-1]
        )
    )

    # Solving the problem
    sol = mdl.solve(log_output=False)
    # log_output=True)

    # --- outputs
    P_ptg_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_ptg_t), orient="index"
    ).loc[:, 0]
    m_H2_ts_df = pd.DataFrame.from_dict(sol.get_value_dict(m_H2_t), orient="index").loc[
        :, 0
    ]
    LoS_H2_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(LoS_H2_t), orient="index"
    ).loc[:, 0]
    m_H2_offtake_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(m_H2_offtake_t), orient="index"
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
    P_ptg_ts = P_ptg_ts_df.reindex(time, fill_value=0).values
    m_H2_ts = m_H2_ts_df.reindex(time, fill_value=0).values
    LoS_H2_ts = LoS_H2_ts_df.reindex(time, fill_value=0).values
    m_H2_offtake_ts = m_H2_offtake_ts_df.reindex(time, fill_value=0).values

    if len(P_HPP_ts_df) < len(wind_ts):
        # print('recomputing p_hpp')
        P_HPP_ts = (
            wind_ts.values
            + solar_ts.values
            + -P_curtailment_ts
            + P_charge_discharge_ts
            - P_ptg_ts
        )
    P_ptg_SB_ts = np.zeros_like(P_ptg_ts)
    for t in range(N_t):
        if abs(P_ptg_ts[t] - min_power_standby * ptg_MW) < 1e-6:
            P_ptg_SB_ts[t] = min_power_standby * ptg_MW
            P_ptg_ts[t] = 0
            m_H2_ts[t] = 0

    return (
        P_HPP_ts,
        P_curtailment_ts,
        P_charge_discharge_ts,
        P_ptg_ts,
        P_ptg_SB_ts,
        E_SOC_ts,
        m_H2_ts,
        m_H2_offtake_ts,
        LoS_H2_ts,
        penalty_ts,
    )
