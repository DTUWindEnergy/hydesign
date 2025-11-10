# Auto-generated combined module
# Generated from: ems_core_defaults.py, ems_core_wind.py, ems_core_solar.py, ems_core_battery.py, ems_core_p2x.py

from typing import Any, Dict

import numpy as np
import pandas as pd
import pyomo.environ as pyo
from docplex.mp.model import Model
from ortools.linear_solver import pywraplp

from hydesign.ems_modular.modules.ems_core_helpers import (
    EMSCoreABC,
    series_from_indexed_var,
    split_in_batch,
)


class EMSCore(EMSCoreABC):
    """Auto-generated combined class (merged bodies of multiple classes)

    NOTE: Please review for semantic issues.
    """

    def __init__(self, **kwargs):
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
        self.P_HPP_ts = np.zeros(n_ts)
        self.P_curtailment_ts = np.zeros(n_ts)
        self.penalty_ts = np.zeros(n_ts)

        self.batches = split_in_batch(list(range(n_ts)), self.batch_size)

        # --- content from ems_core_wind.py ---
        self.wind_ts = kwargs["wind_ts"]

        # --- content from ems_core_solar.py ---
        self.solar_ts = kwargs["solar_ts"]

        # --- content from ems_core_battery.py ---
        self.P_batt_MW = kwargs["P_batt_MW"]
        self.battery_depth_of_discharge = kwargs["battery_depth_of_discharge"]
        self.charge_efficiency = kwargs["charge_efficiency"]
        self.cost_of_battery_P_fluct_in_peak_price_ratio = kwargs.get(
            "cost_of_battery_P_fluct_in_peak_price_ratio", 0.5
        )
        self.E_batt_MWh_t = kwargs["E_batt_MWh_t"]
        self.P_charge_discharge_ts = np.zeros(self.n_ts)
        self.E_SOC_ts = np.zeros(self.n_ts + 1)

        # --- content from ems_core_p2x.py ---
        self.price_H2 = kwargs["price_H2"]
        self.ptg_MW = kwargs["ptg_MW"]
        self.HSS_kg = kwargs["HSS_kg"]
        self.storage_eff = kwargs["storage_eff"]
        self.hhv = kwargs["hhv"]
        self.m_H2_demand_ts = kwargs["m_H2_demand_ts"]
        self.H2_storage_t = kwargs["H2_storage_t"]
        self.penalty_factor_H2 = kwargs["penalty_factor_H2"]
        self.eff_curve = kwargs["eff_curve"]
        self.min_power_standby = kwargs["min_power_standby"]
        self.load_min_penalty_factor = kwargs.get("load_min_penalty_factor", 1e6)
        self.electrolyzer_eff_curve_type = kwargs.get(
            "electrolyzer_eff_curve_type", "production"
        )

        self.LoS_H2_ts = np.zeros(self.n_ts)
        self.P_ptg_ts = np.zeros(self.n_ts)
        self.P_ptg_SB_ts = np.zeros_like(self.P_ptg_ts)
        self.m_H2_ts = np.zeros(self.n_ts)
        self.m_H2_offtake_ts = np.zeros(self.n_ts)

    def solve(self, **kwargs):
        """ """
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

    def batch_data(self, batch):
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

        # --- content from ems_core_wind.py ---
        self.wind_ts_batch = self.wind_ts.iloc[batch]

        # --- content from ems_core_solar.py ---
        self.solar_ts_batch = self.solar_ts.iloc[batch]

        # --- content from ems_core_battery.py ---
        self.E_batt_MWh_t_batch = self.E_batt_MWh_t.iloc[batch]

        # --- content from ems_core_p2x.py ---
        self.price_H2_batch = self.price_H2  # .iloc[batch]
        self.m_H2_demand_ts_batch = self.m_H2_demand_ts.iloc[batch]
        self.H2_storage_t_batch = self.H2_storage_t.iloc[batch]

    def cplex(self, **kwarg):
        """ """
        time_batch = self.time_batch
        SOCtime_batch = self.SOCtime_batch
        # create cplex model
        mdl = Model(name="EMS")
        mdl.context.cplex_parameters.threads = 1
        mdl.context.cplex_parameters.emphasis.mip = 1

        # ------------
        # Variables
        # ------------
        # --- defaults
        P_HPP_t_batch = mdl.continuous_var_dict(
            time_batch, lb=0, ub=self.hpp_grid_connection, name="HPP power output"
        )
        P_curtailment_t_batch = mdl.continuous_var_dict(
            time_batch, lb=0, name="Curtailment"
        )

        penalty_el_batch = mdl.continuous_var(name="penalty", lb=-1e12)
        e_penalty_batch = mdl.continuous_var(name="e_penalty", lb=-1e12)

        # --- content from ems_core_battery.py ---
        # Power charge/discharge from battery
        # Lower bound as large negative number in order to allow the variable to
        # have either positive or negative values
        P_charge_discharge_batch = mdl.continuous_var_dict(
            time_batch,
            lb=-self.P_batt_MW / self.charge_efficiency,
            ub=self.P_batt_MW * self.charge_efficiency,
            name="Battery power",
        )
        # Battery energy level, energy stored
        E_SOC_t_batch = mdl.continuous_var_dict(
            SOCtime_batch,
            lb=0,
            name="Energy level",  # ub=E_batt_MWh_t.max(),
        )

        cost_fluc_batt_batch = mdl.continuous_var(name="cost_fluc_batt", lb=-1e12)

        # --- content from ems_core_p2x.py ---
        price_H2_batch = self.price_H2_batch
        m_H2_demand_ts_batch = self.m_H2_demand_ts_batch
        H2_storage_t_batch = self.H2_storage_t_batch
        # Hydrogen storgae level
        LoS_H2_t_batch = mdl.continuous_var_dict(
            SOCtime_batch, lb=0, name="Hydrogen storage level"
        )

        # Power to gas plant power consumption, produced hydrogen, electrolyzer efficiency
        P_ptg_t_batch = mdl.continuous_var_dict(
            time_batch,
            lb=self.min_power_standby * self.ptg_MW,
            ub=self.ptg_MW,
            name="Power to gas plant consumption",
        )

        m_H2_t_batch = mdl.continuous_var_dict(
            time_batch, lb=0, name="Produced hydrogen"
        )
        m_H2_offtake_t_batch = mdl.continuous_var_dict(
            time_batch, lb=0, name="Hydrogen offtake"
        )

        penalty_p2x_batch = mdl.continuous_var(name="penalty_p2x", lb=-1e12)

        # ------------
        # Power balance
        # ------------
        P_total = {t: 0 for t in time_batch}
        for t in time_batch:
            P_total[t] = -P_curtailment_t_batch[t]

        # --- content from ems_core_wind.py ---
        for t in time_batch:
            P_total[t] += self.wind_ts_batch[t]

        # --- content from ems_core_solar.py ---
        for t in time_batch:
            P_total[t] += self.solar_ts_batch[t]

        # --- content from ems_core_battery.py ---
        for t in time_batch:
            P_total[t] += P_charge_discharge_batch[t]

        # --- content from ems_core_p2x.py ---
        for t in time_batch:
            P_total[t] -= P_ptg_t_batch[t]

        # ------------
        # Constraints
        # ------------
        for t in time_batch:
            mdl.add_constraint(
                P_HPP_t_batch[t] == P_total[t],
                ctname=f"P_HPP_t_balance{t}",
            )
        # --- defaults
        mdl.add_constraint(
            e_penalty_batch
            == (
                self.e_peak_period_expected_batch
                - mdl.sum(
                    P_HPP_t_batch[time_batch[i]] for i in self.peak_hours_index_batch
                )
            )
        )
        # Piecewise function for "only positive" function
        self.f1 = mdl.piecewise(0, [(0, 0)], 1)
        mdl.add_constraint(
            penalty_el_batch == self.price_peak_batch * self.f1(e_penalty_batch),
            ctname="penalty_el",
        )

        # --- content from ems_core_battery.py ---
        # Intitial and end SOC
        mdl.add_constraint(
            E_SOC_t_batch[SOCtime_batch[0]]
            == 0.5 * self.E_batt_MWh_t_batch[time_batch[0]]
        )

        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        mdl.add_constraint(
            E_SOC_t_batch[SOCtime_batch[-1]]
            == 0.5 * self.E_batt_MWh_t_batch[time_batch[0]]
        )

        # pircewise linear representation of charge vs dischrage effciency
        self.f2 = mdl.piecewise(
            self.charge_efficiency, [(0, 0)], 1 / self.charge_efficiency
        )

        self.fabs = mdl.piecewise(-1, [(0, 0)], 1)
        cost_fluc_batt = mdl.sum(
            self.fabs(
                P_charge_discharge_batch[t + pd.Timedelta("1hour")]
                - P_charge_discharge_batch[t]
            )
            * self.cost_of_battery_P_fluct_in_peak_price_ratio
            * self.price_ts_to_max_batch[t]
            for t in time_batch[:-1]
        )

        mdl.add_constraint(
            cost_fluc_batt_batch == cost_fluc_batt,
            ctname="cost_fluc_batt",
        )

        for t in time_batch:
            # Time index for successive time step
            tt = t + pd.Timedelta("1hour")
            # Delta_t of 1 hour
            dt = 1

            # charge/dischrage equation
            mdl.add_constraint(
                E_SOC_t_batch[tt]
                == E_SOC_t_batch[t] - self.f2(P_charge_discharge_batch[t]) * dt
            )

            # Constraining battery energy level to minimum battery level
            mdl.add_constraint(
                E_SOC_t_batch[t]
                >= (1 - self.battery_depth_of_discharge) * self.E_batt_MWh_t_batch[t]
            )

            # Constraining battery energy level to maximum battery level
            mdl.add_constraint(E_SOC_t_batch[t] <= self.E_batt_MWh_t_batch[t])

            # Battery charge/discharge within its power rating
            mdl.add_constraint(
                P_charge_discharge_batch[t] <= self.P_batt_MW * self.charge_efficiency
            )
            mdl.add_constraint(
                P_charge_discharge_batch[t] >= -self.P_batt_MW / self.charge_efficiency
            )

        # --- content from ems_core_p2x.py ---
        # Intitial and end LoS_H2
        mdl.add_constraint(LoS_H2_t_batch[SOCtime_batch[0]] == 0)
        mdl.add_constraint(LoS_H2_t_batch[SOCtime_batch[-1]] == 0)

        # piecewise linear representation of H2 storage vs offtake effciency
        f3 = mdl.piecewise(1 / self.storage_eff, [(0, 0)], self.storage_eff)

        # Caclulating electrolyzer efficiency as a function of load (piecewise linear approximation)
        if self.electrolyzer_eff_curve_type == "production":
            H2_curve_list = [
                (load * self.ptg_MW, H2 * self.ptg_MW) for load, H2 in self.eff_curve
            ]
        elif self.electrolyzer_eff_curve_type == "efficiency":
            H2_curve_list = [
                (load * self.ptg_MW, load * self.ptg_MW * efficiency / self.hhv * 1000)
                for load, efficiency in self.eff_curve
            ]
        else:
            raise ValueError(
                f'electrolyzer_eff_curve_type is: "{self.electrolyzer_eff_curve_type}". Available options are ["production", "efficiency"]'
            )

        H2_prod = mdl.piecewise(0, H2_curve_list, 0)

        penalty_p2x = mdl.sum(
            self.penalty_factor_H2 * (m_H2_demand_ts_batch[t] - m_H2_offtake_t_batch[t])
            for t in time_batch
        )

        mdl.add_constraint(
            penalty_p2x_batch == penalty_p2x,
            ctname="penalty_p2x",
        )

        for t in time_batch:
            # Time index for successive time step
            tt = t + pd.Timedelta("1hour")
            # Delta_t of 1 hour
            dt = 1

            # Hydrogen storgae equation
            mdl.add_constraint(
                LoS_H2_t_batch[tt]
                == LoS_H2_t_batch[t] + f3(m_H2_t_batch[t] - m_H2_offtake_t_batch[t])
            )

            # Constraining hydrogen offtake as per the demand time series and level of storage
            mdl.add_constraint(LoS_H2_t_batch[t] <= H2_storage_t_batch[t])
            mdl.add_constraint(m_H2_offtake_t_batch[t] <= m_H2_demand_ts_batch[t])

            # when the H2 offtake is infinite, there is no storage, then H2_offtake is same as H2_produced
            if H2_storage_t_batch[t] == 0:
                m_H2_offtake_t_batch[t] = m_H2_t_batch[t]

            # Calculating Hydrogen production with HPC
            mdl.add_constraint(m_H2_t_batch[t] == H2_prod(P_ptg_t_batch[t]))

        # ------------
        # Objective
        # ------------

        # --- default
        revenue_el = mdl.sum(
            self.price_ts_batch[t] * P_HPP_t_batch[t] for t in time_batch
        )

        objective_expr = revenue_el - penalty_el_batch

        # --- content from ems_core_battery.py ---

        objective_expr -= cost_fluc_batt_batch

        # --- content from ems_core_p2x.py ---
        revenue_p2x = mdl.sum(
            price_H2_batch * m_H2_offtake_t_batch[t] for t in time_batch
        )
        objective_expr += revenue_p2x - penalty_p2x_batch

        # pass objective to cplex model
        mdl.maximize(objective_expr)

        # ------------
        # solve
        # ------------
        sol = mdl.solve(log_output=False)

        # ------------
        # Outputs
        # ------------
        # --- defaults
        P_HPP_ts_batch_df = pd.DataFrame.from_dict(
            sol.get_value_dict(P_HPP_t_batch), orient="index"
        ).loc[:, 0]

        P_curtailment_ts_batch_df = pd.DataFrame.from_dict(
            sol.get_value_dict(P_curtailment_t_batch), orient="index"
        ).loc[:, 0]

        # make a time series like P_HPP with a constant penalty
        penalty_el_2 = sol.get_value(penalty_el_batch)
        penalty_el_ts_batch = np.ones(self.N_t_batch) * (penalty_el_2 / self.N_t_batch)

        # Cplex sometimes returns missing values :O
        P_HPP_ts_batch = P_HPP_ts_batch_df.reindex(time_batch, fill_value=0).values
        P_curtailment_ts_batch = P_curtailment_ts_batch_df.reindex(
            time_batch, fill_value=0
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

        # --- content from ems_core_battery.py ---
        P_charge_discharge_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(P_charge_discharge_batch), orient="index"
        ).loc[:, 0]

        E_SOC_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(E_SOC_t_batch), orient="index"
        ).loc[:, 0]

        cost_fluc_batt_batch = sol.get_value(cost_fluc_batt_batch)
        # Cplex sometimes returns missing values :O
        P_charge_discharge_ts_batch = P_charge_discharge_ts_df.reindex(
            time_batch, fill_value=0
        ).values
        E_SOC_ts_batch = E_SOC_ts_df.reindex(SOCtime_batch, fill_value=0).values

        self.P_charge_discharge_ts[self.batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[self.batch] = E_SOC_ts_batch[:-1]

        # --- content from ems_core_p2x.py ---
        P_ptg_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(P_ptg_t_batch), orient="index"
        ).loc[:, 0]
        m_H2_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(m_H2_t_batch), orient="index"
        ).loc[:, 0]
        LoS_H2_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(LoS_H2_t_batch), orient="index"
        ).loc[:, 0]
        m_H2_offtake_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(m_H2_offtake_t_batch), orient="index"
        ).loc[:, 0]

        penalty_p2x_batch = sol.get_value(penalty_p2x_batch)

        # Cplex sometimes returns missing values :O
        P_ptg_ts_batch = P_ptg_ts_df_batch.reindex(time_batch, fill_value=0).values
        m_H2_ts_batch = m_H2_ts_df_batch.reindex(time_batch, fill_value=0).values
        LoS_H2_ts_batch = LoS_H2_ts_df_batch.reindex(time_batch, fill_value=0).values
        m_H2_offtake_ts_batch = m_H2_offtake_ts_df_batch.reindex(
            time_batch, fill_value=0
        ).values

        for t in range(self.N_t_batch):
            if abs(P_ptg_ts_batch[t] - self.min_power_standby * self.ptg_MW) < 1e-6:
                self.P_ptg_SB_ts[t] = self.min_power_standby * self.ptg_MW
                P_ptg_ts_batch[t] = 0
                m_H2_ts_batch[t] = 0

        self.P_ptg_ts[self.batch] = P_ptg_ts_batch
        self.m_H2_ts[self.batch] = m_H2_ts_batch

        self.LoS_H2_ts[self.batch] = LoS_H2_ts_batch
        self.m_H2_offtake_ts[self.batch] = m_H2_offtake_ts_batch
        self.penalty_ts[self.batch] += penalty_p2x_batch

    def pyomo(self, **kwarg):
        """ """
        time_batch = self.time_batch
        SOCtime_batch = self.SOCtime_batch
        hpp_grid_connection = self.hpp_grid_connection
        # ------------
        # Variables
        # ------------
        mdl = pyo.ConcreteModel()
        # --- defaults

        mdl.IDX1 = range(len(time_batch))
        mdl.IDX2 = range(1)

        mdl.P_HPP_t = pyo.Var(
            mdl.IDX1, domain=pyo.NonNegativeReals, bounds=(0, hpp_grid_connection)
        )
        mdl.P_curtailment_t_batch = pyo.Var(mdl.IDX1, domain=pyo.NonNegativeReals)
        mdl.power_constraint = pyo.Constraint(mdl.IDX1)
        mdl.penalty = pyo.Var(mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))
        mdl.e_penalty = pyo.Var(mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12))

        # --- content from ems_core_battery.py ---
        time_batch = time_batch
        SOCtime_batch = SOCtime_batch
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

        mdl.P_charge_discharge_with_eff_t = pyo.Var(
            mdl.IDX1,
            domain=pyo.Reals,
            bounds=(-P_batt_MW / charge_efficiency, P_batt_MW * charge_efficiency),
        )

        # Battery delta for battery operation constraints
        mdl.P_battery_delta = pyo.Var(mdl.IDX5, domain=pyo.Reals, bounds=(-1e12, 1e12))

        mdl.P_battery_delta_abs_pw = pyo.Var(
            mdl.IDX5, domain=pyo.Reals, bounds=(0, 1e12)
        )

        # --- content from ems_core_p2x.py ---
        price_H2_batch = self.price_H2_batch
        m_H2_demand_ts_batch = self.m_H2_demand_ts_batch
        H2_storage_t_batch = self.H2_storage_t_batch
        # Hydrogen storage level (use IDX4 for SOCtime index)
        mdl.LoS_H2_t_batch = pyo.Var(mdl.IDX4, domain=pyo.NonNegativeReals)

        # Power to gas plant consumption (use IDX1 for peak hours or full time range)
        mdl.P_ptg_t_batch = pyo.Var(
            mdl.IDX1,
            bounds=(self.min_power_standby * self.ptg_MW, self.ptg_MW),
        )

        # Produced hydrogen (use IDX1)
        mdl.m_H2_t_batch = pyo.Var(mdl.IDX1, domain=pyo.NonNegativeReals)

        # Hydrogen offtake (use IDX1)
        mdl.m_H2_offtake_t_batch = pyo.Var(mdl.IDX1, domain=pyo.NonNegativeReals)

        mdl.penalty_p2x_batch = pyo.Var(
            mdl.IDX2, domain=pyo.Reals, bounds=(-1e12, 1e12)
        )

        # ------------
        # Power balance
        # ------------
        P_total = {t: 0 for t in mdl.IDX1}
        for t in mdl.IDX1:
            P_total[t] -= mdl.P_curtailment_t_batch[t]

        # --- content from ems_core_wind.py ---
        for t in mdl.IDX1:
            P_total[t] += self.wind_ts_batch.iloc[t]

        # --- content from ems_core_solar.py ---
        for t in mdl.IDX1:
            P_total[t] += self.solar_ts_batch.iloc[t]

        # --- content from ems_core_battery.py ---
        for t in mdl.IDX1:
            P_total[t] += mdl.P_charge_discharge_with_eff_t[t]

        # --- content from ems_core_p2x.py ---
        for t in mdl.IDX1:
            P_total[t] -= mdl.P_ptg_t_batch[t]

        # ------------
        # Constraints
        # ------------
        for t in mdl.IDX1:
            mdl.power_constraint[t] = mdl.P_HPP_t[t] == P_total[t]

        # --- defaults
        e_peak_period_expected_batch = self.e_peak_period_expected_batch
        price_peak_batch = self.price_peak_batch

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

        # --- content from ems_core_battery.py ---

        battery_depth_of_discharge = self.battery_depth_of_discharge
        E_batt_MWh_t_batch = self.E_batt_MWh_t_batch

        mdl.battery_energy_constraint = pyo.ConstraintList()
        mdl.battery_energy_min_constraint = pyo.ConstraintList()
        mdl.battery_dynamics_constraint = pyo.ConstraintList()

        mdl.SOC_initial_condition = pyo.Constraint(
            expr=mdl.E_SOC_t[mdl.IDX4[0]] == 0.5 * E_batt_MWh_t_batch.iloc[0]
        )

        # SOC at the end of the year has to be equal to SOC at the beginning of the year
        mdl.SOC_final = pyo.Constraint(
            expr=mdl.E_SOC_t[mdl.IDX4[-1]] == 0.5 * E_batt_MWh_t_batch.iloc[0]
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

        # x-values of the piece-wise function
        f_piecewise_x_vals = [-self.P_batt_MW, 0, self.P_batt_MW]
        # y-values of the piece-wise function
        f_piecewise_y_vals = [
            -self.P_batt_MW / self.charge_efficiency,
            0,
            self.P_batt_MW * self.charge_efficiency,
        ]
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

        # --- content from ems_core_p2x.py ---
        # Initial and end LoS_H2 (use integer indices 0 and last index)
        mdl.LoS_H2_t_batch_constraint_start = pyo.Constraint(
            expr=mdl.LoS_H2_t_batch[0] == 0
        )
        mdl.LoS_H2_t_batch_constraint_end = pyo.Constraint(
            expr=mdl.LoS_H2_t_batch[len(SOCtime_batch) - 1] == 0
        )

        # Build H2 production piecewise map P_ptg_t_batch -> m_H2_t_batch over IDX1 (if IDX1 non-empty)
        if self.electrolyzer_eff_curve_type == "production":
            H2_curve_list = [
                (load * self.ptg_MW, H2 * self.ptg_MW) for load, H2 in self.eff_curve
            ]
        elif self.electrolyzer_eff_curve_type == "efficiency":
            H2_curve_list = [
                (load * self.ptg_MW, load * self.ptg_MW * efficiency / self.hhv * 1000)
                for load, efficiency in self.eff_curve
            ]
        else:
            raise ValueError(
                f'electrolyzer_eff_curve_type is: "{self.electrolyzer_eff_curve_type}". '
                'Available options are ["production", "efficiency"]'
            )

            # Electrolyzer production curve (piecewise)
        H2_curve_x = [pt[0] for pt in H2_curve_list]
        H2_curve_y = [pt[1] for pt in H2_curve_list]
        mdl.H2_prod_constraint = pyo.Piecewise(
            mdl.IDX1,
            mdl.m_H2_t_batch,
            mdl.P_ptg_t_batch,
            pw_pts=H2_curve_x,
            f_rule=H2_curve_y,
            pw_constr_type="EQ",
            pw_repn="DLOG",
        )

        # Create ConstraintLists to collect per-time constraints
        mdl.h2_storage_eq = pyo.ConstraintList()
        mdl.h2_storage_capacity = pyo.ConstraintList()
        mdl.h2_offtake_demand = pyo.ConstraintList()
        mdl.h2_offtake_storage_zero = pyo.ConstraintList()

        # Precompute helper lookups mapping timestamps -> integer indices
        time_to_idx = {tt: i for i, tt in enumerate(time_batch)}
        soctime_to_idx = {tt: i for i, tt in enumerate(SOCtime_batch)}

        for t in time_batch:
            # integer index for current time
            k = time_to_idx[t]

            # successor timestamp t+1h
            tt_time = t + pd.Timedelta("1hour")

            # find SOC integer indices corresponding to t and tt_time (raise helpful KeyError if missing)
            try:
                i_soc = soctime_to_idx[t]
            except KeyError:
                raise KeyError(
                    f"Time {t!r} not found in SOCtime_batch; can't index LoS_H2_t_batch."
                )

            try:
                j_soc = soctime_to_idx[tt_time]
            except KeyError:
                raise KeyError(
                    f"Successor time {tt_time!r} (t+1h) not found in SOCtime_batch; "
                    "ensure SOCtime_batch contains successive timesteps."
                )

            # Determine inflow_minus_offtake for storage eqn: use model vars if present (k in IDX1)
            if k in mdl.IDX1:
                inflow_minus_offtake = mdl.m_H2_t_batch[k] - mdl.m_H2_offtake_t_batch[k]
            else:
                inflow_minus_offtake = 0

            # Hydrogen storage equation (index with integer SOC indices) -> add to ConstraintList
            mdl.h2_storage_eq.add(
                mdl.LoS_H2_t_batch[j_soc]
                == mdl.LoS_H2_t_batch[i_soc] + inflow_minus_offtake / self.storage_eff
            )

            # Storage capacity constraint (LoS <= capacity) -> add to ConstraintList
            mdl.h2_storage_capacity.add(
                mdl.LoS_H2_t_batch[i_soc] <= H2_storage_t_batch[t]
            )

            # m_H2_offtake <= demand (only when var exists) -> add to ConstraintList
            if k in mdl.IDX1:
                mdl.h2_offtake_demand.add(
                    mdl.m_H2_offtake_t_batch[k] <= m_H2_demand_ts_batch.iloc[k]
                )

            # When H2 storage is zero: enforce equality if the variables exist for this k -> add to ConstraintList
            if H2_storage_t_batch[t] == 0 and k in mdl.IDX1:
                mdl.h2_offtake_storage_zero.add(
                    mdl.m_H2_offtake_t_batch[k] == mdl.m_H2_t_batch[k]
                )

        penalty_p2x = sum(
            self.penalty_factor_H2
            * (m_H2_demand_ts_batch.iloc[t] - mdl.m_H2_offtake_t_batch[t])
            for t in mdl.IDX1
        )

        mdl.penalty_constraint.add(mdl.penalty_p2x_batch[0] == penalty_p2x)

        # ------------
        # Objective
        # ------------
        # --- defaults
        price_ts_batch = self.price_ts_batch

        electricity_income = sum(
            price_ts_batch.iloc[t] * mdl.P_HPP_t[t] for t in mdl.IDX1
        )
        penalty = -mdl.penalty[0]
        objective_expr = electricity_income - penalty

        # --- content from ems_core_battery.py ---
        cost_of_battery_P_fluct_in_peak_price_ratio = (
            self.cost_of_battery_P_fluct_in_peak_price_ratio
        )

        price_ts_to_max = self.price_ts_to_max_batch

        # battery
        battery_fluc_cost = sum(
            mdl.P_battery_delta_abs_pw[t]
            * cost_of_battery_P_fluct_in_peak_price_ratio
            * price_ts_to_max.iloc[t]
            for t in mdl.IDX5
        )

        objective_expr -= battery_fluc_cost

        # --- content from ems_core_p2x.py ---
        revenue_p2x = sum(
            price_H2_batch * mdl.m_H2_offtake_t_batch[t] for t in mdl.IDX1
        )
        penalty_p2x = mdl.penalty_p2x_batch[0]
        objective_expr += revenue_p2x - penalty_p2x

        mdl.OBJ = pyo.Objective(
            # revenues and OPEX
            expr=objective_expr,
            sense=pyo.maximize,
        )

        # ------------
        # solve
        # ------------
        solver = pyo.SolverFactory("cplex")  # glpk
        result = solver.solve(mdl, tee=False)  # tee=True if you want solver log

        # ------------
        # Outputs
        # ------------
        N_t = self.N_t_batch
        batch = self.batch
        batch = self.batch

        # --- defaults
        P_HPP_ts_batch_df = series_from_indexed_var(mdl.P_HPP_t, time_batch)
        P_curtailment_ts_batch_df = series_from_indexed_var(
            mdl.P_curtailment_t_batch, time_batch
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

        # --- content from ems_core_battery.py ---

        # Extract time series as pandas Series (index = your time_batch items)

        P_charge_discharge_ts_batch_df = series_from_indexed_var(
            mdl.P_charge_discharge_t, time_batch
        )
        E_SOC_ts_batch_df = series_from_indexed_var(mdl.E_SOC_t, SOCtime_batch)

        P_charge_discharge_ts_batch = P_charge_discharge_ts_batch_df.reindex(
            time_batch, fill_value=0
        ).values
        E_SOC_ts_batch = E_SOC_ts_batch_df.reindex(SOCtime_batch, fill_value=0).values

        self.P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[batch] = E_SOC_ts_batch[:-1]

        # --- content from ems_core_p2x.py ---
        # Extract Pyomo solution values as dictionaries
        P_ptg_ts_dict = series_from_indexed_var(mdl.P_ptg_t_batch, time_batch)
        m_H2_ts_dict = series_from_indexed_var(mdl.m_H2_t_batch, time_batch)
        LoS_H2_ts_dict = series_from_indexed_var(mdl.LoS_H2_t_batch, time_batch)
        m_H2_offtake_ts_dict = series_from_indexed_var(
            mdl.m_H2_offtake_t_batch, time_batch
        )

        # Convert to pandas Series / DataFrame for convenience
        P_ptg_ts_df_batch = pd.Series(P_ptg_ts_dict).reindex(time_batch, fill_value=0)
        m_H2_ts_df_batch = pd.Series(m_H2_ts_dict).reindex(time_batch, fill_value=0)
        LoS_H2_ts_df_batch = pd.Series(LoS_H2_ts_dict).reindex(time_batch, fill_value=0)
        m_H2_offtake_ts_df_batch = pd.Series(m_H2_offtake_ts_dict).reindex(
            time_batch, fill_value=0
        )

        # Convert to numpy arrays
        P_ptg_ts_batch = P_ptg_ts_df_batch.values
        m_H2_ts_batch = m_H2_ts_df_batch.values
        LoS_H2_ts_batch = LoS_H2_ts_df_batch.values
        m_H2_offtake_ts_batch = m_H2_offtake_ts_df_batch.values

        # Adjust for minimum power standby
        for t in range(self.N_t_batch):
            if abs(P_ptg_ts_batch[t] - self.min_power_standby * self.ptg_MW) < 1e-6:
                self.P_ptg_SB_ts[t] = self.min_power_standby * self.ptg_MW
                P_ptg_ts_batch[t] = 0
                m_H2_ts_batch[t] = 0

        # Save results to class attributes
        self.P_ptg_ts[self.batch] = P_ptg_ts_batch
        self.m_H2_ts[self.batch] = m_H2_ts_batch
        self.LoS_H2_ts[self.batch] = LoS_H2_ts_batch
        self.m_H2_offtake_ts[self.batch] = m_H2_offtake_ts_batch

    def ortools(self, **kwarg):
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
        SOCtime_batch = self.SOCtime_batch
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

        # --- content from ems_core_battery.py ---

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

        # --- content from ems_core_p2x.py ---
        price_H2_batch = self.price_H2_batch
        m_H2_demand_ts_batch = self.m_H2_demand_ts_batch
        H2_storage_t_batch = self.H2_storage_t_batch
        # State-of-storage variables indexed by SOCtime_batch (as in CPLEX)
        LoS_H2_t_batch = {
            t: solver.NumVar(0.0, solver.infinity(), f"H2_storage_{t}")
            for t in SOCtime_batch
        }

        # P_ptg, H2 production, H2 offtake for each time in time_batch
        P_ptg_t_batch = {
            t: solver.NumVar(
                self.min_power_standby * self.ptg_MW, self.ptg_MW, f"P_ptg_{t}"
            )
            for t in time_batch
        }
        m_H2_t_batch = {
            t: solver.NumVar(0.0, solver.infinity(), f"H2_prod_{t}") for t in time_batch
        }
        m_H2_offtake_t_batch = {
            t: solver.NumVar(0.0, solver.infinity(), f"H2_offtake_{t}")
            for t in time_batch
        }

        penalty_p2x_batch = solver.NumVar(0.0, solver.infinity(), "penalty_p2x_batch")

        # ------------
        # Power balance
        # ------------
        P_total = {t: 0 for i, t in enumerate(time_batch)}
        for i, t in enumerate(time_batch):
            P_total[t] -= P_curtail_batch[t]

        # --- content from ems_core_wind.py ---
        for i, t in enumerate(time_batch):
            P_total[t] += self.wind_ts_batch[t]

        # --- content from ems_core_solar.py ---
        for i, t in enumerate(time_batch):
            P_total[t] += self.solar_ts_batch[t]

        # --- content from ems_core_battery.py ---
        for i, t in enumerate(time_batch):
            P_total[t] += P_dis_batch[t] - P_ch_batch[t]

        # --- content from ems_core_p2x.py ---
        for i, t in enumerate(time_batch):
            P_total[t] -= P_ptg_t_batch[t]

        # ------------
        # Constraints
        # ------------
        # P_HPP_t_balance constraints
        for i, t in enumerate(time_batch):
            # power balance: P_HPP = wind + solar - curtail + (P_dis - P_ch)
            solver.Add(P_HPP_batch[t] == P_total[t], name=f"P_HPP_t_balance{t}")
        N_t_batch = self.N_t_batch
        e_peak_period_expected_batch = self.e_peak_period_expected_batch
        peak_hours_index_batch = self.peak_hours_index_batch

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

        solver.Add(e_penalty_batch == float(e_peak_period_expected_batch) - expr)

        # Linearize f_pos = max(0, e_penalty)
        f_pos = solver.NumVar(0.0, solver.infinity(), "f_pos")
        solver.Add(f_pos >= e_penalty_batch)

        # Penalty = price_peak * f_pos
        solver.Add(penalty_el_batch == float(self.price_peak_batch) * f_pos)

        # p_nonneg >= e_penalty  (and p_nonneg >= 0 by var bound)
        solver.Add(p_nonneg_batch >= e_penalty_batch)

        # --- content from ems_core_battery.py ---
        N_t_batch = self.N_t_batch

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

            # self.update_constraint_rhs_ortools(
            #     solver,
            #     constraint_name=f"P_HPP_t_balance{t}",
            #     addition_expr=P_dis_batch[t],
            # )  # - P_ch_batch[t])
            # self.update_constraint_rhs_ortools(
            #     solver,
            #     constraint_name=f"P_HPP_t_balance{t}",
            #     addition_expr=P_ch_batch[t],
            #     coeff=-1,
            # )

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

        # --- content from ems_core_p2x.py ---
        # initial and terminal LoS constraints (matching cplex)
        solver.Add(LoS_H2_t_batch[SOCtime_batch[0]] == 0.0)
        solver.Add(LoS_H2_t_batch[SOCtime_batch[-1]] == 0.0)

        # Build electrolyser piecewise points depending on chosen type (match CPLEX)
        # Storage efficiency linear function
        storage_eff = self.storage_eff

        def f3(x):
            return x / storage_eff

        # Prepare electrolyzer piecewise curve
        if self.electrolyzer_eff_curve_type == "production":
            H2_curve_list = [
                (load * self.ptg_MW, H2 * self.ptg_MW) for load, H2 in self.eff_curve
            ]
        elif self.electrolyzer_eff_curve_type == "efficiency":
            H2_curve_list = [
                (load * self.ptg_MW, load * self.ptg_MW * efficiency / self.hhv * 1000)
                for load, efficiency in self.eff_curve
            ]
        else:
            raise ValueError(
                f'electrolyzer_eff_curve_type is: "{self.electrolyzer_eff_curve_type}". Available options are ["production", "efficiency"]'
            )

        for t in time_batch:
            tt = t + pd.Timedelta("1hour")

            # Hydrogen storage balance
            if tt in time_batch:
                solver.Add(
                    LoS_H2_t_batch[tt]
                    == LoS_H2_t_batch[t] + f3(m_H2_t_batch[t] - m_H2_offtake_t_batch[t])
                )

            # Storage and offtake limits
            solver.Add(LoS_H2_t_batch[t] <= H2_storage_t_batch[t])
            solver.Add(m_H2_offtake_t_batch[t] <= m_H2_demand_ts_batch[t])

            # No storage case
            if H2_storage_t_batch[t] == 0:
                solver.Add(m_H2_offtake_t_batch[t] == m_H2_t_batch[t])

            # estimate H2_P relation as a single line
            slope = (H2_curve_list[-2][1] - H2_curve_list[1][1]) / (
                H2_curve_list[-2][0] - H2_curve_list[1][0]
            )

            solver.Add(
                m_H2_t_batch[t] == slope * P_ptg_t_batch[t] + H2_curve_list[1][0]
            )

        # Penalty construction: unmet H2 demand sum (identical formula)
        unmet_sum = solver.Sum(
            [
                self.penalty_factor_H2
                * (float(m_H2_demand_ts_batch[t]) - m_H2_offtake_t_batch[t])
                for t in time_batch
            ]
        )

        solver.Add(penalty_p2x_batch == unmet_sum)

        # ------------
        # Objective
        # ------------
        # --- defaults
        objective = solver.Objective()
        for t in time_batch:
            objective.SetCoefficient(P_HPP_batch[t], float(self.price_ts_batch.loc[t]))

        if penalty_el_batch is not None:
            objective.SetCoefficient(penalty_el_batch, -1.0)

        # --- content from ems_core_battery.py ---
        # battery fluctuation cost
        for i in range(self.N_t_batch - 1):
            t = time_batch[i]
            coef = -float(
                self.cost_of_battery_P_fluct_in_peak_price_ratio
                * self.price_ts_to_max_batch.loc[t]
            )
            objective.SetCoefficient(self.delta[t], coef)

        # --- content from ems_core_p2x.py ---
        for t in time_batch[:-1]:
            # price_H2_batch could be a pandas Series or a scalar; handle both
            try:
                price_t = float(price_H2_batch[t])
            except Exception:
                # fallback: scalar or not indexable
                price_t = float(price_H2_batch)
            objective.SetCoefficient(m_H2_offtake_t_batch[t], price_t)

        # subtract penalty (same sign as CPLEX; CPLEX added penalty to objective with negative sign)
        objective.SetCoefficient(penalty_p2x_batch, -1.0)

        objective.SetMaximization()

        # ------------
        # solve
        # ------------
        result_status = solver.Solve()
        if result_status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(
                f"Solver failed or returned infeasible (status {result_status})."
            )

        # ------------
        # Outputs
        # ------------
        batch = self.batch
        N_t_batch = self.N_t_batch
        price_peak_batch = self.price_peak_batch
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

        # --- content from ems_core_battery.py ---

        P_charge_discharge_ts_batch = np.array(
            [
                (P_dis_batch[t].solution_value() - P_ch_batch[t].solution_value())
                for t in time_batch
            ]
        )
        E_SOC_ts_batch = np.array(
            [E_SOC_batch[t].solution_value() for t in SOCtime_batch]
        )

        self.P_charge_discharge_ts[self.batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[self.batch] = E_SOC_ts_batch[:-1]

        # --- content from ems_core_p2x.py ---
        # collect solution values in the same structure style as cplex_outputs
        P_ptg_ts_batch = [P_ptg_t_batch[t].solution_value() for t in time_batch]
        m_H2_ts_batch = [m_H2_t_batch[t].solution_value() for t in time_batch]
        LoS_H2_ts_batch = [
            LoS_H2_t_batch[t].solution_value() if t in LoS_H2_t_batch else 0.0
            for t in time_batch
        ]
        m_H2_offtake_ts_batch = [
            m_H2_offtake_t_batch[t].solution_value() for t in time_batch
        ]

        # Adjust for minimum power standby exactly as in cplex_outputs
        for t in range(self.N_t_batch):
            if abs(P_ptg_ts_batch[t] - self.min_power_standby * self.ptg_MW) < 1e-6:
                self.P_ptg_SB_ts[t] = self.min_power_standby * self.ptg_MW
                P_ptg_ts_batch[t] = 0.0
                m_H2_ts_batch[t] = 0.0

        # Save results to the same attributes as cplex_outputs expects
        self.P_ptg_ts[self.batch] = P_ptg_ts_batch
        self.m_H2_ts[self.batch] = m_H2_ts_batch
        self.LoS_H2_ts[self.batch] = LoS_H2_ts_batch
        self.m_H2_offtake_ts[self.batch] = m_H2_offtake_ts_batch

    def update_final_outputs(self, out):
        """ """
        # --- defaults
        out_defaults = {
            "P_HPP_ts": self.P_HPP_ts,
            "P_curtailment_ts": self.P_curtailment_ts,
            "penalty_ts": self.penalty_ts,
        }
        out.update(out_defaults)

        # --- content from ems_core_battery.py ---
        # set last SOC element equal to initial SOC.
        self.E_SOC_ts[-1] = self.E_SOC_ts[0]
        out_battery = {
            "P_charge_discharge_ts": self.P_charge_discharge_ts,
            "E_SOC_ts": self.E_SOC_ts,
        }
        out.update(out_battery)

        # --- content from ems_core_p2x.py ---
        out_p2x = {
            "P_ptg_ts": self.P_ptg_ts,
            "P_ptg_SB_ts": self.P_ptg_SB_ts,
            "m_H2_ts": self.m_H2_ts,
            "m_H2_offtake_ts": self.m_H2_offtake_ts,
            "LoS_H2_ts": self.LoS_H2_ts,
        }
        out.update(out_p2x)
