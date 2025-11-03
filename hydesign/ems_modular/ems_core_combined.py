# Auto-generated combined module
# Generated from: ems_core_defaults.py, ems_core_wind.py, ems_core_solar.py, ems_core_battery.py, ems_core_p2x.py

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from hydesign.ems_modular.modules.ems_core_helpers import (
    EMSCoreInterface,
    series_from_indexed_var,
    split_in_batch,
)


class EMSCoreCombined(EMSCoreInterface):
    """Auto-generated combined class (merged bodies of multiple classes)

    NOTE: bodies are concatenated (primary class then others). Please review for semantic issues.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        """ """
        # --- defaults
        self.hpp_grid_connection = kwargs.get("hpp_grid_connection", 0)
        self.price_ts = kwargs["price_ts"]
        self.peak_hr_quantile = kwargs.get("peak_hr_quantile", 0.9)
        self.n_full_power_hours_expected_per_day_at_peak_price = kwargs.get(
            "n_full_power_hours_expected_per_day_at_peak_price", 3
        )
        self.batch_size = kwargs.get("batch_size", 10)
        n_ts = len(self.price_ts)
        self.n_ts = n_ts

        # split in batches
        batches_all = split_in_batch(list(range(n_ts)), self.batch_size)
        self.batches = batches_all

        self.P_HPP_ts = np.zeros(n_ts)
        self.P_curtailment_ts = np.zeros(n_ts)
        self.penalty_ts = np.zeros(n_ts)

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
        self.m_H2_ts = np.zeros(self.n_ts)
        self.m_H2_offtake_ts = np.zeros(self.n_ts)
        self.P_ptg_SB_ts = np.zeros_like(self.P_ptg_ts)

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        """ """
        # --- defaults
        self.price_ts_batch = self.price_ts.iloc[batch]
        self.batch = batch
        self.time_batch = self.price_ts_batch.index

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

        # time set with an additional time slot for the last soc
        self.SOCtime_batch = self.time_batch.append(
            pd.Index([self.time_batch[-1] + pd.Timedelta("1hour")])
        )

        # --- content from ems_core_p2x.py ---
        self.price_H2_batch = self.price_H2  # .iloc[batch]
        self.m_H2_demand_ts_batch = self.m_H2_demand_ts.iloc[batch]
        self.H2_storage_t_batch = self.H2_storage_t.iloc[batch]

        # time set with an additional time slot for the last soc
        self.SOCtime_batch = self.time_batch.append(
            pd.Index([self.time_batch[-1] + pd.Timedelta("1hour")])
        )

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        # --- defaults
        self.P_HPP_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, ub=self.hpp_grid_connection, name="HPP power output"
        )
        self.P_curtailment_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, name="Curtailment"
        )

        self.penalty_batch = mdl.continuous_var(name="penalty", lb=-1e12)
        self.e_penalty_batch = mdl.continuous_var(name="e_penalty", lb=-1e12)

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
        self.LoS_H2_t_batch = mdl.continuous_var_dict(
            self.SOCtime_batch, lb=0, name="Hydrogen storage level"
        )

        # Power to gas plant power consumption, produced hydrogen, electrolyzer efficiency
        self.P_ptg_t_batch = mdl.continuous_var_dict(
            self.time_batch,
            lb=self.min_power_standby * self.ptg_MW,
            ub=self.ptg_MW,
            name="Power to gas plant consumption",
        )

        self.m_H2_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, name="Produced hydrogen"
        )
        self.m_H2_offtake_t_batch = mdl.continuous_var_dict(
            self.time_batch, lb=0, name="Hydrogen offtake"
        )

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
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
            self.penalty_batch == self.price_peak_batch * self.f1(self.e_penalty_batch)
        )

        for t in self.time_batch:
            mdl.add_constraint(
                self.P_HPP_t_batch[t] == -self.P_curtailment_t_batch[t],
                ctname=f"P_HPP_t_balance{t}",
            )

        # --- content from ems_core_wind.py ---
        for t in self.time_batch:
            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.wind_ts_batch[t],
            )

        # --- content from ems_core_solar.py ---
        for t in self.time_batch:
            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.solar_ts_batch[t],
            )

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
        mdl.add_constraint(self.LoS_H2_t_batch[self.SOCtime_batch[0]] == 0)
        mdl.add_constraint(self.LoS_H2_t_batch[self.SOCtime_batch[-1]] == 0)

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

        for t in self.time_batch:
            # Time index for successive time step
            tt = t + pd.Timedelta("1hour")
            # Delta_t of 1 hour
            dt = 1

            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=-self.P_ptg_t_batch[t],
            )

            # Hydrogen storgae equation
            mdl.add_constraint(
                self.LoS_H2_t_batch[tt]
                == self.LoS_H2_t_batch[t]
                + f3(self.m_H2_t_batch[t] - self.m_H2_offtake_t_batch[t])
            )

            # Constraining hydrogen offtake as per the demand time series and level of storage
            mdl.add_constraint(self.LoS_H2_t_batch[t] <= self.H2_storage_t_batch[t])
            mdl.add_constraint(
                self.m_H2_offtake_t_batch[t] <= self.m_H2_demand_ts_batch[t]
            )

            # when the H2 offtake is infinite, there is no storage, then H2_offtake is same as H2_produced
            if self.H2_storage_t_batch[t] == 0:
                self.m_H2_offtake_t_batch[t] = self.m_H2_t_batch[t]

            # Calculating Hydrogen production with HPC
            mdl.add_constraint(self.m_H2_t_batch[t] == H2_prod(self.P_ptg_t_batch[t]))

    def cplex_opjective(self, mdl):
        """ """
        # --- default
        self.fabs = mdl.piecewise(-1, [(0, 0)], 1)
        mdl.maximize(0)

        electricity_income = mdl.sum(
            self.price_ts_batch[t] * self.P_HPP_t_batch[t] for t in self.time_batch
        )
        self.add_to_objective(mdl, addition_expr=electricity_income)

        penalty_grid = self.penalty_batch
        self.add_to_objective(mdl, addition_expr=penalty_grid)

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
        H2_revenue = mdl.sum(
            self.price_H2_batch * self.m_H2_offtake_t_batch[t]
            for t in self.time_batch[:-1]
        )

        self.add_to_objective(mdl, addition_expr=H2_revenue)

    def cplex_outputs(self, sol):
        """ """
        # --- defaults
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
        #     P_HPP_ts_batch = (
        #         wind_ts_batch.values
        #         + solar_ts_batch.values
        #         + -P_curtailment_ts_batch
        #         + P_charge_discharge_ts_batch
        #     )

        self.P_HPP_ts[self.batch] = P_HPP_ts_batch
        self.P_curtailment_ts[self.batch] = P_curtailment_ts_batch
        self.penalty_ts[self.batch] = penalty_ts_batch

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
        P_ptg_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_ptg_t_batch), orient="index"
        ).loc[:, 0]
        m_H2_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(self.m_H2_t_batch), orient="index"
        ).loc[:, 0]
        LoS_H2_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(self.LoS_H2_t_batch), orient="index"
        ).loc[:, 0]
        m_H2_offtake_ts_df_batch = pd.DataFrame.from_dict(
            sol.get_value_dict(self.m_H2_offtake_t_batch), orient="index"
        ).loc[:, 0]

        for t in range(self.N_t_batch):
            if abs(self.P_ptg_ts[t] - self.min_power_standby * self.ptg_MW) < 1e-6:
                self.P_ptg_SB_ts[t] = self.min_power_standby * self.ptg_MW
                self.P_ptg_ts[t] = 0
                self.m_H2_ts[t] = 0

        self.P_ptg_ts[self.batch] = P_ptg_ts_df_batch
        self.m_H2_ts[self.batch] = m_H2_ts_df_batch
        # inside EMSCoreCombined.cplex_outputs, just before the failing line
        print("DEBUG batch index:", self.batch)
        print(
            "type LoS_H2_ts:",
            type(self.LoS_H2_ts[self.batch]),
            "shape:",
            getattr(self.LoS_H2_ts[self.batch], "shape", None),
        )
        print("type LoS_H2_ts_df_batch:", type(LoS_H2_ts_df_batch))
        try:
            print("LoS_H2_ts_df_batch.shape:", LoS_H2_ts_df_batch.shape)
        except Exception:
            # maybe a DataFrame
            print(
                "LoS_H2_ts_df_batch.shape (DF):",
                getattr(LoS_H2_ts_df_batch, "shape", None),
            )

        # If it's a pandas DataFrame, also inspect:
        if hasattr(LoS_H2_ts_df_batch, "columns"):
            print("cols:", LoS_H2_ts_df_batch.columns)
        if hasattr(LoS_H2_ts_df_batch, "index"):
            print("index len:", len(LoS_H2_ts_df_batch.index))

        self.LoS_H2_ts[self.batch] = LoS_H2_ts_df_batch
        self.m_H2_offtake_ts[self.batch] = m_H2_offtake_ts_df_batch

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

    def ortools_variables(self, solver):
        """ """
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

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def ortools_constraints(self, solver):
        """ """
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

        # --- defaults
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

        # --- content from ems_core_wind.py ---
        wind_ts_batch = {
            t: solver.NumVar(
                float(self.wind_ts_batch.loc[t]),
                float(self.wind_ts_batch.loc[t]),
                f"wind_ts_batch_{i}",
            )
            for i, t in enumerate(self.time_batch)
        }
        for t in self.time_batch:
            self.update_constraint_rhs_ortools(
                solver,
                constraint_name=f"P_HPP_t_balance{t}",
                addition_expr=wind_ts_batch[t],
            )

        # --- content from ems_core_solar.py ---
        solar_ts_batch = {
            t: solver.NumVar(
                float(self.solar_ts_batch.loc[t]),
                float(self.solar_ts_batch.loc[t]),
                f"solar_ts_batch_{i}",
            )
            for i, t in enumerate(self.time_batch)
        }
        for t in self.time_batch:
            self.update_constraint_rhs_ortools(
                solver,
                constraint_name=f"P_HPP_t_balance{t}",
                addition_expr=solar_ts_batch[t],
            )

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def ortools_objective(self, solver):
        """ """
        # --- defaults
        objective = solver.Objective()
        for t in self.time_batch:
            objective.SetCoefficient(
                self.P_HPP_batch[t], float(self.price_ts_batch.loc[t])
            )

        # penalty term (note penalty = price_peak * p_nonneg; objective subtracts it)
        objective.SetCoefficient(self.p_nonneg_batch, -float(self.price_peak_batch))

        objective.SetMaximization()

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def ortools_outputs(self):
        """ """
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

        penalty_val_batch = float(price_peak_batch * p_nonneg_batch.solution_value())
        penalty_ts_batch = np.ones(N_t_batch) * (penalty_val_batch / float(N_t_batch))

        self.P_HPP_ts[batch] = P_HPP_ts_batch
        self.P_curtailment_ts[batch] = P_curtail_ts_batch
        self.penalty_ts[batch] = penalty_ts_batch

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def pyomo_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
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

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def pyomo_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        # --- defaults
        time_batch = self.time_batch
        self.P_batt_MW = kwargs["P_batt_MW"]

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

        # --- content from ems_core_wind.py ---
        for t in range(0, len(self.time_batch)):
            self.update_constraint_rhs_pyomo(
                mdl, "power_constraint", added_value=self.wind_ts_batch.iloc[t], idx=t
            )

        # --- content from ems_core_solar.py ---
        for t in range(0, len(self.time_batch)):
            # solar
            self.update_constraint_rhs_pyomo(
                mdl, "power_constraint", added_value=self.solar_ts_batch.iloc[t], idx=t
            )

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def pyomo_opjective(self, mdl, **kwargs):
        """ """
        # --- defaults
        self.P_batt_MW = kwargs["P_batt_MW"]

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

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def pyomo_outputs(self, mdl, **kwargs):
        """ """
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

        # --- content from ems_core_battery.py ---
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

        # --- content from ems_core_p2x.py ---
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

    def final_process(self, **kwargs) -> Dict[str, Any]:
        # set last SOC element equal to initial SOC.
        kwargs["E_SOC_ts"][-1] = kwargs["E_SOC_ts"][0]

        # --- content from ems_core_p2x.py ---
        self.LoS_H2_t_batch[-1] = self.LoS_H2_t_batch[0]
