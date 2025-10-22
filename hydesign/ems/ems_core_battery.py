# basic libraries
from typing import Any, Dict

import numpy as np
import pandas as pd

from .ems_core_helpers import EMSCoreInterface


class EMSCoreBattery(EMSCoreInterface):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        super().__init__(**kwargs)

        # allocate arrays
        self.P_batt_MW = kwargs["P_batt_MW"]
        # self.hpp_grid_connection = kwargs["hpp_grid_connection"]
        self.battery_depth_of_discharge = kwargs["battery_depth_of_discharge"]
        self.charge_efficiency = kwargs["charge_efficiency"]
        self.cost_of_battery_P_fluct_in_peak_price_ratio = kwargs[
            "cost_of_battery_P_fluct_in_peak_price_ratio"
        ]
        self.E_batt_MWh_t = kwargs["E_batt_MWh_t"]
        self.P_charge_discharge_ts = np.zeros(self.n_ts)
        self.E_SOC_ts = np.zeros(self.n_ts + 1)

    def final_process(self, **kwargs) -> Dict[str, Any]:
        """
        Final processing step.
        """
        # set last SOC element equal to initial SOC.
        kwargs["E_SOC_ts"][-1] = kwargs["E_SOC_ts"][0]
        return kwargs

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        """ """
        self.price_ts_batch = self.price_ts.iloc[batch]
        self.E_batt_MWh_t_batch = self.E_batt_MWh_t.iloc[batch]
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

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """

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

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
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

    def cplex_outputs(self, sol):
        P_charge_discharge_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_charge_discharge_batch), orient="index"
        ).loc[:, 0]

        E_SOC_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.E_SOC_t_batch), orient="index"
        ).loc[:, 0]

        # make a time series like P_HPP with a constant penalty

        # Cplex sometimes returns missing values :O

        P_charge_discharge_ts_batch = P_charge_discharge_ts_df.reindex(
            self.time_batch, fill_value=0
        ).values
        E_SOC_ts_batch = E_SOC_ts_df.reindex(self.SOCtime_batch, fill_value=0).values

        batch = self.batch

        self.P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[batch] = E_SOC_ts_batch[:-1]

    def ortools_var_constraints(self, solver):
        """
        EMS solver implemented with OR-Tools linear solver.
        Returns:
            P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts
        """
        time_batch = self.time_batch
        N_t_batch = self.N_t_batch
        SOCtime_batch = self.SOCtime_batch

        # Variables

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

        # delta variables for absolute battery power change cost
        delta = {}
        for i in range(N_t_batch - 1):
            t = time_batch[i]
            delta[t] = solver.NumVar(0.0, solver.infinity(), f"delta_{i}")

        # Constraints
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

        solver.Add(E_SOC_batch[SOCtime_batch[-2]] == 40)

        self.delta = delta
        self.P_dis_batch = P_dis_batch
        self.P_ch_batch = P_ch_batch
        self.E_SOC_batch = E_SOC_batch

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

    def ortools_outputs(self):
        batch = self.batch
        time_batch = self.time_batch
        SOCtime_batch = self.SOCtime_batch
        P_dis_batch = self.P_dis_batch
        P_ch_batch = self.P_ch_batch
        E_SOC_batch = self.E_SOC_batch

        P_charge_discharge_ts_batch = np.array(
            [
                (P_dis_batch[t].solution_value() - P_ch_batch[t].solution_value())
                for t in time_batch
            ]
        )
        E_SOC_ts_batch = np.array(
            [E_SOC_batch[t].solution_value() for t in SOCtime_batch]
        )

        self.P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        self.E_SOC_ts[batch] = E_SOC_ts_batch[:-1]

    def get_final_outputs(self):
        return {
            "P_charge_discharge_ts": self.P_charge_discharge_ts,
            "E_SOC_ts": self.E_SOC_ts,
        }
