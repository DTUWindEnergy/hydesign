# basic libraries
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from hydesign.ems_modular.modules.ems_core_helpers import (
    EMSCoreInterface,
    series_from_indexed_var,
)


class EMSCoreP2x(EMSCoreInterface):
    """
    EMSCoreBattery
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
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
        print(self.batch_size)

    def final_process(self, **kwargs) -> Dict[str, Any]:
        kwargs["LoS_H2_ts"][-1] = kwargs["LoS_H2_ts"][0]

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        self.price_H2_batch = self.price_H2  # .iloc[batch]
        self.m_H2_demand_ts_batch = self.m_H2_demand_ts.iloc[batch]
        self.H2_storage_t_batch = self.H2_storage_t.iloc[batch]

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:
        # Hydrogen storgae level
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

        # # ------ penalty p2x
        # f1 = mdl.piecewise(0, [(0, 0)], 1)

        # mdl.add_constraint(
        #     self.penalty
        #     == self.price_peak_batch * f1(e_penalty)
        #     + mdl.sum(
        #         penalty_factor_H2 * (m_H2_demand_ts[t] - self.m_H2_offtake_t_batch[t]) for t in time
        #     )
        # )
        # # ------

        # Intitial and end LoS_H2
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

            penalty_p2x = mdl.sum(
                self.penalty_factor_H2
                * (self.m_H2_demand_ts_batch[t] - self.m_H2_offtake_t_batch[t])
                for t in self.time_batch
            )

            self.update_constraint_rhs(
                mdl,
                constraint_name=f"penalty_{t}",
                added_value=penalty_p2x,
            )

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

        H2_revenue = mdl.sum(
            self.price_H2_batch * self.m_H2_offtake_t_batch[t]
            for t in self.time_batch[:-1]
        )

        self.add_to_objective(mdl, addition_expr=H2_revenue)

    def cplex_outputs(self, sol):

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

        # Cplex sometimes returns missing values :O
        P_ptg_ts_batch = P_ptg_ts_df_batch.reindex(self.time_batch, fill_value=0).values
        m_H2_ts_batch = m_H2_ts_df_batch.reindex(self.time_batch, fill_value=0).values
        LoS_H2_ts_batch = LoS_H2_ts_df_batch.reindex(
            self.time_batch, fill_value=0
        ).values
        m_H2_offtake_ts_batch = m_H2_offtake_ts_df_batch.reindex(
            self.time_batch, fill_value=0
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

    def pyomo_variables(self, mdl, **kwargs) -> Dict[str, Any]:
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

    def pyomo_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        # --- P2X constraints (fixed: use ConstraintList instead of mdl.add_constraint) ---

        # Initial and end LoS_H2 (use integer indices 0 and last index)
        mdl.LoS_H2_t_batch_constraint_start = pyo.Constraint(
            expr=mdl.LoS_H2_t_batch[0] == 0
        )
        mdl.LoS_H2_t_batch_constraint_end = pyo.Constraint(
            expr=mdl.LoS_H2_t_batch[len(self.SOCtime_batch) - 1] == 0
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

        pw_inputs = [pt[0] for pt in H2_curve_list]
        pw_outputs = [pt[1] for pt in H2_curve_list]

        if len(mdl.IDX1) > 0:
            mdl.H2_prod_pw = pyo.Piecewise(
                mdl.IDX1,
                mdl.m_H2_t_batch,
                mdl.P_ptg_t_batch,
                pw_pts=pw_inputs,
                f_rule=pw_outputs,
                pw_constr_type="EQ",
            )

        # Create ConstraintLists to collect per-time constraints
        mdl.h2_storage_eq = pyo.ConstraintList()
        mdl.h2_storage_capacity = pyo.ConstraintList()
        mdl.h2_offtake_demand = pyo.ConstraintList()
        mdl.h2_offtake_storage_zero = pyo.ConstraintList()

        # Precompute helper lookups mapping timestamps -> integer indices
        time_to_idx = {tt: i for i, tt in enumerate(self.time_batch)}
        soctime_to_idx = {tt: i for i, tt in enumerate(self.SOCtime_batch)}

        for t in self.time_batch:
            # integer index for current time
            k = time_to_idx[t]

            # successor timestamp t+1h
            tt_time = t + pd.Timedelta("1hour")

            # find SOC integer indices corresponding to t and tt_time (raise helpful KeyError if missing)
            try:
                i_soc = soctime_to_idx[t]
            except KeyError:
                raise KeyError(
                    f"Time {t!r} not found in self.SOCtime_batch; can't index LoS_H2_t_batch."
                )

            try:
                j_soc = soctime_to_idx[tt_time]
            except KeyError:
                raise KeyError(
                    f"Successor time {tt_time!r} (t+1h) not found in self.SOCtime_batch; "
                    "ensure SOCtime_batch contains successive timesteps."
                )

            # penalty term for this time t (use model var if available, otherwise treat off-take as 0)
            if k in mdl.IDX1:
                penalty_p2x = self.penalty_factor_H2 * (
                    self.m_H2_demand_ts_batch[t] - mdl.m_H2_offtake_t_batch[k]
                )
            else:
                penalty_p2x = self.penalty_factor_H2 * (
                    self.m_H2_demand_ts_batch[t] - 0
                )

            # (If you need to still call update_constraint_rhs_pyomo for penalty, re-enable here)
            # self.update_constraint_rhs_pyomo(
            #     mdl,
            #     constraint_name=f"penalty_{t}",
            #     added_value=penalty_p2x,
            #     idx=t,
            # )

            # If P_ptg exists at this k (peak hour) include it; else 0
            if k in mdl.IDX1:
                ptg_term = -mdl.P_ptg_t_batch[k]
            else:
                ptg_term = 0

            # Add the P_ptg term into your power_constraint RHS (you already call update_constraint_rhs_pyomo)
            self.update_constraint_rhs_pyomo(
                mdl,
                constraint_name="power_constraint",
                added_value=ptg_term,
                idx=k,
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
                mdl.LoS_H2_t_batch[i_soc] <= self.H2_storage_t_batch[t]
            )

            # m_H2_offtake <= demand (only when var exists) -> add to ConstraintList
            if k in mdl.IDX1:
                mdl.h2_offtake_demand.add(
                    mdl.m_H2_offtake_t_batch[k] <= self.m_H2_demand_ts_batch[t]
                )

            # When H2 storage is zero: enforce equality if the variables exist for this k -> add to ConstraintList
            if self.H2_storage_t_batch[t] == 0 and k in mdl.IDX1:
                mdl.h2_offtake_storage_zero.add(
                    mdl.m_H2_offtake_t_batch[k] == mdl.m_H2_t_batch[k]
                )

    def pyomo_opjective(self, mdl, **kwargs):
        H2_revenue = sum(
            self.price_H2_batch * mdl.m_H2_offtake_t_batch[t] for t in mdl.IDX1
        )

        self.add_to_objective_pyomo(mdl, addition_expr=H2_revenue)

    def pyomo_outputs(self, mdl, **kwargs):
        # Extract Pyomo solution values as dictionaries
        P_ptg_ts_dict = series_from_indexed_var(mdl.P_ptg_t_batch, self.time_batch)
        m_H2_ts_dict = series_from_indexed_var(mdl.m_H2_t_batch, self.time_batch)
        LoS_H2_ts_dict = series_from_indexed_var(mdl.LoS_H2_t_batch, self.time_batch)
        m_H2_offtake_ts_dict = series_from_indexed_var(
            mdl.m_H2_offtake_t_batch, self.time_batch
        )

        # Convert to pandas Series / DataFrame for convenience
        P_ptg_ts_df_batch = pd.Series(P_ptg_ts_dict).reindex(
            self.time_batch, fill_value=0
        )
        m_H2_ts_df_batch = pd.Series(m_H2_ts_dict).reindex(
            self.time_batch, fill_value=0
        )
        LoS_H2_ts_df_batch = pd.Series(LoS_H2_ts_dict).reindex(
            self.time_batch, fill_value=0
        )
        m_H2_offtake_ts_df_batch = pd.Series(m_H2_offtake_ts_dict).reindex(
            self.time_batch, fill_value=0
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

    def ortools_variables(self, solver):

        # Hydrogen storage level
        self.LoS_H2_t_batch = {}
        self.LoS_H2_t_batch = {
            t: solver.NumVar(0, self.H2_storage_t_batch[t], f"LoS_H2_{t}")
            for t in self.time_batch
        }

        # Power to gas plant consumption
        P_min = self.min_power_standby * self.ptg_MW
        P_max = self.ptg_MW
        self.P_ptg_t_batch = {}
        for t in self.time_batch:
            self.P_ptg_t_batch[t] = solver.NumVar(P_min, P_max, f"P_ptg_{t}")

        # Produced hydrogen
        self.m_H2_t_batch = {}
        for t in self.time_batch:
            self.m_H2_t_batch[t] = solver.NumVar(
                0, 1e6, f"m_H2_{t}"
            )  # large upper bound

        # Hydrogen offtake
        self.m_H2_offtake_t_batch = {}
        for t in self.time_batch:
            self.m_H2_offtake_t_batch[t] = solver.NumVar(0, 1e6, f"m_H2_offtake_{t}")

        # Optional penalty variable
        self.penalty_var = solver.NumVar(0, 1e6, "penalty_H2")

    def ortools_constraints(self, solver):
        # Initial and end LoS_H2
        solver.Add(self.LoS_H2_t_batch[self.SOCtime_batch[0]] == 0)
        solver.Add(self.LoS_H2_t_batch[self.SOCtime_batch[-1]] == 0)

        # Prepare electrolyzer piecewise linear
        # H2_curve_list = [(P, H2_prod)] for breakpoints
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
                f'electrolyzer_eff_curve_type is "{self.electrolyzer_eff_curve_type}". Available options: ["production","efficiency"]'
            )

        # Convex combination variables for piecewise linearization
        self.lambda_vars = {}
        for t in self.time_batch:
            self.lambda_vars[t] = []
            for i, (P_i, H2_i) in enumerate(H2_curve_list):
                var = solver.NumVar(0, 1, f"lambda_{t}_{i}")
                self.lambda_vars[t].append(var)
            # Convex sum = 1
            solver.Add(solver.Sum(self.lambda_vars[t]) == 1)
            # Link P_ptg and m_H2
            solver.Add(
                self.P_ptg_t_batch[t]
                == solver.Sum(
                    P_i * self.lambda_vars[t][i]
                    for i, (P_i, _) in enumerate(H2_curve_list)
                )
            )
            solver.Add(
                self.m_H2_t_batch[t]
                == solver.Sum(
                    H2_i * self.lambda_vars[t][i]
                    for i, (_, H2_i) in enumerate(H2_curve_list)
                )
            )

        # Time-step constraints
        for i, t in enumerate(self.time_batch[:-1]):
            tt = self.time_batch[i + 1]

            # Hydrogen storage balance
            solver.Add(
                self.LoS_H2_t_batch[tt]
                == self.LoS_H2_t_batch[t]
                + (self.m_H2_t_batch[t] - self.m_H2_offtake_t_batch[t])
                / self.storage_eff
            )

            # Storage limit
            solver.Add(self.LoS_H2_t_batch[t] <= self.H2_storage_t_batch[t])

            # Offtake limit
            solver.Add(self.m_H2_offtake_t_batch[t] <= self.m_H2_demand_ts_batch[t])

            # If storage=0, offtake=production
            if self.H2_storage_t_batch[t] == 0:
                solver.Add(self.m_H2_offtake_t_batch[t] == self.m_H2_t_batch[t])

        # Penalty constraints
        solver.Add(
            self.penalty_var
            >= solver.Sum(
                self.penalty_factor_H2
                * (self.m_H2_demand_ts_batch[t] - self.m_H2_offtake_t_batch[t])
                for t in self.time_batch
            )
        )

    def ortools_objective(self, solver):
        objective = solver.Objective()
        for t in self.time_batch:
            objective.SetCoefficient(
                self.m_H2_offtake_t_batch[t], float(self.price_H2_batch)
            )
        objective.SetMaximization()

    def ortools_outputs(self):
        P_ptg_ts_batch = [
            self.P_ptg_t_batch[t].solution_value() for t in self.time_batch
        ]
        m_H2_ts_batch = [self.m_H2_t_batch[t].solution_value() for t in self.time_batch]
        LoS_H2_ts_batch = [
            self.LoS_H2_t_batch[t].solution_value() for t in self.time_batch
        ]
        m_H2_offtake_ts_batch = [
            self.m_H2_offtake_t_batch[t].solution_value() for t in self.time_batch
        ]

        # Adjust for minimum power standby
        for t in range(self.N_t_batch):
            if abs(P_ptg_ts_batch[t] - self.min_power_standby * self.ptg_MW) < 1e-6:
                self.P_ptg_SB_ts[t] = self.min_power_standby * self.ptg_MW
                P_ptg_ts_batch[t] = 0
                m_H2_ts_batch[t] = 0

        # Save results
        self.P_ptg_ts[self.batch] = P_ptg_ts_batch
        self.m_H2_ts[self.batch] = m_H2_ts_batch
        self.LoS_H2_ts[self.batch] = LoS_H2_ts_batch
        self.m_H2_offtake_ts[self.batch] = m_H2_offtake_ts_batch

    def update_final_outputs(self, out):
        out_p2x = {
            "P_ptg_ts": self.P_ptg_ts,
            "P_ptg_SB_ts": self.P_ptg_SB_ts,
            "m_H2_ts": self.m_H2_ts,
            "m_H2_offtake_ts": self.m_H2_offtake_ts,
            "LoS_H2_ts": self.LoS_H2_ts,
        }
        out.update(out_p2x)

    # mdl.end()

    # # Cplex sometimes returns missing values :O
    # P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values
    # P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values
    # P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values
    # E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values
    # P_ptg_ts = P_ptg_ts_df.reindex(time, fill_value=0).values
    # m_H2_ts = m_H2_ts_df.reindex(time, fill_value=0).values
    # LoS_H2_ts = LoS_H2_ts_df.reindex(time, fill_value=0).values
    # m_H2_offtake_ts = m_H2_offtake_ts_df.reindex(time, fill_value=0).values

    # if len(P_HPP_ts_df) < len(wind_ts):
    #     # print('recomputing p_hpp')
    #     P_HPP_ts = (
    #         wind_ts.values
    #         + solar_ts.values
    #         + -P_curtailment_ts
    #         + P_charge_discharge_ts
    #         - P_ptg_ts
    #     )
    # P_ptg_SB_ts = np.zeros_like(P_ptg_ts)
    # for t in range(N_t):
    #     if abs(P_ptg_ts[t] - min_power_standby * ptg_MW) < 1e-6:
    #         P_ptg_SB_ts[t] = min_power_standby * ptg_MW
    #         P_ptg_ts[t] = 0
    #         m_H2_ts[t] = 0

    # return (
    #     P_HPP_ts,
    #     P_curtailment_ts,
    #     P_charge_discharge_ts,
    #     P_ptg_ts,
    #     P_ptg_SB_ts,
    #     E_SOC_ts,
    #     m_H2_ts,
    #     m_H2_offtake_ts,
    #     LoS_H2_ts,
    #     penalty_ts,
    # )
