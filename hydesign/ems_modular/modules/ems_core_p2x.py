# basic libraries
from typing import Any, Dict

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from hydesign.ems_modular.modules.ems_core_helpers import (
    EMSCoreInterface,
    add_piecewise_relation_ortools,
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

    def batch_data(self, batch, **kwargs) -> Dict[str, Any]:
        self.price_H2_batch = self.price_H2  # .iloc[batch]
        self.m_H2_demand_ts_batch = self.m_H2_demand_ts.iloc[batch]
        self.H2_storage_t_batch = self.H2_storage_t.iloc[batch]

    def cplex_variables(self, mdl, **kwargs) -> Dict[str, Any]:

        price_H2_batch = self.price_H2_batch
        m_H2_demand_ts_batch = self.m_H2_demand_ts_batch
        H2_storage_t_batch = self.H2_storage_t_batch
        # Hydrogen storgae level
        LoS_H2_t_batch = mdl.continuous_var_dict(
            self.SOCtime_batch, lb=0, name="Hydrogen storage level"
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

    def cplex_powerbalance(self, **kwargs) -> Dict[str, Any]:
        for t in time_batch:
            P_total[t] -= P_ptg_t_batch[t]

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:

        # # ------ penalty p2x
        # f1 = mdl.piecewise(0, [(0, 0)], 1)

        # mdl.add_constraint(
        #     self.penalty
        #     == self.price_peak_batch * f1(e_penalty)
        #     + mdl.sum(
        #         penalty_factor_H2 * (m_H2_demand_ts[t] - m_H2_offtake_t_batch[t]) for t in time
        #     )
        # )
        # # ------

        # Intitial and end LoS_H2
        mdl.add_constraint(LoS_H2_t_batch[self.SOCtime_batch[0]] == 0)
        mdl.add_constraint(LoS_H2_t_batch[self.SOCtime_batch[-1]] == 0)

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

        # self.update_constraint_rhs(
        #     mdl,
        #     constraint_name="penalty_el",
        #     added_value=penalty_p2x,
        # )

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

    def cplex_opjective(self, mdl):
        revenue_p2x = mdl.sum(
            price_H2_batch * m_H2_offtake_t_batch[t] for t in time_batch
        )
        objective_expr += revenue_p2x - penalty_p2x_batch

    def cplex_outputs(self, sol):

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

    def pyomo_variables(self, mdl, **kwargs) -> Dict[str, Any]:
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

    def pyomo_powerbalance(self, **kwargs) -> Dict[str, Any]:
        for t in mdl.IDX1:
            P_total[t] -= mdl.P_ptg_t_batch[t]

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
        time_to_idx = {tt: i for i, tt in enumerate(time_batch)}
        soctime_to_idx = {tt: i for i, tt in enumerate(self.SOCtime_batch)}

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
                    f"Time {t!r} not found in self.SOCtime_batch; can't index LoS_H2_t_batch."
                )

            try:
                j_soc = soctime_to_idx[tt_time]
            except KeyError:
                raise KeyError(
                    f"Successor time {tt_time!r} (t+1h) not found in self.SOCtime_batch; "
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
                    mdl.m_H2_offtake_t_batch[k] <= m_H2_demand_ts_batch[t]
                )

            # When H2 storage is zero: enforce equality if the variables exist for this k -> add to ConstraintList
            if H2_storage_t_batch[t] == 0 and k in mdl.IDX1:
                mdl.h2_offtake_storage_zero.add(
                    mdl.m_H2_offtake_t_batch[k] == mdl.m_H2_t_batch[k]
                )

        # penalty term for this time t (use model var if available, otherwise treat off-take as 0)

        penalty_p2x = sum(
            self.penalty_factor_H2
            * (m_H2_demand_ts_batch[t] - mdl.m_H2_offtake_t_batch[t])
            for t in mdl.IDX1
        )

        mdl.penalty_constraint.add(mdl.penalty_p2x_batch[0] == penalty_p2x)

    def pyomo_opjective(self, mdl, **kwargs):
        revenue_p2x = sum(
            price_H2_batch * mdl.m_H2_offtake_t_batch[t] for t in mdl.IDX1
        )
        penalty_p2x = -mdl.penalty_p2x_batch[0]
        objective_expr += revenue_p2x - penalty_p2x

    def pyomo_outputs(self, mdl, **kwargs):
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

    # --- ortools_variables ---
    def ortools_variables(self, solver):
        price_H2_batch = self.price_H2_batch
        m_H2_demand_ts_batch = self.m_H2_demand_ts_batch
        H2_storage_t_batch = self.H2_storage_t_batch
        # State-of-storage variables indexed by SOCtime_batch (as in CPLEX)
        LoS_H2_t_batch = {
            t: solver.NumVar(0.0, solver.infinity(), f"H2_storage_{t}")
            for t in self.SOCtime_batch
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
        """ # lambda variables placeholders (will be filled per timestep in constraints)
        self.lambdas_ptg_batch = {t: [] for t in time_batch}
        self.lambdas_H2_batch = {t: [] for t in time_batch}

        # penalty variables
        self.penalty_batch = solver.NumVar(0.0, solver.infinity(), "penalty_batch")
        self.e_penalty_batch = solver.NumVar(0.0, solver.infinity(), "e_penalty_batch") """

    def ortools_powerbalance(self, **kwargs) -> Dict[str, Any]:
        for i, t in enumerate(time_batch):
            P_total[t] -= P_ptg_t_batch[t]

    # --- ortools_constraints ---
    def ortools_constraints(self, solver: pywraplp.Solver):

        # initial and terminal LoS constraints (matching cplex)
        solver.Add(LoS_H2_t_batch[self.SOCtime_batch[0]] == 0.0)
        solver.Add(LoS_H2_t_batch[self.SOCtime_batch[-1]] == 0.0)

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

        # --- Constraints ---
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

        # self.update_constraint_rhs_ortools(
        #     solver,"penalty_batch",
        #     addition_expr=unmet_sum,
        #     coeff=1,
        # )

    # --- ortools_objective ---
    def ortools_objective(self, solver: pywraplp.Solver):
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

    # --- ortools_outputs ---
    def ortools_outputs(self):
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
