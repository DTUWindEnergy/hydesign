# basic libraries

from typing import Any, Dict

import pyomo.environ as pyo

from hydesign.ems_modular.modules.ems_core_helpers import EMSCoreInterface


class EMSCoreWind(EMSCoreInterface):
    """
    EMSCoreWind
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        self.wind_ts = kwargs["wind_ts"]

    def batch_data(self, batch, **kwargs) -> Dict[str, Any]:
        self.wind_ts_batch = self.wind_ts.iloc[batch]

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        for t in self.time_batch:
            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.wind_ts_batch[t],
            )

    def ortools_constraints(self, solver, **kwargs) -> Dict[str, Any]:
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

    def pyomo_constraints(self, mdl) -> Dict[str, Any]:
        for t in range(0, len(self.time_batch)):
            self.update_constraint_rhs_pyomo(
                mdl, "power_constraint", added_value=self.wind_ts_batch.iloc[t], idx=t
            )
