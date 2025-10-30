# basic libraries

from typing import Any, Dict

import pandas as pd
import pyomo.environ as pyo

from hydesign.ems_modular.modules.ems_core_helpers import EMSCoreInterface


class EMSCoreSolar(EMSCoreInterface):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        self.solar_ts = kwargs["solar_ts"]

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        self.solar_ts_batch = self.solar_ts.iloc[batch]

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        for t in self.time_batch:
            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.solar_ts_batch[t],
            )

    def ortools_constraints(self, solver) -> Dict[str, Any]:
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

    def pyomo_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        for t in range(0, len(self.time_batch)):
            # solar
            self.update_constraint_rhs_pyomo(
                mdl, "power_constraint", added_value=self.solar_ts_batch.iloc[t], idx=t
            )
