# basic libraries

from typing import Any, Dict

import numpy as np

from hydesign.ems.ems_core_helpers import EMSCoreInterface


class EMSCoreWind(EMSCoreInterface):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        super().__init__(**kwargs)

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        """ """
        self.wind_ts_batch = self.wind_ts.iloc[batch]
        self.price_ts_batch = self.price_ts.iloc[batch]
        self.batch = batch

        self.time_batch = self.price_ts_batch.index

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """
        for t in self.time_batch:
            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.wind_ts_batch[t],
            )

    def ortools_constraints(self, solver, **kwargs) -> Dict[str, Any]:
        """ """
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
