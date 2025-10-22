# basic libraries

from typing import Any, Dict

import numpy as np

# from numpy import newaxis as na
import pandas as pd

from .ems_core_helpers import EMSCoreInterface


class EMSCoreSolar(EMSCoreInterface):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs) -> Dict[str, Any]:
        super().__init__(**kwargs)

    def solver_init(self, batch, **kwargs) -> Dict[str, Any]:
        """ """
        self.price_ts_batch = self.price_ts.iloc[batch]
        self.solar_ts_batch = self.solar_ts.iloc[batch]
        self.batch = batch

        self.time_batch = self.price_ts_batch.index
        # time set with an additional time slot for the last soc
        self.SOCtime_batch = self.time_batch.append(
            pd.Index([self.time_batch[-1] + pd.Timedelta("1hour")])
        )

    def cplex_constraints(self, mdl, **kwargs) -> Dict[str, Any]:
        """ """

        for t in self.time_batch:
            self.update_constraint_rhs(
                mdl,
                constraint_name=f"P_HPP_t_balance{t}",
                added_value=+self.solar_ts_batch[t],
            )

    def ortools_constraints(self, solver) -> Dict[str, Any]:
        """ """
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
