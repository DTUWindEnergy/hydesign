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

    def batch_data(self, batch, **kwargs) -> Dict[str, Any]:
        self.solar_ts_batch = self.solar_ts.iloc[batch]

    def cplex_powerbalance(self, **kwargs) -> Dict[str, Any]:
        for t in time_batch:
            P_total[t] += self.solar_ts_batch[t]

    def pyomo_powerbalance(self, **kwargs) -> Dict[str, Any]:
        for t in mdl.IDX1:
            P_total[t] += self.solar_ts_batch[t]

    def ortools_powerbalance(self, **kwargs) -> Dict[str, Any]:
        for i, t in enumerate(time_batch):
            P_total[t] += self.solar_ts_batch[t]
