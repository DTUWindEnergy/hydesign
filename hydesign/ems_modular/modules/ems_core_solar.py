from typing import Any, Dict

from hydesign.ems_modular.modules.ems_core_helpers import EMSCoreABC


class EMSCoreSolar(EMSCoreABC):
    """
    Concrete EMS core implementation based on original EMSCoreBase code.
    Methods follow the original logic but are instance methods.
    """

    def __init__(self, **kwargs):
        self.solar_ts = kwargs["solar_ts"]

    def batch_data(self, batch, **kwargs):
        self.solar_ts_batch = self.solar_ts.iloc[batch]

    def cplex_powerbalance(self, **kwargs):
        for t in time_batch:
            P_total[t] += self.solar_ts_batch[t]

    def pyomo_powerbalance(self, **kwargs):
        for t in mdl.IDX1:
            P_total[t] += self.solar_ts_batch.iloc[t]

    def ortools_powerbalance(self, **kwargs):
        for i, t in enumerate(time_batch):
            P_total[t] += self.solar_ts_batch[t]
