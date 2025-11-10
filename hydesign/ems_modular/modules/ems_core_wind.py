from hydesign.ems_modular.modules.ems_core_helpers import EMSCoreABC


class EMSCoreWind(EMSCoreABC):
    """
    EMSCoreWind
    """

    def __init__(self, **kwargs):
        self.wind_ts = kwargs["wind_ts"]

    def batch_data(self, batch, **kwargs):
        self.wind_ts_batch = self.wind_ts.iloc[batch]

    def cplex_powerbalance(self, **kwargs):
        for t in time_batch:
            P_total[t] += self.wind_ts_batch[t]

    def pyomo_powerbalance(self, **kwargs):
        for t in mdl.IDX1:
            P_total[t] += self.wind_ts_batch.iloc[t]

    def ortools_powerbalance(self, **kwargs):
        for i, t in enumerate(time_batch):
            P_total[t] += self.wind_ts_batch[t]
