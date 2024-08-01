from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import openmdao.api as om
from docplex.mp.model import Model


class AbstractEMS(om.ExplicitComponent, ABC):
    def __init__(self, N_time, life_h=25 * 365 * 24, weeks_per_season_per_year=None, ems_type='cplex'):
        super().__init__()
        self.weeks_per_season_per_year = weeks_per_season_per_year
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_h = int(life_h)

    def setup_common_inputs(self):
        self.add_input('wind_t', desc="WPP power time series", units='MW', shape=[self.N_time])
        self.add_input('solar_t', desc="PVP power time series", units='MW', shape=[self.N_time])
        self.add_input('price_t', desc="Electricity price time series", shape=[self.N_time])
        self.add_input('b_P', desc="Battery power capacity", units='MW')
        self.add_input('b_E', desc="Battery energy storage capacity")
        self.add_input('G_MW', desc="Grid capacity", units='MW')
        self.add_input('battery_depth_of_discharge', desc="battery depth of discharge", units='MW')
        self.add_input('battery_charge_efficiency', desc="battery charge efficiency")
        self.add_input('peak_hr_quantile',
                       desc="Quantile of price tim series to define peak price hours (above this quantile).\nOnly used for peak production penalty and for cost of battery degradation.")
        self.add_input('cost_of_battery_P_fluct_in_peak_price_ratio', desc="cost of battery power fluctuations computed as a peak price ratio.")
        self.add_input('n_full_power_hours_expected_per_day_at_peak_price',
                       desc="Penalty occurs if number of full power hours expected per day at peak price are not reached.")

    def setup_common_outputs(self):
        self.add_output('wind_t_ext', desc="WPP power time series", units='MW', shape=[self.life_h])
        self.add_output('solar_t_ext', desc="PVP power time series", units='MW', shape=[self.life_h])
        self.add_output('price_t_ext', desc="Electricity price time series", shape=[self.life_h])
        self.add_output('hpp_t', desc="HPP power time series", units='MW', shape=[self.life_h])
        self.add_output('hpp_curt_t', desc="HPP curtailed power time series", units='MW', shape=[self.life_h])
        self.add_output('b_t', desc="Battery charge/discharge power time series", units='MW', shape=[self.life_h])
        self.add_output('b_E_SOC_t', desc="Battery energy SOC time series", shape=[self.life_h + 1])
        self.add_output('penalty_t', desc="penalty for not reaching expected energy production at peak hours", shape=[self.life_h])

    @abstractmethod
    def compute(self, inputs, outputs):
        pass

    def expand_to_lifetime(self, x, life_h=25 * 365 * 24, weeks_per_season_per_year=None):
        """
        Expands (by repeating) a given variable to match an expected lifetime length.

        If weeks_per_season_per_year is an int then it will first build a year out of the selected weeks

        Parameters
        ----------
        x: input variable
        life_h: lifetime in hours.
        weeks_per_season_per_year: None or int.

        Returns
        -------
        x_ext: extended variable
        """
        if weeks_per_season_per_year == None:
            len_x = len(x)
            N_repeats = int(np.ceil(life_h / len_x))
        else:
            x_ext = np.array([])
            for x_batch in self.split_in_batch(x, weeks_per_season_per_year * 7 * 24):
                x_ext = np.append(x_ext, np.tile(x_batch, 20)[:24 * 7 * 13])
            x_ext = np.append(x_ext, x_batch[-24:])
            x = x_ext
            N_repeats = int(np.ceil(life_h / 365 * 24))

        return np.tile(x, N_repeats)[:life_h]

    @staticmethod
    def split_in_batch(array, N):
        batch = []
        counter = 0
        while counter * N < len(array):
            if (counter + 1) * N > len(array):
                end = len(array)
            else:
                end = (counter + 1) * N
            batch += [array[counter * N:end]]
            counter += 1
        return batch
