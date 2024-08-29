# Built by ems_builder
# %%
import inspect
import trace
import sys
import os
import numpy as np
import pandas as pd
import importlib

import openmdao.api as om
from docplex.mp.model import Model
import openmdao.api as om

from . import builded_ems, builded_ems_long_term_operation
from .ems_wind import ems_wind_abc, ems_wind_long_term_operation, ems_cplex_wind, ems_cplex_parts_wind
from .ems_solar import ems_solar_abc, ems_solar_long_term_operation, ems_cplex_solar, ems_cplex_parts_solar
from .ems_bess import ems_bess_abc, ems_bess_long_term_operation, ems_cplex_bess, ems_cplex_parts_bess
from .ems_base import ems_base_abc, ems_base_long_term_operation, ems_cplex_base, ems_cplex_parts_base_pre, ems_cplex_parts_base_post, ems_degredation_base
from .ems_utils import expand_to_lifetime, split_in_batch, print_cplex_model_details
from .ems_builder_utils import insert_ems_section, insert_ems_long_term_operation_section, get_path_builded_ems
from .ems_builder_utils import ems_cplex, ems_cplex_parts, operation_solar_batt_deg
    
class ems(om.ExplicitComponent):
    def __init__(self, inputs):
        super().__init__()
        self.weeks_per_season_per_year = inputs.get('weeks_per_season_per_year')
        self.N_time = int(inputs.get('N_time'))
        self.technology_types = inputs.get('technology_types', ['wind', 'solar', 'bess'])
        self.life_h = int(inputs.get('life_h', 25*365*24))
        self.technology_types_and_base = ['base'] + self.technology_types  # set default order of the technologies

    def setup(self):
        # add inputs to the openmdao model
        # inputs added for base
        self.add_input(
            'price_t',
            desc="Electricity price time series",
            shape=[self.N_time])
        self.add_input(
            'G_MW',
            desc="Grid capacity",
            units='MW')

        self.add_input(
            'peak_hr_quantile',
        )

        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Penalty occurs if number of full power hours expected per day at peak price are not reached.")

        # inputs added for wind
        self.add_input(
            'wind_t',
            desc="WPP power time series",
            units='MW',
            shape=[self.N_time])

        # inputs added for solar
        self.add_input(
            'solar_t',
            desc="PVP power time series",
            units='MW',
            shape=[self.N_time])

        # inputs added for bess
        self.add_input(
            'b_P',
            desc="Battery power capacity",
            units='MW')
        self.add_input(
            'b_E',
            desc="Battery energy storage capacity")
        self.add_input(
            'battery_depth_of_discharge',
            desc="battery depth of discharge",
            units='MW')
        self.add_input(
            'battery_charge_efficiency',
            desc="battery charge efficiency")
        self.add_input(
            'cost_of_battery_P_fluct_in_peak_price_ratio',
            desc="cost of battery power fluctuations computed as a peak price ratio.")

        # add outputs to the openmdao model
        # outputs added for base
        self.add_output(
            'price_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])
        self.add_output(
            'hpp_t',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'penalty_t',
            desc="Penalty for not reaching expected energy production at peak hours",
            shape=[self.life_h])

        # outputs added for wind
        self.add_output(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])

        # outputs added for solar
        self.add_output(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])

        # outputs added for bess
        self.add_output(
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])

    def compute(self, inputs, outputs):
        # Build a sintetic time to avoid problems with time sereis
        # indexing in ems
        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start='01-01-1991 00:00',
                periods=len(inputs['wind_t']),
                freq='1h'))

        # ems arguments
        WSPr_df['price_t'] = inputs['price_t']
        WSPr_df['wind_t'] = inputs['wind_t']
        WSPr_df['solar_t'] = inputs['solar_t']
        WSPr_df['b_E'] = inputs['b_E'][0]
        ems_args = {

            # args base
            'price_ts': WSPr_df.price_t,
            'G_MW': inputs['G_MW'][0],
            'peak_hr_quantile': inputs['peak_hr_quantile'][0],
            'n_full_power_hours_expected_per_day_at_peak_price': inputs['n_full_power_hours_expected_per_day_at_peak_price'][0],
            'technology_types': ['wind', 'solar', 'bess'],

            # args wind
            'wind_ts':  WSPr_df.wind_t,

            # args solar
            'solar_ts': WSPr_df.solar_t,

            # args bess
            'b_P': inputs['b_P'][0],
            'b_E': WSPr_df.b_E,
            'battery_depth_of_discharge': inputs['battery_depth_of_discharge'][0],
            'battery_charge_efficiency': inputs['battery_charge_efficiency'][0],
            'cost_of_battery_P_fluct_in_peak_price_ratio': inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0],
         }

        outputs_ems = ems_cplex(ems_args) 

        # post processing outputs
        # post processing outputs for base
        price_t = inputs['price_t']
        P_HPP_ts = outputs_ems['hpp_t']
        P_curtailment_ts = outputs_ems['hpp_curt_t']
        penalty_ts = outputs_ems['penalty_t']
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['price_t_ext'] = expand_to_lifetime(
            price_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['hpp_t'] = expand_to_lifetime(
            P_HPP_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['hpp_curt_t'] = expand_to_lifetime(
            P_curtailment_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['penalty_t'] = expand_to_lifetime(
            penalty_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)

        # post processing outputs for wind
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['wind_t_ext'] = expand_to_lifetime(
            inputs['wind_t'], life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)

        # post processing outputs for solar
        solar_t = inputs['solar_t']
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['solar_t_ext'] = expand_to_lifetime(
            solar_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)

        # post processing outputs for bess
        P_charge_discharge_ts = outputs_ems['b_t']
        E_SOC_ts = outputs_ems['b_E_SOC_t']
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['b_t'] = expand_to_lifetime(
            P_charge_discharge_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['b_E_SOC_t'] = expand_to_lifetime(
            E_SOC_ts[:-1], life_h=self.life_h + 1, weeks_per_season_per_year=self.weeks_per_season_per_year)

