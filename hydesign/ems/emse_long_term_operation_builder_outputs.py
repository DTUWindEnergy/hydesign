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

class ems_long_term_operation(om.ExplicitComponent):

    def __init__(
        self,
        N_time,
        num_batteries = 1,
        life_h = 25*365*24,
        ems_type='energy_penalty',
        load_min_penalty_factor=1e6,
        technology_types=['wind', 'solar', 'bess']
        ):

        super().__init__()
        self.N_time = N_time
        self.life_h = life_h
        self.ems_type = ems_type
        self.load_min_penalty_factor = load_min_penalty_factor
        self.technology_types = technology_types
        self.build_ems = True

        self.technology_types_and_base = ['base'] + ['solar', 'wind', 'bess']


    def setup(self):
    
        # add inputs to the cplex model
        # inputs added for base
        self.add_input(
            'price_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])
        self.add_input(
            'G_MW',
            desc="Grid capacity",
            units='MW')
        self.add_input(
            'hpp_curt_t',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'peak_hr_quantile',
                 ) # "Only used for peak production penalty and for cost of battery degradation."
        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")
        self.add_input(
            'load_min',
            desc="Minimum electrical load to meet")

        # inputs added for wind
        self.add_input(
            'wind_t_ext_deg',
            desc="Wind time series including degradation",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])

        # inputs added for solar
        self.add_input(
            'solar_t_ext_deg',
            desc="PV time series including degradation",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])

        # inputs added for bess
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_h])
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
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_input(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])

        # add outputs to the cplex model
        # outputs added for base
        self.add_output(
            'hpp_t_with_deg',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t_with_deg',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'penalty_t_with_deg',
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h])
        self.add_output(
            'total_curtailment_with_deg',
            desc="total curtailment in the lifetime after degradation",
            units='GW*h',
        )
        self.add_output(
            'total_curtailment',
            desc="total curtailment in the lifetime",
            units='GW*h',
        )

        # outputs added for wind
        pass

        # outputs added for solar
        pass

        # outputs added for bess
        self.add_output(
            'b_t_with_deg',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'b_E_SOC_t_with_deg',
            desc="Battery energy SOC time series",
            shape=[self.life_h + 1])

    def compute(self, inputs, outputs):
        # ems_long arguments
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
        outputs_ems = energy_penalty(ems_args) 

        # post processing outputs 
        # post processing outputs for base
        outputs['hpp_t_with_deg'] = outputs_ems['Hpp_deg']
        outputs['hpp_curt_t_with_deg'] = outputs_ems['P_curt_deg']
        outputs['penalty_t_with_deg'] = outputs_ems['penalty_ts']
        outputs['total_curtailment_with_deg'] = outputs_ems['P_curt_deg'].sum()
        outputs['total_curtailment'] = inputs['hpp_curt_t'].sum()

        # post processing outputs for wind

        # post processing outputs for solar

        # post processing outputs for bess
        outputs['b_t_with_deg'] = outputs_ems['b_t_sat']
        outputs['b_E_SOC_t_with_deg'] = outputs_ems['b_E_SOC_t_sat']

