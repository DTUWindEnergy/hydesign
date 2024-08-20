# Built by ems_builder
# %%
import numpy as np
import pandas as pd
import openmdao.api as om
from docplex.mp.model import Model
import openmdao.api as om

from .ems_wind import ems_wind_abc, ems_wind_long_term_operation, ems_cplex_wind, ems_cplex_parts_wind
from .ems_solar import ems_solar_abc, ems_solar_long_term_operation, ems_cplex_solar, ems_cplex_parts_solar
from .ems_bess import ems_bess_abc, ems_bess_long_term_operation, ems_cplex_bess, ems_cplex_parts_bess
from .ems_base import ems_base_abc, ems_base_long_term_operation, ems_cplex_base, ems_cplex_parts_base_pre, ems_cplex_parts_base_post, ems_degredation_base
from .ems_utils import expand_to_lifetime, split_in_batch, print_cplex_model_details
from .ems_builder_utils import ems_cplex, ems_cplex_parts, operation_solar_batt_deg
def init(self, inputs):
    self.weeks_per_season_per_year = inputs.get('weeks_per_season_per_year')
    self.N_time = int(inputs.get('N_time'))
    self.ems_type = inputs.get('ems_type', 'cplex')
    self.technology_types = inputs.get('technology_types', ['wind', 'solar', 'bess'])
    self.life_h = int(inputs.get('life_h', 25*365*24))
    self.available_ems_types = {
        'cplex': ems_cplex}
    # set default order of the technologies
    self.technology_types_and_base = ['base'] + self.technology_types
def setup(component):
    """setup of the cplex model"""
    # add inputs to the openmdao model
    # inputs added for base
    component.add_input(
        'price_t',
        desc="Electricity price time series",
        shape=[component.N_time])
    component.add_input(
        'G_MW',
        desc="Grid capacity",
        units='MW')

    component.add_input(
        'peak_hr_quantile',
    )

    component.add_input(
        'n_full_power_hours_expected_per_day_at_peak_price',
        desc="Penalty occurs if number of full power hours expected per day at peak price are not reached.")

    # inputs added for wind
    component.add_input(
        'wind_t',
        desc="WPP power time series",
        units='MW',
        shape=[component.N_time])

    # inputs added for solar
    component.add_input(
        'solar_t',
        desc="PVP power time series",
        units='MW',
        shape=[component.N_time])

    # inputs added for bess
    component.add_input(
        'b_P',
        desc="Battery power capacity",
        units='MW')
    component.add_input(
        'b_E',
        desc="Battery energy storage capacity")
    component.add_input(
        'battery_depth_of_discharge',
        desc="battery depth of discharge",
        units='MW')
    component.add_input(
        'battery_charge_efficiency',
        desc="battery charge efficiency")
    component.add_input(
        'cost_of_battery_P_fluct_in_peak_price_ratio',
        desc="cost of battery power fluctuations computed as a peak price ratio.")

    # add outputs to the openmdao model
    # outputs added for base
    component.add_output(
        'price_t_ext',
        desc="Electricity price time series",
        shape=[component.life_h])
    component.add_output(
        'hpp_t',
        desc="HPP power time series",
        units='MW',
        shape=[component.life_h])
    component.add_output(
        'hpp_curt_t',
        desc="HPP curtailed power time series",
        units='MW',
        shape=[component.life_h])
    component.add_output(
        'penalty_t',
        desc="Penalty for not reaching expected energy production at peak hours",
        shape=[component.life_h])

    # outputs added for wind
    component.add_output(
        'wind_t_ext',
        desc="WPP power time series",
        units='MW',
        shape=[component.life_h])

    # outputs added for solar
    component.add_output(
        'solar_t_ext',
        desc="PVP power time series",
        units='MW',
        shape=[component.life_h])

    # outputs added for bess
    component.add_output(
        'b_t',
        desc="Battery charge/discharge power time series",
        units='MW',
        shape=[component.life_h])
    component.add_output(
        'b_E_SOC_t',
        desc="Battery energy SOC time series",
        shape=[component.life_h + 1])
def compute(self, inputs, outputs):
    # Get the appropriate EMS function name based on ems_type
    ems_type = self.available_ems_types.get(self.ems_type)

    # Raise a warning if the ems_type is not found in the dictionary
    if ems_type is None:
        raise Warning("Selected ems type is not supported")

    # Build a sintetic time to avoid problems with time sereis
    # indexing in ems
    WSPr_df = pd.DataFrame(
        index=pd.date_range(
            start='01-01-1991 00:00',
            periods=len(inputs['wind_t']),
            freq='1h'))

    ems_args = {}

    # ems arguments
    # arguments added for base
    G_MW = inputs['G_MW']
    peak_hr_quantile = inputs['peak_hr_quantile'][0]
    n_full_power_hours_expected_per_day_at_peak_price = inputs[
        'n_full_power_hours_expected_per_day_at_peak_price'][0]
    WSPr_df['price_t'] = inputs['price_t']


    added_ems_args = {
        'price_ts': WSPr_df.price_t,
        'G_MW': G_MW[0],
        'peak_hr_quantile': peak_hr_quantile,
        'n_full_power_hours_expected_per_day_at_peak_price': n_full_power_hours_expected_per_day_at_peak_price,
        'technology_types': ['wind', 'solar', 'bess'],
        }

    ems_args.update(added_ems_args)

    # arguments added for wind
    WSPr_df['wind_t'] = inputs['wind_t']
    added_ems_args = {'wind_ts': WSPr_df.wind_t}

    ems_args.update(added_ems_args)
    # make sure the variable is not overwritten


    # arguments added for solar
    WSPr_df['solar_t'] = inputs['solar_t']

    added_ems_args = {'solar_ts': WSPr_df.solar_t,}

    ems_args.update(added_ems_args)

    # arguments added for bess
    b_P = inputs['b_P']
    b_E = inputs['b_E']
    WSPr_df['b_E'] = inputs['b_E'][0]

    battery_depth_of_discharge = inputs['battery_depth_of_discharge']
    battery_charge_efficiency = inputs['battery_charge_efficiency']
    cost_of_battery_P_fluct_in_peak_price_ratio = inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0]

    added_ems_args = {
        'b_P': b_P[0],
        'b_E': WSPr_df.b_E,
        'battery_depth_of_discharge': battery_depth_of_discharge[0],
        'battery_charge_efficiency': battery_charge_efficiency[0],
        'cost_of_battery_P_fluct_in_peak_price_ratio': cost_of_battery_P_fluct_in_peak_price_ratio
        }

    ems_args.update(added_ems_args)

    outputs_ems = ems_type(ems_args) 
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

