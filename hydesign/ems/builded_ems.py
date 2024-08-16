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
def init_ems(self):
    class_map = {
        'base': ems_base_abc,
        'wind': ems_wind_abc,
        'solar': ems_solar_abc,
        'bess': ems_bess_abc
    }

    # Dynamically create and assign instances of EMS classes for each technology type
    for techonogy_type in self.technology_types_and_base:
        setattr(self, f"ems_{techonogy_type}_abc_instance", class_map[techonogy_type]())
def init_ems_long_term_operation(self):
    class_map = {
        'base': ems_base_long_term_operation,
        'wind': ems_wind_long_term_operation,
        'solar': ems_solar_long_term_operation,
        'bess': ems_bess_long_term_operation
    }

    # Dynamically create and assign instances of ems_long_term_operation classes for each technology type
    for techonogy_type in self.technology_types_and_base:
        setattr(self, f"ems_{techonogy_type}_long_term_operation_instance", class_map[techonogy_type]())
def setup_ems(component):
    """setup of the cplex model"""

    # add inputs to the cplex model
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

    # add outputs to the cplex model
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
def setup_ems_long_term_operation(component):
    """setup of the cplex model"""

    # add inputs to the cplex model
    # inputs added for base
    component.add_input(
        'price_t_ext',
        desc="Electricity price time series",
        shape=[component.life_h])
    component.add_input(
        'G_MW',
        desc="Grid capacity",
        units='MW')
    component.add_input(
        'hpp_curt_t',
        desc="HPP curtailed power time series",
        units='MW',
        shape=[component.life_h])
    component.add_input(
        'peak_hr_quantile',
             ) # "Only used for peak production penalty and for cost of battery degradation."
    component.add_input(
        'n_full_power_hours_expected_per_day_at_peak_price',
        desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")
    component.add_input(
        'load_min',
        desc="Minimum electrical load to meet")

    # inputs added for solar
    component.add_input(
        'solar_t_ext_deg',
        desc="PV time series including degradation",
        units='MW',
        shape=[component.life_h])
    component.add_input(
        'solar_t_ext',
        desc="PVP power time series",
        units='MW',
        shape=[component.life_h])

    # inputs added for wind
    component.add_input(
        'wind_t_ext_deg',
        desc="Wind time series including degradation",
        units='MW',
        shape=[component.life_h])
    component.add_input(
        'wind_t_ext',
        desc="WPP power time series",
        units='MW',
        shape=[component.life_h])

    # inputs added for bess
    component.add_input(
        'SoH',
        desc="Battery state of health at discretization levels",
        shape=[component.life_h])
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
        'b_t',
        desc="Battery charge/discharge power time series",
        units='MW',
        shape=[component.life_h])
    component.add_input(
        'b_E_SOC_t',
        desc="Battery energy SOC time series",
        shape=[component.life_h + 1])

    # add outputs to the cplex model
    # outputs added for base
    component.add_output(
        'hpp_t_with_deg',
        desc="HPP power time series",
        units='MW',
        shape=[component.life_h])
    component.add_output(
        'hpp_curt_t_with_deg',
        desc="HPP curtailed power time series",
        units='MW',
        shape=[component.life_h])
    component.add_output(
        'penalty_t_with_deg',
        desc="penalty for not reaching expected energy productin at peak hours",
        shape=[component.life_h])
    component.add_output(
        'total_curtailment_with_deg',
        desc="total curtailment in the lifetime after degradation",
        units='GW*h',
    )
    component.add_output(
        'total_curtailment',
        desc="total curtailment in the lifetime",
        units='GW*h',
    )

    # outputs added for solar
    pass

    # outputs added for wind
    pass

    # outputs added for bess
    component.add_output(
        'b_t_with_deg',
        desc="Battery charge/discharge power time series",
        units='MW',
        shape=[component.life_h])
    component.add_output(
        'b_E_SOC_t_with_deg',
        desc="Battery energy SOC time series",
        shape=[component.life_h + 1])
def operation_solar_batt_deg(inputs):
    """EMS operation for degraded PV and battery based on an existing EMS."""

    # instantial the class
    ems_degredation_instance = ems_degredation_base(inputs)

    # wind + solar after degredation
    wind_solar_t_deg = ems_degredation_instance.compute_wind_solar_deg()
    G_MW = inputs.get('G_MW')

    # addjust battery power based on the available power
    b_t_less_sol = ems_degredation_instance.adjust_battery_power(wind_solar_t_deg)

    # battery state of charge
    b_E_SOC_t_sat = ems_degredation_instance.initialize_soc()
    b_E_SOC_t_sat = ems_degredation_instance.update_soc(b_t_less_sol, b_E_SOC_t_sat)

    # update battery power
    b_t_sat = ems_degredation_instance.recompute_battery_power(b_E_SOC_t_sat)

    # compute Hpp and P_curt after degredation
    Hpp_deg = np.minimum(wind_solar_t_deg + b_t_sat, G_MW)
    P_curt_deg = np.maximum(wind_solar_t_deg + b_t_sat - G_MW, 0)

    # compuate penalty time series with degraded valud of HPP power
    penalty_ts = ems_degredation_instance.compute_penalty(Hpp_deg)

    # collect outputs
    outputs = {
        'Hpp_deg': Hpp_deg,
        'P_curt_deg': P_curt_deg,
        'b_t_sat': b_t_sat,
        'b_E_SOC_t_sat': b_E_SOC_t_sat,
        'penalty_ts': penalty_ts
    }

    return outputs
def compute_ems(self, inputs, outputs):
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

def compute_ems_long_term_operation(self, inputs, outputs):
    # life_h = self.life_h

    if self.ems_type == 'energy_penalty':
        ems_longterm = operation_solar_batt_deg
    else:
        raise Warning("This class should only be used for energy_penalty")

    ems_args = {}

    # ems_long arguments
    # arguments added for base
    price_t_ext = inputs['price_t_ext']
    G_MW = inputs['G_MW']
    hpp_curt_t = inputs['hpp_curt_t']
    peak_hr_quantile = inputs['peak_hr_quantile'][0]
    n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

    added_ems_args = dict(
                hpp_curt_t=hpp_curt_t,
                G_MW=G_MW[0],
                price_ts=price_t_ext,
                peak_hr_quantile=peak_hr_quantile,
                n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price, )

    ems_args.update(added_ems_args)

    # arguments added for solar
    solar_t_ext_deg = inputs['solar_t_ext_deg']
    solar_t_ext = inputs['solar_t_ext']

    added_ems_args = dict(
                solar_t_deg=solar_t_ext_deg,
                solar_t=solar_t_ext,
                )


    ems_args.update(added_ems_args)

    # arguments added for wind
    wind_t_ext_deg = inputs['wind_t_ext_deg']
    wind_t_ext = inputs['wind_t_ext']

    added_ems_args = dict(
                wind_t_deg=wind_t_ext_deg,
                wind_t=wind_t_ext,
                )


    ems_args.update(added_ems_args)

    # arguments added for bess
    SoH = inputs['SoH']
    b_E = inputs['b_E']
    battery_depth_of_discharge = inputs['battery_depth_of_discharge']
    battery_charge_efficiency = inputs['battery_charge_efficiency']
    b_t = inputs['b_t']
    b_E_SOC_t = inputs['b_E_SOC_t']

    added_ems_args = dict(
                batt_degradation=SoH,
                b_t=b_t,
                b_E_SOC_t=b_E_SOC_t,
                b_E=b_E[0],
                battery_depth_of_discharge=battery_depth_of_discharge[0],
                battery_charge_efficiency=battery_charge_efficiency[0],
                b_E_SOC_0=None,
                )

    ems_args.update(added_ems_args)

    outputs_ems = ems_longterm(ems_args)
    # post processing outputs
    # post processing outputs for base
    outputs['hpp_t_with_deg'] = outputs_ems['Hpp_deg']
    outputs['hpp_curt_t_with_deg'] = outputs_ems['P_curt_deg']
    outputs['penalty_t_with_deg'] = outputs_ems['penalty_ts']
    outputs['total_curtailment_with_deg'] = outputs_ems['P_curt_deg'].sum()
    outputs['total_curtailment'] = inputs['hpp_curt_t'].sum()


    # post processing outputs for solar


    # post processing outputs for wind


    # post processing outputs for bess
    outputs['b_t_with_deg'] = outputs_ems['b_t_sat']
    outputs['b_E_SOC_t_with_deg'] = outputs_ems['b_E_SOC_t_sat']

