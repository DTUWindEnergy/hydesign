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
def init(self):
    ''
def setup(component):
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
def compute(self, inputs, outputs):
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

