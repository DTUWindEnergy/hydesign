import numpy as np
import pandas as pd
from docplex.mp.model import Model
import inspect
import os
from .ems_utils import split_in_batch
from .ems_wind import ems_wind_abc, ems_wind_long_term_operation, ems_cplex_wind, ems_cplex_parts_wind
from .ems_solar import ems_solar_abc, ems_solar_long_term_operation, ems_cplex_solar, ems_cplex_parts_solar
from .ems_bess import ems_bess_abc, ems_bess_long_term_operation, ems_cplex_bess, ems_cplex_parts_bess
from .ems_base import ems_base_abc, ems_base_long_term_operation, ems_cplex_base, ems_cplex_parts_base_pre,\
                      ems_cplex_parts_base_post, ems_degredation_base

def insert_ems_section(section):
    code = inspect.getsource(section)
    # Split the code into lines
    lines = code.splitlines()

    # Remove the function definition line (the first line) and remove leading indentation from the remaining lines
    indented_code = "\n".join(line[4:] if line.startswith("    ") else line for line in lines[1:]) +"\n"

    # Save the combined string to a Python (.py) file
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the new file in the same directory as the script
    file_path = os.path.join(script_dir, "builded_ems.py")

    # Write to the file
    with open(file_path, "a") as file:
        file.write(indented_code)

def insert_ems_long_term_operation_section(section):
    code = inspect.getsource(section)
    # Split the code into lines
    lines = code.splitlines()

    # Remove the function definition line (the first line) and remove leading indentation from the remaining lines
    indented_code = "\n".join(line[4:] if line.startswith("    ") else line for line in lines[1:]) +"\n"

    # Save the combined string to a Python (.py) file
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the new file in the same directory as the script
    file_path = os.path.join(script_dir, "builded_ems_long_term_operation.py")

    # Write to the file
    with open(file_path, "a") as file:
        file.write(indented_code)

def append_ems_section(code, section):
    section_code = inspect.getsource(section)
    # Split the code into lines
    lines = section_code.splitlines()

    # Remove the function definition line (the first line) and remove leading indentation from the remaining lines
    indented_code = "\n".join(line[4:] if line.startswith("    ") else line for line in lines[1:]) +"\n"

    code = code + indented_code

    return code

def get_path_builded_ems(short_long):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Create the full path for the new file in the same directory as the script
    if short_long == 'short':
        file_path = os.path.join(script_dir, "builded_ems.py")
    elif short_long == 'long':
        file_path = os.path.join(script_dir, "builded_ems_long_term_operation.py")
    return file_path


def ems_cplex(inputs):
    ems_cplex_map = {
        'base': ems_cplex_base,
        'wind': ems_cplex_wind,
        'solar': ems_cplex_solar,
        'bess': ems_cplex_bess
    }

    technology_types = inputs['technology_types']
    technology_types_and_base = ['base'] + technology_types

    # Create a dictionary to store the instances
    ems_cplex_instances = {}

    # Loop through the technology types and create instances
    for tech_type in technology_types_and_base:
        if tech_type in ems_cplex_map:
            ems_cplex_instances[tech_type] = ems_cplex_map[tech_type]()

    # split in batches, ussually a week
    wind_ts = inputs['wind_ts']
    batch_size_default = 2*24 # should depend on technology_types
    batch_size = inputs.get('batch_size', batch_size_default)
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]

    # allocate vars
    length_ts = len(wind_ts)
    allocated_vars = {}

    for tech_type in technology_types_and_base:
        allocated_vars = ems_cplex_instances[tech_type].allocate_vars(allocated_vars, length_ts)

    for ib, batch in enumerate(batches):
        ems_cplex_parts_inputs = {}
        for tech_type in technology_types_and_base:
            ems_cplex_parts_inputs = ems_cplex_instances[tech_type].inputs_to_ems_cplex_parts(inputs, batch, ems_cplex_parts_inputs)

        #print(f'batch {ib+1} out of {len(batches)}')
        outputs_ems_cplex_parts = ems_cplex_parts(ems_cplex_parts_inputs)

        for tech_type in technology_types_and_base:
            allocated_vars = ems_cplex_instances[tech_type].outputs_batch(outputs_ems_cplex_parts, batch, allocated_vars)

    outputs_ems_cplex = {}
    for tech_type in technology_types_and_base:
        outputs_ems_cplex = ems_cplex_instances[tech_type].ems_cplex_outputs(outputs_ems_cplex, allocated_vars)

    return outputs_ems_cplex

def ems_cplex_parts(inputs):
    """EMS solver implemented in cplex

    Parameters
    ----------
    wind_ts : WPP power time series
    solar_ts : PVP power time series
    price_ts : price time series
    b_P : battery power
    b_E : battery energy capacity time series
    G_MW : grid connection
    battery_depth_of_discharge : battery depth of discharge
    battery_charge_efficiency : battery charge efficiency
    peak_hr_quantile : quantile of price time series to define peak price hours
    cost_of_battery_P_fluct_in_peak_price_ratio : cost of battery power fluctuations computed as a peak price ratio
    n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached

    Returns
    -------
    P_HPP_ts: HPP power time series
    P_curtailment_ts: HPP curtailed power time series
    P_charge_discharge_ts: Battery charge/discharge power time series
    E_SOC_ts: Battery energy SOC time series
    penalty_ts: penalty time series for not reaching expected energy production at peak hours
    """
    ###############################################
    # SETUP
    mdl = Model(name='EMS')
    mdl.context.cplex_parameters.threads = 1

    # CPLEX parameter pg 87 Emphasize feasibility over optimality
    mdl.context.cplex_parameters.emphasis.mip = 1

    price_ts=inputs.get('price_ts')
    G_MW=inputs.get('G_MW')
    peak_hr_quantile = inputs.get('peak_hr_quantile', 0.9)
    n_full_power_hours_expected_per_day_at_peak_price = inputs.get('n_full_power_hours_expected_per_day_at_peak_price', 3)
    technology_types = inputs.get('technology_types', ['wind', 'solar', 'bess'])

    N_t = len(price_ts.index)
    N_days = N_t / 24
    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price * G_MW
    e_peak_period_expected = e_peak_day_expected * N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts >= price_peak)[0]
    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max < 0] = 0
    price_ts_to_max.iloc[:-1] = 0.5 * price_ts_to_max.iloc[:-1].values + 0.5 * price_ts_to_max.iloc[1:].values
    time = price_ts.index
    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))
    extra_inputs=dict(
        N_t=N_t,
        N_days=N_days,
        e_peak_day_expected=e_peak_day_expected,
        e_peak_period_expected=e_peak_period_expected,
        price_peak=price_peak,
        peak_hours_index=peak_hours_index,
        price_ts_to_max=price_ts_to_max,
        time=time,
        SOCtime=SOCtime
    )

    # base model
    ems_cplex_parts_map ={
        'base_pre': ems_cplex_parts_base_pre,
        'wind': ems_cplex_parts_wind,
        'solar': ems_cplex_parts_solar,
        'bess': ems_cplex_parts_bess,
        'base_post': ems_cplex_parts_base_post
    }

    technology_types_and_base = ['base_pre'] + technology_types + ['base_post']
    ems_cplex_parts_instances = {}
    for tech_type in technology_types_and_base :
        ems_cplex_parts_instances[tech_type] = ems_cplex_parts_map[tech_type](mdl, inputs, extra_inputs)
        mdl = ems_cplex_parts_instances[tech_type].get_updated_model()

    # # print mdl details
    # print('dict_hydesign_equations =', mdl.dict_hydesign_equations)
    # print('decision variables =', mdl.dict_hydesign_equations['decision variables'])
    # print('objective function =', mdl.dict_hydesign_equations['objective function'])
    # print('constraints =', mdl.dict_hydesign_equations['constraints'])


    # Solving the problem
    sol = mdl.solve(
        log_output=False)
        #log_output=True)

    outputs = {}
    for tech_type in ['base_pre'] + technology_types + ['base_post']:
        outputs = ems_cplex_parts_instances[tech_type].outputs_cplex_parts(sol, outputs)

    mdl.end()

    return outputs

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