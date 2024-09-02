# %%
import inspect
import trace
import sys
import os
import re
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
from .ems_base import ems_base_abc, ems_base_long_term_operation, ems_cplex_base, ems_cplex_parts_base_pre,\
                      ems_cplex_parts_base_post, ems_degredation_base
from .ems_utils import expand_to_lifetime, split_in_batch, print_cplex_model_details
from .ems_builder_utils import insert_ems_section, insert_ems_long_term_operation_section, get_path_builded_ems
from .ems_builder_utils import ems_cplex, ems_cplex_parts, operation_solar_batt_deg
from .ems_sections import imports, init_ems, init_ems_long, setup_ems, setup_ems_long, compute_ems, compute_ems_long, section10

def ems_builder(self):
    # Get the appropriate EMS function name based on ems_type
    ems_type = self.available_ems_types.get(self.ems_type)

    # Raise a warning if the ems_type is not found in the dictionary
    if ems_type is None:
        raise Warning("Selected ems type is not supported")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "ems_builder_outputs.py")

    code_builder = """# Built by ems_builder
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
    """
    code_builder += """
class ems(om.ExplicitComponent):
    def __init__(self, inputs):
        super().__init__()
        self.weeks_per_season_per_year = inputs.get('weeks_per_season_per_year')
        self.N_time = int(inputs.get('N_time'))
        self.technology_types = inputs.get('technology_types', ['wind', 'solar', 'bess'])
        self.life_h = int(inputs.get('life_h', 25*365*24))
        self.technology_types_and_base = ['base'] + self.technology_types  # set default order of the technologies

    def setup(self):"""



    code_builder += "\n        # add inputs to the openmdao model"
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_class_map[tech_type]() #getattr(self, f"ems_{tech_type}_abc_instance")
        #class_name.add_inputs(self)
        code_class = class_name.get_code_addinputs()
        code_builder += f"\n        # inputs added for {tech_type}"
        code_builder += "\n" + code_class + "\n"

    code_builder += "\n        # add outputs to the openmdao model"
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_class_map[tech_type]() #getattr(self, f"ems_{tech_type}_abc_instance")
        #class_name.add_outputs(self)
        code_class = class_name.get_code_addoutputs()
        code_builder += f"\n        # outputs added for {tech_type}"
        code_builder += "\n" + code_class + "\n"

    code_builder +="""
    def compute(self, inputs, outputs):
        # Build a sintetic time to avoid problems with time sereis
        # indexing in ems
        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start='01-01-1991 00:00',
                periods=len(inputs['wind_t']),
                freq='1h'))

        # ems arguments"""

    code_added_ems_args =""
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_class_map[tech_type]() # getattr(self, f"ems_{tech_type}_abc_instance")
        #ems_args = class_name.add_ems_args(ems_args, WSPr_df, inputs)

        WSPr_df, added_ems_args = class_name.add_ems_args()
        code_builder += "\n        " + WSPr_df
        code_added_ems_args += "\n" + added_ems_args

    code_builder += "\n        ems_args = {"
    code_builder += f"{code_added_ems_args}\n"
    code_builder += "         }\n"

    code_builder += f"\n        outputs_ems = {ems_type}(ems_args) \n\n        # post processing outputs\n"
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_class_map[tech_type]() #  getattr(self, f"ems_{tech_type}_abc_instance")
        # extention of the outputs to the project lifetime
        #outputs = class_name.postprocessing_outputs(component=self, inputs=inputs, outputs=outputs, outputs_ems=outputs_ems)

        code_class = class_name.get_code_postprocessing_outputs()
        code_builder += f"        # post processing outputs for {tech_type}"
        code_builder += "\n" + code_class + "\n"

    ######### final code
    with open(file_path, "w") as file:
        file.write(code_builder)
        file.flush()  # Ensure all data is written to disk
        os.fsync(file.fileno())  # Ensure the file is physically written to disk

def ems_long_term_operation_builder(self):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "ems_long_term_operation_builder_outputs.py")

    code_builder = """# Built by ems_builder
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
"""

    code_builder += """
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
    """
    code_builder += "\n        # add inputs to the cplex model"
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_long_class_map[tech_type]()  # getattr(self, f"ems_{tech_type}_long_term_operation_instance")
        # class_name.add_inputs(self)
        code_class = class_name.get_code_addinputs()
        code_builder += f"\n        # inputs added for {tech_type}"
        code_builder += "\n" + code_class + "\n"

    code_builder = code_builder + "\n        # add outputs to the cplex model"
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_long_class_map[tech_type]()  # getattr(self, f"ems_{tech_type}_long_term_operation_instance")
        # class_name.add_outputs(self)
        code_class = class_name.get_code_addoutputs()
        code_builder += f"\n        # outputs added for {tech_type}"
        code_builder += "\n" + code_class + "\n"

    code_builder += """
    def compute(self, inputs, outputs):"""
    if self.ems_long_type == 'energy_penalty':
        ems_longterm = 'operation_solar_batt_deg'
    else:
        raise Warning("This class should only be used for energy_penalty")
        """# ems_long arguments"""

    code_added_ems_args =""
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_long_class_map[tech_type]() # getattr(self, f"ems_{tech_type}_abc_instance")
        #ems_args = class_name.add_ems_args(ems_args, WSPr_df, inputs)

        WSPr_df, added_ems_args = class_name.add_ems_args()
        if WSPr_df:
            code_builder += "\n        " + WSPr_df
        code_added_ems_args += added_ems_args

    code_builder += "\n        ems_args = {"
    code_builder += f"\n {code_added_ems_args}\n"
    code_builder += "         }\n"


    code_builder += f"""
        outputs_ems = {ems_longterm}(ems_args) \n
        # post processing outputs \n"""
    for tech_type in self.technology_types_and_base:
        class_name = self.ems_long_class_map[tech_type]()  # getattr(self, f"ems_{tech_type}_long_term_operation_instance")
        # outputs = class_name.postprocessing_outputs(component=self, inputs=inputs, outputs=outputs, outputs_ems=outputs_ems_longterm)

        code_class = class_name.get_code_postprocessing_outputs()
        code_builder += f"        # post processing outputs for {tech_type}\n"
        code_builder += code_class + "\n"


    ######### final code
    with open(file_path, "w") as file:
        file.write(code_builder)
        file.flush()  # Ensure all data is written to disk
        os.fsync(file.fileno())  # Ensure the file is physically written to disk


import os
import inspect
import re


def ems_cplex_builder(self):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "ems_cplex_builder_output.py")

    code_builder = """# Built by ems_builder
def ems_cplex(inputs):
    # Variable allocation"""
    # Create a dictionary to store the instances
    ems_cplex_instances = {}

    # Loop through the technology types and create instances
    allocated_vars = {}
    output_string = {}
    modified_output_string = {}

    allocated_vars_all = '{\n'

    for tech_type in self.technology_types_and_base:
        if tech_type in self.ems_cplex_map:
            ems_cplex_instances[tech_type] = self.ems_cplex_map[tech_type]()
            # allocated_vars_instance = ems_cplex_instances[tech_type].allocate_vars2(100)
            # allocated_vars.update(allocated_vars_instance)
            #code = inspect.getsource(ems_cplex_instances[tech_type].allocate_vars2)

            code_builder += f"\n    # allocated variables for {tech_type}: \n"
            code_allocated_vars = ems_cplex_instances[tech_type].get_function_code(ems_cplex_instances[tech_type].allocate_vars2, indent_adjustment=4)

            if 'dict' in code_allocated_vars or '{' in code_allocated_vars:

                new_variable_name = f"allocated_vars_{tech_type}"
                allocated_vars_all += f"                    **{new_variable_name}, \n"
                code_allocated_vars_renamed = re.sub(r"^\s*\w+\s*=", f"    {new_variable_name} =", code_allocated_vars, count=1)
                code_builder += code_allocated_vars_renamed + "\n"

    allocated_vars_all += "    }\n"
    code_builder += "\n    # allocated variables for all technologies: \n"
    code_builder += f"    allocated_vars_all = {allocated_vars_all}"

    code_builder += """
    # split in batches, ussually a week
    wind_ts = inputs['wind_ts']
    batch_size_default = 2*24 # should depend on technology_types
    batch_size = inputs.get('batch_size', batch_size_default)
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]
    
    for ib, batch in enumerate(batches):
    """

    code_builder += "\n        # inputs_to_ems_cplex_parts"
    inputs_to_ems_cplex_parts_all = '{\n'
    for tech_type in self.technology_types_and_base:
        if tech_type in self.ems_cplex_map:
            ems_cplex_instances[tech_type] = self.ems_cplex_map[tech_type]()

            code_builder += f"\n        # inputs_to_ems_cplex_parts for {tech_type}: \n"
            code_inputs_to_ems_cplex_parts = ems_cplex_instances[tech_type].get_function_code(ems_cplex_instances[tech_type].inputs_to_ems_cplex_parts2, indent_adjustment=8)

            if 'dict' in code_inputs_to_ems_cplex_parts or '{' in code_inputs_to_ems_cplex_parts:

                new_variable_name = f"inputs_to_ems_cplex_parts_{tech_type}"
                inputs_to_ems_cplex_parts_all += f"                    **{new_variable_name}, \n"
                code_inputs_to_ems_cplex_parts_renamed = re.sub(r"^\s*\w+\s*=", f"    {new_variable_name} =", code_inputs_to_ems_cplex_parts, count=1)
                code_builder += "    " + code_inputs_to_ems_cplex_parts_renamed + "\n"

    inputs_to_ems_cplex_parts_all += "        }\n"
    code_builder += "\n        # inputs_to_ems_cplex_parts for all technologies: \n"
    code_builder += f"        inputs_to_ems_cplex_parts_all = {inputs_to_ems_cplex_parts_all}\n"

    code_builder += """        # run ems cplex parts
        outputs_ems_cplex_parts = ems_cplex_parts(inputs_to_ems_cplex_parts_all)\n"""

    code_builder += f"\n        # outputs_batch"
    for tech_type in self.technology_types_and_base:
        if tech_type in self.ems_cplex_map:
            ems_cplex_instances[tech_type] = self.ems_cplex_map[tech_type]()

            code_builder += f"\n        # outputs_batch for {tech_type}: \n"
            code_outputs_batch = "    " + ems_cplex_instances[tech_type].get_function_code(ems_cplex_instances[tech_type].outputs_batch, indent_adjustment=8)
            code_builder += "    " + code_outputs_batch +"\n"

    code_builder += f"\n    # outputs of ems cplex"
    ems_cplex_outputs_all = '{\n'
    for tech_type in self.technology_types_and_base:
        if tech_type in self.ems_cplex_map:
            ems_cplex_instances[tech_type] = self.ems_cplex_map[tech_type]()

            code_builder += f"\n    # ems_cplex_outputs for {tech_type}: \n"
            code_ems_cplex_outputs = ems_cplex_instances[tech_type].get_function_code(ems_cplex_instances[tech_type].ems_cplex_outputs2, indent_adjustment=8)

            if 'dict' in code_ems_cplex_outputs or '{' in code_ems_cplex_outputs:

                new_variable_name = f"ems_cplex_outputs_{tech_type}"
                ems_cplex_outputs_all += f"                    **{new_variable_name}, \n"
                code_ems_cplex_outputs_renamed = re.sub(r"^\s*\w+\s*=", f"    {new_variable_name} =", code_ems_cplex_outputs, count=1)
                code_builder += code_ems_cplex_outputs_renamed + "\n"

    ems_cplex_outputs_all += "    }\n"
    code_builder += "\n    # ems_cplex_outputs for all technologies: \n"
    code_builder += f"    ems_cplex_outputs_all = {ems_cplex_outputs_all}"

    code_builder += "\n    return ems_cplex_outputs_all"

    # Save the generated code to the output file
    with open(file_path, 'w') as file:
        file.write(code_builder)


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

        self.class_map = {
            'base': ems_base_long_term_operation,
            'wind': ems_wind_long_term_operation,
            'solar': ems_solar_long_term_operation,
            'bess': ems_bess_long_term_operation
        }

        # Start ems_build
        file_path = get_path_builded_ems('long')

        # Write to the file
        with open(file_path, "w") as file:
            file.write("# Built by ems_builder\n")

        if self.build_ems:
            insert_ems_long_term_operation_section(imports)
            insert_ems_long_term_operation_section(init_ems_long)

        # Reload the module
        importlib.reload(builded_ems)
        builded_ems_long_term_operation.init(self)

    def setup(self):
        if self.build_ems:
            code = insert_ems_long_term_operation_section(setup_ems_long)
            code_setup = "\n    # add inputs to the cplex model"
            for tech_type in self.technology_types_and_base:
                class_name = self.class_map[tech_type]() # getattr(self, f"ems_{tech_type}_long_term_operation_instance")
                #class_name.add_inputs(self)
                code_class = class_name.get_code_addinputs()
                code_setup = code_setup + f"\n    # inputs added for {tech_type}"
                code_setup = code_setup + "\n" + code_class + "\n"

            code_setup = code_setup + "\n    # add outputs to the cplex model"
            for tech_type in self.technology_types_and_base:
                class_name = self.class_map[tech_type]() #  getattr(self, f"ems_{tech_type}_long_term_operation_instance")
                #class_name.add_outputs(self)
                code_class = class_name.get_code_addoutputs()
                code_setup = code_setup + f"\n    # outputs added for {tech_type}"
                code_setup = code_setup + "\n" + code_class + "\n"

            # Save the combined string to a Python (.py) file
            file_path = get_path_builded_ems('long')
            with open(file_path, "a") as file:
                file.write(code_setup)

        # Reload the module
        importlib.reload(builded_ems_long_term_operation)
        builded_ems_long_term_operation.setup(self)

        insert_ems_long_term_operation_section(section10)

    def compute(self, inputs, outputs):
        if self.build_ems:
            code_setup = "\n    # ems_long arguments"
            for tech_type in self.technology_types_and_base:
                class_name = self.class_map[tech_type]() #  getattr(self, f"ems_{tech_type}_long_term_operation_instance")
                #ems_args = class_name.add_ems_args(ems_args, inputs)
                code_class = class_name.get_code_add_ems_args()
                code_setup = code_setup + f"\n    # arguments added for {tech_type}"
                code_setup = code_setup + "\n" + code_class + "\n"

            # Save the combined string to a Python (.py) file
            insert_ems_long_term_operation_section(compute_ems_long)
            file_path = get_path_builded_ems('long')
            with open(file_path, "a") as file:
                file.write(code_setup)

            code_setup = "\n    outputs_ems = ems_longterm(ems_args)\n    # post processing outputs"
            for tech_type in self.technology_types_and_base:
                class_name = self.class_map[tech_type]() #  getattr(self, f"ems_{tech_type}_long_term_operation_instance")
                #outputs = class_name.postprocessing_outputs(component=self, inputs=inputs, outputs=outputs, outputs_ems=outputs_ems_longterm)

                code_class = class_name.get_code_postprocessing_outputs()
                code_setup = code_setup + f"\n    # post processing outputs for {tech_type}"
                code_setup = code_setup + "\n" + code_class + "\n"

            # Save the combined string to a Python (.py) file
            with open(file_path, "a") as file:
                file.write(code_setup)
        # Reload the module
        importlib.reload(builded_ems_long_term_operation)
        builded_ems_long_term_operation.compute(self, inputs, outputs)


# ------------
# ems_cplex
# ------------



# def operation_solar_batt_deg(inputs):
#
#     """EMS operation for degraded PV and battery based on an existing EMS.
#
#     Parameters
#     ----------
#     wind_t_deg: Wind time series including degradation
#     solar_t_deg: PV time series including degradation
#     batt_degradation: Battery degradation as health factor [0=dead,1=new]
#     wind_t: WPP power time series
#     solar_t: PVP power time series
#     hpp_curt_t: HPP curtailment time series results form an EMS planed without degradation
#     b_t: HPP battery power (charge/discharge) time series results form an EMS planed without degradation
#     b_E_SOC_t: HPP battery state of charge (SoC) time series results form an EMS planed without degradation
#     b_E_SOC_0: Initial charge status of the actual operation
#     price_ts : price time series
#     G_MW : grid connection
#     b_E : battery energy capacity time series
#     battery_depth_of_discharge : battery depth of discharge
#     battery_charge_efficiency : battery charge efficiency
#     peak_hr_quantile : quantile of price time series to define peak price hours
#     cost_of_battery_P_fluct_in_peak_price_ratio : cost of battery power fluctuations computed as a peak price ratio
#     n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached
#
#     Returns
#     -------
#     Hpp_deg : HPP power time series
#     P_curt_deg : HPP curtailed power time series
#     b_t_sat : Battery charge/discharge power time series
#     b_E_SOC_t_sat : Battery energy SOC time series
#     penalty_ts : penalty for not reaching expected energy production at peak hours
#     """
#     wind_t_deg=inputs.get('wind_t_deg')
#     solar_t_deg=inputs.get('solar_t_deg')
#     batt_degradation=inputs.get('batt_degradation')
#     wind_t=inputs.get('wind_t')
#     solar_t=inputs.get('solar_t')
#     hpp_curt_t=inputs.get('hpp_curt_t')
#     b_t=inputs.get('b_t')
#     b_E_SOC_t=inputs.get('b_E_SOC_t')
#     G_MW=inputs.get('G_MW')
#     b_E=inputs.get('b_E')
#     battery_depth_of_discharge=inputs.get('battery_depth_of_discharge')
#     battery_charge_efficiency=inputs.get('battery_charge_efficiency')
#     price_ts=inputs.get('price_ts')
#     b_E_SOC_0 = inputs.get('b_E_SOC_0', None)
#     peak_hr_quantile = inputs.get('peak_hr_quantile', 0.9)
#     n_full_power_hours_expected_per_day_at_peak_price = inputs.get('n_full_power_hours_expected_per_day_at_peak_price', 3)
#
#     B_p = np.max(np.abs(b_t))
#
#     # wind_solar_t = solar_t + wind_t
#     wind_solar_t_deg = solar_t_deg + wind_t_deg
#     # P_deg_t_sat_loss = (solar_t - solar_t_deg) + (wind_t - wind_t_deg)
#
#     # P_loss =  np.maximum( 0 , P_deg_t_sat_loss  - hpp_curt_t)
#     b_t_less_sol = b_t.copy()
#     dt = 1
#
#     # Reduction in power to battery due to reduction of solar
#     for i in range(len(b_t)):
#         if b_t[i] < 0:
#             # Try to keep the ratio of b_t[i] / wind_solar_t[i]  SoC to the maximum
#             # b_t_less_sol[i] = ( b_t[i] / wind_solar_t[i] ) * ( wind_solar_t_deg[i] )
#
#             # Hajar's method
#             #b_t_less_sol[i] = np.minimum(b_t[i] + P_loss[i],0)
#
#             # Try to follow SoC to the maximum
#             if -b_t[i] > wind_solar_t_deg[i]:
#                 b_t_less_sol[i] = -wind_solar_t_deg[i]
#
#             b_t_less_sol[i] = np.clip(
#                 b_t_less_sol[i],
#                 -B_p,
#                 B_p)
#
#     # Initialize the SoC
#     b_E_SOC_t_sat = b_E_SOC_t.copy()
#     if b_E_SOC_0 == None:
#         try:
#             b_E_SOC_t_sat[0]= b_E_SOC_t[0]
#         except:
#             raise('len(b_E_SOC_t):', len(b_E_SOC_t))
#     else:
#         b_E_SOC_t_sat[0]= b_E_SOC_0
#
#     # Update the SoC
#     for i in range(len(b_t_less_sol)):
#         if b_t_less_sol[i] < 0: # charging
#             b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt * battery_charge_efficiency
#         if b_t_less_sol[i] >= 0: # discharging
#             b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt / battery_charge_efficiency
#
#         b_E_SOC_t_sat[i+1] = np.clip(
#             b_E_SOC_t_sat[i+1],
#             (1-battery_depth_of_discharge) * b_E * batt_degradation[i],
#             b_E * batt_degradation[i])
#
#     # Recompute the battery power
#     b_t_sat = b_t.copy()
#     for i in range(len(b_t_sat)):
#         if b_t[i] < 0:
#             b_t_sat[i] = ( ( b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] ) / battery_charge_efficiency ) / dt
#         elif b_t[i] >= 0:
#             b_t_sat[i] = ( (b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] )  * battery_charge_efficiency ) / dt
#
#     Hpp_deg = np.minimum( wind_t_deg + solar_t_deg + b_t_sat, G_MW)
#     P_curt_deg = np.maximum( wind_t_deg + solar_t_deg + b_t_sat - G_MW, 0)
#
#     # Compute penalty per week
#     H_df = pd.DataFrame(
#         np.copy(Hpp_deg),
#         columns=['hpp_t_with_deg'],
#         index=pd.date_range(
#         start='01-01-1991 00:00',
#         periods=len(Hpp_deg),
#         freq='1h')
#     )
#     H_df['price_t_ext'] = price_ts
#     price_peak = np.quantile(H_df['price_t_ext'],peak_hr_quantile)
#     H_df['peak'] = H_df['price_t_ext']>=price_peak
#     H_df.loc[~H_df['peak'],'hpp_t_with_deg']=0
#
#     N_days = 7
#     H_df_week = H_df.resample(f'{N_days}d').sum()
#
#     e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*G_MW
#     e_peak_period_expected = e_peak_day_expected*N_days
#
#     H_df_week['e_peak_period_expected'] = e_peak_period_expected
#     H_df_week['e_penalty'] = H_df_week['e_peak_period_expected'] - H_df_week['hpp_t_with_deg']
#     H_df_week['penalty'] = price_peak*np.maximum(0, H_df_week['e_penalty'])
#
#     penalty_df = H_df_week['penalty'].reindex_like(H_df).fillna(0.0)
#     penalty_ts = penalty_df.values.flatten()
#
#     outputs ={'Hpp_deg': Hpp_deg,
#               'P_curt_deg': P_curt_deg,
#               'b_t_sat': b_t_sat,
#               'b_E_SOC_t_sat': b_E_SOC_t_sat,
#               'penalty_ts': penalty_ts}
#
#     return outputs
#

#######################################
#######################################

# def operation_rule_base_no_penalty(
#     wind_t_deg,
#     solar_t_deg,
#     batt_degradation,
#     wind_t,
#     solar_t,
#     hpp_curt_t,
#     b_t,
#     b_E_SOC_t,
#     G_MW,
#     b_E,
#     battery_depth_of_discharge,
#     battery_charge_efficiency,
#     b_E_SOC_0 = None,
#     load_min = 3,
#     load_min_penalty_factor = 1e6,
#     change_BES_charging = 'only_for_less_power',
#     ):
#
#     """EMS operation for degraded PV and battery based on an existing EMS.
#
#     Parameters
#     ----------
#     wind_t_deg: Wind time series including degradation
#     solar_t_deg: PV time series including degradation
#     batt_degradation: Battery degradation as health factor [0=dead,1=new]
#     b_t: HPP battery power (charge/discharge) time series results form an EMS planed without degradation
#     b_E_SOC_t: HPP battery state of charge (SoC) time series results form an EMS planed without degradation
#     G_MW : grid connection
#     b_E : battery energy capacity time series
#     battery_depth_of_discharge : battery depth of discharge
#     battery_charge_efficiency : battery charge efficiency
#     b_E_SOC_0: Initial charge status of the actual operation
#     load_min: minimum electrical load to meet [MW]
#     load_min_penalty_factor: penalty factor to scale the penalty when not meeting required load
#
#     Returns
#     -------
#     Hpp_deg : HPP power time series
#     P_curt_deg : HPP curtailed power time series
#     b_t_sat : Battery charge/discharge power time series
#     b_E_SOC_t_sat : Battery energy SOC time series
#     penalty_ts : penalty for not reaching minimum electrical load constraint
#     """
#
#     B_p = np.max(np.abs(b_t))
#
#     wind_solar_t = solar_t + wind_t
#     wind_solar_t_deg = solar_t_deg + wind_t_deg
#
#     P_deg_t_sat_loss = (solar_t - solar_t_deg) + (wind_t - wind_t_deg)
#     P_loss =  np.maximum( 0 , P_deg_t_sat_loss  - hpp_curt_t)
#
#     b_t_less_sol = b_t.copy()
#     dt = 1
#
#     # Reduction in power to battery due to reduction of solar
#     for i in range(len(b_t)):
#         if b_t[i] < 0:
#             if change_BES_charging == 'proportional':
#                 if wind_solar_t[i] != 0:
#                     # Try to keep the ratio of b_t[i] / wind_solar_t[i]  SoC to the maximum
#                     b_t_less_sol[i] = ( b_t[i] / wind_solar_t[i] ) * ( wind_solar_t_deg[i] )
#
#             elif change_BES_charging == 'only_for_less_power':
#                 if -b_t[i] > P_loss[i]:
#                     b_t_less_sol[i] = b_t_less_sol[i] + P_loss[i]
#
#             elif change_BES_charging == 'always':
#                 # Try to follow SoC to the maximum
#                 if -b_t[i] > wind_solar_t_deg[i]:
#                     b_t_less_sol[i] = -wind_solar_t_deg[i]
#
#             b_t_less_sol[i] = np.clip(
#                 b_t_less_sol[i],
#                 -B_p,
#                 B_p)
#
#     # Initialize the SoC
#     b_E_SOC_t_sat = b_E_SOC_t.copy()
#     if b_E_SOC_0 == None:
#         try:
#             b_E_SOC_t_sat[0]= b_E_SOC_t[0]
#         except:
#             raise('len(b_E_SOC_t):', len(b_E_SOC_t))
#     else:
#         b_E_SOC_t_sat[0]= b_E_SOC_0
#
#     # Update the SoC
#     for i in range(len(b_t_less_sol)):
#         if b_t_less_sol[i] < 0: # charging
#             b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt * battery_charge_efficiency
#         if b_t_less_sol[i] >= 0: # discharging
#             b_E_SOC_t_sat[i+1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt / battery_charge_efficiency
#
#         b_E_SOC_t_sat[i+1] = np.clip(
#             b_E_SOC_t_sat[i+1],
#             (1-battery_depth_of_discharge) * b_E * batt_degradation[i],
#             b_E * batt_degradation[i]  )
#
#     # Recompute the battery power
#     b_t_sat = b_t.copy()
#     for i in range(len(b_t_sat)):
#         if b_t[i] < 0:
#             b_t_sat[i] = ( ( b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] ) / battery_charge_efficiency ) / dt
#         elif b_t[i] >= 0:
#             b_t_sat[i] = ( (b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i+1] )  * battery_charge_efficiency ) / dt
#
#     Hpp_deg = np.minimum( wind_t_deg + solar_t_deg + b_t_sat, G_MW)
#     P_curt_deg = np.maximum( wind_t_deg + solar_t_deg + b_t_sat - G_MW, 0)
#
#
#     return Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat
