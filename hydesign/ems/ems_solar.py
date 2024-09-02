import pandas as pd
import numpy as np
from .ems_abc import ems_abc, ems_cplex_abc, ems_cplex_parts_abc
from .ems_utils import expand_to_lifetime

class ems_solar_abc(ems_abc):

    def add_inputs(self, component):
        component.add_input(
            'solar_t',
            desc="PVP power time series",
            units='MW',
            shape=[component.N_time])

    def add_outputs(self, component):
        component.add_output(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[component.life_h])

    def add_ems_args(self):
        WSPr_df = """WSPr_df['solar_t'] = inputs['solar_t']"""
        added_ems_args = """
            # args solar
            'solar_ts': WSPr_df.solar_t,"""
        return WSPr_df, added_ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        solar_t = inputs['solar_t']
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['solar_t_ext'] = expand_to_lifetime(
            solar_t, life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)

        return outputs

class ems_solar_long_term_operation(ems_abc):

    def add_inputs(self, component):
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

    def add_outputs(self, component):
        pass

    def add_ems_args(self):
        WSPr_df = ''
        added_ems_args = """
            # args solar
            'solar_t_deg': inputs['solar_t_ext_deg'],
            'solar_t': inputs['solar_t_ext'],
            """
        return WSPr_df, added_ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        return outputs

class ems_cplex_solar(ems_cplex_abc):
    def allocate_vars(self, allocated_vars, length_ts):
        return allocated_vars

    def allocate_vars2(self, length_ts):
        pass

    def inputs_to_ems_cplex_parts(self, inputs, batch, outputs):
        added_outputs=dict(solar_ts=inputs['solar_ts'].iloc[batch])
        outputs.update(added_outputs)
        return outputs

    def inputs_to_ems_cplex_parts2(self, inputs, batch, outputs):
        added_outputs=dict(solar_ts=inputs['solar_ts'].iloc[batch])

    def outputs_batch(self, outputs_ems_cplex_parts, batch, allocated_vars):
        return allocated_vars

    def ems_cplex_outputs(self, outputs, allocated_vars):
        return outputs

    def ems_cplex_outputs2(self):
        return None

class ems_cplex_parts_solar(ems_cplex_parts_abc):
    def __init__(self, mdl, inputs, extra_inputs={}):
        self.mdl = mdl
        self.inputs = inputs
        self.extra_inputs = extra_inputs
        self.solar_ts = inputs['solar_ts']
        self.time = extra_inputs['time']

    def _variables(self):
        pass

    def _constraints(self):
        # Iterate over each time step
        for t in self.time:
            tt = t + pd.Timedelta('1hour')

            # Update specific constraints
            constraint = self.mdl.get_constraint_by_name(f'power_balance_{t}')
            self.mdl.remove_constraint(constraint)
            self.mdl.add_constraint(constraint.lhs == constraint.rhs + self.solar_ts[t], f'power_balance_{t}')

    def _objective_function(self):
        pass

    def _dict_hydesign_equations(self):
        self.mdl.dict_hydesign_equations['constraints']['power balance'] = self.mdl.dict_hydesign_equations['constraints'].get('power balance', '') +\
                                                                           '+solar_ts[t]'

    def outputs_cplex_parts(self, sol, outputs):
        return outputs

