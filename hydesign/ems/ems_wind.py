import pandas as pd
import numpy as np
from .ems_abc import ems_abc, ems_cplex_abc, ems_cplex_parts_abc
from .ems_utils import expand_to_lifetime

class ems_wind_abc(ems_abc):

    def add_inputs(self, component):
        component.add_input(
            'wind_t',
            desc="WPP power time series",
            units='MW',
            shape=[component.N_time])

    def add_outputs(self, component):
        component.add_output(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[component.life_h])

    def add_ems_args(self):
        added_ems_args = """'wind_ts': inputs['wind_t'],"""
        return added_ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['wind_t_ext'] = expand_to_lifetime(
            inputs['wind_t'], life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)

        return outputs

class ems_wind_long_term_operation(ems_abc):

    def add_inputs(self, component):
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

    def add_outputs(self, component):
        pass

    def add_ems_args(self):
        added_ems_args ="""
wind_t_deg=inputs['wind_t_ext_deg'],
wind_t=inputs['wind_t_ext'],"""
        return added_ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        return outputs

class ems_cplex_wind(ems_cplex_abc):
    def allocate_vars(self, allocated_vars, length_ts):
        return allocated_vars

    def inputs_to_ems_cplex_parts(self, inputs, batch, outputs):
        added_outputs=dict(wind_ts=inputs['wind_ts'].iloc[batch])
        outputs.update(added_outputs)
        return outputs

    def outputs_batch(self, outputs_ems_cplex_parts, batch, allocated_vars):
        return allocated_vars

    def ems_cplex_outputs(self, outputs, allocated_vars):
        return outputs

class ems_cplex_parts_wind(ems_cplex_parts_abc):
    def __init__(self, mdl, inputs, extra_inputs={}):
        self.mdl = mdl
        self.inputs = inputs
        self.extra_inputs = extra_inputs
        self.wind_ts = inputs['wind_ts']
        self.time = extra_inputs['time']

    def _variables(self):
        pass

    def _constraints(self):
        for t in self.time:
            tt = t + pd.Timedelta('1hour')
            dt = 1

            # Update specific constraints
            constraint = self.mdl.get_constraint_by_name(f'power_balance_{t}')
            self.mdl.remove_constraint(constraint)
            self.mdl.add_constraint(constraint.lhs == constraint.rhs + self.wind_ts[t], f'power_balance_{t}')

    def _objective_function(self):
        pass

    def _dict_hydesign_equations(self):
        self.mdl.dict_hydesign_equations['constraints']['power balance'] = self.mdl.dict_hydesign_equations['constraints'].get('power balance', '') +\
                                                                           '+wind_ts[t]'

    def outputs_cplex_parts(self, sol, outputs):
        return outputs
