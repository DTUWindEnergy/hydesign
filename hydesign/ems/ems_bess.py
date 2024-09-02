import pandas as pd
import numpy as np
from .ems_abc import ems_abc, ems_cplex_abc, ems_cplex_parts_abc
from .ems_utils import expand_to_lifetime

class ems_bess_abc(ems_abc):
    def add_inputs(self, component):
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

    def add_outputs(self, component):
        component.add_output(
            'b_t',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[component.life_h])
        component.add_output(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[component.life_h + 1])

    def add_ems_args(self):
        WSPr_df = """WSPr_df['b_E'] = inputs['b_E'][0]"""
        added_ems_args = """
            # args bess
            'b_P': inputs['b_P'][0],
            'b_E': WSPr_df.b_E,
            'battery_depth_of_discharge': inputs['battery_depth_of_discharge'][0],
            'battery_charge_efficiency': inputs['battery_charge_efficiency'][0],
            'cost_of_battery_P_fluct_in_peak_price_ratio': inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0],"""
        return WSPr_df, added_ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        P_charge_discharge_ts = outputs_ems['b_t']
        E_SOC_ts = outputs_ems['b_E_SOC_t']
        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['b_t'] = expand_to_lifetime(
            P_charge_discharge_ts, life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)
        outputs['b_E_SOC_t'] = expand_to_lifetime(
            E_SOC_ts[:-1], life_h=component.life_h + 1, weeks_per_season_per_year=component.weeks_per_season_per_year)

        return outputs

class ems_bess_long_term_operation(ems_abc):

    def add_inputs(self, component):
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

    def add_outputs(self, component):
        component.add_output(
            'b_t_with_deg',
            desc="Battery charge/discharge power time series",
            units='MW',
            shape=[component.life_h])
        component.add_output(
            'b_E_SOC_t_with_deg',
            desc="Battery energy SOC time series",
            shape=[component.life_h + 1])

    def add_ems_args(self):
        WSPr_df = ''
        added_ems_args = """
            # args bess
            'batt_degradation': inputs['SoH'],
            'b_t': inputs['b_t'],
            'b_E_SOC_t': inputs['b_E_SOC_t'],
            'b_E': inputs['b_E'][0],
            'battery_depth_of_discharge': inputs['battery_depth_of_discharge'][0],
            'battery_charge_efficiency': inputs['battery_charge_efficiency'][0],
            'b_E_SOC_0': None,
            """
        return WSPr_df, added_ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        outputs['b_t_with_deg'] = outputs_ems['b_t_sat']
        outputs['b_E_SOC_t_with_deg'] = outputs_ems['b_E_SOC_t_sat']

        return outputs

class ems_cplex_bess(ems_cplex_abc):
    def allocate_vars(self, allocated_vars, length_ts):
        added_allocated_vars = dict(
            P_charge_discharge_ts = np.zeros(length_ts),
            E_SOC_ts = np.zeros(length_ts + 1)
        )
        allocated_vars.update(added_allocated_vars)
        return allocated_vars

    def allocate_vars2(self, length_ts):
        allocated_vars_bess = dict(
            P_charge_discharge_ts = np.zeros(length_ts),
            E_SOC_ts = np.zeros(length_ts + 1)
        )


    def inputs_to_ems_cplex_parts(self, inputs, batch, outputs):
        added_outputs = dict(
            b_P=inputs['b_P'],
            battery_depth_of_discharge=inputs['battery_depth_of_discharge'],
            battery_charge_efficiency=inputs['battery_charge_efficiency'],
            cost_of_battery_P_fluct_in_peak_price_ratio=inputs['cost_of_battery_P_fluct_in_peak_price_ratio'],  # [0, 0.8]. For higher values might cause errors
            b_E=inputs['b_E'].iloc[batch]
        )
        outputs.update(added_outputs)
        return outputs

    def inputs_to_ems_cplex_parts2(self, inputs, batch, outputs):
        added_outputs = dict(
            b_P=inputs['b_P'],
            battery_depth_of_discharge=inputs['battery_depth_of_discharge'],
            battery_charge_efficiency=inputs['battery_charge_efficiency'],
            cost_of_battery_P_fluct_in_peak_price_ratio=inputs['cost_of_battery_P_fluct_in_peak_price_ratio'],  # [0, 0.8]. For higher values might cause errors
            b_E=inputs['b_E'].iloc[batch]
        )

    def outputs_batch(self, outputs_ems_cplex_parts, batch, allocated_vars):
        allocated_vars['P_charge_discharge_ts'][batch] = outputs_ems_cplex_parts['P_charge_discharge_ts']
        allocated_vars['E_SOC_ts'][batch] = outputs_ems_cplex_parts['E_SOC_ts'][:-1]
        return allocated_vars

    def ems_cplex_outputs(self, outputs, allocated_vars):
        added_outputs = {
            'b_t': allocated_vars['P_charge_discharge_ts'],
            'b_E_SOC_t': allocated_vars['E_SOC_ts']
        }
        outputs.update(added_outputs)
        return outputs

    def ems_cplex_outputs2(self):
        added_outputs = {
            'b_t': allocated_vars['P_charge_discharge_ts'],
            'b_E_SOC_t': allocated_vars['E_SOC_ts']
        }

class ems_cplex_parts_bess(ems_cplex_parts_abc):
    def __init__(self, mdl, inputs, extra_inputs={}):
        self.mdl = mdl
        self.inputs = inputs
        self.extra_inputs = extra_inputs
        self.b_P=inputs.get('b_P')
        self.b_E=inputs.get('b_E')
        self.battery_depth_of_discharge=inputs.get('battery_depth_of_discharge')
        self.battery_charge_efficiency=inputs.get('battery_charge_efficiency')
        self.cost_of_battery_P_fluct_in_peak_price_ratio = inputs.get('cost_of_battery_P_fluct_in_peak_price_ratio', 0.5) #[0, 0.8]. For higher values might cause errors
        self.time = extra_inputs['time']
        self.SOCtime = extra_inputs['SOCtime']
        self.price_peak = extra_inputs['price_peak']
        self.price_ts_to_max = extra_inputs['price_ts_to_max']

    def _variables(self):
        self.P_charge_discharge = self.mdl.continuous_var_dict(self.time, lb=-self.b_P / self.battery_charge_efficiency, ub=self.b_P * self.battery_charge_efficiency, name='Battery_power')
        self.E_SOC_t = self.mdl.continuous_var_dict(self.SOCtime, lb=0, name='Energy_level')

    def _constraints(self):
        self.mdl.add_constraint(self.E_SOC_t[self.SOCtime[0]] == 0.5 * self.b_E[self.time[0]])
        self.mdl.add_constraint(self.E_SOC_t[self.SOCtime[-1]] == 0.5 * self.b_E[self.time[0]])

        f2 = self.mdl.piecewise(self.battery_charge_efficiency, [(0, 0)], 1 / self.battery_charge_efficiency)

        # Iterate over each time step
        for t in self.time:
            tt = t + pd.Timedelta('1hour')
            dt = 1

            # add new constraints
            self.mdl.add_constraint(self.E_SOC_t[tt] == self.E_SOC_t[t] - f2(self.P_charge_discharge[t]) * dt, f'E_SOC_tt_{tt}') # charge/dischrage equation
            self.mdl.add_constraint(self.E_SOC_t[t] >= (1 - self.battery_depth_of_discharge) * self.b_E[t], f'E_SOC_{t}') # Constraining battery energy level to minimum battery level
            self.mdl.add_constraint(self.E_SOC_t[t] <= self.b_E[t], f'E_SOC_lt_b_E{t}') # Constraining battery energy level to maximum battery level (lt: less than)
            self.mdl.add_constraint(self.P_charge_discharge[t] <= self.b_P * self.battery_charge_efficiency, f'P_charge_discharge_lt_b_P_{t}') # Battery charge/discharge within its power rating
            self.mdl.add_constraint(self.P_charge_discharge[t] >= -self.b_P / self.battery_charge_efficiency, f'P_charge_discharge_gt_neg_b_P_{t}') # gt_neg: greater than negative of

            # Update specific constraints
            constraint = self.mdl.get_constraint_by_name(f'power_balance_{t}')
            self.mdl.remove_constraint(constraint)
            self.mdl.add_constraint(constraint.lhs == constraint.rhs + self.P_charge_discharge[t], f'power_balance_{t}')


    def _objective_function(self):
        # Extract current objective function
        current_objective = self.mdl.objective_expr

        # Add additional cost term to the objective function
        # Piecewise function for "absolute value" function
        fabs = self.mdl.piecewise(-1, [(0, 0)], 1)

        additional_cost = self.mdl.sum(
           fabs(self.P_charge_discharge[t + pd.Timedelta('1hour')] - \
                self.P_charge_discharge[t])*self.cost_of_battery_P_fluct_in_peak_price_ratio*self.price_ts_to_max[t]
           for t in self.time[:-1])

        new_objective = current_objective - additional_cost
        self.mdl.remove_objective()
        self.mdl.maximize(new_objective) # This line actually overwrites the objective function (no need to remove the objective function in the previous line).


    def _dict_hydesign_equations(self):
        # new variables
        self.mdl.dict_hydesign_equations['decision variables']['Energy_level'] = 'E_SOC_t'
        self.mdl.dict_hydesign_equations['decision variables']['Battery_power'] = 'P_charge_discharge'

        # updated objective function
        self.mdl.dict_hydesign_equations['objective function']['equation'] = self.mdl.dict_hydesign_equations['objective function']['equation'] + \
                                                                 '-sum(Piecewise(P_charge_discharge[t + 1hour] - P_charge_discharge[t]) * cost_of_battery_P_fluct_in_peak_price_ratio * self.price_ts_to_max[t] for t in time[:-1])'
        # updated constraints
        self.mdl.dict_hydesign_equations['constraints']['power balance'] = self.mdl.dict_hydesign_equations['constraints'].get('power balance', '') +\
                                                                           '+P_charge_discharge[t]'
        # new constraints
        self.mdl.dict_hydesign_equations['constraints']['E_SOC_t[tt]'] = 'E_SOC_t[t+1hour] == E_SOC_t[t] - Piecewise(P_charge_discharge[t]) * dt, E_SOC_tt[t+1hour])'  # charge/dischrage equation
        self.mdl.dict_hydesign_equations['constraints']['E_SOC_t[tt]'] = 'E_SOC_t[t] >= (1 - self.battery_depth_of_discharge) * b_E[t]'
        self.mdl.dict_hydesign_equations['constraints']['E_SOC_lt_b_E[t]'] = 'E_SOC_t[t] <= self.b_E[t]'
        self.mdl.dict_hydesign_equations['constraints']['P_charge_discharge_lt_b_P[t]'] = 'P_charge_discharge[t] <= b_P * battery_charge_efficiency'
        self.mdl.dict_hydesign_equations['constraints']['P_charge_discharge_gt_neg_b_P[t]'] = 'P_charge_discharge[t] >= -self.b_P / self.battery_charge_efficiency'

    def outputs_cplex_parts(self, sol, outputs):
        N_t = self.extra_inputs['N_t']
        time = self.extra_inputs['time']
        SOCtime = self.extra_inputs['SOCtime']

        P_charge_discharge_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_charge_discharge), orient='index').loc[:, 0]

        E_SOC_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.E_SOC_t), orient='index').loc[:, 0]

        # Cplex sometimes returns missing values :O
        P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values
        E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values


        added_outputs = {
            'P_charge_discharge_ts': P_charge_discharge_ts,
            'E_SOC_ts': E_SOC_ts
        }
        outputs.update(added_outputs)

        return outputs
