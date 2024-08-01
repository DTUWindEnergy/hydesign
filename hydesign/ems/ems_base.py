import pandas as pd
import numpy as np
from .ems_abc import ems_abc, ems_cplex_abc, ems_cplex_parts_abc

from .ems_utils import expand_to_lifetime

class ems_base_abc(ems_abc):

    def add_inputs(self, component):
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
            desc="Quantile of price time series to define peak price hours (above this quantile)." +
                 " Only used for peak production penalty and for cost of battery degradation.")
        component.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Penalty occurs if number of full power hours expected per day at peak price are not reached.")

    def add_outputs(self, component):
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


    def added_ems_WSB_args(self, ems_WSB_args,  WSPr_df, inputs):
        G_MW = inputs['G_MW']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs[
            'n_full_power_hours_expected_per_day_at_peak_price'][0]
        WSPr_df['price_t'] = inputs['price_t']


        added_ems_WSB_args = {
            'price_ts': WSPr_df.price_t,
            'G_MW': G_MW[0],
            'peak_hr_quantile': peak_hr_quantile,
            'n_full_power_hours_expected_per_day_at_peak_price': n_full_power_hours_expected_per_day_at_peak_price,
            'technology_types': ['base', 'wind', 'solar', 'bess'],
            }

        ems_WSB_args.update(added_ems_WSB_args)
        return ems_WSB_args

    def outputs_extension(self, component, inputs, outputs, outputs_ems_WSB):
        price_t = inputs['price_t']
        P_HPP_ts = outputs_ems_WSB['hpp_t']
        P_curtailment_ts = outputs_ems_WSB['hpp_curt_t']
        penalty_ts = outputs_ems_WSB['penalty_t']

        # Extend (by repeating them and stacking) all variable to full lifetime
        outputs['price_t_ext'] = expand_to_lifetime(
            price_t, life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)
        outputs['hpp_t'] = expand_to_lifetime(
            P_HPP_ts, life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)
        outputs['hpp_curt_t'] = expand_to_lifetime(
            P_curtailment_ts, life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)
        outputs['penalty_t'] = expand_to_lifetime(
            penalty_ts, life_h=component.life_h, weeks_per_season_per_year=component.weeks_per_season_per_year)

        return outputs

class ems_base_long_term_operation(ems_abc):

    def add_inputs(self, component):
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
            desc="Quantile of price tim sereis to define peak price hours (above this quantile).\n" +
                 "Only used for peak production penalty and for cost of battery degradation.")
        component.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")
        component.add_input(
            'load_min',
            desc="Minimum electrical load to meet")

    def add_outputs(self, component):
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

    def added_ems_WSB_args(self, ems_WSB_args, inputs):
        price_t_ext = inputs['price_t_ext']
        G_MW = inputs['G_MW']
        hpp_curt_t = inputs['hpp_curt_t']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

        added_ems_WSB_args = dict(
                    hpp_curt_t=hpp_curt_t,
                    G_MW=G_MW[0],
                    price_ts=price_t_ext,
                    peak_hr_quantile=peak_hr_quantile,
                    n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price, )

        ems_WSB_args.update(added_ems_WSB_args)
        return ems_WSB_args

    def outputs_extension(self, component, inputs, outputs, outputs_ems_WSB):
        outputs['hpp_t_with_deg'] = outputs_ems_WSB['Hpp_deg']
        outputs['hpp_curt_t_with_deg'] = outputs_ems_WSB['P_curt_deg']
        outputs['penalty_t_with_deg'] = outputs_ems_WSB['penalty_ts']
        outputs['total_curtailment_with_deg'] = outputs_ems_WSB['P_curt_deg'].sum()
        outputs['total_curtailment'] = inputs['hpp_curt_t'].sum()

        return outputs

class ems_cplex_base(ems_cplex_abc):
    def allocate_vars(self, allocated_vars, length_ts):
        added_allocated_vars = dict(
            P_HPP_ts = np.zeros(length_ts),
            P_curtailment_ts = np.zeros(length_ts),
            penalty_ts = np.zeros(length_ts)
        )
        allocated_vars.update(added_allocated_vars)
        return allocated_vars

    def inputs_to_ems_cplex_parts(self, inputs, batch, outputs):
        added_outputs = dict(
            price_ts=inputs['price_ts'].iloc[batch],
            G_MW=inputs['G_MW'],
            peak_hr_quantile=inputs['peak_hr_quantile'],
            cost_of_battery_P_fluct_in_peak_price_ratio=inputs['cost_of_battery_P_fluct_in_peak_price_ratio'],  # [0, 0.8]. For higher values might cause errors
            n_full_power_hours_expected_per_day_at_peak_price=inputs['n_full_power_hours_expected_per_day_at_peak_price'],
            technology_types=inputs['technology_types'],
        )

        outputs.update(added_outputs)

        return outputs

    def outputs_batch(self, outputs_ems_cplex_parts, batch, allocated_vars):
        allocated_vars['P_HPP_ts'][batch] = outputs_ems_cplex_parts['P_HPP_ts']
        allocated_vars['P_curtailment_ts'][batch] = outputs_ems_cplex_parts['P_curtailment_ts']
        allocated_vars['penalty_ts'][batch] = outputs_ems_cplex_parts['penalty_ts']
        return allocated_vars

    def ems_cplex_outputs(self, outputs, allocated_vars):
        added_outputs = {
            'hpp_t': allocated_vars['P_HPP_ts'],
            'hpp_curt_t': allocated_vars['P_curtailment_ts'],
            'penalty_t': allocated_vars['penalty_ts'],
        }
        outputs.update(added_outputs)
        return outputs

class ems_cplex_parts_base_pre(ems_cplex_parts_abc):
    def __init__(self, mdl, time, SOCtime, G_MW, peak_hours_index,
                 e_peak_period_expected, price_peak, price_ts):
        self.mdl = mdl
        self.time = time
        self.SOCtime = SOCtime
        self.G_MW = G_MW
        self.peak_hours_index = peak_hours_index
        self.e_peak_period_expected = e_peak_period_expected
        self.price_peak = price_peak
        self.price_ts = price_ts

        self._variables()
        self._constraints()
        self._objective_function()

    def _variables(self):

        self.P_HPP_t = self.mdl.continuous_var_dict(
            self.time, lb=0, ub=self.G_MW,
            name='HPP power output')

        self.P_curtailment_t = self.mdl.continuous_var_dict(
            self.time, lb=0,
            name='Curtailment')

        self.penalty = self.mdl.continuous_var(name='penalty', lb=-1e12)
        self.e_penalty = self.mdl.continuous_var(name='e_penalty', lb=-1e12)


    def _constraints(self):
        # add constraints
        for t in self.time:
            tt = t + pd.Timedelta('1hour')
            dt = 1
            self.mdl.add_constraint(self.P_HPP_t[t] == 0, f'power_balance_{t}')


    def _objective_function(self):
        pass

    def get_added_variables(self):
        return {'P_HPP_t': self.P_HPP_t, 'P_curtailment_t': self.P_curtailment_t, 'penalty': self.penalty, 'e_penalty': self.e_penalty}


class ems_cplex_parts_base_post(ems_cplex_parts_abc):
    def __init__(self, mdl, time, SOCtime, G_MW, peak_hours_index,
                 e_peak_period_expected, price_peak, price_ts, vars):
        self.mdl = mdl
        self.time = time
        self.SOCtime = SOCtime
        self.G_MW = G_MW
        self.peak_hours_index = peak_hours_index
        self.e_peak_period_expected = e_peak_period_expected
        self.price_peak = price_peak
        self.price_ts = price_ts

        self._variables()
        self._constraints()
        self._objective_function()

    def _variables(self):
        pass


    def _constraints(self):
        for t in self.time:
            tt = t + pd.Timedelta('1hour')
            dt = 1

            # Update specific constraints
            constraint = self.mdl.get_constraint_by_name(f'power_balance_{t}')
            self.mdl.remove_constraint(constraint)
            self.mdl.add_constraint(constraint.lhs == constraint.rhs - self.mdl.get_var_by_name(f'Curtailment_{t}'), f'power_balance_{t}')

        e_peak_period_expr = self.e_peak_period_expected - self.mdl.sum(self.mdl.get_var_by_name(f'HPP power output_{self.time[i]}' for i in self.peak_hours_index))
        self.mdl.add_constraint(self.mdl.get_var_by_name(f'e_penalty') == e_peak_period_expr, f'e_penalty')
        # Piecewise function for "only positive" function
        f1 = self.mdl.piecewise(0, [(0, 0)], 1)
        self.mdl.add_constraint(self.mdl.get_var_by_name(f'penalty') == self.price_peak * f1(self.mdl.get_var_by_name(f'e_penalty')), 'penalty')


    def _objective_function(self):
        # Extract current objective function
        current_objective = self.mdl.objective_expr

        # Revenues and OPEX
        revenue_opex = self.mdl.sum(
            self.price_ts[t] *
            self.mdl.get_var_by_name(f'HPP power output_{t}') *
            (self.mdl.get_var_by_name(f'Hydrogen storage level_{t}') if self.mdl.get_var_by_name(f'Hydrogen storage level_{t}') is not None else 1)
            for t in self.time) - self.mdl.get_var_by_name('penalty')

        # Define the new objective
        new_objective = current_objective + revenue_opex

        # Update the objective function
        self.mdl.remove_objective()
        self.mdl.maximize(new_objective)

    def get_added_variables(self):
        return {}