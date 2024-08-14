import pandas as pd
import numpy as np
from .ems_abc import ems_abc, ems_cplex_abc, ems_cplex_parts_abc, ems_degredation_abc

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


    def add_ems_args(self, ems_args,  WSPr_df, inputs):
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
        return ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        price_t = inputs['price_t']
        P_HPP_ts = outputs_ems['hpp_t']
        P_curtailment_ts = outputs_ems['hpp_curt_t']
        penalty_ts = outputs_ems['penalty_t']

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

    def add_ems_args(self, ems_args, inputs):
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
        return ems_args

    def postprocessing_outputs(self, component, inputs, outputs, outputs_ems):
        outputs['hpp_t_with_deg'] = outputs_ems['Hpp_deg']
        outputs['hpp_curt_t_with_deg'] = outputs_ems['P_curt_deg']
        outputs['penalty_t_with_deg'] = outputs_ems['penalty_ts']
        outputs['total_curtailment_with_deg'] = outputs_ems['P_curt_deg'].sum()
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
    def __init__(self, mdl, inputs, extra_inputs={}):
        self.mdl = mdl
        self.inputs = inputs
        self.extra_inputs = extra_inputs
        self.G_MW = inputs['G_MW']
        self.time = extra_inputs['time']


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

    def _dict_hydesign_equations(self):
        # initialize dict
        self.mdl.dict_hydesign_equations = {}
        self.mdl.dict_hydesign_equations['decision variables'] = {}
        self.mdl.dict_hydesign_equations['objective function'] = {}
        self.mdl.dict_hydesign_equations['constraints'] = {}

        # set some of the values
        self.mdl.dict_hydesign_equations['decision variables']['HPP power output'] = 'P_HPP_t'
        self.mdl.dict_hydesign_equations['decision variables']['Curtailment'] = 'P_curtailment_t'
        self.mdl.dict_hydesign_equations['objective function']['sense'] = 'maximize'
        self.mdl.dict_hydesign_equations['objective function']['equation'] = ''
        self.mdl.dict_hydesign_equations['constraints']['power balance'] = 'P_HPP_t[t] == '

    def outputs_cplex_parts(self, sol, outputs):
        N_t = self.extra_inputs['N_t']
        time = self.extra_inputs['time']
        SOCtime = self.extra_inputs['SOCtime']

        P_HPP_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_HPP_t), orient='index').loc[:, 0]

        P_curtailment_ts_df = pd.DataFrame.from_dict(
            sol.get_value_dict(self.P_curtailment_t), orient='index').loc[:, 0]

        # make a time series like P_HPP with a constant penalty
        penalty = sol.get_value(self.penalty)
        penalty_ts = np.ones(N_t) * (penalty/ N_t)

        # Cplex sometimes returns missing values :O
        P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values
        P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values

        added_outputs = {'P_HPP_ts': P_HPP_ts,
                         'P_curtailment_ts': P_curtailment_ts,
                         'penalty_ts': penalty_ts
                         }
        outputs.update(added_outputs)

        return outputs

class ems_cplex_parts_base_post(ems_cplex_parts_abc):
    def __init__(self, mdl, inputs, extra_inputs={}):
        self.inputs = inputs
        self.extra_inputs = extra_inputs
        self.mdl = mdl
        self.price_ts = inputs['price_ts']
        self.time = extra_inputs['time']
        self.e_peak_period_expected = extra_inputs['e_peak_period_expected']
        self.peak_hours_index = extra_inputs['peak_hours_index']
        self.price_peak = extra_inputs['price_peak']

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

    def _dict_hydesign_equations(self):
        self.mdl.dict_hydesign_equations['objective function']['equation'] = self.mdl.dict_hydesign_equations['objective function']['equation'] +\
                                                                 'sum(price[t] * P_HPP[t] * Hydrogen_storage_level[t]) - penalty'

        self.mdl.dict_hydesign_equations['constraints']['power balance'] = self.mdl.dict_hydesign_equations['constraints'].get('power balance', '') +\
                                                                           '-P_Curtailment[t] '

        self.mdl.dict_hydesign_equations['constraints']['e_penalty'] = 'e_penalty == e_peak_period_expected - sum(P_HPP[i] for i in peak_hours_index)'
        self.mdl.dict_hydesign_equations['constraints']['penalty'] = 'penalty == price_peak * e_penalty'

    def outputs_cplex_parts(self, sol, outputs):
        # if len(P_HPP_ts_df) < len(wind_ts):
        #     # print('recomputing p_hpp')
        #     P_HPP_ts = wind_ts.values + solar_ts.values + \
        #                - P_curtailment_ts + P_charge_discharge_ts

        return outputs



class ems_degredation_base(ems_degredation_abc):

    def __init__(self, inputs):
        self.inputs = inputs

    def compute_wind_solar_deg(self):
        """Compute the combined degraded wind and solar power."""
        wind_t_deg = self.inputs.get('wind_t_deg')
        solar_t_deg = self.inputs.get('solar_t_deg')
        return wind_t_deg + solar_t_deg

    def initialize_soc(self):
        """Initialize the state of charge (SoC)."""
        b_E_SOC_t = self.inputs.get('b_E_SOC_t')
        b_E_SOC_0 = self.inputs.get('b_E_SOC_0', None)
        b_E_SOC_t_sat = b_E_SOC_t.copy()
        if b_E_SOC_0 is None:
            b_E_SOC_t_sat[0] = b_E_SOC_t[0]
        else:
            b_E_SOC_t_sat[0] = b_E_SOC_0
        return b_E_SOC_t_sat

    def adjust_battery_power(self, wind_solar_t_deg):
        """Adjust battery power for solar degradation."""
        b_t = self.inputs.get('b_t')
        B_p = np.max(np.abs(b_t))

        b_t_less_sol = b_t.copy()
        for i in range(len(b_t)):
            if b_t[i] < 0:  # Charging
                if -b_t[i] > wind_solar_t_deg[i]:
                    b_t_less_sol[i] = -wind_solar_t_deg[i]
                b_t_less_sol[i] = np.clip(b_t_less_sol[i], -B_p, B_p)
        return b_t_less_sol

    def update_soc(self, b_t_less_sol, b_E_SOC_t_sat):
        """Update the state of charge (SoC) based on adjusted battery power."""
        battery_charge_efficiency = self.inputs.get('battery_charge_efficiency')
        battery_depth_of_discharge = self.inputs.get('battery_depth_of_discharge')
        b_E = self.inputs.get('b_E')
        batt_degradation = self.inputs.get('batt_degradation')

        dt = 1  # Assuming time step is 1 hour
        for i in range(len(b_t_less_sol)):
            if b_t_less_sol[i] < 0:  # Charging
                b_E_SOC_t_sat[i + 1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt * battery_charge_efficiency
            else:  # Discharging
                b_E_SOC_t_sat[i + 1] = b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt / battery_charge_efficiency
            b_E_SOC_t_sat[i + 1] = np.clip(b_E_SOC_t_sat[i + 1],
                                           (1 - battery_depth_of_discharge) * b_E * batt_degradation[i],
                                           b_E * batt_degradation[i])
        return b_E_SOC_t_sat

    def recompute_battery_power(self, b_E_SOC_t_sat):
        """Recompute the battery power based on updated SoC."""
        b_t = self.inputs.get('b_t')
        battery_charge_efficiency = self.inputs.get('battery_charge_efficiency')

        b_t_sat = b_t.copy()
        dt = 1  # Assuming time step is 1 hour
        for i in range(len(b_t_sat)):
            if b_t[i] < 0:  # Charging
                b_t_sat[i] = ((b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i + 1]) / battery_charge_efficiency) / dt
            else:  # Discharging
                b_t_sat[i] = ((b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i + 1]) * battery_charge_efficiency) / dt
        return b_t_sat

    def compute_penalty(self, Hpp_deg):
        """Compute penalties for not meeting expected energy production at peak hours."""
        price_ts = self.inputs.get('price_ts')
        peak_hr_quantile = self.inputs.get('peak_hr_quantile', 0.9)
        n_full_power_hours_expected_per_day_at_peak_price = self.inputs.get('n_full_power_hours_expected_per_day_at_peak_price', 3)
        G_MW = self.inputs.get('G_MW')

        H_df = pd.DataFrame(
            {'hpp_t_with_deg': Hpp_deg},
            index=pd.date_range(start='01-01-1991 00:00', periods=len(Hpp_deg), freq='1h')
        )
        H_df['price_t_ext'] = price_ts
        price_peak = np.quantile(H_df['price_t_ext'], peak_hr_quantile)
        H_df['peak'] = H_df['price_t_ext'] >= price_peak
        H_df.loc[~H_df['peak'], 'hpp_t_with_deg'] = 0

        N_days = 7
        H_df_week = H_df.resample(f'{N_days}d').sum()
        e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price * G_MW
        e_peak_period_expected = e_peak_day_expected * N_days
        H_df_week['e_peak_period_expected'] = e_peak_period_expected
        H_df_week['e_penalty'] = H_df_week['e_peak_period_expected'] - H_df_week['hpp_t_with_deg']
        H_df_week['penalty'] = price_peak * np.maximum(0, H_df_week['e_penalty'])

        penalty_df = H_df_week['penalty'].reindex_like(H_df).fillna(0.0)
        penalty_ts = penalty_df.values.flatten()
        return penalty_ts
