import pandas as pd
from .ems_abc import ems_abc, ems_cplex_parts_abc
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

    def added_ems_WSB_args(self, ems_WSB_args,  WSPr_df, inputs):
        WSPr_df['solar_t'] = inputs['solar_t']

        added_ems_WSB_args = {'solar_ts': WSPr_df.solar_t,}

        ems_WSB_args.update(added_ems_WSB_args)
        return ems_WSB_args

    def outputs_extension(self, component, inputs, outputs, outputs_ems_WSB):
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

    def added_ems_WSB_args(self, ems_WSB_args, inputs):
        solar_t_ext_deg = inputs['solar_t_ext_deg']
        solar_t_ext = inputs['solar_t_ext']

        added_ems_WSB_args = dict(
                    solar_t_deg=solar_t_ext_deg,
                    solar_t=solar_t_ext,
                    )


        ems_WSB_args.update(added_ems_WSB_args)
        return ems_WSB_args

    def outputs_extension(self, component, inputs, outputs, outputs_ems_WSB):
        return outputs

class ems_cplex_parts_solar(ems_cplex_parts_abc):
    def __init__(self, mdl, time, SOCtime, G_MW, b_P, b_E, wind_ts, solar_ts,
                 cost_of_battery_P_fluct_in_peak_price_ratio, price_ts_to_max,
                 charge_efficiency, battery_depth_of_discharge, peak_hours_index,
                 e_peak_period_expected, price_peak, price_ts, vars):
        self.mdl = mdl
        self.time = time
        self.SOCtime = SOCtime
        self.G_MW = G_MW
        self.b_P = b_P
        self.b_E = b_E
        self.wind_ts = wind_ts
        self.solar_ts = solar_ts
        self.cost_of_battery_P_fluct_in_peak_price_ratio = cost_of_battery_P_fluct_in_peak_price_ratio
        self.price_ts_to_max = price_ts_to_max
        self.charge_efficiency = charge_efficiency
        self.battery_depth_of_discharge = battery_depth_of_discharge
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
        # Iterate over each time step
        for t in self.time:
            tt = t + pd.Timedelta('1hour')

            # Update specific constraints
            constraint = self.mdl.get_constraint_by_name(f'power_balance_{t}')
            self.mdl.remove_constraint(constraint)
            self.mdl.add_constraint(constraint.lhs == constraint.rhs + self.solar_ts[t], f'power_balance_{t}')

        # for i in self.peak_hours_index:
        #     # Update specific constraints
        #     constraint = self.mdl.get_constraint_by_name(f'e_penalty_{i}')
        #     self.mdl.remove_constraint(constraint)
        #     self.mdl.add_constraint(constraint.lhs == constraint.rhs - self.mdl.sum(self.solar_ts[self.time[i]]), f'e_penalty_{t}')
        # # Piecewise function for "only positive" function
        # f1 = self.mdl.piecewise(0, [(0,0)], 1)
        # constraint = self.mdl.get_constraint_by_name('penalty')
        # self.mdl.remove_constraint(constraint)
        # self.mdl.add_constraint( self.penalty == self.price_peak*f1(self.e_penalty), 'penalty' )

    def _objective_function(self):
        pass

    def get_added_variables(self):
        return {}
