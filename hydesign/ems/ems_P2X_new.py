import pandas as pd
from .ems_abc import ems_cplex_abc

class ems_cplex_p2x(ems_cplex_abc):
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
        self.price_ts = price_ts
        self.peak_hours_index = peak_hours_index
        self.e_peak_period_expected = e_peak_period_expected
        self.price_peak = price_peak
        self.price_ts = price_ts

        self._variables()
        self._constraints()
        self._objective_function()

    def _variables(self):
        # Hydrogen storgae level
        self.LoS_H2_t = self.mdl.continuous_var_dict(SOCtime, lb=0, name='Hydrogen storage level')

        # Power to gas plant power consumption, produced hydrogen, electrolyzer efficiency
        self.P_ptg_t = self.mdl.continuous_var_dict(time, lb=self.min_power_standby * self.ptg_MW, ub=self.ptg_MW, name = "Power to gas plant consumption")

        self.m_H2_t = self.mdl.continuous_var_dict(self.time, lb = 0, name = 'Produced hydrogen')
        self.m_H2_offtake_t = self.mdl.continuous_var_dict(self.time, lb = 0, name = 'Hydrogen offtake')

    def _constraints(self):
        # Intitial and end LoS_H2
        mdl.add_constraint(LoS_H2_t[SOCtime[0]] == 0, 'LoS_H2_initial')
        mdl.add_constraint(LoS_H2_t[SOCtime[-1]] == 0, 'LoS_H2_end')

        # piecewise linear representation of battery charge vs dischrage effciency
        f2 = mdl.piecewise(charge_efficiency, [(0, 0)], 1 / charge_efficiency)

        # piecewise linear representation of H2 storage vs offtake effciency
        f3 = mdl.piecewise(1 / storage_eff, [(0, 0)], storage_eff)

        # Caclulating Hydrogen production as a function of load (piecewise linear approximation)
        self.H2_curve_list = [(load * self.ptg_MW, hydrogen * self.ptg_MW) for load, hydrogen in eff_curve]
        self.H2_prod = mdl.piecewise(0, H2_curve_list, 0)

        for t in time:
            # Time index for successive time step
            tt = t + pd.Timedelta('1hour')
            # Delta_t of 1 hour
            dt = 1

            # Update specific constraints
            constraint = self.mdl.get_constraint_by_name(f'power_balance_{t}')
            self.mdl.remove_constraint(constraint)
            self.mdl.add_constraint(constraint.lhs == constraint.rhs - self.P_ptg_t[t], f'power_balance_{t}')

            # Hydrogen storgae equation
            self.mdl.add_constraint(LoS_H2_t[tt] == LoS_H2_t[t] + f3(m_H2_t[t] - m_H2_offtake_t[t]), f'LoS_H2_t_{t}')

            # Constraining hydrogen offtake as per the demand time series and level of storage
            self.mdl.add_constraint(LoS_H2_t[t] <= H2_storage_t[t], f'LoS_H2_storage_t_{t}')
            self.mdl.add_constraint(m_H2_offtake_t[t] <= m_H2_demand_ts[t], f'm_H2_offtake_t_{t}')


            if H2_storage_t[t] == 0:
                m_H2_offtake_t[t] = m_H2_t[t]

            # Calculating Hydrogen production with HPC
            self.mdl.add_constraint(m_H2_t[t] == H2_prod(P_ptg_t[t]), f'm_H2_t_{t}')


    def _objective_function(self):
        pass


    def get_added_variables(self):
        return {'LoS_H2_t': self.LoS_H2_t, 'P_ptg_t': self.P_ptg_t, 'm_H2_t': self.m_H2_t, 'm_H2_offtake_t': self.m_H2_offtake_t}