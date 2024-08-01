from abstract_ems import AbstractEMS


class EMS(AbstractEMS):
    def __init__(self, N_time, life_h=25 * 365 * 24, weeks_per_season_per_year=None, ems_type='cplex'):
        super().__init__(N_time, life_h, weeks_per_season_per_year, ems_type)

    def setup(self):
        self.setup_common_inputs()
        self.setup_common_outputs()

    def compute(self, inputs, outputs):
        wind_t = inputs['wind_t']
        solar_t = inputs['solar_t']
        price_t = inputs['price_t']
        b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']

        if self.ems_type == 'cplex':
            ems_WSB = ems_cplex
        elif self.ems_type == 'pyomo':
            ems_WSB = ems_Wind_Solar_Battery_Pyomo
        else:
            raise Warning("This class should only be used for cplex or pyomo")

        battery_depth_of_discharge = inputs['battery_depth_of_discharge']
        battery_charge_efficiency = inputs['battery_charge_efficiency']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        cost_of_battery_P_fluct_in_peak_price_ratio = inputs['cost_of_battery_P_fluct_in_peak_price_ratio'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

        WSPr_df = pd.DataFrame(index=pd.date_range(start='01-01-1991 00:00', periods=len(wind_t), freq='1h'))
        WSPr_df['wind_t'] = wind_t
        WSPr_df['solar_t'] = solar_t
        WSPr_df['price_t'] = price_t
        WSPr_df['E_batt_MWh_t'] = b_E[0]

        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts = ems_WSB(
            wind_ts=WSPr_df.wind_t,
            solar_ts=WSPr_df.solar_t,
            price_ts=WSPr_df.price_t,
            P_batt_MW=b_P[0],
            E_batt_MWh_t=WSPr_df.E_batt_MWh_t,
            hpp_grid_connection=G_MW[0],
            battery_depth_of_discharge=battery_depth_of_discharge[0],
            charge_efficiency=battery_charge_efficiency[0],
            peak_hr_quantile=peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price,
        )

        outputs['wind_t_ext'] = self.expand_to_lifetime(wind_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['solar_t_ext'] = self.expand_to_lifetime(solar_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['price_t_ext'] = self.expand_to_lifetime(price_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['hpp_t'] = self.expand_to_lifetime(P_HPP_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['hpp_curt_t'] = self.expand_to_lifetime(P_curtailment_ts, life_h=self.life_h,
                                                        weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['b_t'] = self.expand_to_lifetime(P_charge_discharge_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['b_E_SOC_t'] = self.expand_to_lifetime(E_SOC_ts[:-1], life_h=self.life_h + 1,
                                                       weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['penalty_t'] = self.expand_to_lifetime(penalty_ts, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
