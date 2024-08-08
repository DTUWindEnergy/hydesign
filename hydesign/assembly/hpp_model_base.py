import glob
import os
import time
import ast

import numpy as np
import pandas as pd
import xarray as xr

from .hpp_model_abc import hpp_model_abc
from .assembly_utils import mkdir
from hydesign.look_up_tables import lut_filepath
from hydesign.weather.weather import extract_weather_for_HPP, ABL, select_years
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, wpp_with_degradation, get_rotor_area, get_rotor_d
from hydesign.pv.pv import pvp, pvp_with_degradation
from hydesign.ems.ems import ems, ems_long_term_operation
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance.finance import finance

class hpp_model_base(hpp_model_abc):
    def __init__(self, excel_data, sim_pars, verbose, weather, N_time, genWT_fn, N_ws, genWake_fn, price, **kwargs):
        self.sim_pars = sim_pars
        self.weather = weather
        self.N_time = N_time
        self.genWT_fn = genWT_fn
        self.N_ws = N_ws
        self.genWake_fn = genWake_fn
        self.price = price


    def add_subsystems(self, model):
        sim_pars = self.sim_pars
        weather = self.weather
        N_time = self.N_time
        genWT_fn = self.genWT_fn
        N_ws = self.N_ws
        genWake_fn = self.genWake_fn


        latitude = sim_pars.get('latitude')
        longitude = float(sim_pars.get('longitude'))
        altitude = sim_pars.get('altitude')
        wpp_efficiency = sim_pars['wpp_efficiency']
        wind_deg = sim_pars.get('wind_deg', [0, 0.25])
        wind_deg_yr = sim_pars.get('wind_deg_yr', [0, 25])
        share_WT_deg_types = sim_pars.get('share_WT_deg_types', 0.5)
        N_life = sim_pars['N_life']
        life_h = N_life * 365 * 24
        max_num_batteries_allowed = sim_pars.get('max_num_batteries_allowed', 3)
        factor_battery_cost = sim_pars.get('factor_battery_cost', 1)
        ems_type = sim_pars.get('ems_type', 'cplex')
        weeks_per_season_per_year = sim_pars.get('weeks_per_season_per_year')

        model.add_subsystem(
            'abl',
            ABL(
                weather=weather,
                N_time=N_time),
            promotes_inputs=['hh']
        )
        model.add_subsystem(
            'genericWT',
            genericWT_surrogate(
                genWT_fn=genWT_fn,
                N_ws=N_ws),
            promotes_inputs=[
                'hh',
                'd',
                'p_rated',
            ])

        model.add_subsystem(
            'genericWake',
            genericWake_surrogate(
                genWake_fn=genWake_fn,
                N_ws=N_ws),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'd',
                'p_rated',
            ])

        model.add_subsystem(
            'wpp',
            wpp(
                N_time=N_time,
                N_ws=N_ws,
                wpp_efficiency=wpp_efficiency, )
        )

        model.add_subsystem(
            'wpp_with_degradation',
            wpp_with_degradation(
                N_time=N_time,
                N_ws=N_ws,
                wpp_efficiency=wpp_efficiency,
                life_h=life_h,
                wind_deg_yr=wind_deg_yr,
                wind_deg=wind_deg,
                share_WT_deg_types=share_WT_deg_types,
                weeks_per_season_per_year=weeks_per_season_per_year,

            )
        )

        model.add_subsystem(
            'wpp_cost',
            wpp_cost(
                wind_turbine_cost=sim_pars['wind_turbine_cost'],
                wind_civil_works_cost=sim_pars['wind_civil_works_cost'],
                wind_fixed_onm_cost=sim_pars['wind_fixed_onm_cost'],
                wind_variable_onm_cost=sim_pars['wind_variable_onm_cost'],
                d_ref=sim_pars['d_ref'],
                hh_ref=sim_pars['hh_ref'],
                p_rated_ref=sim_pars['p_rated_ref'],
                N_time=N_time,
            ),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'hh',
                'd',
                'p_rated'])

        model.add_subsystem(
            'pvp',
            pvp(
                weather=weather,
                N_time=N_time,
                latitude=latitude,
                longitude=longitude,
                altitude=altitude,
                tracking=sim_pars['tracking']
            ),
            promotes_inputs=[
                'surface_tilt',
                'surface_azimuth',
                'DC_AC_ratio',
                'solar_MW',
                'land_use_per_solar_MW',
            ])
        model.add_subsystem(
            'ems',
            ems(
                N_time=N_time,
                weeks_per_season_per_year=weeks_per_season_per_year,
                life_h=life_h,
                ems_type=ems_type),
            promotes_inputs=[
                'price_t',
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile',
                'cost_of_battery_P_fluct_in_peak_price_ratio',
                'n_full_power_hours_expected_per_day_at_peak_price',
            ]
        )
        model.add_subsystem(
            'battery_degradation',
            battery_degradation(
                weather=weather,  # for extracting temperature
                num_batteries=max_num_batteries_allowed,
                life_h=life_h,
                weeks_per_season_per_year=weeks_per_season_per_year,
            ),
            promotes_inputs=[
                'min_LoH'
            ])

        model.add_subsystem(
            'battery_loss_in_capacity_due_to_temp',
            battery_loss_in_capacity_due_to_temp(
                weather=weather,  # for extracting temperature
                life_h=life_h,
                weeks_per_season_per_year=weeks_per_season_per_year,
            ),
        )


        model.add_subsystem(
            'pvp_with_degradation',
            pvp_with_degradation(
                life_h=life_h,
                pv_deg_yr=sim_pars['pv_deg_yr'],
                pv_deg=sim_pars['pv_deg'],
            )
        )

        model.add_subsystem(
            'ems_long_term_operation',
            ems_long_term_operation(
                N_time=N_time,
                life_h=life_h),
            promotes_inputs=[
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile',
                'n_full_power_hours_expected_per_day_at_peak_price'
            ],
            promotes_outputs=[
                'total_curtailment',
                'total_curtailment_with_deg'
            ])


        model.add_subsystem(
            'pvp_cost',
            pvp_cost(
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_inverter_cost=sim_pars['solar_inverter_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=factor_battery_cost * sim_pars['battery_energy_cost'],
                battery_power_cost=factor_battery_cost * sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=factor_battery_cost * sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=factor_battery_cost * sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=factor_battery_cost * sim_pars['battery_energy_onm_cost'],
                battery_price_reduction_per_year=factor_battery_cost * sim_pars['battery_price_reduction_per_year'],
                N_life=N_life,
                life_h=life_h
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'battery_price_reduction_per_year'])

        model.add_subsystem(
            'shared_cost',
            shared_cost(
                hpp_BOS_soft_cost=sim_pars['hpp_BOS_soft_cost'],
                hpp_grid_connection_cost=sim_pars['hpp_grid_connection_cost'],
                land_cost=sim_pars['land_cost'],
            ),
            promotes_inputs=[
                'G_MW',
                'Awpp',
            ])

        model.add_subsystem(
            'finance',
            finance(
                N_time=N_time,
                # Depreciation curve
                depreciation_yr=sim_pars['depreciation_yr'],
                depreciation=sim_pars['depreciation'],
                # Inflation curve
                inflation_yr=sim_pars['inflation_yr'],
                inflation=sim_pars['inflation'],
                ref_yr_inflation=sim_pars['ref_yr_inflation'],
                # Early paying or CAPEX Phasing
                phasing_yr=sim_pars['phasing_yr'],
                phasing_CAPEX=sim_pars['phasing_CAPEX'],
                life_h=life_h),
            promotes_inputs=['wind_WACC',
                             'solar_WACC',
                             'battery_WACC',
                             'tax_rate'
                             ],
            promotes_outputs=['NPV',
                              'IRR',
                              'NPV_over_CAPEX',
                              'LCOE',
                              'revenues',
                              'mean_AEP',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX',
                              'break_even_PPA_price',
                              ],
        )

        return model


    def connect_subsytems(self, model):
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('shared_cost.CAPEX_sh', 'finance.CAPEX_el')
        model.connect('shared_cost.OPEX_sh', 'finance.OPEX_el')
        model.connect('ems.price_t_ext', 'finance.price_t_ext')
        model.connect('ems_long_term_operation.hpp_t_with_deg', 'finance.hpp_t_with_deg')
        model.connect('ems_long_term_operation.penalty_t_with_deg', 'finance.penalty_t')

        return model

    def set_prob_values(self, prob):
        price = self.price
        sim_pars = self.sim_pars
        # Additional parameters
        prob.set_val('price_t', price)
        prob.set_val('G_MW', sim_pars['G_MW'])
        # prob.set_val('pv_deg_per_year', sim_pars['pv_deg_per_year'])
        prob.set_val('peak_hr_quantile', sim_pars['peak_hr_quantile'])
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])

    def list_vars_and_outvars(self, list_vars, list_out_vars):
        added_list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'Revenues [MEuro]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'Shared CAPEX [MEuro]',
            'Shared Opex [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP [GWh]',
            'GUF',
            'grid [MW]',
            'Total curtailment [GWh]',
            'Total curtailment with deg [GWh]',
            'Plant area [km2]',
            'Break-even PPA price [Euro/MWh]',
        ]

        added_list_vars = []

        list_vars = list_vars + added_list_vars
        list_out_vars = list_out_vars + added_list_out_vars

        return list_vars, list_out_vars

    def evalute_set_and_get_prob_values(self, model):
        pass

    def evalute_outputs(self, model):
        pass

    def evaluation_in_csv_design_df(self, model):
        pass
