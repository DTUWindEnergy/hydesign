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
    def __init__(self, excel_data, verbose, **kwargs):
        excel_data_main = excel_data['dashboard']

        # Reset the header to the 6th row (index 5) and reassign the DataFrame
        excel_data_main.columns = excel_data_main.iloc[5]
        excel_data_main = excel_data_main[6:].reset_index(drop=True)

        # Drop rows where 'Parameter' is NaN
        excel_data_main.dropna(subset=['Parameter'], inplace=True)

        # Extract 'Parameter' and 'Value' columns as a dictionary
        sim_pars = dict(zip(excel_data_main['Parameter'], excel_data_main['Value']))

        # Convert specific sim_pars to np.array using ast.literal_eval
        for key in ['wind_deg', 'wind_deg_yr', 'pv_deg', 'pv_deg_yr', 'inflation', 'inflation_yr', 'phasing_yr', 'phasing_CAPEX', 'depreciation_yr',
                    'depreciation', 'technology_types']:
            if key in sim_pars:
                sim_pars[key] = np.array(ast.literal_eval(sim_pars[key]))

        # Overwrite parameters with values from kwargs if provided
        for key in sim_pars.keys():
            if key in kwargs:
                sim_pars[key] = kwargs[key]

        # Extract sim_pars from the dictionary
        latitude = sim_pars.get('latitude')
        longitude = float(sim_pars.get('longitude'))
        altitude = sim_pars.get('altitude')
        work_dir = sim_pars.get('work_dir', './')
        max_num_batteries_allowed = sim_pars.get('max_num_batteries_allowed', 3)
        factor_battery_cost = sim_pars.get('factor_battery_cost', 1)
        ems_type = sim_pars.get('ems_type', 'cplex')
        weeks_per_season_per_year = sim_pars.get('weeks_per_season_per_year')
        seed = sim_pars.get('seed', 0)
        genWT_fn = sim_pars.get('genWT_fn', lut_filepath + './genWT_v3.nc')
        genWake_fn = sim_pars.get('genWake_fn', lut_filepath + './genWake_v3.nc')
        verbose = verbose
        name = sim_pars.get('name', '')
        ppa_price = sim_pars.get('ppa_price')
        ems_type = sim_pars.get('ems_type', 'cplex')
        technology_types = sim_pars.get('technology_types')

        work_dir = mkdir(work_dir)

        # Extract additional data from separate sheets if available
        input_ts_data = excel_data.get('input_ts_DA')

        if altitude is None:
            elevation_fn = sim_pars['elevation_fn']  # Altitude map for extracting altitude
            elevation_ds = xr.open_dataset(elevation_fn)
            altitude = elevation_ds['elev'].interp(
                latitude=latitude,
                longitude=longitude,
                kwargs={"fill_value": 0.0}
            ).values
        if verbose:
            print('\nFixed parameters on the site')
            print('-------------------------------')
            print('longitude =', longitude)
            print('latitude =', latitude)
            print('altitude =', altitude)

        # Parameters of the simulation
        year_start = sim_pars.get('year_start', sim_pars['year'])
        year_end = sim_pars.get('year_end', sim_pars['year'])
        N_life = sim_pars['N_life']
        life_h = N_life * 365 * 24
        G_MW = sim_pars['G_MW']
        battery_depth_of_discharge = sim_pars['battery_depth_of_discharge']
        battery_charge_efficiency = sim_pars['battery_charge_efficiency']
        min_LoH = sim_pars['min_LoH']
        wpp_efficiency = sim_pars['wpp_efficiency']
        land_use_per_solar_MW = sim_pars['land_use_per_solar_MW']

        wind_deg = sim_pars.get('wind_deg', [0, 0.25])
        wind_deg_yr = sim_pars.get('wind_deg_yr', [0, 25])
        share_WT_deg_types = sim_pars.get('share_WT_deg_types', 0.5)

        # Extract weather timeseries
        if input_ts_data.empty:
            # Weather database
            era5_zarr = sim_pars['era5_zarr']  # location of wind speed renalysis
            ratio_gwa_era5 = sim_pars['ratio_gwa_era5']  # location of mean wind speed correction factor
            era5_ghi_zarr = sim_pars['era5_ghi_zarr']  # location of GHI renalysis

            weather = extract_weather_for_HPP(
                longitude=longitude,
                latitude=latitude,
                altitude=altitude,
                era5_zarr=era5_zarr,
                ratio_gwa_era5=ratio_gwa_era5,
                era5_ghi_zarr=era5_ghi_zarr,
                year_start=year_start,
                year_end=year_end
            )

            price_fn = sim_pars.get('price_fn')
            if type(price_fn) is str:
                price = pd.read_csv(price_fn, index_col=0, parse_dates=True)
            else:
                price = price_fn
            try:
                weather['Price'] = price.loc[weather.index].bfill()
            except Exception as e:
                raise Exception('Price timeseries does not match the weather')

            # Check for complete years in the input_ts: Hydesign works with years of 365 days
            N_time = len(weather)
            if np.mod(N_time, 365 * 24) == 0:
                pass
            else:
                N_sel = N_time - np.mod(N_time, 365 * 24)
                weather = weather.iloc[:N_sel]

            input_ts_fn = f'{work_dir}/input_ts{name}.csv'
            if verbose:
                print(f'\ninput_ts_fn extracted and stored in {input_ts_fn}')
            #weather.to_csv(input_ts_fn, index=True)
            N_time = len(weather)

        else:  # User provided weather timeseries
            weather = excel_data['input_ts_DA']
            weather.set_index(input_ts_data.columns[0], inplace=True)
            weather.index = pd.to_datetime(weather.index)
            N_time = len(weather)

        # Check for complete years in the input_ts
        if np.mod(N_time, 365 * 24) == 0:
            pass
        else:
            N_sel = N_time - np.mod(N_time, 365 * 24)
            weather = weather.iloc[:N_sel]

            if verbose:
                print(f'\ninput_ts_fn length is not a complete number of years (hyDesign handles years as 365 days).')
                print(f'The file has been modified accordingly')
            N_time = len(weather)

        # Assign PPA to the full input_ts
        if ppa_price is None:
            price = weather['Price']
        else:
            price = ppa_price * np.ones_like(weather['Price'])

        # Randomly sample the weather to generate representative years
        if weeks_per_season_per_year is not None:
            weather = select_years(
                weather,
                seed=seed,
                weeks_per_season_per_year=weeks_per_season_per_year,
            )
            N_time = len(weather)
            input_ts_fn = f'{work_dir}/input_ts_sel.csv'
            if verbose:
                print(f'\n\nSelected input time series based on {weeks_per_season_per_year} weeks per season are stored in {input_ts_fn}')

        # extract number of ws in the look-up tables
        with xr.open_dataset(genWT_fn) as ds:
            # number of points in the power curves
            N_ws = len(ds.ws.values)


        self.sim_pars = sim_pars
        self.weather = weather
        self.N_time = N_time
        self.genWT_fn = genWT_fn
        self.N_ws = N_ws
        self.genWake_fn = genWake_fn
        self.price = price

    def get_simpars(self):
        return self.sim_pars


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
        model.connect('genericWT.ws', 'genericWake.ws')
        model.connect('genericWT.pc', 'genericWake.pc')
        model.connect('genericWT.ct', 'genericWake.ct')
        model.connect('genericWT.ws', 'wpp.ws')
        model.connect('genericWake.pcw', 'wpp.pcw')
        model.connect('abl.wst', 'wpp.wst')

        model.connect('wpp.wind_t', 'ems.wind_t')
        model.connect('pvp.solar_t', 'ems.solar_t')

        model.connect('ems.b_E_SOC_t', 'battery_degradation.b_E_SOC_t')

        model.connect('battery_degradation.SoH', 'battery_loss_in_capacity_due_to_temp.SoH')
        model.connect('battery_loss_in_capacity_due_to_temp.SoH_all', 'ems_long_term_operation.SoH')

        model.connect('genericWT.ws', 'wpp_with_degradation.ws')
        model.connect('genericWake.pcw', 'wpp_with_degradation.pcw')
        model.connect('abl.wst', 'wpp_with_degradation.wst')
        model.connect('wpp_with_degradation.wind_t_ext_deg', 'ems_long_term_operation.wind_t_ext_deg')

        model.connect('ems.solar_t_ext', 'pvp_with_degradation.solar_t_ext')
        model.connect('pvp_with_degradation.solar_t_ext_deg', 'ems_long_term_operation.solar_t_ext_deg')

        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('ems.b_E_SOC_t', 'ems_long_term_operation.b_E_SOC_t')
        model.connect('ems.b_t', 'ems_long_term_operation.b_t')

        model.connect('wpp.wind_t', 'wpp_cost.wind_t')

        model.connect('battery_degradation.SoH', 'battery_cost.SoH')

        model.connect('pvp.Apvp', 'shared_cost.Apvp')

        model.connect('wpp_cost.CAPEX_w', 'finance.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance.OPEX_w')

        model.connect('pvp_cost.CAPEX_s', 'finance.CAPEX_s')
        model.connect('pvp_cost.OPEX_s', 'finance.OPEX_s')

        model.connect('battery_cost.CAPEX_b', 'finance.CAPEX_b')
        model.connect('battery_cost.OPEX_b', 'finance.OPEX_b')

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
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])
        prob.set_val('peak_hr_quantile', sim_pars['peak_hr_quantile'])
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])
        prob.set_val('min_LoH', sim_pars['min_LoH'])
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])

    def list_vars_and_outvars(self, list_vars, list_out_vars):
        list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'Revenues [MEuro]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'Wind CAPEX [MEuro]',
            'Wind OPEX [MEuro]',
            'PV CAPEX [MEuro]',
            'PV OPEX [MEuro]',
            'Batt CAPEX [MEuro]',
            'Batt OPEX [MEuro]',
            'Shared CAPEX [MEuro]',
            'Shared Opex [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP [GWh]',
            'GUF',
            'grid [MW]',
            'wind [MW]',
            'solar [MW]',
            'Battery Energy [MWh]',
            'Battery Power [MW]',
            'Total curtailment [GWh]',
            'Total curtailment with deg [GWh]',
            'Awpp [km2]',
            'Apvp [km2]',
            'Plant area [km2]',
            'Rotor diam [m]',
            'Hub height [m]',
            'Number of batteries used in lifetime',
            'Break-even PPA price [Euro/MWh]',
            'Capacity factor wind [-]'
        ]

        list_vars = [
            'clearance [m]',
            'sp [W/m2]',
            'p_rated [MW]',
            'Nwt',
            'wind_MW_per_km2 [MW/km2]',
            'solar_MW [MW]',
            'surface_tilt [deg]',
            'surface_azimuth [deg]',
            'DC_AC_ratio',
            'b_P [MW]',
            'b_E_h [h]',
            'cost_of_battery_P_fluct_in_peak_price_ratio'
        ]

        return list_vars, list_out_vars

    def evalute_set_and_get_prob_values(self, model):
        pass

    def evalute_outputs(self, model):
        pass

    def evaluation_in_csv_design_df(self, model):
        pass
