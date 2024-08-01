# %%
import glob
import os
import time
import ast

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
# import seaborn as sns
import openmdao.api as om
import yaml
import scipy as sp
from scipy import stats
import xarray as xr

from hydesign.weather.weather import extract_weather_for_HPP, ABL, select_years
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, wpp_with_degradation, get_rotor_area, get_rotor_d
from hydesign.pv.pv import pvp, pvp_with_degradation
from hydesign.ems.ems import ems, ems_long_term_operation
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance.finance import finance
from hydesign.look_up_tables import lut_filepath


class hpp_model:
    """HPP design evaluator"""

    def __init__(self, excel_file_path, verbose=True):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        latitude : Latitude at chosen location
        longitude : Longitude at chosen location
        altitude : Altitude at chosen location, if not provided, elevation is calculated using elevation map datasets
        sims_pars_fn : Case study input values of the HPP 
        work_dir : Working directory path
        max_num_batteries_allowed : Maximum number of batteries allowed including start and replacements
        weeks_per_season_per_year: Number of weeks per season to select from the input data, to reduce computation time. Default is `None` which uses all the input time series
        seed: seed number for week selection
        ems_type : Energy management system optimization type: cplex solver or rule based
        inputs_ts_fn : User provided weather timeseries, if not provided, the weather data is calculated using ERA5 datasets
        price_fn : Price timeseries
        era5_zarr : Location of wind speed renalysis
        ratio_gwa_era5 : Location of mean wind speed correction factor
        era5_ghi_zarr : Location of GHI renalysis
        elevation_fn : Location of GHI renalysis
        genWT_fn : Wind turbine power curve look-up tables
        genWake_fn : Wind turbine wake look-up tables
        """

        excel_data = pd.read_excel(excel_file_path, sheet_name=None)
        self.excel_data = excel_data
        self.excel_file_path = excel_file_path

        excel_data_main = excel_data['dashboard']

        # Reset the header to the 6th row (index 5) and reassign the DataFrame
        excel_data_main.columns = excel_data_main.iloc[5]
        excel_data_main = excel_data_main[6:].reset_index(drop=True)

        # Drop rows where 'Parameter' is NaN
        excel_data_main.dropna(subset=['Parameter'], inplace=True)

        # Extract 'Parameter' and 'Value' columns as a dictionary
        parameters = dict(zip(excel_data_main['Parameter'], excel_data_main['Value']))

        # Convert specific parameters to np.array using ast.literal_eval
        parameters['wind_deg'] = np.array(ast.literal_eval(parameters['wind_deg']))
        parameters['wind_deg_yr'] = np.array(ast.literal_eval(parameters['wind_deg_yr']))
        parameters['pv_deg'] = np.array(ast.literal_eval(parameters['pv_deg']))
        parameters['pv_deg_yr'] = np.array(ast.literal_eval(parameters['pv_deg_yr']))
        parameters['inflation'] = np.array(ast.literal_eval(parameters['inflation']))
        parameters['inflation_yr'] = np.array(ast.literal_eval(parameters['inflation_yr']))
        parameters['phasing_yr'] = np.array(ast.literal_eval(parameters['phasing_yr']))
        parameters['phasing_CAPEX'] = np.array(ast.literal_eval(parameters['phasing_CAPEX']))
        parameters['depreciation_yr'] = np.array(ast.literal_eval(parameters['depreciation_yr']))
        parameters['depreciation'] = np.array(ast.literal_eval(parameters['depreciation']))

        # Extract parameters from the dictionary
        latitude = parameters.get('latitude')
        longitude = float(parameters.get('longitude'))
        altitude = parameters.get('altitude')
        work_dir = parameters.get('work_dir', './')
        max_num_batteries_allowed = parameters.get('max_num_batteries_allowed', 3)
        factor_battery_cost = parameters.get('factor_battery_cost', 1)
        ems_type = parameters.get('ems_type', 'cplex')
        weeks_per_season_per_year = parameters.get('weeks_per_season_per_year')
        seed = parameters.get('seed', 0)
        genWT_fn = parameters.get('genWT_fn', lut_filepath + 'genWT_v3.nc')
        genWake_fn = parameters.get('genWake_fn', lut_filepath + 'genWake_v3.nc')
        verbose = verbose
        name = parameters.get('name', '')
        ppa_price = parameters.get('ppa_price', 0)

        work_dir = mkdir(work_dir)

        # Extract additional data from separate sheets if available
        input_ts_data = excel_data.get('input_ts_DA')
        input_ts_fn = 'input_ts_data.csv'  # Define the path where you want to save it
        input_ts_data.to_csv(input_ts_fn, index=False)
        price_data = excel_data.get('2030_EL_PRICE')
        
        if altitude == None:
            
            elevation_fn = parameters['elevation_fn'] # Altitude map for extracting altitude
            elevation_ds = xr.open_dataset(elevation_fn)
            altitude = elevation_ds['elev'].interp(
                                latitude=latitude,
                                longitude=longitude,
                                kwargs={"fill_value": 0.0}
                            ).values
        if verbose:
            print('\nFixed parameters on the site')
            print('-------------------------------')
            print('longitude =',longitude)
            print('latitude =',latitude)
            print('altitude =',altitude)
        
        # Parameters of the simulation
        if 'year_start' in parameters.keys():
            year_start = parameters['year_start']
        else:
            year_start = parameters['year']


        if 'year_end' in parameters.keys():
            year_end = parameters['year_end']
        else:
            year_end = parameters['year']
            
        N_life = parameters['N_life']
        life_h = N_life*365*24
        G_MW = parameters['G_MW']
        battery_depth_of_discharge = parameters['battery_depth_of_discharge']
        battery_charge_efficiency = parameters['battery_charge_efficiency']
        min_LoH = parameters['min_LoH']
        #pv_deg_per_year = parameters['pv_deg_per_year']
        wpp_efficiency = parameters['wpp_efficiency']
        land_use_per_solar_MW = parameters['land_use_per_solar_MW']
        
        if 'wind_deg' in parameters:
            wind_deg = parameters['wind_deg']
            wind_deg_yr = parameters['wind_deg_yr']
            share_WT_deg_types = parameters['share_WT_deg_types']
        else:
            wind_deg = [0, 0.25]
            wind_deg_yr = [0, 25]
            share_WT_deg_types = 0.5
        
        # Extract weather timeseries
        if input_ts_data.empty:
            
            # Weather database
            era5_zarr = parameters['era5_zarr'] # location of wind speed renalysis
            ratio_gwa_era5 = parameters['ratio_gwa_era5'] # location of mean wind speed correction factor
            era5_ghi_zarr = parameters['era5_ghi_zarr'] # location of GHI renalysis

            weather = extract_weather_for_HPP(
                longitude = longitude, 
                latitude = latitude,
                altitude = altitude,
                era5_zarr = era5_zarr,
                ratio_gwa_era5 = ratio_gwa_era5,
                era5_ghi_zarr = era5_ghi_zarr,
                year_start = year_start,
                year_end = year_end)
                
            if type(price_fn) is str:
                price = pd.read_csv(price_fn, index_col=0, parse_dates=True)
            else:
                price = price_fn
            try:
                weather['Price'] = price.loc[weather.index].bfill()
            except:
                raise('Price timeseries does not match the weather')

            # Check for complete years in the input_ts: Hydesign works with years of 365 days
            N_time = len(weather)
            if np.mod(N_time,365)/24 == 0:
                pass
            else:
                N_sel = N_time - np.mod(N_time,365)
                weather = weather.iloc[:N_sel]
            
            input_ts_fn = f'{work_dir}input_ts{name}.csv'
            print(f'\ninput_ts_fn extracted and stored in {input_ts_fn}')
            weather.to_csv(input_ts_fn)
            N_time = len(weather)
            
        else: # User provided weather timeseries
            weather= excel_data['input_ts_DA']
            weather.set_index(input_ts_data.columns[0], inplace=True)
            weather.index = pd.to_datetime(weather.index)
            # weather = pd.read_csv(r"C:\Users\amia\Git_projects\hydesign\hydesign\examples\Europe\GWA2\input_ts_Denmark_good_wind.csv", index_col=0, parse_dates=True)
            N_time = len(weather)



        # Check for complete years in the input_ts
        if np.mod(N_time,365)/24 == 0:
            pass
        else:
            N_sel = N_time - np.mod(N_time,365)
            weather = weather.iloc[:N_sel]
            
            input_ts_fn = f'{work_dir}input_ts_modified.csv'
            print(f'\ninput_ts_fn length is not a complete number of years (hyDesign handles years as 365 days).')
            print(f'The file has been modified and stored in {input_ts_fn}')
            weather.to_csv(input_ts_fn)
            N_time = len(weather)

        # Assign PPA to the full input_ts
        if ppa_price is None:
            price = weather['Price']
        else:
            price = ppa_price * np.ones_like(weather['Price'])

        # Randomly sample the weather to generate representative years
        if weeks_per_season_per_year != None:
            weather = select_years(
                weather,
                seed=seed,
                weeks_per_season_per_year=weeks_per_season_per_year,
            )
            N_time = len(weather)
            input_ts_fn = f'{work_dir}input_ts_sel.csv'
            print(f'\n\nSelected input time series based on {weeks_per_season_per_year} weeks per season are stored in {input_ts_fn}')
            weather.to_csv(input_ts_fn)

        # extract number of ws in the look-up tables
        with xr.open_dataset(genWT_fn) as ds: 
            # number of points in the power curves
            N_ws = len(ds.ws.values)
        
        model = om.Group()
        
        model.add_subsystem(
            'abl', 
            ABL(
                weather_fn=input_ts_fn, 
                N_time=N_time),
            promotes_inputs=['hh']
            )
        model.add_subsystem(
            'genericWT', 
            genericWT_surrogate(
                genWT_fn=genWT_fn,
                N_ws = N_ws),
            promotes_inputs=[
               'hh',
               'd',
               'p_rated',
            ])
        
        model.add_subsystem(
            'genericWake', 
            genericWake_surrogate(
                genWake_fn=genWake_fn,
                N_ws = N_ws),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'd',
                'p_rated',
                ])
        
        model.add_subsystem(
            'wpp', 
            wpp(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,)
                )
        
        model.add_subsystem(
            'pvp', 
            pvp(
                weather_fn = input_ts_fn, 
                N_time = N_time,
                latitude = latitude,
                longitude = longitude,
                altitude = altitude,
                tracking = parameters['tracking']
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
                N_time = N_time,
                weeks_per_season_per_year = weeks_per_season_per_year,
                life_h = life_h, 
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
                weather_fn = input_ts_fn, # for extracting temperature
                num_batteries = max_num_batteries_allowed,
                life_h = life_h,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            promotes_inputs=[
                'min_LoH'
                ])

        model.add_subsystem(
            'battery_loss_in_capacity_due_to_temp', 
            battery_loss_in_capacity_due_to_temp(
                weather_fn = input_ts_fn, # for extracting temperature
                life_h = life_h,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            )

        model.add_subsystem(
            'wpp_with_degradation', 
            wpp_with_degradation(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,
                life_h = life_h,
                wind_deg_yr = wind_deg_yr,
                wind_deg = wind_deg,
                share_WT_deg_types = share_WT_deg_types,
                weeks_per_season_per_year = weeks_per_season_per_year,
                
            )
        )

        model.add_subsystem(
            'pvp_with_degradation', 
            pvp_with_degradation(
                life_h = life_h,
                pv_deg_yr = parameters['pv_deg_yr'],
                pv_deg = parameters['pv_deg'],
            )
        )
        
        
        model.add_subsystem(
            'ems_long_term_operation', 
            ems_long_term_operation(
                N_time = N_time,
                life_h = life_h),
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
                wind_turbine_cost=parameters['wind_turbine_cost'],
                wind_civil_works_cost=parameters['wind_civil_works_cost'],
                wind_fixed_onm_cost=parameters['wind_fixed_onm_cost'],
                wind_variable_onm_cost=parameters['wind_variable_onm_cost'],
                d_ref=parameters['d_ref'],
                hh_ref=parameters['hh_ref'],
                p_rated_ref=parameters['p_rated_ref'],
                N_time = N_time, 
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
                solar_PV_cost=parameters['solar_PV_cost'],
                solar_hardware_installation_cost=parameters['solar_hardware_installation_cost'],
                solar_inverter_cost=parameters['solar_inverter_cost'],
                solar_fixed_onm_cost=parameters['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=factor_battery_cost*parameters['battery_energy_cost'],
                battery_power_cost=factor_battery_cost*parameters['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=factor_battery_cost*parameters['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=factor_battery_cost*parameters['battery_control_system_cost'],
                battery_energy_onm_cost=factor_battery_cost*parameters['battery_energy_onm_cost'],
                N_life = N_life,
                life_h = life_h
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'battery_price_reduction_per_year'])

        model.add_subsystem(
            'shared_cost',
            shared_cost(
                hpp_BOS_soft_cost=parameters['hpp_BOS_soft_cost'],
                hpp_grid_connection_cost=parameters['hpp_grid_connection_cost'],
                land_cost=parameters['land_cost'],
            ),
            promotes_inputs=[
                'G_MW',
                'Awpp',
            ])

        model.add_subsystem(
            'finance', 
            finance(
                N_time = N_time, 
                # Depreciation curve
                depreciation_yr = parameters['depreciation_yr'],
                depreciation = parameters['depreciation'],
                # Inflation curve
                inflation_yr = parameters['inflation_yr'],
                inflation = parameters['inflation'],
                ref_yr_inflation = parameters['ref_yr_inflation'],
                # Early paying or CAPEX Phasing
                phasing_yr = parameters['phasing_yr'],
                phasing_CAPEX = parameters['phasing_CAPEX'],
                life_h = life_h),
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

        model.connect('ems.solar_t_ext','pvp_with_degradation.solar_t_ext')
        model.connect('pvp_with_degradation.solar_t_ext_deg', 'ems_long_term_operation.solar_t_ext_deg')
        
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('ems.b_E_SOC_t', 'ems_long_term_operation.b_E_SOC_t')
        model.connect('ems.b_t', 'ems_long_term_operation.b_t')

        model.connect('wpp.wind_t', 'wpp_cost.wind_t')
        
        model.connect('battery_degradation.SoH','battery_cost.SoH')
        
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
        
        prob = om.Problem(
            model,
            reports=None
        )

        prob.setup()        
        
        # Additional parameters
        prob.set_val('price_t', price)
        prob.set_val('G_MW', parameters['G_MW'])
        #prob.set_val('pv_deg_per_year', parameters['pv_deg_per_year'])
        prob.set_val('battery_depth_of_discharge', parameters.get('battery_depth_of_discharge', 0.9))
        prob.set_val('battery_charge_efficiency', parameters.get('battery_charge_efficiency', 0.985))
        prob.set_val('peak_hr_quantile', parameters.get('peak_hr_quantile', 0.9) )
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     parameters.get('n_full_power_hours_expected_per_day_at_peak_price', 0))
        prob.set_val('min_LoH', parameters.get('min_LoH', 0.7))
        prob.set_val('wind_WACC', parameters.get('wind_WACC', 0.06))
        prob.set_val('solar_WACC', parameters.get('solar_WACC', 0.06))
        prob.set_val('battery_WACC', parameters.get('battery_WACC', 0.06))
        prob.set_val('tax_rate', parameters.get('tax_rate', 0.22))
        prob.set_val('land_use_per_solar_MW', parameters.get('land_use_per_solar_MW', 0.01226))

        self.parameters = parameters
        self.prob = prob
        self.max_num_batteries_allowed = max_num_batteries_allowed
        self.input_ts_fn = input_ts_fn
        self.altitude = altitude
    
        self.list_out_vars = [
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

        self.list_vars = [
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
    
    
    def evaluate(
        self,
        # Wind plant design
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio
        ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        clearance : Distance from the ground to the tip of the blade [m]
        sp : Specific power of the turbine [W/m2] 
        p_rated : Rated powe of the turbine [MW] 
        Nwt : Number of wind turbines
        wind_MW_per_km2 : Wind power installation density [MW/km2]
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['mean_AEP']/(self.parameters['G_MW']*365*24) : Grid utilization factor
        self.parameters['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        self.num_batteries : Number of allowed replacements of the battery
        """
        # Ensure weather data is correct
        if 'weather' in self.__dict__:
            weather = self.weather
            print("Weather DataFrame:")
            print(weather.head())
            print(weather.dtypes)

            # Convert relevant columns to numeric types if necessary
            numeric_columns = ['temperature', 'pressure', 'wind_speed']  # Add all relevant columns
            for column in numeric_columns:
                weather[column] = pd.to_numeric(weather[column], errors='coerce')
                if weather[column].isnull().any():
                    print(f"Warning: NaN values found in column {column} after conversion.")

        prob = self.prob

        d = get_rotor_d(p_rated*1e6/sp)
        hh = (d/2)+clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2 
        #Awpp = Awpp + 1e-10*(Awpp==0)
        b_E = b_E_h * b_P
        
        # pass design variables        
        prob.set_val('hh', hh)
        prob.set_val('d', d)
        prob.set_val('p_rated', p_rated)
        prob.set_val('Nwt', Nwt)
        prob.set_val('Awpp', Awpp)

        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)
        
        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)        
        
        prob.run_model()
        
        self.prob = prob
        
        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = prob.get_val('wpp_with_degradation.wind_t_ext_deg').mean() / p_rated / Nwt  # Capacity factor of wind only

        # Collect the results in a DataFrame
        results = pd.DataFrame({
            'NPV_over_CAPEX': [prob['NPV_over_CAPEX']],
            'NPV': [prob['NPV'] / 1e6],
            'IRR': [prob['IRR']],
            'LCOE': [prob['LCOE']],
            'revenues': [prob['revenues'] / 1e6],
            'CAPEX': [prob['CAPEX'] / 1e6],
            'OPEX': [prob['OPEX'] / 1e6],
            'CAPEX_w': [prob.get_val('finance.CAPEX_w') / 1e6],
            'OPEX_w': [prob.get_val('finance.OPEX_w') / 1e6],
            'CAPEX_s': [prob.get_val('finance.CAPEX_s') / 1e6],
            'OPEX_s': [prob.get_val('finance.OPEX_s') / 1e6],
            'CAPEX_b': [prob.get_val('finance.CAPEX_b') / 1e6],
            'OPEX_b': [prob.get_val('finance.OPEX_b') / 1e6],
            'CAPEX_el': [prob.get_val('finance.CAPEX_el') / 1e6],
            'OPEX_el': [prob.get_val('finance.OPEX_el') / 1e6],
            'penalty_lifetime': [prob['penalty_lifetime'] / 1e6],
            'mean_AEP': [prob['mean_AEP'] / 1e3],  # [GWh]
            'grid_utilization_factor': [prob['mean_AEP'] / (self.parameters['G_MW'] * 365 * 24)],
            'grid_connection': [self.parameters['G_MW']],
            'wind_MW': [wind_MW],
            'solar_MW': [solar_MW],
            'battery_energy': [b_E],
            'battery_power': [b_P],
            'total_curtailment': [prob['total_curtailment'] / 1e3],  # [GWh]
            'total_curtailment_with_deg': [prob['total_curtailment_with_deg'] / 1e3],  # [GWh]
            'Awpp': [Awpp],
            'Apvp': [prob.get_val('shared_cost.Apvp')],
            'max_Awpp_Apvp': [max(Awpp, prob.get_val('shared_cost.Apvp'))],
            'rotor_diameter': [d],
            'hub_height': [hh],
            'num_batteries': [prob.get_val('battery_degradation.n_batteries') * (b_P > 0)],
            'break_even_PPA_price': [prob['break_even_PPA_price']],
            'cf_wind': [cf_wind],
        })

        # Determine the value in cell C5
        output_option = self.excel_data['dashboard'].iloc[3, 2]  # Cell C5 corresponds to row 4 (0-based index) and column 2 (0-based index)
        print(f"Output option in C5: {output_option}")

        # Function to save output to a separate file
        def save_output_to_separate_file(output_data, output_file_path):
            output_data.to_excel(output_file_path, sheet_name='output_data', index=False)
            print(f"Output saved to a separate file: {output_file_path}")

        # Function to save output to the same file
        def save_output_to_same_file(output_data, file_path, new_sheet_name, reopen=True):
            def close_excel_process():
                try:
                    os.system("taskkill /f /im EXCEL.EXE")
                except Exception as e:
                    print(f"Error closing the Excel process: {e}")

            # Ensure the Excel file is closed
            close_excel_process()

            # Give the system a moment to close Excel
            time.sleep(2)
            #
            # # Write the output data to the Excel file
            # output_data.to_excel(file_path, sheet_name=new_sheet_name, index=False)
            #
            # print(f"Output saved to the same file in sheet: {new_sheet_name}")
            #
            # # Optionally, reopen the Excel file
            # if reopen:
            #     os.system(f'start EXCEL.EXE {file_path}')

        # Check the value in C5 and save accordingly
        if output_option == "Outputs in a seperate excel file":
            output_file_path = 'output_file.xlsx'
            save_output_to_separate_file(results, output_file_path)
        elif output_option == "Outputs in this excel file (The excel file will be saved and closed!)":
            save_output_to_same_file(results, self.excel_file_path, 'output_data')
        else:
            print(f"Unrecognized option in C5: {output_option}")

        return np.hstack([
            prob['NPV_over_CAPEX'], 
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['revenues']/1e6,
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob.get_val('finance.CAPEX_w')/1e6,
            prob.get_val('finance.OPEX_w')/1e6,
            prob.get_val('finance.CAPEX_s')/1e6,
            prob.get_val('finance.OPEX_s')/1e6,
            prob.get_val('finance.CAPEX_b')/1e6,
            prob.get_val('finance.OPEX_b')/1e6,
            prob.get_val('finance.CAPEX_el')/1e6,
            prob.get_val('finance.OPEX_el')/1e6,
            prob['penalty_lifetime']/1e6,
            prob['mean_AEP']/1e3, #[GWh]
            # Grid Utilization factor
            prob['mean_AEP']/(self.parameters['G_MW']*365*24),
            self.parameters['G_MW'],
            wind_MW,
            solar_MW,
            b_E,
            b_P,
            prob['total_curtailment']/1e3, #[GWh]
            prob['total_curtailment_with_deg']/1e3, #[GWh]
            Awpp,
            prob.get_val('shared_cost.Apvp'),
            max( Awpp , prob.get_val('shared_cost.Apvp') ),
            d,
            hh,
            prob.get_val('battery_degradation.n_batteries') * (b_P>0),
            prob['break_even_PPA_price'],
            cf_wind,
            ])
    
    def print_design(self, x_opt, outs):
        print() 
        print('Design:') 
        print('---------------') 

        for i_v, var in enumerate(self.list_vars):
                print(f'{var}: {x_opt[i_v]:.3f}')
        print()    
        print()
        for i_v, var in enumerate(self.list_out_vars):
            print(f'{var}: {outs[i_v]:.3f}')
        print()


    def evaluation_in_csv(self, name_file ,longitude, latitude, altitude, x_opt, outs ):
        design_df = pd.DataFrame(columns = ['longitude',
                                            'latitude',
                                            'altitude',
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
                                            'cost_of_battery_P_fluct_in_peak_price_ratio',
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
                                            'Shared OPEX [MEuro]',
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
                                            ]  , index=range(1))
        design_df.iloc[0] =  [longitude,latitude,altitude] + list(x_opt) + list(outs)
        design_df.to_csv(f'{name_file}.csv')
        
    
# -----------------------------------------------------------------------
# Auxiliar functions for ems modelling
# -----------------------------------------------------------------------
    
def mkdir(dir_):
    if str(dir_).startswith('~'):
        dir_ = str(dir_).replace('~', os.path.expanduser('~'))
    try:
        os.stat(dir_)
    except BaseException:
        try:
            os.mkdir(dir_)
            #Path(dir_).mkdir(parents=True, exist_ok=True)
        except BaseException:
            pass
    return dir_

