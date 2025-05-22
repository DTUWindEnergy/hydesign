# %%
import os

# basic libraries
import numpy as np
import pandas as pd
import openmdao.api as om
import yaml
import xarray as xr
import datetime

from hydesign.weather.weather import extract_weather_for_HPP, ABL, select_years
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, wpp_with_degradation, get_rotor_d  # , get_rotor_area
from hydesign.pv.pv import pvp, pvp_with_degradation
from hydesign.ems.ems import ems, ems_long_term_operation
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance.finance import finance
from hydesign.look_up_tables import lut_filepath
from hydesign.reliability import battery_with_reliability, wpp_with_reliability, pvp_with_reliability

class hpp_base:
    def __init__(self,
                 sim_pars_fn,
                 defaults={},
                 **kwargs
                 ):
        self.sim_pars_fn = sim_pars_fn

        # Extract simulation parameters:
        # First gets default values, then updates with values provided in yml-file and finally updating with values provided in instantiation
        sim_pars = self.get_defaults()
        sim_pars.update(defaults)
        with open(sim_pars_fn) as file:
            sim_pars.update(yaml.load(file, Loader=yaml.FullLoader))
        sim_pars.update(kwargs)
        self.check_inputs(sim_pars)
        
        # Combine Wind farm surrogate directory and file paths
        genWT_fn = os.path.join(sim_pars['gen_lut_dir'], sim_pars['genWT_fn'])
        genWake_fn = os.path.join(sim_pars['gen_lut_dir'], sim_pars['genWake_fn'])
        sim_pars['genWT_fn'] = genWT_fn
        sim_pars['genWake_fn'] = genWake_fn

        work_dir = sim_pars['work_dir']
        altitude = sim_pars['altitude']
        latitude = sim_pars['latitude']
        longitude = sim_pars['longitude']
        verbose = sim_pars['verbose']
        input_ts_fn = sim_pars['input_ts_fn']
        price_fn = sim_pars['price_fn']
        name = sim_pars['name']
        ppa_price = sim_pars['ppa_price']
        weeks_per_season_per_year = sim_pars['weeks_per_season_per_year']
        seed = sim_pars['seed']
        genWT_fn = sim_pars['genWT_fn']
        max_num_batteries_allowed = sim_pars['max_num_batteries_allowed']
            
        work_dir = mkdir(work_dir)
        
        if altitude == None:
            
            elevation_fn = sim_pars['elevation_fn'] # Altitude map for extracting altitude
            elevation_ds = xr.open_dataset(elevation_fn, engine='h5netcdf')
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
        if 'year_start' in sim_pars.keys():
            year_start = sim_pars['year_start']
        else:
            year_start = sim_pars['year']


        if 'year_end' in sim_pars.keys():
            year_end = sim_pars['year_end']
        else:
            year_end = sim_pars['year']
        if 'life_y' in sim_pars:
            life_y = sim_pars['life_y']
        elif 'N_life' in sim_pars:
            life_y = sim_pars['N_life']
        
        if 'wind_deg' in sim_pars:
            wind_deg = sim_pars['wind_deg']
        else:
            wind_deg = [0, 0]
        if 'wind_deg_yr' in sim_pars:
            wind_deg_yr = sim_pars['wind_deg_yr']
        else:
            wind_deg_yr = [0, 25]
        if 'share_WT_deg_types' in sim_pars:
            share_WT_deg_types = sim_pars['share_WT_deg_types']
        else:
            share_WT_deg_types = 0.5
        
        # Extract weather timeseries
        if input_ts_fn == None:
            
            # Weather database
            era5_zarr = sim_pars['era5_zarr'] # location of wind speed renalysis
            ratio_gwa_era5 = sim_pars['ratio_gwa_era5'] # location of mean wind speed correction factor
            era5_ghi_zarr = sim_pars['era5_ghi_zarr'] # location of GHI renalysis
            
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
            weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
            N_time = len(weather)

        # Check for complete years in the input_ts
        if np.mod(N_time,365)/24 == 0:
            pass
        else:
            N_sel = N_time - np.mod(N_time,365)
            weather = weather.iloc[:N_sel]
            
            input_ts_fn = f'{work_dir}input_ts_modified.csv'
            print('\ninput_ts_fn length is not a complete number of years (hyDesign handles years as 365 days).')
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
        with xr.open_dataset(genWT_fn, engine='h5netcdf') as ds: 
            # number of points in the power curves
            self.N_ws = len(ds.ws.values)
            
        sim_pars['time_str'] = datetime.datetime.now().strftime('%y_%m_%H_%M_%S')
        
        self.N_time = N_time
        self.sim_pars = sim_pars
        self.wind_deg = wind_deg
        self.wind_deg_yr = wind_deg_yr
        self.share_WT_deg_types = share_WT_deg_types
        # self.N_life = N_life
        self.price = price
        self.wpp_efficiency = sim_pars['wpp_efficiency']
        # self.life_h = N_life*365*24
        self.max_num_batteries_allowed = max_num_batteries_allowed
        self.input_ts_fn = input_ts_fn
        self.longitude = longitude
        self.latitude = latitude
        self.altitude = altitude
        self.life_y = life_y
        
    def get_defaults(self):
        return dict(work_dir = './',
                    max_num_batteries_allowed = 3,
                    ems_type='cplex',
                    weeks_per_season_per_year = None,
                    seed=0,
                    input_ts_fn = None, 
                    price_fn = None, 
                    gen_lut_dir = lut_filepath,
                    genWT_fn = 'genWT_v3.nc',
                    genWake_fn = 'genWake_v3.nc',
                    verbose = True,
                    name = '',
                    ppa_price=None,
                    reliability_ts_battery=None,
                    reliability_ts_trans=None,
                    reliability_ts_wind=None,
                    reliability_ts_pv=None,
                    save_finance_ts=False,)
    
    def check_inputs(self, sim_pars):
        req_vars = ['altitude', 'latitude', 'longitude']
        for var in req_vars:
            if var not in sim_pars:
                raise ValueError(f"variable: '{var}' must be provided either in input yml file or when instantiating HPP model")
            else:
                if sim_pars[var] is None:
                    raise ValueError(f"variable: '{var}' cannot be provided as None")
    
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
        
    def evaluation_in_csv(self, name_file ,longitude, latitude, altitude, x_opt, outs):
        design_df = pd.DataFrame(columns = ['longitude',
                                            'latitude',
                                            'altitude',] + self.list_vars + self.list_out_vars, index=range(1))
        design_df.iloc[0] =  [longitude,latitude,altitude] + list(x_opt) + list(outs)
        design_df.to_csv(f'{name_file}.csv')
        


class hpp_model(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
        **kwargs
        ):
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
        hpp_base.__init__(self,
                          sim_pars_fn=sim_pars_fn,
                          **kwargs
                          )

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        # life_h = self.life_h
        life_y = self.life_y
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        # N_life = self.N_life
        price = self.price
        
        input_ts_fn = sim_pars['input_ts_fn']
        genWT_fn = sim_pars['genWT_fn']
        genWake_fn = sim_pars['genWake_fn']
        latitude = sim_pars['latitude']
        longitude = sim_pars['longitude']
        altitude = sim_pars['altitude']
        weeks_per_season_per_year = sim_pars['weeks_per_season_per_year']
        ems_type = sim_pars['ems_type']
        max_num_batteries_allowed = sim_pars['max_num_batteries_allowed']
        reliability_ts_battery = sim_pars['reliability_ts_battery']
        reliability_ts_trans = sim_pars['reliability_ts_trans']
        reliability_ts_wind = sim_pars['reliability_ts_wind']
        reliability_ts_pv = sim_pars['reliability_ts_pv']
        battery_price_reduction_per_year = sim_pars['battery_price_reduction_per_year']
        work_dir = sim_pars['work_dir']

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
                tracking = sim_pars['tracking']
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
                life_y = life_y, 
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
                life_y = life_y,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            promotes_inputs=[
                'min_LoH'
                ])

        model.add_subsystem(
            'battery_loss_in_capacity_due_to_temp', 
            battery_loss_in_capacity_due_to_temp(
                weather_fn = input_ts_fn, # for extracting temperature
                life_y = life_y,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            )

        model.add_subsystem(
            'wpp_with_degradation', 
            wpp_with_degradation(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,
                life_y = life_y,
                wind_deg_yr = wind_deg_yr,
                wind_deg = wind_deg,
                share_WT_deg_types = share_WT_deg_types,
                weeks_per_season_per_year = weeks_per_season_per_year,
                
            )
        )

        model.add_subsystem(
            'pvp_with_degradation', 
            pvp_with_degradation(
                life_y = life_y,
                pv_deg_yr = sim_pars['pv_deg_yr'],
                pv_deg = sim_pars['pv_deg'],
                )
            )

        
        model.add_subsystem(
            'battery_with_reliability', 
            battery_with_reliability(
                life_y = life_y,
                reliability_ts_battery=reliability_ts_battery,
                reliability_ts_trans=reliability_ts_trans,
                ),
            )        
        

        model.add_subsystem(
            'wpp_with_reliability', 
            wpp_with_reliability(
                life_y = life_y,
                reliability_ts_wind=reliability_ts_wind,
                reliability_ts_trans=reliability_ts_trans,
                ),
            # promotes_inputs=['Nwt',]
            )        
        

        model.add_subsystem(
            'pvp_with_reliability', 
            pvp_with_reliability(
                life_y = life_y,
                reliability_ts_pv=reliability_ts_pv,
                # reliability_ts_inv_fn=reliability_ts_inv_fn,
                reliability_ts_trans=reliability_ts_trans,
                ),
            # promotes_inputs=['solar_MW',]
            )        
        
        model.add_subsystem(
            'ems_long_term_operation', 
            ems_long_term_operation(
                N_time = N_time,
                life_y = life_y),
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
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_inverter_cost=sim_pars['solar_inverter_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=sim_pars['battery_energy_cost'],
                battery_power_cost=sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=sim_pars['battery_energy_onm_cost'],
                # N_life = N_life,
                life_y = life_y,
                battery_price_reduction_per_year = battery_price_reduction_per_year,

            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                ])

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
                N_time = N_time, 
                # Depreciation curve
                depreciation_yr = sim_pars['depreciation_yr'],
                depreciation = sim_pars['depreciation'],
                # Inflation curve
                inflation_yr = sim_pars['inflation_yr'],
                inflation = sim_pars['inflation'],
                ref_yr_inflation = sim_pars['ref_yr_inflation'],
                # Early paying or CAPEX Phasing
                phasing_yr = sim_pars['phasing_yr'],
                phasing_CAPEX = sim_pars['phasing_CAPEX'],
                life_y = life_y,
                save_finance_ts = sim_pars['save_finance_ts'],
                work_dir = work_dir,
                time_str = sim_pars['time_str']),
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
        model.connect('wpp_with_degradation.wind_t_ext_deg', 'wpp_with_reliability.wind_t')
        model.connect('wpp_with_reliability.wind_t_rel', 'ems_long_term_operation.wind_t_ext_deg')

        model.connect('ems.solar_t_ext','pvp_with_degradation.solar_t_ext')
        model.connect('pvp_with_degradation.solar_t_ext_deg', 'pvp_with_reliability.solar_t')
        model.connect('pvp_with_reliability.solar_t_rel', 'ems_long_term_operation.solar_t_ext_deg')
        
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('ems.b_E_SOC_t', 'ems_long_term_operation.b_E_SOC_t')
        model.connect('ems.b_t', 'battery_with_reliability.b_t')
        model.connect('battery_with_reliability.b_t_rel', 'ems_long_term_operation.b_t')

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
        prob.set_val('G_MW', sim_pars['G_MW'])
        #prob.set_val('pv_deg_per_year', sim_pars['pv_deg_per_year'])
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])      
        prob.set_val('peak_hr_quantile',sim_pars['peak_hr_quantile'] )
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])        
        prob.set_val('min_LoH', sim_pars['min_LoH'])
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])

        

        self.prob = prob
    
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
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        self.num_batteries : Number of allowed replacements of the battery
        """

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
            prob['mean_AEP']/(self.sim_pars['G_MW']*365*24),
            self.sim_pars['G_MW'],
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

if __name__ == '__main__':
    
    import time
    from hydesign.examples import examples_filepath
    
    name = 'France_good_wind'
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    ex_site = examples_sites.loc[examples_sites.name == name]
    
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
    
    hpp = hpp_model(latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    sim_pars_fn=sim_pars_fn,
                    input_ts_fn=input_ts_fn,)
    
    start = time.time()
    
    x=[55.0, 257.0, 10.000000000000002, 75.0, 5.916666666666667, 75.0, 28.125, 191.25, 1.4791666666666665, 27.0, 4.0, 8.75]
    
    outs = hpp.evaluate(*x)
    
    hpp.print_design(x, outs)
    
    end = time.time()
    print('exec. time [min]:', (end - start)/60 )
    
    print(hpp.prob['NPV_over_CAPEX'])
