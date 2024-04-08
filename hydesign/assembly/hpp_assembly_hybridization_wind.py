# %%
# import glob
import os
# import time

# basic libraries
import numpy as np
# from numpy import newaxis as na
# import numpy_financial as npf
import pandas as pd
# import seaborn as sns
import openmdao.api as om
import yaml
# import scipy as sp
# from scipy import stats
import xarray as xr


from hydesign.weather.weather import extract_weather_for_HPP, select_years
from hydesign.weather.weather_wind_hybridization import ABL_WD
from hydesign.wind.wind_hybridization import existing_wpp, existing_wpp_with_degradation # , genericWT_surrogate, genericWake_surrogate, get_rotor_area, get_rotor_d
from hydesign.pv.pv import pvp
from hydesign.pv.pv_hybridization import pvp_with_degradation
from hydesign.ems.ems import ems, ems_long_term_operation
from hydesign.utils import hybridization_shifted
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost
from hydesign.costs.costs_hybridized_wind import shared_cost, decommissioning_cost
from hydesign.finance.finance_hybridized_wind import finance
from hydesign.look_up_tables import lut_filepath


class hpp_model:
    """HPP design evaluator"""

    def __init__(
        self,
        latitude,
        longitude,
        altitude=None,
        sim_pars_fn=None,
        work_dir = './',
        num_batteries = 3,
        factor_battery_cost = 1,
        ems_type='cplex',
        weeks_per_season_per_year = None,
        seed=0, # For selecting random weeks pe season to reduce the computational time
        # existing_wpp_power_curve_xr_fn = None, # Input file for existing wind plant. For docs look at wind.existing_wpp_with_degradation
        input_ts_fn = None, # If None then it computes the weather
        price_fn = None, # If input_ts_fn is given it should include Price column.
        genWT_fn = lut_filepath+'genWT_v3.nc',
        genWake_fn = lut_filepath+'genWake_v3.nc',
        verbose = True,
        name = '',
        ppa_price = None,
        # pv_reduction=0,
        # batt_reduction=0,
        **kwargs,
        ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        latitude : Latitude at chosen location
        longitude : Longitude at chosen location
        altitude : Altitude at chosen location, if not provided, elevation is calculated using elevation map datasets
        sims_pars_fn : Case study input values of the HPP 
        work_dir : Working directory path
        num_batteries : Number of battery replacements
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
        work_dir = mkdir(work_dir)
        
        # Extract simulation parameters
        try:
            with open(sim_pars_fn) as file:
                sim_pars = yaml.load(file, Loader=yaml.FullLoader)
        except:
            raise(f'sim_pars_fn="{sim_pars_fn}" can not be read')
        
        if altitude == None:
            
            elevation_fn = sim_pars['elevation_fn'] # Altitude map for extracting altitude
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
        if 'year_start' in sim_pars.keys():
            year_start = sim_pars['year_start']
        else:
            year_start = sim_pars['year']


        if 'year_end' in sim_pars.keys():
            year_end = sim_pars['year_end']
        else:
            year_end = sim_pars['year']

        N_life = sim_pars['N_life']
        life_h = (N_life+15)*365*24
        N_limit = 15 # delta_life limit

        # G_MW = sim_pars['G_MW']
        # battery_depth_of_discharge = sim_pars['battery_depth_of_discharge']
        # battery_charge_efficiency = sim_pars['battery_charge_efficiency']
        # min_LoH = sim_pars['min_LoH']
        wpp_efficiency = sim_pars['wpp_efficiency']
        # land_use_per_solar_MW = sim_pars['land_use_per_solar_MW']
        
        if 'wind_deg' in sim_pars:
            wind_deg = sim_pars['wind_deg']
            wind_deg_yr = sim_pars['wind_deg_yr']
            share_WT_deg_types = sim_pars['share_WT_deg_types']
        else:
            wind_deg = [0, 0]
            wind_deg_yr = [0, 25]
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
            
            input_ts_fn = f'{work_dir}input_ts{name}.csv'
            print(f'\ninput_ts_fn extracted and stored in {input_ts_fn}')
            weather.to_csv(input_ts_fn)
            N_time = len(weather)
            
        else: # User provided weather timeseries
            weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
            N_time = len(weather)

        if ppa_price is None:
            price = weather['Price']

        else:
            price = ppa_price * np.ones_like(weather['Price'])
            
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
        
        # with xr.open_dataset(genWT_fn) as ds: 
        #     # number of points in the power curves
        #     N_ws = len(ds.ws.values)
        
        existing_wpp_power_curve_xr_fn = os.path.join(os.path.dirname(sim_pars_fn), sim_pars['existing_wpp_power_curve_xr_fn'])
        
        model = om.Group()

        model.add_subsystem(
            'abl_wd', 
            ABL_WD(
                weather_fn=input_ts_fn, 
                N_time=N_time),
            promotes_inputs=['hh']
            )
        
        model.add_subsystem(
            'existing_wpp', 
            existing_wpp(
                N_time = N_time,
                wpp_efficiency = wpp_efficiency,
                existing_wpp_power_curve_xr_fn = existing_wpp_power_curve_xr_fn,
            )
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
                life_h = life_h,
                N_time = N_time,
                weeks_per_season_per_year = weeks_per_season_per_year,
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
                'n_full_power_hours_expected_per_day_at_peak_price'
                ]
            )
        model.add_subsystem(
            'battery_degradation', 
            battery_degradation(
                life_h = life_h,
                weather_fn = input_ts_fn, # for extracting temperature
                num_batteries = num_batteries,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            promotes_inputs=[
                'min_LoH'
                ])

        model.add_subsystem(
            'hybridization_shifted',
            hybridization_shifted(
                N_limit=N_limit,
                N_life=N_life,
                N_time=N_time,
                life_h=life_h,
            ),
            promotes_inputs=[
                'delta_life',
            ])

        model.add_subsystem(
            'battery_loss_in_capacity_due_to_temp', 
            battery_loss_in_capacity_due_to_temp(
                life_h = life_h,
                weather_fn = input_ts_fn, # for extracting temperature
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            )

        model.add_subsystem(
            'existing_wpp_with_degradation', 
            existing_wpp_with_degradation(
                life_h = life_h,
                N_time = N_time,
                existing_wpp_power_curve_xr_fn =  existing_wpp_power_curve_xr_fn,
                wpp_efficiency = wpp_efficiency,
                wind_deg_yr = wind_deg_yr,
                wind_deg = wind_deg,
                share_WT_deg_types = share_WT_deg_types,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            )

        model.add_subsystem(
            'pvp_with_degradation',
            pvp_with_degradation(
                N_life = N_life,
                N_limit = N_limit,
                life_h = life_h,
                pv_deg = sim_pars['pv_deg'],
            ),
            promotes_inputs=[
                'delta_life'
            ])

        model.add_subsystem(
            'ems_long_term_operation', 
            ems_long_term_operation(
                N_time = N_time,
                life_h = life_h,
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile',
                'n_full_power_hours_expected_per_day_at_peak_price',
                ],
            promotes_outputs=[
                'total_curtailment'
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
                # pv_reduction=pv_reduction,
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_inverter_cost=sim_pars['solar_inverter_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                life_h = life_h,
                # batt_reduction=batt_reduction,
                battery_energy_cost=factor_battery_cost*sim_pars['battery_energy_cost'],
                battery_power_cost=factor_battery_cost*sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=factor_battery_cost*sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=factor_battery_cost*sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=factor_battery_cost*sim_pars['battery_energy_onm_cost'],
                N_life = N_life,
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'battery_price_reduction_per_year',
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
                'p_rated',
                'Nwt',
                'solar_MW',
                'Awpp',
            ])

        model.add_subsystem(
            'decommissioning_cost',
            decommissioning_cost(
                decommissioning_cost_w=sim_pars['decommissioning_cost_w'],
                decommissioning_cost_s=sim_pars['decommissioning_cost_s'],
            ),
            promotes_inputs=['solar_MW',
                             ],
            promotes_outputs=['decommissioning_cost_tot_w',
                              'decommissioning_cost_tot_s',
                              ],
            )

        model.add_subsystem(
            'finance', 
            finance(
                N_limit = N_limit,
                N_life = N_life,
                life_h = life_h,
                N_time = N_time,
                # Depreciation curve
                depreciation_yr = sim_pars['depreciation_yr'],
                depreciation = sim_pars['depreciation'],
                depre_rate = sim_pars['depre_rate'],
                # Inflation curve
                inflation_yr = sim_pars['inflation_yr'],
                inflation = sim_pars['inflation'],
                ref_yr_inflation = sim_pars['ref_yr_inflation'],
                # Early paying or CAPEX Phasing
                # phasing_yr = sim_pars['phasing_yr'],
                # phasing_CAPEX = sim_pars['phasing_CAPEX']
                ),
            promotes_inputs=[
                             'delta_life',
                             # 'wind_WACC',
                             # 'solar_WACC', 
                             # 'battery_WACC',
                              'hpp_WACC',
                             'tax_rate',
                             'decommissioning_cost_tot_w',
                             'decommissioning_cost_tot_s',
                            ],
            promotes_outputs=['NPV',
                              'IRR',
                              'NPV_over_CAPEX',
                              'LCOE',
                              'COVE',
                              'mean_AEP',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX',
                              'break_even_PPA_price',
                              ],
        )
                  
                      
        model.connect('abl_wd.wst', 'existing_wpp.wst')
        model.connect('abl_wd.wdt', 'existing_wpp.wdt')
        
        model.connect('existing_wpp.wind_t', 'ems.wind_t')
        model.connect('pvp.solar_t', 'ems.solar_t')
        
        model.connect('ems.b_E_SOC_t', 'battery_degradation.b_E_SOC_t')
        
        model.connect('battery_degradation.SoH', 'hybridization_shifted.SoH')
        model.connect('hybridization_shifted.SoH_shifted', 'battery_loss_in_capacity_due_to_temp.SoH')
        model.connect('battery_loss_in_capacity_due_to_temp.SoH_all', 'ems_long_term_operation.SoH')

        model.connect('abl_wd.wst', 'existing_wpp_with_degradation.wst')
        model.connect('abl_wd.wdt', 'existing_wpp_with_degradation.wdt')
        
        model.connect('existing_wpp_with_degradation.wind_t_ext_deg', 'ems_long_term_operation.wind_t_ext_deg')

        model.connect('ems.solar_t_ext','pvp_with_degradation.solar_t_ext')
        model.connect('pvp_with_degradation.solar_t_ext_deg', 'ems_long_term_operation.solar_t_ext_deg')
        
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('ems.b_E_SOC_t', 'ems_long_term_operation.b_E_SOC_t')
        model.connect('ems.b_t', 'ems_long_term_operation.b_t')

        model.connect('existing_wpp.wind_t', 'wpp_cost.wind_t')
        
        model.connect('battery_degradation.SoH','battery_cost.SoH')
        #model.connect('hybridization_shifted.SoH_shifted', 'battery_cost.SoH')

        model.connect('pvp.Apvp', 'shared_cost.Apvp')
        
        model.connect('wpp_cost.CAPEX_w', 'finance.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance.OPEX_w')

        model.connect('pvp_cost.CAPEX_s', 'finance.CAPEX_s')
        model.connect('pvp_cost.OPEX_s', 'finance.OPEX_s')

        model.connect('battery_cost.CAPEX_b', 'finance.CAPEX_b')
        model.connect('battery_cost.OPEX_b', 'finance.OPEX_b')

        model.connect('shared_cost.CAPEX_sh_w', 'finance.CAPEX_el_w')
        model.connect('shared_cost.CAPEX_sh_s', 'finance.CAPEX_el_s')
        model.connect('shared_cost.OPEX_sh', 'finance.OPEX_el')

        model.connect('wpp_cost.CAPEX_w', 'decommissioning_cost.CAPEX_w')

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
        
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])      
        prob.set_val('peak_hr_quantile',sim_pars['peak_hr_quantile'] )
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])        
        prob.set_val('min_LoH', sim_pars['min_LoH'])
        # prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        # prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        # prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('hpp_WACC', sim_pars['hpp_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])

        prob.set_val('hh', sim_pars['hh'])
        prob.set_val('d', sim_pars['d'])
        prob.set_val('p_rated', sim_pars['p_rated'])
        prob.set_val('Nwt', sim_pars['Nwt'])
        prob.set_val('Awpp', sim_pars['Awpp'])

        self.sim_pars = sim_pars
        self.prob = prob
        self.num_batteries = num_batteries
        self.input_ts_fn = input_ts_fn
        self.altitude = altitude
    
        self.list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'COVE [-]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'Wind CAPEX [MEuro]',
            'Wind OPEX [MEuro]',
            'PV CAPEX [MEuro]',
            'PV OPEX [MEuro]',
            'Batt CAPEX [MEuro]',
            'Batt OPEX [MEuro]',
            'Shared CAPEX W [MEuro]',
            'Shared CAPEX S [MEuro]',
            'Shared OPEX [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP per year [GWh]',
            'GUF',
            'grid [MW]',
            'wind [MW]',
            'solar [MW]',
            'Battery Energy [MWh]',
            'Battery Power [MW]',
            'Total curtailment [GWh]',
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
            'solar_MW [MW]',
            'surface_tilt [deg]', 
            'surface_azimuth [deg]', 
            'DC_AC_ratio', 
            'b_P [MW]', 
            'b_E_h [h]',
            'cost_of_battery_P_fluct_in_peak_price_ratio',
            'delta_life [years]'
            ]   
    
    
    def evaluate(
        self,
        # PV plant design
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio,
        # Time desig
        delta_life,
        ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]
        delta_life : Distance in years between the starting operations of the wind plant and of the PV+batteries

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['COVE'] : Cost of Valued Energy
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

        b_E = b_E_h * b_P
        
        # pass design variables
        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)
        
        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)

        prob.set_val('delta_life', delta_life)

        prob.run_model()
        
        self.prob = prob

        hh = prob['hh']
        d = prob['d']
        p_rated = prob['p_rated']
        Nwt = prob['Nwt']
        Awpp = prob['Awpp']

        wind_MW = p_rated * Nwt
        
        if Nwt == 0:
            cf_wind = np.nan
        else:
            wind_t_ext_degg = prob.get_val('existing_wpp_with_degradation.wind_t_ext_deg')
            cf_wind = np.mean(wind_t_ext_degg[wind_t_ext_degg != 0]) / wind_MW   # Capacity factor of wind only

        return np.hstack([
            prob['NPV_over_CAPEX'], 
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['COVE'],
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob.get_val('finance.CAPEX_w')/1e6,
            prob.get_val('finance.OPEX_w')/1e6,
            prob.get_val('finance.CAPEX_s')/1e6,
            prob.get_val('finance.OPEX_s')/1e6,
            prob.get_val('finance.CAPEX_b')/1e6,
            prob.get_val('finance.OPEX_b')/1e6,
            prob.get_val('finance.CAPEX_el_w')/1e6,
            prob.get_val('finance.CAPEX_el_s') / 1e6,
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
                                            'COVE [-]',
                                            'CAPEX [MEuro]',
                                            'OPEX [MEuro]',
                                            'Wind CAPEX [MEuro]',
                                            'Wind OPEX [MEuro]',
                                            'PV CAPEX [MEuro]',
                                            'PV OPEX [MEuro]',
                                            'Batt CAPEX [MEuro]',
                                            'Batt OPEX [MEuro]',
                                            'Shared CAPEX W [MEuro]',
                                            'Shared CAPEX S [MEuro]',
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
                                            'Awpp [km2]',
                                            'Apvp [km2]',
                                            'Plant area [km2]',
                                            'Rotor diam [m]',
                                            'Hub height [m]',
                                            'Number of batteries used in lifetime',
                                            'Break-even PPA price [Euro/MWh]',
                                            'Capacity factor wind [-]',
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

