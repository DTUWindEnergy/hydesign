# %%
import glob
import os
import time

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

from hydesign.weather import extract_weather_for_HPP, ABL
from hydesign.wind import genericWT_surrogate, genericWake_surrogate, wpp, get_rotor_area, get_rotor_d
from hydesign.pv import pvp, pvp_degradation_linear
from hydesign.ems import ems, ems_long_term_operation
from hydesign.battery_degradation import battery_degradation
from hydesign.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance import finance
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
        num_batteries = 1,
        ems_type='pyomo',
        input_ts_fn = None, # If None then it computes the weather
        price_fn = None, # If input_ts_fn is given it should include Price column.
        genWT_fn = lut_filepath+'genWT_v3.nc',
        genWake_fn = lut_filepath+'genWake_v3.nc',
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
        
        print('longitude =',longitude)
        print('latitude =',latitude)
        print('altitude =',altitude)
        
        # Parameters of the simulation
        year_start = sim_pars['year']
        year_end = sim_pars['year']
        N_life = sim_pars['N_life']
        life_h = N_life*365*24
        n_steps_in_LoH = sim_pars['n_steps_in_LoH']
        G_MW = sim_pars['G_MW']
        battery_depth_of_discharge = sim_pars['battery_depth_of_discharge']
        battery_charge_efficiency = sim_pars['battery_charge_efficiency']
        min_LoH = sim_pars['min_LoH']
        pv_deg_per_year = sim_pars['pv_deg_per_year']
        wpp_efficiency = sim_pars['wpp_efficiency']
        land_use_per_solar_MW = sim_pars['land_use_per_solar_MW']
        
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
            
            input_ts_fn = f'{work_dir}input_ts.csv'
            weather.to_csv(input_ts_fn)
            N_time = len(weather)
            
        else: # User provided weather timeseries
            weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
            N_time = len(weather)
        
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
                'n_full_power_hours_expected_per_day_at_peak_price'
                ]
            )
        model.add_subsystem(
            'battery_degradation', 
            battery_degradation(
                num_batteries = num_batteries,
                n_steps_in_LoH = n_steps_in_LoH,
                life_h = life_h),
            promotes_inputs=[
                'min_LoH'
                ])
        
        model.add_subsystem(
            'pvp_degradation_linear', 
            pvp_degradation_linear(
                life_h = life_h),
            promotes_inputs=[
                'pv_deg_per_year'
                ])
        
        model.add_subsystem(
            'ems_long_term_operation', 
            ems_long_term_operation(
                N_time = N_time,
                num_batteries = num_batteries,
                n_steps_in_LoH = n_steps_in_LoH,
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
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=sim_pars['battery_energy_cost'],
                battery_power_cost=sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=sim_pars['battery_energy_onm_cost'],
                num_batteries = num_batteries,
                n_steps_in_LoH = n_steps_in_LoH,
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
                              'mean_AEP',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX'
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
        
        model.connect('battery_degradation.ii_time', 'ems_long_term_operation.ii_time')
        model.connect('battery_degradation.SoH', 'ems_long_term_operation.SoH')
        model.connect('pvp_degradation_linear.SoH_pv', 'ems_long_term_operation.SoH_pv')
        
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.price_t_ext', 'ems_long_term_operation.price_t_ext')

        model.connect('wpp.wind_t', 'wpp_cost.wind_t')
        
        model.connect('battery_degradation.ii_time','battery_cost.ii_time')
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
        prob.set_val('price_t', weather['Price'])
        prob.set_val('G_MW', sim_pars['G_MW'])
        prob.set_val('pv_deg_per_year', sim_pars['pv_deg_per_year'])
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
        

        self.sim_pars = sim_pars
        self.prob = prob
        self.num_batteries = num_batteries
    
        self.list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP',
            'GUF',
            'grid [MW]',
            'wind [MW]',
            'solar [MW]',
            'Battery Energy [MWh]',
            'Battery Power [MW]',
            'Total curtailment [GWh]',
            'Awpp [km2]',
            #'Apvp [km2]',
            'Rotor diam [m]',
            'Hub height [m]',
            'Number_of_batteries',
            ]

        self.list_vars = [
            'clearance [m]', 
            'sp [m2/W]', 
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
        sp : Specific power of the turbine [MW/m2] 
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
        #Apvp = solar_MW * self.sim_pars['land_use_per_solar_MW']
        #prob.set_val('Apvp', Apvp)

        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)
        
        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)        
        
        prob.run_model()
        
        self.prob = prob
        
        return np.hstack([
            prob['NPV_over_CAPEX'], 
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob['penalty_lifetime']/1e6,
            prob['mean_AEP'],
            # Grid Utilization factor
            prob['mean_AEP']/(self.sim_pars['G_MW']*365*24),
            self.sim_pars['G_MW'],
            wind_MW,
            solar_MW,
            b_E,
            b_P,
            prob['total_curtailment']/1e3, #[GWh]
            Awpp,
            d,
            hh,
            self.num_batteries
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


# %%
