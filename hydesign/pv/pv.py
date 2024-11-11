# %%
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import xarray as xr
import openmdao.api as om

# pvlib imports
from pvlib import pvsystem, tools, irradiance, atmosphere
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


class pvp(om.ExplicitComponent):
    """PV power plant model : It computes the solar power output during the lifetime of the plant using solar plant AC capacity, DC/AC ratio, location coordinates and PV module angles"""

    def __init__(self,
                 weather_fn,
                 N_time,
                 latitude,
                 longitude,
                 altitude,
                 tracking = 'single_axis'):

        """Initialization of the PV power plant model

        Parameters
        ----------
        weather_fn : Weather timeseries
        N_time : Length of the representative data
        latitude : Latitude at chosen location
        longitude : Longitude at chosen location
        altitude : Altitude at chosen location
        tracking : Tracking type of the PV modules, ex:'single_axis'

        """   
        super().__init__()
        self.weather_fn = weather_fn
        self.N_time = N_time
        self.tracking = tracking

        pvloc = Location(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            name='Plant')

        weather = pd.read_csv(
            weather_fn, 
            index_col=0,
            parse_dates=True)

        weather = weather.rename(columns={'GHI': 'ghi',
                                  'DNI': 'dni',
                                  'DHI': 'dhi',
                                  })
        weather['temp_air'] = weather['temp_air_1'] - 273.15  # Celcium
        heights = [int(x.split('WS_')[-1]) for x in list(weather) if x.startswith('WS_')]
        min_key = f'WS_{int(np.min(heights))}'
        weather['wind_speed'] = weather[min_key]

        self.weather = weather
        self.pvloc = pvloc

    def setup(self):
        self.add_input(
            'surface_tilt',
            val=20,
            desc="Solar PV tilt angle in degs")

        self.add_input(
            'surface_azimuth',
            val=180,
            desc="Solar PV azimuth angle in degs, 180=south facing")

        self.add_input(
            'DC_AC_ratio',
            desc="DC/AC PV ratio")

        self.add_input(
            'solar_MW',
            val=1,
            desc="Solar PV plant installed capacity",
            units='MW')
        
        self.add_input(
            'land_use_per_solar_MW',
            val=1,
            desc="Solar land use per solar MW",
            units='km**2/MW')
        
        self.add_output(
            'solar_t',
            desc="PV power time series",
            units='MW',
            shape=[self.N_time])
        
        self.add_output(
            'Apvp',
            desc="Land use area of WPP",
            units='km**2')
        
    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):
        surface_tilt = inputs['surface_tilt']
        surface_azimuth = inputs['surface_azimuth']
        solar_MW = inputs['solar_MW'][0]
        land_use_per_solar_MW = inputs['land_use_per_solar_MW'][0]
        DC_AC_ratio = inputs['DC_AC_ratio']
        
        Apvp = solar_MW * land_use_per_solar_MW
        solar_t = get_solar_time_series(
            surface_tilt = surface_tilt, 
            surface_azimuth = surface_azimuth, 
            solar_MW = solar_MW, 
            land_use_per_solar_MW = land_use_per_solar_MW, 
            DC_AC_ratio = DC_AC_ratio, 
            tracking = self.tracking, 
            pvloc = self.pvloc, 
            weather = self.weather)
        outputs['solar_t'] = solar_t
        outputs['Apvp'] = Apvp

        

class pvp_with_degradation(om.ExplicitComponent):
    """
    PV degradation model providing the PV degradation time series throughout the lifetime of the plant
    """
    def __init__(
        self, 
        life_y = 25,
        intervals_per_hour=1,
        pv_deg_yr = [0, 25],
        pv_deg = [0, 25*1/100],
        ):
        """Initialization of the PV degradation model

        Parameters
        ----------
        life_h : lifetime of the plant

        """ 
        super().__init__()
        self.life_y = life_y
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.intervals_per_hour = intervals_per_hour
        
        # PV degradation curve
        self.pv_deg_yr = pv_deg_yr
        self.pv_deg = pv_deg        
        
    def setup(self):
        self.add_input(
            'solar_t_ext', 
            desc="PVP power time series", 
            units='MW',
            shape=[self.life_intervals])
        
        self.add_output(
            'solar_t_ext_deg', 
            desc="PVP power time series with degradation", 
            units='MW',
            shape=[self.life_intervals])   

    def compute(self, inputs, outputs):

        solar_t_ext = inputs['solar_t_ext']
        # t_over_year = np.arange(self.life_h)/(365*24)
        t_over_year = np.arange(self.life_intervals)/(365*24*self.intervals_per_hour)
        degradation = np.interp(t_over_year, self.pv_deg_yr, self.pv_deg)

        outputs['solar_t_ext_deg'] = (1-degradation)*solar_t_ext
    
class pvp_degradation_linear(om.ExplicitComponent):
    """
    PV degradation model providing the PV degradation time series throughout the lifetime of the plant, 
    considering a fixed linear degradation of the PV panels
    """
    def __init__(self,
                 life_y = 25,
                 intervals_per_hour=1,
                 ):
        """Initialization of the PV degradation model

        Parameters
        ----------
        life_h : lifetime of the plant

        """ 
        super().__init__()
        self.life_y = life_y
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.intervals_per_hour = intervals_per_hour
        
    def setup(self):
        self.add_input('pv_deg_per_year', desc="PV degradation per year", val=0.5 / 100)
        self.add_output('SoH_pv', desc="PV state of health time series", shape=[self.life_h])   

    def compute(self, inputs, outputs):
        pv_deg_per_year = inputs['pv_deg_per_year']
        outputs['SoH_pv'] = get_linear_solar_degradation(pv_deg_per_year, self.life_intervals, self.intervals_per_hour)   

class shadow(om.ExplicitComponent):
    """pv loss model due to shadows of wt"""

    # TODO implement degradation model in pcw
    # 1. Add input for:
    #    - turbine locations x_wt, y_wt in lat long
    #    - Pv locations
    #    - Altitude at the site
    # 2. Compute sun poisition:
    #    - sun position
    #    - simple wt shadow model to estimate covered area
    # 3. Estimate efficiency_t due to shadows

    def __init__(self, N_time):
        super().__init__()
        self.N_time = N_time

    def setup(self):
        self.add_input('solar_deg_t',
                       desc="PV power time series with degradation",
                       units='W',
                       shape=[self.N_time])
        self.add_output(
            'solar_deg_shad_t',
            desc="PV power time series with degradation and shadow losses",
            units='W',
            shape=[
                self.N_time])

    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):

        solar_deg_t = inputs['solar_deg_t']
        outputs['solar_deg_shad_t'] = solar_deg_t


# -----------------------------------------------------------------------
# Auxiliar functions 
# -----------------------------------------------------------------------        

def get_solar_time_series(
    surface_tilt, 
    surface_azimuth, 
    solar_MW, 
    land_use_per_solar_MW, 
    DC_AC_ratio, 
    tracking, 
    pvloc, 
    weather):

    """ Computing the output power time series of the PV plant

    Parameters
    ----------
    surface_tilt : surface tilt of the PV panels
    surface_azimuth : azimuth of the PV panels
    DC_AC_ratio : DC-AC ratio of the PV converter
    solar_MW : AC nominal capacity of the PV power plant

    Returns
    -------
    solar_t : PV power time series 
    """
    
    # Sandia
    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    module_name = 'Canadian_Solar_CS5P_220M___2009_'
    module = sandia_modules[module_name]
    module['aoi_model'] = irradiance.aoi

    # 2. Inverter
    # -------------
    inverters = pvsystem.retrieve_sam('cecinverter')
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    
    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
            
    if tracking == 'single_axis':
      
        mount = pvsystem.SingleAxisTrackerMount(
            axis_tilt=float(surface_tilt),
            axis_azimuth=float(surface_azimuth), 
            max_angle = 90.0, 
            backtrack = True, 
            gcr = 0.2857142857142857, 
            cross_axis_tilt = 0.0,
            #module_height = 1
            )
        array = pvsystem.Array(
            mount=mount, 
            module_parameters=module,
            temperature_model_parameters=temp_model)
        system = pvsystem.PVSystem(
            arrays=[array],
            inverter_parameters=inverter,
            )
    else:
        system = pvsystem.PVSystem(
            module_parameters=module,
            inverter_parameters=inverter,
            temperature_model_parameters=temp_model,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth)

    mc = ModelChain(system, pvloc)

    # Run solar with the WRF weather
    mc.run_model(weather)

    
    DC_AC_ratio_ref = inverter.Pdco / inverter.Paco
    Paco = inverter.Paco * DC_AC_ratio_ref / DC_AC_ratio
    solar_t = (mc.results.ac / Paco)
   
    solar_t[solar_t>1] = 1
    solar_t[solar_t<0] = 0
    return solar_MW * solar_t.fillna(0.0)

def get_linear_solar_degradation(pv_deg_per_year, life, intervals_per_hour=1):
    """ 
    Computes the PV degradation

    Parameters
    ----------
    pv_deg_per_year : fixed yearly degradation of PV panels
    life : lifetime of the plant in intervals

    Returns
    -------
    SoH_pv : degradation of the PV plant throughout the lifetime
    """
    t_over_year = np.arange(life)/(365*24*intervals_per_hour)
    degradation = pv_deg_per_year * t_over_year

    y = 1 - degradation
    if len(y[y < 0]) > 0:
        y[y < 0] = 0  # No replacement of PV panels
    return y