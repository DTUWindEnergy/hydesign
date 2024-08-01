"""
I have a script and want the user to be able to following thing (or any other possible flexibility in developing the code
1. chnage name of the methods
2. add or remove any part of the code (in any class/function/method ...), for example:
    make new 'self.add_input' or 'self.add_output', remove the existing ones, or rename any of them
3. add and remove code to any part of the scripts

I want make it such that the developers inherit this script and make the changes according to their needs. If I share with you the script can you please help me with this?

"""

import numpy as np
# from numpy import newaxis as na
import pandas as pd
import openmdao.api as om
# import yaml

# import xarray as xr
from docplex.mp.model import Model

class ems(om.ExplicitComponent):
    """
    """

    def __init__(
        self, 
        N_time, 
        life_h = 25*365*24, 
        weeks_per_season_per_year = None,
        ems_type='cplex'):

        super().__init__()
        self.weeks_per_season_per_year = weeks_per_season_per_year
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_h = int(life_h)
        
    def setup(self):
        self.add_input(
            'wind_t',
            desc="WPP power time series",
            units='MW',
            shape=[self.N_time])
        self.add_input(
            'solar_t',
            desc="PVP power time series",
            units='MW',
            shape=[self.N_time])
        self.add_input(
            'price_t',
            desc="Electricity price time series",
            shape=[self.N_time])

        # ----------------------------------------------------------------------------------------------------------
        self.add_output(
            'wind_t_ext',
            desc="WPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])


    def compute(self, inputs, outputs):
        
        wind_t = inputs['wind_t']
        solar_t = inputs['solar_t']
        price_t = inputs['price_t']

        print('some code here')

class ems_long_term_operation(om.ExplicitComponent):
    """
    """

    def __init__(
        self, 
        N_time,
        num_batteries = 1,
        life_h = 25*365*24, 
        ems_type='energy_penalty',
        load_min_penalty_factor=1e6,
        ):

        super().__init__()
        self.N_time = N_time
        self.life_h = life_h
        self.ems_type = ems_type
        self.load_min_penalty_factor = load_min_penalty_factor

    def setup(self):
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_h])
        self.add_input(
            'wind_t_ext_deg',
            desc="Wind time series including degradation",
            units='MW',
            shape=[self.life_h])
        # -------------------------------------------------------
        self.add_output(
            'hpp_t_with_deg',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t_with_deg',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])

    def compute(self, inputs, outputs):
        
        SoH = inputs['SoH']
        wind_t_ext_deg = inputs['wind_t_ext_deg']
        solar_t_ext_deg = inputs['solar_t_ext_deg']
        
        wind_t_ext = inputs['wind_t_ext']
        solar_t_ext = inputs['solar_t_ext']
        price_t_ext = inputs['price_t_ext']
        # b_P = inputs['b_P']
        b_E = inputs['b_E']
        G_MW = inputs['G_MW']

        if self.ems_type == 'energy_penalty':
            ems_longterm = operation_solar_batt_deg
        else:
            raise Warning("This class should only be used for energy_penalty")

        print('some codes here')
        
        outputs['hpp_t_with_deg'] = Hpp_deg
        outputs['hpp_curt_t_with_deg'] = P_curt_deg

# -----------------------------------------------------------------------
# Auxiliar functions for ems modelling
# -----------------------------------------------------------------------

def expand_to_lifetime(x, life_h = 25*365*24, weeks_per_season_per_year=None):
    """
    """    
    if weeks_per_season_per_year == None:
        
        # Extend the data to match the expected lifetime
        len_x = len(x)
        N_repeats = int(np.ceil(life_h/len_x))
    
    else:
        x_ext = np.array([])
    
        # extend selected weeks into a year: 4 season of 13 weeks + one extra day.
        for x_batch in split_in_batch(x,weeks_per_season_per_year*7*24):
            x_ext = np.append( x_ext, np.tile(x_batch,20)[:24*7*13] )
        x_ext = np.append( x_ext, x_batch[-24:] )    
        
        # extend the constructed year to match the expected lifetime
        x = x_ext    
        N_repeats = int(np.ceil(life_h/365*24))
    
    return np.tile(x,N_repeats)[:life_h]
