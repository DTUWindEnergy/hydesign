# %%

import glob
import os
import time
import copy

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import scipy as sp
import openmdao.api as om
import yaml

import xarray as xr
from docplex.mp.model import Model

import rainflow

from hydesign.ems.ems import expand_to_lifetime

class battery_degradation(om.ExplicitComponent):
    """
    Battery degradation model to predict the degradation of the battery throughout the lifetime of the plant

    Parameters
    ----------
    b_E_SOC_t : battery energy SOC time series
    min_LoH : minimum level of health before death of battery

    Returns
    -------
    SoH : battery state of health at discretization levels
    """

    def __init__(
        self, 
        weather_fn,
        num_batteries = 1,
        life_y = 25,
        intervals_per_hour = 1,
        weeks_per_season_per_year = None,
        battery_deg = True,
    ):

        super().__init__()
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.yearly_intervals = 365 * 24 * intervals_per_hour
        self.num_batteries = num_batteries
        self.weather_fn = weather_fn
        self.battery_deg = battery_deg
        self.battery_rf_matrix = None

        weather = pd.read_csv(
            weather_fn, 
            index_col=0,
            parse_dates=True)

        air_temp_K_t = expand_to_lifetime(
            weather.temp_air_1.values, 
            life = self.life_intervals,
            weeks_per_season_per_year = weeks_per_season_per_year)

        self.air_temp_K_t = air_temp_K_t
        # print(life_y, self.life_h)

    def setup(self):
        self.add_input(
            'b_E_SOC_t',
            desc="Battery energy SOC time series",
            shape=[self.life_intervals + 1])
        self.add_input(
            'min_LoH',
            desc="minimum level of health before death of battery")
        
        # -------------------------------------------------------

        self.add_output(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_intervals])
        self.add_output(
            'n_batteries',
            desc="Number of batteries used.",
            )

    def compute(self, inputs, outputs):

        num_batteries = self.num_batteries
        life_intervals = self.life_intervals

        b_E_SOC_t = inputs['b_E_SOC_t']
        min_LoH = inputs['min_LoH'][0]

        air_temp_K_t = self.air_temp_K_t
        
        if self.battery_deg:
            if np.max(b_E_SOC_t) == 0 or num_batteries==0:
                outputs['SoH'] = np.zeros(life_intervals)
                outputs['n_batteries'] = 0
            else:
                SoC = b_E_SOC_t/np.max(b_E_SOC_t)
                rf_DoD, rf_SoC, rf_count, rf_i_start, self.battery_rf_matrix = RFcount(SoC)   

                # use the temperature time-series
                avr_tem = np.mean(air_temp_K_t)

                # loop to determine the maximum number of replacements
                for n_batteries in np.arange(num_batteries, dtype=int) + 1:
                    LoC, ind_q, _ = battery_replacement(
                        rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, 
                        min_LoH, num_batteries=n_batteries)
                    if 1-LoC[-1] >= min_LoH: # stop replacing batteries
                        break         

                SoH_all = np.interp( 
                    x = np.arange(life_intervals)/self.yearly_intervals,
                    xp = np.array(rf_i_start)/self.yearly_intervals,
                    fp = 1-LoC )
                outputs['SoH'] = SoH_all
                outputs['n_batteries'] = n_batteries
        else:
            outputs['SoH'] = np.ones(self.life_intervals)
            outputs['n_batteries'] = 1
            

class battery_loss_in_capacity_due_to_temp(om.ExplicitComponent):
    """
    Battery non-permanent loss of capacity due to low temp

    Parameters
    ----------
    SoH : battery state of health at discretization levels

    Returns
    -------
    SoH_all : battery state of health at discretization levels
    """

    def __init__(
        self, 
        weather_fn,
        num_batteries = 1,
        life_y = 25,
        intervals_per_hour = 1,
        weeks_per_season_per_year = None,
        battery_deg = True,
    ):

        super().__init__()
        self.life_h = 365 * 24 * life_y
        self.yearly_intervals = 365 * 24 * intervals_per_hour
        self.life_intervals = self.life_h * intervals_per_hour
        self.num_batteries = num_batteries
        self.weather_fn = weather_fn
        self.battery_deg = battery_deg

        weather = pd.read_csv(
            weather_fn, 
            index_col=0,
            parse_dates=True)

        air_temp_C_t = expand_to_lifetime(
            (weather.temp_air_1 - 273.15).values, 
            life=self.life_intervals,
            weeks_per_season_per_year = weeks_per_season_per_year)

        self.air_temp_C_t = air_temp_C_t

    def setup(self):
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_intervals])
        
        # -------------------------------------------------------

        self.add_output(
            'SoH_all',
            desc="Battery state of health at discretization levels",
            shape=[self.life_intervals])

    def compute(self, inputs, outputs):
        
        # life_h = self.life_h
        air_temp_C_t = self.air_temp_C_t

        B_E_loss_due_to_low_temp = thermal_loss_of_storage(air_temp_C_t)
        
        if self.battery_deg:
            outputs['SoH_all'] = B_E_loss_due_to_low_temp * inputs['SoH']
        else:
            outputs['SoH_all'] = np.ones(self.life_intervals)
         

# -----------------------------------------------------------------------
# Auxiliar functions for bat_deg modelling
# -----------------------------------------------------------------------

def incerase_resolution(ii_time, SoH, life, nn, hourly_intervals=1):
    iis = 1
    n_obtained = len(ii_time)
    while nn > n_obtained:
        ii_add = range(life - 24*iis*hourly_intervals, life, 24*hourly_intervals)
        ii_time_new = np.unique( np.sort( np.append( ii_time, ii_add) ) )
        n_obtained = len(ii_time_new)
        iis += 1
        #print(nn, n_obtained)
    
    ii_time_interp = np.append(ii_time,life)
    SoH_new = sp.interpolate.interp1d(
        x=0.5*ii_time_interp[1:] + 0.5*ii_time_interp[:-1],
        y=SoH,
        kind='nearest',
        fill_value='extrapolate')(ii_time_new)
    
    for ii in ii_time_new:
        ind_ = np.where(ii_time == ii)[0]
        if len(ind_)>0:
            SoH_new[ind_] = SoH[ind_]        
    
    return ii_time_new, SoH_new
    
def battery_replacement(
    rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, 
    min_LoH, n_steps_in_LoH=30, num_batteries=2):
    """
    Battery degradation in steps and battery replacement

    Parameters
    ----------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    avr_tem: average temperature in the location, yearly or more long. default value is 20
    min_LoH: minimum level of health before death of battery
    n_steps_in_LoH: number of discretizations in battery state of health
    num_batteries: number of battery replacements

    Returns
    -------
    LoC: battery level of capacity
    ind_q: time indices for constant health levels
    ind_q_last: time index for battery replacement
    """

    #rf_DoD: depth of discharge after rainflow counting
    #rf_SoC: mean SoC after rainflow counting
    #rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    #rf_i_start: time index for the cycles [in hours]
    #avr_tem: average temperature in the location, yearly or more long. default value is 20

    #LoC: loss of capacity: LoC = 1 - LoH 
    
    LoC, LoC1, LLoC  = degradation(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, LLoC_0=0)
    
    if np.min(1-LoC) > min_LoH: # First battery is NOT fully used after the full lifetime
        try: #split the minimum into the number of levels
            ind_q = [np.where(1-LoC < q)[0][0] 
                     for q in np.linspace(1,np.min(1-LoC),n_steps_in_LoH+1, endpoint = False)]
            ind_q_last = ind_q[-1]
        except: #split the time into equal number of levels
            ind_q = np.linspace(0, len(rf_i_start), n_steps_in_LoH+1, dtype=int, endpoint = False)
            ind_q_last = ind_q[-1]
        
    else: # First battery is fully used after the full lifetime
        ind_q = [np.where(1-LoC < q)[0][0] 
                 for q in np.linspace(1,min_LoH,n_steps_in_LoH+1, endpoint = True)]
    
        ind_q_last = ind_q[-1]
        LoC[ind_q_last:] = 1

    # Battery replacement
    for i in range(num_batteries-1):
        try:
            # Degradation is computed after the new battery is installed: ind_q_last
            LoC_new, LoC1_new, LLoC_new  = degradation(
                rf_DoD[ind_q_last:], 
                rf_SoC[ind_q_last:], 
                rf_count[ind_q_last:], 
                rf_i_start[ind_q_last:]-rf_i_start[ind_q_last], 
                avr_tem, 
                LLoC_0=0, # starts with new battery without degradation
            )

            LoC[ind_q_last:] = LoC_new
            
            if min_LoH >  (1 - LoC_new[-1]):
                
                ind_q_new = [np.where(1-LoC_new < q)[0][0] + ind_q_last
                             for q in np.linspace(1,min_LoH,n_steps_in_LoH+1, endpoint = False)]
                ind_q_last = ind_q_new[-1] 
                ind_q = ind_q + ind_q_new[1:]

                LoC[ind_q_last:] = 1
            else:
                ind_q_new = [np.where(1-LoC_new < q)[0][0] + ind_q_last
                             for q in np.linspace(1,1-LoC_new[-1],n_steps_in_LoH+1, endpoint = False)]
                ind_q_last = ind_q_new[-1] 
                ind_q = ind_q + ind_q_new[1:]        

        except:
            raise('This many bateries are not required. Reduce the number.')

    return LoC, ind_q, ind_q_last

def degradation(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, LLoC_0=0):
    """
    Calculating the new level of capacity of the battery.
    
    Xu, B., Oudalov, A., Ulbig, A., Andersson, G., and Kirschen, D. S.: Modeling of lithium-ion battery degradation for cell life assessment, 
    IEEE Transactions on Smart Grid, 9, 1131–1140, 2016.

    Parameters
    ----------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    avr_tem: average temperature in the location, yearly or more long. default value is 20

    Returns
    -------
    LoC: battery level of capacity
    LoC1: 
    LLoC: 
    """
    #rf_DoD: depth of discharge after rainflow counting
    #rf_SoC: mean SoC after rainflow counting
    #rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    #rf_i_start: time index for the cycles [in hours]
    #avr_tem: average temperature in the location, yearly or more long. default value is 20

    #SoH: state of health = 1 - loss of capacity, between 0 and 1
    #LoC: loss of capacity
    #LLoC: linear estimation of LoC 

    LLoC_hist = Linear_Degfun(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem)
    
    alpha = 0.0575
    beta = 121
    LLoC = LLoC_0 + np.cumsum(LLoC_hist)
    LLoC1 = LLoC.copy()
    LoC1 = 1-alpha*np.exp(-LLoC*beta)-(1-alpha)*np.exp(-LLoC)
    
    SoH_l = 1-LoC1

    if np.min(SoH_l) <= 0.92:
        ind_SoH_lt_92 = np.where(SoH_l<=0.92)[0]
        
        LoC = LoC1.copy()
        LoC[ind_SoH_lt_92] = 1-(1-LoC1[ind_SoH_lt_92])*np.exp(-(LLoC[ind_SoH_lt_92]-LoC1[ind_SoH_lt_92]))
        LoC[ind_SoH_lt_92] = LoC[ind_SoH_lt_92] + LoC1[ind_SoH_lt_92[0]] - LoC[ind_SoH_lt_92[0]]
    else:
        #print( 'np.min(SoH_l) = ',np.min(SoH_l) , '> 0.92')
        LoC = LoC1.copy()
    
    return LoC, LoC1, LLoC 

def Linear_Degfun(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem): 
    """
    Linear degradation function.

    Xu, B., Oudalov, A., Ulbig, A., Andersson, G., and Kirschen, D. S.: Modeling of lithium-ion battery degradation for cell life assessment, 
    IEEE Transactions on Smart Grid, 9, 1131–1140, 2016.

    Parameters
    ----------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    avr_tem: average temperature in the location, yearly or more long. default value is 20

    Returns
    -------
    np.array(LLoC_hist): 

    """
    #LLoC:linear estimation of LoC
    #S_DoD:stress model of depth of discharge
    #S_time:stress model of time duration
    #S_SoC: stress model of state of charge
    #S_T: stress model of cell temperature
        
    kdelta1 = 1.4e5
    kdelta2 = -5.01e-1
    kdelta3 = -1.23e5
    ksigma = 1.04
    sigma_ref = 0.5
    kT = 6.93e-2
    Tref = 293.15 # in Kelvin
    kti = 4.14e-10
          
    LLoC_hist = []
    for j in range(len(rf_DoD)):        
        # To ensure no divide by zero problems
        if rf_DoD[j] != 0:
            term = rf_DoD[j]**kdelta2
        else:
            term = 0
        S_DoD = ( (kdelta1*term+kdelta3)**(-1) )
        
        #S_time = kti*(age_day*24/sum(rf_count)*rf_count[j]*3600)
        S_time = kti* rf_i_start[j]
        S_SoC = np.exp(ksigma*(rf_SoC[j]-sigma_ref))

        # instead force it to be 1 the factor on bellow the normal operating range [15-25]
        if avr_tem>Tref:
            S_T = np.exp(kT*(avr_tem-Tref)*Tref/avr_tem)
        else:
            S_T = 1
                                
        LLoC_i = (S_DoD+S_time)*S_SoC*S_T *rf_count[j]*0.7
        LLoC_hist += [LLoC_i]
                
                            
    return np.array(LLoC_hist)
        
        
def RFcount(SoC):
    rf_df = pd.DataFrame(
        data=np.array([[rng, mean, count, i_start, i_end]  
                       for rng, mean, count, i_start, i_end  in rainflow.extract_cycles(SoC)]),
        columns=['rng_', 'mean_', 'count_', 'i_start', 'i_end']
    )
    """
    Rainflow count

    Parameters
    ----------
    SoC : state of charge time series

    Returns
    -------
    rf_DoD: depth of discharge after rainflow counting
    rf_SoC: mean SoC after rainflow counting
    rf_count: half or full cycle after rainflow counting, ethier 0.5 or 1
    rf_i_start: time index for the cycles [in hours]
    """
    
    rf_df = rf_df.sort_values(by='i_start')
    return rf_df.rng_.values, rf_df.mean_.values, rf_df.count_.values, rf_df.i_start.astype(int).values, rf_df


def thermal_loss_of_storage(air_temp_C_t):
    '''
    Battery temporary loss of storage at low temperatures. Simple piecewise linear fit from:

    Lv, S., Wang, X., Lu, W., Zhang, J., & Ni, H. (2021). The influence of temperature on the capacity of lithium ion batteries with different anodes. Energies, 15(1), 60.
    '''
    B_E_loss_due_to_low_temp = np.interp( 
            x = air_temp_C_t,
            xp = [-60, -30, 0, 15, 25, 40, 70], 
            fp = [0, 0.5, 0.9, 1, 1, 1, 1] )
    return B_E_loss_due_to_low_temp
