# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 13:05:20 2022

@author: ruzhu
"""

import DEMS as EMS
import utils
from matplotlib import pyplot as plt
import pandas as pd
from hydesign.examples import examples_filepath




parameter_dict = {
        
        # hpp parameters
        'hpp_grid_connection': 100,  # in MW

        # hpp wind parameters
        'wind_capacity': 120, #in MW


        # hpp solar parameters
        'solar_capacity': 10,  # in MW
       

        # hpp battery parameters
        'battery_energy_capacity': 120,  # in MWh
        'battery_power_capacity': 40,  # in MW
        'battery_minimum_SoC': 0.1,
        'battery_maximum_SoC': 0.9,
        'battery_initial_SoC': 0.1,
        'battery_hour_discharge_efficiency': 0.985,  #
        'battery_hour_charge_efficiency': 0.975,
        'battery_self_discharge_efficiency': 0,
        # hpp battery degradation parameters
        'battery_initial_degradation': 0,  
        'battery_marginal_degradation_cost': 142000, # in /MWh
        'battery_capital_cost': 142000, # in /MWh
        'degradation_in_optimization': 0, # 1:yes 0:no
        
        # bid parameters
        'max_up_bid': 50,
        'max_dw_bid': 50,
        'min_up_bid': 5,
        'min_dw_bid': 5,
        
        # interval parameters: note that DI must <= SI
        'dispatch_interval': 1/4,
        'settlement_interval': 1/4,
        
        'imbalance_fee': 0.13,  # DK: 0.13 €/MWh, other Nordic countries: , others: 0.001
    }

simulation_dict = {
        'wind_as_component': 1,
        'solar_as_component': 1,  # The code does not support for solar power plant
        'battery_as_component': 1,
        'start_date': '1/1/22',
        'number_of_run_day': 360,   # 
        'out_dir':"./test/",

        'DA_wind': "DA",   #DA, Measurement
        'HA_wind': "HA" ,  #HA, Measurement
        'FMA_wind':"RT",#5min_ahead, Measurement
        'DA_solar': "DA",
        'HA_solar': "HA",
        'FMA_solar': "RT",
        'SP': "SM_forecast",  # SM_forecast;SM_cleared
        'RP': "reg_forecast", #reg_cleared;reg_forecast_pre
        'BP': 1, #1:forecast value 2: perfect value
        
        # Data
        'wind_fn': examples_filepath + "HiFiEMS_inputs/Winddata2021_15min.csv",
        'solar_fn': examples_filepath + "HiFiEMS_inputs/Solardata2021_15min.csv",
        'market_fn': examples_filepath + "HiFiEMS_inputs/Market2021.csv",
        
        # for DDEMS (spot market) -- Historical data
        'history_wind_fn': examples_filepath + "HiFiEMS_inputs/Winddata2022_15min.csv",
        'history_market_fn': examples_filepath + "HiFiEMS_inputs/Market2021.csv",
        
        # for REMS (balancing market)
        'HA_wind_error_ub': "5%_fc_error",
        'HA_wind_error_lb': "95%_fc_error",
        
        # for SEMS
        #'wind_scenario_fn': "../Data/Winddata2022_15min.csv",  # "../Data/probabilistic_wind2022.csv"
        'price_scenario_fn': None,  # "../Data/xxx.csv", if None then use the build in method to generate price scenarios
        'number_of_wind_scenario': 3, 
        'number_of_price_scenario': 3, 
    }

out_keys = ['P_HPP_SM_t_opt',
                'SM_price_cleared',
                'BM_dw_price_cleared',
                'BM_up_price_cleared',
                'P_HPP_RT_ts',
                'P_HPP_RT_refs',
                'P_HPP_UP_bid_ts',
                'P_HPP_DW_bid_ts',
                's_UP_t',
                's_DW_t',
                'residual_imbalance',
                'RES_RT_cur_ts',
                'P_dis_RT_ts',
                'P_cha_RT_ts',
                'SoC_ts']
res = utils.run(
        parameter_dict = parameter_dict,
        simulation_dict = simulation_dict,
        EMS = EMS,
        EMStype="DEMS",
        BM_model=False,
        RD_model=False
       )   # run EMS with only spot market optimization
    
for k, r in zip(out_keys, res):
    print(k, r.shape)

#EMS.run_SM_RD(
#        parameter_dict = parameter_dict,
#        simulation_dict = simulation_dict
#       )   # run EMS with spot market optimization and re-dispatch optimization

#EMS.run_SM_BM(
#        parameter_dict = parameter_dict,
#        simulation_dict = simulation_dict
#       )   # run EMS with spot market optimization and balancing market optimization

#EMS.run(
#        parameter_dict = parameter_dict,
#        simulation_dict = simulation_dict
#       )   # run EMS with all optimization models    


   
    