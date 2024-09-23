# -*- coding: utf-8 -*-
"""

Created on Mon Mar 15 11:42:56 2021

@author: ruzhu
"""

import pandas as pd
import numpy as np
from numpy import matlib as mb
import rainflow
import math
from hydesign.HiFiEMS import Deg_Calculation as DegCal
#import random
#import matplotlib.pyplot as plt
from docplex.mp.model import Model
import os
import time
import openpyxl
from datetime import datetime
#from scipy import interpolate

def ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict): 
    datetime_str = simulation_dict["start_date"]
    day_num_start = datetime.strptime(datetime_str, '%m/%d/%y').timetuple().tm_yday

    # skips1 = range(1, ((day_num - 1 + day_num_start - 1) * T)%(359*T) + 1)
    # skips2 = range(1, ((day_num - 1 + day_num_start - 1) * 24)%(359*24) + 1)

    skips1 = ((day_num - 1 + day_num_start - 1) * T)%(359*T)
    skips2 = ((day_num - 1 + day_num_start - 1) * 24)%(359*24)

    # Wind_data = pd.read_csv(simulation_dict["wind_fn"], skiprows = skips1, nrows=T+exten_num)
    # Solar_data = pd.read_csv(simulation_dict["solar_fn"], skiprows = skips1, nrows=T+exten_num)
    # Market_data = pd.read_csv(simulation_dict["market_fn"], skiprows = skips2, nrows=int(T/DI_num)+int(exten_num/DI_num))
    
    Wind_data = simulation_dict["wind_df"].iloc[skips1: skips1+T+exten_num]
    Solar_data = simulation_dict["solar_df"].iloc[skips1: skips1+T+exten_num]
    Market_data = simulation_dict["market_df"].iloc[skips2: skips2 + int(T/DI_num)+int(exten_num/DI_num)]
    
    Wind_data.reset_index(inplace=True)
    Solar_data.reset_index(inplace=True)
    Market_data.reset_index(inplace=True)
    
    if 'Measurement' in Wind_data:
        Wind_measurement = Wind_data['Measurement'] * PwMax
    else:
        Wind_measurement = None
    if 'Measurement' in Solar_data:
        Solar_measurement = Solar_data['Measurement'] * PsMax
    else:
        Solar_measurement = None


    DA_wind_forecast = Wind_data[simulation_dict["DA_wind"]] * PwMax
    #DA_wind_forecast = Wind_data['Measurement'] * PwMax
    HA_wind_forecast = Wind_data[simulation_dict["HA_wind"]] * PwMax
    #HA_wind_forecast = Wind_data['Measurement'] * PwMax
    RT_wind_forecast = Wind_data[simulation_dict["FMA_wind"]] * PwMax
    #RT_wind_forecast = Wind_data['Measurement'] * PwMax

    DA_solar_forecast = Solar_data[simulation_dict["DA_solar"]] * PsMax
    HA_solar_forecast = Solar_data[simulation_dict["HA_solar"]] * PsMax
    #HA_solar_forecast = Solar_data['Measurement'] * PsMax
    RT_solar_forecast = Solar_data[simulation_dict["FMA_solar"]] * PsMax
    #RT_solar_forecast = Solar_data['Measurement'] * PsMax

    SM_price_cleared = Market_data['SM_cleared'] 
    SM_price_forecast = Market_data[simulation_dict["SP"]] 
    #SM_price_forecast = Market_data['SM_cleared'] 

    Reg_price_forecast = Market_data[simulation_dict["RP"]]
    Reg_price_cleared = Market_data['reg_cleared']
    #Reg_price_forecast = Market_data['reg_cleared']

    BM_dw_price_forecasts = []
    BM_up_price_forecasts = []
    reg_up_sign_forecasts = []
    reg_dw_sign_forecasts = []
    
    for i in range(int(T/DI_num)+int(exten_num/DI_num)):
        if Reg_price_forecast.iloc[i] > SM_price_cleared.iloc[i]:
            BM_up_price_forecast_i = Reg_price_forecast.iloc[i]
            BM_dw_price_forecast_i = SM_price_cleared.iloc[i]
            reg_up_sign_forecast_i = 1
            reg_dw_sign_forecast_i = 0
            
        elif Reg_price_forecast.iloc[i] < SM_price_cleared.iloc[i]:
            BM_up_price_forecast_i = SM_price_cleared.iloc[i]
            BM_dw_price_forecast_i = Reg_price_forecast.iloc[i]
            reg_up_sign_forecast_i = 0
            reg_dw_sign_forecast_i = 1
            
        else:
            BM_up_price_forecast_i = SM_price_cleared.iloc[i]
            BM_dw_price_forecast_i = SM_price_cleared.iloc[i]
            reg_up_sign_forecast_i = 0
            reg_dw_sign_forecast_i = 0

        BM_dw_price_forecasts.append({'Up': BM_up_price_forecast_i})
        BM_up_price_forecasts.append({'Down': BM_dw_price_forecast_i})
        reg_up_sign_forecasts.append({'up_sign': reg_up_sign_forecast_i})
        reg_dw_sign_forecasts.append({'dw_sign': reg_dw_sign_forecast_i})
    
    BM_dw_price_forecast = pd.DataFrame(BM_dw_price_forecasts).squeeze()
    BM_up_price_forecast = pd.DataFrame(BM_up_price_forecasts).squeeze()
    reg_up_sign_forecast = pd.DataFrame(reg_up_sign_forecasts).squeeze()
    reg_dw_sign_forecast = pd.DataFrame(reg_dw_sign_forecasts).squeeze()        
    
    if simulation_dict["BP"] == 2:
       BM_dw_price_forecast = Market_data['BM_Down_cleared']
       BM_up_price_forecast = Market_data['BM_Up_cleared']
    
    
    BM_dw_price_cleared = Market_data['BM_Down_cleared']
    BM_up_price_cleared = Market_data['BM_Up_cleared']
    
  

    reg_vol_up = Market_data['reg_vol_Up']
    reg_vol_dw = Market_data['reg_vol_Down']
    
    time_index = Wind_data['time']
    return DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index




def f_xmin_to_ymin(x,reso_x, reso_y):  #x: dataframe reso: in hour
    y = pd.DataFrame()
    
    a=0
    num = int(reso_y/reso_x)
    
    for ii in range(len(x)):        
        if ii%num == num-1:
            a = (a + x.iloc[ii][0]) /num   
            y = y.append(pd.DataFrame([a]))
            a = 0
        else:                       
            a = a + x.iloc[ii][0]     
    y.index = range(int(len(x)/num))
    return y


def get_var_value_from_sol(x, sol):
    
    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient='index')
    
    return y
   


def SMOpt(dt, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                    DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator, verbose=False):
    
    # Optimization modelling by CPLEX
    setT = [i for i in range(T)] 
    set_SoCT = [i for i in range(T + 1)] 
    setK = [i for i in range(24)] 
    dt_num = int(1/dt)

    eta_cha_ha = eta_cha**(1/dt_num)
    eta_dis_ha = eta_dis**(1/dt_num)
    eta_leak_ha = 1 - (1-eta_leak)**(1/dt_num)


    SMOpt_mdl = Model()
  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_SM_t = SMOpt_mdl.continuous_var_dict(setT, name='SM bidding 5min')
    P_HPP_SM_k = SMOpt_mdl.continuous_var_dict(setK, name='SM bidding H')
    P_W_SM_t   = {t: SMOpt_mdl.continuous_var(lb=0, ub=DA_wind_forecast[t], name="SM Wind bidding {}".format(t)) for t in setT}
    P_S_SM_t   = {t: SMOpt_mdl.continuous_var(lb=0, ub=DA_solar_forecast[t], name="DA Solar bidding {}".format(t)) for t in setT}
    P_dis_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM discharge') 
    P_cha_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='SM charge') 
    P_b_SM_t   = SMOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='SM Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_SM_t   = SMOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='SM SoC')
    z_t        = SMOpt_mdl.binary_var_dict(setT, name='Cha or Discha')
    #an_var     = SMOpt_mdl.continuous_var(lb=0, ub=0.5, name='anciliary var')
 #   z_t        = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=0.4, name='Cha or Discha')
  # Define constraints
    for t in setT:
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] == P_W_SM_t[t] + P_S_SM_t[t] + P_b_SM_t[t])
        SMOpt_mdl.add_constraint(P_b_SM_t[t]   == P_dis_SM_t[t] - P_cha_SM_t[t])
        SMOpt_mdl.add_constraint(P_dis_SM_t[t] <= (PbMax - PreUp) * z_t[t] )
        SMOpt_mdl.add_constraint(P_cha_SM_t[t] <= (PbMax - PreDw) * (1-z_t[t]))
        SMOpt_mdl.add_constraint(SoC_SM_t[t+1]  == SoC_SM_t[t] * (1-eta_leak_ha) - 1/Emax * P_dis_SM_t[t]/eta_dis_ha * dt + 1/Emax * P_cha_SM_t[t] * eta_cha_ha * dt)
        SMOpt_mdl.add_constraint(SoC_SM_t[t+1]   <= SoCmax )
        SMOpt_mdl.add_constraint(SoC_SM_t[t+1]   >= SoCmin )
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] <= P_grid_limit - PreUp)
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] >= -P_grid_limit + PreDw)
    for k in setK:
        for t in setT:
            if t//dt_num == k:
               SMOpt_mdl.add_constraint(P_HPP_SM_k[k] == P_HPP_SM_t[t])
   
    SMOpt_mdl.add_constraint(SoC_SM_t[0] == SoC0)



  # Define objective function
    Revenue = SMOpt_mdl.sum(SM_price_forecast[t] * P_HPP_SM_t[t] *dt for t in setT) 
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * SMOpt_mdl.sum((P_dis_SM_t[t]+P_cha_SM_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0

    SMOpt_mdl.maximize(Revenue - Deg_cost)

  # Solve SMOpt Model
    if verbose:
        SMOpt_mdl.print_information()
    sol = SMOpt_mdl.solve()

    if sol:
        P_HPP_SM_t_opt = get_var_value_from_sol(P_HPP_SM_t, sol)        
        P_HPP_SM_k_opt = get_var_value_from_sol(P_HPP_SM_k, sol)
        P_HPP_SM_t_opt.columns = ['SM']
        
        
        P_W_SM_t_opt = get_var_value_from_sol(P_W_SM_t, sol) 
        P_S_SM_t_opt = get_var_value_from_sol(P_S_SM_t, sol) 
        P_dis_SM_t_opt = get_var_value_from_sol(P_dis_SM_t, sol) 
        P_cha_SM_t_opt = get_var_value_from_sol(P_cha_SM_t, sol) 
        SoC_SM_t_opt = get_var_value_from_sol(SoC_SM_t, sol) 
        
        

        E_HPP_SM_t_opt = P_HPP_SM_t_opt * dt

        P_W_SM_cur_t_opt = np.array(DA_wind_forecast[:T].T) - np.array(P_W_SM_t_opt).flatten()
        P_W_SM_cur_t_opt = pd.DataFrame(P_W_SM_cur_t_opt)
        P_S_SM_cur_t_opt = np.array(DA_solar_forecast[:T].T) - np.array(P_S_SM_t_opt).flatten()
        P_S_SM_cur_t_opt = pd.DataFrame(P_S_SM_cur_t_opt)


        z_t_opt = get_var_value_from_sol(z_t, sol) 
        if verbose:
            print(P_HPP_SM_t_opt)
            print(P_dis_SM_t_opt)
            print(P_cha_SM_t_opt)
            print(SoC_SM_t_opt)
            print(z_t_opt)


    else:
        print("DA EMS Model has no solution")
    return E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt



def BMOpt(dt, ds, dk, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    HA_wind_forecast, HA_solar_forecast, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, start, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator):
    
    # Optimization modelling by CPLEX

    dt_num = int(1/dt) #DI
    

    dk_num = int(1/dk) #BI
    T_dk = int(24/dk)
    
    ds_num = int(1/ds) #SI
    T_ds = int(24/ds)
    dsdt_num = int(ds/dt) 

    eta_cha_ha = eta_cha**(1/dt_num)
    eta_dis_ha = eta_dis**(1/dt_num)
    eta_leak_ha = 1 - (1-eta_leak)**(1/dt_num)

    
    reg_up_sign_forecast1 = reg_up_sign_forecast.squeeze().repeat(dt_num)
    reg_dw_sign_forecast1 = reg_dw_sign_forecast.squeeze().repeat(dt_num)
    reg_up_sign_forecast1.index = range(T + exten_num)
    reg_dw_sign_forecast1.index = range(T + exten_num)

    setT = [i for i in range(start*dt_num, T + exten_num)] 
    setT1 = [i for i in range((start + 1) * dt_num, T + exten_num)] 
    setK = [i for i in range(start*dk_num, T_dk + int(exten_num/dt_num))]
    setK1 = [i for i in range((start + 1) * dk_num, T_dk + int(exten_num/dt_num))]
    setS = [i for i in range(start*ds_num, T_ds + int(exten_num/dsdt_num))]
    set_SoCT = [i for i in range(start*dt_num, T + 1 + exten_num)] 

    BMOpt_mdl = Model()

  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_UP_t = BMOpt_mdl.continuous_var_dict(setT1, lb=0, ub=P_grid_limit, name='BM UP bidding 5min')
    P_HPP_DW_t = BMOpt_mdl.continuous_var_dict(setT1, lb=0, ub=P_grid_limit, name='BM DW bidding 5min')
    P_HPP_UP_k = BMOpt_mdl.continuous_var_dict(setK1, lb=0, ub=P_grid_limit, name='BM UP bidding H')
    P_HPP_DW_k = BMOpt_mdl.continuous_var_dict(setK1, lb=0, ub=P_grid_limit, name='BM DW bidding H')

    P_HPP_HA_t = BMOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_W_HA_t   = {t: BMOpt_mdl.continuous_var(lb=0, ub=HA_wind_forecast[t-start*dt_num], name="HA Wind schedule {}".format(t)) for t in setT}
    P_S_HA_t   = {t: BMOpt_mdl.continuous_var(lb=0, ub=HA_solar_forecast[t-start*dt_num], name="HA Solar schedule {}".format(t)) for t in setT}
    P_dis_HA_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA discharge') 
    P_cha_HA_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA charge') 
    P_b_HA_t   = BMOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)

    SoC_HA_t   = BMOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = BMOpt_mdl.binary_var_dict(setT, name='Cha or Discha')

    delta_P_HPP_s = BMOpt_mdl.continuous_var_dict(setS, lb=-P_grid_limit, ub=P_grid_limit, name='HA imbalance')
    delta_P_HPP_UP_s = BMOpt_mdl.continuous_var_dict(setS, lb=0, name='HA up imbalance')
    delta_P_HPP_DW_s = BMOpt_mdl.continuous_var_dict(setS, lb=0, name='HA dw imbalance')

    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  #dataframe to series
  # Define constraints
    for t in setT:

        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] == P_W_HA_t[t] + P_S_HA_t[t] + P_b_HA_t[t])
        BMOpt_mdl.add_constraint(P_b_HA_t[t]   == P_dis_HA_t[t] - P_cha_HA_t[t])
        BMOpt_mdl.add_constraint(P_dis_HA_t[t] <= (PbMax - PreUp ) * z_t[t] )
        BMOpt_mdl.add_constraint(P_cha_HA_t[t] <= (PbMax - PreDw) * (1-z_t[t]))
        
        BMOpt_mdl.add_constraint(SoC_HA_t[t + 1] == SoC_HA_t[t] * (1-eta_leak_ha) - 1/Emax * (P_dis_HA_t[t])/eta_dis_ha * dt + 1/Emax * (P_cha_HA_t[t]) * eta_cha_ha * dt)

        BMOpt_mdl.add_constraint(SoC_HA_t[t] <= SoCmax + 1/Emax * (-PreDw*eta_cha_ha) * dt)
        BMOpt_mdl.add_constraint(SoC_HA_t[t] >= SoCmin + 1/Emax * (PreUp/eta_dis_ha) * dt)
        
        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] <= P_grid_limit - PreUp )
        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] >= -P_grid_limit + PreDw) 
        
    for s in setS:
        if s < (start + 1) * ds_num:        
        #BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (s_UP_t[s * dsdt_num + m] * P_HPP_UP_t[s * dsdt_num + m] - s_DW_t[s * dsdt_num + m] * P_HPP_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
            BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (P_HPP_UP_t0 * s_UP_t[s * dsdt_num + m] - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
        else:
            BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (reg_up_sign_forecast1[s * dsdt_num + m] * P_HPP_UP_t[s * dsdt_num + m] - reg_dw_sign_forecast1[s * dsdt_num + m] * P_HPP_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
        BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s])
       
    for k in setK1:
        for j in range(0, dt_num):
                
                BMOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
                BMOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])
 

    BMOpt_mdl.add_constraint(SoC_HA_t[start*dt_num] == SoC0)


    Revenue = BMOpt_mdl.sum(BM_up_price_forecast[k] * reg_up_sign_forecast[k] * P_HPP_UP_k[k] *dk - BM_dw_price_forecast[k] * reg_dw_sign_forecast[k] * P_HPP_DW_k[k] *dk for k in setK1) + BMOpt_mdl.sum((BM_dw_price_forecast_settle[s]-0.001) * delta_P_HPP_UP_s[s] *ds - (BM_up_price_forecast_settle[s]+0.001) * delta_P_HPP_DW_s[s] *ds for s in setS)
    
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * BMOpt_mdl.sum((P_dis_HA_t[t] + P_cha_HA_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0 

    BMOpt_mdl.maximize(Revenue - Deg_cost)

  # Solve BMOpt Model
    BMOpt_mdl.print_information()
    print('BMOpt is running')
    sol = BMOpt_mdl.solve()
    aa = BMOpt_mdl.get_solve_details()
    print(aa.status)
    if sol:

        P_HPP_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_HA_t), orient='index').reindex(setT, fill_value=0)
        P_HPP_HA_t_opt.columns = ['HA']
        P_W_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_W_HA_t), orient='index').reindex(setT, fill_value=0)
        P_S_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_S_HA_t), orient='index').reindex(setT, fill_value=0)
        P_dis_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_dis_HA_t), orient='index').reindex(setT, fill_value=0)
        P_cha_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_cha_HA_t), orient='index').reindex(setT, fill_value=0)

        SoC_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_HA_t), orient='index').reindex(set_SoCT, fill_value=0)
        P_HPP_UP_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_t), orient='index').reindex(setT1, fill_value=0)
        P_HPP_DW_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_t), orient='index').reindex(setT1, fill_value=0)
        P_HPP_UP_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_k), orient='index').reindex(setK1, fill_value=0)
        P_HPP_DW_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_k), orient='index').reindex(setK1, fill_value=0)
        delta_P_HPP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_s), orient='index').reindex(setS, fill_value=0)
        delta_P_HPP_UP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_UP_s), orient='index').reindex(setS, fill_value=0)
        delta_P_HPP_DW_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_DW_s), orient='index').reindex(setS, fill_value=0)


        E_HPP_HA_t_opt = P_HPP_HA_t_opt * dt

        P_W_HA_cur_t_opt = np.array(HA_wind_forecast[0:].T) - np.array(P_W_HA_t_opt).flatten()
        P_W_HA_cur_t_opt = pd.DataFrame(P_W_HA_cur_t_opt)
        P_S_HA_cur_t_opt = np.array(HA_solar_forecast[0:].T) - np.array(P_S_HA_t_opt).flatten()
        P_S_HA_cur_t_opt = pd.DataFrame(P_S_HA_cur_t_opt)


        z_ts = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient='index').reindex(setT, fill_value=0)

    else:
        print("BMOpt has no solution")
        #print(SMOpt_mdl.export_to_string())
    return E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt  


    
def RDOpt(dt, ds, dk, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    RD_wind_forecast, RD_solar_forecast, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, start, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, P_HPP_UP_t1, P_HPP_DW_t1, SoC0, exist_imbalance, exten_num, deg_indicator):
          
    # Optimization modelling by CPLEX
    dt_num = int(1/dt) #DI
    

    dk_num = int(1/dk) #BI
    T_dk = int(24/dk)
    
    ds_num = int(1/ds) #SI
    T_ds = int(24/ds)
    dsdt_num = int(ds/dt) 

    eta_cha_ha = eta_cha**(1/dt_num)
    eta_dis_ha = eta_dis**(1/dt_num)
    eta_leak_ha = 1 - (1-eta_leak)**(1/dt_num)
    
    reg_up_sign_forecast1 = reg_up_sign_forecast.repeat(dt_num)
    reg_dw_sign_forecast1 = reg_dw_sign_forecast.repeat(dt_num)
    reg_up_sign_forecast1.index = range(T + exten_num)
    reg_dw_sign_forecast1.index = range(T + exten_num)
    
    current_SI = start//dsdt_num
    current_hour = start//dt_num
    setT = [i for i in range(start, T + exten_num)]
    setT1 = [i for i in range((current_hour + 2) * dt_num, T + exten_num)] 
    setK = [i for i in range(current_hour * dk_num, T_dk + int(exten_num/dt_num))]
    setK1 = [i for i in range((current_hour + 2) * dk_num, T_dk + int(exten_num/dt_num))]
    setS = [i for i in range(current_SI, T_ds + int(exten_num/dsdt_num))]
    set_SoCT = [i for i in range(start, T + 1 + exten_num)] 
    print('RDOpt model') 
    RDOpt_mdl = Model()
    print('RDOpt model is constructed')
  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_UP_t = RDOpt_mdl.continuous_var_dict(setT1, lb=0, name='BM UP bidding 5min')
    P_HPP_DW_t = RDOpt_mdl.continuous_var_dict(setT1, lb=0, name='BM DW bidding 5min')
    P_HPP_UP_k = RDOpt_mdl.continuous_var_dict(setK1, lb=0, name='BM UP bidding')
    P_HPP_DW_k = RDOpt_mdl.continuous_var_dict(setK1, lb=0, name='BM DW bidding')
    
    P_HPP_RD_t = RDOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_W_RD_t   = {t: RDOpt_mdl.continuous_var(lb=0, ub=RD_wind_forecast[t-start], name="HA Wind schedule {}".format(t)) for t in setT}
    P_S_RD_t   = {t: RDOpt_mdl.continuous_var(lb=0, ub=RD_solar_forecast[t-start], name="HA Solar schedule {}".format(t)) for t in setT}
    P_dis_RD_t = RDOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA discharge') 
    P_cha_RD_t = RDOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA charge') 
    P_b_RD_t   = RDOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RD_t   = RDOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = RDOpt_mdl.binary_var_dict(setT, name='Cha or Discha')
   
    delta_P_HPP_s = RDOpt_mdl.continuous_var_dict(setS, lb=-P_grid_limit, ub=P_grid_limit, name='HA imbalance')
    delta_P_HPP_UP_s = RDOpt_mdl.continuous_var_dict(setS, lb=0, name='HA up imbalance')
    delta_P_HPP_DW_s = RDOpt_mdl.continuous_var_dict(setS, lb=0, name='HA dw imbalance')
   
    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  #dataframe to series
  # Define constraints
    for t in setT:
        
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] == P_W_RD_t[t] + P_S_RD_t[t] + P_b_RD_t[t])
        RDOpt_mdl.add_constraint(P_b_RD_t[t]   == P_dis_RD_t[t] - P_cha_RD_t[t])
        RDOpt_mdl.add_constraint(P_dis_RD_t[t] <= (PbMax - PreUp) * z_t[t] )
        RDOpt_mdl.add_constraint(P_cha_RD_t[t] <= (PbMax - PreDw) * (1-z_t[t]))

        RDOpt_mdl.add_constraint(SoC_RD_t[t + 1] == SoC_RD_t[t] * (1-eta_leak_ha) - 1/Emax * P_dis_RD_t[t]/eta_dis_ha * dt + 1/Emax * P_cha_RD_t[t] * eta_cha_ha * dt)
        
        RDOpt_mdl.add_constraint(SoC_RD_t[t]   <= SoCmax + 1/Emax * (- PreDw * eta_cha_ha) * dt )
        RDOpt_mdl.add_constraint(SoC_RD_t[t]   >= SoCmin + 1/Emax * (PreUp/eta_dis_ha) * dt ) 
       
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] <= P_grid_limit - PreUp)
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] >= -P_grid_limit + PreDw)
        
    for s in setS:
        RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s])
        if s < (current_hour + 1)*ds_num:            
            if s == current_SI:
               RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == (exist_imbalance + RDOpt_mdl.sum((P_HPP_RD_t[s * dsdt_num + j] - (P_HPP_UP_t0 * s_UP_t[s * dsdt_num + j] - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j]) * dt for j in range(start%dsdt_num, dsdt_num)))/ds)
            else:
           #RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == RDOpt_mdl.sum(P_HPP_RD_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
               RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == RDOpt_mdl.sum(P_HPP_RD_t[s * dsdt_num + j] - (P_HPP_UP_t0 * s_UP_t[s * dsdt_num + j] - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
        elif s >= (current_hour + 1)*ds_num and s < (current_hour + 2)*ds_num:
            RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == RDOpt_mdl.sum(P_HPP_RD_t[s * dsdt_num + j] - (reg_up_sign_forecast1[s * dsdt_num + j] * P_HPP_UP_t1 - reg_dw_sign_forecast1[s * dsdt_num + j] * P_HPP_DW_t1) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
        else:
            RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == RDOpt_mdl.sum(P_HPP_RD_t[s * dsdt_num + j] - (reg_up_sign_forecast1[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - reg_dw_sign_forecast1[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
            
    for k in setK1:
        for j in range(0, dt_num):
            RDOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
            RDOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])



    RDOpt_mdl.add_constraint(SoC_RD_t[start] == SoC0)

    

    Revenue = RDOpt_mdl.sum(BM_up_price_forecast[k] * reg_up_sign_forecast[k] * P_HPP_UP_k[k] *dk - BM_dw_price_forecast[k] * reg_dw_sign_forecast[k] * P_HPP_DW_k[k] *dk for k in setK1) + RDOpt_mdl.sum((BM_dw_price_forecast_settle[s]-0.001) * delta_P_HPP_UP_s[s] *ds - (BM_up_price_forecast_settle[s]+0.001) * delta_P_HPP_DW_s[s] *ds for s in setS)
    
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * RDOpt_mdl.sum((P_dis_RD_t[t] + P_cha_RD_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0 

    RDOpt_mdl.maximize(Revenue - Deg_cost)

  # Solve BMOpt Model
    RDOpt_mdl.print_information()
    print('RDOpt is running')
    sol = RDOpt_mdl.solve()
    aa = RDOpt_mdl.get_solve_details()    
    print(aa.status)

    if sol:

        P_HPP_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_RD_t), orient='index').reindex(setT, fill_value=0)
        P_HPP_RD_t_opt.columns = ['RD']
        P_W_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_W_RD_t), orient='index').reindex(setT, fill_value=0)
        P_S_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_S_RD_t), orient='index').reindex(setT, fill_value=0)
        P_dis_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_dis_RD_t), orient='index').reindex(setT, fill_value=0)
        P_cha_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_cha_RD_t), orient='index').reindex(setT, fill_value=0)
        SoC_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_RD_t), orient='index').reindex(set_SoCT, fill_value=0)
        P_HPP_UP_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_t), orient='index').reindex(setT1, fill_value=0)
        P_HPP_DW_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_t), orient='index').reindex(setT1, fill_value=0)
        P_HPP_UP_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_k), orient='index').reindex(setK1, fill_value=0)
        P_HPP_DW_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_k), orient='index').reindex(setK1, fill_value=0)
        delta_P_HPP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_s), orient='index').reindex(setS, fill_value=0)
        delta_P_HPP_UP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_UP_s), orient='index').reindex(setS, fill_value=0)
        delta_P_HPP_DW_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_DW_s), orient='index').reindex(setS, fill_value=0)



        E_HPP_RD_t_opt = P_HPP_RD_t_opt * dt

        P_W_RD_cur_t_opt = np.array(RD_wind_forecast[0:].T) - np.array(P_W_RD_t_opt).flatten()
        P_W_RD_cur_t_opt = pd.DataFrame(P_W_RD_cur_t_opt)
        P_S_RD_cur_t_opt = np.array(RD_solar_forecast[0:].T) - np.array(P_S_RD_t_opt).flatten()
        P_S_RD_cur_t_opt = pd.DataFrame(P_S_RD_cur_t_opt)


        z_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient='index')


    else:
        print("RDOpt has no solution")
        #print(SMOpt_mdl.expoRD_to_string())
    return E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt  

def RTSim(dt, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_t0, start, P_activated_UP_t, P_activated_DW_t):  
    RES_error = Wind_measurement[start] + Solar_measurement[start] - RT_wind_forecast[start] - RT_solar_forecast[start] 


    eta_cha_ha = eta_cha**(dt)
    eta_dis_ha = eta_dis**(dt)
    eta_leak_ha = 1 - (1-eta_leak)**(dt)

    # Optimization modelling by CPLEX    
    set_SoCT = [0, 1] 
    RTSim_mdl = Model()
  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_W_RT_t   = RTSim_mdl.continuous_var(lb=0, ub=Wind_measurement[start], name='HA Wind schedule')
    P_S_RT_t   = RTSim_mdl.continuous_var(lb=0, ub=Solar_measurement[start], name='HA Solar schedule')
    P_HPP_RT_t = RTSim_mdl.continuous_var(lb=-P_grid_limit, ub=P_grid_limit, name='HA schedule without balancing bidding')
    P_dis_RT_t = RTSim_mdl.continuous_var(lb=0, ub=PbMax, name='HA discharge') 
    P_cha_RT_t = RTSim_mdl.continuous_var(lb=0, ub=PbMax, name='HA charge') 
    P_b_RT_t   = RTSim_mdl.continuous_var(lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RT_t   = RTSim_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = RTSim_mdl.binary_var(name='Cha or Discha')
    
  # Define constraints

    RTSim_mdl.add_constraint(P_HPP_RT_t == P_W_RT_t + P_S_RT_t + P_b_RT_t)
    RTSim_mdl.add_constraint(P_b_RT_t == P_dis_RT_t - P_cha_RT_t)
    RTSim_mdl.add_constraint(P_dis_RT_t <= (PbMax - PreUp) * z_t )
    RTSim_mdl.add_constraint(P_cha_RT_t <= (PbMax - PreDw) * (1-z_t))
    RTSim_mdl.add_constraint(SoC_RT_t[1] == SoC_RT_t[0] * (1-eta_leak_ha) - 1/Emax * P_dis_RT_t/eta_dis_ha * dt + 1/Emax * P_cha_RT_t * eta_cha_ha * dt)
    RTSim_mdl.add_constraint(SoC_RT_t[0]   <= SoCmax )
    RTSim_mdl.add_constraint(SoC_RT_t[0]   >= SoCmin )
    RTSim_mdl.add_constraint(P_HPP_RT_t <= P_grid_limit - PreUp)
    RTSim_mdl.add_constraint(P_HPP_RT_t >= -P_grid_limit + PreDw)
    RTSim_mdl.add_constraint(SoC_RT_t[0] == SoC0)
    
    
    if math.isclose(P_activated_UP_t, 0, abs_tol=1e-5) and math.isclose(P_activated_DW_t, 0, abs_tol=1e-5):
        obj = 1e5 * (Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t - P_S_RT_t) + (P_HPP_RT_t - P_HPP_t0) * (P_HPP_RT_t - P_HPP_t0)
    else:
        obj = (Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t - P_S_RT_t) + 1e5*(P_HPP_RT_t - P_HPP_t0) * (P_HPP_RT_t - P_HPP_t0)
    
    
   
    RTSim_mdl.minimize(obj)

  # Solve BMOpt Model
    RTSim_mdl.print_information()
    sol = RTSim_mdl.solve()
    aa = RTSim_mdl.get_solve_details()
    print(aa.status)
    if sol:
    #    SMOpt_mdl.print_solution()
        #imbalance_RT_to_ref = sol.get_objective_value() * dt
        P_HPP_RT_t_opt = sol.get_value(P_HPP_RT_t)
        P_W_RT_t_opt = sol.get_value(P_W_RT_t)
        P_S_RT_t_opt = sol.get_value(P_S_RT_t)
        P_dis_RT_t_opt = sol.get_value(P_dis_RT_t)
        P_cha_RT_t_opt = sol.get_value(P_cha_RT_t)
        SoC_RT_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_RT_t), orient='index')
        E_HPP_RT_t_opt = P_HPP_RT_t_opt * dt
        
        RES_RT_cur_t_opt = Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t_opt - P_S_RT_t_opt
        #P_W_RT_cur_t_opt = Wind_measurement[start] - P_W_RT_t_opt
        #P_W_RT_cur_t_opt = pd.DataFrame(P_W_RT_cur_t_opt)
        #P_S_RT_cur_t_opt = Solar_measurement[start] - P_S_RT_t_opt
        #P_S_RT_cur_t_opt = pd.DataFrame(P_S_RT_cur_t_opt)


        z_t_opt = sol.get_value(z_t)

    else:
        print("RTOpt has no solution")
        #print(SMOpt_mdl.export_to_string())
    return E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt

def RBOpt(dt, ds, dk, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    RB_wind_forecast, RB_solar_forecast, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, start, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, P_HPP_UP_t1, P_HPP_DW_t1, SoC0, exist_imbalance, exten_num, deg_indicator):
    
    # Optimization modelling by CPLEX
    dt_num = int(1/dt) #DI
    

    dk_num = int(1/dk) #BI
    T_dk = int(24/dk)
    
    ds_num = int(1/ds) #SI
    T_ds = int(24/ds)
    dsdt_num = int(ds/dt) 

    eta_cha_ha = eta_cha**(1/dt_num)
    eta_dis_ha = eta_dis**(1/dt_num)
    eta_leak_ha = 1 - (1-eta_leak)**(1/dt_num)

    
    reg_up_sign_forecast1 = reg_up_sign_forecast.repeat(dt_num)
    reg_dw_sign_forecast1 = reg_dw_sign_forecast.repeat(dt_num)
    reg_up_sign_forecast1.index = range(T + exten_num)
    reg_dw_sign_forecast1.index = range(T + exten_num)
    
    current_SI = start//dsdt_num
    current_hour = start//dt_num
    setT = [i for i in range(start, T + exten_num)]
    setT1 = [i for i in range((current_hour + 2) * dt_num, T + exten_num)] 
    setK = [i for i in range(current_hour * dk_num, T_dk + int(exten_num/dt_num))]
    setK1 = [i for i in range((current_hour + 2) * dk_num, T_dk + int(exten_num/dt_num))]
    setS = [i for i in range(current_SI, T_ds + int(exten_num/dsdt_num))]
    set_SoCT = [i for i in range(start, T + 1 + exten_num)] 
    print('RBOpt model')
    RBOpt_mdl = Model()
    print('RBOpt model is constructed')
  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    #P_HPP_all_t = BMOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_HPP_RB_t = RBOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_W_RB_t   = {t: RBOpt_mdl.continuous_var(lb=0, ub=RB_wind_forecast[t-start], name="HA Wind schedule {}".format(t)) for t in setT}
    P_S_RB_t   = {t: RBOpt_mdl.continuous_var(lb=0, ub=RB_solar_forecast[t-start], name="HA Solar schedule {}".format(t)) for t in setT}
    P_dis_RB_t = RBOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA discharge') 
    P_cha_RB_t = RBOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA charge') 
    P_b_RB_t   = RBOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RB_t   = RBOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = RBOpt_mdl.binary_var_dict(setT, name='Cha or Discha')
    #an_var     = RBOpt_mdl.continuous_var(lb=0, ub=0.5, name='anciliary var')
    #v_t        = RBOpt_mdl.binary_var_dict(setT, name='Ban up or ban dw')
    delta_P_HPP_s = RBOpt_mdl.continuous_var_dict(setS, lb=-P_grid_limit, ub=P_grid_limit, name='HA imbalance')
    delta_P_HPP_UP_s = RBOpt_mdl.continuous_var_dict(setS, lb=0, name='HA up imbalance')
    delta_P_HPP_DW_s = RBOpt_mdl.continuous_var_dict(setS, lb=0, name='HA dw imbalance')
    # delta_E_HPP_DW_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min dw imbalance')
    # delta_E_HPP_UP_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min up imbalance')
    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  #dataframe to series
  # Define constraints
    for t in setT:
        RBOpt_mdl.add_constraint(P_HPP_RB_t[t] == P_W_RB_t[t] + P_S_RB_t[t] + P_b_RB_t[t])
        RBOpt_mdl.add_constraint(P_b_RB_t[t]   == P_dis_RB_t[t] - P_cha_RB_t[t])
        RBOpt_mdl.add_constraint(P_dis_RB_t[t] <= (PbMax - PreUp) * z_t[t] )
        RBOpt_mdl.add_constraint(P_cha_RB_t[t] <= (PbMax - PreDw) * (1-z_t[t]))
        RBOpt_mdl.add_constraint(SoC_RB_t[t + 1] == SoC_RB_t[t] * (1-eta_leak_ha) - 1/Emax * P_dis_RB_t[t]/eta_dis_ha * dt + 1/Emax * P_cha_RB_t[t] * eta_cha_ha * dt)
        RBOpt_mdl.add_constraint(SoC_RB_t[t]   <= SoCmax + 1/Emax * (- PreDw * eta_cha_ha) * dt )
        RBOpt_mdl.add_constraint(SoC_RB_t[t]   >= SoCmin + 1/Emax * (PreUp/eta_dis_ha) * dt ) 
        RBOpt_mdl.add_constraint(P_HPP_RB_t[t] <= P_grid_limit - PreUp)
        RBOpt_mdl.add_constraint(P_HPP_RB_t[t] >= -P_grid_limit + PreDw)
    
    for s in setS:
        RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s])
        if s < (current_hour + 1)*ds_num:            
            if s == current_SI:
           #RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == (exist_imbalance + RBOpt_mdl.sum((P_HPP_RB_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j]) * dt for j in range(start%dsdt_num, dsdt_num)))/ds)
                RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == (exist_imbalance + RBOpt_mdl.sum((P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j]) * dt for j in range(start%dsdt_num, dsdt_num)))/ds)
            else:
           #RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == RBOpt_mdl.sum(P_HPP_RB_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
                RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == RBOpt_mdl.sum(P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
        elif s >= (current_hour + 1)*ds_num and s < (current_hour + 2)*ds_num:
            RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == RBOpt_mdl.sum(P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
        else:
            RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == RBOpt_mdl.sum(P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
            


 #   {t : BMOpt_mdl.add_constraint(ct=delta_P_HPP_t[t] == P_HPP_BM_t[t] - P_HPP_DA_ts[t], ctname="constraint_{0}".format(t)) for t in setT } 

    RBOpt_mdl.add_constraint(SoC_RB_t[start] == SoC0)
#    RBOpt_mdl.add_constraint(SoC_RB_t[T] <= 0.6)
#    RBOpt_mdl.add_constraint(SoC_RB_t[T] >= 0.4)
    

    #Revenue = RBOpt_mdl.sum(BM_dw_price_forecast[k] * delta_P_HPP_UP_k[k] *dk - BM_up_price_forecast[k] * delta_P_HPP_DW_k[k] *dk for k in setK)
    Revenue = RBOpt_mdl.sum((BM_dw_price_forecast_settle[s]-0.001) * delta_P_HPP_UP_s[s] *ds - (BM_up_price_forecast_settle[s]+0.001) * delta_P_HPP_DW_s[s] *ds for s in setS)
    #Revenue = RBOpt_mdl.sum(BM_dw_price_forecast[k] * delta_P_HPP_UP_k[k] *dk - BM_up_price_forecast[k] * delta_P_HPP_DW_k[k] *dk for k in setK)
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * RBOpt_mdl.sum((P_dis_RB_t[t] + P_cha_RB_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0 
    #RBOpt_mdl.maximize(Revenue-Deg_cost - 1e7*an_var)
    RBOpt_mdl.maximize(Revenue-Deg_cost)

  # Solve BMOpt Model
    RBOpt_mdl.print_information()
    sol = RBOpt_mdl.solve()
    aa = RBOpt_mdl.get_solve_details()
    print(aa.status)

    if sol:
    #    SMOpt_mdl.print_solution()
        P_HPP_RB_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_RB_t), orient='index')
        P_HPP_RB_t_opt.columns = ['RB']
        P_W_RB_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_W_RB_t), orient='index')
        P_S_RB_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_S_RB_t), orient='index')
        P_dis_RB_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_dis_RB_t), orient='index')
        P_cha_RB_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_cha_RB_t), orient='index')
        SoC_RB_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_RB_t), orient='index')
        delta_P_HPP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_s), orient='index')
        delta_P_HPP_UP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_UP_s), orient='index')
        delta_P_HPP_DW_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_DW_s), orient='index')

        #print(SoC_RB_t_opt.iloc[12:15])

        E_HPP_RB_t_opt = P_HPP_RB_t_opt * dt

        P_W_RB_cur_t_opt = np.array(RB_wind_forecast[0:].T) - np.array(P_W_RB_t_opt).flatten()
        P_W_RB_cur_t_opt = pd.DataFrame(P_W_RB_cur_t_opt)
        P_S_RB_cur_t_opt = np.array(RB_solar_forecast[0:].T) - np.array(P_S_RB_t_opt).flatten()
        P_S_RB_cur_t_opt = pd.DataFrame(P_S_RB_cur_t_opt)


        z_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient='index')


    else:
        print("RBOpt has no solution")
        #print(SMOpt_mdl.expoRB_to_string())
    return E_HPP_RB_t_opt, P_HPP_RB_t_opt, P_dis_RB_t_opt, P_cha_RB_t_opt, SoC_RB_t_opt, P_W_RB_cur_t_opt, P_S_RB_cur_t_opt, P_W_RB_t_opt, P_S_RB_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt  

def revenue_process(results_path):
    results_file_names = os.listdir(results_path)
    
    output_accu_revenue = pd.DataFrame(list(), columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost'])
    
    for i in results_file_names:
        revenue = pd.read_csv(results_path + '/' + i +'/revenue.csv')
        accu_revenue = revenue.sum()
        accu_revenue =pd.DataFrame([accu_revenue])
       # accu_revenue.columns = ['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'BM_total', 'Deg_cost']
        
        output_accu_revenue = pd.concat([output_accu_revenue, accu_revenue], axis=0)
    
    return output_accu_revenue


    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    