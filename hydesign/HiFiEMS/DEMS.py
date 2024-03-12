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
#from scipy import interpolate

def ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict): 
    skips1 = range(1, ((day_num - 1) * T)%(359*T) + 1)
    skips2 = range(1, ((day_num - 1) * 24)%(359*24) + 1)

    Wind_data = pd.read_csv(simulation_dict["wind_dir"], skiprows = skips1, nrows=T+exten_num)
    Solar_data = pd.read_csv(simulation_dict["solar_dir"], skiprows = skips1, nrows=T+exten_num)
    Market_data = pd.read_csv(simulation_dict["market_dir"], skiprows = skips2, nrows=int(T/DI_num)+int(exten_num/DI_num))
    
    
    
    Wind_measurement = Wind_data['Measurement'] * PwMax
    Solar_measurement = Solar_data['Measurement'] * PsMax


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

    BM_dw_price_forecast = pd.DataFrame(columns=['Down'])
    BM_up_price_forecast = pd.DataFrame(columns=['Up'])
    reg_up_sign_forecast = pd.DataFrame(columns=['up_sign'])
    reg_dw_sign_forecast = pd.DataFrame(columns=['dw_sign'])
    
    for i in range(int(T/DI_num)+int(exten_num/DI_num)):
        if Reg_price_forecast.iloc[i] > SM_price_cleared.iloc[i]:
            BM_up_price_forecast = pd.concat([BM_up_price_forecast, pd.DataFrame([Reg_price_forecast.iloc[i]], columns=['Up'])], ignore_index=True)
            BM_dw_price_forecast = pd.concat([BM_dw_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Down'])], ignore_index=True)
            
            reg_up_sign_forecast = pd.concat([reg_up_sign_forecast, pd.DataFrame([1], columns=['up_sign'])], ignore_index=True)
            reg_dw_sign_forecast = pd.concat([reg_dw_sign_forecast, pd.DataFrame([0], columns=['dw_sign'])], ignore_index=True)
        elif Reg_price_forecast.iloc[i] < SM_price_cleared.iloc[i]:
            BM_up_price_forecast = pd.concat([BM_up_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Up'])], ignore_index=True)
            BM_dw_price_forecast = pd.concat([BM_dw_price_forecast, pd.DataFrame([Reg_price_forecast.iloc[i]], columns=['Down'])], ignore_index=True)
            
            reg_up_sign_forecast = pd.concat([reg_up_sign_forecast, pd.DataFrame([0], columns=['up_sign'])], ignore_index=True)
            reg_dw_sign_forecast = pd.concat([reg_dw_sign_forecast, pd.DataFrame([1], columns=['dw_sign'])], ignore_index=True)
        else:
            BM_up_price_forecast = pd.concat([BM_up_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Up'])], ignore_index=True)
            BM_dw_price_forecast = pd.concat([BM_dw_price_forecast, pd.DataFrame([SM_price_cleared.iloc[i]], columns=['Down'])], ignore_index=True)   
            
            reg_up_sign_forecast = pd.concat([reg_up_sign_forecast, pd.DataFrame([0], columns=['up_sign'])], ignore_index=True)
            reg_dw_sign_forecast = pd.concat([reg_dw_sign_forecast, pd.DataFrame([0], columns=['dw_sign'])], ignore_index=True)
    
    BM_dw_price_forecast = BM_dw_price_forecast.squeeze()
    BM_up_price_forecast = BM_up_price_forecast.squeeze()
    reg_up_sign_forecast = reg_up_sign_forecast.squeeze()
    reg_dw_sign_forecast = reg_dw_sign_forecast.squeeze()        
    
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

def Input():  #x: dataframe reso: in hour
    c_exp = 1
    para = pd.DataFrame([c_exp,1], columns=['para'])
    para = para.squeeze()
    #para.append({'para': c_exp}, ignore_index=True)

    return para

def get_var_value_from_sol(x, sol):
    
    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient='index')
    
    return y
   
def Revenue_calculation(DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_k_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num):    
    P_HPP_SM = P_HPP_SM_k_opt.squeeze().to_numpy()
    SM_price = SM_price_cleared.to_numpy()
    
    BM_up_price_settle_cleared = BM_up_price_cleared.squeeze().repeat(SI_num)
    BM_up_price_settle_cleared.index = range(T_SI + int(exten_num/SIDI_num))
    BM_dw_price_settle_cleared = BM_dw_price_cleared.squeeze().repeat(SI_num)
    BM_dw_price_settle_cleared.index = range(T_SI + int(exten_num/SIDI_num))    
    
    
    
    
    #P_HPP_RT = P_HPP_RT_ts.to_numpy()
    BM_up_settle_price = BM_up_price_settle_cleared.to_numpy()
    BM_dw_settle_price = BM_dw_price_settle_cleared.to_numpy() 
    BM_up_bid_price = BM_up_settle_price[0:T_SI:SI_num]
    BM_dw_bid_price = BM_dw_settle_price[0:T_SI:SI_num]
    pos_imbalance = residual_imbalance[residual_imbalance>0]
    pos_imbalance = pos_imbalance.replace(np.nan, 0)
    pos_imbalance = pos_imbalance.to_numpy()
    pos_imbalance = pos_imbalance[:,0]
    neg_imbalance = residual_imbalance[residual_imbalance<0]
    neg_imbalance = neg_imbalance.replace(np.nan, 0)
    neg_imbalance = neg_imbalance.to_numpy()
    neg_imbalance = neg_imbalance[:,0]
    
    s_UP_t = s_UP_t[0:T:DI_num]
    s_DW_t = s_DW_t[0:T:DI_num]
    P_HPP_UP_bid_ts = P_HPP_UP_bid_ts.to_numpy()
    P_HPP_UP_bid_ts = P_HPP_UP_bid_ts[:,0]
    P_HPP_DW_bid_ts = P_HPP_DW_bid_ts.to_numpy()
    P_HPP_DW_bid_ts = P_HPP_DW_bid_ts[:,0]
    
    # spot market revenue
    SM_revenue = np.sum(P_HPP_SM*SM_price)
    # regulation market revenue
    reg_revenue = np.sum(s_UP_t*P_HPP_UP_bid_ts*BI*BM_up_bid_price) - np.sum(s_DW_t*P_HPP_DW_bid_ts*BI*BM_dw_bid_price) 
    # imbalance revenue
    im_revenue = np.sum(pos_imbalance*BM_dw_settle_price) + np.sum(neg_imbalance*BM_up_settle_price)
    
    
    # power imbalance expenses (only for DK1)  
    if SI == 1 or SI == 1/4:
      AU_up_price = SM_price + 13.45 # 100DKK = 13.45 €
      AU_dw_price = SM_price - 13.45
    
      BM_up_price_cleared_DK1 = BM_up_price_cleared.squeeze().repeat(4)
      BM_up_price_cleared_DK1.index = range(96 + int(exten_num/3))
      BM_dw_price_cleared_DK1 = BM_dw_price_cleared.squeeze().repeat(4)
      BM_dw_price_cleared_DK1.index = range(96 + int(exten_num/3)) 
    
      P_HPP_RT_ts_15min = f_xmin_to_ymin(P_HPP_RT_ts, DI, 1/4)    
      P_HPP_RT_refs_15min = f_xmin_to_ymin(P_HPP_RT_refs, DI, 1/4)
    
      SI_num_DK1 = 4
    
      im_power_cost_DK1 = 0
      for ii in range(len(P_HPP_RT_ts_15min)):
          notification = P_HPP_SM[int(ii/SI_num_DK1)] + s_UP_t[int(ii/SI_num_DK1)]*P_HPP_UP_bid_ts[int(ii/SI_num_DK1)] - s_DW_t[int(ii/SI_num_DK1)]*P_HPP_DW_bid_ts[int(ii/SI_num_DK1)]
          metered_result = P_HPP_RT_ts_15min.iloc[ii][0]
          power_schedule = P_HPP_RT_refs_15min.iloc[ii][0]
        
          power_imbalance = metered_result - power_schedule
          if power_imbalance <=10 and power_imbalance >=-10:
             power_imbalance = 0
          elif power_imbalance > 10:
             power_imbalance = power_imbalance - 10
          else:
             power_imbalance = power_imbalance + 10
          if metered_result > power_schedule and power_schedule > notification:
             
              punish_price = BM_dw_price_cleared_DK1.iloc[ii] - AU_dw_price[int(ii/SI_num_DK1)]  
             
              if punish_price < 0:
                  punish_price = 0                                                                          
             
          elif power_schedule > metered_result and metered_result > notification:
             
              punish_price = BM_dw_price_cleared_DK1.iloc[ii] - AU_up_price[int(ii/SI_num_DK1)]    
                                                                                           
          elif metered_result > notification and notification > power_schedule:
             
              punish_price = BM_up_price_cleared_DK1.iloc[ii] - BM_dw_price_cleared_DK1.iloc[ii]
        
          elif notification > power_schedule and power_schedule > metered_result:
             
              punish_price = BM_up_price_cleared_DK1.iloc[ii] - AU_up_price[int(ii/SI_num_DK1)] 
             
              if punish_price > 0:
                  punish_price = 0   
          elif notification > metered_result and metered_result > power_schedule:
            
              punish_price = BM_up_price_cleared_DK1.iloc[ii] - AU_dw_price[int(ii/SI_num_DK1)] 
            
          elif power_schedule > notification and notification > metered_result:
            
              punish_price = BM_dw_price_cleared_DK1.iloc[ii] - BM_up_price_cleared_DK1.iloc[ii]
          elif power_schedule == notification:
              if metered_result > power_schedule:
                  punish_price = max(np.array([BM_dw_price_cleared_DK1.iloc[ii] - AU_dw_price[int(ii/SI_num_DK1)], BM_up_price_cleared_DK1.iloc[ii] - BM_dw_price_cleared_DK1.iloc[ii]]))
              elif metered_result < power_schedule:
                  punish_price = min(np.array([BM_dw_price_cleared_DK1.iloc[ii] - BM_up_price_cleared_DK1.iloc[ii], BM_up_price_cleared_DK1.iloc[ii] - AU_up_price[int(ii/SI_num_DK1)]]))
              else:
                  punish_price = 0                  
          else:
              punish_price = 0
                  
          im_power_cost_DK1 = im_power_cost_DK1 + power_imbalance * 1/4 * punish_price
    else:
        im_power_cost_DK1 = 0
        
    BM_revenue = reg_revenue + im_revenue - im_power_cost_DK1 
    return SM_revenue, reg_revenue, im_revenue, BM_revenue, im_power_cost_DK1





def SMOpt(dt, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                    DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator):
    
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
                
    #SMOpt_mdl.add_constraint(an_var >= SoC_SM_t[T] - 0.5) 
    #SMOpt_mdl.add_constraint(an_var >= 0.5 - SoC_SM_t[T])        
    SMOpt_mdl.add_constraint(SoC_SM_t[0] == SoC0)
    #SMOpt_mdl.add_constraint(SoC_SM_t[T] == 0.5)
#    SMOpt_mdl.add_constraint(SoC_SM_t[T] >= 0.4)


  # Define objective function
    Revenue = SMOpt_mdl.sum(SM_price_forecast[t] * P_HPP_SM_t[t] *dt for t in setT) 
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * SMOpt_mdl.sum((P_dis_SM_t[t]+P_cha_SM_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0
    #SMOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    SMOpt_mdl.maximize(Revenue - Deg_cost)

  # Solve SMOpt Model
    SMOpt_mdl.print_information()
    sol = SMOpt_mdl.solve()

    if sol:
    #    SMOpt_mdl.print_solution()
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
        
        print(P_HPP_SM_t_opt)
        print(P_dis_SM_t_opt)
        print(P_cha_SM_t_opt)
        print(SoC_SM_t_opt)
        print(z_t_opt)
    #  write to excel
    #    DA_schedule = np.array(P_HPP_DA_ts.T, P_W_DA_ts.T, P_S_DA_ts.T, P_dis_DA_ts.T, P_cha_DA_ts.T, E_HPP_DA_ts.T)  
       
        #writer = pd.ExcelWriter('DA_schedule.xlsx')
        #P_HPP_SM_t_opt.to_csv('Scheule.csv', mode='a')
        #writer.save()
        #writer.close()

       # DateTime = pd.date_range('20200101','20201231',freq='BM')

    else:
        print("DA EMS Model has no solution")
        #print(SMOpt_mdl.export_to_string())
    return E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt



def BMOpt(dt, ds, dk, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    HA_wind_forecast, HA_solar_forecast, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, start, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator):
    
    # Optimization modelling by CPLEX
    print('2')
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
    print('3')
    setT = [i for i in range(start*dt_num, T + exten_num)] 
    setT1 = [i for i in range((start + 1) * dt_num, T + exten_num)] 
    setK = [i for i in range(start*dk_num, T_dk + int(exten_num/dt_num))]
    setK1 = [i for i in range((start + 1) * dk_num, T_dk + int(exten_num/dt_num))]
    setS = [i for i in range(start*ds_num, T_ds + int(exten_num/dsdt_num))]
    set_SoCT = [i for i in range(start*dt_num, T + 1 + exten_num)] 
    print('construct BMOpt model')    
    BMOpt_mdl = Model()
    print('BMOpt model is constructed')
  # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_UP_t = BMOpt_mdl.continuous_var_dict(setT1, lb=0, ub=P_grid_limit, name='BM UP bidding 5min')
    P_HPP_DW_t = BMOpt_mdl.continuous_var_dict(setT1, lb=0, ub=P_grid_limit, name='BM DW bidding 5min')
    P_HPP_UP_k = BMOpt_mdl.continuous_var_dict(setK1, lb=0, ub=P_grid_limit, name='BM UP bidding H')
    P_HPP_DW_k = BMOpt_mdl.continuous_var_dict(setK1, lb=0, ub=P_grid_limit, name='BM DW bidding H')
    #P_b_UP_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='BM UP bidding')
    #P_b_DW_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='BM DW bidding')    
    #P_HPP_all_t = BMOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_HPP_HA_t = BMOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_W_HA_t   = {t: BMOpt_mdl.continuous_var(lb=0, ub=HA_wind_forecast[t-start*dt_num], name="HA Wind schedule {}".format(t)) for t in setT}
    P_S_HA_t   = {t: BMOpt_mdl.continuous_var(lb=0, ub=HA_solar_forecast[t-start*dt_num], name="HA Solar schedule {}".format(t)) for t in setT}
    P_dis_HA_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA discharge') 
    P_cha_HA_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA charge') 
    P_b_HA_t   = BMOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
#    SoC_HA_t1   = BMOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
#    SoC_HA_t2   = BMOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    SoC_HA_t   = BMOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = BMOpt_mdl.binary_var_dict(setT, name='Cha or Discha')
    #an_var        = BMOpt_mdl.continuous_var(lb=0, ub=0.5, name='anciliary var')
    #v_t        = BMOpt_mdl.binary_var_dict(setT, name='Ban up or ban dw')
    delta_P_HPP_s = BMOpt_mdl.continuous_var_dict(setS, lb=-P_grid_limit, ub=P_grid_limit, name='HA imbalance')
    delta_P_HPP_UP_s = BMOpt_mdl.continuous_var_dict(setS, lb=0, name='HA up imbalance')
    delta_P_HPP_DW_s = BMOpt_mdl.continuous_var_dict(setS, lb=0, name='HA dw imbalance')
    # delta_E_HPP_DW_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min dw imbalance')
    # delta_E_HPP_UP_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min up imbalance')
 #   z_t        = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=0.4, name='Cha or Discha')
    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  #dataframe to series
  # Define constraints
    for t in setT:
        #if t - start*dt_num < dt_num:
        #    BMOpt_mdl.add_constraint(P_HPP_UP_t[t] == P_HPP_UP_t0 * s_UP_t[t])
        #    BMOpt_mdl.add_constraint(P_HPP_DW_t[t] == P_HPP_DW_t0 * s_DW_t[t])
        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] == P_W_HA_t[t] + P_S_HA_t[t] + P_b_HA_t[t])
        BMOpt_mdl.add_constraint(P_b_HA_t[t]   == P_dis_HA_t[t] - P_cha_HA_t[t])
        BMOpt_mdl.add_constraint(P_dis_HA_t[t] <= (PbMax - PreUp ) * z_t[t] )
        BMOpt_mdl.add_constraint(P_cha_HA_t[t] <= (PbMax - PreDw) * (1-z_t[t]))
        #BMOpt_mdl.add_constraint(P_HPP_UP_t[t] == P_b_UP_t[t] + HA_wind_forecast[t-start*dt_num] + HA_solar_forecast[t-start*dt_num] - (DA_wind_forecast[t] + DA_solar_forecast[t]))
        #BMOpt_mdl.add_constraint(P_HPP_DW_t[t] == P_b_DW_t[t] + HA_wind_forecast[t-start*dt_num] + HA_solar_forecast[t-start*dt_num] - (DA_wind_forecast[t] + DA_solar_forecast[t]))
        #BMOpt_mdl.add_constraint(P_b_HA_t[t] + (1-s_UP_t[t]) * P_HPP_UP_t[t] <= (PbMax - PreUp))
        #BMOpt_mdl.add_constraint(-P_b_HA_t[t] + (1-s_DW_t[t]) * P_HPP_DW_t[t] <= (PbMax - PreDw))
#        if t >= (start + 1)*dt_num:
#            BMOpt_mdl.add_constraint(P_b_HA_t[t] - P_HPP_DW_t[t] >= -(PbMax - PreDw))
#            BMOpt_mdl.add_constraint(P_b_HA_t[t] + P_HPP_UP_t[t] <= (PbMax - PreUp))        
#            BMOpt_mdl.add_constraint(P_b_HA_t[t] + P_HPP_DW_t[t] <= (PbMax - PreDw))
#            BMOpt_mdl.add_constraint(P_b_HA_t[t] - P_HPP_UP_t[t] >= -(PbMax - PreUp))     
#            if reg_up_sign_forecast1[t] == 1:
#                BMOpt_mdl.add_constraint(P_HPP_DW_t[t] == PbMax + P_b_HA_t[t])
#            if reg_dw_sign_forecast1[t] == 1:
#                BMOpt_mdl.add_constraint(P_HPP_UP_t[t] == PbMax - P_b_HA_t[t])
#            if reg_dw_sign_forecast1[t] == 0 and reg_up_sign_forecast1[t] == 0:
#                BMOpt_mdl.add_constraint(P_HPP_UP_t[t] == PbMax - P_b_HA_t[t])
#                BMOpt_mdl.add_constraint(P_HPP_DW_t[t] == PbMax + P_b_HA_t[t])
        BMOpt_mdl.add_constraint(SoC_HA_t[t + 1] == SoC_HA_t[t] * (1-eta_leak_ha) - 1/Emax * (P_dis_HA_t[t])/eta_dis_ha * dt + 1/Emax * (P_cha_HA_t[t]) * eta_cha_ha * dt)
        #BMOpt_mdl.add_constraint(SoC_HA_t1[t + 1] == SoC_HA_t1[t] * (1-eta_leak) - 1/Emax * (P_dis_HA_t[t] + P_HPP_UP_t[t]) / eta_dis * dt)
        #BMOpt_mdl.add_constraint(SoC_HA_t2[t + 1] == SoC_HA_t2[t] * (1-eta_leak) + 1/Emax * (P_cha_HA_t[t] + P_HPP_DW_t[t]) * eta_cha * dt)
        #BMOpt_mdl.add_constraint(SoC_HA_t[t] <= SoCmax + 1/Emax * (P_b_HA_t[t] -PreDw) * dt - 1/Emax * (BMOpt_mdl.sum((1-s_DW_t[t//dt_num * dt_num + ii]) * P_HPP_DW_t[t//dt_num * dt_num + ii] *dt for ii in range(t%dt_num, dt_num))))
        #BMOpt_mdl.add_constraint(SoC_HA_t[t] >= SoCmin + 1/Emax * (P_b_HA_t[t] + PreUp/eta_dis_ha) * dt + 1/Emax * (BMOpt_mdl.sum((1-s_UP_t[t//dt_num * dt_num + ii]) * P_HPP_UP_t[t//dt_num * dt_num + ii]/eta_dis *dt for ii in range(t%dt_num, dt_num))))
        BMOpt_mdl.add_constraint(SoC_HA_t[t] <= SoCmax + 1/Emax * (-PreDw*eta_cha_ha) * dt)
        BMOpt_mdl.add_constraint(SoC_HA_t[t] >= SoCmin + 1/Emax * (PreUp/eta_dis_ha) * dt)
        #BMOpt_mdl.add_constraint(P_HPP_HA_t[t] <= P_grid_limit - PreUp - (1-s_UP_t[t]) * P_HPP_UP_t[t])
        #BMOpt_mdl.add_constraint(P_HPP_HA_t[t] >= -P_grid_limit + PreDw + (1-s_DW_t[t]) * P_HPP_DW_t[t])  
        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] <= P_grid_limit - PreUp )
        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] >= -P_grid_limit + PreDw) 
        #BMOpt_mdl.add_constraint(an_var >= SoC_HA_t[T] - 0.5) 
        #BMOpt_mdl.add_constraint(an_var >= 0.5 - SoC_HA_t[T]) 
        #BMOpt_mdl.add_constraint(P_HPP_HA_t[t] == P_HPP_SM_t_opt[t] + P_HPP_UP_t[t] - P_HPP_DW_t[t])
        #BMOpt_mdl.add_constraint(P_dis_HA_t[start*dt_num] == P_dis_HA_t0)
        #BMOpt_mdl.add_constraint(P_cha_HA_t[start*dt_num] == P_cha_HA_t0) 
    for s in setS:
        if s < (start + 1) * ds_num:        
        #BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (s_UP_t[s * dsdt_num + m] * P_HPP_UP_t[s * dsdt_num + m] - s_DW_t[s * dsdt_num + m] * P_HPP_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
            BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (P_HPP_UP_t0 * s_UP_t[s * dsdt_num + m] - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
        else:
            BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (reg_up_sign_forecast1[s * dsdt_num + m] * P_HPP_UP_t[s * dsdt_num + m] - reg_dw_sign_forecast1[s * dsdt_num + m] * P_HPP_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
        BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s])
       
    for k in setK1:
        for j in range(0, dt_num):
#            if reg_sign_forecast[k//dk_num] == 1:    #up regu
#                BMOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
#                BMOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == 0)
#            elif reg_sign_forecast[k//dk_num] == -1: #dw regu
#                BMOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == 0)
#                BMOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])
#            elif reg_sign_forecast[k//dk_num] == 2: #both up and dw regu
                BMOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
                BMOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])
 #   {t : BMOpt_mdl.add_constraint(ct=delta_P_HPP_t[t] == P_HPP_BM_t[t] - P_HPP_DA_ts[t], ctname="constraint_{0}".format(t)) for t in setT } 

    BMOpt_mdl.add_constraint(SoC_HA_t[start*dt_num] == SoC0)
#    BMOpt_mdl.add_constraint(SoC_HA_t1[start*dt_num] == SoC0)
#    BMOpt_mdl.add_constraint(SoC_HA_t2[start*dt_num] == SoC0)
#    BMOpt_mdl.add_constraint(SoC_HA_t[T] <= 0.6)
#    BMOpt_mdl.add_constraint(SoC_HA_t[T] >= 0.200001)

    Revenue = BMOpt_mdl.sum(BM_up_price_forecast[k] * reg_up_sign_forecast[k] * P_HPP_UP_k[k] *dk - BM_dw_price_forecast[k] * reg_dw_sign_forecast[k] * P_HPP_DW_k[k] *dk for k in setK1) + BMOpt_mdl.sum((BM_dw_price_forecast_settle[s]-0.001) * delta_P_HPP_UP_s[s] *ds - (BM_up_price_forecast_settle[s]+0.001) * delta_P_HPP_DW_s[s] *ds for s in setS)
    #Revenue = BMOpt_mdl.sum(BM_up_price_forecast[k] * P_HPP_UP_k[k] *dk - BM_dw_price_forecast[k] * P_HPP_DW_k[k] *dk for k in setK)
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * BMOpt_mdl.sum((P_dis_HA_t[t] + P_cha_HA_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0 
    #BMOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    BMOpt_mdl.maximize(Revenue - Deg_cost)

  # Solve BMOpt Model
    BMOpt_mdl.print_information()
    print('BMOpt is running')
    sol = BMOpt_mdl.solve()
    aa = BMOpt_mdl.get_solve_details()
    print(aa.status)
    if sol:
    #    SMOpt_mdl.print_solution()
        P_HPP_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_HA_t), orient='index')
        P_HPP_HA_t_opt.columns = ['HA']
        P_W_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_W_HA_t), orient='index')
        P_S_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_S_HA_t), orient='index')
        P_dis_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_dis_HA_t), orient='index')
        P_cha_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_cha_HA_t), orient='index')
 #       SoC_HA_t_opt1 = pd.DataFrame.from_dict(sol.get_value_dict(SoC_HA_t1), orient='index')
 #       SoC_HA_t_opt2 = pd.DataFrame.from_dict(sol.get_value_dict(SoC_HA_t2), orient='index')
        SoC_HA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_HA_t), orient='index')
        P_HPP_UP_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_t), orient='index')
        P_HPP_DW_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_t), orient='index')
        P_HPP_UP_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_k), orient='index')
        P_HPP_DW_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_k), orient='index')
        delta_P_HPP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_s), orient='index')
        delta_P_HPP_UP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_UP_s), orient='index')
        delta_P_HPP_DW_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_DW_s), orient='index')

#        for t in setT:
#            if t >= dt_num * (start + 1):
#               if reg_up_sign_forecast1[t] == 1:
#                   P_HPP_DW_t_opt.loc[t] = PbMax + P_dis_HA_t_opt.loc[t] - P_cha_HA_t_opt.loc[t] - P_HPP_UP_t_opt.loc[t]
#               if reg_dw_sign_forecast1[t] == 1:
#                   P_HPP_UP_t_opt.loc[t] = PbMax - P_dis_HA_t_opt.loc[t] + P_cha_HA_t_opt.loc[t] - P_HPP_DW_t_opt.loc[t]
#               if reg_dw_sign_forecast1[t] == 0 and reg_up_sign_forecast1[t] == 0:
#                   P_HPP_DW_t_opt.loc[t] = PbMax + P_dis_HA_t_opt.loc[t] - P_cha_HA_t_opt.loc[t] - P_HPP_UP_t_opt.loc[t]
#                   P_HPP_UP_t_opt.loc[t] = PbMax - P_dis_HA_t_opt.loc[t] + P_cha_HA_t_opt.loc[t] - P_HPP_DW_t_opt.loc[t]
        #print(SoC_HA_t_opt.iloc[12:15])

        E_HPP_HA_t_opt = P_HPP_HA_t_opt * dt

        P_W_HA_cur_t_opt = np.array(HA_wind_forecast[0:].T) - np.array(P_W_HA_t_opt).flatten()
        P_W_HA_cur_t_opt = pd.DataFrame(P_W_HA_cur_t_opt)
        P_S_HA_cur_t_opt = np.array(HA_solar_forecast[0:].T) - np.array(P_S_HA_t_opt).flatten()
        P_S_HA_cur_t_opt = pd.DataFrame(P_S_HA_cur_t_opt)


        z_ts = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient='index')

    #  write to excel
    #    DA_schedule = np.array(P_HPP_DA_ts.T, P_W_DA_ts.T, P_S_DA_ts.T, P_dis_DA_ts.T, P_cha_DA_ts.T, E_HPP_DA_ts.T)  
       
       # writer = pd.ExcelWriter('HA_schedule.xlsx')
       # E_HPP_DA_ts.to_excel(writer, 'HA_schedule')
       # writer.save()
      # writer.close()
        #P_HPP_HA_t_opt.to_csv(r'Scheule.csv', mode='a', index=False)
       # DateTime = pd.date_range('20200101','20201231',freq='BM')

    else:
        print("BMOpt has no solution")
        #print(SMOpt_mdl.export_to_string())
    return E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt  

'''
def test():
    mdl = Model()
    x  = mdl.continuous_var(lb=0, ub=10, name='HA Wind schedule')
    mdl.add_constraint(x>=10)
    mdl.maximize(x)
    sol = mdl.solve()
    aa = mdl.get_solve_details()
    print(aa.status)
'''    
    
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
    P_HPP_UP_t = RDOpt_mdl.continuous_var_dict(setT1, lb=0, name='BM UP bidding')
    P_HPP_DW_t = RDOpt_mdl.continuous_var_dict(setT1, lb=0, name='BM DW bidding')
    P_HPP_UP_k = RDOpt_mdl.continuous_var_dict(setK1, lb=0, name='BM UP bidding')
    P_HPP_DW_k = RDOpt_mdl.continuous_var_dict(setK1, lb=0, name='BM DW bidding')
    #P_HPP_all_t = BMOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_HPP_RD_t = RDOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_W_RD_t   = {t: RDOpt_mdl.continuous_var(lb=0, ub=RD_wind_forecast[t-start], name="HA Wind schedule {}".format(t)) for t in setT}
    P_S_RD_t   = {t: RDOpt_mdl.continuous_var(lb=0, ub=RD_solar_forecast[t-start], name="HA Solar schedule {}".format(t)) for t in setT}
    P_dis_RD_t = RDOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA discharge') 
    P_cha_RD_t = RDOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name='HA charge') 
    P_b_RD_t   = RDOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RD_t   = RDOpt_mdl.continuous_var_dict(set_SoCT, lb=SoCmin, ub=SoCmax, name='HA SoC')
    z_t        = RDOpt_mdl.binary_var_dict(setT, name='Cha or Discha')
    #an_var     = RDOpt_mdl.continuous_var(lb=0, ub=0.5, name='anciliary var')
    #v_t        = RDOpt_mdl.binary_var_dict(setT, name='Ban up or ban dw')
    delta_P_HPP_s = RDOpt_mdl.continuous_var_dict(setS, lb=-P_grid_limit, ub=P_grid_limit, name='HA imbalance')
    delta_P_HPP_UP_s = RDOpt_mdl.continuous_var_dict(setS, lb=0, name='HA up imbalance')
    delta_P_HPP_DW_s = RDOpt_mdl.continuous_var_dict(setS, lb=0, name='HA dw imbalance')
    # delta_E_HPP_DW_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min dw imbalance')
    # delta_E_HPP_UP_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min up imbalance')
    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  #dataframe to series
  # Define constraints
    for t in setT:
        #if t < dt_num * (current_hour + 1):
        #    RDOpt_mdl.add_constraint(P_HPP_UP_t[t] == P_HPP_UP_t0 * s_UP_t[t])
        #    RDOpt_mdl.add_constraint(P_HPP_DW_t[t] == P_HPP_DW_t0 * s_DW_t[t])
        #elif t >= dt_num * (current_hour + 1) and t < dt_num * (current_hour + 2):
        #    RDOpt_mdl.add_constraint(P_HPP_UP_t[t] == P_HPP_UP_t1)
        #    RDOpt_mdl.add_constraint(P_HPP_DW_t[t] == P_HPP_DW_t1)
#            RDOpt_mdl.add_constraint(P_HPP_UP_t[t] == 0)
#            RDOpt_mdl.add_constraint(P_HPP_DW_t[t] == 0)
#        else:
#            RDOpt_mdl.add_constraint(P_HPP_UP_t[t] == 0)
#            RDOpt_mdl.add_constraint(P_HPP_DW_t[t] == 0)
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] == P_W_RD_t[t] + P_S_RD_t[t] + P_b_RD_t[t])
        RDOpt_mdl.add_constraint(P_b_RD_t[t]   == P_dis_RD_t[t] - P_cha_RD_t[t])
        RDOpt_mdl.add_constraint(P_dis_RD_t[t] <= (PbMax - PreUp) * z_t[t] )
        RDOpt_mdl.add_constraint(P_cha_RD_t[t] <= (PbMax - PreDw) * (1-z_t[t]))
        #RDOpt_mdl.add_constraint(P_b_RD_t[t] + (1-s_UP_t[t]) * P_HPP_UP_t[t] <= (PbMax - PreUp))
        #RDOpt_mdl.add_constraint(-P_b_RD_t[t] + (1-s_DW_t[t]) * P_HPP_DW_t[t] <= (PbMax - PreDw))
#        if t >= dt_num * (current_hour + 2):
#            RDOpt_mdl.add_constraint(P_b_RD_t[t] + P_HPP_DW_t[t] <= (PbMax - PreDw))
#            RDOpt_mdl.add_constraint(P_b_RD_t[t] - P_HPP_UP_t[t] >= -(PbMax - PreUp))
            
#            if reg_up_sign_forecast1[t] == 1:
#                RDOpt_mdl.add_constraint(P_HPP_DW_t[t] == PbMax + P_b_RD_t[t])
#            if reg_dw_sign_forecast1[t] == 1:
#                RDOpt_mdl.add_constraint(P_HPP_UP_t[t] == PbMax - P_b_RD_t[t])
#            if reg_dw_sign_forecast1[t] == 0 and reg_up_sign_forecast1[t] == 0:
#                RDOpt_mdl.add_constraint(P_HPP_UP_t[t] == PbMax - P_b_RD_t[t])
#                RDOpt_mdl.add_constraint(P_HPP_DW_t[t] == PbMax + P_b_RD_t[t])
        #RDOpt_mdl.add_constraint(P_HPP_UP_t[t] <= 1e5 * v_t[t])
        #RDOpt_mdl.add_constraint(P_HPP_DW_t[t] <= 1e5 * (1-v_t[t]))
        #BMOpt_mdl.add_constraint(SoC_HA_t[t + 1] == SoC_HA_t[t] * (1-eta_leak) - 1/Emax * (P_dis_HA_t[t] + P_HPP_UP_t[t])/eta_dis_ha * dt + 1/Emax * P_cha_HA_t[t] * eta_cha_ha * dt)
        RDOpt_mdl.add_constraint(SoC_RD_t[t + 1] == SoC_RD_t[t] * (1-eta_leak_ha) - 1/Emax * P_dis_RD_t[t]/eta_dis_ha * dt + 1/Emax * P_cha_RD_t[t] * eta_cha_ha * dt)
        #RDOpt_mdl.add_constraint(SoC_RD_t[t]   <= SoCmax + 1/Emax * (P_b_RD_t[t] -PreDw) * dt - 1/Emax * (RDOpt_mdl.sum((1-s_DW_t[t//dt_num * dt_num + ii]) * P_HPP_DW_t[t//dt_num * dt_num + ii] *dt for ii in range(t%dt_num, dt_num))))
        #RDOpt_mdl.add_constraint(SoC_RD_t[t]   >= SoCmin + 1/Emax * (P_b_RD_t[t] + PreUp/eta_dis_ha) * dt + 1/Emax * (RDOpt_mdl.sum((1-s_UP_t[t//dt_num * dt_num + ii]) * P_HPP_UP_t[t//dt_num * dt_num + ii]/eta_dis *dt for ii in range(t%dt_num, dt_num)))) 
        RDOpt_mdl.add_constraint(SoC_RD_t[t]   <= SoCmax + 1/Emax * (- PreDw * eta_cha_ha) * dt )
        RDOpt_mdl.add_constraint(SoC_RD_t[t]   >= SoCmin + 1/Emax * (PreUp/eta_dis_ha) * dt ) 
        #RDOpt_mdl.add_constraint(P_HPP_RD_t[t] <= P_grid_limit - PreUp - (1-s_UP_t[t]) * P_HPP_UP_t[t])
        #RDOpt_mdl.add_constraint(P_HPP_RD_t[t] >= -P_grid_limit + PreDw + (1-s_DW_t[t]) * P_HPP_DW_t[t])
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] <= P_grid_limit - PreUp)
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] >= -P_grid_limit + PreDw)
        #RDOpt_mdl.add_constraint(an_var >= SoC_RD_t[T] - 0.5) 
        #RDOpt_mdl.add_constraint(an_var >= 0.5 - SoC_RD_t[T]) 
        #RDOpt_mdl.add_constraint(P_dis_RD_t[start] == P_dis_HA_t0)
        #RDOpt_mdl.add_constraint(P_cha_RD_t[start] == P_cha_HA_t0)
    
    for s in setS:
        RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s])
        if s < (current_hour + 1)*ds_num:            
            if s == current_SI:
           #RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == (exist_imbalance + RDOpt_mdl.sum((P_HPP_RD_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j]) * dt for j in range(start%dsdt_num, dsdt_num)))/ds)
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

 #   {t : BMOpt_mdl.add_constraint(ct=delta_P_HPP_t[t] == P_HPP_BM_t[t] - P_HPP_DA_ts[t], ctname="constraint_{0}".format(t)) for t in setT } 

    RDOpt_mdl.add_constraint(SoC_RD_t[start] == SoC0)
#    RDOpt_mdl.add_constraint(SoC_RD_t[T] <= 0.6)
#    RDOpt_mdl.add_constraint(SoC_RD_t[T] >= 0.4)
    

    #Revenue = RDOpt_mdl.sum(BM_dw_price_forecast[k] * delta_P_HPP_UP_k[k] *dk - BM_up_price_forecast[k] * delta_P_HPP_DW_k[k] *dk for k in setK)
    Revenue = RDOpt_mdl.sum(BM_up_price_forecast[k] * reg_up_sign_forecast[k] * P_HPP_UP_k[k] *dk - BM_dw_price_forecast[k] * reg_dw_sign_forecast[k] * P_HPP_DW_k[k] *dk for k in setK1) + RDOpt_mdl.sum((BM_dw_price_forecast_settle[s]-0.001) * delta_P_HPP_UP_s[s] *ds - (BM_up_price_forecast_settle[s]+0.001) * delta_P_HPP_DW_s[s] *ds for s in setS)
    #Revenue = RDOpt_mdl.sum(BM_dw_price_forecast[k] * delta_P_HPP_UP_k[k] *dk - BM_up_price_forecast[k] * delta_P_HPP_DW_k[k] *dk for k in setK)
    if deg_indicator == 1:
       Deg_cost = mu * EBESS * ad * RDOpt_mdl.sum((P_dis_RD_t[t] + P_cha_RD_t[t]) * dt for t in setT)
    else:
       Deg_cost = 0 
    #RDOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    RDOpt_mdl.maximize(Revenue - Deg_cost)

  # Solve BMOpt Model
    RDOpt_mdl.print_information()
    print('RDOpt is running')
    sol = RDOpt_mdl.solve()
    aa = RDOpt_mdl.get_solve_details()    
    print(aa.status)

    if sol:
    #    SMOpt_mdl.print_solution()
        P_HPP_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_RD_t), orient='index')
        P_HPP_RD_t_opt.columns = ['RD']
        P_W_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_W_RD_t), orient='index')
        P_S_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_S_RD_t), orient='index')
        P_dis_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_dis_RD_t), orient='index')
        P_cha_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_cha_RD_t), orient='index')
        SoC_RD_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(SoC_RD_t), orient='index')
        P_HPP_UP_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_t), orient='index')
        P_HPP_DW_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_t), orient='index')
        P_HPP_UP_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_UP_k), orient='index')
        P_HPP_DW_k_opt = pd.DataFrame.from_dict(sol.get_value_dict(P_HPP_DW_k), orient='index')
        delta_P_HPP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_s), orient='index')
        delta_P_HPP_UP_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_UP_s), orient='index')
        delta_P_HPP_DW_s_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_P_HPP_DW_s), orient='index')

#        for t in setT:
#            if t >= dt_num * (current_hour + 2):
#               if reg_up_sign_forecast1[t] == 1:
#                   P_HPP_DW_t_opt.loc[t] = PbMax + P_dis_RD_t_opt.loc[t] - P_cha_RD_t_opt.loc[t] - P_HPP_UP_t_opt.loc[t]
#               if reg_dw_sign_forecast1[t] == 1:
#                   P_HPP_UP_t_opt.loc[t] = PbMax - P_dis_RD_t_opt.loc[t] + P_cha_RD_t_opt.loc[t] - P_HPP_DW_t_opt.loc[t]
#               if reg_dw_sign_forecast1[t] == 0 and reg_up_sign_forecast1[t] == 0:
#                   P_HPP_DW_t_opt.loc[t] = PbMax + P_dis_RD_t_opt.loc[t] - P_cha_RD_t_opt.loc[t] - P_HPP_UP_t_opt.loc[t]
#                   P_HPP_UP_t_opt.loc[t] = PbMax - P_dis_RD_t_opt.loc[t] + P_cha_RD_t_opt.loc[t] - P_HPP_DW_t_opt.loc[t]
        #print(SoC_RD_t_opt.iloc[12:15])

        E_HPP_RD_t_opt = P_HPP_RD_t_opt * dt

        P_W_RD_cur_t_opt = np.array(RD_wind_forecast[0:].T) - np.array(P_W_RD_t_opt).flatten()
        P_W_RD_cur_t_opt = pd.DataFrame(P_W_RD_cur_t_opt)
        P_S_RD_cur_t_opt = np.array(RD_solar_forecast[0:].T) - np.array(P_S_RD_t_opt).flatten()
        P_S_RD_cur_t_opt = pd.DataFrame(P_S_RD_cur_t_opt)


        z_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient='index')

    #  write to excel
    #    DA_schedule = np.array(P_HPP_DA_ts.T, P_W_DA_ts.T, P_S_DA_ts.T, P_dis_DA_ts.T, P_cha_DA_ts.T, E_HPP_DA_ts.T)  
       
       # writer = pd.ExcelWriter('HA_schedule.xlsx')
       # E_HPP_DA_ts.to_excel(writer, 'HA_schedule')
       # writer.save()
      # writer.close()
        #P_HPP_RD_t_opt.to_csv(r'Scheule.csv', mode='a', index=False)
       # DateTime = pd.date_range('20200101','20201231',freq='BM')

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



def write_results(data, filename, filetype, x_row, y_col, sheetname):
    if not x_row == 0:
        idn = False
        add = 1
    else:
        idn = True
        add = 0
    if filetype == 'xlsx':
       excel_book = openpyxl.load_workbook(filename)
       with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            writer.book = excel_book
            writer.sheets = {worksheet.title: worksheet for worksheet in excel_book.worksheets}
           #P_HPP_SM_t_opt.to_excel(writer, index=False, startrow=(day_num-1)*T+1)
            for iii in range(0,len(y_col)):
                data.iloc[:,iii].to_excel(writer, index=False, header = idn, sheet_name=sheetname, startrow=x_row+add, startcol=y_col[iii])
            writer.save()
            writer.close()
    elif filetype == 'csv':
        data.to_csv(filename, mode='a', index=False, header=False)  
        
        
        
        
def run(parameter_dict, simulation_dict):
    
    DI = 1/12
    DI_num = int(1/DI)    
    T = int(1/DI*24)
        
    
    SI = 1/4
    SI_num = int(1/SI)
    T_SI = int(24/SI)
    SIDI_num = int(SI/DI)
    
  
    BI = 1
    BI_num = int(1/BI)
    T_BI = int(24/BI)
    
    Wind_component = simulation_dict["wind_as_component"]
    Solar_component = simulation_dict["solar_as_component"]
    BESS_component = simulation_dict["battery_as_component"]
    
    PwMax = parameter_dict["wind_capacity"] * Wind_component
    PsMax = parameter_dict["solar_capacity"] * Solar_component
    EBESS = parameter_dict["battery_energy_capacity"]     
    PbMax = parameter_dict["battery_power_capacity"] * BESS_component  
    SoCmin = parameter_dict["battery_minimum_SoC"] * BESS_component  
    SoCmax = parameter_dict["battery_maximum_SoC"] * BESS_component
    SoCini = parameter_dict["battery_initial_SoC"] * BESS_component
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"] * BESS_component
    
    
    day_num = 1
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1.11E-06   # slope   
    capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    replace_percent = 0.2     

                             
    PreUp = PreDw = 0
    P_grid_limit = parameter_dict["hpp_grid_connection"]

    mu = parameter_dict["battery_marginal_degradation_cost"]
    
    deg_indicator = parameter_dict["degradation_in_optimization"]
    
    P_HPP_UP_t0 = 0
    P_HPP_DW_t0 = 0

    exten_num = 0
    
    out_dir = simulation_dict['out_dir']
        
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    
    re  = pd.DataFrame(list(), columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost'])
    sig = pd.DataFrame(list(), columns=['signal_up','signal_down'])
    cur = pd.DataFrame(list(), columns=['RES_cur'])
    de  = pd.DataFrame(list(), columns=['nld','ld'])
    ei  = pd.DataFrame(list(), columns=['energy_imbalance'])
    reg = pd.DataFrame(list(), columns=['bid_up','bid_dw'])
    shc = pd.DataFrame(list(), columns=['SM','RT','Ref','dis_RT','cha_RT'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])

    sig.to_csv(out_dir+'act_signal.csv',index=False)
    cur.to_csv(out_dir+'curtailment.csv',index=False)
    de.to_csv(out_dir+'Degradation.csv',index=False)
    ei.to_csv(out_dir+'energy_imbalance.csv',index=False)
    reg.to_csv(out_dir+'reg_bids.csv',index=False)
    re.to_csv(out_dir+'revenue.csv',index=False)
    shc.to_csv(out_dir+'schedule.csv',index=False)
    slo.to_csv(out_dir+'slope.csv',index=False)
    soc.to_csv(out_dir+'SoC.csv',index=False)    
    
    while day_num:
        Emax = EBESS*(1-pre_nld)
        
        DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
        

        
        
        SM_price_forecast = SM_price_forecast.squeeze().repeat(DI_num)
        SM_price_forecast.index = range(T + exten_num)
        
        
        P_HPP_RT_ref = RT_wind_forecast[0] + RT_solar_forecast[0]
        
        
    # Call EMS Model
        # Run SMOpt
        E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt = SMOpt(DI, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                    DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator)
        
        P_HPP_SM_t_opt.index = time_index[:T]
        
        #write_results(P_HPP_SM_t_opt, 'results_run.xlsx', (day_num-1)*T, 0, 'power schedule')

                      
        P_HPP_RT_ts = pd.DataFrame(columns=['RT'])
        P_HPP_RT_refs = pd.DataFrame(columns=['Ref'])
        RES_RT_cur_ts = pd.DataFrame(columns=['RES_cur'])
        residual_imbalance = pd.DataFrame(columns=['energy_imbalance'])
        SoC_ts = pd.DataFrame(columns=['SoC'])
        P_dis_RT_ts = pd.DataFrame(columns=['dis_RT'])
        P_cha_RT_ts = pd.DataFrame(columns=['cha_RT'])
        
        

        P_HPP_UP_bid_ts = pd.DataFrame([P_HPP_UP_t0], columns=['bid_up'])
        P_HPP_DW_bid_ts = pd.DataFrame([P_HPP_DW_t0], columns=['bid_dw'])
        s_UP_t = np.zeros(T)
        s_DW_t = np.zeros(T)

        
              
    
        for i in range(0,24):
           if reg_vol_up[i]>0 and reg_vol_dw[i]<0:
               if P_HPP_UP_t0 < reg_vol_up[i]:
                  s_UP_t[i*DI_num:int((i+1/2)*DI_num)] = 1
                  s_DW_t[i*DI_num:int((i+1/2)*DI_num)] = 0
               if -P_HPP_DW_t0 > reg_vol_dw[i]:
                  s_DW_t[int((i+1/2)*DI_num):(i+1)*DI_num] = 1
                  s_UP_t[int((i+1/2)*DI_num):(i+1)*DI_num] = 0
                          
           else:
               if P_HPP_UP_t0 < reg_vol_up[i]:
                  s_UP_t[i*DI_num:(i+1)*DI_num] = 1
                  s_DW_t[i*DI_num:(i+1)*DI_num] = 0
               elif -P_HPP_DW_t0 > reg_vol_dw[i]:
                  s_UP_t[i*DI_num:(i+1)*DI_num] = 0
                  s_DW_t[i*DI_num:(i+1)*DI_num] = 1     
               
           HA_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8 * (DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
           HA_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8 * (DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])
           
#           BM_dw_price_forecast1 = BM_dw_price_forecast
#           BM_up_price_forecast1 = BM_up_price_forecast
#           reg_up_sign_forecast0 = reg_up_sign_forecast
#           reg_dw_sign_forecast0 = reg_dw_sign_forecast
#           if i<24-1:
#               BM_dw_price_forecast1[i+1] = BM_dw_price_cleared[i]
#               BM_up_price_forecast1[i+1] = BM_up_price_cleared[i]
#               if Reg_price_cleared[i] > SM_price_cleared[i+1]:
#                   reg_up_sign_forecast0[i+1] = 1
#                   reg_dw_sign_forecast0[i+1] = 0
#               elif Reg_price_cleared[i] < SM_price_cleared[i+1]:
#                   reg_up_sign_forecast0[i+1] = 0
#                   reg_dw_sign_forecast0[i+1] = 1
#               else:
#                   reg_up_sign_forecast0[i+1] = 0
#                   reg_dw_sign_forecast0[i+1] = 0
#                       
#           BM_dw_price_forecast1[i] = BM_dw_price_cleared[i]
#           BM_up_price_forecast1[i] = BM_up_price_cleared[i]           

           
           BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
           BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
           BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
           BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        
           BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
           BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
           BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
           BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
           
           SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
           # Run BMOpt
           E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = BMOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    HA_wind_forecast1, HA_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator)

           #P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[0, 0]
          # Run RTSim
           
           E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, i * DI_num) 
           
           P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
           P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
           RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
           P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
           P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
           
           P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[1, 0] 
           
          
           exist_imbalance = (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num] - P_HPP_DW_t0 * s_DW_t[i*DI_num]) - P_HPP_SM_t_opt.iloc[i * DI_num,0]) * DI
           #P_HPP_UP_t0 = P_HPP_UP_t_opt.iloc[0, 0]
           #P_HPP_DW_t0 = P_HPP_DW_t_opt.iloc[0, 0]
           if i < 24 - 1:
              P_HPP_UP_t1 = P_HPP_UP_t_opt.iloc[0, 0]
              P_HPP_DW_t1 = P_HPP_DW_t_opt.iloc[0, 0]   
              P_HPP_UP_bid_ts = P_HPP_UP_bid_ts.append(pd.DataFrame([P_HPP_UP_t1], columns=['bid_up']))
              P_HPP_DW_bid_ts = P_HPP_DW_bid_ts.append(pd.DataFrame([P_HPP_DW_t1], columns=['bid_dw']))
           else:
               P_HPP_UP_t1 = 0
               P_HPP_DW_t1 = 0
               

           
           P_dis_t0 = P_dis_HA_t_opt.iloc[1,0] 
           P_cha_t0 = P_cha_HA_t_opt.iloc[1,0]     
           #SoC0 = SoC_HA_t_opt.iloc[1,0]
           SoC0 = SoC_RT_t_opt.iloc[1,0]
           for j in range(1, DI_num):              
               #BM_dw_price = BM_dw_price_forecast
               #BM_up_price = BM_up_price_forecast
               #BM_dw_price[i] = BM_dw_price_cleared[i]
               #BM_up_price[i] = BM_up_price_cleared[i]
               RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):]-Wind_measurement.values[(i+2)*int(1/DI):])])
               RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement[(i+2)*int(1/DI):])])
               SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
               
               RT_interval = i * DI_num + j
               # Run RDOpt
 
               E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = RDOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                      RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, RT_interval, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, P_HPP_UP_t1, P_HPP_DW_t1, SoC0, exist_imbalance, exten_num, deg_indicator)
                   #P_HPP_RD_t_opt = P_HPP_HA_t_opt
                   #P_dis_RD_t_opt = P_dis_HA_t_opt
                   #P_cha_RD_t_opt = P_cha_HA_t_opt
                              # Run RTSim
               E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                        Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval)
               P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
               P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
               RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
               P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
               P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
               
               if RT_interval < T - 1:
                   P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1,0]  
               
                   
               if RT_interval%SIDI_num == SIDI_num-1:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                   residual_imbalance = residual_imbalance.append(pd.DataFrame([exist_imbalance], columns=['energy_imbalance'])) 
                   exist_imbalance = 0
               elif RT_interval%SIDI_num == 0: 
                   exist_imbalance = (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               else:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               #P_dis_t0 = P_dis_RD_t_opt.iloc[1,0] 
               #P_cha_t0 = P_cha_RD_t_opt.iloc[1,0]
               #SoC0 = SoC_RD_t_opt.iloc[1,0]
               SoC0 = SoC_RT_t_opt.iloc[1,0]
               
           P_HPP_UP_t0 = P_HPP_UP_t1
           P_HPP_DW_t0 = P_HPP_DW_t1
        
        SM_revenue, reg_revenue, im_revenue, BM_revenue, im_special_revenue_DK1 = Revenue_calculation(DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_k_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num)   
        

        
        #SoC_all = pd.read_excel('results_run.xlsx', sheet_name = 'SoC', nrows=(day_num-1)*T, engine='openpyxl')
        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_all = SoC_all.append(SoC_ts) 
        
        SoC_for_rainflow = SoC_all.values.tolist()
        SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num*T))]
    
    
        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
        
        Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost
        
    
        P_HPP_RT_ts.index = time_index[:T]
        P_HPP_RT_refs.index = time_index[:T]
        P_dis_RT_ts.index = time_index[:T]
        P_cha_RT_ts.index = time_index[:T]
        #P_HPP_UP_bid_ts.index = time_index[:T]
        #P_HPP_DW_bid_ts.index = time_index[:T]
        #RES_RT_cur_ts = time_index[:T]
        
        output_schedule = pd.concat([P_HPP_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, P_dis_RT_ts, P_cha_RT_ts], axis=1)
        output_revenue = pd.DataFrame([SM_revenue, reg_revenue, im_revenue, im_special_revenue_DK1, Deg_cost]).T
        output_revenue.columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost']
        output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts], axis=1)
        output_act_signal = pd.concat([pd.DataFrame(s_UP_t, columns=['signal_up']), pd.DataFrame(s_DW_t, columns=['signal_down'])], axis=1)
        
        if day_num == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld'])], axis=1)
                             
        write_results(output_schedule, out_dir+'schedule.csv', 'csv', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        write_results(output_bids, out_dir+'reg_bids.csv', 'csv', (day_num-1)*T_BI, [0,1], 'power bids')    
        write_results(output_act_signal, out_dir+'act_signal.csv', 'csv', (day_num-1)*T, [0,1], 'act_signal')
        write_results(output_deg, out_dir+'Degradation.csv', 'csv', (day_num-1)*T, [0,1], 'Degradation')
        write_results(SoC_ts, out_dir+'SoC.csv', 'csv', (day_num-1)*T, [0], 'SoC')
        write_results(residual_imbalance, out_dir+'energy_imbalance.csv', 'csv', (day_num-1)*T_SI, [0], 'energy imbalance')
        write_results(RES_RT_cur_ts, out_dir+'curtailment.csv', 'csv', (day_num-1)*T, [0], 'RES curtailment')
        write_results(output_revenue, out_dir+'revenue.csv', 'csv', day_num-1, [0,1,2,3], 'Revenue')

        
        
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[3])
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[4])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all)
        
        write_results(pd.DataFrame([ad], columns=['slope']), out_dir+'slope.csv', 'csv', day_num-1, [0], 'Degradation')

        if nld>0.2:
            break
        
        pre_nld = nld
        day_num = day_num + 1
        if day_num > simulation_dict["number_of_run_day"]:
            print(P_grid_limit)
            break
                             
#    return P_HPP_SM_t_opt, P_HPP_RT_refs, P_HPP_RT_ts, P_dis_RT_ts, P_cha_RT_ts, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, RES_RT_cur_ts, SM_revenue, reg_revenue, im_revenue, im_special_revenue_DK1, output_deg 
    
    
    
    
def run_SM(parameter_dict, simulation_dict):

    DI = 1/4
    DI_num = int(1/DI)    
    T = int(1/DI*24)
        
    
    SI = 1/4
    SI_num = int(1/SI)
    T_SI = int(24/SI)
    SIDI_num = int(SI/DI)
    
  
    BI = 1
    BI_num = int(1/BI)
    T_BI = int(24/BI)
    
    Wind_component = simulation_dict["wind_as_component"]
    Solar_component = simulation_dict["solar_as_component"]
    BESS_component = simulation_dict["battery_as_component"]
    
    PwMax = parameter_dict["wind_capacity"] * Wind_component
    PsMax = parameter_dict["solar_capacity"] * Solar_component
    EBESS = parameter_dict["battery_energy_capacity"]     
    PbMax = parameter_dict["battery_power_capacity"] * BESS_component  
    SoCmin = parameter_dict["battery_minimum_SoC"] * BESS_component  
    SoCmax = parameter_dict["battery_maximum_SoC"] * BESS_component
    SoCini = parameter_dict["battery_initial_SoC"] * BESS_component
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"] * BESS_component
    
    
    day_num = 1
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1.11e-6   # slope   
    capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    replace_percent = 0.2     
    total_cycles = 3500
                             
    PreUp = PreDw = 0
    P_grid_limit = parameter_dict["hpp_grid_connection"]

    mu = parameter_dict["battery_marginal_degradation_cost"]
    
    deg_indicator = parameter_dict["degradation_in_optimization"]
    
    
    
    
    P_HPP_UP_t0 = 0
    P_HPP_DW_t0 = 0
      
    
    
    #SoC_all = pd.DataFrame(columns = ['SoC_all'])
    
    exten_num = 0
    out_dir = simulation_dict['out_dir']

        
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    
    re  = pd.DataFrame(list(), columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost','Deg_cost_by_cycle'])
    sig = pd.DataFrame(list(), columns=['signal_up','signal_down'])
    cur = pd.DataFrame(list(), columns=['RES_cur'])
    de  = pd.DataFrame(list(), columns=['nld','ld','cycles'])
    ei  = pd.DataFrame(list(), columns=['energy_imbalance'])
    reg = pd.DataFrame(list(), columns=['bid_up','bid_dw'])
    shc = pd.DataFrame(list(), columns=['SM','RT','Ref','dis_RT','cha_RT'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])

    sig.to_csv(out_dir+'act_signal.csv',index=False)
    cur.to_csv(out_dir+'curtailment.csv',index=False)
    de.to_csv(out_dir+'Degradation.csv',index=False)
    ei.to_csv(out_dir+'energy_imbalance.csv',index=False)
    reg.to_csv(out_dir+'reg_bids.csv',index=False)
    re.to_csv(out_dir+'revenue.csv',index=False)
    shc.to_csv(out_dir+'schedule.csv',index=False)
    slo.to_csv(out_dir+'slope.csv',index=False)
    soc.to_csv(out_dir+'SoC.csv',index=False)
    while day_num:
        Emax = EBESS*(1-pre_nld)
        
        DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
        

        
        
        SM_price_forecast = SM_price_forecast.squeeze().repeat(DI_num)
        SM_price_forecast.index = range(T + exten_num)
        
        
        
    # Call EMS Model
        # Run SMOpt
        E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt = SMOpt(DI, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                    DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator)
        
        P_HPP_SM_t_opt.index = time_index[:T]
        
        #write_results(P_HPP_SM_t_opt, 'results_run.xlsx', (day_num-1)*T, 0, 'power schedule')

                      
        P_HPP_RT_ts = pd.DataFrame(columns=['RT'])
        P_HPP_RT_refs = pd.DataFrame(columns=['Ref'])
        RES_RT_cur_ts = pd.DataFrame(columns=['RES_cur'])
        residual_imbalance = pd.DataFrame(columns=['energy_imbalance'])
        SoC_ts = pd.DataFrame(columns=['SoC'])
        P_dis_RT_ts = pd.DataFrame(columns=['dis_RT'])
        P_cha_RT_ts = pd.DataFrame(columns=['cha_RT'])
        
        P_HPP_UP_bid_ts = np.zeros(24)
        P_HPP_DW_bid_ts = np.zeros(24)
        P_HPP_UP_bid_ts = pd.DataFrame(P_HPP_UP_bid_ts, columns=['bid_up'])
        P_HPP_DW_bid_ts = pd.DataFrame(P_HPP_DW_bid_ts, columns=['bid_dw'])
        s_UP_t = np.zeros(T)
        s_DW_t = np.zeros(T)
              
        
        for i in range(0,24):
           exist_imbalance = 0
           for j in range(0, DI_num):              
               BM_dw_price = BM_dw_price_forecast
               BM_up_price = BM_up_price_forecast
               BM_dw_price[i] = BM_dw_price_cleared[i]
               BM_up_price[i] = BM_up_price_cleared[i]
               #RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], DA_wind_forecast.values[(i+2)*int(1/DI):]])
               #RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], DA_solar_forecast.values[(i+2)*int(1/DI):]])
               SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
               
               RT_interval = i * DI_num + j
               P_HPP_RT_ref = P_HPP_SM_t_opt.iloc[RT_interval, 0]  
               # Run RTSim
               E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                        Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval)
               P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
               P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
               RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
               P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
               P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
               

               
                   
               if RT_interval%SIDI_num == SIDI_num-1:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                   residual_imbalance = residual_imbalance.append(pd.DataFrame([exist_imbalance], columns=['energy_imbalance']))
                   exist_imbalance = 0
               elif RT_interval%SIDI_num == 0: 
                   exist_imbalance = (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               else:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               #P_dis_t0 = P_dis_RD_t_opt.iloc[1,0] 
               #P_cha_t0 = P_cha_RD_t_opt.iloc[1,0]
               #SoC0 = SoC_RD_t_opt.iloc[1,0]
               SoC0 = SoC_RT_t_opt.iloc[1,0]
               

        
        SM_revenue, reg_revenue, im_revenue, BM_revenue, im_special_revenue_DK1 = Revenue_calculation(DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_k_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num)    
        
        
        #SoC_all = pd.read_excel('results_run.xlsx', sheet_name = 'SoC', nrows=(day_num-1)*T, engine='openpyxl')
        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_all = SoC_all.append(SoC_ts) 
        
        SoC_for_rainflow = SoC_all.values.tolist()
        SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num*T))]
    
    
        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
        
        Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost

        if day_num==1:
           Deg_cost_by_cycle = cycles.iloc[0,0]/total_cycles * EBESS * capital_cost  
        else:                
           Deg = pd.read_csv(out_dir+'Degradation.csv') 
           cycle_of_day = Deg.iloc[-1,2] - Deg.iloc[-2,2] 
           Deg_cost_by_cycle = cycle_of_day/total_cycles * EBESS * capital_cost        
    
        P_HPP_RT_ts.index = time_index[:T]
        P_HPP_RT_refs.index = time_index[:T]
        P_dis_RT_ts.index = time_index[:T]
        P_cha_RT_ts.index = time_index[:T]
        output_schedule = pd.concat([P_HPP_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, P_dis_RT_ts, P_cha_RT_ts], axis=1)
        output_revenue = pd.DataFrame([SM_revenue, reg_revenue, im_revenue, im_special_revenue_DK1, Deg_cost, Deg_cost_by_cycle]).T
        output_revenue.columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost','Deg_cost_by_cycle']
        output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts], axis=1)
        output_act_signal = pd.concat([pd.DataFrame(s_UP_t, columns=['signal_up']), pd.DataFrame(s_DW_t, columns=['signal_down'])], axis=1)
        if day_num == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld']), pd.DataFrame([0, cycles.iloc[0,0]], columns=['cycles'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld']), cycles], axis=1)

        #write_results(output_schedule, 'results_run.xlsx', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        #write_results(output_bids, 'results_run.xlsx', (day_num-1)*T_BI, [0,1], 'power bids')
        #write_results(pd.DataFrame(s_UP_t, columns=['signal_up']), 'results_run.xlsx', (day_num-1)*T, [0], 'act_signal')
        #write_results(pd.DataFrame(s_DW_t, columns=['signal_down']), 'results_run.xlsx', (day_num-1)*T, [1], 'act_signal')
        #write_results(SoC_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'SoC')
        #write_results(residual_imbalance, 'results_run.xlsx', (day_num-1)*T_SI, [0], 'energy imbalance')
        #write_results(RES_RT_cur_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'RES curtailment')
        #write_results(output_revenue, 'results_run.xlsx', day_num-1, [0,1,2,3], 'Revenue')
        #write_results(pd.DataFrame([Ini_nld, nld], columns=['nld']), 'results_run.xlsx', day_num-1, [0], 'Degradation')
        #write_results(pd.DataFrame([ld0, ld], columns=['ld']), 'results_run.xlsx', day_num-1, [1], 'Degradation')
        
        write_results(output_schedule, out_dir+'schedule.csv', 'csv', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        write_results(output_bids, out_dir+'reg_bids.csv', 'csv', (day_num-1)*T_BI, [0,1], 'power bids')    
        write_results(output_act_signal, out_dir+'act_signal.csv', 'csv', (day_num-1)*T, [0,1], 'act_signal')
        write_results(output_deg, out_dir+'Degradation.csv', 'csv', (day_num-1)*T, [0,1], 'Degradation')
        write_results(SoC_ts, out_dir+'SoC.csv', 'csv', (day_num-1)*T, [0], 'SoC')
        write_results(residual_imbalance, out_dir+'energy_imbalance.csv', 'csv', (day_num-1)*T_SI, [0], 'energy imbalance')
        write_results(RES_RT_cur_ts, out_dir+'curtailment.csv', 'csv', (day_num-1)*T, [0], 'RES curtailment')
        write_results(output_revenue, out_dir+'revenue.csv', 'csv', day_num-1, [0,1,2,3], 'Revenue')

        
        
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[3])
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[4])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all)
        
        write_results(pd.DataFrame([ad], columns=['slope']), out_dir+'slope.csv', 'csv', day_num-1, [0], 'Degradation')
        if nld>0.2:
            break
        
        
        pre_nld = nld
        day_num = day_num + 1  
        if day_num > simulation_dict["number_of_run_day"]:
            print(P_grid_limit)
            break

def run_SM_BM(parameter_dict, simulation_dict):
    DI = 1/4
    DI_num = int(1/DI)    
    T = int(1/DI*24)
        
    
    SI = 1/4
    SI_num = int(1/SI)
    T_SI = int(24/SI)
    SIDI_num = int(SI/DI)
    
  
    BI = 1
    BI_num = int(1/BI)
    T_BI = int(24/BI)
    
    Wind_component = simulation_dict["wind_as_component"]
    Solar_component = simulation_dict["solar_as_component"]
    BESS_component = simulation_dict["battery_as_component"]
    
    PwMax = parameter_dict["wind_capacity"] * Wind_component
    PsMax = parameter_dict["solar_capacity"] * Solar_component
    EBESS = parameter_dict["battery_energy_capacity"]     
    PbMax = parameter_dict["battery_power_capacity"] * BESS_component  
    SoCmin = parameter_dict["battery_minimum_SoC"] * BESS_component  
    SoCmax = parameter_dict["battery_maximum_SoC"] * BESS_component
    SoCini = parameter_dict["battery_initial_SoC"] * BESS_component
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"] * BESS_component
    
    
    day_num = 1
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1.11e-6   # slope   
    capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    replace_percent = 0.2     

                             
    PreUp = PreDw = 0
    P_grid_limit = parameter_dict["hpp_grid_connection"]

    mu = parameter_dict["battery_marginal_degradation_cost"]
    
    deg_indicator = parameter_dict["degradation_in_optimization"]
    
    
    
    
    P_HPP_UP_t0 = 0
    P_HPP_DW_t0 = 0
      
    
    
    #SoC_all = pd.DataFrame(columns = ['SoC_all'])
    
    exten_num = 0
    out_dir = simulation_dict['out_dir']
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    
    re  = pd.DataFrame(list(), columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost'])
    sig = pd.DataFrame(list(), columns=['signal_up','signal_down'])
    cur = pd.DataFrame(list(), columns=['RES_cur'])
    de  = pd.DataFrame(list(), columns=['nld','ld'])
    ei  = pd.DataFrame(list(), columns=['energy_imbalance'])
    reg = pd.DataFrame(list(), columns=['bid_up','bid_dw'])
    shc = pd.DataFrame(list(), columns=['SM','RT','Ref','dis_RT','cha_RT'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])

    sig.to_csv(out_dir+'act_signal.csv',index=False)
    cur.to_csv(out_dir+'curtailment.csv',index=False)
    de.to_csv(out_dir+'Degradation.csv',index=False)
    ei.to_csv(out_dir+'energy_imbalance.csv',index=False)
    reg.to_csv(out_dir+'reg_bids.csv',index=False)
    re.to_csv(out_dir+'revenue.csv',index=False)
    shc.to_csv(out_dir+'schedule.csv',index=False)
    slo.to_csv(out_dir+'slope.csv',index=False)
    soc.to_csv(out_dir+'SoC.csv',index=False)
    while day_num:
        Emax = EBESS*(1-pre_nld)
        
        DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
        
#        reg_vol_up = reg_vol_up * c_exp
#        reg_vol_dw = reg_vol_dw * c_exp
      
        
        SM_price_forecast = SM_price_forecast.squeeze().repeat(DI_num)
        SM_price_forecast.index = range(T + exten_num)
        
        
        P_HPP_RT_ref = RT_wind_forecast[0] + RT_solar_forecast[0]
        
        
    # Call EMS Model
        # Run SMOpt
        E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt = SMOpt(DI, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                    DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator)
        
        P_HPP_SM_t_opt.index = time_index[:T]
        
        #write_results(P_HPP_SM_t_opt, 'results_run.xlsx', (day_num-1)*T, 0, 'power schedule')

                      
        P_HPP_RT_ts = pd.DataFrame(columns=['RT'])
        P_HPP_RT_refs = pd.DataFrame(columns=['Ref'])
        RES_RT_cur_ts = pd.DataFrame(columns=['RES_cur'])
        residual_imbalance = pd.DataFrame(columns=['energy_imbalance'])
        SoC_ts = pd.DataFrame(columns=['SoC'])
        P_dis_RT_ts = pd.DataFrame(columns=['dis_RT'])
        P_cha_RT_ts = pd.DataFrame(columns=['cha_RT'])
        
        

        P_HPP_UP_bid_ts = pd.DataFrame([P_HPP_UP_t0], columns=['bid_up'])
        P_HPP_DW_bid_ts = pd.DataFrame([P_HPP_DW_t0], columns=['bid_dw'])
        s_UP_t = np.zeros(T)
        s_DW_t = np.zeros(T)

        
              
    
        for i in range(0,24):
           if reg_vol_up[i]>0 and reg_vol_dw[i]<0:
               if P_HPP_UP_t0 < reg_vol_up[i] and P_HPP_UP_t0 != 0:
                  s_UP_t[i*DI_num:int((i+1/2)*DI_num)] = 1
                  s_DW_t[i*DI_num:int((i+1/2)*DI_num)] = 0
               if -P_HPP_DW_t0 > reg_vol_dw[i] and P_HPP_DW_t0 != 0:
                  s_DW_t[int((i+1/2)*DI_num):(i+1)*DI_num] = 1
                  s_UP_t[int((i+1/2)*DI_num):(i+1)*DI_num] = 0
                          
           else:
               if P_HPP_UP_t0 < reg_vol_up[i] and P_HPP_UP_t0 != 0:
                  s_UP_t[i*DI_num:(i+1)*DI_num] = 1
                  s_DW_t[i*DI_num:(i+1)*DI_num] = 0
               elif -P_HPP_DW_t0 > reg_vol_dw[i] and P_HPP_DW_t0 != 0:
                  s_UP_t[i*DI_num:(i+1)*DI_num] = 0
                  s_DW_t[i*DI_num:(i+1)*DI_num] = 1      
               
           HA_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8*(DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
           HA_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8*(DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])

#           BM_dw_price_forecast1 = BM_dw_price_forecast
#           BM_up_price_forecast1 = BM_up_price_forecast
#           reg_up_sign_forecast0 = reg_up_sign_forecast
#           reg_dw_sign_forecast0 = reg_dw_sign_forecast
#           if i<24-1:
#               BM_dw_price_forecast1[i+1] = BM_dw_price_cleared[i]
#               BM_up_price_forecast1[i+1] = BM_up_price_cleared[i]
#               if Reg_price_cleared[i] > SM_price_cleared[i+1]:
#                   reg_up_sign_forecast0[i+1] = 1
#                   reg_dw_sign_forecast0[i+1] = 0
#               elif Reg_price_cleared[i] < SM_price_cleared[i+1]:
#                   reg_up_sign_forecast0[i+1] = 0
#                   reg_dw_sign_forecast0[i+1] = 1
#               else:
#                   reg_up_sign_forecast0[i+1] = 0
#                   reg_dw_sign_forecast0[i+1] = 0
#                       
#           BM_dw_price_forecast1[i] = BM_dw_price_cleared[i]
#           BM_up_price_forecast1[i] = BM_up_price_cleared[i]           

           
           BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
           BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
           BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
           BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        
           BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
           BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
           BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
           BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))

           SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
           # Run BMOpt
           E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = BMOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    HA_wind_forecast1, HA_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator)

          # Run RTSim
           
           E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, i * DI_num) 
           
           P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
           P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
           RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
           P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
           P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
           
           P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[1, 0] 
           
          
           exist_imbalance = (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num] - P_HPP_DW_t0 * s_DW_t[i*DI_num]) - P_HPP_SM_t_opt.iloc[i * DI_num,0]) * DI
           #P_HPP_UP_t0 = P_HPP_UP_t_opt.iloc[0, 0]
           #P_HPP_DW_t0 = P_HPP_DW_t_opt.iloc[0, 0]
           if i < 24 - 1:
              P_HPP_UP_t1 = P_HPP_UP_t_opt.iloc[0, 0]
              P_HPP_DW_t1 = P_HPP_DW_t_opt.iloc[0, 0]   
              P_HPP_UP_bid_ts = P_HPP_UP_bid_ts.append(pd.DataFrame([P_HPP_UP_t1], columns=['bid_up']))
              P_HPP_DW_bid_ts = P_HPP_DW_bid_ts.append(pd.DataFrame([P_HPP_DW_t1], columns=['bid_dw']))
           else:
               P_HPP_UP_t1 = 0
               P_HPP_DW_t1 = 0
               

           
           P_dis_t0 = P_dis_HA_t_opt.iloc[1,0] 
           P_cha_t0 = P_cha_HA_t_opt.iloc[1,0]     
           #SoC0 = SoC_HA_t_opt.iloc[1,0]
           SoC0 = SoC_RT_t_opt.iloc[1,0]
           for j in range(1, DI_num):              
               BM_dw_price = BM_dw_price_forecast
               BM_up_price = BM_up_price_forecast
               BM_dw_price[i] = BM_dw_price_cleared[i]
               BM_up_price[i] = BM_up_price_cleared[i]
               RD_wind_forecast1 = pd.Series(np.r_[HA_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], DA_wind_forecast.values[(i+2)*int(1/DI):]])
               RD_solar_forecast1 = pd.Series(np.r_[HA_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], DA_solar_forecast.values[(i+2)*int(1/DI):]])
               SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
               
               RT_interval = i * DI_num + j
               # Run RDOpt
 
               #E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = DEMS.RDOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
               #       RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, RT_interval, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, P_HPP_UP_t1, P_HPP_DW_t1, SoC0, exist_imbalance, exten_num)
                   #P_HPP_RD_t_opt = P_HPP_HA_t_opt
                   #P_dis_RD_t_opt = P_dis_HA_t_opt
                   #P_cha_RD_t_opt = P_cha_HA_t_opt

               # Run RTSim
               E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                        Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval)
               P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
               P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
               RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
               P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
               P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
               
               if RT_interval < T - 1:
                   P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[j+1,0]  
               
                   
               if RT_interval%SIDI_num == SIDI_num-1:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                   residual_imbalance = residual_imbalance.append(pd.DataFrame([exist_imbalance], columns=['energy_imbalance'])) 
                   exist_imbalance = 0
               elif RT_interval%SIDI_num == 0: 
                   exist_imbalance = (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               else:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               #P_dis_t0 = P_dis_RD_t_opt.iloc[1,0] 
               #P_cha_t0 = P_cha_RD_t_opt.iloc[1,0]
               #SoC0 = SoC_RD_t_opt.iloc[1,0]
               SoC0 = SoC_RT_t_opt.iloc[1,0]
               
           P_HPP_UP_t0 = P_HPP_UP_t1
           P_HPP_DW_t0 = P_HPP_DW_t1
        
        SM_revenue, reg_revenue, im_revenue, BM_revenue, im_special_revenue_DK1 = Revenue_calculation(DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_k_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num)     
        

        #SoC_all = pd.read_excel('results_run.xlsx', sheet_name = 'SoC', nrows=(day_num-1)*T, engine='openpyxl')
        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_all = SoC_all.append(SoC_ts) 
        
        SoC_for_rainflow = SoC_all.values.tolist()
        SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num*T))]
    
    
        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
        
        Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost
        
    
        P_HPP_RT_ts.index = time_index[:T]
        P_HPP_RT_refs.index = time_index[:T]
        P_dis_RT_ts.index = time_index[:T]
        P_cha_RT_ts.index = time_index[:T]
        output_schedule = pd.concat([P_HPP_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, P_dis_RT_ts, P_cha_RT_ts], axis=1)
        output_revenue = pd.DataFrame([SM_revenue, reg_revenue, im_revenue, im_special_revenue_DK1, Deg_cost]).T
        output_revenue.columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost']
        output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts], axis=1)
        output_act_signal = pd.concat([pd.DataFrame(s_UP_t, columns=['signal_up']), pd.DataFrame(s_DW_t, columns=['signal_down'])], axis=1)
        if day_num == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld'])], axis=1)
        #write_results(output_schedule, 'results_run.xlsx', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        #write_results(output_bids, 'results_run.xlsx', (day_num-1)*T_BI, [0,1], 'power bids')
        #write_results(pd.DataFrame(s_UP_t, columns=['signal_up']), 'results_run.xlsx', (day_num-1)*T, [0], 'act_signal')
        #write_results(pd.DataFrame(s_DW_t, columns=['signal_down']), 'results_run.xlsx', (day_num-1)*T, [1], 'act_signal')
        #write_results(SoC_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'SoC')
        #write_results(residual_imbalance, 'results_run.xlsx', (day_num-1)*T_SI, [0], 'energy imbalance')
        #write_results(RES_RT_cur_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'RES curtailment')
        #write_results(output_revenue, 'results_run.xlsx', day_num-1, [0,1,2,3], 'Revenue')
        #write_results(pd.DataFrame([Ini_nld, nld], columns=['nld']), 'results_run.xlsx', day_num-1, [0], 'Degradation')
        #write_results(pd.DataFrame([ld0, ld], columns=['ld']), 'results_run.xlsx', day_num-1, [1], 'Degradation')
        write_results(output_schedule, out_dir+'schedule.csv', 'csv', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        write_results(output_bids, out_dir+'reg_bids.csv', 'csv', (day_num-1)*T_BI, [0,1], 'power bids')    
        write_results(output_act_signal, out_dir+'act_signal.csv', 'csv', (day_num-1)*T, [0,1], 'act_signal')
        write_results(output_deg, out_dir+'Degradation.csv', 'csv', (day_num-1)*T, [0,1], 'Degradation')
        write_results(SoC_ts, out_dir+'SoC.csv', 'csv', (day_num-1)*T, [0], 'SoC')
        write_results(residual_imbalance, out_dir+'energy_imbalance.csv', 'csv', (day_num-1)*T_SI, [0], 'energy imbalance')
        write_results(RES_RT_cur_ts, out_dir+'curtailment.csv', 'csv', (day_num-1)*T, [0], 'RES curtailment')
        write_results(output_revenue, out_dir+'revenue.csv', 'csv', day_num-1, [0,1,2,3], 'Revenue')

        
        
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[3])
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[4])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all)
        
        write_results(pd.DataFrame([ad], columns=['slope']), out_dir+'slope.csv', 'csv', day_num-1, [0], 'Degradation')
        if nld>0.2:
            break
        

        pre_nld = nld
        day_num = day_num + 1 
        if day_num > simulation_dict["number_of_run_day"]:
            print(P_grid_limit)
            break


def run_SM_RD(parameter_dict, simulation_dict):

    DI = 1/12
    DI_num = int(1/DI)    
    T = int(1/DI*24)
        
    
    SI = 1/4
    SI_num = int(1/SI)
    T_SI = int(24/SI)
    SIDI_num = int(SI/DI)
    
  
    BI = 1
    BI_num = int(1/BI)
    T_BI = int(24/BI)
    
    Wind_component = simulation_dict["wind_as_component"]
    Solar_component = simulation_dict["solar_as_component"]
    BESS_component = simulation_dict["battery_as_component"]
    
    PwMax = parameter_dict["wind_capacity"] * Wind_component
    PsMax = parameter_dict["solar_capacity"] * Solar_component
    EBESS = parameter_dict["battery_energy_capacity"]     
    PbMax = parameter_dict["battery_power_capacity"] * BESS_component  
    SoCmin = parameter_dict["battery_minimum_SoC"] * BESS_component  
    SoCmax = parameter_dict["battery_maximum_SoC"] * BESS_component
    SoCini = parameter_dict["battery_initial_SoC"] * BESS_component
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"] * BESS_component
    
    
    day_num = 1
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1.11e-6   # slope   
    capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    replace_percent = 0.2     

                             
    PreUp = PreDw = 0
    P_grid_limit = parameter_dict["hpp_grid_connection"]

    mu = parameter_dict["battery_marginal_degradation_cost"]
    total_cycles = 3700
    deg_indicator = parameter_dict["degradation_in_optimization"]
    
    
    
    
    P_HPP_UP_t0 = 0
    P_HPP_DW_t0 = 0
      
    
    
    #SoC_all = pd.DataFrame(columns = ['SoC_all'])
    
    exten_num = 0
    out_dir = simulation_dict['out_dir']

        
    if not os.path.exists(out_dir):
       os.makedirs(out_dir)
    
    re  = pd.DataFrame(list(), columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost','Deg_cost_by_cycle'])
    sig = pd.DataFrame(list(), columns=['signal_up','signal_down'])
    cur = pd.DataFrame(list(), columns=['RES_cur'])
    de  = pd.DataFrame(list(), columns=['nld','ld','cycles'])
    ei  = pd.DataFrame(list(), columns=['energy_imbalance'])
    reg = pd.DataFrame(list(), columns=['bid_up','bid_dw'])
    shc = pd.DataFrame(list(), columns=['SM','RT','Ref','dis_RT','cha_RT'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])

    sig.to_csv(out_dir+'act_signal.csv',index=False)
    cur.to_csv(out_dir+'curtailment.csv',index=False)
    de.to_csv(out_dir+'Degradation.csv',index=False)
    ei.to_csv(out_dir+'energy_imbalance.csv',index=False)
    reg.to_csv(out_dir+'reg_bids.csv',index=False)
    re.to_csv(out_dir+'revenue.csv',index=False)
    shc.to_csv(out_dir+'schedule.csv',index=False)
    slo.to_csv(out_dir+'slope.csv',index=False)
    soc.to_csv(out_dir+'SoC.csv',index=False)
    while day_num:
        Emax = EBESS*(1-pre_nld)
        
        DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
        
       
        
        SM_price_forecast = SM_price_forecast.squeeze().repeat(DI_num)
        SM_price_forecast.index = range(T + exten_num)
        
        
        P_HPP_RT_ref = RT_wind_forecast[0] + RT_solar_forecast[0]
        
        
    # Call EMS Model
        # Run SMOpt
        E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt = SMOpt(DI, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                    DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator)
        
        P_HPP_SM_t_opt.index = time_index[:T]
        
        #write_results(P_HPP_SM_t_opt, 'results_run.xlsx', (day_num-1)*T, 0, 'power schedule')

                      
        P_HPP_RT_ts = pd.DataFrame(columns=['RT'])
        P_HPP_RT_refs = pd.DataFrame(columns=['Ref'])
        RES_RT_cur_ts = pd.DataFrame(columns=['RES_cur'])
        residual_imbalance = pd.DataFrame(columns=['energy_imbalance'])
        SoC_ts = pd.DataFrame(columns=['SoC'])
        P_dis_RT_ts = pd.DataFrame(columns=['dis_RT'])
        P_cha_RT_ts = pd.DataFrame(columns=['cha_RT'])
        
        

        P_HPP_UP_bid_ts = np.zeros(24)
        P_HPP_DW_bid_ts = np.zeros(24)
        P_HPP_UP_bid_ts = pd.DataFrame(P_HPP_UP_bid_ts, columns=['bid_up'])
        P_HPP_DW_bid_ts = pd.DataFrame(P_HPP_DW_bid_ts, columns=['bid_dw'])
        s_UP_t = np.zeros(T)
        s_DW_t = np.zeros(T)

        
              
    
        for i in range(0,24):   
           RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8*(DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
           RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8*(DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])

#           BM_dw_price_forecast1 = BM_dw_price_forecast
#           BM_up_price_forecast1 = BM_up_price_forecast
#           reg_up_sign_forecast0 = reg_up_sign_forecast
#           reg_dw_sign_forecast0 = reg_dw_sign_forecast
#           if i<24-1:
#               BM_dw_price_forecast1[i+1] = BM_dw_price_cleared[i]
#               BM_up_price_forecast1[i+1] = BM_up_price_cleared[i]
#               if Reg_price_cleared[i] > SM_price_cleared[i+1]:
#                   reg_up_sign_forecast0[i+1] = 1
#                   reg_dw_sign_forecast0[i+1] = 0
#               elif Reg_price_cleared[i] < SM_price_cleared[i+1]:
#                   reg_up_sign_forecast0[i+1] = 0
#                   reg_dw_sign_forecast0[i+1] = 1
#               else:
#                   reg_up_sign_forecast0[i+1] = 0
#                   reg_dw_sign_forecast0[i+1] = 0
#                       
#           BM_dw_price_forecast1[i] = BM_dw_price_cleared[i]
#           BM_up_price_forecast1[i] = BM_up_price_cleared[i]           

           
           BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
           BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
           BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
           BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        
           BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
           BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
           BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
           BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))



           exist_imbalance = 0
           SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
           # Run BMOpt
           #E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = DEMS.BMOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
           #         HA_wind_forecast1, HA_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num)
           E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = RBOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                     RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i*DI_num, s_UP_t, s_DW_t, 0, 0, 0, 0, SoC0, exist_imbalance, exten_num, deg_indicator)
          # Run RTSim
           
           E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, i * DI_num) 
           
           P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
           P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
           RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
           P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
           P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
           
           P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1, 0] 
           
          
           exist_imbalance = (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[i * DI_num,0]) * DI
           #P_HPP_UP_t0 = P_HPP_UP_t_opt.iloc[0, 0]
           #P_HPP_DW_t0 = P_HPP_DW_t_opt.iloc[0, 0]
           if DI == 1/4:
              residual_imbalance = residual_imbalance.append(pd.DataFrame([exist_imbalance], columns=['energy_imbalance'])) 
              exist_imbalance = 0               

           
           #P_dis_t0 = P_dis_HA_t_opt.iloc[1,0] 
           #P_cha_t0 = P_cha_HA_t_opt.iloc[1,0]     
           #SoC0 = SoC_HA_t_opt.iloc[1,0]
           SoC0 = SoC_RT_t_opt.iloc[1,0]
           for j in range(1, DI_num):              

               RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):] - Wind_measurement.values[(i+2)*int(1/DI):])])
               RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement.values[(i+2)*int(1/DI):])])
               SoC_ts = SoC_ts.append(pd.DataFrame([SoC0], columns=['SoC']))
               
               RT_interval = i * DI_num + j
               # Run RDOpt
 
               E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = RBOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                      RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, RT_interval, s_UP_t, s_DW_t, 0, 0, 0, 0, SoC0, exist_imbalance, exten_num, deg_indicator)
                   #P_HPP_RD_t_opt = P_HPP_HA_t_opt
                   #P_dis_RD_t_opt = P_dis_HA_t_opt
                   #P_cha_RD_t_opt = P_cha_HA_t_opt

               # Run RTSim
               E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                        Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval)
               P_HPP_RT_ts = P_HPP_RT_ts.append(pd.DataFrame([P_HPP_RT_t_opt], columns=['RT']))
               P_HPP_RT_refs = P_HPP_RT_refs.append(pd.DataFrame([P_HPP_RT_ref], columns=['Ref']))
               RES_RT_cur_ts = RES_RT_cur_ts.append(pd.DataFrame([RES_RT_cur_t_opt], columns=['RES_cur']))
               P_dis_RT_ts = P_dis_RT_ts.append(pd.DataFrame([P_dis_RT_t_opt], columns=['dis_RT']))
               P_cha_RT_ts = P_cha_RT_ts.append(pd.DataFrame([P_cha_RT_t_opt], columns=['cha_RT']))
               
               if RT_interval < T - 1:
                   P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1,0]  
               
                   
               if RT_interval%SIDI_num == SIDI_num-1:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                   residual_imbalance = residual_imbalance.append(pd.DataFrame([exist_imbalance], columns=['energy_imbalance'])) 
                   exist_imbalance = 0
               elif RT_interval%SIDI_num == 0: 
                   exist_imbalance = (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               else:
                   exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
               #P_dis_t0 = P_dis_RD_t_opt.iloc[1,0] 
               #P_cha_t0 = P_cha_RD_t_opt.iloc[1,0]
               #SoC0 = SoC_RD_t_opt.iloc[1,0]
               SoC0 = SoC_RT_t_opt.iloc[1,0]
               
        
        SM_revenue, reg_revenue, im_revenue, BM_revenue, im_special_revenue_DK1 = Revenue_calculation(DI_num, T_SI, SI_num, SIDI_num, T, DI, SI, BI, P_HPP_SM_k_opt, P_HPP_RT_ts, P_HPP_RT_refs, SM_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, s_UP_t, s_DW_t, residual_imbalance, exten_num)    
        
        #SoC_all = pd.read_excel('results_run.xlsx', sheet_name = 'SoC', nrows=(day_num-1)*T, engine='openpyxl')
        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_all = SoC_all.append(SoC_ts) 
        
        SoC_for_rainflow = SoC_all.values.tolist()
        SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num*T))]
    
    
        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
        
        Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost

        if day_num==1:
           Deg_cost_by_cycle = cycles.iloc[0,0]/total_cycles * EBESS * capital_cost  
        else:                
           Deg = pd.read_csv(out_dir+'Degradation.csv') 
           cycle_of_day = Deg.iloc[-1,2] - Deg.iloc[-2,2] 
           Deg_cost_by_cycle = cycle_of_day/total_cycles * EBESS * capital_cost        
    
        P_HPP_RT_ts.index = time_index[:T]
        P_HPP_RT_refs.index = time_index[:T]
        P_dis_RT_ts.index = time_index[:T]
        P_cha_RT_ts.index = time_index[:T]
        output_schedule = pd.concat([P_HPP_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, P_dis_RT_ts, P_cha_RT_ts], axis=1)
        output_revenue = pd.DataFrame([SM_revenue, reg_revenue, im_revenue, im_special_revenue_DK1, Deg_cost, Deg_cost_by_cycle]).T
        output_revenue.columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost','Deg_cost_by_cycle']
        output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts], axis=1)
        output_act_signal = pd.concat([pd.DataFrame(s_UP_t, columns=['signal_up']), pd.DataFrame(s_DW_t, columns=['signal_down'])], axis=1)
        if day_num == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld']), pd.DataFrame([0, cycles.iloc[0,0]], columns=['cycles'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld']), cycles], axis=1)
        #write_results(output_schedule, 'results_run.xlsx', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        #write_results(output_bids, 'results_run.xlsx', (day_num-1)*T_BI, [0,1], 'power bids')
        #write_results(pd.DataFrame(s_UP_t, columns=['signal_up']), 'results_run.xlsx', (day_num-1)*T, [0], 'act_signal')
        #write_results(pd.DataFrame(s_DW_t, columns=['signal_down']), 'results_run.xlsx', (day_num-1)*T, [1], 'act_signal')
        #write_results(SoC_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'SoC')
        #write_results(residual_imbalance, 'results_run.xlsx', (day_num-1)*T_SI, [0], 'energy imbalance')
        #write_results(RES_RT_cur_ts, 'results_run.xlsx', (day_num-1)*T, [0], 'RES curtailment')
        #write_results(output_revenue, 'results_run.xlsx', day_num-1, [0,1,2,3], 'Revenue')
        #write_results(pd.DataFrame([Ini_nld, nld], columns=['nld']), 'results_run.xlsx', day_num-1, [0], 'Degradation')
        #write_results(pd.DataFrame([ld0, ld], columns=['ld']), 'results_run.xlsx', day_num-1, [1], 'Degradation')
        #out_dir = './results/'
        write_results(output_schedule, out_dir+'schedule.csv', 'csv', (day_num-1)*T, [0,1,2,3,4], 'power schedule')
        write_results(output_bids, out_dir+'reg_bids.csv', 'csv', (day_num-1)*T_BI, [0,1], 'power bids')    
        write_results(output_act_signal, out_dir+'act_signal.csv', 'csv', (day_num-1)*T, [0,1], 'act_signal')
        write_results(output_deg, out_dir+'Degradation.csv', 'csv', (day_num-1)*T, [0,1], 'Degradation')
        write_results(SoC_ts, out_dir+'SoC.csv', 'csv', (day_num-1)*T, [0], 'SoC')
        write_results(residual_imbalance, out_dir+'energy_imbalance.csv', 'csv', (day_num-1)*T_SI, [0], 'energy imbalance')
        write_results(RES_RT_cur_ts, out_dir+'curtailment.csv', 'csv', (day_num-1)*T, [0], 'RES curtailment')
        write_results(output_revenue, out_dir+'revenue.csv', 'csv', day_num-1, [0,1,2,3], 'Revenue')

        
        
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[3])
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[4])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all)
        
        write_results(pd.DataFrame([ad], columns=['slope']), out_dir+'slope.csv', 'csv', day_num-1, [0], 'Degradation')
        
        if nld>0.2:
            break
        pre_nld = nld
        day_num = day_num + 1  




        if day_num > simulation_dict["number_of_run_day"]:
            print(P_grid_limit)
            break
    
   
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    