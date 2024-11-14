import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import os
from hydesign.HiFiEMS import Deg_Calculation as DegCal
import math
from docplex.mp.model import Model
from tqdm import tqdm

def ReadHistoricalData(PsMax, PwMax, T, DI_num, demension): 
    History_wind = pd.read_csv('../Data/probabilistic_forecast2022.csv')
    History_price = pd.read_csv('../Data/Market2021.csv')  
    mean_wind_DA_error = (History_wind['DA'] - History_wind['Measurement']).mean() * PwMax
    mean_wind_HA_error = (History_wind['HA'] - History_wind['Measurement']).mean() * PwMax
    
    History_wind_DA_error = (History_wind['DA'] - History_wind['Measurement']) * PwMax 
    History_wind_HA_error = (History_wind['HA'][0::int(4/DI_num)] - History_wind['Measurement'][0::int(4/DI_num)]) * PwMax
    
    History_spot_price_error = (History_price['SM_forecast'] - History_price['SM_cleared'])

    History_wind_DA_error = History_wind_DA_error.to_numpy()
    History_spot_price_error = History_spot_price_error.to_numpy()
    
    History_wind_DA_error = History_wind_DA_error.reshape((int(len(History_wind_DA_error)/(T/demension)), int(T/demension)))
    History_wind_DA_error = np.mean(History_wind_DA_error,axis=1)
    History_wind_DA_error = pd.DataFrame(History_wind_DA_error.reshape((int(len(History_wind_DA_error)/demension), demension)))
    History_wind_DA_error = History_wind_DA_error.sample(frac=1) # shuffle
    
    
    
    demension = 24
    History_spot_price_error = History_spot_price_error.reshape((int(len(History_spot_price_error)/(24/demension)), int(24/demension)))
    History_spot_price_error = np.mean(History_spot_price_error,axis=1)    
    History_spot_price_error = pd.DataFrame(History_spot_price_error.reshape((int(len(History_spot_price_error)/demension), demension)))    

    return History_wind_DA_error, History_wind_HA_error, mean_wind_DA_error, mean_wind_HA_error, History_spot_price_error, History_price




def f_xmin_to_ymin(x,reso_x, reso_y):  #x: dataframe reso: in hour
    x = np.asarray(x).squeeze()
    y = pd.DataFrame()
    if reso_y > reso_x:
        a=0
        num = int(reso_y/reso_x)
        
        for ii in range(len(x)):        
            if ii%num == num-1:
                a = (a + x[ii]) /num   
                y = y.append(pd.DataFrame([a]))
                a = 0
            else:                       
                a = a + x[ii]    
    else:
        y = pd.DataFrame(np.repeat(x,int(reso_x/reso_y)))
        num = int(reso_x/reso_y)
    y.index = range(int(len(x)/num))
    return y


def scenario_generation(simulation_dict, SM_price_cleared, BM_dw_price_cleared, BM_up_price_cleared, Reg_price_cleared,PsMax, PwMax, T, DI_num, demension):

    History_wind_DA_error, History_wind_HA_error, mean_wind_DA_error, mean_wind_HA_error, History_spot_price_error, History_price = ReadHistoricalData(PsMax, PwMax,T, DI_num, demension)   
    
    spot_price = pd.concat([History_price['SM_cleared'], SM_price_cleared])
    spot_price.index = range(len(spot_price))
    spot_price = spot_price.to_numpy().reshape((int(len(History_price['SM_cleared'])/24+1), 24))
    
    # step 1: clustering based on spot price to find similar days
    # wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(spot_price)
    #     wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()
    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0).fit(spot_price)
    
    cluster_num = kmeans.predict([SM_price_cleared.values.tolist()])
    
    similar_spot_prices_indces = np.where(kmeans.labels_ == cluster_num)[0]
    
    
    # step 2: From the cluster that contains spot price of operating day, do clustering again for regulting price
    
    reg_price = History_price['reg_cleared'].to_numpy().reshape((int(len(History_price['reg_cleared'])/24), 24))
    reg_price = reg_price[similar_spot_prices_indces[:-1],:]

    # wcss = []
    # for i in range(1, 11):
    #     kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    #     kmeans.fit(reg_price)
    #     wcss.append(kmeans.inertia_)
    # plt.plot(range(1, 11), wcss)
    # plt.title('Elbow Method')
    # plt.xlabel('Number of clusters')
    # plt.ylabel('WCSS')
    # plt.show()


    kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kM=kmeans.fit(reg_price)
    labels = kM.predict(reg_price)

    clusterCount = np.bincount(labels)
    probability = clusterCount/clusterCount.sum()
    scenarios = kmeans.cluster_centers_
    
    reg_forecast = np.zeros(24)     
    for j in range(24):
        reg_forecast[j] = sum(probability[i]*scenarios[i,j] for i in range(4))
        
    BP_up_forecast = np.zeros(24)
    BP_dw_forecast = np.zeros(24)   
    for j in range(24):
        if reg_forecast[j] >= SM_price_cleared.iloc[j]:
           BP_up_forecast[j] = reg_forecast[j] 
           BP_dw_forecast[j] = SM_price_cleared[j]
        else:
           BP_up_forecast[j] = SM_price_cleared[j] 
           BP_dw_forecast[j] = reg_forecast[j]        
    
    if simulation_dict['BP'] == 2:
       BP_up_forecast = BM_up_price_cleared.to_numpy()
       BP_dw_forecast = BM_dw_price_cleared.to_numpy()
       reg_forecast = Reg_price_cleared 
    return probability, BP_up_forecast, BP_dw_forecast, reg_forecast

def _revenue_calculation(parameter_dict,
                         P_HPP_SM_t_opt,
                         P_HPP_RT_ts,
                         P_HPP_RT_refs,
                         SM_price_cleared,
                         BM_dw_price_cleared,
                         BM_up_price_cleared,
                         P_HPP_UP_bid_ts,
                         P_HPP_DW_bid_ts,
                         s_UP_t,
                         s_DW_t,
                         BI,):
    DI = parameter_dict["dispatch_interval"]
    DI_num = int(1/DI)    
    
    SI = parameter_dict["settlement_interval"]
    SI_num = int(1/SI)

    # Spot market revenue
    SM_price_cleared_DI = SM_price_cleared.repeat(DI_num)
    SM_revenue = (P_HPP_SM_t_opt.squeeze()*SM_price_cleared_DI*DI)

    # Regulation revenue
    BM_up_price_cleared_DI = BM_up_price_cleared.repeat(DI_num)
    BM_dw_price_cleared_DI = BM_dw_price_cleared.repeat(DI_num)
   
    s_UP_t = pd.Series(s_UP_t)
    s_DW_t = pd.Series(s_DW_t)

    reg_revenue = (s_UP_t*P_HPP_UP_bid_ts.squeeze()*DI*BM_up_price_cleared_DI) - (s_DW_t*P_HPP_DW_bid_ts.squeeze()*BI*BM_dw_price_cleared_DI) 
    
    # Imbalance revenue
    BM_up_price_cleared_SI = BM_up_price_cleared.repeat(SI_num)
    BM_dw_price_cleared_SI = BM_dw_price_cleared.repeat(SI_num)
    P_HPP_RT_ts_15min = f_xmin_to_ymin(P_HPP_RT_ts, DI, 1/4)    
    P_HPP_RT_refs_15min = f_xmin_to_ymin(P_HPP_RT_refs, DI, 1/4)
    
    power_imbalance = pd.Series((P_HPP_RT_ts_15min.values -P_HPP_RT_refs_15min.values)[:,0])

    pos_imbalance = power_imbalance.apply(lambda x: x if x > 0 else 0)
    neg_imbalance = power_imbalance.apply(lambda x: x if x < 0 else 0)

    im_revenue = np.asarray(pos_imbalance*SI*BM_dw_price_cleared_SI) + np.asarray(neg_imbalance*SI*BM_up_price_cleared_SI)
    
    # imbalance fee

    im_power_cost_DK1 = (abs(power_imbalance*SI)*parameter_dict['imbalance_fee'])

    # Balancing market revenue    
    BM_revenue = reg_revenue + im_revenue - im_power_cost_DK1 
    return SM_revenue, reg_revenue, im_revenue, BM_revenue, im_power_cost_DK1

def Revenue_calculation(parameter_dict,
                        P_HPP_SM_t_opt,
                        P_HPP_RT_ts,
                        P_HPP_RT_refs,
                        SM_price_cleared,
                        BM_dw_price_cleared,
                        BM_up_price_cleared,
                        P_HPP_UP_bid_ts,
                        P_HPP_DW_bid_ts,
                        s_UP_t,
                        s_DW_t,
                        BI=1,
                        ):
    SM_revenue, reg_revenue, im_revenue, BM_revenue, im_power_cost_DK1 = _revenue_calculation(parameter_dict,
                         P_HPP_SM_t_opt,
                         P_HPP_RT_ts,
                         P_HPP_RT_refs,
                         SM_price_cleared,
                         BM_dw_price_cleared,
                         BM_up_price_cleared,
                         P_HPP_UP_bid_ts,
                         P_HPP_DW_bid_ts,
                         s_UP_t,
                         s_DW_t,
                         BI,
                         )    
   
    return SM_revenue.sum(), reg_revenue.sum(), im_revenue.sum(), BM_revenue.sum(), im_power_cost_DK1.sum()


def get_var_value_from_sol(x, sol):
    
    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient='index')
    
    return y

 
        
def RTSim(dt, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_t0, start, P_activated_UP_t, P_activated_DW_t, verbose=False):  
    # RES_error = Wind_measurement[start] + Solar_measurement[start] - RT_wind_forecast[start] - RT_solar_forecast[start] 


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
    sol = RTSim_mdl.solve()
    if verbose:
        RTSim_mdl.print_information()
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


        # z_t_opt = sol.get_value(z_t)

    else:
        print("RTOpt has no solution")
        #print(SMOpt_mdl.export_to_string())
    return E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt



def run(parameter_dict, simulation_dict, EMS, EMStype, BM_model=False, RD_model=False):
    
    DI = parameter_dict["dispatch_interval"]
    DI_num = int(1/DI)    
    T = int(1/DI*24)
        
    
    SI = parameter_dict["settlement_interval"]
    SI_num = int(1/SI)
    T_SI = int(24/SI)
    SIDI_num = int(SI/DI)
    
  
    BI = 1
    # BI_num = int(1/BI)
    # T_BI = int(24/BI)
    
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
    # PUPMax = parameter_dict["max_up_bid"] 
    # PDWMax = parameter_dict["max_dw_bid"] 
    # PUPMin = parameter_dict["min_up_bid"] 
    # PDWMin = parameter_dict["min_dw_bid"] 
    
    day_num = 1
    Ini_nld = parameter_dict["battery_initial_degradation"]
    pre_nld = Ini_nld
    SoC0 = SoCini
    ld1 = 0
    nld1 = Ini_nld
    ad = 1e-7   # slope   
    capital_cost = parameter_dict["battery_capital_cost"] # €/MWh 
    replace_percent = 0.2     
    total_cycles = 3500
                             
    PreUp = PreDw = 0
    P_grid_limit = parameter_dict["hpp_grid_connection"]

    mu = parameter_dict["battery_marginal_degradation_cost"]
    
    deg_indicator = parameter_dict["degradation_in_optimization"]
    
    
    
    
    P_HPP_UP_t0 = 0
    P_HPP_DW_t0 = 0
    
    C_dev = parameter_dict["imbalance_fee"]
        
    
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
    #reg = pd.DataFrame(list(), columns=['bid_up','bid_dw','w_up','w_dw','b_up','b_dw'])
    shc = pd.DataFrame(list(), columns=['SM','dis_SM','cha_SM','w_SM','RT','Ref','dis_RT','cha_RT'])
    slo = pd.DataFrame([ad], columns=['slope'])
    soc = pd.DataFrame(list(), columns=['SoC'])
    #bounds = pd.DataFrame(list(), columns=['UB','LB'])
    #worst_reg = pd.DataFrame(list(), columns=['up','down'])
    #worst_wind = pd.DataFrame(list(), columns=['wind'])
    #times = pd.DataFrame(list(), columns=['time-1','time12'])
    
    sig.to_csv(out_dir+'act_signal.csv',index=False)
    cur.to_csv(out_dir+'curtailment.csv',index=False)
    de.to_csv(out_dir+'Degradation.csv',index=False)
    ei.to_csv(out_dir+'energy_imbalance.csv',index=False)
    #reg.to_csv(out_dir+'reg_bids.csv',index=False)
    re.to_csv(out_dir+'revenue.csv',index=False)
    shc.to_csv(out_dir+'schedule.csv',index=False)
    slo.to_csv(out_dir+'slope.csv',index=False)
    soc.to_csv(out_dir+'SoC.csv',index=False)
    #bounds.to_csv(out_dir+'bounds.csv',index=False)
    #worst_reg.to_csv(out_dir+'worst_reg.csv',index=False)
    #worst_wind.to_csv(out_dir+'worst_wind.csv',index=False)
    #times.to_csv(out_dir+'time.csv',index=False)
    pbar = tqdm(total = simulation_dict["number_of_run_day"]+1)
    while day_num:
        pbar.update(1)
        Emax = EBESS*(1-pre_nld)
        
        if EMStype == "DEMS":
           DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = EMS.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
        elif EMStype == "SEMS":
           DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index, HA_wind_forecast_scenario, probability_wind = EMS.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
    
           if simulation_dict['price_scenario_fn'] == None:      
              probability_price, SP_scenario, RP_scenario = scenario_generation(PsMax, PwMax, T, DI_num, simulation_dict)

           
           probability = np.zeros(len(probability_wind)*len(probability_price))
           
           # Produce the final probability. Here assume the price uncertainty is independent with wind uncertainty
           for i in range(len(probability_wind)):
               for j in range(len(probability_price)):
                   probability[i*len(probability_price)+j] = probability_price[j] * probability_wind[i]
           HA_wind_forecast_scenario = np.repeat(HA_wind_forecast_scenario, len(probability_price), axis=0) 
           SP_scenario = np.matlib.repmat(SP_scenario,len(probability_wind),1)
           RP_scenario = np.matlib.repmat(RP_scenario,len(probability_wind),1)
        else:
           DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = EMS.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
 
      
        
        SM_price_forecast = SM_price_forecast.squeeze().repeat(DI_num)
        SM_price_forecast.index = range(T + exten_num)
        
        
        
        
        
    # Call EMS Model
        # Run SMOpt
        if EMStype == "DEMS":
             E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_w_SM_cur_t_opt, P_s_SM_cur_t_opt, P_w_SM_t_opt, P_s_SM_t_opt = EMS.SMOpt(DI, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                         DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator)

        elif EMStype == "SEMS":    
             E_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, P_w_SM_t_opt = EMS.SMOpt(DI, SI, BI, T, EBESS, PbMax, PwMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                            HA_wind_forecast_scenario, SP_scenario, RP_scenario, probability, SoC0, exten_num, len(probability), C_dev)
        else:
             E_HPP_SM_t_opt, P_HPP_SM_t_opt, P_HPP_SM_k_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, SoC_SM_t_opt, P_W_SM_cur_t_opt, P_S_SM_cur_t_opt, P_W_SM_t_opt, P_S_SM_t_opt = EMS.SMOpt(DI, T, PbMax, EBESS, SoCmin, SoCmax, eta_dis, eta_cha, eta_leak, Emax, PreUp, PreDw, P_grid_limit, mu, ad,
                         DA_wind_forecast, DA_solar_forecast, SM_price_forecast, SoC0, deg_indicator)

        #P_HPP_SM_t_opt.index = time_index[:T]
        P_HPP_SM_t_opt.index = range(T)
        #write_results(P_HPP_SM_t_opt, 'results_run.xlsx', (day_num-1)*T, 0, 'power schedule')

        P_HPP_RT_ref = P_HPP_SM_t_opt.iloc[0,0]
                      
        P_HPP_RT_ts = []
        P_HPP_RT_refs = []
        RES_RT_cur_ts = []
        residual_imbalance = []
        SoC_ts = []
        P_dis_RT_ts = []
        P_cha_RT_ts = []
        
        
        s_UP_t = np.zeros(T)
        s_DW_t = np.zeros(T)
        P_HPP_UP_bid_ts = pd.DataFrame(np.zeros(T))
        P_HPP_DW_bid_ts = pd.DataFrame(np.zeros(T))

        
        if BM_model == True and RD_model == True:    
           for i in range(0,24):
                BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
                BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
                BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
                BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        
                BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
                BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
                BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
                BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num)) 

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
           
                # Run BMOpt
                E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = EMS.BMOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    HA_wind_forecast1, HA_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator)
                
                if i < 24 - 1:
                    P_HPP_UP_t1 = P_HPP_UP_k_opt.loc[i+1].iloc[0]
                    P_HPP_DW_t1 = P_HPP_DW_k_opt.loc[i+1].iloc[0]
                    P_HPP_UP_bid_ts.iloc[(i+1)*DI_num:(i+2)*DI_num,0] = P_HPP_UP_t1
                    P_HPP_DW_bid_ts.iloc[(i+1)*DI_num:(i+2)*DI_num,0] = P_HPP_UP_t1
                else:
                    P_HPP_UP_t1 = 0
                    P_HPP_DW_t1 = 0


                # Run RTSim
           
                E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, i * DI_num, P_HPP_UP_t0, P_HPP_DW_t0) 
           
                SoC_ts.append({'SoC': SoC0})
                P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt})
                
                P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[1, 0] 
           
          
                exist_imbalance = (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num] - P_HPP_DW_t0 * s_DW_t[i*DI_num]) - P_HPP_SM_t_opt.iloc[i * DI_num,0]) * DI

                if DI == 1/4:
                    residual_imbalance.append({'energy_imbalance': exist_imbalance})
                    exist_imbalance = 0

                SoC0 = SoC_RT_t_opt.iloc[1,0]

                for j in range(1, DI_num):
                    
                    RT_interval = i * DI_num + j
                    
                 
                    RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):]-Wind_measurement.values[(i+2)*int(1/DI):])])
                    RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement[(i+2)*int(1/DI):])])
                                        
                    # Run RDOpt       
                    E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = EMS.RDOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                            RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, RT_interval, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, P_HPP_UP_t1, P_HPP_DW_t1, SoC0, exist_imbalance, exten_num, deg_indicator)

   


                    # Run RTSim
                    E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                                Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval, P_HPP_UP_t0, P_HPP_DW_t0)
                    SoC_ts.append({'SoC': SoC0})
                    P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                    P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                    RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                    P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt})
                    


                    if RT_interval < T - 1:
                        P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1,0]  
      

                    
                
                        
                    if RT_interval%SIDI_num == SIDI_num-1:
                        exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                        residual_imbalance.append({'energy_imbalance': exist_imbalance})
                        exist_imbalance = 0
                    else:
                        exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI

                    SoC0 = SoC_RT_t_opt.iloc[1,0]
                    
                P_HPP_UP_t0 = P_HPP_UP_t1
                P_HPP_DW_t0 = P_HPP_DW_t1

        elif BM_model == True and RD_model == False:
             
             for i in range(0,24):
                BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
                BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
                BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
                BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        
                BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
                BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
                BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
                BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num)) 

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
           
                # Run BMOpt
                E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = EMS.BMOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                    HA_wind_forecast1, HA_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i, s_UP_t, s_DW_t, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator)
                
                if i < 24 - 1:
                    P_HPP_UP_t1 = P_HPP_UP_k_opt.loc[i+1].iloc[0]
                    P_HPP_DW_t1 = P_HPP_DW_k_opt.loc[i+1].iloc[0]
                    P_HPP_UP_bid_ts.iloc[(i+1)*DI_num:(i+2)*DI_num,0] = P_HPP_UP_t1
                    P_HPP_DW_bid_ts.iloc[(i+1)*DI_num:(i+2)*DI_num,0] = P_HPP_UP_t1
                else:
                    P_HPP_UP_t1 = 0
                    P_HPP_DW_t1 = 0

                # Run RTSim
           
                E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, i * DI_num, P_HPP_UP_t0, P_HPP_DW_t0) 
           
                SoC_ts.append({'SoC': SoC0})
                P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt})
                
                P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[1, 0]
           
          
                exist_imbalance = (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num] - P_HPP_DW_t0 * s_DW_t[i*DI_num]) - P_HPP_SM_t_opt.iloc[i * DI_num,0]) * DI

                if DI == 1/4:
                    residual_imbalance.append({'energy_imbalance': exist_imbalance})
                    exist_imbalance = 0

                SoC0 = SoC_RT_t_opt.iloc[1,0]

                for j in range(1, DI_num):
                    BM_dw_price = BM_dw_price_forecast
                    BM_up_price = BM_up_price_forecast
                    BM_dw_price[i] = BM_dw_price_cleared[i]
                    BM_up_price[i] = BM_up_price_cleared[i]

                    RT_interval = i * DI_num + j
                    
                 
                    RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):]-Wind_measurement.values[(i+2)*int(1/DI):])])
                    RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement[(i+2)*int(1/DI):])])
                                        
                    

                    # Run RTSim
                    E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                                Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval, P_HPP_UP_t0, P_HPP_DW_t0)
                    SoC_ts.append({'SoC': SoC0})
                    P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                    P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                    RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                    P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt})
                    


                    if RT_interval < T - 1:
                        P_HPP_RT_ref = P_HPP_HA_t_opt.loc[RT_interval+1].iloc[0]
      

                    if RT_interval%SIDI_num == SIDI_num-1:
                        exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt - (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI     
                        residual_imbalance.append({'energy_imbalance': exist_imbalance})    
                        exist_imbalance = 0          
                    else:    
                        exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- (P_HPP_UP_t0 * s_UP_t[i*DI_num + j] - P_HPP_DW_t0 * s_DW_t[i*DI_num + j]) - P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI

                    SoC0 = SoC_RT_t_opt.iloc[1,0]
                    
                P_HPP_UP_t0 = P_HPP_UP_t1
                P_HPP_DW_t0 = P_HPP_DW_t1
        

        elif BM_model == False and RD_model == True:

            for i in range(0,24):   
                RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8*(DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
                RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8*(DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])

                BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
                BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
                BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
                BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
        
                BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
                BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
                BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
                BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))


                exist_imbalance = 0

                E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = EMS.RBOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                     RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, i*DI_num, s_UP_t, s_DW_t, 0, 0, 0, 0, SoC0, exist_imbalance, exten_num, deg_indicator)
          
          
          
          # Run RTSim
           
                E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                            Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, i * DI_num, P_HPP_UP_t0, P_HPP_DW_t0) 
                
                SoC_ts.append({'SoC': SoC0})
                P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt})
                
                P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1, 0] 

                
                exist_imbalance = (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[i * DI_num,0]) * DI

                if DI == 1/4:
                    residual_imbalance.append({'energy_imbalance': exist_imbalance}) 
                    exist_imbalance = 0               


                SoC0 = SoC_RT_t_opt.iloc[1,0]

                for j in range(1, DI_num):              

                    RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):] - Wind_measurement.values[(i+2)*int(1/DI):])])
                    RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement.values[(i+2)*int(1/DI):])])
                    
                    
                    RT_interval = i * DI_num + j
                    # Run RDOpt
        
                    E_HPP_RD_t_opt, P_HPP_RD_t_opt, P_dis_RD_t_opt, P_cha_RD_t_opt, SoC_RD_t_opt, P_W_RD_cur_t_opt, P_S_RD_cur_t_opt, P_W_RD_t_opt, P_S_RD_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt = EMS.RBOpt(DI, SI, BI, T, EBESS, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad,
                            RD_wind_forecast1, RD_solar_forecast1, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_forecast_settle, BM_up_price_forecast_settle, reg_up_sign_forecast, reg_dw_sign_forecast, P_HPP_SM_t_opt, RT_interval, s_UP_t, s_DW_t, 0, 0, 0, 0, SoC0, exist_imbalance, exten_num, deg_indicator)
                        #P_HPP_RD_t_opt = P_HPP_HA_t_opt
                        #P_dis_RD_t_opt = P_dis_HA_t_opt
                        #P_cha_RD_t_opt = P_cha_HA_t_opt

                    # Run RTSim
                    E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                                Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval, P_HPP_UP_t0, P_HPP_DW_t0)
                    SoC_ts.append({'SoC': SoC0})
                    P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                    P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                    RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                    P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt})
                    
                    if RT_interval < T - 1:
                        P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1,0] 
                    
                        
                    if RT_interval%SIDI_num == SIDI_num-1:
                        exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                        residual_imbalance.append({'energy_imbalance': exist_imbalance})
                        exist_imbalance = 0
                    else:
                        exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI

                    SoC0 = SoC_RT_t_opt.iloc[1,0]
        else:
            for i in range(0,24):
                exist_imbalance = 0
                for j in range(0, DI_num):    
                    RT_interval = i * DI_num + j
                    # run RTSim

                    P_HPP_RT_ref = P_HPP_SM_t_opt.iloc[RT_interval,0] 
        
                        
                    E_HPP_RT_t_opt, P_HPP_RT_t_opt, P_dis_RT_t_opt, P_cha_RT_t_opt, SoC_RT_t_opt, RES_RT_cur_t_opt, P_W_RT_t_opt, P_S_RT_t_opt = RTSim(DI, PbMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak,
                                    Wind_measurement, Solar_measurement, RT_wind_forecast, RT_solar_forecast, SoC0, P_HPP_RT_ref, RT_interval, P_HPP_UP_t0, P_HPP_DW_t0) 
                    
                    SoC_ts.append({'SoC': SoC0})
                    P_HPP_RT_ts.append({'RT': P_HPP_RT_t_opt}) 
                    P_HPP_RT_refs.append({'Ref': P_HPP_RT_ref}) 
                    RES_RT_cur_ts.append({'RES_cur': RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({'dis_RT': P_dis_RT_t_opt})
                    P_cha_RT_ts.append({'cha_RT': P_cha_RT_t_opt}) 
                
                        
        

                    exist_imbalance = exist_imbalance + (P_HPP_RT_t_opt- P_HPP_SM_t_opt.iloc[RT_interval, 0]) * DI
                    residual_imbalance.append({'energy_imbalance': exist_imbalance})
                    SoC0 = SoC_RT_t_opt.iloc[1,0]

        residual_imbalance = pd.DataFrame(residual_imbalance)
        P_HPP_RT_ts = pd.DataFrame(P_HPP_RT_ts)
        P_HPP_RT_refs = pd.DataFrame(P_HPP_RT_refs)
        P_dis_RT_ts = pd.DataFrame(P_dis_RT_ts)
        P_cha_RT_ts = pd.DataFrame(P_cha_RT_ts)
        RES_RT_cur_ts = pd.DataFrame(RES_RT_cur_ts)

        
        
        (SM_revenue,
        reg_revenue,
        im_revenue, 
        BM_revenue,
        im_special_revenue_DK1,
        ) = Revenue_calculation(parameter_dict,
                                P_HPP_SM_t_opt,
                                P_HPP_RT_ts,
                                P_HPP_RT_refs,
                                SM_price_cleared,
                                BM_dw_price_cleared,
                                BM_up_price_cleared,
                                P_HPP_UP_bid_ts,
                                P_HPP_DW_bid_ts,
                                s_UP_t,
                                s_DW_t,
                                )
        

        
        #SoC_all = pd.read_excel('results_run.xlsx', sheet_name = 'SoC', nrows=(day_num-1)*T, engine='openpyxl')
        SoC_all = pd.read_csv(out_dir+'SoC.csv')
        SoC_ts = pd.DataFrame(SoC_ts)
        if SoC_all.empty:
           SoC_all = SoC_ts
        else:
           SoC_all = pd.concat([SoC_all, SoC_ts]) 
        
        SoC_all = SoC_all.values.tolist()
        
        SoC_for_rainflow = SoC_all
        SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num*T))]
    
    
        ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles = DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
        
        Deg_cost = (nld - pre_nld)/replace_percent * EBESS * capital_cost
        
    
        if day_num==1:
           Deg_cost_by_cycle = cycles.iloc[0,0]/total_cycles * EBESS * capital_cost  
        else:                
           Deg = pd.read_csv(out_dir+'Degradation.csv') 
           cycle_of_day = Deg.iloc[-1,2] - Deg.iloc[-2,2] 
           Deg_cost_by_cycle = cycle_of_day/total_cycles * EBESS * capital_cost        
    
        P_HPP_RT_ts.index = range(T)
        P_HPP_RT_refs.index = range(T)
        P_dis_RT_ts.index = range(T)
        P_cha_RT_ts.index = range(T)   
        
        '''
        P_HPP_SM_t_opt, Spot market schedule, power [MW]
        P_dis_SM_t_opt, Battery discharge schedule in spot market, power [MW]
        P_cha_SM_t_opt, 
        P_w_SM_t_opt, 
        P_HPP_RT_ts, Final HPP output, power [MW] 
        P_HPP_RT_refs, 
        P_dis_RT_ts, Final discharge operation of battery, power [MW]
        P_cha_RT_ts: Final charge operation of battery, power [MW]
        '''
        output_schedule = pd.concat([P_HPP_SM_t_opt, P_dis_SM_t_opt, P_cha_SM_t_opt, P_w_SM_t_opt, P_HPP_RT_ts, P_HPP_RT_refs, P_dis_RT_ts, P_cha_RT_ts], axis=1)
        output_revenue = pd.DataFrame([SM_revenue, reg_revenue, im_revenue, im_special_revenue_DK1, Deg_cost, Deg_cost_by_cycle]).T
        output_revenue.columns=['SM_revenue','reg_revenue','im_revenue','im_special_revenue_DK1', 'Deg_cost','Deg_cost_by_cycle']
        output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts], axis=1)
        output_act_signal = pd.concat([pd.DataFrame(s_UP_t, columns=['signal_up']), pd.DataFrame(s_DW_t, columns=['signal_down'])], axis=1)
        #output_time = pd.concat([pd.DataFrame([run_time], columns=['time-1']), pd.DataFrame([run_time2], columns=['time0'])], axis=1)
        #output_time = pd.concat([pd.DataFrame([run_time], columns=['time-1']), pd.DataFrame([run_time2], columns=['time0'])], axis=1)
        #output_bounds = pd.concat([pd.DataFrame(UBs, columns=['UB']), pd.DataFrame(LBs, columns=['LB'])], axis=1)
        if day_num == 1:
            output_deg = pd.concat([pd.DataFrame([Ini_nld, nld], columns=['nld']), pd.DataFrame([0, ld], columns=['ld']), pd.DataFrame([0, cycles.iloc[0,0]], columns=['cycles'])], axis=1)
        else:
            output_deg = pd.concat([pd.DataFrame([nld], columns=['nld']), pd.DataFrame([ld], columns=['ld']), cycles], axis=1)

        output_schedule.to_csv(out_dir+'schedule.csv', mode='a', index=False, header=False)    
        output_bids.to_csv(out_dir+'reg_bids.csv', mode='a', index=False, header=False)                    
        output_act_signal.to_csv(out_dir+'act_signal.csv', mode='a', index=False, header=False)  
        output_deg.to_csv(out_dir+'Degradation.csv', mode='a', index=False, header=False)  
        SoC_ts.to_csv(out_dir+'SoC.csv', mode='a', index=False, header=False)  
        residual_imbalance.to_csv(out_dir+'energy_imbalance.csv', mode='a', index=False, header=False)  
        RES_RT_cur_ts.to_csv(out_dir+'curtailment.csv', mode='a', index=False, header=False)  
        output_revenue.to_csv(out_dir+'revenue.csv', mode='a', index=False, header=False)  


        
        
        Pdis_all = pd.read_csv(out_dir+'schedule.csv', usecols=[3])
        Pcha_all = pd.read_csv(out_dir+'schedule.csv', usecols=[4])
        nld_all = pd.read_csv(out_dir+'Degradation.csv', usecols=[0])
        ad_all =pd.read_csv(out_dir+'slope.csv', usecols=[0])
        ad = DegCal.slope_update(Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all)
        
        
        pd.DataFrame([ad], columns=['slope']).to_csv(out_dir+'slope.csv', mode='a', index=False, header=False) 
        if nld>0.2:
            break
        
        pre_nld = nld
        day_num = day_num + 1
        if day_num > simulation_dict["number_of_run_day"]:
            print(P_grid_limit)
            break
    pbar.close()
    # return P_HPP_RT_ts, P_HPP_SM_k_opt, P_HPP_RT_refs, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, RES_RT_cur_ts, P_cha_RT_ts, P_dis_RT_ts, SoC_ts
    return (P_HPP_SM_t_opt.values.ravel(),
            SM_price_cleared.values,
            BM_dw_price_cleared.values,
            BM_up_price_cleared.values,
            P_HPP_RT_ts.values.ravel(),
            P_HPP_RT_refs.values.ravel(),
            P_HPP_UP_bid_ts.values.ravel(),
            P_HPP_DW_bid_ts.values.ravel(),
            s_UP_t,s_DW_t,
            residual_imbalance.values.ravel(), 
            RES_RT_cur_ts.values.ravel(), 
            P_dis_RT_ts.values.ravel(), 
            P_cha_RT_ts.values.ravel(), 
            pd.DataFrame(SoC_ts).values.ravel(),
            )


