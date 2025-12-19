# -*- coding: utf-8 -*-
"""

Created on Mon Mar 15 11:42:56 2021

@author: ruzhu
"""

from datetime import datetime

import numpy as np
import pandas as pd
from docplex.mp.model import Model
from numpy import matlib as mb

from hydesign.HiFiEMS.utils import DataReaderBase


def get_var_value_from_sol(x, sol):

    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient="index")

    return y


def SMOpt(parameter_dict, simulation_dict, dynamic_inputs, verbose=False):
    dt = parameter_dict["dispatch_interval"]
    dt_num = int(1 / dt)
    T = int(1 / dt * 24)

    day_num = dynamic_inputs["day_num"]
    SoC0 = dynamic_inputs["SoC0"]
    Emax = dynamic_inputs["Emax"]
    ad = dynamic_inputs["ad"]

    PwMax = parameter_dict["wind_capacity"]
    PsMax = parameter_dict["solar_capacity"]
    EBESS = parameter_dict["battery_energy_capacity"]
    PbMax = parameter_dict["battery_power_capacity"]
    SoCmin = parameter_dict["battery_minimum_SoC"]
    SoCmax = parameter_dict["battery_maximum_SoC"]
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"]

    P_grid_limit = parameter_dict["hpp_grid_connection"]
    mu = parameter_dict["battery_marginal_degradation_cost"]
    deg_indicator = parameter_dict["degradation_in_optimization"]

    # Optimization modelling by CPLEX
    setT = [i for i in range(T)]
    set_SoCT = [i for i in range(T + 1)]
    setK = [i for i in range(24)]
    dt_num = int(1 / dt)

    # Read data
    ReadData = DataReaderBase(
        day_num=day_num,
        DI_num=dt_num,
        T=T,
        PsMax=PsMax,
        PwMax=PwMax,
        simulation_dict=simulation_dict,
    )
    Inputs = ReadData.execute()

    DA_wind_forecast = Inputs["DA_wind_forecast"]
    DA_solar_forecast = Inputs["DA_solar_forecast"]

    SM_price_forecast = Inputs["SM_price_forecast"]
    SM_price_forecast = SM_price_forecast.squeeze().repeat(dt_num)
    SM_price_forecast.index = range(T)

    SMOpt_mdl = Model()
    # Define variables
    P_HPP_SM_t = SMOpt_mdl.continuous_var_dict(setT, name="SM bidding 5min")
    P_HPP_SM_k = SMOpt_mdl.continuous_var_dict(setK, name="SM bidding H")
    P_W_SM_t = {
        t: SMOpt_mdl.continuous_var(
            lb=0, ub=DA_wind_forecast[t], name="SM Wind bidding {}".format(t)
        )
        for t in setT
    }
    P_S_SM_t = {
        t: SMOpt_mdl.continuous_var(
            lb=0, ub=DA_solar_forecast[t], name="DA Solar bidding {}".format(t)
        )
        for t in setT
    }
    P_dis_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="SM discharge"
    )
    P_cha_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name="SM charge")
    P_b_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=-PbMax, ub=PbMax, name="SM Battery schedule"
    )  # (must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_SM_t = SMOpt_mdl.continuous_var_dict(
        set_SoCT, lb=SoCmin, ub=SoCmax, name="SM SoC"
    )
    z_t = SMOpt_mdl.binary_var_dict(setT, name="Cha or Discha")

    # Define constraints
    for t in setT:
        SMOpt_mdl.add_constraint(
            P_HPP_SM_t[t] == P_W_SM_t[t] + P_S_SM_t[t] + P_b_SM_t[t]
        )
        SMOpt_mdl.add_constraint(P_b_SM_t[t] == P_dis_SM_t[t] - P_cha_SM_t[t])
        SMOpt_mdl.add_constraint(P_dis_SM_t[t] <= (PbMax) * z_t[t])
        SMOpt_mdl.add_constraint(P_cha_SM_t[t] <= (PbMax) * (1 - z_t[t]))
        SMOpt_mdl.add_constraint(
            SoC_SM_t[t + 1]
            == SoC_SM_t[t] * (1 - eta_leak)
            - 1 / Emax * P_dis_SM_t[t] / eta_dis * dt
            + 1 / Emax * P_cha_SM_t[t] * eta_cha * dt
        )
        SMOpt_mdl.add_constraint(SoC_SM_t[t + 1] <= SoCmax)
        SMOpt_mdl.add_constraint(SoC_SM_t[t + 1] >= SoCmin)
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] <= P_grid_limit)
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] >= -P_grid_limit)
    for k in setK:
        for t in setT:
            if t // dt_num == k:
                SMOpt_mdl.add_constraint(P_HPP_SM_k[k] == P_HPP_SM_t[t])

    SMOpt_mdl.add_constraint(SoC_SM_t[0] == SoC0)

    # Define objective function
    Revenue = SMOpt_mdl.sum(SM_price_forecast[t] * P_HPP_SM_t[t] * dt for t in setT)
    if deg_indicator == 1:
        Deg_cost = (
            mu
            * EBESS
            * ad
            * SMOpt_mdl.sum((P_dis_SM_t[t] + P_cha_SM_t[t]) * dt for t in setT)
        )
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
        P_HPP_SM_t_opt.columns = ["SM"]

        P_W_SM_t_opt = get_var_value_from_sol(P_W_SM_t, sol)
        P_S_SM_t_opt = get_var_value_from_sol(P_S_SM_t, sol)
        P_dis_SM_t_opt = get_var_value_from_sol(P_dis_SM_t, sol)
        P_cha_SM_t_opt = get_var_value_from_sol(P_cha_SM_t, sol)
        SoC_SM_t_opt = get_var_value_from_sol(SoC_SM_t, sol)

        E_HPP_SM_t_opt = P_HPP_SM_t_opt * dt

        P_W_SM_cur_t_opt = (
            np.array(DA_wind_forecast[:T].T) - np.array(P_W_SM_t_opt).flatten()
        )
        P_W_SM_cur_t_opt = pd.DataFrame(P_W_SM_cur_t_opt)
        P_S_SM_cur_t_opt = (
            np.array(DA_solar_forecast[:T].T) - np.array(P_S_SM_t_opt).flatten()
        )
        P_S_SM_cur_t_opt = pd.DataFrame(P_S_SM_cur_t_opt)

        z_t_opt = get_var_value_from_sol(z_t, sol)
        if verbose:
            print(P_HPP_SM_t_opt)
            print(P_dis_SM_t_opt)
            print(P_cha_SM_t_opt)
            print(SoC_SM_t_opt)
            print(z_t_opt)

    else:
        print("SMOpt has no solution")
    return (
        E_HPP_SM_t_opt,
        P_HPP_SM_t_opt,
        P_HPP_SM_k_opt,
        P_dis_SM_t_opt,
        P_cha_SM_t_opt,
        SoC_SM_t_opt,
        P_W_SM_cur_t_opt,
        P_S_SM_cur_t_opt,
        P_W_SM_t_opt,
        P_S_SM_t_opt,
    )


def BMOpt(parameter_dict, simulation_dict, dynamic_inputs, verbose=False):

    day_num = dynamic_inputs["day_num"]
    Emax = dynamic_inputs["Emax"]
    ad = dynamic_inputs["ad"]
    P_HPP_SM_t_opt = dynamic_inputs["P_HPP_SM_t_opt"]
    start = dynamic_inputs["Current_hour"]
    s_UP_t = dynamic_inputs["s_UP_t"]
    s_DW_t = dynamic_inputs["s_DW_t"]
    P_HPP_UP_t0 = dynamic_inputs["P_HPP_UP_t0"]
    P_HPP_DW_t0 = dynamic_inputs["P_HPP_DW_t0"]
    SoC0 = dynamic_inputs["SoC0"]

    dt = parameter_dict["dispatch_interval"]
    dt_num = int(1 / dt)
    T = int(1 / dt * 24)

    ds = parameter_dict["settlement_interval"]
    ds_num = int(1 / ds)
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    dk = parameter_dict["offer_interval"]
    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    setT = [i for i in range(start * dt_num, T)]
    setT1 = [i for i in range((start + 1) * dt_num, T)]
    setK = [i for i in range(start * dk_num, T_dk)]
    setK1 = [i for i in range((start + 1) * dk_num, T_dk)]
    setS = [i for i in range(start * ds_num, T_ds)]
    set_SoCT = [i for i in range(start * dt_num, T + 1)]

    PwMax = parameter_dict["wind_capacity"]
    PsMax = parameter_dict["solar_capacity"]
    EBESS = parameter_dict["battery_energy_capacity"]
    PbMax = parameter_dict["battery_power_capacity"]
    SoCmin = parameter_dict["battery_minimum_SoC"]
    SoCmax = parameter_dict["battery_maximum_SoC"]
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"]

    P_grid_limit = parameter_dict["hpp_grid_connection"]
    mu = parameter_dict["battery_marginal_degradation_cost"]
    deg_indicator = parameter_dict["degradation_in_optimization"]

    # Read data
    ReadData = DataReaderBase(
        day_num=day_num,
        DI_num=dt_num,
        T=T,
        PsMax=PsMax,
        PwMax=PwMax,
        simulation_dict=simulation_dict,
    )
    Inputs = ReadData.execute()

    DA_wind_forecast = Inputs["DA_wind_forecast"]
    DA_solar_forecast = Inputs["DA_solar_forecast"]
    HA_wind_forecast = Inputs["HA_wind_forecast"]
    HA_solar_forecast = Inputs["HA_solar_forecast"]
    RT_wind_forecast = Inputs["RT_wind_forecast"]
    RT_solar_forecast = Inputs["RT_solar_forecast"]
    Wind_measurement = Inputs["Wind_measurement"]
    Solar_measurement = Inputs["Solar_measurement"]

    HA_wind_forecast = pd.Series(
        np.r_[
            RT_wind_forecast.values[start * dt_num : start * dt_num + 2],
            HA_wind_forecast.values[start * dt_num + 2 : (start + 2) * dt_num],
            Wind_measurement.values[(start + 2) * dt_num :]
            + 0.8
            * (
                DA_wind_forecast.values[(start + 2) * dt_num :]
                - Wind_measurement.values[(start + 2) * dt_num :]
            ),
        ]
    )
    HA_solar_forecast = pd.Series(
        np.r_[
            RT_solar_forecast.values[start * dt_num : start * dt_num + 2],
            HA_solar_forecast.values[start * dt_num + 2 : (start + 2) * dt_num],
            Solar_measurement.values[(start + 2) * dt_num :]
            + 0.8
            * (
                DA_solar_forecast.values[(start + 2) * dt_num :]
                - Solar_measurement.values[(start + 2) * dt_num :]
            ),
        ]
    )

    BM_dw_price_forecast = Inputs["BM_dw_price_forecast"]
    BM_up_price_forecast = Inputs["BM_up_price_forecast"]

    reg_up_sign_forecast = Inputs["reg_up_sign_forecast"]
    reg_dw_sign_forecast = Inputs["reg_dw_sign_forecast"]

    BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(ds_num)
    BM_up_price_forecast_settle.index = range(T_ds)

    BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(ds_num)
    BM_dw_price_forecast_settle.index = range(T_ds)

    reg_up_sign_forecast1 = reg_up_sign_forecast.squeeze().repeat(dt_num)
    reg_dw_sign_forecast1 = reg_dw_sign_forecast.squeeze().repeat(dt_num)

    reg_up_sign_forecast1.index = range(T)
    reg_dw_sign_forecast1.index = range(T)

    # Optimization modelling by CPLEX
    BMOpt_mdl = Model()

    # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_UP_t = BMOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=P_grid_limit, name="BM UP bidding 5min"
    )
    P_HPP_DW_t = BMOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=P_grid_limit, name="BM DW bidding 5min"
    )
    P_HPP_UP_k = BMOpt_mdl.continuous_var_dict(
        setK1, lb=0, ub=P_grid_limit, name="BM UP bidding H"
    )
    P_HPP_DW_k = BMOpt_mdl.continuous_var_dict(
        setK1, lb=0, ub=P_grid_limit, name="BM DW bidding H"
    )

    P_HPP_HA_t = BMOpt_mdl.continuous_var_dict(
        setT, name="HA schedule with balancing bidding"
    )
    P_W_HA_t = {
        t: BMOpt_mdl.continuous_var(
            lb=0,
            ub=HA_wind_forecast[t - start * dt_num],
            name="HA Wind schedule {}".format(t),
        )
        for t in setT
    }
    P_S_HA_t = {
        t: BMOpt_mdl.continuous_var(
            lb=0,
            ub=HA_solar_forecast[t - start * dt_num],
            name="HA Solar schedule {}".format(t),
        )
        for t in setT
    }
    P_dis_HA_t = BMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="HA discharge"
    )
    P_cha_HA_t = BMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name="HA charge")
    P_b_HA_t = BMOpt_mdl.continuous_var_dict(
        setT, lb=-PbMax, ub=PbMax, name="HA Battery schedule"
    )  # (must define lb and ub, otherwise may cause unknown issues on cplex)

    SoC_HA_t = BMOpt_mdl.continuous_var_dict(
        set_SoCT, lb=SoCmin, ub=SoCmax, name="HA SoC"
    )
    z_t = BMOpt_mdl.binary_var_dict(setT, name="Cha or Discha")

    delta_P_HPP_s = BMOpt_mdl.continuous_var_dict(
        setS, lb=-P_grid_limit, ub=P_grid_limit, name="HA imbalance"
    )
    delta_P_HPP_UP_s = BMOpt_mdl.continuous_var_dict(
        setS, lb=0, ub=P_grid_limit, name="HA up imbalance"
    )
    delta_P_HPP_DW_s = BMOpt_mdl.continuous_var_dict(
        setS, lb=0, ub=P_grid_limit, name="HA dw imbalance"
    )

    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  # dataframe to series
    # Define constraints
    for t in setT:

        BMOpt_mdl.add_constraint(
            P_HPP_HA_t[t] == P_W_HA_t[t] + P_S_HA_t[t] + P_b_HA_t[t]
        )
        BMOpt_mdl.add_constraint(P_b_HA_t[t] == P_dis_HA_t[t] - P_cha_HA_t[t])
        BMOpt_mdl.add_constraint(P_dis_HA_t[t] <= (PbMax) * z_t[t])
        BMOpt_mdl.add_constraint(P_cha_HA_t[t] <= (PbMax) * (1 - z_t[t]))

        BMOpt_mdl.add_constraint(
            SoC_HA_t[t + 1]
            == SoC_HA_t[t] * (1 - eta_leak)
            - 1 / Emax * (P_dis_HA_t[t]) / eta_dis * dt
            + 1 / Emax * (P_cha_HA_t[t]) * eta_cha * dt
        )

        BMOpt_mdl.add_constraint(SoC_HA_t[t] <= SoCmax)
        BMOpt_mdl.add_constraint(SoC_HA_t[t] >= SoCmin)

        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] <= P_grid_limit)
        BMOpt_mdl.add_constraint(P_HPP_HA_t[t] >= -P_grid_limit)

    for s in setS:
        if s < (start + 1) * ds_num:
            # BMOpt_mdl.add_constraint(delta_P_HPP_s[s] == BMOpt_mdl.sum(P_HPP_HA_t[s * dsdt_num + m] - (s_UP_t[s * dsdt_num + m] * P_HPP_UP_t[s * dsdt_num + m] - s_DW_t[s * dsdt_num + m] * P_HPP_DW_t[s * dsdt_num + m]) - P_HPP_SM_t_opt[s * dsdt_num + m] for m in range(0, dsdt_num))/dsdt_num)
            BMOpt_mdl.add_constraint(
                delta_P_HPP_s[s]
                == BMOpt_mdl.sum(
                    P_HPP_HA_t[s * dsdt_num + m]
                    - (
                        P_HPP_UP_t0 * s_UP_t[s * dsdt_num + m]
                        - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + m]
                    )
                    - P_HPP_SM_t_opt[s * dsdt_num + m]
                    for m in range(0, dsdt_num)
                )
                / dsdt_num
            )
        else:
            BMOpt_mdl.add_constraint(
                delta_P_HPP_s[s]
                == BMOpt_mdl.sum(
                    P_HPP_HA_t[s * dsdt_num + m]
                    - (
                        reg_up_sign_forecast1[s * dsdt_num + m]
                        * P_HPP_UP_t[s * dsdt_num + m]
                        - reg_dw_sign_forecast1[s * dsdt_num + m]
                        * P_HPP_DW_t[s * dsdt_num + m]
                    )
                    - P_HPP_SM_t_opt[s * dsdt_num + m]
                    for m in range(0, dsdt_num)
                )
                / dsdt_num
            )
        BMOpt_mdl.add_constraint(
            delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s]
        )

    for k in setK1:
        for j in range(0, dt_num):

            BMOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
            BMOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])

    BMOpt_mdl.add_constraint(SoC_HA_t[start * dt_num] == SoC0)

    Revenue = BMOpt_mdl.sum(
        BM_up_price_forecast[k] * reg_up_sign_forecast[k] * P_HPP_UP_k[k] * dk
        - BM_dw_price_forecast[k] * reg_dw_sign_forecast[k] * P_HPP_DW_k[k] * dk
        for k in setK1
    ) + BMOpt_mdl.sum(
        (BM_dw_price_forecast_settle[s] - 0.001) * delta_P_HPP_UP_s[s] * ds
        - (BM_up_price_forecast_settle[s] + 0.001) * delta_P_HPP_DW_s[s] * ds
        for s in setS
    )

    if deg_indicator == 1:
        Deg_cost = (
            mu
            * EBESS
            * ad
            * BMOpt_mdl.sum((P_dis_HA_t[t] + P_cha_HA_t[t]) * dt for t in setT)
        )
    else:
        Deg_cost = 0

    BMOpt_mdl.maximize(Revenue - Deg_cost)

    # Solve BMOpt Model
    sol = BMOpt_mdl.solve()
    if verbose:
        BMOpt_mdl.print_information()
        aa = BMOpt_mdl.get_solve_details()
        print(aa.status)
    if sol:

        P_HPP_HA_t_opt = get_var_value_from_sol(P_HPP_HA_t, sol)
        P_HPP_HA_t_opt.columns = [
            "HA"
        ]  # Keeping this because it's likely a display tweak

        P_W_HA_t_opt = get_var_value_from_sol(P_W_HA_t, sol)
        P_S_HA_t_opt = get_var_value_from_sol(P_S_HA_t, sol)
        P_dis_HA_t_opt = get_var_value_from_sol(P_dis_HA_t, sol)
        P_cha_HA_t_opt = get_var_value_from_sol(P_cha_HA_t, sol)

        SoC_HA_t_opt = get_var_value_from_sol(SoC_HA_t, sol)
        P_HPP_UP_t_opt = get_var_value_from_sol(P_HPP_UP_t, sol)
        P_HPP_DW_t_opt = get_var_value_from_sol(P_HPP_DW_t, sol)
        P_HPP_UP_k_opt = get_var_value_from_sol(P_HPP_UP_k, sol)
        P_HPP_DW_k_opt = get_var_value_from_sol(P_HPP_DW_k, sol)

        delta_P_HPP_s_opt = get_var_value_from_sol(delta_P_HPP_s, sol)
        delta_P_HPP_UP_s_opt = get_var_value_from_sol(delta_P_HPP_UP_s, sol)
        delta_P_HPP_DW_s_opt = get_var_value_from_sol(delta_P_HPP_DW_s, sol)

        E_HPP_HA_t_opt = P_HPP_HA_t_opt * dt

        P_W_HA_cur_t_opt = (
            np.array(HA_wind_forecast[0:].T) - np.array(P_W_HA_t_opt).flatten()
        )
        P_W_HA_cur_t_opt = pd.DataFrame(P_W_HA_cur_t_opt)
        P_S_HA_cur_t_opt = (
            np.array(HA_solar_forecast[0:].T) - np.array(P_S_HA_t_opt).flatten()
        )
        P_S_HA_cur_t_opt = pd.DataFrame(P_S_HA_cur_t_opt)

        z_ts = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient="index").reindex(
            setT, fill_value=0
        )

    else:
        aa = BMOpt_mdl.get_solve_details()
        print(aa.status)
        # print(SMOpt_mdl.export_to_string())
    # return E_HPP_HA_t_opt, P_HPP_HA_t_opt, P_dis_HA_t_opt, P_cha_HA_t_opt, P_HPP_UP_t_opt, P_HPP_DW_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt, SoC_HA_t_opt, P_W_HA_cur_t_opt, P_S_HA_cur_t_opt, P_W_HA_t_opt, P_S_HA_t_opt, delta_P_HPP_s_opt, delta_P_HPP_UP_s_opt, delta_P_HPP_DW_s_opt
    return P_HPP_HA_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt


def RDOpt(parameter_dict, simulation_dict, dynamic_inputs, verbose=False):

    day_num = dynamic_inputs["day_num"]
    Emax = dynamic_inputs["Emax"]
    ad = dynamic_inputs["ad"]
    P_HPP_SM_t_opt = dynamic_inputs["P_HPP_SM_t_opt"]
    start = dynamic_inputs["Current_DI"]
    s_UP_t = dynamic_inputs["s_UP_t"]
    s_DW_t = dynamic_inputs["s_DW_t"]
    P_HPP_UP_t0 = dynamic_inputs["P_HPP_UP_t0"]
    P_HPP_DW_t0 = dynamic_inputs["P_HPP_DW_t0"]
    P_HPP_UP_t1 = dynamic_inputs["P_HPP_UP_t1"]
    P_HPP_DW_t1 = dynamic_inputs["P_HPP_DW_t1"]
    SoC0 = dynamic_inputs["SoC0"]
    exist_imbalance = dynamic_inputs["exist_imbalance"]

    # Optimization modelling by CPLEX
    dt = parameter_dict["dispatch_interval"]
    dt_num = int(1 / dt)
    T = int(1 / dt * 24)

    ds = parameter_dict["settlement_interval"]
    ds_num = int(1 / ds)
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    dk = parameter_dict["offer_interval"]
    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    current_SI = start // dsdt_num
    current_hour = start // dt_num
    setT = [i for i in range(start, T)]
    setT1 = [i for i in range((current_hour + 2) * dt_num, T)]
    setK = [i for i in range(current_hour * dk_num, T_dk)]
    setK1 = [i for i in range((current_hour + 2) * dk_num, T_dk)]
    setS = [i for i in range(current_SI, T_ds)]
    set_SoCT = [i for i in range(start, T + 1)]

    PwMax = parameter_dict["wind_capacity"]
    PsMax = parameter_dict["solar_capacity"]
    EBESS = parameter_dict["battery_energy_capacity"]
    PbMax = parameter_dict["battery_power_capacity"]
    SoCmin = parameter_dict["battery_minimum_SoC"]
    SoCmax = parameter_dict["battery_maximum_SoC"]
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"]

    P_grid_limit = parameter_dict["hpp_grid_connection"]
    mu = parameter_dict["battery_marginal_degradation_cost"]
    deg_indicator = parameter_dict["degradation_in_optimization"]

    # Read data
    ReadData = DataReaderBase(
        day_num=day_num,
        DI_num=dt_num,
        T=T,
        PsMax=PsMax,
        PwMax=PwMax,
        simulation_dict=simulation_dict,
    )
    Inputs = ReadData.execute()

    DA_wind_forecast = Inputs["DA_wind_forecast"]
    DA_solar_forecast = Inputs["DA_solar_forecast"]
    HA_wind_forecast = Inputs["HA_wind_forecast"]
    HA_solar_forecast = Inputs["HA_solar_forecast"]
    RT_wind_forecast = Inputs["RT_wind_forecast"]
    RT_solar_forecast = Inputs["RT_solar_forecast"]
    Wind_measurement = Inputs["Wind_measurement"]
    Solar_measurement = Inputs["Solar_measurement"]

    RD_wind_forecast = pd.Series(
        np.r_[
            RT_wind_forecast.values[start : start + 2],
            HA_wind_forecast.values[start + 2 : (current_hour + 2) * dt_num],
            Wind_measurement.values[(current_hour + 2) * dt_num :]
            + 0.8
            * (
                DA_wind_forecast.values[(current_hour + 2) * dt_num :]
                - Wind_measurement.values[(current_hour + 2) * dt_num :]
            ),
        ]
    )
    RD_solar_forecast = pd.Series(
        np.r_[
            RT_solar_forecast.values[start : start + 2],
            HA_solar_forecast.values[start + 2 : (current_hour + 2) * dt_num],
            Solar_measurement[(current_hour + 2) * dt_num :]
            + 0.8
            * (
                DA_solar_forecast.values[(current_hour + 2) * dt_num :]
                - Solar_measurement[(current_hour + 2) * dt_num :]
            ),
        ]
    )

    BM_dw_price_forecast = Inputs["BM_dw_price_forecast"]
    BM_up_price_forecast = Inputs["BM_up_price_forecast"]

    reg_up_sign_forecast = Inputs["reg_up_sign_forecast"]
    reg_dw_sign_forecast = Inputs["reg_dw_sign_forecast"]

    BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(ds_num)
    BM_up_price_forecast_settle.index = range(T_ds)

    BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(ds_num)
    BM_dw_price_forecast_settle.index = range(T_ds)

    reg_up_sign_forecast1 = reg_up_sign_forecast.repeat(dt_num)
    reg_dw_sign_forecast1 = reg_dw_sign_forecast.repeat(dt_num)
    reg_up_sign_forecast1.index = range(T)
    reg_dw_sign_forecast1.index = range(T)

    RDOpt_mdl = Model()

    # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_UP_t = RDOpt_mdl.continuous_var_dict(
        setT1,
        lb=0,
        ub=dynamic_inputs["RDOpt_mFRREAM_enabler"] * P_grid_limit,
        name="BM UP bidding 5min",
    )
    P_HPP_DW_t = RDOpt_mdl.continuous_var_dict(
        setT1,
        lb=0,
        ub=dynamic_inputs["RDOpt_mFRREAM_enabler"] * P_grid_limit,
        name="BM DW bidding 5min",
    )
    P_HPP_UP_k = RDOpt_mdl.continuous_var_dict(
        setK1,
        lb=0,
        ub=dynamic_inputs["RDOpt_mFRREAM_enabler"] * P_grid_limit,
        name="BM UP bidding",
    )
    P_HPP_DW_k = RDOpt_mdl.continuous_var_dict(
        setK1,
        lb=0,
        ub=dynamic_inputs["RDOpt_mFRREAM_enabler"] * P_grid_limit,
        name="BM DW bidding",
    )

    P_HPP_RD_t = RDOpt_mdl.continuous_var_dict(
        setT, name="HA schedule with balancing bidding"
    )
    P_W_RD_t = {
        t: RDOpt_mdl.continuous_var(
            lb=0, ub=RD_wind_forecast[t - start], name="HA Wind schedule {}".format(t)
        )
        for t in setT
    }
    P_S_RD_t = {
        t: RDOpt_mdl.continuous_var(
            lb=0, ub=RD_solar_forecast[t - start], name="HA Solar schedule {}".format(t)
        )
        for t in setT
    }
    P_dis_RD_t = RDOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="HA discharge"
    )
    P_cha_RD_t = RDOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name="HA charge")
    P_b_RD_t = RDOpt_mdl.continuous_var_dict(
        setT, lb=-PbMax, ub=PbMax, name="HA Battery schedule"
    )  # (must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RD_t = RDOpt_mdl.continuous_var_dict(
        set_SoCT, lb=SoCmin, ub=SoCmax, name="HA SoC"
    )
    z_t = RDOpt_mdl.binary_var_dict(setT, name="Cha or Discha")

    delta_P_HPP_s = RDOpt_mdl.continuous_var_dict(
        setS, lb=-P_grid_limit, ub=P_grid_limit, name="HA imbalance"
    )
    delta_P_HPP_UP_s = RDOpt_mdl.continuous_var_dict(
        setS, lb=0, ub=P_grid_limit, name="HA up imbalance"
    )
    delta_P_HPP_DW_s = RDOpt_mdl.continuous_var_dict(
        setS, lb=0, ub=P_grid_limit, name="HA dw imbalance"
    )

    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  # dataframe to series
    # Define constraints
    for t in setT:

        RDOpt_mdl.add_constraint(
            P_HPP_RD_t[t] == P_W_RD_t[t] + P_S_RD_t[t] + P_b_RD_t[t]
        )
        RDOpt_mdl.add_constraint(P_b_RD_t[t] == P_dis_RD_t[t] - P_cha_RD_t[t])
        RDOpt_mdl.add_constraint(P_dis_RD_t[t] <= (PbMax) * z_t[t])
        RDOpt_mdl.add_constraint(P_cha_RD_t[t] <= (PbMax) * (1 - z_t[t]))

        RDOpt_mdl.add_constraint(
            SoC_RD_t[t + 1]
            == SoC_RD_t[t] * (1 - eta_leak)
            - 1 / Emax * P_dis_RD_t[t] / eta_dis * dt
            + 1 / Emax * P_cha_RD_t[t] * eta_cha * dt
        )

        RDOpt_mdl.add_constraint(SoC_RD_t[t] <= SoCmax)
        RDOpt_mdl.add_constraint(SoC_RD_t[t] >= SoCmin)

        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] <= P_grid_limit)
        RDOpt_mdl.add_constraint(P_HPP_RD_t[t] >= -P_grid_limit)

    for s in setS:
        RDOpt_mdl.add_constraint(
            delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s]
        )
        if s < (current_hour + 1) * ds_num:
            if s == current_SI:
                RDOpt_mdl.add_constraint(
                    delta_P_HPP_s[s]
                    == (
                        exist_imbalance
                        + RDOpt_mdl.sum(
                            (
                                P_HPP_RD_t[s * dsdt_num + j]
                                - (
                                    P_HPP_UP_t0 * s_UP_t[s * dsdt_num + j]
                                    - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + j]
                                )
                                - P_HPP_SM_t_opt[s * dsdt_num + j]
                            )
                            * dt
                            for j in range(start % dsdt_num, dsdt_num)
                        )
                    )
                    / ds
                )
            else:
                # RDOpt_mdl.add_constraint(delta_P_HPP_s[s] == RDOpt_mdl.sum(P_HPP_RD_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
                RDOpt_mdl.add_constraint(
                    delta_P_HPP_s[s]
                    == RDOpt_mdl.sum(
                        P_HPP_RD_t[s * dsdt_num + j]
                        - (
                            P_HPP_UP_t0 * s_UP_t[s * dsdt_num + j]
                            - P_HPP_DW_t0 * s_DW_t[s * dsdt_num + j]
                        )
                        - P_HPP_SM_t_opt[s * dsdt_num + j]
                        for j in range(0, dsdt_num)
                    )
                    / dsdt_num
                )
        elif s >= (current_hour + 1) * ds_num and s < (current_hour + 2) * ds_num:
            RDOpt_mdl.add_constraint(
                delta_P_HPP_s[s]
                == RDOpt_mdl.sum(
                    P_HPP_RD_t[s * dsdt_num + j]
                    - (
                        reg_up_sign_forecast1[s * dsdt_num + j] * P_HPP_UP_t1
                        - reg_dw_sign_forecast1[s * dsdt_num + j] * P_HPP_DW_t1
                    )
                    - P_HPP_SM_t_opt[s * dsdt_num + j]
                    for j in range(0, dsdt_num)
                )
                / dsdt_num
            )
        else:
            RDOpt_mdl.add_constraint(
                delta_P_HPP_s[s]
                == RDOpt_mdl.sum(
                    P_HPP_RD_t[s * dsdt_num + j]
                    - (
                        reg_up_sign_forecast1[s * dsdt_num + j]
                        * P_HPP_UP_t[s * dsdt_num + j]
                        - reg_dw_sign_forecast1[s * dsdt_num + j]
                        * P_HPP_DW_t[s * dsdt_num + j]
                    )
                    - P_HPP_SM_t_opt[s * dsdt_num + j]
                    for j in range(0, dsdt_num)
                )
                / dsdt_num
            )

    for k in setK1:
        for j in range(0, dt_num):
            RDOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
            RDOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])

    RDOpt_mdl.add_constraint(SoC_RD_t[start] == SoC0)

    Revenue = RDOpt_mdl.sum(
        BM_up_price_forecast[k] * reg_up_sign_forecast[k] * P_HPP_UP_k[k] * dk
        - BM_dw_price_forecast[k] * reg_dw_sign_forecast[k] * P_HPP_DW_k[k] * dk
        for k in setK1
    ) + RDOpt_mdl.sum(
        (BM_dw_price_forecast_settle[s] - 0.001) * delta_P_HPP_UP_s[s] * ds
        - (BM_up_price_forecast_settle[s] + 0.001) * delta_P_HPP_DW_s[s] * ds
        for s in setS
    )

    if deg_indicator == 1:
        Deg_cost = (
            mu
            * EBESS
            * ad
            * RDOpt_mdl.sum((P_dis_RD_t[t] + P_cha_RD_t[t]) * dt for t in setT)
        )
    else:
        Deg_cost = 0

    RDOpt_mdl.maximize(Revenue - Deg_cost)

    # Solve RDOpt Model
    sol = RDOpt_mdl.solve()
    if verbose:
        RDOpt_mdl.print_information()
        aa = RDOpt_mdl.get_solve_details()
        print(aa.status)

    if sol:

        P_HPP_RD_t_opt = get_var_value_from_sol(P_HPP_RD_t, sol)
        P_HPP_RD_t_opt.columns = ["RD"]  # Preserved custom column name

        P_W_RD_t_opt = get_var_value_from_sol(P_W_RD_t, sol)
        P_S_RD_t_opt = get_var_value_from_sol(P_S_RD_t, sol)
        P_dis_RD_t_opt = get_var_value_from_sol(P_dis_RD_t, sol)
        P_cha_RD_t_opt = get_var_value_from_sol(P_cha_RD_t, sol)

        SoC_RD_t_opt = get_var_value_from_sol(SoC_RD_t, sol)
        P_HPP_UP_t_opt = get_var_value_from_sol(P_HPP_UP_t, sol)
        P_HPP_DW_t_opt = get_var_value_from_sol(P_HPP_DW_t, sol)
        P_HPP_UP_k_opt = get_var_value_from_sol(P_HPP_UP_k, sol)
        P_HPP_DW_k_opt = get_var_value_from_sol(P_HPP_DW_k, sol)

        delta_P_HPP_s_opt = get_var_value_from_sol(delta_P_HPP_s, sol)
        delta_P_HPP_UP_s_opt = get_var_value_from_sol(delta_P_HPP_UP_s, sol)
        delta_P_HPP_DW_s_opt = get_var_value_from_sol(delta_P_HPP_DW_s, sol)

        E_HPP_RD_t_opt = P_HPP_RD_t_opt * dt

        P_W_RD_cur_t_opt = (
            np.array(RD_wind_forecast[0:].T) - np.array(P_W_RD_t_opt).flatten()
        )
        P_W_RD_cur_t_opt = pd.DataFrame(P_W_RD_cur_t_opt)
        P_S_RD_cur_t_opt = (
            np.array(RD_solar_forecast[0:].T) - np.array(P_S_RD_t_opt).flatten()
        )
        P_S_RD_cur_t_opt = pd.DataFrame(P_S_RD_cur_t_opt)

        z_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient="index")

    else:
        print("RDOpt has no solution")
        # print(SMOpt_mdl.expoRD_to_string())
    return P_HPP_RD_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt


def RBOpt(parameter_dict, simulation_dict, dynamic_inputs, verbose=False):

    day_num = dynamic_inputs["day_num"]
    Emax = dynamic_inputs["Emax"]
    ad = dynamic_inputs["ad"]
    P_HPP_SM_t_opt = dynamic_inputs["P_HPP_SM_t_opt"]
    start = dynamic_inputs["Current_DI"]
    SoC0 = dynamic_inputs["SoC0"]
    exist_imbalance = dynamic_inputs["exist_imbalance"]
    # Optimization modelling by CPLEX
    dt = parameter_dict["dispatch_interval"]
    dt_num = int(1 / dt)
    T = int(1 / dt * 24)

    ds = parameter_dict["settlement_interval"]
    ds_num = int(1 / ds)
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    dk = parameter_dict["offer_interval"]
    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    current_SI = start // dsdt_num
    current_hour = start // dt_num
    setT = [i for i in range(start, T)]
    setT1 = [i for i in range((current_hour + 2) * dt_num, T)]
    setK = [i for i in range(current_hour * dk_num, T_dk)]
    setK1 = [i for i in range((current_hour + 2) * dk_num, T_dk)]
    setS = [i for i in range(current_SI, T_ds)]
    set_SoCT = [i for i in range(start, T + 1)]

    PwMax = parameter_dict["wind_capacity"]
    PsMax = parameter_dict["solar_capacity"]
    EBESS = parameter_dict["battery_energy_capacity"]
    PbMax = parameter_dict["battery_power_capacity"]
    SoCmin = parameter_dict["battery_minimum_SoC"]
    SoCmax = parameter_dict["battery_maximum_SoC"]
    eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
    eta_cha = parameter_dict["battery_hour_charge_efficiency"]
    eta_leak = parameter_dict["battery_self_discharge_efficiency"]

    P_grid_limit = parameter_dict["hpp_grid_connection"]
    mu = parameter_dict["battery_marginal_degradation_cost"]
    deg_indicator = parameter_dict["degradation_in_optimization"]

    # Read data
    ReadData = DataReaderBase(
        day_num=day_num,
        DI_num=dt_num,
        T=T,
        PsMax=PsMax,
        PwMax=PwMax,
        simulation_dict=simulation_dict,
    )
    Inputs = ReadData.execute()

    DA_wind_forecast = Inputs["DA_wind_forecast"]
    DA_solar_forecast = Inputs["DA_solar_forecast"]
    HA_wind_forecast = Inputs["HA_wind_forecast"]
    HA_solar_forecast = Inputs["HA_solar_forecast"]
    RT_wind_forecast = Inputs["RT_wind_forecast"]
    RT_solar_forecast = Inputs["RT_solar_forecast"]
    Wind_measurement = Inputs["Wind_measurement"]
    Solar_measurement = Inputs["Solar_measurement"]

    RB_wind_forecast = pd.Series(
        np.r_[
            RT_wind_forecast.values[start : start + 2],
            HA_wind_forecast.values[start + 2 : (current_hour + 2) * dt_num],
            Wind_measurement.values[(current_hour + 2) * dt_num :]
            + 0.8
            * (
                DA_wind_forecast.values[(current_hour + 2) * dt_num :]
                - Wind_measurement.values[(current_hour + 2) * dt_num :]
            ),
        ]
    )
    RB_solar_forecast = pd.Series(
        np.r_[
            RT_solar_forecast.values[start : start + 2],
            HA_solar_forecast.values[start + 2 : (current_hour + 2) * dt_num],
            Solar_measurement[(current_hour + 2) * dt_num :]
            + 0.8
            * (
                DA_solar_forecast.values[(current_hour + 2) * dt_num :]
                - Solar_measurement[(current_hour + 2) * dt_num :]
            ),
        ]
    )

    BM_dw_price_forecast = Inputs["BM_dw_price_forecast"]
    BM_up_price_forecast = Inputs["BM_up_price_forecast"]

    reg_up_sign_forecast = Inputs["reg_up_sign_forecast"]
    reg_dw_sign_forecast = Inputs["reg_dw_sign_forecast"]

    BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(ds_num)
    BM_up_price_forecast_settle.index = range(T_ds)

    BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(ds_num)
    BM_dw_price_forecast_settle.index = range(T_ds)

    reg_up_sign_forecast1 = reg_up_sign_forecast.repeat(dt_num)
    reg_dw_sign_forecast1 = reg_dw_sign_forecast.repeat(dt_num)
    reg_up_sign_forecast1.index = range(T)
    reg_dw_sign_forecast1.index = range(T)

    RBOpt_mdl = Model()

    # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    # P_HPP_all_t = BMOpt_mdl.continuous_var_dict(setT, name='HA schedule with balancing bidding')
    P_HPP_RB_t = RBOpt_mdl.continuous_var_dict(
        setT, name="HA schedule with balancing bidding"
    )
    P_W_RB_t = {
        t: RBOpt_mdl.continuous_var(
            lb=0, ub=RB_wind_forecast[t - start], name="HA Wind schedule {}".format(t)
        )
        for t in setT
    }
    P_S_RB_t = {
        t: RBOpt_mdl.continuous_var(
            lb=0, ub=RB_solar_forecast[t - start], name="HA Solar schedule {}".format(t)
        )
        for t in setT
    }
    P_dis_RB_t = RBOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="HA discharge"
    )
    P_cha_RB_t = RBOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name="HA charge")
    P_b_RB_t = RBOpt_mdl.continuous_var_dict(
        setT, lb=-PbMax, ub=PbMax, name="HA Battery schedule"
    )  # (must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RB_t = RBOpt_mdl.continuous_var_dict(
        set_SoCT, lb=SoCmin, ub=SoCmax, name="HA SoC"
    )
    z_t = RBOpt_mdl.binary_var_dict(setT, name="Cha or Discha")
    # an_var     = RBOpt_mdl.continuous_var(lb=0, ub=0.5, name='anciliary var')
    # v_t        = RBOpt_mdl.binary_var_dict(setT, name='Ban up or ban dw')
    delta_P_HPP_s = RBOpt_mdl.continuous_var_dict(
        setS, lb=-P_grid_limit, ub=P_grid_limit, name="HA imbalance"
    )
    delta_P_HPP_UP_s = RBOpt_mdl.continuous_var_dict(
        setS, lb=0, ub=P_grid_limit, name="HA up imbalance"
    )
    delta_P_HPP_DW_s = RBOpt_mdl.continuous_var_dict(
        setS, lb=0, ub=P_grid_limit, name="HA dw imbalance"
    )
    # delta_E_HPP_DW_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min dw imbalance')
    # delta_E_HPP_UP_k = BMOpt_mdl.continuous_var_dict(setK, name='HA 15min up imbalance')
    P_HPP_SM_t_opt = P_HPP_SM_t_opt.squeeze()  # dataframe to series
    # Define constraints
    for t in setT:
        RBOpt_mdl.add_constraint(
            P_HPP_RB_t[t] == P_W_RB_t[t] + P_S_RB_t[t] + P_b_RB_t[t]
        )
        RBOpt_mdl.add_constraint(P_b_RB_t[t] == P_dis_RB_t[t] - P_cha_RB_t[t])
        RBOpt_mdl.add_constraint(P_dis_RB_t[t] <= (PbMax) * z_t[t])
        RBOpt_mdl.add_constraint(P_cha_RB_t[t] <= (PbMax) * (1 - z_t[t]))
        RBOpt_mdl.add_constraint(
            SoC_RB_t[t + 1]
            == SoC_RB_t[t] * (1 - eta_leak)
            - 1 / Emax * P_dis_RB_t[t] / eta_dis * dt
            + 1 / Emax * P_cha_RB_t[t] * eta_cha * dt
        )
        RBOpt_mdl.add_constraint(SoC_RB_t[t] <= SoCmax)
        RBOpt_mdl.add_constraint(SoC_RB_t[t] >= SoCmin)
        RBOpt_mdl.add_constraint(P_HPP_RB_t[t] <= P_grid_limit)
        RBOpt_mdl.add_constraint(P_HPP_RB_t[t] >= -P_grid_limit)

    for s in setS:
        RBOpt_mdl.add_constraint(
            delta_P_HPP_s[s] == delta_P_HPP_UP_s[s] - delta_P_HPP_DW_s[s]
        )
        if s < (current_hour + 1) * ds_num:
            if s == current_SI:
                # RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == (exist_imbalance + RBOpt_mdl.sum((P_HPP_RB_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j]) * dt for j in range(start%dsdt_num, dsdt_num)))/ds)
                RBOpt_mdl.add_constraint(
                    delta_P_HPP_s[s]
                    == (
                        exist_imbalance
                        + RBOpt_mdl.sum(
                            (
                                P_HPP_RB_t[s * dsdt_num + j]
                                - P_HPP_SM_t_opt[s * dsdt_num + j]
                            )
                            * dt
                            for j in range(start % dsdt_num, dsdt_num)
                        )
                    )
                    / ds
                )
            else:
                # RBOpt_mdl.add_constraint(delta_P_HPP_s[s] == RBOpt_mdl.sum(P_HPP_RB_t[s * dsdt_num + j] - (s_UP_t[s * dsdt_num + j] * P_HPP_UP_t[s * dsdt_num + j] - s_DW_t[s * dsdt_num + j] * P_HPP_DW_t[s * dsdt_num + j]) - P_HPP_SM_t_opt[s * dsdt_num + j] for j in range(0, dsdt_num))/dsdt_num)
                RBOpt_mdl.add_constraint(
                    delta_P_HPP_s[s]
                    == RBOpt_mdl.sum(
                        P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j]
                        for j in range(0, dsdt_num)
                    )
                    / dsdt_num
                )
        elif s >= (current_hour + 1) * ds_num and s < (current_hour + 2) * ds_num:
            RBOpt_mdl.add_constraint(
                delta_P_HPP_s[s]
                == RBOpt_mdl.sum(
                    P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j]
                    for j in range(0, dsdt_num)
                )
                / dsdt_num
            )
        else:
            RBOpt_mdl.add_constraint(
                delta_P_HPP_s[s]
                == RBOpt_mdl.sum(
                    P_HPP_RB_t[s * dsdt_num + j] - P_HPP_SM_t_opt[s * dsdt_num + j]
                    for j in range(0, dsdt_num)
                )
                / dsdt_num
            )

    #   {t : BMOpt_mdl.add_constraint(ct=delta_P_HPP_t[t] == P_HPP_BM_t[t] - P_HPP_DA_ts[t], ctname="constraint_{0}".format(t)) for t in setT }

    RBOpt_mdl.add_constraint(SoC_RB_t[start] == SoC0)
    #    RBOpt_mdl.add_constraint(SoC_RB_t[T] <= 0.6)
    #    RBOpt_mdl.add_constraint(SoC_RB_t[T] >= 0.4)

    # Revenue = RBOpt_mdl.sum(BM_dw_price_forecast[k] * delta_P_HPP_UP_k[k] *dk - BM_up_price_forecast[k] * delta_P_HPP_DW_k[k] *dk for k in setK)
    Revenue = RBOpt_mdl.sum(
        (BM_dw_price_forecast_settle[s] - 0.001) * delta_P_HPP_UP_s[s] * ds
        - (BM_up_price_forecast_settle[s] + 0.001) * delta_P_HPP_DW_s[s] * ds
        for s in setS
    )
    # Revenue = RBOpt_mdl.sum(BM_dw_price_forecast[k] * delta_P_HPP_UP_k[k] *dk - BM_up_price_forecast[k] * delta_P_HPP_DW_k[k] *dk for k in setK)
    if deg_indicator == 1:
        Deg_cost = (
            mu
            * EBESS
            * ad
            * RBOpt_mdl.sum((P_dis_RB_t[t] + P_cha_RB_t[t]) * dt for t in setT)
        )
    else:
        Deg_cost = 0
    # RBOpt_mdl.maximize(Revenue-Deg_cost - 1e7*an_var)
    RBOpt_mdl.maximize(Revenue - Deg_cost)

    # Solve RbOpt Model

    sol = RBOpt_mdl.solve()
    if verbose:
        RBOpt_mdl.print_information()
        aa = RBOpt_mdl.get_solve_details()
        print(aa.status)

    if sol:
        #    SMOpt_mdl.print_solution()
        P_HPP_RB_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(P_HPP_RB_t), orient="index"
        )
        P_HPP_RB_t_opt.columns = ["RB"]
        P_W_RB_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(P_W_RB_t), orient="index"
        )
        P_S_RB_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(P_S_RB_t), orient="index"
        )
        P_dis_RB_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(P_dis_RB_t), orient="index"
        )
        P_cha_RB_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(P_cha_RB_t), orient="index"
        )
        SoC_RB_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(SoC_RB_t), orient="index"
        )
        delta_P_HPP_s_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(delta_P_HPP_s), orient="index"
        )
        delta_P_HPP_UP_s_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(delta_P_HPP_UP_s), orient="index"
        )
        delta_P_HPP_DW_s_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(delta_P_HPP_DW_s), orient="index"
        )

        # print(SoC_RB_t_opt.iloc[12:15])

        E_HPP_RB_t_opt = P_HPP_RB_t_opt * dt

        P_W_RB_cur_t_opt = (
            np.array(RB_wind_forecast[0:].T) - np.array(P_W_RB_t_opt).flatten()
        )
        P_W_RB_cur_t_opt = pd.DataFrame(P_W_RB_cur_t_opt)
        P_S_RB_cur_t_opt = (
            np.array(RB_solar_forecast[0:].T) - np.array(P_S_RB_t_opt).flatten()
        )
        P_S_RB_cur_t_opt = pd.DataFrame(P_S_RB_cur_t_opt)

        z_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_t), orient="index")

    else:
        aa = RBOpt_mdl.get_solve_details()
        print(aa.status)
        # print(SMOpt_mdl.expoRB_to_string())
    return (
        E_HPP_RB_t_opt,
        P_HPP_RB_t_opt,
        P_dis_RB_t_opt,
        P_cha_RB_t_opt,
        SoC_RB_t_opt,
        P_W_RB_cur_t_opt,
        P_S_RB_cur_t_opt,
        P_W_RB_t_opt,
        P_S_RB_t_opt,
        delta_P_HPP_s_opt,
        delta_P_HPP_UP_s_opt,
        delta_P_HPP_DW_s_opt,
    )
