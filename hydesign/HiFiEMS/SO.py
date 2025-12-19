# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:57:55 2023

@author: ruzhu



single imbalance settlement
"""

import os
from datetime import datetime

import cplex
import numpy as np
import pandas as pd
from docplex.mp.model import Model
from scipy.stats import norm

from hydesign.HiFiEMS.utils import DataReaderBase


class DataReader(DataReaderBase):
    def __init__(self, day_num, DI_num, T, PsMax, PwMax, simulation_dict):
        super().__init__(day_num, DI_num, T, PsMax, PwMax, simulation_dict)

    def execute(self):
        Inputs = super().execute()

        scenario_num = self.sim["number_of_scenario"]

        T0 = 96
        indices = ["DA_" + str(i) for i in range(1, scenario_num + 1)]
        DA_wind_forecast_scenario = self.Wind_data[indices]
        DA_wind_forecast_scenario = DA_wind_forecast_scenario.to_numpy().transpose()
        DA_wind_forecast_scenario = (
            DA_wind_forecast_scenario[:, 0 : T0 : int(4 / self.DI_num)] * self.PwMax
        )

        # probability_wind = [1/wind_scenario_num ]*wind_scenario_num

        indices = ["DA_" + str(i) for i in range(1, scenario_num + 1)]
        DA_solar_forecast_scenario = self.Solar_data[indices]
        DA_solar_forecast_scenario = DA_solar_forecast_scenario.to_numpy().transpose()
        DA_solar_forecast_scenario = (
            DA_solar_forecast_scenario[:, 0 : T0 : int(4 / self.DI_num)] * self.PsMax
        )

        # probability_solar = [1/solar_scenario_num ]*solar_scenario_num

        indices = ["SM_forecast_" + str(i) for i in range(1, scenario_num + 1)]
        SP_scenario = self.Market_data[indices]
        SP_scenario = SP_scenario.to_numpy().transpose()
        # SP_scenario = SP_scenario[:,0:T0:int(4/DI_num)]

        indices = ["reg_forecast_" + str(i) for i in range(1, scenario_num + 1)]
        RP_scenario = self.Market_data[indices]
        RP_scenario = RP_scenario.to_numpy().transpose()
        # RP_scenario = RP_scenario[:,0:T0:int(4/DI_num)]
        if self.sim["probability"] is None:
            probability = [1 / scenario_num] * scenario_num
        else:
            probability = self.sim["probability"]

        Inputs["SP_scenario"] = SP_scenario
        Inputs["RP_scenario"] = RP_scenario
        Inputs["probability"] = probability
        Inputs["DA_wind_forecast_scenario"] = DA_wind_forecast_scenario
        Inputs["DA_solar_forecast_scenario"] = DA_solar_forecast_scenario
        Inputs["scenario_num"] = scenario_num

        return Inputs


def f_xmin_to_ymin(x, reso_x, reso_y):  # x: dataframe reso: in hour
    y = pd.DataFrame()
    if reso_y > reso_x:
        a = 0
        num = int(reso_y / reso_x)

        for ii in range(len(x)):
            if ii % num == num - 1:
                a = (a + x.iloc[ii][0]) / num
                y = y.append(pd.DataFrame([a]))
                a = 0
            else:
                a = a + x.iloc[ii][0]
        y.index = range(int(len(x) / num))
    else:
        y = pd.DataFrame(np.repeat(x.iloc[:, 0], int(reso_x / reso_y)))
        y.index = range(int(24 / reso_y))
    return y


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
    C_dev = parameter_dict["imbalance_fee"]
    deviation = parameter_dict["deviation"]

    ds = parameter_dict["settlement_interval"]
    ds_num = int(1 / ds)
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    dk = parameter_dict["offer_interval"]
    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    #
    setT = [i for i in range(T)]
    setS = [i for i in range(T_ds)]
    set_SoCT = [i for i in range(T + 1)]
    setK = [i for i in range(T_dk)]

    ReadData = DataReader(
        day_num=day_num,
        DI_num=dt_num,
        T=T,
        PsMax=PsMax,
        PwMax=PwMax,
        simulation_dict=simulation_dict,
    )
    Inputs = ReadData.execute()

    DA_wind_forecast_scenario = Inputs["DA_wind_forecast_scenario"]
    DA_solar_forecast_scenario = Inputs["DA_solar_forecast_scenario"]
    SP_scenario = Inputs["SP_scenario"]
    RP_scenario = Inputs["RP_scenario"]
    probability = Inputs["probability"]
    scenario_num = Inputs["scenario_num"]

    RP_scenario_sub = np.repeat(RP_scenario, ds_num, axis=1)

    SMOpt_mdl = Model()

    # Define variables (must define lb and ub, otherwise may cause issues on cplex)

    P_HPP_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, name="SM schedule subhourly")
    P_HPP_SM_k = SMOpt_mdl.continuous_var_dict(setK, lb=0, name="SM schedule hourly")
    P_w_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PwMax, name="SM wind subhourly"
    )
    P_s_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PsMax, name="SM solar subhourly"
    )
    P_dis_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="SM discharge subhourly"
    )
    P_cha_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="SM charge subhourly"
    )
    E_SM_t = SMOpt_mdl.continuous_var_dict(
        set_SoCT, lb=-cplex.infinity, ub=cplex.infinity, name="SM SoC"
    )
    z_t = SMOpt_mdl.binary_var_dict(setT, name="Cha or Discha")

    # Define constraints
    for t in setT:
        SMOpt_mdl.add_constraint(
            P_HPP_SM_t[t] == P_w_SM_t[t] + P_s_SM_t[t] + P_dis_SM_t[t] - P_cha_SM_t[t]
        )
        SMOpt_mdl.add_constraint(P_dis_SM_t[t] <= (PbMax) * z_t[t])
        SMOpt_mdl.add_constraint(P_cha_SM_t[t] <= (PbMax) * (1 - z_t[t]))

        SMOpt_mdl.add_constraint(
            E_SM_t[t + 1]
            == E_SM_t[t] * (1 - eta_leak)
            - (P_dis_SM_t[t]) / eta_dis * dt
            + (P_cha_SM_t[t]) * eta_cha * dt
        )
        SMOpt_mdl.add_constraint(E_SM_t[t + 1] <= SoCmax * Emax)
        SMOpt_mdl.add_constraint(E_SM_t[t + 1] >= SoCmin * Emax)

        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] <= P_grid_limit)

    for k in setK:
        for j in range(dt_num):
            SMOpt_mdl.add_constraint(P_HPP_SM_t[k * dt_num + j] == P_HPP_SM_k[k])
            SMOpt_mdl.add_constraint(P_HPP_SM_t[k * dt_num + j] == P_HPP_SM_k[k])

    SMOpt_mdl.add_constraint(E_SM_t[0] == SoC0 * Emax)

    # second-stage variables and constriants

    setV = [i for i in range(scenario_num)]
    setTV = [(i, j) for i in setT for j in setV]
    setSV = [(i, j) for i in setS for j in setV]

    set_SoCTV = [(i, j) for i in set_SoCT for j in setV]

    P_tilde_SM_dis_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=cplex.infinity, name="RT discHArge"
    )
    P_tilde_SM_cha_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=cplex.infinity, name="RT charge"
    )
    P_tilde_w_SM_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=PwMax, name="RT wind 15min"
    )  # (must define lb and ub, otherwise may cause unknown issues on cplex)
    P_tilde_s_SM_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=PsMax, name="RT solar 15min"
    )
    E_tilde_SM_t = SMOpt_mdl.continuous_var_dict(
        set_SoCTV, lb=SoCmin * Emax, ub=Emax, name="RT SoC"
    )

    delta_tilde_P_HPP_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=-cplex.infinity, ub=cplex.infinity, name="RT imbalance"
    )
    delta_tilde_P_HPP_s = SMOpt_mdl.continuous_var_dict(
        setSV, lb=-cplex.infinity, ub=cplex.infinity, name="RT imbalance 15min"
    )

    tau_s = SMOpt_mdl.continuous_var_dict(
        setSV, lb=-cplex.infinity, ub=cplex.infinity, name="aux"
    )

    z_tilde_t = SMOpt_mdl.binary_var_dict(setTV, name="RT Cha or DiscSM")

    for v in setV:
        for t in setT:
            SMOpt_mdl.add_constraint(P_tilde_SM_dis_t[t, v] <= PbMax * z_tilde_t[t, v])
            SMOpt_mdl.add_constraint(
                P_tilde_SM_cha_t[t, v] <= PbMax * (1 - z_tilde_t[t, v])
            )

            SMOpt_mdl.add_constraint(
                E_tilde_SM_t[t + 1, v]
                == E_tilde_SM_t[t, v] * (1 - eta_leak)
                - (P_tilde_SM_dis_t[t, v]) / eta_dis * dt
                + (P_tilde_SM_cha_t[t, v]) * eta_cha * dt
            )
            SMOpt_mdl.add_constraint(E_tilde_SM_t[t + 1, v] <= Emax * SoCmax)
            SMOpt_mdl.add_constraint(E_tilde_SM_t[t + 1, v] >= Emax * SoCmin)

            SMOpt_mdl.add_constraint(
                delta_tilde_P_HPP_t[t, v]
                == P_tilde_w_SM_t[t, v]
                + P_tilde_s_SM_t[t, v]
                + P_tilde_SM_dis_t[t, v]
                - P_tilde_SM_cha_t[t, v]
                - P_HPP_SM_t[t]
            )

            SMOpt_mdl.add_constraint(
                P_tilde_w_SM_t[t, v]
                + P_tilde_s_SM_t[t, v]
                + P_tilde_SM_dis_t[t, v]
                - P_tilde_SM_cha_t[t, v]
                <= P_grid_limit
            )
            SMOpt_mdl.add_constraint(
                P_tilde_w_SM_t[t, v]
                + P_tilde_s_SM_t[t, v]
                + P_tilde_SM_dis_t[t, v]
                - P_tilde_SM_cha_t[t, v]
                >= 0
            )

            SMOpt_mdl.add_constraint(
                P_tilde_w_SM_t[t, v] <= DA_wind_forecast_scenario[v, t]
            )
            SMOpt_mdl.add_constraint(
                P_tilde_s_SM_t[t, v] <= DA_solar_forecast_scenario[v, t]
            )

        for s in setS:
            SMOpt_mdl.add_constraint(tau_s[s, v] >= delta_tilde_P_HPP_s[s, v])
            SMOpt_mdl.add_constraint(tau_s[s, v] >= -delta_tilde_P_HPP_s[s, v])

            SMOpt_mdl.add_constraint(tau_s[s, v] <= deviation)

            for t in range(dsdt_num):
                SMOpt_mdl.add_constraint(
                    delta_tilde_P_HPP_t[s * dsdt_num + t, v]
                    == delta_tilde_P_HPP_s[s, v]
                )

        SMOpt_mdl.add_constraint(E_tilde_SM_t[0, v] == SoC0 * Emax)

    if deg_indicator == 1:

        SMOpt_mdl.maximize(
            SMOpt_mdl.sum(
                probability[i] * SP_scenario[i, k] * P_HPP_SM_k[k]
                for k in setK
                for i in setV
            )
            + SMOpt_mdl.sum(
                probability[i] * RP_scenario_sub[i, s] * delta_tilde_P_HPP_s[s, i] * ds
                for i in setV
                for s in setS
            )
            - SMOpt_mdl.sum(
                probability[i]
                * ad
                * EBESS
                * mu
                * (P_tilde_SM_dis_t[t, i] + P_tilde_SM_cha_t[t, i])
                * dt
                for i in setV
                for t in setT
            )
            - SMOpt_mdl.sum(
                probability[i] * C_dev * tau_s[s, i] * ds for s in setS for i in setV
            )
        )
    else:
        SMOpt_mdl.maximize(
            SMOpt_mdl.sum(
                probability[i] * SP_scenario[i, k] * P_HPP_SM_k[k]
                for k in setK
                for i in setV
            )
            + SMOpt_mdl.sum(
                probability[i] * RP_scenario_sub[i, s] * delta_tilde_P_HPP_s[s, i] * ds
                for i in setV
                for s in setS
            )
            - SMOpt_mdl.sum(
                probability[i] * C_dev * tau_s[s, i] * ds for s in setS for i in setV
            )
        )

    # Solve MasterOpt Model
    sol = SMOpt_mdl.solve()
    if verbose:
        SMOpt_mdl.print_information()

        aa = SMOpt_mdl.get_solve_details()
        print(aa.status)
    if sol:
        P_dis_SM_t_opt = get_var_value_from_sol(P_dis_SM_t, sol)
        P_cha_SM_t_opt = get_var_value_from_sol(P_cha_SM_t, sol)
        P_w_SM_t_opt = get_var_value_from_sol(P_w_SM_t, sol)
        P_s_SM_t_opt = get_var_value_from_sol(P_s_SM_t, sol)
        P_HPP_SM_t_opt = get_var_value_from_sol(P_HPP_SM_t, sol)
        P_HPP_SM_k_opt = get_var_value_from_sol(P_HPP_SM_k, sol)
        E_SM_t_opt = get_var_value_from_sol(E_SM_t, sol)
        E_HPP_SM_t_opt = P_HPP_SM_t_opt * dt
        SoC_SM_t_opt = E_SM_t_opt / Emax
        P_w_SM_cur_t_opt = DA_wind_forecast_scenario.T - np.array(P_w_SM_t_opt)
        P_w_SM_cur_t_opt = pd.DataFrame(P_w_SM_cur_t_opt)
        P_s_SM_cur_t_opt = DA_solar_forecast_scenario.T - np.array(P_s_SM_t_opt)
        P_s_SM_cur_t_opt = pd.DataFrame(P_s_SM_cur_t_opt)

        P_tilde_SM_dis_t_opt = get_var_value_from_sol(P_tilde_SM_dis_t, sol)
        P_tilde_SM_cha_t_opt = get_var_value_from_sol(P_tilde_SM_cha_t, sol)
        P_tilde_w_SM_t_opt = get_var_value_from_sol(P_tilde_w_SM_t, sol)
        delta_tilde_P_HPP_s_opt = get_var_value_from_sol(delta_tilde_P_HPP_s, sol)
        tau_s_opt = get_var_value_from_sol(tau_s, sol)
        obj = sol.get_objective_value()
    else:
        aa = SMOpt_mdl.get_solve_details()
        print(aa.status)

    return (
        E_HPP_SM_t_opt,
        P_HPP_SM_t_opt,
        P_HPP_SM_k_opt,
        P_dis_SM_t_opt,
        P_cha_SM_t_opt,
        SoC_SM_t_opt,
        P_w_SM_cur_t_opt,
        P_s_SM_cur_t_opt,
        P_w_SM_t_opt,
        P_s_SM_t_opt,
    )
