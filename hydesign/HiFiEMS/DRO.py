# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 10:11:36 2022

@author: ruzhu
"""

import time

import cplex
import numpy as np
import pandas as pd
from docplex.mp.model import Model
from numdifftools import Jacobian
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize
from sklearn.datasets import make_blobs

from hydesign.HiFiEMS.utils import DataReaderBase


class DataReader(DataReaderBase):
    def __init__(self, day_num, DI_num, T, PsMax, PwMax, simulation_dict):
        super().__init__(day_num, DI_num, T, PsMax, PwMax, simulation_dict)

    def execute(self):
        Inputs = super().execute()

        History_wind = pd.read_csv(self.sim["history_wind_fn"])
        History_wind_DA_error = (
            History_wind["DA_1"] - History_wind["Measurement"]
        ) * self.PwMax
        History_solar = pd.read_csv(self.sim["history_solar_fn"])
        History_solar_DA_error = (
            History_solar["DA_1"] - History_solar["Measurement"]
        ) * self.PsMax

        scenario_num = self.sim["number_of_scenario"]

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
        Inputs["scenario_num"] = scenario_num
        Inputs["History_wind_DA_error"] = History_wind_DA_error
        Inputs["History_solar_DA_error"] = History_solar_DA_error
        Inputs["N_Samples"] = self.sim["N_Samples"]
        Inputs["epsilon"] = self.sim["epsilon"]
        Inputs["epsilon1"] = self.sim["epsilon1"]

        return Inputs


def get_var_value_from_sol(x, sol):

    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient="index")

    return y


def f_xmin_to_ymin(x, reso_x, reso_y):  # x: dataframe reso: in hour
    y = pd.DataFrame()

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
    return y


def Uncertain_set(epsilon, Samples, N):
    Samples = Samples.to_numpy()

    setN = [i for i in range(N)]

    # epsilon = 0.5
    l_min = 0
    l_max = 2000

    pho = 0.3

    l_max = np.percentile(Samples, 95)
    l_min = np.percentile(Samples, 5)
    l_max = 1000
    l_min = 0

    Samples_uncorr = np.var(Samples) ** (-0.5) * (Samples - np.mean(Samples))

    # epsilon = 0.25

    while l_max - l_min > 1e-3:
        l = 0.5 * (l_min + l_max)
        fun = lambda x: x * epsilon + 1 / N * sum(
            max(1 - x * max(l - abs(Samples_uncorr[n, 0]), 0), 0) for n in setN
        )
        fun_Jac = lambda x: Jacobian(lambda x: fun(x))(x).ravel()
        cons = {"type": "ineq", "fun": lambda x: x - 0}
        # bnds = ((None))
        x0 = 0.1
        res = minimize(fun, x0, method="SLSQP", constraints=cons, jac=fun_Jac)
        if res.fun > pho:
            l_min = l
        else:
            l_max = l
    # if res.fun > pho:
    #    print('l_max is too small')

    # l = np.percentile(Samples_uncorr,95)
    l_max = l * np.var(Samples) ** (0.5) + np.mean(Samples)
    l_min = -l * np.var(Samples) ** (0.5) + np.mean(Samples)
    # l = 0.5*(l_min + l_max)

    return l_max, l_min


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
    # C_dev = parameter_dict['imbalance_fee']
    # deviation = parameter_dict['deviation']

    ds = parameter_dict["settlement_interval"]
    ds_num = int(1 / ds)
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    dk = parameter_dict["offer_interval"]
    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    ReadData = DataReader(
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
    History_wind_DA_error = Inputs["History_wind_DA_error"]
    History_solar_DA_error = Inputs["History_solar_DA_error"]
    SP_scenario = Inputs["SP_scenario"]
    RP_scenario = Inputs["RP_scenario"]
    probability = Inputs["probability"]
    N_price = Inputs["scenario_num"]
    N_RE = Inputs["N_Samples"]
    epsilon = Inputs["epsilon"]
    epsilon1 = Inputs["epsilon1"]

    History_RE_DA_error = History_wind_DA_error + History_solar_DA_error

    Samples_RE = pd.DataFrame(
        [np.linspace(History_RE_DA_error.min(), History_RE_DA_error.max(), num=N_RE)]
    ).T
    Samples_RE.index = range(N_RE)

    # Optimization modelling by cplex
    # setN_SP = [i for i in range(N_SP)]   # set of historical samples
    setN_RE = [i for i in range(N_RE)]
    setT = [i for i in range(T)]
    set_SoCT = [i for i in range(T + 1)]
    # setNTSP2 = [i for i in range(N_SP*T_BI*2)]
    # setNTRE2 = [i for i in range(N_RE*T*2)]
    setK = [i for i in range(24)]

    # epsilon1 = Wasserstein_radius(N_RE, Samples_RE)

    l_max_RE, l_min_RE = Uncertain_set(epsilon1, Samples_RE, N_RE)

    vertices = []
    for i in setN_RE:
        vertices.append([l_max_RE, l_min_RE, Samples_RE.iloc[i, 0]])
    vertices = np.array(vertices).reshape(3 * N_RE, 1)

    setTV = [
        (t, v, i)
        for t in range(T)
        for v in range(len(vertices))
        for i in range(N_price)
    ]
    set_SoCTV = [
        (t, v, i)
        for t in range(T + 1)
        for v in range(len(vertices))
        for i in range(N_price)
    ]
    setN_REP = [(v, i) for v in range(N_RE) for i in range(N_price)]

    BP_up_scenario = np.where(RP_scenario > SP_scenario, RP_scenario, SP_scenario)
    BP_dw_scenario = np.where(RP_scenario > SP_scenario, SP_scenario, RP_scenario)

    start_time = time.time()
    SMOpt_mdl = Model()
    # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=P_grid_limit, name="SM bidding 15min"
    )
    P_HPP_SM_k = SMOpt_mdl.continuous_var_dict(
        setK, lb=0, ub=P_grid_limit, name="SM bidding H"
    )
    # P_W_SM_t   = {t: SMOpt_mdl.continuous_var(lb=0, ub=DA_wind_forecast[t], name="SM Wind bidding {}".format(t)) for t in setT}
    # P_S_SM_t   = {t: SMOpt_mdl.continuous_var(lb=0, ub=DA_solar_forecast[t], name="DA Solar bidding {}".format(t)) for t in setT}
    P_W_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PwMax, name="wind power generation"
    )
    P_S_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PsMax, name="solar power generation"
    )
    P_dis_SM_t = SMOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="SM discharge"
    )
    P_cha_SM_t = SMOpt_mdl.continuous_var_dict(setT, lb=0, ub=PbMax, name="SM charge")
    # P_b_SM_t   = SMOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='SM Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    E_SM_t = SMOpt_mdl.continuous_var_dict(
        set_SoCT, lb=SoCmin * Emax, ub=Emax, name="SM SoC"
    )
    z_t = SMOpt_mdl.binary_var_dict(setT, lb=0, ub=1, name="Cha or Discha")

    Delta_P_dis_SM_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=-cplex.infinity, ub=cplex.infinity, name="delta SM discharge"
    )
    Delta_P_cha_SM_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=-cplex.infinity, ub=cplex.infinity, name="delta SM charge"
    )
    P_tilde_W_SM_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=PwMax, name="real wind power generation"
    )
    E_tilde_SM_t = SMOpt_mdl.continuous_var_dict(
        set_SoCTV, lb=SoCmin * Emax, ub=Emax, name="real SoC"
    )
    Delta_P_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=-cplex.infinity, ub=cplex.infinity, name="imbalance power"
    )
    Delta_P_up_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=cplex.infinity, name="positive imbalance power"
    )
    Delta_P_dw_t = SMOpt_mdl.continuous_var_dict(
        setTV, lb=0, ub=cplex.infinity, name="negative imbalance power"
    )

    # pho_2nt      = SMOpt_mdl.continuous_var_dict(setNTSP2, lb=0, name='Auxiliary variable variable 1')
    #    tau_n     = SMOpt_mdl.continuous_var_dict(setN_SP, lb=0, name='Auxiliary variable variable 1')
    v_n = SMOpt_mdl.continuous_var_dict(
        setN_REP, lb=0, name="Auxiliary variable variable 2"
    )
    #    pho_2nt      = SMOpt_mdl.addVars(setT_SP2,vtype=GRB.CONTINUOUS,  lb=0, name='Auxiliary variable variable 2')

    # au1 = SMOpt_mdl.continuous_var_dict(setTV, lb=0, name='Auxiliary variable variable 11')
    # au2 = SMOpt_mdl.continuous_var_dict(setTV, lb=0, name='Auxiliary variable variable 12')
    # obj1      = SMOpt_mdl.continuous_var_dict(setN_SP, lb=-cplex.infinity, ub=cplex.infinity, name='obj1')

    # lambda1      = SMOpt_mdl.continuous_var(lb=0, name='DRO variable 1')
    lambda2 = SMOpt_mdl.continuous_var(lb=0, name="DRO variable 2")

    # an_var     = SMOpt_mdl.continuous_var(lb=0, ub=0.5, name='anciliary var')
    # Define constraints
    for t in setT:
        SMOpt_mdl.add_constraint(
            P_HPP_SM_t[t] == P_W_SM_t[t] + P_S_SM_t[t] + P_dis_SM_t[t] - P_cha_SM_t[t]
        )
        SMOpt_mdl.add_constraint(P_dis_SM_t[t] <= (PbMax) * z_t[t])
        SMOpt_mdl.add_constraint(P_cha_SM_t[t] <= (PbMax) * (1 - z_t[t]))
        SMOpt_mdl.add_constraint(
            E_SM_t[t + 1]
            == E_SM_t[t] * (1 - eta_leak)
            - P_dis_SM_t[t] / eta_dis * dt
            + P_cha_SM_t[t] * eta_cha * dt
        )
        SMOpt_mdl.add_constraint(E_SM_t[t] <= Emax)
        SMOpt_mdl.add_constraint(E_SM_t[t] >= SoCmin * Emax)
        SMOpt_mdl.add_constraint(P_HPP_SM_t[t] <= P_grid_limit)

    for t in setT:
        for v in range(len(vertices)):
            for i in range(N_price):
                SMOpt_mdl.add_constraint(
                    P_dis_SM_t[t] - Delta_P_dis_SM_t[t, v, i] <= (PbMax)
                )
                SMOpt_mdl.add_constraint(
                    P_cha_SM_t[t] - Delta_P_cha_SM_t[t, v, i] <= (PbMax)
                )
                SMOpt_mdl.add_constraint(P_dis_SM_t[t] - Delta_P_dis_SM_t[t, v, i] >= 0)
                SMOpt_mdl.add_constraint(P_cha_SM_t[t] - Delta_P_cha_SM_t[t, v, i] >= 0)
                if DA_wind_forecast[t] - vertices[v, 0] < 0:
                    SMOpt_mdl.add_constraint(P_tilde_W_SM_t[t, v, i] <= 0)
                else:
                    SMOpt_mdl.add_constraint(
                        P_tilde_W_SM_t[t, v, i] <= DA_wind_forecast[t] - vertices[v, 0]
                    )
                SMOpt_mdl.add_constraint(P_tilde_W_SM_t[t, v, i] >= 0)
                SMOpt_mdl.add_constraint(P_tilde_W_SM_t[t, v, i] <= PwMax)
                SMOpt_mdl.add_constraint(
                    E_tilde_SM_t[t + 1, v, i]
                    == E_tilde_SM_t[t, v, i] * (1 - eta_leak)
                    - P_dis_SM_t[t] / eta_dis * dt
                    + P_cha_SM_t[t] * eta_cha * dt
                    + Delta_P_dis_SM_t[t, v, i] / eta_dis * dt
                    - Delta_P_cha_SM_t[t, v, i] * eta_cha * dt
                )
                SMOpt_mdl.add_constraint(E_tilde_SM_t[t + 1, v, i] <= Emax)
                SMOpt_mdl.add_constraint(E_tilde_SM_t[t + 1, v, i] >= SoCmin * Emax)
                SMOpt_mdl.add_constraint(
                    P_tilde_W_SM_t[t, v, i]
                    + P_dis_SM_t[t]
                    - P_cha_SM_t[t]
                    - Delta_P_dis_SM_t[t, v, i]
                    + Delta_P_cha_SM_t[t, v, i]
                    <= P_grid_limit
                )
                SMOpt_mdl.add_constraint(
                    Delta_P_t[t, v, i]
                    == P_tilde_W_SM_t[t, v, i]
                    - Delta_P_dis_SM_t[t, v, i]
                    + Delta_P_cha_SM_t[t, v, i]
                    - P_W_SM_t[t]
                )
                SMOpt_mdl.add_constraint(
                    Delta_P_t[t, v, i] == Delta_P_up_t[t, v, i] - Delta_P_dw_t[t, v, i]
                )
                if t == 0:
                    SMOpt_mdl.add_constraint(E_tilde_SM_t[0, v, i] == SoC0 * Emax)
            # SMOpt_mdl.add_constraint(au1[t+i*T] <= Delta_P_t[t+i*T])
            # SMOpt_mdl.add_constraint(-au1[t+i*T] <= Delta_P_t[t+i*T])
            # SMOpt_mdl.add_constraint(au1[t+i*T] >= Delta_P_t[t+i*T])
            # SMOpt_mdl.add_constraint(au1[t+i*T] >= -Delta_P_t[t+i*T])
            # SMOpt_mdl.add_constraint(Delta_P_t[t+i*T] == Delta_P_up_t[t+i*T] - Delta_P_dw_t[t+i*T])

    # for n in setN_SP:
    #    for t in setT_SP:
    #        temp = pd.DataFrame(error_C_SP.T[t,:])
    #        SMOpt_mdl.add_constraint(sum(temp.iloc[i,0] * pho_2nt[i + n*T_BI*2] for i in setT_SP2) - (P_HPP_SM_t[t*dt_num]) <= lambda1)
    #        SMOpt_mdl.add_constraint(-sum(temp.iloc[i,0] * pho_2nt[i + n*T_BI*2] for i in setT_SP2) + (P_HPP_SM_t[t*dt_num]) <= lambda1)

    # for n in setN_SP:
    #    temp = pd.DataFrame(error_d_SP - np.matmul(error_C_SP, Samples_SP.to_numpy()[n,:]))
    #    SMOpt_mdl.add_constraint(obj1[n] == sum(pho_2nt[i + n*T_BI*2] * temp.iloc[i,0] for i in setT_SP2))

    for v in range(len(vertices)):
        for ii in range(N_price):
            # SMOpt_mdl.add_constraint(sum(37*au1[t+n*T]*dt + mu*EBESS*ad*(P_dis_SM_t[t] - Delta_P_dis_SM_t[t+n*T] + P_cha_SM_t[t] - Delta_P_cha_SM_t[t+n*T])*dt for t in setT)  - lambda2*abs(vertices[n,0] - Samples_RE.iloc[int(n/3),0]) <= v_n[int(n/3)] )
            # SMOpt_mdl.add_constraint(sum(-(BM_dw_price_forecast1[t]*Delta_P_up_t[t+n*T]*dt - BM_up_price_forecast1[t]*Delta_P_dw_t[t+n*T]*dt) + mu*EBESS*ad*(P_dis_SM_t[t] - Delta_P_dis_SM_t[t+n*T] + P_cha_SM_t[t] - Delta_P_cha_SM_t[t+n*T]) for t in setT_SP)  - lambda2*abs(vertices[n,0] - Samples_RE.iloc[int(n/3),0]) <= v_n[int(n/3)] )
            # SMOpt_mdl.add_constraint(sum(-60*Delta_P_t[t+n*T]*dt + mu*EBESS*ad*(P_dis_SM_t[t] - Delta_P_dis_SM_t[t+n*T] + P_cha_SM_t[t] - Delta_P_cha_SM_t[t+n*T])*dt for t in setT)  - lambda2*abs(vertices[n,0] - Samples_RE.iloc[int(n/3),0]) <= v_n[int(n/3)] )
            SMOpt_mdl.add_constraint(
                sum(
                    -probability[ii]
                    * BP_dw_scenario[ii, int(t * dt)]
                    * Delta_P_up_t[t, v, ii]
                    * dt
                    + probability[ii]
                    * BP_up_scenario[ii, int(t * dt)]
                    * Delta_P_dw_t[t, v, ii]
                    * dt
                    + mu
                    * EBESS
                    * ad
                    * (
                        P_dis_SM_t[t]
                        - Delta_P_dis_SM_t[t, v, ii]
                        + P_cha_SM_t[t]
                        - Delta_P_cha_SM_t[t, v, ii]
                    )
                    * dt
                    for t in setT
                )
                - lambda2 * abs(vertices[v, 0] - Samples_RE.iloc[int(v / 3), 0])
                <= v_n[int(v / 3), ii]
            )

    for k in setK:
        for t in setT:
            if t // dt_num == k:
                SMOpt_mdl.add_constraint(P_HPP_SM_k[k] == P_HPP_SM_t[t])

    SMOpt_mdl.add_constraint(E_SM_t[0] == SoC0 * Emax)

    # Define objective function

    SM_revenue_minus = (
        sum(
            -probability[ii] * SP_scenario[ii, int(t * dt)] * P_HPP_SM_t[t] * dt
            for ii in range(N_price)
            for t in setT
        )
        + epsilon * lambda2
        + 1 / N_RE * sum(v_n[n, ii] for n in setN_RE for ii in range(N_price))
    )

    SMOpt_mdl.minimize(SM_revenue_minus)

    # SMOpt_mdl.parameters.mip.tolerances.mipgap=0.1;
    # SMOpt_mdl.parameters.timelimit=100;
    # SMOpt_mdl.print_information()
    # start_time = time.time()

    sol = SMOpt_mdl.solve()
    if verbose:
        SMOpt_mdl.print_information()

        aa = SMOpt_mdl.get_solve_details()
        print(aa.status)

    if sol:
        P_HPP_SM_t_opt = get_var_value_from_sol(P_HPP_SM_t, sol)
        P_HPP_SM_k_opt = get_var_value_from_sol(P_HPP_SM_k, sol)
        P_W_SM_t_opt = get_var_value_from_sol(P_W_SM_t, sol)
        P_S_SM_t_opt = get_var_value_from_sol(P_S_SM_t, sol)
        P_dis_SM_t_opt = get_var_value_from_sol(P_dis_SM_t, sol)
        P_cha_SM_t_opt = get_var_value_from_sol(P_cha_SM_t, sol)
        SoC_SM_t_opt = get_var_value_from_sol(E_SM_t, sol) / Emax
        # Delta_P_up_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(Delta_P_up_t), orient='index')
        # Delta_P_dw_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(Delta_P_dw_t), orient='index')
        # lambda1_opt = sol.get_value(lambda1)
    else:
        aa = SMOpt_mdl.get_solve_details()
        print(aa.status)

    E_HPP_SM_t_opt = P_HPP_SM_t_opt * dt
    P_W_SM_cur_t_opt = (
        np.array(DA_wind_forecast[:T].T) - np.array(P_W_SM_t_opt).flatten()
    )
    P_W_SM_cur_t_opt = pd.DataFrame(P_W_SM_cur_t_opt)
    P_S_SM_cur_t_opt = (
        np.array(DA_solar_forecast[:T].T) - np.array(P_S_SM_t_opt).flatten()
    )
    P_S_SM_cur_t_opt = pd.DataFrame(P_S_SM_cur_t_opt)
    obj = sol.get_objective_value()

    # print(P_HPP_SM_t_opt)
    # print(P_dis_SM_t_opt)
    # print(P_cha_SM_t_opt)

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
