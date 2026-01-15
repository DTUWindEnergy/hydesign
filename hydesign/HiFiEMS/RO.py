# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 14:57:55 2023

@author: ruzhu

"""
import os

os.environ["OMP_NUM_THREADS"] = "1"  # avoid warnning
import math
import time

import cplex
import numpy as np
import pandas as pd
from docplex.mp.model import Model
from scipy.linalg import fractional_matrix_power
from scipy.optimize import minimize

from hydesign.HiFiEMS.utils import DataReaderBase


class DataReader(DataReaderBase):
    def __init__(self, day_num, DI_num, T, PsMax, PwMax, simulation_dict):
        super().__init__(day_num, DI_num, T, PsMax, PwMax, simulation_dict)

    def execute(self):
        Inputs = super().execute()

        T0 = 96

        HA_wind_forecast_error_ub = (
            self.Wind_data[self.sim["wind_error_ub"]] * self.PwMax
        )
        HA_wind_forecast_error_lb = (
            self.Wind_data[self.sim["wind_error_lb"]] * self.PwMax
        )

        HA_wind_forecast_error_ub = HA_wind_forecast_error_ub[
            0 : T0 : int(4 / self.DI_num)
        ]
        HA_wind_forecast_error_ub.index = range(int(T0 / (4 / self.DI_num)))
        HA_wind_forecast_error_lb = HA_wind_forecast_error_lb[
            0 : T0 : int(4 / self.DI_num)
        ]
        HA_wind_forecast_error_lb.index = range(int(T0 / (4 / self.DI_num)))

        Inputs["HA_wind_forecast_error_ub"] = HA_wind_forecast_error_ub
        Inputs["HA_wind_forecast_error_lb"] = HA_wind_forecast_error_lb

        return Inputs


def get_var_value_from_sol(x, sol):

    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient="index")

    return y


def MasterPro(
    dt,
    ds,
    dk,
    T,
    EBESS,
    PbMax,
    PwMax,
    PUPMax,
    PDWMax,
    PUPMin,
    PDWMin,
    P_grid_limit,
    SoCmin,
    SoCmax,
    Emax,
    eta_dis,
    eta_cha,
    mu,
    ad,
    HA_wind_forecast,
    HA_solar_forecast,
    xi_tilde_s_opt,
    P_HPP_UP_t0,
    P_HPP_DW_t0,
    P_HPP_SM_t_opt,
    reg_forecast,
    SoC0,
    iter1,
    deg_indicator,
    z_hat1_s_opt,
    z_hat2_s_opt,
    C_dev,
    start,
    Cp,
):

    dt_num = int(1 / dt)  # DI

    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    ds_num = int(1 / ds)  # SI
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    eta_cha_ha = eta_cha
    eta_dis_ha = eta_dis

    if start == -1:
        setT = [i for i in range((start + 1) * dt_num, T)]
        setS = [i for i in range((start + 1) * ds_num, T_ds)]
        set_SoCT = [i for i in range((start + 1) * dt_num, T + 1)]
        start1 = 0
    else:
        setT = [i for i in range(start * dt_num, T)]
        setS = [i for i in range(start * ds_num, T_ds)]
        set_SoCT = [i for i in range(start * dt_num, T + 1)]
        start1 = start
    setT1 = [i for i in range((start + 1) * dt_num, T)]
    # setK = [i for i in range(start*dk_num, T_dk + int(exten_num/dt_num))]
    setK1 = [i for i in range((start + 1) * dk_num, T_dk)]

    reg_forecast = np.repeat(reg_forecast, dt_num)
    reg_forecast.index = range(T)

    MasterOpt_mdl = Model()

    # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_HPP_UP_k = MasterOpt_mdl.continuous_var_dict(
        setK1, lb=0, ub=min(P_grid_limit, PUPMax), name="BM UP bidding hour"
    )
    P_HPP_DW_k = MasterOpt_mdl.continuous_var_dict(
        setK1, lb=0, ub=min(P_grid_limit, PDWMax), name="BM DW bidding hour"
    )
    P_HPP_UP_t = MasterOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=min(P_grid_limit, PUPMax), name="BM UP bidding 15min"
    )
    P_HPP_DW_t = MasterOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=min(P_grid_limit, PDWMax), name="BM DW bidding 15min"
    )

    P_b_UP_t = MasterOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=PbMax, name="BM UP battery bidding 15min"
    )
    P_b_DW_t = MasterOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=PbMax, name="BM DW battery bidding 15min"
    )
    P_w_UP_t = MasterOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=PwMax, name="BM UP wind bidding 15min"
    )
    P_w_DW_t = MasterOpt_mdl.continuous_var_dict(
        setT1, lb=0, ub=PwMax, name="BM DW wind bidding 15min"
    )

    P_HPP_HA_t = MasterOpt_mdl.continuous_var_dict(
        setT, lb=0, name="HA schedule with balancing bidding"
    )
    P_w_HA_t = MasterOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PwMax, name="HA wind 15min"
    )
    P_dis_HA_t = MasterOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="HA discharge"
    )
    P_cha_HA_t = MasterOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=PbMax, name="HA charge"
    )
    # P_b_HA_s   = MasterOpt_mdl.continuous_var_dict(setT, lb=-PbMax, ub=PbMax, name='HA Battery schedule')  #(must define lb and ub, otherwise may cause unknown issues on cplex)
    E_HA_t = MasterOpt_mdl.continuous_var_dict(
        set_SoCT, lb=-cplex.infinity, ub=cplex.infinity, name="HA SoC"
    )
    z_t = MasterOpt_mdl.binary_var_dict(setT, name="Cha or Discha")

    z_UP_bidlimit_t = MasterOpt_mdl.binary_var_dict(setT1, name="up >5 MW or 0 MW")
    z_DW_bidlimit_t = MasterOpt_mdl.binary_var_dict(setT1, name="dw >5 MW or 0 MW")

    eta = MasterOpt_mdl.continuous_var(
        lb=-cplex.infinity, ub=cplex.infinity, name="obj"
    )

    # Define constraints
    for t in setT:
        MasterOpt_mdl.add_constraint(
            P_HPP_HA_t[t] == P_w_HA_t[t] + P_dis_HA_t[t] - P_cha_HA_t[t]
        )
        MasterOpt_mdl.add_constraint(P_HPP_HA_t[t] == P_HPP_SM_t_opt.loc[t].iloc[0])
        # MasterOpt_mdl.add_constraint(P_HPP_HA_t[t] == 1)
        # MasterOpt_mdl.add_constraint(P_b_HA_s[t]   == P_dis_HA_s[t] - P_cHA_HA_s[t])
        MasterOpt_mdl.add_constraint(P_dis_HA_t[t] <= (PbMax) * z_t[t])
        MasterOpt_mdl.add_constraint(P_cha_HA_t[t] <= (PbMax) * (1 - z_t[t]))

        MasterOpt_mdl.add_constraint(
            E_HA_t[t + 1]
            == E_HA_t[t]
            - (P_dis_HA_t[t]) / eta_dis_ha * dt
            + (P_cha_HA_t[t]) * eta_cha_ha * dt
        )
        MasterOpt_mdl.add_constraint(E_HA_t[t + 1] <= SoCmax * Emax)
        MasterOpt_mdl.add_constraint(E_HA_t[t + 1] >= SoCmin * Emax)

        if t < (start + 1) * dt_num:
            MasterOpt_mdl.add_constraint(
                P_HPP_HA_t[t] + P_HPP_UP_t0[t - start * dt_num] <= P_grid_limit
            )
            MasterOpt_mdl.add_constraint(
                P_HPP_HA_t[t] - P_HPP_DW_t0[t - start * dt_num] >= 0
            )
        else:
            MasterOpt_mdl.add_constraint(P_HPP_HA_t[t] + P_HPP_UP_t[t] <= P_grid_limit)
            MasterOpt_mdl.add_constraint(P_HPP_HA_t[t] - P_HPP_DW_t[t] >= 0)
    for t in setT1:
        MasterOpt_mdl.add_constraint(P_w_HA_t[t] + P_w_UP_t[t] <= PwMax)
        MasterOpt_mdl.add_constraint(P_w_HA_t[t] - P_w_DW_t[t] >= 0)
        MasterOpt_mdl.add_constraint(
            P_dis_HA_t[t] - P_cha_HA_t[t] + P_b_UP_t[t] <= PbMax
        )
        MasterOpt_mdl.add_constraint(
            P_dis_HA_t[t] - P_cha_HA_t[t] - P_b_DW_t[t] >= -PbMax
        )

        MasterOpt_mdl.add_constraint(P_HPP_UP_t[t] == P_w_UP_t[t] + P_b_UP_t[t])
        MasterOpt_mdl.add_constraint(P_HPP_DW_t[t] == P_w_DW_t[t] + P_b_DW_t[t])

        MasterOpt_mdl.add_constraint(P_HPP_UP_t[t] <= PUPMax * z_UP_bidlimit_t[t])
        MasterOpt_mdl.add_constraint(P_HPP_UP_t[t] >= -PUPMax * z_UP_bidlimit_t[t])
        MasterOpt_mdl.add_constraint(
            P_HPP_UP_t[t] >= PUPMin - PUPMax * (1 - z_UP_bidlimit_t[t])
        )
        # MasterOpt_mdl.add_constraint(z_UP_bidlimit_t[t] == 1)

        MasterOpt_mdl.add_constraint(P_HPP_DW_t[t] <= PDWMax * z_DW_bidlimit_t[t])
        MasterOpt_mdl.add_constraint(P_HPP_DW_t[t] >= -PDWMax * z_DW_bidlimit_t[t])
        MasterOpt_mdl.add_constraint(
            P_HPP_DW_t[t] >= PDWMin - PDWMax * (1 - z_DW_bidlimit_t[t])
        )

        MasterOpt_mdl.add_constraint(z_UP_bidlimit_t[t] + z_DW_bidlimit_t[t] <= 1)
        # MasterOpt_mdl.add_constraint(z_DW_bidlimit_t[t] == 1)
        # if reg_forecast[int(t/dt_num)] >= SM_price_cleared[int(t/dt_num)]:
        #     MasterOpt_mdl.add_constraint(P_HPP_DW_t[t] == 0)
        # elif reg_forecast[int(t/dt_num)] <= SM_price_cleared[int(t/dt_num)]:
        #     MasterOpt_mdl.add_constraint(P_HPP_UP_t[t] == 0)
        # MasterOpt_mdl.add_constraint(P_HPP_DW_t[t] == 0)

    # MasterOpt_mdl.add_constraint(MasterOpt_mdl.sum(z_UP_bidlimit_t[t] for t in setT1)==5  )
    # for k in setK1:
    #    for j in range(dt_num):
    #        MasterOpt_mdl.add_constraint(P_HPP_UP_t[k * dt_num + j] == P_HPP_UP_k[k])
    #        MasterOpt_mdl.add_constraint(P_HPP_DW_t[k * dt_num + j] == P_HPP_DW_k[k])

    if start == -1:
        MasterOpt_mdl.add_constraint(E_HA_t[0] == SoC0 * Emax)
    else:
        MasterOpt_mdl.add_constraint(E_HA_t[start * dt_num] == SoC0 * Emax)

    # MasterOpt_mdl.add_constraint(P_HPP_UP_s[23] == 5)
    # MasterOpt_mdl.add_constraint(P_HPP_DW_s[20] == 5)

    # added variables and constriants
    if iter1 > 0:
        setV = [i for i in range(iter1)]
        setTV = [(i, j) for i in setT for j in setV]
        # setTV1 = [i for i in range(start*dt_num*iter1, (T-(start+1)*dt_num) *iter1)]
        set_SoCTV = [(i, j) for i in set_SoCT for j in setV]

        setTV1 = [(i, j) for i in setT1 for j in setV]

        P_tilde_dis_HA_t = MasterOpt_mdl.continuous_var_dict(
            setTV, lb=0, ub=cplex.infinity, name="RT discharge"
        )
        # P_tilde_b_HA_aux_t = MasterOpt_mdl.continuous_var_dict(setTV, lb=-cplex.infinity, ub=cplex.infinity, name='RT battery aux')
        P_tilde_cha_HA_t = MasterOpt_mdl.continuous_var_dict(
            setTV, lb=0, ub=cplex.infinity, name="RT charge"
        )
        # delta_tilde_P_b_HA_s = MasterOpt_mdl.continuous_var_dict(setTV, lb=-cplex.infinity, ub=cplex.infinity, name='RT cHArge')
        P_tilde_w_HA_t = MasterOpt_mdl.continuous_var_dict(
            setTV, lb=0, ub=PwMax, name="RT wind 15min"
        )  # (must define lb and ub, otherwise may cause unknown issues on cplex)

        # delta_tilde_P_HPP_t = MasterOpt_mdl.continuous_var_dict(setTV, lb=-cplex.infinity, ub=cplex.infinity, name='RT imbalance')

        delta_tilde_P_HPP_UP_t = MasterOpt_mdl.continuous_var_dict(
            setTV, lb=0, ub=cplex.infinity, name="RT imbalance up"
        )
        delta_tilde_P_HPP_DW_t = MasterOpt_mdl.continuous_var_dict(
            setTV, lb=0, ub=cplex.infinity, name="RT imbalance dw"
        )

        # tau_t = MasterOpt_mdl.continuous_var_dict(setTV, lb=-cplex.infinity, ub=cplex.infinity, name='RT special imbalance aux')

        w_tilde_t = MasterOpt_mdl.continuous_var_dict(
            setTV1, lb=-cplex.infinity, ub=cplex.infinity, name="aux penalty"
        )

        P_new_tilde_HPP_UP_t = MasterOpt_mdl.continuous_var_dict(
            setTV1, lb=0, ub=cplex.infinity, name="new up before projection"
        )
        P_new_tilde_HPP_DW_t = MasterOpt_mdl.continuous_var_dict(
            setTV1, lb=0, ub=cplex.infinity, name="new dw before projection"
        )

        P_projected_tilde_HPP_UP_t = MasterOpt_mdl.continuous_var_dict(
            setTV1, lb=0, ub=cplex.infinity, name="projected up"
        )
        P_projected_tilde_HPP_DW_t = MasterOpt_mdl.continuous_var_dict(
            setTV1, lb=0, ub=cplex.infinity, name="projected dw"
        )

        for v in setV:
            for t in setT:
                MasterOpt_mdl.add_constraint(P_tilde_dis_HA_t[t, v] <= PbMax)
                MasterOpt_mdl.add_constraint(P_tilde_cha_HA_t[t, v] <= PbMax)

                # MasterOpt_mdl.add_constraint(E_tilde_SM_t[t+1,v] == E_tilde_SM_t[t,v] * (1-eta_leak_ha) - (P_dis_SM_t[t])/eta_dis_ha * dt + (P_cha_SM_t[t]) * eta_cha_ha * dt + delta_tilde_P_b_SM_t[t,v] * dt )
                MasterOpt_mdl.add_constraint(
                    SoC0 * Emax
                    - MasterOpt_mdl.sum(
                        (
                            P_tilde_dis_HA_t[i, v] / eta_dis_ha
                            - P_tilde_cha_HA_t[i, v] * eta_cha_ha
                        )
                        * dt
                        for i in range(start1 * dt_num, t + 1)
                    )
                    <= SoCmax * Emax
                )
                MasterOpt_mdl.add_constraint(
                    SoC0 * Emax
                    - MasterOpt_mdl.sum(
                        (
                            P_tilde_dis_HA_t[i, v] / eta_dis_ha
                            - P_tilde_cha_HA_t[i, v] * eta_cha_ha
                        )
                        * dt
                        for i in range(start1 * dt_num, t + 1)
                    )
                    >= SoCmin * Emax
                )

                if t < (start + 1) * dt_num:
                    MasterOpt_mdl.add_constraint(
                        delta_tilde_P_HPP_UP_t[t, v] - delta_tilde_P_HPP_DW_t[t, v]
                        == P_tilde_w_HA_t[t, v]
                        + (P_tilde_dis_HA_t[t, v] - P_tilde_cha_HA_t[t, v])
                        - P_HPP_SM_t_opt.loc[t].iloc[0]
                        - P_HPP_UP_t0[t - start1 * dt_num]
                        + P_HPP_DW_t0[t - start1 * dt_num]
                    )
                else:
                    MasterOpt_mdl.add_constraint(
                        delta_tilde_P_HPP_UP_t[t, v] - delta_tilde_P_HPP_DW_t[t, v]
                        == P_tilde_w_HA_t[t, v]
                        + (P_tilde_dis_HA_t[t, v] - P_tilde_cha_HA_t[t, v])
                        - P_HPP_SM_t_opt.loc[t].iloc[0]
                        - P_projected_tilde_HPP_UP_t[t, v]
                        + P_projected_tilde_HPP_DW_t[t, v]
                    )

                    # MasterOpt_mdl.add_constraint(delta_tilde_P_HPP_t[t,v] <= 3*P_grid_limit*(1-z_hat1_s_opt.loc[t].iloc[v]) + 3 )
                    # MasterOpt_mdl.add_constraint(delta_tilde_P_HPP_t[t,v] >= -3*P_grid_limit*(1-z_hat1_s_opt.loc[t].iloc[v]) - 3 )
                # MasterOpt_mdl.add_constraint(delta_tilde_P_HPP_t[t,v] == delta_tilde_P_HPP_UP_t[t,v] - delta_tilde_P_HPP_DW_t[t,v])
                # MasterOpt_mdl.add_constraint(delta_tilde_P_HPP_UP_t[t,v] <= z_delta_tilde_t[t,v] * P_grid_limit)
                # MasterOpt_mdl.add_constraint(delta_tilde_P_HPP_DW_t[t,v] <= (1 - z_delta_tilde_t[t,v]) * P_grid_limit)

                MasterOpt_mdl.add_constraint(
                    P_tilde_w_HA_t[t, v]
                    + (P_tilde_dis_HA_t[t, v] - P_tilde_cha_HA_t[t, v])
                    <= P_grid_limit
                )
                MasterOpt_mdl.add_constraint(
                    P_tilde_w_HA_t[t, v]
                    + (P_tilde_dis_HA_t[t, v] - P_tilde_cha_HA_t[t, v])
                    >= 0
                )

                MasterOpt_mdl.add_constraint(
                    P_tilde_w_HA_t[t, v]
                    <= HA_wind_forecast[t] - xi_tilde_s_opt.loc[t].iloc[v]
                )

                # if t < (start + 1) * dt_num:
                #   MasterOpt_mdl.add_constraint(delta_tilde_P_special_HPP_t[t,v] + P_HPP_UP_t0 - P_HPP_DW_t0 == P_tilde_w_HA_t[t,v] + P_tilde_HA_dis_t[t,v]  - P_tilde_HA_cha_t[t,v]  - P_HPP_HA_t[t])

                # else:
                #   MasterOpt_mdl.add_constraint(delta_tilde_P_special_HPP_t[t,v] + P_projected_tilde_HPP_UP_t[t,v] - P_projected_tilde_HPP_DW_t[t,v] == P_tilde_w_HA_t[t,v] + P_tilde_HA_dis_t[t,v]  - P_tilde_HA_cha_t[t,v]  - P_HPP_HA_t[t])

                # MasterOpt_mdl.add_constraint(tau_t[t,v] >= delta_tilde_P_HPP_t[t,v])
                # MasterOpt_mdl.add_constraint(tau_t[t,v] >= -delta_tilde_P_HPP_t[t,v])

            for t in setT1:
                MasterOpt_mdl.add_constraint(w_tilde_t[t, v] >= 0)
                MasterOpt_mdl.add_constraint(
                    w_tilde_t[t, v]
                    >= delta_tilde_P_HPP_UP_t[t, v]
                    + delta_tilde_P_HPP_DW_t[t, v]
                    - P_grid_limit * (1 - z_UP_bidlimit_t[t])
                    - P_grid_limit * (1 - z_hat1_s_opt.loc[t].iloc[v])
                )
                MasterOpt_mdl.add_constraint(
                    w_tilde_t[t, v]
                    >= delta_tilde_P_HPP_UP_t[t, v]
                    + delta_tilde_P_HPP_DW_t[t, v]
                    - P_grid_limit * (1 - z_DW_bidlimit_t[t])
                    - P_grid_limit * (1 - z_hat2_s_opt.loc[t].iloc[v])
                )

        # DDU mapping

        for v in setV:

            # up
            num_of_active_cons = 0
            inactive_set = list()
            for t in setT1:

                MasterOpt_mdl.add_constraint(
                    P_new_tilde_HPP_UP_t[t, v]
                    == P_HPP_UP_t[t] * z_hat1_s_opt.loc[t].iloc[v]
                )
                MasterOpt_mdl.add_constraint(
                    P_new_tilde_HPP_DW_t[t, v]
                    == P_HPP_DW_t[t] * z_hat2_s_opt.loc[t].iloc[v]
                )
            print("full rank")

            # projection
            for t in setT1:
                MasterOpt_mdl.add_constraint(
                    P_projected_tilde_HPP_UP_t[t, v] == P_new_tilde_HPP_UP_t[t, v]
                )
                MasterOpt_mdl.add_constraint(
                    P_projected_tilde_HPP_DW_t[t, v] == P_new_tilde_HPP_DW_t[t, v]
                )

        for v in setV:
            MasterOpt_mdl.add_constraint(
                eta
                >= MasterOpt_mdl.sum(
                    -reg_forecast[t] * P_projected_tilde_HPP_UP_t[t, v] * dt
                    + reg_forecast[t] * P_projected_tilde_HPP_DW_t[t, v] * dt
                    for t in setT1
                )
                - MasterOpt_mdl.sum(
                    reg_forecast[t]
                    * (delta_tilde_P_HPP_UP_t[t, v] - delta_tilde_P_HPP_DW_t[t, v])
                    * ds
                    for t in setS
                )
                + MasterOpt_mdl.sum(
                    ad
                    * EBESS
                    * mu
                    * (P_tilde_dis_HA_t[t, v] + P_tilde_cha_HA_t[t, v])
                    * dt
                    for t in setT
                )
                + MasterOpt_mdl.sum(
                    C_dev
                    * (delta_tilde_P_HPP_UP_t[t, v] + delta_tilde_P_HPP_DW_t[t, v])
                    * dt
                    for t in setT
                )
                + MasterOpt_mdl.sum(Cp * (w_tilde_t[t, v]) * dt for t in setT1)
                + MasterOpt_mdl.sum(
                    (
                        0
                        * max(reg_forecast)
                        * math.exp(-(t - start1) / (2 * EBESS / PbMax))
                    )
                    * (delta_tilde_P_HPP_DW_t[t, v])
                    * dt
                    for t in setT
                )
            )
    else:
        MasterOpt_mdl.add_constraint(eta >= -1e6)
        # for t in setT1:
        #    MasterOpt_mdl.add_constraint(P_w_SM_s[t] + P_w_UP_s[t] <= SM_wind_forecast[t])

    MasterOpt_mdl.minimize(eta)

    # Solve MasterOpt Model
    MasterOpt_mdl.print_information()
    sol = MasterOpt_mdl.solve()
    aa = MasterOpt_mdl.get_solve_details()
    print(aa.status)
    if sol:
        if iter1 > 0:
            # P_tilde_b_HA_t_opt           = get_var_value_from_sol(P_tilde_b_HA_t, sol)
            P_tilde_HA_dis_t_opt = get_var_value_from_sol(P_tilde_dis_HA_t, sol)
            P_tilde_HA_cha_t_opt = get_var_value_from_sol(P_tilde_cha_HA_t, sol)
            P_tilde_w_HA_t_opt = get_var_value_from_sol(P_tilde_w_HA_t, sol)
            # delta_tilde_P_HPP_t_opt     = get_var_value_from_sol(delta_tilde_P_HPP_t, sol)
            # delta_tilde_P_HPP_UP_t_opt     = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt>0, 0)
            # delta_tilde_P_HPP_DW_t_opt     = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt<0, 0)
            delta_tilde_P_HPP_UP_t_opt = get_var_value_from_sol(
                delta_tilde_P_HPP_UP_t, sol
            )
            delta_tilde_P_HPP_DW_t_opt = get_var_value_from_sol(
                delta_tilde_P_HPP_DW_t, sol
            )
            # delta_tilde_P_special_HPP_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_tilde_P_special_HPP_t), orient='index')
            # tau_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(tau_t), orient='index')
            P_projected_tilde_HPP_UP_t_opt = get_var_value_from_sol(
                P_projected_tilde_HPP_UP_t, sol
            )
            P_projected_tilde_HPP_DW_t_opt = get_var_value_from_sol(
                P_projected_tilde_HPP_DW_t, sol
            )

        P_HPP_UP_k_opt = get_var_value_from_sol(P_HPP_UP_k, sol)
        P_HPP_DW_k_opt = get_var_value_from_sol(P_HPP_DW_k, sol)
        P_b_UP_t_opt = get_var_value_from_sol(P_b_UP_t, sol)
        P_b_DW_t_opt = get_var_value_from_sol(P_b_DW_t, sol)
        P_w_UP_t_opt = get_var_value_from_sol(P_w_UP_t, sol)
        P_w_DW_t_opt = get_var_value_from_sol(P_w_DW_t, sol)
        P_HPP_UP_t_opt = get_var_value_from_sol(P_HPP_UP_t, sol)
        P_HPP_DW_t_opt = get_var_value_from_sol(P_HPP_DW_t, sol)
        P_dis_HA_t_opt = get_var_value_from_sol(P_dis_HA_t, sol)
        P_cha_HA_t_opt = get_var_value_from_sol(P_cha_HA_t, sol)
        P_w_HA_t_opt = get_var_value_from_sol(P_w_HA_t, sol)
        P_HPP_HA_t_opt = get_var_value_from_sol(P_HPP_HA_t, sol)
        E_HA_t_opt = get_var_value_from_sol(E_HA_t, sol)
        z_UP_bidlimit_t_opt = get_var_value_from_sol(z_UP_bidlimit_t, sol)
        z_DW_bidlimit_t_opt = get_var_value_from_sol(z_DW_bidlimit_t, sol)
        obj = sol.get_objective_value()

    # E = np.zeros(T+1)
    # E[0] = SoC0*Emax
    # Pb = P_dis_SM_t_opt - P_cha_SM_t_opt
    # for t in setT:
    #     if Pb.iloc[t,0]>=0:
    #        E[t+1] = E[t] -  Pb.iloc[t,0]/eta_dis_ha*dt
    #     else:
    #        E[t+1] = E[t] -  Pb.iloc[t,0]*eta_cha_ha*dt

    #     if E[t+1]>Emax or  E[t+1]<SoCmin*Emax:
    #        print("Infeasible battery operation")
    #        break
    return (
        E_HA_t_opt,
        P_HPP_UP_t_opt,
        P_HPP_DW_t_opt,
        P_HPP_UP_k_opt,
        P_HPP_DW_k_opt,
        P_b_UP_t_opt,
        P_b_DW_t_opt,
        P_w_UP_t_opt,
        P_w_DW_t_opt,
        P_HPP_HA_t_opt,
        P_dis_HA_t_opt,
        P_cha_HA_t_opt,
        P_w_HA_t_opt,
        obj,
        z_UP_bidlimit_t_opt,
        z_DW_bidlimit_t_opt,
    )


def SubDualPro(
    dt,
    ds,
    dk,
    T,
    EBESS,
    PbMax,
    PwMax,
    PreUp,
    PreDw,
    P_grid_limit,
    SoCmin,
    SoCmax,
    Emax,
    eta_dis,
    eta_cha,
    eta_leak,
    mu,
    ad,
    HA_wind_forecast,
    P_HPP_HA_t_opt,
    P_HPP_SM_t_opt,
    P_HPP_UP_t0,
    P_HPP_DW_t0,
    SoC0,
    exten_num,
    deg_indicator,
    probability,
    BP_up_forecast,
    BP_dw_forecast,
    reg_forecast,
    SM_price_cleared,
    Cp,
    start,
    P_HPP_UP_t_opt,
    P_HPP_DW_t_opt,
    xi_max,
    xi_min,
    C_dev,
    error_C,
    error_d,
    z_UP_bidlimit_t_opt,
    z_DW_bidlimit_t_opt,
    time_limit,
    z_hat1_t_opt,
    z_hat2_t_opt,
    xi_tilde_t_opt,
):

    dt_num = int(1 / dt)  # DI

    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    ds_num = int(1 / ds)  # SI
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    eta_cha_ha = eta_cha
    eta_dis_ha = eta_dis
    # eta_cha_ha = 1
    # eta_dis_ha = 1

    if start == -1:
        setT = [i for i in range((start + 1) * dt_num, T + exten_num)]
        setS = [
            i for i in range((start + 1) * ds_num, T_ds + int(exten_num / dsdt_num))
        ]
        set_SoCT = [i for i in range((start + 1) * dt_num, T + 1 + exten_num)]

        start1 = 0
    else:
        setT = [i for i in range(start * dt_num, T + exten_num)]
        setS = [i for i in range(start * ds_num, T_ds + int(exten_num / dsdt_num))]
        set_SoCT = [i for i in range(start * dt_num, T + 1 + exten_num)]

        start1 = start
    setT1 = [i for i in range((start + 1) * dt_num, T + exten_num)]
    # setK = [i for i in range(start*dk_num, T_dk + int(exten_num/dt_num))]
    # setK1 = [i for i in range((start + 1) * dk_num, T_dk + int(exten_num/dt_num))]

    setT1_up = [
        t for t in setT1 if math.isclose(z_UP_bidlimit_t_opt.loc[t, 0], 1, abs_tol=1e-3)
    ]
    setT1_dw = [
        t for t in setT1 if math.isclose(z_DW_bidlimit_t_opt.loc[t, 0], 1, abs_tol=1e-3)
    ]
    setT1_no = [t for t in setT1 if t not in setT1_up and t not in setT1_dw]

    reg_forecast = np.repeat(reg_forecast, dt_num)
    reg_forecast.index = range(T)
    SubOpt_mdl = Model()

    # xi_tilde_t[0].start = xi_max[0]

    alpha1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="alpha1")
    alpha2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="alpha2")
    beta1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="beta1")
    beta2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="beta2")
    # gamma_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, name='gamma')
    m_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="m")
    n_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="n")
    l1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="l1")
    l2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="l2")
    mu_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="mu")
    v_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="v")
    r_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, name="r")
    p_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="p")
    q_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="q")

    theta1_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="theta1")
    omega1_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="omega1")
    omega2_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="omega2")

    ## dual constraints
    for t in setT:
        SubOpt_mdl.add_constraint(
            -alpha1_t[t]
            + alpha2_t[t]
            + l1_t[t]
            - l2_t[t]
            - r_t[t]
            - SubOpt_mdl.sum(m_t[i] for i in range(t, T)) / eta_dis_ha * dt
            + SubOpt_mdl.sum(n_t[i] for i in range(t, T)) / eta_dis_ha * dt
            + ad * EBESS * mu * dt
            == 0
        )  # P_dis
        SubOpt_mdl.add_constraint(
            -beta1_t[t]
            + beta2_t[t]
            - l1_t[t]
            + l2_t[t]
            + r_t[t]
            + SubOpt_mdl.sum(m_t[i] for i in range(t, T)) * eta_cha_ha * dt
            - SubOpt_mdl.sum(n_t[i] for i in range(t, T)) * eta_cha_ha * dt
            + ad * EBESS * mu * dt
            == 0
        )  # P_b_aux
        SubOpt_mdl.add_constraint(
            l1_t[t] - l2_t[t] - mu_t[t] + v_t[t] - r_t[t] == 0
        )  # P_w
        # if t < (start + 1) * dt_num:
        SubOpt_mdl.add_constraint(
            -(reg_forecast[t]) * dt + r_t[t] + p_t[t] - q_t[t] == 0
        )  # delta_P
        # else:
        #   SubOpt_mdl.add_constraint(-(reg_forecast[t])*dt + r_t[t] + p_t[t] - q_t[t] + omega1_t[t] - omega2_t[t]  == 0)

        if t < (start + 1) * dt_num:
            SubOpt_mdl.add_constraint(-p_t[t] - q_t[t] + C_dev * dt == 0)  # tau
        else:

            # if t in setT1_no:
            SubOpt_mdl.add_constraint(
                -p_t[t] - q_t[t] + C_dev * dt + omega1_t[t] + omega2_t[t] == 0
            )  # tau
            SubOpt_mdl.add_constraint(
                -theta1_t[t] - omega1_t[t] - omega2_t[t] + Cp * dt == 0
            )  # w

            # SubOpt_mdl.add_constraint(y1_t[t] + bigM[0]*0.9 * dt == 0) # w_tilde_up_t
            # SubOpt_mdl.add_constraint(y2_t[t] + bigM[0]*0.9 * dt == 0) # w_tilde_dw_t

            #
            # if t in setT1_up:

            # SubOpt_mdl.add_constraint(- p_t[t] - q_t[t] + C_dev*dt - theta2_t[t] == 0)   # tau
            # SubOpt_mdl.add_constraint(- p_t[t] - q_t[t] + C_dev*dt + omega1_t[t]  == 0)
            # SubOpt_mdl.add_constraint( - theta1_t[t] - omega1_t[t] + bigM[0]*0.9 * dt== 0) # w_tilde_up_t
            # SubOpt_mdl.add_constraint(y2_t[t] + bigM[0]*0.9 * dt == 0) # w_tilde_dw_t

            # if t in setT1_dw:
            # SubOpt_mdl.add_constraint(- p_t[t] - q_t[t] + C_dev*dt + omega2_t[t]  == 0)
            # SubOpt_mdl.add_constraint( - theta1_t[t] - omega2_t[t] + bigM[0]*0.9 * dt== 0) # w_tilde_up_t
            # SubOpt_mdl.add_constraint(- p_t[t] - q_t[t] + C_dev*dt - omega2_t[t] == 0)   # tau
            # SubOpt_mdl.add_constraint( - omega1_t[t] - omega2_t[t] + bigM[0]*0.9 * dt == 0) # w_tilde_dw_t
            # SubOpt_mdl.add_constraint(y1_t[t] + bigM[0]*0.9 * dt == 0) # w_tilde_up_t
    ## aux constriants

    ## dual constraints and complementary slackness condition of uncertainty set

    # for t in setT1:
    #     SubOpt_mdl.add_constraint(f1_tilde_t[t] >= -bigM[t]*(z_hat1_t_opt.loc[t,0]) )
    #     SubOpt_mdl.add_constraint(f1_tilde_t[t] <= bigM[t]*(z_hat1_t_opt.loc[t,0]) )
    #     SubOpt_mdl.add_constraint(-bigM[t]*(1-z_hat1_t_opt.loc[t,0]) + r_t[t]<=f1_tilde_t[t])
    #     SubOpt_mdl.add_constraint(f1_tilde_t[t] <= bigM[t]*(1-z_hat1_t_opt.loc[t,0]) + r_t[t])

    #     SubOpt_mdl.add_constraint(f2_tilde_t[t] >= -bigM[t]*(z_hat2_t_opt.loc[t,0]))
    #     SubOpt_mdl.add_constraint(f2_tilde_t[t] <= bigM[t]*(z_hat2_t_opt.loc[t,0]))
    #     SubOpt_mdl.add_constraint(-bigM[t]*(1-z_hat2_t_opt.loc[t,0]) + r_t[t]<=f2_tilde_t[t])
    #     SubOpt_mdl.add_constraint(f2_tilde_t[t] <= bigM[t]*(1-z_hat2_t_opt.loc[t,0]) + r_t[t])

    #     SubOpt_mdl.add_constraint(f3_tilde_t[t] >= -bigM[t]*(z_hat1_t_opt.loc[t,0]) )
    #     SubOpt_mdl.add_constraint(f3_tilde_t[t] <= bigM[t]*(z_hat1_t_opt.loc[t,0]) )
    #     SubOpt_mdl.add_constraint(-bigM[t]*(1-z_hat1_t_opt.loc[t,0]) + omega1_t[t]<=f3_tilde_t[t])
    #     SubOpt_mdl.add_constraint(f3_tilde_t[t] <= bigM[t]*(1-z_hat1_t_opt.loc[t,0]) + omega1_t[t])

    #     SubOpt_mdl.add_constraint(f4_tilde_t[t] >= -bigM[t]*(z_hat2_t_opt.loc[t,0]))
    #     SubOpt_mdl.add_constraint(f4_tilde_t[t] <= bigM[t]*(z_hat2_t_opt.loc[t,0]))
    #     SubOpt_mdl.add_constraint(-bigM[t]*(1-z_hat2_t_opt.loc[t,0]) + omega2_t[t]<=f4_tilde_t[t])
    #     SubOpt_mdl.add_constraint(f4_tilde_t[t] <= bigM[t]*(1-z_hat2_t_opt.loc[t,0]) + omega2_t[t])

    if start == -1:
        minus_profit = (
            SubOpt_mdl.sum(
                -reg_forecast[t]
                * P_HPP_UP_t_opt.loc[t].iloc[0]
                * z_hat1_t_opt.loc[t, 0]
                * dt
                + reg_forecast[t]
                * P_HPP_DW_t_opt.loc[t].iloc[0]
                * z_hat2_t_opt.loc[t, 0]
                * dt
                for t in setT1
            )
            + SubOpt_mdl.sum(
                -PbMax * alpha2_t[t]
                - PbMax * beta2_t[t]
                + n_t[t] * (SoCmin * Emax - SoC0 * Emax)
                + m_t[t] * (SoC0 * Emax - SoCmax * Emax)
                - v_t[t] * HA_wind_forecast[t]
                + v_t[t] * xi_tilde_t_opt.loc[t].iloc[0]
                - l1_t[t] * P_grid_limit
                + r_t[t] * P_HPP_SM_t_opt.loc[t].iloc[0]
                for t in setT
            )
            + SubOpt_mdl.sum(
                P_HPP_UP_t_opt.loc[t].iloc[0] * r_t[t] * z_hat1_t_opt.loc[t, 0]
                - P_HPP_DW_t_opt.loc[t].iloc[0] * r_t[t] * z_hat2_t_opt.loc[t, 0]
                for t in setT1
            )
            - SubOpt_mdl.sum(
                omega1_t[t] * P_grid_limit * (1 - z_UP_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega1_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(
                omega1_t[t] * P_grid_limit * z_hat1_t_opt.loc[t, 0] for t in setT1
            )
            - SubOpt_mdl.sum(
                omega2_t[t] * P_grid_limit * (1 - z_DW_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega2_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(
                omega2_t[t] * P_grid_limit * z_hat2_t_opt.loc[t, 0] for t in setT1
            )
        )
    else:
        minus_profit = (
            SubOpt_mdl.sum(
                -reg_forecast[t]
                * P_HPP_UP_t_opt.loc[t].iloc[0]
                * z_hat1_t_opt.loc[t, 0]
                * dt
                + reg_forecast[t]
                * P_HPP_DW_t_opt.loc[t].iloc[0]
                * z_hat2_t_opt.loc[t, 0]
                * dt
                for t in setT1
            )
            + SubOpt_mdl.sum(
                -PbMax * alpha2_t[t]
                - PbMax * beta2_t[t]
                + n_t[t] * (SoCmin * Emax - SoC0 * Emax)
                + m_t[t] * (SoC0 * Emax - SoCmax * Emax)
                - v_t[t] * HA_wind_forecast[t]
                + v_t[t] * xi_tilde_t_opt.loc[t].iloc[0]
                - l1_t[t] * P_grid_limit
                + r_t[t] * P_HPP_SM_t_opt.loc[t].iloc[0]
                for t in setT
            )
            + SubOpt_mdl.sum(
                P_HPP_UP_t_opt.loc[t].iloc[0] * r_t[t] * z_hat1_t_opt.loc[t, 0]
                - P_HPP_DW_t_opt.loc[t].iloc[0] * r_t[t] * z_hat2_t_opt.loc[t, 0]
                for t in setT1
            )
            + SubOpt_mdl.sum(
                P_HPP_UP_t0[t - start1 * dt_num] * r_t[t]
                - P_HPP_DW_t0[t - start1 * dt_num] * r_t[t]
                for t in range(start1 * dt_num, (start1 + 1) * dt_num)
            )
            - SubOpt_mdl.sum(
                omega1_t[t] * P_grid_limit * (1 - z_UP_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega1_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(
                omega1_t[t] * P_grid_limit * z_hat1_t_opt.loc[t, 0] for t in setT1
            )
            - SubOpt_mdl.sum(
                omega2_t[t] * P_grid_limit * (1 - z_DW_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega2_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(
                omega2_t[t] * P_grid_limit * z_hat2_t_opt.loc[t, 0] for t in setT1
            )
        )

    # SubOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    SubOpt_mdl.maximize(minus_profit)
    # SubOpt_mdl.maximize(SubOpt_mdl.sum(alpha1_s[t] for t in setTV))
    # SubOpt_mdl.parameters.mip.tolerances.mipgap=0.5

    SubOpt_mdl.parameters.timelimit = time_limit
    SubOpt_mdl.parameters.preprocessing.presolve = "on"
    SubOpt_mdl.parameters.mip.strategy.fpheur = 1
    # Solve DualSubSubOpt Model
    SubOpt_mdl.print_information()
    # SubOpt_mdl.parameters.mip.strategy.heuristicfreq = 1  # Frequency of applying heuristics (1 = always)
    # SubOpt_mdl.parameters.mip.strategy.rinsheur = 1
    # SubOpt_mdl.parameters.preprocessing.presolve = 2
    # Set the RINSHeur parameter
    # SubOpt_mdl.parameters.mip.strategy.rinsheur = 1  # Enable RINS heuristic
    # sol = SubOpt_mdl.solve(log_output=True)
    sol = SubOpt_mdl.solve(log_output=False)
    aa = SubOpt_mdl.get_solve_details()
    print(aa.status)
    print(SubOpt_mdl.solve_details.mip_relative_gap)

    if sol:

        # z_hat1_t_opt = pd.DataFrame([])
        # z_hat2_t_opt = pd.DataFrame([])

        obj_LB = sol.get_objective_value()
        obj_UB = SubOpt_mdl.solve_details.best_bound

        #    delta_tilde_P_HPP_t_opt = get_var_value_from_sol(delta_tilde_P_HPP_t, sol)
        #    delta_tilde_P_HPP_UP_t_opt = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt > 0, 0)
        #    delta_tilde_P_HPP_DW_t_opt = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt < 0, 0)
        #    P_tilde_HPP_UP_t_opt       = get_var_value_from_sol(P_tilde_HPP_UP_t, sol)
        #    P_tilde_HPP_DW_t_opt       = get_var_value_from_sol(P_tilde_HPP_DW_t, sol)
        #    P_tilde_w_HA_t_opt         = get_var_value_from_sol(P_tilde_w_HA_t, sol)
        #    P_tilde_dis_HA_t_opt       = get_var_value_from_sol(P_tilde_dis_HA_t, sol)
        #    P_tilde_cha_HA_t_opt       = get_var_value_from_sol(P_tilde_cha_HA_t, sol)
        delta_tilde_P_HPP_t_opt = pd.DataFrame([])
        delta_tilde_P_HPP_UP_t_opt = pd.DataFrame([])
        delta_tilde_P_HPP_DW_t_opt = pd.DataFrame([])
        P_tilde_HPP_UP_t_opt = pd.DataFrame([])
        P_tilde_HPP_DW_t_opt = pd.DataFrame([])
        P_tilde_w_HA_t_opt = pd.DataFrame([])
        P_tilde_dis_HA_t_opt = pd.DataFrame([])
        P_tilde_cha_HA_t_opt = pd.DataFrame([])

        # z_bigM = pd.concat([z_bigM_alpha1_opt, z_bigM_alpha2_opt, z_bigM_beta1_opt, z_bigM_m_opt, z_bigM_n_opt, z_bigM_l1_opt, z_bigM_l2_opt,
        #                    z_bigM_mu_opt, z_bigM_v_opt, z_bigM_p_opt, z_bigM_q_opt], axis = 1)
        z_bigM = pd.DataFrame([])

        m_opt = get_var_value_from_sol(m_t, sol)
        n_opt = get_var_value_from_sol(n_t, sol)
        alpha1_opt = get_var_value_from_sol(alpha1_t, sol)
        alpha2_opt = get_var_value_from_sol(alpha2_t, sol)
        beta1_opt = get_var_value_from_sol(beta1_t, sol)
        # beta2_opt = get_var_value_from_sol(beta2_t, sol)
        l1_opt = get_var_value_from_sol(l1_t, sol)
        l2_opt = get_var_value_from_sol(l2_t, sol)
        mu_opt = get_var_value_from_sol(mu_t, sol)
        v_opt = get_var_value_from_sol(v_t, sol)
        p_opt = get_var_value_from_sol(p_t, sol)
        q_opt = get_var_value_from_sol(q_t, sol)
        r_opt = get_var_value_from_sol(r_t, sol)
        # omega1_opt = get_var_value_from_sol(omega1_t, sol)
        # omega2_opt = get_var_value_from_sol(omega2_t, sol)
        # theta1_opt = get_var_value_from_sol(theta1_t, sol)

        # gamma1_opt = get_var_value_from_sol(gamma1_t, sol)
        # gamma3_opt = get_var_value_from_sol(gamma3_t, sol)

    #       z_hat_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_hat_t), orient='index')
    return obj_LB


def SubPrimalPro(
    dt,
    ds,
    dk,
    T,
    EBESS,
    PbMax,
    P_grid_limit,
    SoCmin,
    SoCmax,
    Emax,
    eta_dis,
    eta_cha,
    mu,
    ad,
    HA_wind_forecast,
    P_HPP_SM_t_opt,
    P_HPP_UP_t0,
    P_HPP_DW_t0,
    SoC0,
    deg_indicator,
    reg_forecast,
    Cp,
    start,
    P_HPP_UP_t_opt,
    P_HPP_DW_t_opt,
    C_dev,
    z_UP_bidlimit_t_opt,
    z_DW_bidlimit_t_opt,
    time_limit,
    z_hat1_t_opt,
    z_hat2_t_opt,
    xi_tilde_t_opt,
):

    dt_num = int(1 / dt)  # DI

    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    ds_num = int(1 / ds)  # SI
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    eta_cha_ha = eta_cha
    eta_dis_ha = eta_dis

    if start == -1:
        setT = [i for i in range((start + 1) * dt_num, T)]
        setS = [i for i in range((start + 1) * ds_num, T_ds)]
        set_SoCT = [i for i in range((start + 1) * dt_num, T + 1)]

        start1 = 0
    else:
        setT = [i for i in range(start * dt_num, T)]
        setS = [i for i in range(start * ds_num, T_ds)]
        set_SoCT = [i for i in range(start * dt_num, T + 1)]

        start1 = start
    setT1 = [i for i in range((start + 1) * dt_num, T)]
    # setK = [i for i in range(start*dk_num, T_dk + int(exten_num/dt_num))]
    # setK1 = [i for i in range((start + 1) * dk_num, T_dk + int(exten_num/dt_num))]

    setT1_up = [
        t for t in setT1 if math.isclose(z_UP_bidlimit_t_opt.loc[t, 0], 1, abs_tol=1e-3)
    ]
    setT1_dw = [
        t for t in setT1 if math.isclose(z_DW_bidlimit_t_opt.loc[t, 0], 1, abs_tol=1e-3)
    ]
    setT1_no = [t for t in setT1 if t not in setT1_up and t not in setT1_dw]

    reg_forecast = np.repeat(reg_forecast, dt_num)
    reg_forecast.index = range(T)
    SubOpt_mdl = Model()

    # omega4_t = SubOpt_mdl.continuous_var_dict(setT1_dw, lb=0, name='omega4')

    # gamma1_t = SubOpt_mdl.continuous_var_dict(setT1_up, lb=0, name='gamma1')
    # gamma3_t = SubOpt_mdl.continuous_var_dict(setT1_dw, lb=0, name='gamma3')
    # y_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name='y')

    P_tilde_w_HA_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="RT wind 15min")
    P_tilde_dis_HA_t = SubOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=cplex.infinity, name="RT discharge"
    )
    P_tilde_cha_HA_t = SubOpt_mdl.continuous_var_dict(
        setT, lb=0, ub=cplex.infinity, name="RT charge"
    )
    #
    # #P_tilde_b_HA_aux_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, ub=cplex.infinity, name='RT battery aux')
    # #E_tilde_DA_t   = SubOpt_mdl.continuous_var_dict(set_SoCT, lb=-cplex.infinity, ub=cplex.infinity, name='RT SoC')
    # #z_tilde_s        = SubOpt_mdl.binary_var_dict(setT, name='RT Cha or Discha')
    # #z_delta_tilde_s        = SubSubOpt_mdl.binary_var_dict(setT, name='pos imbalance or neg')

    # delta_tilde_P_HPP_t = SubOpt_mdl.continuous_var_dict(setS, lb=-cplex.infinity, ub=cplex.infinity, name='RT imbalance')
    # delta_tilde_P_special_HPP_s = SubOpt_mdl.continuous_var_dict(setS, lb=-cplex.infinity, ub=cplex.infinity, name='RT special imbalance')
    delta_tilde_P_HPP_UP_t = SubOpt_mdl.continuous_var_dict(
        setS, lb=0, name="RT up imbalance"
    )
    delta_tilde_P_HPP_DW_t = SubOpt_mdl.continuous_var_dict(
        setS, lb=0, name="RT dw imbalance"
    )

    # delta_tilde_P_special_HPP_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, ub=cplex.infinity, name='RT special imbalance')

    # tau_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, ub=cplex.infinity, name='special imbalance')

    w_tilde_t = SubOpt_mdl.continuous_var_dict(
        setT1, lb=-cplex.infinity, ub=cplex.infinity, name="aux penalty"
    )

    # z_bigM_alpha1 = SubOpt_mdl.binary_var_dict(setT, name='bigM alpha1')
    # z_bigM_beta1 = SubOpt_mdl.binary_var_dict(setT, name='bigM beta1')
    # z_bigM_alpha2 = SubOpt_mdl.binary_var_dict(setT, name='bigM alpha2')
    # z_bigM_beta2 = SubOpt_mdl.binary_var_dict(setT, name='bigM beta2')
    # z_bigM_gamma = SubOpt_mdl.binary_var_dict(setT, name='bigM gamma')
    # z_bigM_l1 = SubOpt_mdl.binary_var_dict(setT, name='bigM l')
    # z_bigM_l2 = SubOpt_mdl.binary_var_dict(setT, name='bigM 2')
    # z_bigM_m = SubOpt_mdl.binary_var_dict(setT, name='bigM m')
    # z_bigM_n = SubOpt_mdl.binary_var_dict(setT, name='bigM n')
    # z_bigM_mu = SubOpt_mdl.binary_var_dict(setT, name='bigM mu')
    # z_bigM_v = SubOpt_mdl.binary_var_dict(setT, name='bigM v')
    # z_bigM_p = SubOpt_mdl.binary_var_dict(setT, name='bigM p')
    # z_bigM_q = SubOpt_mdl.binary_var_dict(setT, name='bigM q')
    # z_bigM_theta1 = SubOpt_mdl.binary_var_dict(setT1, name='bigM theta 1')
    # z_bigM_theta2 = SubOpt_mdl.binary_var_dict(setT1_up, name='bigM theta 2')
    # z_bigM_theta3 = SubOpt_mdl.binary_var_dict(setT1_dw, name='bigM theta 3')
    # z_bigM_theta4 = SubOpt_mdl.binary_var_dict(setT1_dw, name='bigM theta 4')
    # z_bigM_omega = SubOpt_mdl.binary_var_dict(setT, name='bigM omega')
    # z_bigM_omega1 = SubOpt_mdl.binary_var_dict(setT1_up, name='bigM omega 1')
    # z_bigM_omega2 = SubOpt_mdl.binary_var_dict(setT1_dw, name='bigM omega 2')
    # z_bigM_omega3 = SubOpt_mdl.binary_var_dict(setT1_dw, name='bigM omega 3')
    # z_bigM_omega4 = SubOpt_mdl.binary_var_dict(setT1_dw, name='bigM omega 4')

    # z_bigM_gamma1 = SubOpt_mdl.binary_var_dict(setT1_up, name='bigM gamma 1')
    # z_bigM_gamma3 = SubOpt_mdl.binary_var_dict(setT1_dw, name='bigM gamma 3')

    # z_hat_t = SubOpt_mdl.binary_var_dict(setT1, name='up or down')

    # setTup = set(list(np.where(reg_forecast>SM_price_cleared)[0]))
    # setTdw = set(list(np.where(reg_forecast<SM_price_cleared)[0]))
    # setTup = list(setTup.intersection(set(setT1)))
    # setTdw = list(setTdw.intersection(set(setT1)))

    # SubOpt_mdl.add_constraint(z_hat1_t[t] + z_hat2_t[t] <= 1)

    # else:

    # for i in range(2*T):
    #    SubOpt_mdl.add_constraint(SubOpt_mdl.sum(error_C[i,t]*xi_tilde_t[t] for t in setT) <= error_d[i])

    # primal constraints
    for t in setT:
        SubOpt_mdl.add_constraint(P_tilde_dis_HA_t[t] <= PbMax)  # alpha 2
        SubOpt_mdl.add_constraint(P_tilde_cha_HA_t[t] <= PbMax)  # beta  2

        # SubOpt_mdl.add_constraint(P_tilde_b_HA_t[t] <= PbMax)  # alpha 1
        # SubOpt_mdl.add_constraint(P_tilde_b_HA_t[t] >= -PbMax) # beta  1

        # SubOpt_mdl.add_constraint(P_tilde_b_HA_aux_t[t] >= P_tilde_b_HA_t[t]/eta_dis_ha) # alpha 2
        # SubOpt_mdl.add_constraint(P_tilde_b_HA_aux_t[t] >= P_tilde_b_HA_t[t]*eta_cha_ha) # beta  2

        SubOpt_mdl.add_constraint(
            SoC0 * Emax
            - SubOpt_mdl.sum(
                (P_tilde_dis_HA_t[i] / eta_dis_ha - P_tilde_cha_HA_t[i] * eta_cha_ha)
                * dt
                for i in range(start1 * dt_num, t + 1)
            )
            <= SoCmax * Emax
        )  # m
        SubOpt_mdl.add_constraint(
            SoC0 * Emax
            - SubOpt_mdl.sum(
                (P_tilde_dis_HA_t[i] / eta_dis_ha - P_tilde_cha_HA_t[i] * eta_cha_ha)
                * dt
                for i in range(start1 * dt_num, t + 1)
            )
            >= SoCmin * Emax
        )  # n

        SubOpt_mdl.add_constraint(
            (P_tilde_dis_HA_t[t] - P_tilde_cha_HA_t[t]) + P_tilde_w_HA_t[t]
            <= P_grid_limit
        )  # l1
        SubOpt_mdl.add_constraint(
            (P_tilde_dis_HA_t[t] - P_tilde_cha_HA_t[t]) + P_tilde_w_HA_t[t] >= 0
        )  # l2
        SubOpt_mdl.add_constraint(P_tilde_w_HA_t[t] >= 0)  # mu
        SubOpt_mdl.add_constraint(
            P_tilde_w_HA_t[t] <= HA_wind_forecast[t] - xi_tilde_t_opt.loc[t, 0]
        )  # v

        if t < (start + 1) * dt_num:
            SubOpt_mdl.add_constraint(
                delta_tilde_P_HPP_UP_t[t] - delta_tilde_P_HPP_DW_t[t]
                == (P_tilde_dis_HA_t[t] - P_tilde_cha_HA_t[t])
                + P_tilde_w_HA_t[t]
                - P_HPP_SM_t_opt.loc[t].iloc[0]
                - P_HPP_UP_t0[t - start1 * dt_num]
                + P_HPP_DW_t0[t - start1 * dt_num]
            )
        else:
            SubOpt_mdl.add_constraint(
                delta_tilde_P_HPP_UP_t[t] - delta_tilde_P_HPP_DW_t[t]
                == (P_tilde_dis_HA_t[t] - P_tilde_cha_HA_t[t])
                + P_tilde_w_HA_t[t]
                - P_HPP_SM_t_opt.loc[t].iloc[0]
                - P_HPP_UP_t_opt.loc[t, 0] * z_hat1_t_opt.loc[t, 0]
                + P_HPP_DW_t_opt.loc[t, 0] * z_hat2_t_opt.loc[t, 0]
            )  # r

        # SubOpt_mdl.add_constraint(tau_t[t] >= delta_tilde_P_HPP_t[t])  # p
        # SubOpt_mdl.add_constraint(tau_t[t] >= -delta_tilde_P_HPP_t[t]) # q

    for t in setT1:
        SubOpt_mdl.add_constraint(w_tilde_t[t] >= 0)
        SubOpt_mdl.add_constraint(
            w_tilde_t[t]
            >= delta_tilde_P_HPP_UP_t[t]
            + delta_tilde_P_HPP_DW_t[t]
            - P_grid_limit * (1 - z_UP_bidlimit_t_opt.loc[t, 0])
            - P_grid_limit * (1 - z_hat1_t_opt.loc[t, 0])
        )
        SubOpt_mdl.add_constraint(
            w_tilde_t[t]
            >= delta_tilde_P_HPP_UP_t[t]
            + delta_tilde_P_HPP_DW_t[t]
            - P_grid_limit * (1 - z_DW_bidlimit_t_opt.loc[t, 0])
            - P_grid_limit * (1 - z_hat2_t_opt.loc[t, 0])
        )

    ## dual constraints and complementary slackness condition of uncertainty set

    minus_profit = (
        SubOpt_mdl.sum(
            -reg_forecast[t] * P_HPP_UP_t_opt.loc[t, 0] * z_hat1_t_opt.loc[t, 0] * dt
            + reg_forecast[t] * P_HPP_DW_t_opt.loc[t, 0] * z_hat2_t_opt.loc[t, 0] * dt
            for t in setT1
        )
        - SubOpt_mdl.sum(
            reg_forecast[t]
            * (delta_tilde_P_HPP_UP_t[t] - delta_tilde_P_HPP_DW_t[t])
            * ds
            for t in setS
        )
        + SubOpt_mdl.sum(
            ad * EBESS * mu * (P_tilde_dis_HA_t[t] + P_tilde_cha_HA_t[t]) * dt
            for t in setT
        )
        + SubOpt_mdl.sum(
            C_dev * (delta_tilde_P_HPP_UP_t[t] + delta_tilde_P_HPP_DW_t[t]) * dt
            for t in setT
        )
        + SubOpt_mdl.sum(Cp * (w_tilde_t[t]) * dt for t in setT1)
        + SubOpt_mdl.sum(
            (0 * max(reg_forecast) * math.exp(-(t - start1) / (2 * EBESS / PbMax)))
            * (delta_tilde_P_HPP_DW_t[t])
            * dt
            for t in setT
        )
    )

    # SubOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    SubOpt_mdl.minimize(minus_profit)
    # SubOpt_mdl.maximize(SubOpt_mdl.sum(alpha1_s[t] for t in setTV))
    # SubOpt_mdl.parameters.mip.tolerances.mipgap=0.5

    SubOpt_mdl.parameters.timelimit = time_limit
    SubOpt_mdl.parameters.preprocessing.presolve = "on"
    SubOpt_mdl.parameters.mip.strategy.fpheur = 1
    # Solve DualSubSubOpt Model
    SubOpt_mdl.print_information()
    # SubOpt_mdl.parameters.mip.strategy.heuristicfreq = 1  # Frequency of applying heuristics (1 = always)
    # SubOpt_mdl.parameters.mip.strategy.rinsheur = 1
    # SubOpt_mdl.parameters.preprocessing.presolve = 2
    # Set the RINSHeur parameter
    # SubOpt_mdl.parameters.mip.strategy.rinsheur = 1  # Enable RINS heuristic
    # sol = SubOpt_mdl.solve(log_output=True)
    sol = SubOpt_mdl.solve(log_output=False)
    aa = SubOpt_mdl.get_solve_details()
    print(aa.status)
    print(SubOpt_mdl.solve_details.mip_relative_gap)

    if sol:

        # z_hat1_t_opt = pd.DataFrame([])
        # z_hat2_t_opt = pd.DataFrame([])

        obj_LB = sol.get_objective_value()
        obj_UB = SubOpt_mdl.solve_details.best_bound

        # delta_tilde_P_HPP_t_opt = get_var_value_from_sol(delta_tilde_P_HPP_t, sol)
        # delta_tilde_P_HPP_UP_t_opt = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt > 0, 0)
        # delta_tilde_P_HPP_DW_t_opt = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt < 0, 0)
        delta_tilde_P_HPP_UP_t_opt = get_var_value_from_sol(delta_tilde_P_HPP_UP_t, sol)
        delta_tilde_P_HPP_DW_t_opt = get_var_value_from_sol(delta_tilde_P_HPP_DW_t, sol)
        # P_tilde_HPP_UP_t_opt       = get_var_value_from_sol(P_tilde_HPP_UP_t, sol)
        # P_tilde_HPP_DW_t_opt       = get_var_value_from_sol(P_tilde_HPP_DW_t, sol)
        P_tilde_w_HA_t_opt = get_var_value_from_sol(P_tilde_w_HA_t, sol)
        P_tilde_dis_HA_t_opt = get_var_value_from_sol(P_tilde_dis_HA_t, sol)
        P_tilde_cha_HA_t_opt = get_var_value_from_sol(P_tilde_cha_HA_t, sol)

        # gamma1_opt = get_var_value_from_sol(gamma1_t, sol)
        # gamma3_opt = get_var_value_from_sol(gamma3_t, sol)

    #       z_hat_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_hat_t), orient='index')
    return (
        obj_LB,
        delta_tilde_P_HPP_UP_t_opt,
        delta_tilde_P_HPP_DW_t_opt,
        P_tilde_w_HA_t_opt,
        P_tilde_dis_HA_t_opt,
        P_tilde_cha_HA_t_opt,
    )


def SubPro(
    dt,
    ds,
    dk,
    T,
    EBESS,
    PbMax,
    P_grid_limit,
    SoCmin,
    SoCmax,
    Emax,
    eta_dis,
    eta_cha,
    mu,
    ad,
    HA_wind_forecast,
    P_HPP_SM_t_opt,
    P_HPP_UP_t0,
    P_HPP_DW_t0,
    SoC0,
    deg_indicator,
    reg_forecast,
    bigM,
    start,
    P_HPP_UP_t_opt,
    P_HPP_DW_t_opt,
    xi_max,
    xi_min,
    C_dev,
    z_UP_bidlimit_t_opt,
    z_DW_bidlimit_t_opt,
    time_limit,
    Cp,
):

    dt_num = int(1 / dt)  # DI

    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    ds_num = int(1 / ds)  # SI
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    eta_cha_ha = eta_cha
    eta_dis_ha = eta_dis

    if start == -1:
        setT = [i for i in range((start + 1) * dt_num, T)]
        setS = [i for i in range((start + 1) * ds_num, T_ds)]
        set_SoCT = [i for i in range((start + 1) * dt_num, T + 1)]

        start1 = 0
    else:
        setT = [i for i in range(start * dt_num, T)]
        setS = [i for i in range(start * ds_num, T_ds)]
        set_SoCT = [i for i in range(start * dt_num, T + 1)]

        start1 = start
    setT1 = [i for i in range((start + 1) * dt_num, T)]
    # setK = [i for i in range(start*dk_num, T_dk + int(exten_num/dt_num))]
    # setK1 = [i for i in range((start + 1) * dk_num, T_dk + int(exten_num/dt_num))]

    setT1_up = [
        t for t in setT1 if math.isclose(z_UP_bidlimit_t_opt.loc[t, 0], 1, abs_tol=1e-3)
    ]
    setT1_dw = [
        t for t in setT1 if math.isclose(z_DW_bidlimit_t_opt.loc[t, 0], 1, abs_tol=1e-3)
    ]
    setT1_no = [t for t in setT1 if t not in setT1_up and t not in setT1_dw]

    reg_forecast = np.repeat(reg_forecast, dt_num)
    reg_forecast.index = range(T)
    SubOpt_mdl = Model()

    xi_tilde_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, name="xi")
    P_tilde_HPP_UP_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="P_HPP_UP")
    P_tilde_HPP_DW_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="P_HPP_DW")

    # xi_tilde_t[0].start = xi_max[0]

    alpha1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="alpha1")
    alpha2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="alpha2")
    beta1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="beta1")
    beta2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="beta2")
    # gamma_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, name='gamma')
    m_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="m")
    n_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="n")
    l1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="l1")
    l2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="l2")
    mu_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="mu")
    v_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="v")
    r_t = SubOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, name="r")
    p_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="p")
    q_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="q")

    theta1_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="theta1")
    omega1_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="omega1")
    omega2_t = SubOpt_mdl.continuous_var_dict(setT1, lb=0, name="omega2")

    y1_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="y min")
    y2_t = SubOpt_mdl.continuous_var_dict(setT, lb=0, name="y max")

    # y3_i = SubOpt_mdl.continuous_var_dict(list(range(2)), lb=0, name='y error_C')

    f1_tilde_t = SubOpt_mdl.continuous_var_dict(
        setT1, lb=-cplex.infinity, ub=cplex.infinity, name="aux for f1"
    )
    f2_tilde_t = SubOpt_mdl.continuous_var_dict(
        setT1, lb=-cplex.infinity, ub=cplex.infinity, name="aux for f2"
    )
    f3_tilde_t = SubOpt_mdl.continuous_var_dict(
        setT1, lb=-cplex.infinity, ub=cplex.infinity, name="aux for f3"
    )
    f4_tilde_t = SubOpt_mdl.continuous_var_dict(
        setT1, lb=-cplex.infinity, ub=cplex.infinity, name="aux for f4"
    )

    z_bigM_y1 = SubOpt_mdl.binary_var_dict(setT, name="bigM y1")
    z_bigM_y2 = SubOpt_mdl.binary_var_dict(setT, name="bigM y2")
    # z_bigM_y3 = SubOpt_mdl.binary_var_dict(list(range(2)), name='bigM y3')

    z_hat1_t = SubOpt_mdl.binary_var_dict(setT1, name="up or zero")
    z_hat2_t = SubOpt_mdl.binary_var_dict(setT1, name="dw or zero")

    for t in setT1:
        SubOpt_mdl.add_constraint(
            P_tilde_HPP_UP_t[t] == P_HPP_UP_t_opt.loc[t, 0] * z_hat1_t[t]
        )
        SubOpt_mdl.add_constraint(
            P_tilde_HPP_DW_t[t] == P_HPP_DW_t_opt.loc[t, 0] * z_hat2_t[t]
        )

    SubOpt_mdl.add_constraint(
        SubOpt_mdl.sum(z_hat1_t[t] for t in setT1) >= len(setT1) * 0.8
    )
    SubOpt_mdl.add_constraint(
        SubOpt_mdl.sum(z_hat2_t[t] for t in setT1) >= len(setT1) * 0.5
    )

    # if len(setT1)==T:
    #     SubOpt_mdl.add_constraint(SubOpt_mdl.sum(z_hat1_t[t] for t in setT1)>=23*dt_num)
    #     SubOpt_mdl.add_constraint(SubOpt_mdl.sum(z_hat2_t[t] for t in setT1)>=10*dt_num)
    # else:
    #     SubOpt_mdl.add_constraint(SubOpt_mdl.sum(z_hat1_t[t] for t in setT1)>=10*dt_num)
    #     SubOpt_mdl.add_constraint(SubOpt_mdl.sum(z_hat2_t[t] for t in setT1)>=5*dt_num)

    for t in setT:
        SubOpt_mdl.add_constraint(xi_tilde_t[t] <= xi_max[t])
        SubOpt_mdl.add_constraint(xi_tilde_t[t] >= xi_min[t])

    # for i in range(2):
    #    SubOpt_mdl.add_constraint(SubOpt_mdl.sum(error_C[i,t-start1]*xi_tilde_t[t] for t in setT) <= error_d[i])

    # SubOpt_mdl.add_constraint(SubOpt_mdl.sum(xi_tilde_t[t] for t in setT) <= 400)
    # SubOpt_mdl.add_constraint(SubOpt_mdl.sum(xi_tilde_t[t] for t in setT) >= -1500)

    ## dual constraints
    for t in setT:
        SubOpt_mdl.add_constraint(
            -alpha1_t[t]
            + alpha2_t[t]
            + l1_t[t]
            - l2_t[t]
            - r_t[t]
            - SubOpt_mdl.sum(m_t[i] for i in range(t, T)) / eta_dis_ha * dt
            + SubOpt_mdl.sum(n_t[i] for i in range(t, T)) / eta_dis_ha * dt
            + ad * EBESS * mu * dt
            == 0
        )  # P_dis
        SubOpt_mdl.add_constraint(
            -beta1_t[t]
            + beta2_t[t]
            - l1_t[t]
            + l2_t[t]
            + r_t[t]
            + SubOpt_mdl.sum(m_t[i] for i in range(t, T)) * eta_cha_ha * dt
            - SubOpt_mdl.sum(n_t[i] for i in range(t, T)) * eta_cha_ha * dt
            + ad * EBESS * mu * dt
            == 0
        )  # P_b_aux
        SubOpt_mdl.add_constraint(
            l1_t[t] - l2_t[t] - mu_t[t] + v_t[t] - r_t[t] == 0
        )  # P_w

        # SubOpt_mdl.add_constraint(-(reg_forecast[t])*dt + r_t[t] + p_t[t] - q_t[t]  == 0) # delta_P

        if t < (start + 1) * dt_num:
            SubOpt_mdl.add_constraint(
                -(reg_forecast[t]) * dt + C_dev * dt + r_t[t] - p_t[t] == 0
            )  # delta_P_UP
            SubOpt_mdl.add_constraint(
                (reg_forecast[t]) * dt
                + C_dev * dt
                + (
                    0
                    * max(reg_forecast)
                    * math.exp(-(t - start1) / (2 * EBESS / PbMax))
                )
                * dt
                - r_t[t]
                - q_t[t]
                == 0
            )  # delta_P_DW

        else:
            SubOpt_mdl.add_constraint(
                -(reg_forecast[t]) * dt
                + C_dev * dt
                + r_t[t]
                - p_t[t]
                + omega1_t[t]
                + omega2_t[t]
                == 0
            )  # delta_P_UP
            SubOpt_mdl.add_constraint(
                (reg_forecast[t]) * dt
                + C_dev * dt
                + (
                    0
                    * max(reg_forecast)
                    * math.exp(-(t - start1) / (2 * EBESS / PbMax))
                )
                * dt
                - r_t[t]
                - q_t[t]
                + omega1_t[t]
                + omega2_t[t]
                == 0
            )  # delta_P_DW
            # SubOpt_mdl.add_constraint(- p_t[t] - q_t[t] + C_dev*dt*1e3*(T-t) + omega1_t[t] + omega2_t[t]  == 0) # tau
            SubOpt_mdl.add_constraint(
                -theta1_t[t] - omega1_t[t] - omega2_t[t] + Cp * dt == 0
            )  # w_tilde_t

    ## aux constriants
    for t in setT1:
        SubOpt_mdl.add_constraint(f1_tilde_t[t] >= -bigM[t] * (z_hat1_t[t]))
        SubOpt_mdl.add_constraint(f1_tilde_t[t] <= bigM[t] * (z_hat1_t[t]))
        SubOpt_mdl.add_constraint(
            -bigM[t] * (1 - z_hat1_t[t]) + r_t[t] <= f1_tilde_t[t]
        )
        SubOpt_mdl.add_constraint(f1_tilde_t[t] <= bigM[t] * (1 - z_hat1_t[t]) + r_t[t])

        SubOpt_mdl.add_constraint(f2_tilde_t[t] >= -bigM[t] * (z_hat2_t[t]))
        SubOpt_mdl.add_constraint(f2_tilde_t[t] <= bigM[t] * (z_hat2_t[t]))
        SubOpt_mdl.add_constraint(
            -bigM[t] * (1 - z_hat2_t[t]) + r_t[t] <= f2_tilde_t[t]
        )
        SubOpt_mdl.add_constraint(f2_tilde_t[t] <= bigM[t] * (1 - z_hat2_t[t]) + r_t[t])

        SubOpt_mdl.add_constraint(f3_tilde_t[t] >= -bigM[t] * (z_hat1_t[t]))
        SubOpt_mdl.add_constraint(f3_tilde_t[t] <= bigM[t] * (z_hat1_t[t]))
        SubOpt_mdl.add_constraint(
            -bigM[t] * (1 - z_hat1_t[t]) + omega1_t[t] <= f3_tilde_t[t]
        )
        SubOpt_mdl.add_constraint(
            f3_tilde_t[t] <= bigM[t] * (1 - z_hat1_t[t]) + omega1_t[t]
        )

        SubOpt_mdl.add_constraint(f4_tilde_t[t] >= -bigM[t] * (z_hat2_t[t]))
        SubOpt_mdl.add_constraint(f4_tilde_t[t] <= bigM[t] * (z_hat2_t[t]))
        SubOpt_mdl.add_constraint(
            -bigM[t] * (1 - z_hat2_t[t]) + omega2_t[t] <= f4_tilde_t[t]
        )
        SubOpt_mdl.add_constraint(
            f4_tilde_t[t] <= bigM[t] * (1 - z_hat2_t[t]) + omega2_t[t]
        )

    ## dual constraints and complementary slackness condition of uncertainty set

    for t in setT:
        # SubOpt_mdl.add_constraint(v_t[t] + y1_t[t] - y2_t[t] - SubOpt_mdl.sum(y3_i[i]*error_C[i,t-start1] for i in range(2)) ==0)
        SubOpt_mdl.add_constraint(v_t[t] + y1_t[t] - y2_t[t] == 0)
        # SubOpt_mdl.add_constraint(v_t[t] + y1_t[t] - y2_t[t] - y3_i[0] ==0)

        SubOpt_mdl.add_constraint(y1_t[t] <= bigM[t] * (1 - z_bigM_y1[t]))
        SubOpt_mdl.add_constraint(xi_tilde_t[t] - xi_min[t] <= bigM[t] * (z_bigM_y1[t]))

        SubOpt_mdl.add_constraint(y2_t[t] <= bigM[t] * (1 - z_bigM_y2[t]))
        SubOpt_mdl.add_constraint(
            -xi_tilde_t[t] + xi_max[t] <= bigM[t] * (z_bigM_y2[t])
        )

    # for i in range(2):
    #    SubOpt_mdl.add_constraint(y3_i[i]<=bigM[0]*(1-z_bigM_y3[i]))
    #    SubOpt_mdl.add_constraint(- SubOpt_mdl.sum(error_C[i,t-start1]*xi_tilde_t[t] for t in setT) + error_d[i] <=3*error_d[i]*(z_bigM_y3[i]))

    # SubOpt_mdl.add_constraint(y3_i[0]<=bigM[0]/10*(1-z_bigM_y3[0]))
    # SubOpt_mdl.add_constraint(- SubOpt_mdl.sum(xi_tilde_t[t] for t in setT) + 400 <=1e3*(z_bigM_y3[0]))

    # SubOpt_mdl.add_constraint(y3_i[1]<=bigM[0]*(1-z_bigM_y3[1]))
    # SubOpt_mdl.add_constraint( SubOpt_mdl.sum(xi_tilde_t[t] for t in setT) + 1500 <=1e4*(z_bigM_y3[1]))

    if start == -1:
        minus_profit = (
            SubOpt_mdl.sum(
                -reg_forecast[t] * P_tilde_HPP_UP_t[t] * dt
                + reg_forecast[t] * P_tilde_HPP_DW_t[t] * dt
                for t in setT1
            )
            + SubOpt_mdl.sum(
                -PbMax * alpha2_t[t]
                - PbMax * beta2_t[t]
                + n_t[t] * (SoCmin * Emax - SoC0 * Emax)
                + m_t[t] * (SoC0 * Emax - SoCmax * Emax)
                - v_t[t] * HA_wind_forecast[t]
                - l1_t[t] * P_grid_limit
                + r_t[t] * P_HPP_SM_t_opt.loc[t].iloc[0]
                for t in setT
            )
            + SubOpt_mdl.sum(-y1_t[t] * xi_min[t] + y2_t[t] * xi_max[t] for t in setT)
            + SubOpt_mdl.sum(
                P_HPP_UP_t_opt.loc[t, 0] * f1_tilde_t[t]
                - P_HPP_DW_t_opt.loc[t, 0] * f2_tilde_t[t]
                for t in setT1
            )
            - SubOpt_mdl.sum(
                omega1_t[t] * P_grid_limit * (1 - z_UP_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega1_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(f3_tilde_t[t] * P_grid_limit for t in setT1)
            - SubOpt_mdl.sum(
                omega2_t[t] * P_grid_limit * (1 - z_DW_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega2_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(f4_tilde_t[t] * P_grid_limit for t in setT1)
        )
    else:  # + SubOpt_mdl.sum(-y3_i[i]*error_d[i] for i in range(2*len(setT)))
        minus_profit = (
            SubOpt_mdl.sum(
                -reg_forecast[t] * P_tilde_HPP_UP_t[t] * dt
                + reg_forecast[t] * P_tilde_HPP_DW_t[t] * dt
                for t in setT1
            )
            + SubOpt_mdl.sum(
                -PbMax * alpha2_t[t]
                - PbMax * beta2_t[t]
                + n_t[t] * (SoCmin * Emax - SoC0 * Emax)
                + m_t[t] * (SoC0 * Emax - SoCmax * Emax)
                - v_t[t] * HA_wind_forecast[t]
                - l1_t[t] * P_grid_limit
                + r_t[t] * P_HPP_SM_t_opt.loc[t].iloc[0]
                for t in setT
            )
            + SubOpt_mdl.sum(-y1_t[t] * xi_min[t] + y2_t[t] * xi_max[t] for t in setT)
            + SubOpt_mdl.sum(
                P_HPP_UP_t_opt.loc[t, 0] * f1_tilde_t[t]
                - P_HPP_DW_t_opt.loc[t, 0] * f2_tilde_t[t]
                for t in setT1
            )
            + SubOpt_mdl.sum(
                P_HPP_UP_t0[t - start1 * dt_num] * r_t[t]
                - P_HPP_DW_t0[t - start1 * dt_num] * r_t[t]
                for t in range(start1 * dt_num, (start1 + 1) * dt_num)
            )
            - SubOpt_mdl.sum(
                omega1_t[t] * P_grid_limit * (1 - z_UP_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega1_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(f3_tilde_t[t] * P_grid_limit for t in setT1)
            - SubOpt_mdl.sum(
                omega2_t[t] * P_grid_limit * (1 - z_DW_bidlimit_t_opt.loc[t, 0])
                for t in setT1
            )
            - SubOpt_mdl.sum(omega2_t[t] * P_grid_limit for t in setT1)
            + SubOpt_mdl.sum(f4_tilde_t[t] * P_grid_limit for t in setT1)
        )

    # SubOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    SubOpt_mdl.maximize(minus_profit)
    # SubOpt_mdl.maximize(SubOpt_mdl.sum(alpha1_s[t] for t in setTV))
    # SubOpt_mdl.parameters.mip.tolerances.mipgap=0.5

    SubOpt_mdl.parameters.timelimit = time_limit
    # SubOpt_mdl.parameters.preprocessing.presolve = 'on'
    # SubOpt_mdl.parameters.mip.strategy.fpheur = 1
    # Solve DualSubSubOpt Model
    SubOpt_mdl.print_information()
    # SubOpt_mdl.parameters.mip.strategy.heuristicfreq = 1  # Frequency of applying heuristics (1 = always)
    # SubOpt_mdl.parameters.mip.strategy.rinsheur = 1
    # SubOpt_mdl.parameters.preprocessing.presolve = 2
    # Set the RINSHeur parameter
    # SubOpt_mdl.parameters.mip.strategy.rinsheur = 1  # Enable RINS heuristic
    # sol = SubOpt_mdl.solve(log_output=True)
    sol = SubOpt_mdl.solve(log_output=False)
    aa = SubOpt_mdl.get_solve_details()
    print(aa.status)
    print(SubOpt_mdl.solve_details.mip_relative_gap)

    if sol:
        z_hat1_t_opt = get_var_value_from_sol(z_hat1_t, sol)
        z_hat2_t_opt = get_var_value_from_sol(z_hat2_t, sol)

        #    z_hat1_t_opt = pd.DataFrame([])
        #    z_hat2_t_opt = pd.DataFrame([])

        obj_LB = sol.get_objective_value()
        obj_UB = SubOpt_mdl.solve_details.best_bound

        #    delta_tilde_P_HPP_t_opt = get_var_value_from_sol(delta_tilde_P_HPP_t, sol)
        #    delta_tilde_P_HPP_UP_t_opt = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt > 0, 0)
        #    delta_tilde_P_HPP_DW_t_opt = delta_tilde_P_HPP_t_opt.where(delta_tilde_P_HPP_t_opt < 0, 0)
        #    P_tilde_HPP_UP_t_opt       = get_var_value_from_sol(P_tilde_HPP_UP_t, sol)
        #    P_tilde_HPP_DW_t_opt       = get_var_value_from_sol(P_tilde_HPP_DW_t, sol)
        #    P_tilde_w_HA_t_opt         = get_var_value_from_sol(P_tilde_w_HA_t, sol)
        #    P_tilde_dis_HA_t_opt       = get_var_value_from_sol(P_tilde_dis_HA_t, sol)
        #    P_tilde_cha_HA_t_opt       = get_var_value_from_sol(P_tilde_cha_HA_t, sol)
        delta_tilde_P_HPP_t_opt = pd.DataFrame([])
        delta_tilde_P_HPP_UP_t_opt = pd.DataFrame([])
        delta_tilde_P_HPP_DW_t_opt = pd.DataFrame([])
        P_tilde_HPP_UP_t_opt = pd.DataFrame([])
        P_tilde_HPP_DW_t_opt = pd.DataFrame([])
        P_tilde_w_HA_t_opt = pd.DataFrame([])
        P_tilde_dis_HA_t_opt = pd.DataFrame([])
        P_tilde_cha_HA_t_opt = pd.DataFrame([])

        xi_tilde_t_opt = get_var_value_from_sol(xi_tilde_t, sol)
        P_tilde_HPP_UP_t_opt = get_var_value_from_sol(P_tilde_HPP_UP_t, sol)
        P_tilde_HPP_DW_t_opt = get_var_value_from_sol(P_tilde_HPP_DW_t, sol)

        # w_tilde_t_opt           = get_var_value_from_sol(w_tilde_t, sol)
        # w_tilde_dw_t_opt           = get_var_value_from_sol(w_tilde_dw_t, sol)
        f1_tilde_t_opt = get_var_value_from_sol(f1_tilde_t, sol)
        f2_tilde_t_opt = get_var_value_from_sol(f2_tilde_t, sol)
        f3_tilde_t_opt = get_var_value_from_sol(f3_tilde_t, sol)

        z_bigM_y1_opt = get_var_value_from_sol(z_bigM_y1, sol)
        z_bigM_y2_opt = get_var_value_from_sol(z_bigM_y2, sol)

        # z_bigM = pd.concat([z_bigM_alpha1_opt, z_bigM_alpha2_opt, z_bigM_beta1_opt, z_bigM_m_opt, z_bigM_n_opt, z_bigM_l1_opt, z_bigM_l2_opt,
        #                    z_bigM_mu_opt, z_bigM_v_opt, z_bigM_p_opt, z_bigM_q_opt], axis = 1)
        z_bigM = pd.concat([z_bigM_y1_opt, z_bigM_y2_opt], axis=1)

        m_opt = get_var_value_from_sol(m_t, sol)
        n_opt = get_var_value_from_sol(n_t, sol)
        alpha1_opt = get_var_value_from_sol(alpha1_t, sol)
        alpha2_opt = get_var_value_from_sol(alpha2_t, sol)
        beta1_opt = get_var_value_from_sol(beta1_t, sol)
        beta2_opt = get_var_value_from_sol(beta2_t, sol)
        l1_opt = get_var_value_from_sol(l1_t, sol)
        l2_opt = get_var_value_from_sol(l2_t, sol)
        mu_opt = get_var_value_from_sol(mu_t, sol)
        v_opt = get_var_value_from_sol(v_t, sol)
        p_opt = get_var_value_from_sol(p_t, sol)
        q_opt = get_var_value_from_sol(q_t, sol)
        r_opt = get_var_value_from_sol(r_t, sol)
        omega1_opt = get_var_value_from_sol(omega1_t, sol)
        # omega2_opt = get_var_value_from_sol(omega2_t, sol)
        # omega3_opt = get_var_value_from_sol(omega3_t, sol)
        # omega4_opt = get_var_value_from_sol(omega4_t, sol)
        y1_t_opt = get_var_value_from_sol(y1_t, sol)
        y2_t_opt = get_var_value_from_sol(y2_t, sol)
        # theta2_opt = get_var_value_from_sol(theta2_t, sol)
        # theta1_opt = get_var_value_from_sol(theta1_t, sol)

        # gamma1_opt = get_var_value_from_sol(gamma1_t, sol)
        # gamma3_opt = get_var_value_from_sol(gamma3_t, sol)

    #       z_hat_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(z_hat_t), orient='index')
    return (
        xi_tilde_t_opt,
        P_tilde_HPP_UP_t_opt,
        P_tilde_HPP_DW_t_opt,
        z_hat1_t_opt,
        z_hat2_t_opt,
        delta_tilde_P_HPP_UP_t_opt,
        delta_tilde_P_HPP_DW_t_opt,
        P_tilde_w_HA_t_opt,
        P_tilde_dis_HA_t_opt,
        P_tilde_cha_HA_t_opt,
        obj_LB,
        obj_UB,
        z_bigM,
    )


def BigMPro(
    dt,
    ds,
    dk,
    T,
    EBESS,
    PbMax,
    PwMax,
    PreUp,
    PreDw,
    P_grid_limit,
    SoCmin,
    SoCmax,
    Emax,
    eta_dis,
    eta_cha,
    eta_leak,
    mu,
    ad,
    HA_wind_forecast,
    P_HPP_HA_t_opt,
    P_HPP_SM_t_opt,
    P_HPP_UP_t0,
    P_HPP_DW_t0,
    SoC0,
    exten_num,
    deg_indicator,
    probability,
    reg_forecast,
    bigM,
    P_HPP_UP_t_opt,
    P_HPP_DW_t_opt,
    xi_max,
    xi_min,
    C_dev,
    start,
    xi_tilde_t_opt,
    P_tilde_HPP_UP_t_opt,
    P_tilde_HPP_DW_t_opt,
    z_hat1_t_opt,
    z_hat2_t_opt,
):

    dt_num = int(1 / dt)  # DI

    dk_num = int(1 / dk)  # BI
    T_dk = int(24 / dk)

    ds_num = int(1 / ds)  # SI
    T_ds = int(24 / ds)
    dsdt_num = int(ds / dt)

    eta_cha_ha = eta_cha ** (1 / dt_num)
    eta_dis_ha = eta_dis ** (1 / dt_num)
    eta_leak_ha = 1 - (1 - eta_leak) ** (1 / dt_num)
    eta_equ = (1 / eta_dis_ha + eta_cha_ha) / 2

    if start == -1:
        setT = [i for i in range((start + 1) * dt_num, T + exten_num)]
        setS = [
            i for i in range((start + 1) * ds_num, T_ds + int(exten_num / dsdt_num))
        ]
        set_SoCT = [i for i in range((start + 1) * dt_num, T + 1 + exten_num)]
        start1 = 0
    else:
        setT = [i for i in range(start * dt_num, T + exten_num)]
        setS = [i for i in range(start * ds_num, T_ds + int(exten_num / dsdt_num))]
        set_SoCT = [i for i in range(start * dt_num, T + 1 + exten_num)]
        start1 = start
    setT1 = [i for i in range((start + 1) * dt_num, T + exten_num)]

    # setI = range(len(BP_up_scenario))

    BigMOpt_mdl = Model()

    w_alpha1_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="alpha1")
    w_beta1_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="beta1")
    w_alpha2_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="alpha2")
    w_beta2_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="beta2")
    # w_gamma_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name='gamma')
    w_m_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="m")
    w_n_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="n")
    w_l1_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="l1")
    w_l2_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="l2")
    w_mu_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="mu")
    w_v_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="v")
    # r_t = BigMOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, name='r')
    w_p_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="p")
    w_q_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="q")
    # w_theta1_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name='theta1')
    # w_theta2_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name='theta2')

    # w_omega_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name='omega')
    # w_omega2_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name='omega2')
    # w_y_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name='y')

    P_tilde_w_HA_t = BigMOpt_mdl.continuous_var_dict(setT, lb=0, name="RT wind 15min")
    P_tilde_HA_dis_t = BigMOpt_mdl.continuous_var_dict(
        setT, lb=-cplex.infinity, ub=cplex.infinity, name="RT discharge"
    )
    P_tilde_HA_cha_t = BigMOpt_mdl.continuous_var_dict(
        setT, lb=-cplex.infinity, ub=cplex.infinity, name="RT charge"
    )
    # E_tilde_DA_t   = BigMOpt_mdl.continuous_var_dict(set_SoCT, lb=-cplex.infinity, ub=cplex.infinity, name='RT SoC')
    # z_tilde_s        = BigMOpt_mdl.binary_var_dict(setT, name='RT Cha or Discha')
    # z_delta_tilde_s        = SubBigMOpt_mdl.binary_var_dict(setT, name='pos imbalance or neg')

    delta_tilde_P_HPP_t = BigMOpt_mdl.continuous_var_dict(
        setS, lb=-cplex.infinity, ub=cplex.infinity, name="RT imbalance"
    )
    # delta_tilde_P_special_HPP_t = BigMOpt_mdl.continuous_var_dict(setS, lb=-cplex.infinity, ub=cplex.infinity, name='RT special imbalance')
    # delta_tilde_P_HPP_UP_t = BigMOpt_mdl.continuous_var_dict(setS, lb=0, name='RT up imbalance')
    # delta_tilde_P_HPP_DW_t = BigMOpt_mdl.continuous_var_dict(setS, lb=0, name='RT dw imbalance')

    tau_t = BigMOpt_mdl.continuous_var_dict(
        setT, lb=-cplex.infinity, ub=cplex.infinity, name="special imbalance"
    )
    # tau2_t = BigMOpt_mdl.continuous_var_dict(setT, lb=-cplex.infinity, ub=cplex.infinity, name='RT special imbalance aux')

    ## primal constraints
    for t in setT:
        # BigMOpt_mdl.add_constraint(P_tilde_HA_dis_t[t] <= PbMax + w_alpha2_t[t])
        # BigMOpt_mdl.add_constraint(P_tilde_HA_cha_t[t] <= PbMax + w_beta2_t[t])
        BigMOpt_mdl.add_constraint(P_tilde_HA_dis_t[t] + w_alpha1_t[t] >= 0)
        BigMOpt_mdl.add_constraint(P_tilde_HA_cha_t[t] + w_beta1_t[t] >= 0)
        BigMOpt_mdl.add_constraint(P_tilde_HA_dis_t[t] <= PbMax + w_alpha2_t[t])
        BigMOpt_mdl.add_constraint(P_tilde_HA_cha_t[t] <= PbMax + w_beta2_t[t])

        # BigMOpt_mdl.add_constraint(E_tilde_DA_t[t + 1] == E_tilde_DA_t[t] * (1-eta_leak) - (P_dis_SM_t_opt.iloc[t,0])/eta_dis_ha * dt + (P_cha_SM_t_opt.iloc[t,0]) * eta_cha_ha * dt + delta_tilde_P_b_DA_t[t] * dt )
        BigMOpt_mdl.add_constraint(
            SoC0 * Emax * (1 - eta_leak)
            - BigMOpt_mdl.sum(
                (P_tilde_HA_dis_t[i]) / eta_dis_ha * dt
                - (P_tilde_HA_cha_t[i]) * eta_cha_ha * dt
                for i in range(start1, t + 1)
            )
            <= Emax * SoCmax + w_m_t[t]
        )
        BigMOpt_mdl.add_constraint(
            SoC0 * Emax * (1 - eta_leak)
            - BigMOpt_mdl.sum(
                (P_tilde_HA_dis_t[i]) / eta_dis_ha * dt
                - (P_tilde_HA_cha_t[i]) * eta_cha_ha * dt
                for i in range(start1, t + 1)
            )
            + w_n_t[t]
            >= SoCmin * Emax
        )

        BigMOpt_mdl.add_constraint(
            P_tilde_HA_dis_t[t] - P_tilde_HA_cha_t[t] + P_tilde_w_HA_t[t]
            <= P_grid_limit + w_l1_t[t]
        )
        BigMOpt_mdl.add_constraint(
            P_tilde_HA_dis_t[t] - P_tilde_HA_cha_t[t] + P_tilde_w_HA_t[t] + w_l2_t[t]
            >= 0
        )
        BigMOpt_mdl.add_constraint(P_tilde_w_HA_t[t] + w_mu_t[t] >= 0)
        BigMOpt_mdl.add_constraint(
            P_tilde_w_HA_t[t]
            <= HA_wind_forecast[t] - xi_tilde_t_opt.loc[t, 0] + w_v_t[t]
        )

        if t < (start + 1) * dt_num:
            BigMOpt_mdl.add_constraint(
                delta_tilde_P_HPP_t[t]
                == P_tilde_HA_dis_t[t]
                - P_tilde_HA_cha_t[t]
                + P_tilde_w_HA_t[t]
                - P_HPP_SM_t_opt.iloc[t, 0]
                - P_HPP_UP_t0
                + P_HPP_DW_t0
            )
        else:
            BigMOpt_mdl.add_constraint(
                delta_tilde_P_HPP_t[t]
                == P_tilde_HA_dis_t[t]
                - P_tilde_HA_cha_t[t]
                + P_tilde_w_HA_t[t]
                - P_HPP_SM_t_opt.iloc[t, 0]
                - P_tilde_HPP_UP_t_opt.loc[t, 0]
                + P_tilde_HPP_DW_t_opt.loc[t, 0]
            )

        BigMOpt_mdl.add_constraint(tau_t[t] + w_p_t[t] >= delta_tilde_P_HPP_t[t])
        BigMOpt_mdl.add_constraint(tau_t[t] + w_q_t[t] >= -delta_tilde_P_HPP_t[t])

        # if t < (start + 1) * dt_num:
        #   BigMOpt_mdl.add_constraint(delta_tilde_P_special_HPP_t[t] + P_HPP_UP_t0 - P_HPP_DW_t0 == P_tilde_w_HA_t[t] + P_tilde_HA_dis_t[t]  - P_tilde_HA_cha_t[t]  - P_HPP_HA_t_opt.loc[t,0])

        # else:
        #   BigMOpt_mdl.add_constraint(delta_tilde_P_special_HPP_t[t] + P_tilde_HPP_UP_t_opt.loc[t,0] - P_tilde_HPP_DW_t_opt.loc[t,0] == P_tilde_w_HA_t[t] + P_tilde_HA_dis_t[t]  - P_tilde_HA_cha_t[t]  - P_HPP_HA_t_opt.loc[t,0])

        # BigMOpt_mdl.add_constraint(tau_t[t] + w_theta1_t[t] >= delta_tilde_P_special_HPP_t[t])
        # BigMOpt_mdl.add_constraint(tau_t[t] + w_theta2_t[t] >= -delta_tilde_P_special_HPP_t[t])

        # BigMOpt_mdl.add_constraint(tau2_t[t] + w_omega1_t[t] >= -10*P_grid_limit * (z_hat1_t_opt.iloc[t,0] + z_hat2_t_opt.iloc[t,0]))
        # BigMOpt_mdl.add_constraint(tau2_t[t] <= 10*P_grid_limit * (z_hat1_t_opt.iloc[t,0] + z_hat2_t_opt.iloc[t,0]) + w_omega2_t[t])

        # BigMOpt_mdl.add_constraint(tau2_t[t] + w_y_t[t] >= delta_tilde_P_HPP_UP_t[t] + delta_tilde_P_HPP_DW_t[t] - 10*P_grid_limit * (1 - z_hat1_t_opt.iloc[t,0] - z_hat2_t_opt.iloc[t,0]))

    # BigMOpt_mdl.add_constraint(E_tilde_DA_t[0] == SoC0*Emax)

    minus_profit = (
        BigMOpt_mdl.sum(
            -reg_forecast[t] * P_tilde_HPP_UP_t_opt.loc[t, 0] * dt
            + reg_forecast[t] * P_tilde_HPP_DW_t_opt.loc[t, 0] * dt
            for t in setT1
        )
        - BigMOpt_mdl.sum((reg_forecast[t]) * delta_tilde_P_HPP_t[t] * dt for t in setT)
        + BigMOpt_mdl.sum(
            ad * EBESS * mu * ((P_tilde_HA_dis_t[t] + P_tilde_HA_cha_t[t])) * dt
            for t in setT
        )
        + BigMOpt_mdl.sum(C_dev * (tau_t[t]) * dt for t in setT)
        + BigMOpt_mdl.sum(
            bigM[t]
            * (
                w_alpha1_t[t]
                + w_beta1_t[t]
                + w_alpha2_t[t]
                + w_beta2_t[t]
                + w_m_t[t]
                + w_n_t[t]
                + w_l1_t[t]
                + w_l2_t[t]
                + w_mu_t[t]
                + w_v_t[t]
                + w_p_t[t]
                + w_q_t[t]
            )
            for t in setT
        )
    )

    # BigMOpt_mdl.maximize(Revenue - Deg_cost - 1e7*an_var)
    BigMOpt_mdl.minimize(minus_profit)
    # BigMOpt_mdl.maximize(BigMOpt_mdl.sum(alpha1_s[t] for t in setTV))
    # BigMOpt_mdl.parameters.mip.tolerances.mipgap=0.5
    BigMOpt_mdl.parameters.timelimit = 100
    # Solve DualSubSubOpt Model
    BigMOpt_mdl.print_information()
    # BigMOpt_mdl.parameters.mip.strategy.heuristicfreq = 1  # Frequency of applying heuristics (1 = always)
    # BigMOpt_mdl.parameters.mip.strategy.rinsheur = 1
    # BigMOpt_mdl.parameters.preprocessing.presolve = 2
    # Set the RINSHeur parameter
    # BigMOpt_mdl.parameters.mip.strategy.rinsheur = 1  # Enable RINS heuristic
    sol = BigMOpt_mdl.solve()
    aa = BigMOpt_mdl.get_solve_details()
    print(aa.status)
    print(BigMOpt_mdl.solve_details.mip_relative_gap)

    if sol:
        w_alpha1_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(w_alpha1_t), orient="index"
        )
        w_beta1_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(w_beta1_t), orient="index"
        )
        w_alpha2_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(w_alpha2_t), orient="index"
        )
        w_beta2_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(w_beta2_t), orient="index"
        )
        # w_omega_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_omega_t), orient='index')
        w_m_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_m_t), orient="index")
        w_n_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_n_t), orient="index")
        w_l1_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_l1_t), orient="index")
        w_l2_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_l2_t), orient="index")
        w_mu_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_mu_t), orient="index")
        w_v_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_v_t), orient="index")
        w_p_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_p_t), orient="index")
        w_q_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_q_t), orient="index")
        # w_theta1_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_theta1_t), orient='index')
        # w_theta2_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_theta2_t), orient='index')
        # w_omega1_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_omega1_t), orient='index')
        # w_omega2_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_omega2_t), orient='index')
        # w_y_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(w_y_t), orient='index')
        obj = sol.get_objective_value()
        delta_tilde_P_HPP_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(delta_tilde_P_HPP_t), orient="index"
        )
        # delta_tilde_P_HPP_DW_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_tilde_P_HPP_DW_t), orient='index')
        P_tilde_w_HA_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(P_tilde_w_HA_t), orient="index"
        )
        # delta_tilde_P_b_DA_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(delta_tilde_P_b_DA_t), orient='index')
        # tau_t_opt = pd.DataFrame.from_dict(sol.get_value_dict(tau_t), orient='index')

        w_opt = pd.concat(
            [
                w_alpha1_t_opt,
                w_alpha2_t_opt,
                w_beta1_t_opt,
                w_beta2_t_opt,
                w_m_t_opt,
                w_n_t_opt,
                w_l1_t_opt,
                w_l2_t_opt,
                w_mu_t_opt,
                w_v_t_opt,
                w_p_t_opt,
                w_q_t_opt,
            ],
            axis=1,
        )

    else:
        w_opt = pd.DataFrame(
            [
                "alpha1",
                "alpha2",
                "beta1",
                "beta2",
                "m",
                "n",
                "l1",
                "l2",
                "mu",
                "v",
                "p",
                "q",
            ]
        )
    w_opt.columns = [
        "alpha1",
        "alpha2",
        "beta1",
        "beta2",
        "m",
        "n",
        "l1",
        "l2",
        "mu",
        "v",
        "p",
        "q",
    ]
    return obj, w_opt


def BMOpt(parameter_dict, simulation_dict, dynamic_inputs, verbose=False):

    day_num = dynamic_inputs["day_num"]
    Emax = dynamic_inputs["Emax"]
    ad = dynamic_inputs["ad"]
    P_HPP_SM_t_opt = dynamic_inputs["P_HPP_SM_t_opt"]
    start = dynamic_inputs["Current_hour"]
    P_HPP_UP_t0 = dynamic_inputs["P_HPP_UP_t0"]
    P_HPP_DW_t0 = dynamic_inputs["P_HPP_DW_t0"]
    s_UP_t = dynamic_inputs["s_UP_t"]
    s_DW_t = dynamic_inputs["s_DW_t"]
    SoC0 = dynamic_inputs["SoC0"]

    dt = parameter_dict["dispatch_interval"]
    dt_num = int(1 / dt)
    T = int(1 / dt * 24)

    ds = parameter_dict["settlement_interval"]
    dk = parameter_dict["offer_interval"]

    P_HPP_UP_t0 = P_HPP_UP_t0 * s_UP_t[start * dt_num : (start + 1) * dt_num]
    P_HPP_DW_t0 = P_HPP_DW_t0 * s_DW_t[start * dt_num : (start + 1) * dt_num]

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

    PUPMax = parameter_dict["max_up_bid"]
    PDWMax = parameter_dict["max_dw_bid"]
    PUPMin = parameter_dict["min_up_bid"]
    PDWMin = parameter_dict["min_dw_bid"]

    ReadData = DataReader(
        day_num=day_num,
        DI_num=dt_num,
        T=T,
        PsMax=PsMax,
        PwMax=PwMax,
        simulation_dict=simulation_dict,
    )
    Inputs = ReadData.execute()

    HA_wind_forecast = Inputs["HA_wind_forecast"]
    HA_solar_forecast = Inputs["HA_solar_forecast"]
    reg_forecast = Inputs["Reg_price_forecast"]
    SM_price_cleared = Inputs["SM_price_cleared"]
    xi_max = Inputs["HA_wind_forecast_error_ub"]
    xi_min = Inputs["HA_wind_forecast_error_lb"]

    UB = 1e10
    LB = -1e10
    Subobj_LB = LB
    iteration = 0
    time_limit = 100

    Cp = simulation_dict["Cp"]

    bigM = np.array([max(reg_forecast) * 100] * T)
    P_tilde_HPP_UP_t_opts = pd.DataFrame([])
    P_tilde_HPP_DW_t_opts = pd.DataFrame([])
    xi_tilde_t_opts = pd.DataFrame([])
    z_hat1_t_opts = pd.DataFrame([])
    z_hat2_t_opts = pd.DataFrame([])

    z_bigMs = pd.DataFrame([])
    w_opts = pd.DataFrame(
        [],
        columns=[
            "alpha1",
            "alpha2",
            "beta1",
            "beta2",
            "m",
            "n",
            "l1",
            "l2",
            "mu",
            "v",
            "p",
            "q",
        ],
    )

    UBs = list()
    LBs = list([LB])
    start_time = time.time()
    while iteration > -1 and iteration <= 15:

        (
            E_HA_t_opt,
            P_HPP_UP_t_opt,
            P_HPP_DW_t_opt,
            P_HPP_UP_k_opt,
            P_HPP_DW_k_opt,
            P_b_UP_t_opt,
            P_b_DW_t_opt,
            P_w_UP_t_opt,
            P_w_DW_t_opt,
            P_HPP_HA_t_opt,
            P_dis_HA_t_opt,
            P_cha_HA_t_opt,
            P_w_HA_t_opt,
            Masterobj,
            z_UP_bidlimit_t_opt,
            z_DW_bidlimit_t_opt,
        ) = MasterPro(
            dt,
            ds,
            dk,
            T,
            EBESS,
            PbMax,
            PwMax,
            PUPMax,
            PDWMax,
            PUPMin,
            PDWMin,
            P_grid_limit,
            SoCmin,
            SoCmax,
            Emax,
            eta_dis,
            eta_cha,
            mu,
            ad,
            HA_wind_forecast,
            HA_solar_forecast,
            xi_tilde_t_opts,
            P_HPP_UP_t0,
            P_HPP_DW_t0,
            P_HPP_SM_t_opt,
            reg_forecast,
            SoC0,
            iteration,
            deg_indicator,
            z_hat1_t_opts,
            z_hat2_t_opts,
            C_dev,
            start,
            Cp,
        )

        LB = Masterobj
        LBs.append(LB)
        Gap1 = UB - LB
        # print(Gap1)
        # if abs(UB-LB)/abs(LB) <1e-4 or UB-LB<=0:
        # if UB-LB <1e-3 :
        #   break

        # if math.isclose(LBs[-1] , LBs[-2], abs_tol=1e-3):
        #   time_limit = time_limit*2

        (
            xi_tilde_t_opt,
            P_tilde_HPP_UP_t_opt,
            P_tilde_HPP_DW_t_opt,
            z_hat1_t_opt,
            z_hat2_t_opt,
            delta_tilde_P_HPP_UP_t_opt,
            delta_tilde_P_HPP_DW_t_opt,
            P_tilde_w_HA_t_opt,
            P_tilde_dis_HA_t_opt,
            P_tilde_cha_HA_t_opt,
            Subobj_LB,
            Subobj_UB,
            z_bigM,
        ) = SubPro(
            dt,
            ds,
            dk,
            T,
            EBESS,
            PbMax,
            P_grid_limit,
            SoCmin,
            SoCmax,
            Emax,
            eta_dis,
            eta_cha,
            mu,
            ad,
            HA_wind_forecast,
            P_HPP_SM_t_opt,
            P_HPP_UP_t0,
            P_HPP_DW_t0,
            SoC0,
            deg_indicator,
            reg_forecast,
            bigM,
            start,
            P_HPP_UP_t_opt,
            P_HPP_DW_t_opt,
            xi_max,
            xi_min,
            C_dev,
            z_UP_bidlimit_t_opt,
            z_DW_bidlimit_t_opt,
            time_limit,
            Cp,
        )

        # obj_SubDual = SubDualPro(dt, ds, dk, T, EBESS, PbMax, PwMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad, HA_wind_forecast, P_HPP_HA_t_opt, P_HPP_SM_t_opt, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator, probability, BP_up_forecast, BP_dw_forecast, reg_forecast, SM_price_cleared, Cp, start, P_HPP_UP_t_opt, P_HPP_DW_t_opt, xi_max, xi_min, C_dev, error_C, error_d, z_UP_bidlimit_t_opt, z_DW_bidlimit_t_opt, time_limit, z_hat1_t_opt, z_hat2_t_opt, xi_tilde_t_opt)
        # obj_SubPrimal, delta_tilde_P_HPP_UP_t_opt, delta_tilde_P_HPP_DW_t_opt, P_tilde_w_HA_t_opt, P_tilde_dis_HA_t_opt, P_tilde_cha_HA_t_opt = SubPrimalPro(dt, ds, dk, T, EBESS, PbMax, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, mu, ad, HA_wind_forecast, P_HPP_SM_t_opt, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, deg_indicator, reg_forecast, Cp, start, P_HPP_UP_t_opt, P_HPP_DW_t_opt, C_dev, z_UP_bidlimit_t_opt, z_DW_bidlimit_t_opt, time_limit, z_hat1_t_opt, z_hat2_t_opt, xi_tilde_t_opt)

        # while Subobj_LB < LB:
        #      time_limit = time_limit*2
        #      xi_tilde_t_opt, P_tilde_HPP_UP_t_opt, P_tilde_HPP_DW_t_opt, z_hat1_t_opt, z_hat2_t_opt, delta_tilde_P_HPP_UP_t_opt, delta_tilde_P_HPP_DW_t_opt, P_tilde_w_HA_t_opt, P_tilde_dis_HA_t_opt, P_tilde_cha_HA_t_opt, Subobj_LB, Subobj_UB, z_bigM = SubPro(dt, ds, dk, T, EBESS, PbMax, PwMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad, HA_wind_forecast, P_HPP_HA_t_opt, P_HPP_SM_t_opt, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator, probability, BP_up_forecast, BP_dw_forecast, reg_forecast, SM_price_cleared, bigM, start, P_HPP_UP_t_opt, P_HPP_DW_t_opt, xi_max, xi_min, C_dev, error_C, error_d, z_UP_bidlimit_t_opt, z_DW_bidlimit_t_opt, time_limit)

        UB = min(UB, Subobj_LB)
        UBs.append(UB)
        print(
            f"####################################################External gap: {UB-LB}  ###################################"
        )
        # print(UB-LB)
        Gap2 = UB - LB

        # BigMobj, w_opt = BigMPro(dt, ds, dk, T, EBESS, PbMax, PwMax, PreUp, PreDw, P_grid_limit, SoCmin, SoCmax, Emax, eta_dis, eta_cha, eta_leak, mu, ad, HA_wind_forecast, P_HPP_HA_t_opt, P_HPP_SM_t_opt, P_HPP_UP_t0, P_HPP_DW_t0, SoC0, exten_num, deg_indicator, probability, reg_forecast, bigM, P_HPP_UP_t_opt, P_HPP_DW_t_opt, xi_max, xi_min, C_dev, start, xi_tilde_t_opt, P_tilde_HPP_UP_t_opt, P_tilde_HPP_DW_t_opt, z_hat1_t_opt, z_hat2_t_opt)
        # w_opts = pd.concat([w_opts, w_opt])
        # step 3: Expand vertex set

        P_tilde_HPP_UP_t_opts = pd.concat(
            [P_tilde_HPP_UP_t_opts, P_tilde_HPP_UP_t_opt], axis=1
        )
        P_tilde_HPP_DW_t_opts = pd.concat(
            [P_tilde_HPP_DW_t_opts, P_tilde_HPP_DW_t_opt], axis=1
        )
        xi_tilde_t_opts = pd.concat([xi_tilde_t_opts, xi_tilde_t_opt], axis=1)
        z_hat1_t_opts = pd.concat([z_hat1_t_opts, z_hat1_t_opt], axis=1)
        z_hat2_t_opts = pd.concat([z_hat2_t_opts, z_hat2_t_opt], axis=1)

        z_bigMs = pd.concat([z_bigMs, z_bigM])

        if math.isclose(LB, 0, abs_tol=1e-4):
            if UB - LB <= 1e-4:
                break
        else:
            if abs(UB - LB) / abs(LB) < 5e-4 or UB - LB <= 0:
                break

        iteration = iteration + 1
    end_time = time.time()
    run_time = end_time - start_time

    print(f"Total run time: {run_time} seconds")
    obj = LB

    E_HPP_HA_t_opt = P_HPP_HA_t_opt * dt
    SoC_HA_t_opt = E_HA_t_opt / Emax

    return P_HPP_HA_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt
