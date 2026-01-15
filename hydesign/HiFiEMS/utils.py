import math
from datetime import datetime

import numpy as np
import pandas as pd
from docplex.mp.model import Model


class DataReaderBase:
    def __init__(self, day_num, DI_num, T, PsMax, PwMax, simulation_dict):
        self.day_num = day_num
        self.DI_num = DI_num
        self.T = T
        self.PsMax = PsMax
        self.PwMax = PwMax
        self.sim = simulation_dict

    def execute(self):
        T, DI_num, sim = self.T, self.DI_num, self.sim
        PwMax, PsMax = self.PwMax, self.PsMax

        day_num_start = (
            datetime.strptime(self.sim["start_date"], "%m/%d/%y").timetuple().tm_yday
        )
        skips1 = ((self.day_num - 1 + day_num_start - 1) * T) % (359 * T)
        skips2 = ((self.day_num - 1 + day_num_start - 1) * 24) % (359 * 24)

        Wind_data = sim["wind_df"].iloc[skips1 : skips1 + T].reset_index(drop=True)
        Solar_data = sim["solar_df"].iloc[skips1 : skips1 + T].reset_index(drop=True)
        Market_data = (
            sim["market_df"]
            .iloc[skips2 : skips2 + int(T / DI_num)]
            .reset_index(drop=True)
        )

        self.Wind_data = Wind_data
        self.Solar_data = Solar_data
        self.Market_data = Market_data

        Wind_measurement = (
            Wind_data["Measurement"] * PwMax if "Measurement" in Wind_data else None
        )
        Solar_measurement = (
            Solar_data["Measurement"] * PsMax if "Measurement" in Solar_data else None
        )

        DA_wind_forecast = Wind_data[sim["DA_wind"]] * PwMax
        HA_wind_forecast = Wind_data[sim["HA_wind"]] * PwMax
        RT_wind_forecast = Wind_data[sim["FMA_wind"]] * PwMax

        DA_solar_forecast = Solar_data[sim["DA_solar"]] * PsMax
        HA_solar_forecast = Solar_data[sim["HA_solar"]] * PsMax
        RT_solar_forecast = Solar_data[sim["FMA_solar"]] * PsMax

        SM_price_forecast = Market_data[sim["SP"]]
        SM_price_cleared = Market_data["SM_cleared"]
        Reg_price_forecast = Market_data[sim["RP"]]
        Reg_price_cleared = Market_data["reg_cleared"]

        BM_dw_price_forecasts = []
        BM_up_price_forecasts = []
        reg_up_sign_forecasts = []
        reg_dw_sign_forecasts = []

        for i in range(int(T / DI_num)):
            reg_p = Reg_price_forecast.iloc[i]
            sm_p = SM_price_cleared.iloc[i]
            if reg_p > sm_p:
                up, dw, up_sign, dw_sign = reg_p, sm_p, 1, 0
            elif reg_p < sm_p:
                up, dw, up_sign, dw_sign = sm_p, reg_p, 0, 1
            else:
                up = dw = sm_p
                up_sign = dw_sign = 0

            BM_dw_price_forecasts.append({"Up": up})
            BM_up_price_forecasts.append({"Down": dw})
            reg_up_sign_forecasts.append({"up_sign": up_sign})
            reg_dw_sign_forecasts.append({"dw_sign": dw_sign})

        BM_dw_price_forecast = pd.DataFrame(BM_dw_price_forecasts).squeeze()
        BM_up_price_forecast = pd.DataFrame(BM_up_price_forecasts).squeeze()
        reg_up_sign_forecast = pd.DataFrame(reg_up_sign_forecasts).squeeze()
        reg_dw_sign_forecast = pd.DataFrame(reg_dw_sign_forecasts).squeeze()

        if sim.get("BP") == 2:
            BM_dw_price_forecast = Market_data["BM_Down_cleared"]
            BM_up_price_forecast = Market_data["BM_Up_cleared"]

        return {
            "DA_wind_forecast": DA_wind_forecast,
            "HA_wind_forecast": HA_wind_forecast,
            "RT_wind_forecast": RT_wind_forecast,
            "DA_solar_forecast": DA_solar_forecast,
            "HA_solar_forecast": HA_solar_forecast,
            "RT_solar_forecast": RT_solar_forecast,
            "SM_price_forecast": SM_price_forecast,
            "SM_price_cleared": SM_price_cleared,
            "Wind_measurement": Wind_measurement,
            "Solar_measurement": Solar_measurement,
            "BM_dw_price_forecast": BM_dw_price_forecast,
            "BM_up_price_forecast": BM_up_price_forecast,
            "BM_dw_price_cleared": Market_data["BM_Down_cleared"],
            "BM_up_price_cleared": Market_data["BM_Up_cleared"],
            "reg_up_sign_forecast": reg_up_sign_forecast,
            "reg_dw_sign_forecast": reg_dw_sign_forecast,
            "reg_vol_up": Market_data["reg_vol_Up"],
            "reg_vol_dw": Market_data["reg_vol_Down"],
            "Reg_price_forecast": Reg_price_forecast,
            "Reg_price_cleared": Reg_price_cleared,
            "time_index": Wind_data["time"],
        }


def f_xmin_to_ymin(x, reso_x, reso_y):  # x: dataframe reso: in hour
    x = np.asarray(x).squeeze()
    y = pd.DataFrame()
    if reso_y > reso_x:
        a = 0
        num = int(reso_y / reso_x)

        for ii in range(len(x)):
            if ii % num == num - 1:
                a = (a + x[ii]) / num
                y = y.append(pd.DataFrame([a]))
                a = 0
            else:
                a = a + x[ii]
    else:
        y = pd.DataFrame(np.repeat(x, int(reso_x / reso_y)))
        num = int(reso_x / reso_y)
    y.index = range(int(len(x) / num))
    return y


def _revenue_calculation(
    parameter_dict,
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
):
    DI = parameter_dict["dispatch_interval"]
    DI_num = int(1 / DI)

    SI = parameter_dict["settlement_interval"]
    SI_num = int(1 / SI)

    # Spot market revenue
    SM_price_cleared_DI = SM_price_cleared.repeat(DI_num)
    SM_revenue = P_HPP_SM_t_opt.squeeze() * SM_price_cleared_DI * DI

    # Regulation revenue
    BM_up_price_cleared_DI = BM_up_price_cleared.repeat(DI_num)
    BM_dw_price_cleared_DI = BM_dw_price_cleared.repeat(DI_num)

    s_UP_t = pd.Series(s_UP_t)
    s_DW_t = pd.Series(s_DW_t)

    reg_revenue = (s_UP_t * P_HPP_UP_bid_ts.squeeze() * DI * BM_up_price_cleared_DI) - (
        s_DW_t * P_HPP_DW_bid_ts.squeeze() * BI * BM_dw_price_cleared_DI
    )

    # Imbalance revenue
    BM_up_price_cleared_SI = BM_up_price_cleared.repeat(SI_num)
    BM_dw_price_cleared_SI = BM_dw_price_cleared.repeat(SI_num)
    P_HPP_RT_ts_15min = f_xmin_to_ymin(P_HPP_RT_ts, DI, 1 / 4)
    P_HPP_RT_refs_15min = f_xmin_to_ymin(P_HPP_RT_refs, DI, 1 / 4)

    power_imbalance = pd.Series(
        (P_HPP_RT_ts_15min.values - P_HPP_RT_refs_15min.values)[:, 0]
    )

    pos_imbalance = power_imbalance.apply(lambda x: x if x > 0 else 0)
    neg_imbalance = power_imbalance.apply(lambda x: x if x < 0 else 0)

    im_revenue = np.asarray(pos_imbalance * SI * BM_dw_price_cleared_SI) + np.asarray(
        neg_imbalance * SI * BM_up_price_cleared_SI
    )

    # imbalance fee

    im_power_cost_DK1 = abs(power_imbalance * SI) * parameter_dict["imbalance_fee"]

    # Balancing market revenue
    BM_revenue = reg_revenue + im_revenue - im_power_cost_DK1
    return SM_revenue, reg_revenue, im_revenue, BM_revenue, im_power_cost_DK1


def Revenue_calculation(
    parameter_dict,
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
    SM_revenue, reg_revenue, im_revenue, BM_revenue, im_power_cost_DK1 = (
        _revenue_calculation(
            parameter_dict,
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
    )

    return (
        SM_revenue.sum(),
        reg_revenue.sum(),
        im_revenue.sum(),
        BM_revenue.sum(),
        im_power_cost_DK1.sum(),
    )


def get_var_value_from_sol(x, sol):

    y = {}

    for key, var in x.items():
        y[key] = sol.get_var_value(var)

    y = pd.DataFrame.from_dict(y, orient="index")

    return y


def RTSim(
    dt,
    PbMax,
    PreUp,
    PreDw,
    P_grid_limit,
    SoCmin,
    SoCmax,
    Emax,
    eta_dis,
    eta_cha,
    eta_leak,
    Wind_measurement,
    Solar_measurement,
    RT_wind_forecast,
    RT_solar_forecast,
    SoC0,
    P_HPP_t0,
    start,
    P_activated_UP_t,
    P_activated_DW_t,
    verbose=False,
):
    # RES_error = Wind_measurement[start] + Solar_measurement[start] - RT_wind_forecast[start] - RT_solar_forecast[start]

    eta_cha_ha = eta_cha ** (dt)
    eta_dis_ha = eta_dis ** (dt)
    eta_leak_ha = 1 - (1 - eta_leak) ** (dt)

    # Optimization modelling by CPLEX
    set_SoCT = [0, 1]
    RTSim_mdl = Model()
    # Define variables (must define lb and ub, otherwise may cause issues on cplex)
    P_W_RT_t = RTSim_mdl.continuous_var(
        lb=0, ub=Wind_measurement[start], name="HA Wind schedule"
    )
    P_S_RT_t = RTSim_mdl.continuous_var(
        lb=0, ub=Solar_measurement[start], name="HA Solar schedule"
    )
    P_HPP_RT_t = RTSim_mdl.continuous_var(
        lb=-P_grid_limit, ub=P_grid_limit, name="HA schedule without balancing bidding"
    )
    P_dis_RT_t = RTSim_mdl.continuous_var(lb=0, ub=PbMax, name="HA discharge")
    P_cha_RT_t = RTSim_mdl.continuous_var(lb=0, ub=PbMax, name="HA charge")
    P_b_RT_t = RTSim_mdl.continuous_var(
        lb=-PbMax, ub=PbMax, name="HA Battery schedule"
    )  # (must define lb and ub, otherwise may cause unknown issues on cplex)
    SoC_RT_t = RTSim_mdl.continuous_var_dict(
        set_SoCT, lb=SoCmin, ub=SoCmax, name="HA SoC"
    )
    z_t = RTSim_mdl.binary_var(name="Cha or Discha")

    # Define constraints

    RTSim_mdl.add_constraint(P_HPP_RT_t == P_W_RT_t + P_S_RT_t + P_b_RT_t)
    RTSim_mdl.add_constraint(P_b_RT_t == P_dis_RT_t - P_cha_RT_t)
    RTSim_mdl.add_constraint(P_dis_RT_t <= (PbMax - PreUp) * z_t)
    RTSim_mdl.add_constraint(P_cha_RT_t <= (PbMax - PreDw) * (1 - z_t))
    RTSim_mdl.add_constraint(
        SoC_RT_t[1]
        == SoC_RT_t[0] * (1 - eta_leak_ha)
        - 1 / Emax * P_dis_RT_t / eta_dis_ha * dt
        + 1 / Emax * P_cha_RT_t * eta_cha_ha * dt
    )
    RTSim_mdl.add_constraint(SoC_RT_t[0] <= SoCmax)
    RTSim_mdl.add_constraint(SoC_RT_t[0] >= SoCmin)
    RTSim_mdl.add_constraint(P_HPP_RT_t <= P_grid_limit - PreUp)
    RTSim_mdl.add_constraint(P_HPP_RT_t >= -P_grid_limit + PreDw)
    RTSim_mdl.add_constraint(SoC_RT_t[0] == SoC0)

    if math.isclose(P_activated_UP_t, 0, abs_tol=1e-5) and math.isclose(
        P_activated_DW_t, 0, abs_tol=1e-5
    ):
        obj = 1e5 * (
            Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t - P_S_RT_t
        ) + (P_HPP_RT_t - P_HPP_t0) * (P_HPP_RT_t - P_HPP_t0)
    else:
        obj = (
            Wind_measurement[start] + Solar_measurement[start] - P_W_RT_t - P_S_RT_t
        ) + 1e5 * (P_HPP_RT_t - P_HPP_t0) * (P_HPP_RT_t - P_HPP_t0)

    RTSim_mdl.minimize(obj)

    # Solve BMOpt Model
    sol = RTSim_mdl.solve()
    if verbose:
        RTSim_mdl.print_information()
        aa = RTSim_mdl.get_solve_details()
        print(aa.status)
    if sol:
        #    SMOpt_mdl.print_solution()
        # imbalance_RT_to_ref = sol.get_objective_value() * dt
        P_HPP_RT_t_opt = sol.get_value(P_HPP_RT_t)
        P_W_RT_t_opt = sol.get_value(P_W_RT_t)
        P_S_RT_t_opt = sol.get_value(P_S_RT_t)
        P_dis_RT_t_opt = sol.get_value(P_dis_RT_t)
        P_cha_RT_t_opt = sol.get_value(P_cha_RT_t)
        SoC_RT_t_opt = pd.DataFrame.from_dict(
            sol.get_value_dict(SoC_RT_t), orient="index"
        )
        E_HPP_RT_t_opt = P_HPP_RT_t_opt * dt

        RES_RT_cur_t_opt = (
            Wind_measurement[start]
            + Solar_measurement[start]
            - P_W_RT_t_opt
            - P_S_RT_t_opt
        )
        # P_W_RT_cur_t_opt = Wind_measurement[start] - P_W_RT_t_opt
        # P_W_RT_cur_t_opt = pd.DataFrame(P_W_RT_cur_t_opt)
        # P_S_RT_cur_t_opt = Solar_measurement[start] - P_S_RT_t_opt
        # P_S_RT_cur_t_opt = pd.DataFrame(P_S_RT_cur_t_opt)

        # z_t_opt = sol.get_value(z_t)

    else:
        print("RTOpt has no solution")
        # print(SMOpt_mdl.export_to_string())
    return (
        E_HPP_RT_t_opt,
        P_HPP_RT_t_opt,
        P_dis_RT_t_opt,
        P_cha_RT_t_opt,
        SoC_RT_t_opt,
        RES_RT_cur_t_opt,
        P_W_RT_t_opt,
        P_S_RT_t_opt,
    )
