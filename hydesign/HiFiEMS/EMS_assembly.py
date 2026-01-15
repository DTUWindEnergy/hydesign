import importlib
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

import hydesign.HiFiEMS.Deg_Calculation as DegCal
import hydesign.HiFiEMS.utils as utils
from hydesign.HiFiEMS import HIFIEMS_PACKAGE
from hydesign.HiFiEMS.utils import DataReaderBase


class EMS:
    def __init__(self, config):
        self.config = config
        self.ems_models = {}
        self._load_opts()
        self._validate_config()

    def _load_opts(self):
        for opt_type in list(self.config.keys()):
            ems_type = self.config.get(opt_type)
            if not ems_type:
                self.ems_models[opt_type] = None
            else:
                try:
                    module = importlib.import_module(f"{HIFIEMS_PACKAGE}.{ems_type}")
                    self.ems_models[opt_type] = getattr(module, opt_type)
                except (ModuleNotFoundError, AttributeError) as e:
                    raise RuntimeError(
                        f"Failed to load {opt_type} from '{ems_type}.py': {e}"
                    )

    def _validate_config(self):
        if self.config.get("SMOpt") == None:
            raise ValueError("Invalid config: SMOpt is required.")

    def run(self, parameter_dict, simulation_dict):

        DI = parameter_dict["dispatch_interval"]
        DI_num = int(1 / DI)
        T = int(1 / DI * 24)

        SI = parameter_dict["settlement_interval"]
        SI_num = int(1 / SI)
        T_SI = int(24 / SI)
        SIDI_num = int(SI / DI)

        BI = parameter_dict["offer_interval"]
        # BI_num = int(1/BI)
        # T_BI = int(24/BI)

        PwMax = parameter_dict["wind_capacity"]
        PsMax = parameter_dict["solar_capacity"]
        EBESS = parameter_dict["battery_energy_capacity"]
        PbMax = parameter_dict["battery_power_capacity"]
        SoCmin = parameter_dict["battery_minimum_SoC"]
        SoCmax = parameter_dict["battery_maximum_SoC"]
        SoCini = parameter_dict["battery_initial_SoC"]
        eta_dis = parameter_dict["battery_hour_discharge_efficiency"]
        eta_cha = parameter_dict["battery_hour_charge_efficiency"]
        eta_leak = parameter_dict["battery_self_discharge_efficiency"]
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
        ad = 1e-7  # slope
        capital_cost = parameter_dict["battery_capital_cost"]  # Euro/MWh
        replace_percent = 0.2
        total_cycles = 3500

        PreUp = PreDw = 0
        P_grid_limit = parameter_dict["hpp_grid_connection"]

        mu = parameter_dict["battery_marginal_degradation_cost"]

        deg_indicator = parameter_dict["degradation_in_optimization"]

        C_dev = parameter_dict["imbalance_fee"]

        # SoC_all = pd.DataFrame(columns = ['SoC_all'])

        exten_num = 0
        out_dir = simulation_dict["out_dir"]
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        re = pd.DataFrame(
            list(),
            columns=[
                "SM_revenue",
                "reg_revenue",
                "im_revenue",
                "im_special_revenue_DK1",
                "Deg_cost",
                "Deg_cost_by_cycle",
            ],
        )
        sig = pd.DataFrame(list(), columns=["signal_up", "signal_down"])
        cur = pd.DataFrame(list(), columns=["RES_cur"])
        de = pd.DataFrame(list(), columns=["nld", "ld", "cycles"])
        ei = pd.DataFrame(list(), columns=["energy_imbalance"])
        # reg = pd.DataFrame(list(), columns=['bid_up','bid_dw','w_up','w_dw','b_up','b_dw'])
        shc = pd.DataFrame(
            list(),
            columns=["SM", "dis_SM", "cha_SM", "w_SM", "RT", "Ref", "dis_RT", "cha_RT"],
        )
        slo = pd.DataFrame([ad], columns=["slope"])
        soc = pd.DataFrame(list(), columns=["SoC"])
        # bounds = pd.DataFrame(list(), columns=['UB','LB'])
        # worst_reg = pd.DataFrame(list(), columns=['up','down'])
        # worst_wind = pd.DataFrame(list(), columns=['wind'])
        # times = pd.DataFrame(list(), columns=['time-1','time12'])

        sig.to_csv(out_dir + "act_signal.csv", index=False)
        cur.to_csv(out_dir + "curtailment.csv", index=False)
        de.to_csv(out_dir + "Degradation.csv", index=False)
        ei.to_csv(out_dir + "energy_imbalance.csv", index=False)
        # reg.to_csv(out_dir+'reg_bids.csv',index=False)
        re.to_csv(out_dir + "revenue.csv", index=False)
        shc.to_csv(out_dir + "schedule.csv", index=False)
        slo.to_csv(out_dir + "slope.csv", index=False)
        soc.to_csv(out_dir + "SoC.csv", index=False)
        # bounds.to_csv(out_dir+'bounds.csv',index=False)
        # worst_reg.to_csv(out_dir+'worst_reg.csv',index=False)
        # worst_wind.to_csv(out_dir+'worst_wind.csv',index=False)
        # times.to_csv(out_dir+'time.csv',index=False)
        P_HPP_SM_t_opt_list = []
        SM_price_cleared_list = []
        BM_dw_price_cleared_list = []
        BM_up_price_cleared_list = []
        P_HPP_RT_ts_list = []
        P_HPP_RT_refs_list = []
        P_HPP_UP_bid_ts_list = []
        P_HPP_DW_bid_ts_list = []
        s_UP_t_list = []
        s_DW_t_list = []
        residual_imbalance_list = []
        RES_RT_cur_ts_list = []
        P_dis_RT_ts_list = []
        P_cha_RT_ts_list = []
        SoC_ts_list = []

        pbar = tqdm(total=simulation_dict["number_of_run_day"] + 1)
        while day_num:
            pbar.update(1)
            Emax = EBESS * (1 - pre_nld)
            P_HPP_UP_t0 = 0
            P_HPP_DW_t0 = 0
            P_HPP_UP_t1 = 0
            P_HPP_DW_t1 = 0
            # if not self.ems_modules.get('SMOpt'):
            #     module = self.ems_modules.get('SMOpt')
            #     DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = module.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)
            # elif EMStype == "SEMS":
            #     DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index, HA_wind_forecast_scenario, probability_wind = EMS.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)

            # if simulation_dict['price_scenario_fn'] == None:
            #     probability_price, SP_scenario, RP_scenario = scenario_generation(PsMax, PwMax, T, DI_num, simulation_dict)

            #     probability = np.zeros(len(probability_wind)*len(probability_price))

            #     # Produce the final probability. Here assume the price uncertainty is independent with wind uncertainty
            #     for i in range(len(probability_wind)):
            #         for j in range(len(probability_price)):
            #             probability[i*len(probability_price)+j] = probability_price[j] * probability_wind[i]
            #     HA_wind_forecast_scenario = np.repeat(HA_wind_forecast_scenario, len(probability_price), axis=0)
            #     SP_scenario = np.matlib.repmat(SP_scenario,len(probability_wind),1)
            #     RP_scenario = np.matlib.repmat(RP_scenario,len(probability_wind),1)
            # else:
            #     DA_wind_forecast, HA_wind_forecast, RT_wind_forecast, DA_solar_forecast, HA_solar_forecast, RT_solar_forecast, SM_price_forecast, SM_price_cleared, Wind_measurement, Solar_measurement, BM_dw_price_forecast, BM_up_price_forecast, BM_dw_price_cleared, BM_up_price_cleared, reg_up_sign_forecast, reg_dw_sign_forecast, reg_vol_up, reg_vol_dw, Reg_price_cleared, time_index = EMS.ReadData(day_num, exten_num, DI_num, T, PsMax, PwMax, simulation_dict)

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

            dynamic_inputs = {
                "day_num": day_num,
                "SoC0": SoC0,
                "Emax": Emax,
                "ad": ad,
                "P_HPP_UP_t0": P_HPP_UP_t0,
                "P_HPP_DW_t0": P_HPP_DW_t0,
                "P_HPP_UP_t1": P_HPP_UP_t1,
                "P_HPP_DW_t1": P_HPP_DW_t1,
                "RDOpt_mFRREAM_enabler": True,
                "s_UP_t": s_UP_t,
                "s_DW_t": s_DW_t,
            }

            ReadData = DataReaderBase(
                day_num=day_num,
                DI_num=DI_num,
                T=T,
                PsMax=PsMax,
                PwMax=PwMax,
                simulation_dict=simulation_dict,
            )
            Inputs = ReadData.execute()

            reg_vol_up = Inputs["reg_vol_up"]
            reg_vol_dw = Inputs["reg_vol_dw"]
            RT_wind_forecast = Inputs["RT_wind_forecast"]
            RT_solar_forecast = Inputs["RT_solar_forecast"]
            Wind_measurement = Inputs["Wind_measurement"]
            Solar_measurement = Inputs["Solar_measurement"]
            SM_price_cleared = Inputs["SM_price_cleared"]
            BM_up_price_cleared = Inputs["BM_up_price_cleared"]
            BM_dw_price_cleared = Inputs["BM_dw_price_cleared"]
            # Call EMS Model
            # Run SMOpt
            if callable(self.ems_models.get("SMOpt")):
                SMOpt = self.ems_models.get("SMOpt")

                (
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
                ) = SMOpt(
                    parameter_dict, simulation_dict, dynamic_inputs, verbose=False
                )

                # P_HPP_SM_t_opt.index = time_index[:T]
                P_HPP_SM_t_opt.index = range(T)

                P_HPP_RT_ref = P_HPP_SM_t_opt.iloc[0, 0]

                dynamic_inputs["P_HPP_SM_t_opt"] = P_HPP_SM_t_opt

            if callable(self.ems_models.get("BMOpt")) and callable(
                self.ems_models.get("RDOpt")
            ):
                BMOpt = self.ems_models.get("BMOpt")
                RDOpt = self.ems_models.get("RDOpt")
                for i in range(0, 24):
                    # BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
                    # BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
                    # BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
                    # BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))

                    # BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
                    # BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
                    # BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
                    # BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))

                    if reg_vol_up[i] > 0 and reg_vol_dw[i] < 0:
                        if P_HPP_UP_t0 < reg_vol_up[i]:
                            s_UP_t[i * DI_num : int((i + 1 / 2) * DI_num)] = 1
                            s_DW_t[i * DI_num : int((i + 1 / 2) * DI_num)] = 0
                        if -P_HPP_DW_t0 > reg_vol_dw[i]:
                            s_DW_t[int((i + 1 / 2) * DI_num) : (i + 1) * DI_num] = 1
                            s_UP_t[int((i + 1 / 2) * DI_num) : (i + 1) * DI_num] = 0

                    else:
                        if P_HPP_UP_t0 < reg_vol_up[i]:
                            s_UP_t[i * DI_num : (i + 1) * DI_num] = 1
                            s_DW_t[i * DI_num : (i + 1) * DI_num] = 0
                        elif -P_HPP_DW_t0 > reg_vol_dw[i]:
                            s_UP_t[i * DI_num : (i + 1) * DI_num] = 0
                            s_DW_t[i * DI_num : (i + 1) * DI_num] = 1

                    # HA_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8 * (DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
                    # HA_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8 * (DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])

                    # Run BMOpt
                    dynamic_inputs["Current_hour"] = i
                    dynamic_inputs["P_HPP_UP_t0"] = P_HPP_UP_t0
                    dynamic_inputs["P_HPP_DW_t0"] = P_HPP_DW_t0
                    dynamic_inputs["SoC0"] = SoC0
                    dynamic_inputs["s_UP_t"] = s_UP_t
                    dynamic_inputs["s_DW_t"] = s_DW_t
                    P_HPP_HA_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt = BMOpt(
                        parameter_dict, simulation_dict, dynamic_inputs, verbose=False
                    )

                    if i < 24 - 1:
                        P_HPP_UP_t1 = P_HPP_UP_k_opt.loc[i + 1].iloc[0]
                        P_HPP_DW_t1 = P_HPP_DW_k_opt.loc[i + 1].iloc[0]
                        P_HPP_UP_bid_ts.iloc[(i + 1) * DI_num : (i + 2) * DI_num, 0] = (
                            P_HPP_UP_t1
                        )
                        P_HPP_DW_bid_ts.iloc[(i + 1) * DI_num : (i + 2) * DI_num, 0] = (
                            P_HPP_UP_t1
                        )
                    else:
                        P_HPP_UP_t1 = 0
                        P_HPP_DW_t1 = 0

                    # Run RTSim

                    (
                        E_HPP_RT_t_opt,
                        P_HPP_RT_t_opt,
                        P_dis_RT_t_opt,
                        P_cha_RT_t_opt,
                        SoC_RT_t_opt,
                        RES_RT_cur_t_opt,
                        P_W_RT_t_opt,
                        P_S_RT_t_opt,
                    ) = utils.RTSim(
                        DI,
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
                        P_HPP_RT_ref,
                        i * DI_num,
                        P_HPP_UP_t0,
                        P_HPP_DW_t0,
                    )

                    SoC_ts.append({"SoC": SoC0})
                    P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                    P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                    RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                    P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                    P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[1, 0]

                    exist_imbalance = (
                        P_HPP_RT_t_opt
                        - (
                            P_HPP_UP_t0 * s_UP_t[i * DI_num]
                            - P_HPP_DW_t0 * s_DW_t[i * DI_num]
                        )
                        - P_HPP_SM_t_opt.iloc[i * DI_num, 0]
                    ) * DI

                    if DI == 1 / 4:
                        residual_imbalance.append({"energy_imbalance": exist_imbalance})
                        exist_imbalance = 0

                    SoC0 = SoC_RT_t_opt.iloc[1, 0]

                    for j in range(1, DI_num):

                        RT_interval = i * DI_num + j

                        # RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):]-Wind_measurement.values[(i+2)*int(1/DI):])])
                        # RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement[(i+2)*int(1/DI):])])

                        # Run RDOpt
                        dynamic_inputs["SoC0"] = SoC0
                        dynamic_inputs["exist_imbalance"] = exist_imbalance
                        dynamic_inputs["Current_DI"] = RT_interval
                        dynamic_inputs["P_HPP_UP_t1"] = P_HPP_UP_t1
                        dynamic_inputs["P_HPP_DW_t1"] = P_HPP_DW_t1
                        P_HPP_RD_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt = RDOpt(
                            parameter_dict,
                            simulation_dict,
                            dynamic_inputs,
                            verbose=False,
                        )

                        # Run RTSim
                        (
                            E_HPP_RT_t_opt,
                            P_HPP_RT_t_opt,
                            P_dis_RT_t_opt,
                            P_cha_RT_t_opt,
                            SoC_RT_t_opt,
                            RES_RT_cur_t_opt,
                            P_W_RT_t_opt,
                            P_S_RT_t_opt,
                        ) = utils.RTSim(
                            DI,
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
                            P_HPP_RT_ref,
                            RT_interval,
                            P_HPP_UP_t0,
                            P_HPP_DW_t0,
                        )
                        SoC_ts.append({"SoC": SoC0})
                        P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                        P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                        RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                        P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                        P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                        if RT_interval < T - 1:
                            P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1, 0]

                        if RT_interval % SIDI_num == SIDI_num - 1:
                            exist_imbalance = (
                                exist_imbalance
                                + (
                                    P_HPP_RT_t_opt
                                    - (
                                        P_HPP_UP_t0 * s_UP_t[i * DI_num + j]
                                        - P_HPP_DW_t0 * s_DW_t[i * DI_num + j]
                                    )
                                    - P_HPP_SM_t_opt.iloc[RT_interval, 0]
                                )
                                * DI
                            )
                            residual_imbalance.append(
                                {"energy_imbalance": exist_imbalance}
                            )
                            exist_imbalance = 0
                        else:
                            exist_imbalance = (
                                exist_imbalance
                                + (
                                    P_HPP_RT_t_opt
                                    - (
                                        P_HPP_UP_t0 * s_UP_t[i * DI_num + j]
                                        - P_HPP_DW_t0 * s_DW_t[i * DI_num + j]
                                    )
                                    - P_HPP_SM_t_opt.iloc[RT_interval, 0]
                                )
                                * DI
                            )

                        SoC0 = SoC_RT_t_opt.iloc[1, 0]

                    P_HPP_UP_t0 = P_HPP_UP_t1
                    P_HPP_DW_t0 = P_HPP_DW_t1

            elif callable(self.ems_models.get("BMOpt")) and not callable(
                self.ems_models.get("RDOpt")
            ):
                BMOpt = self.ems_models.get("BMOpt")

                for i in range(0, 24):
                    # BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
                    # BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
                    # BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
                    # BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))

                    # BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
                    # BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
                    # BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
                    # BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))

                    if reg_vol_up[i] > 0 and reg_vol_dw[i] < 0:
                        if P_HPP_UP_t0 < reg_vol_up[i]:
                            s_UP_t[i * DI_num : int((i + 1 / 2) * DI_num)] = 1
                            s_DW_t[i * DI_num : int((i + 1 / 2) * DI_num)] = 0
                        if -P_HPP_DW_t0 > reg_vol_dw[i]:
                            s_DW_t[int((i + 1 / 2) * DI_num) : (i + 1) * DI_num] = 1
                            s_UP_t[int((i + 1 / 2) * DI_num) : (i + 1) * DI_num] = 0

                    else:
                        if P_HPP_UP_t0 < reg_vol_up[i]:
                            s_UP_t[i * DI_num : (i + 1) * DI_num] = 1
                            s_DW_t[i * DI_num : (i + 1) * DI_num] = 0
                        elif -P_HPP_DW_t0 > reg_vol_dw[i]:
                            s_UP_t[i * DI_num : (i + 1) * DI_num] = 0
                            s_DW_t[i * DI_num : (i + 1) * DI_num] = 1

                    # HA_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8 * (DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
                    # HA_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8 * (DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])

                    # Run BMOpt
                    dynamic_inputs["Current_hour"] = i
                    dynamic_inputs["P_HPP_UP_t0"] = P_HPP_UP_t0
                    dynamic_inputs["P_HPP_DW_t0"] = P_HPP_DW_t0
                    dynamic_inputs["SoC0"] = SoC0
                    dynamic_inputs["s_UP_t"] = s_UP_t
                    dynamic_inputs["s_DW_t"] = s_DW_t
                    P_HPP_HA_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt = BMOpt(
                        parameter_dict, simulation_dict, dynamic_inputs, verbose=False
                    )

                    if i < 24 - 1:
                        P_HPP_UP_t1 = P_HPP_UP_k_opt.loc[i + 1].iloc[0]
                        P_HPP_DW_t1 = P_HPP_DW_k_opt.loc[i + 1].iloc[0]
                        P_HPP_UP_bid_ts.iloc[(i + 1) * DI_num : (i + 2) * DI_num, 0] = (
                            P_HPP_UP_t1
                        )
                        P_HPP_DW_bid_ts.iloc[(i + 1) * DI_num : (i + 2) * DI_num, 0] = (
                            P_HPP_UP_t1
                        )
                    else:
                        P_HPP_UP_t1 = 0
                        P_HPP_DW_t1 = 0

                    # Run RTSim

                    (
                        E_HPP_RT_t_opt,
                        P_HPP_RT_t_opt,
                        P_dis_RT_t_opt,
                        P_cha_RT_t_opt,
                        SoC_RT_t_opt,
                        RES_RT_cur_t_opt,
                        P_W_RT_t_opt,
                        P_S_RT_t_opt,
                    ) = utils.RTSim(
                        DI,
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
                        P_HPP_RT_ref,
                        i * DI_num,
                        P_HPP_UP_t0,
                        P_HPP_DW_t0,
                    )

                    SoC_ts.append({"SoC": SoC0})
                    P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                    P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                    RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                    P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                    P_HPP_RT_ref = P_HPP_HA_t_opt.iloc[1, 0]

                    exist_imbalance = (
                        P_HPP_RT_t_opt
                        - (
                            P_HPP_UP_t0 * s_UP_t[i * DI_num]
                            - P_HPP_DW_t0 * s_DW_t[i * DI_num]
                        )
                        - P_HPP_SM_t_opt.iloc[i * DI_num, 0]
                    ) * DI

                    if DI == 1 / 4:
                        residual_imbalance.append({"energy_imbalance": exist_imbalance})
                        exist_imbalance = 0

                    SoC0 = SoC_RT_t_opt.iloc[1, 0]

                    for j in range(1, DI_num):
                        # BM_dw_price = BM_dw_price_forecast
                        # BM_up_price = BM_up_price_forecast
                        # BM_dw_price[i] = BM_dw_price_cleared[i]
                        # BM_up_price[i] = BM_up_price_cleared[i]

                        RT_interval = i * DI_num + j

                        # RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):]-Wind_measurement.values[(i+2)*int(1/DI):])])
                        # RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement[(i+2)*int(1/DI):])])

                        # Run RTSim
                        (
                            E_HPP_RT_t_opt,
                            P_HPP_RT_t_opt,
                            P_dis_RT_t_opt,
                            P_cha_RT_t_opt,
                            SoC_RT_t_opt,
                            RES_RT_cur_t_opt,
                            P_W_RT_t_opt,
                            P_S_RT_t_opt,
                        ) = utils.RTSim(
                            DI,
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
                            P_HPP_RT_ref,
                            RT_interval,
                            P_HPP_UP_t0,
                            P_HPP_DW_t0,
                        )
                        SoC_ts.append({"SoC": SoC0})
                        P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                        P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                        RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                        P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                        P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                        if RT_interval < T - 1:
                            P_HPP_RT_ref = P_HPP_HA_t_opt.loc[RT_interval + 1].iloc[0]

                        if RT_interval % SIDI_num == SIDI_num - 1:
                            exist_imbalance = (
                                exist_imbalance
                                + (
                                    P_HPP_RT_t_opt
                                    - (
                                        P_HPP_UP_t0 * s_UP_t[i * DI_num + j]
                                        - P_HPP_DW_t0 * s_DW_t[i * DI_num + j]
                                    )
                                    - P_HPP_SM_t_opt.iloc[RT_interval, 0]
                                )
                                * DI
                            )
                            residual_imbalance.append(
                                {"energy_imbalance": exist_imbalance}
                            )
                            exist_imbalance = 0
                        else:
                            exist_imbalance = (
                                exist_imbalance
                                + (
                                    P_HPP_RT_t_opt
                                    - (
                                        P_HPP_UP_t0 * s_UP_t[i * DI_num + j]
                                        - P_HPP_DW_t0 * s_DW_t[i * DI_num + j]
                                    )
                                    - P_HPP_SM_t_opt.iloc[RT_interval, 0]
                                )
                                * DI
                            )

                        SoC0 = SoC_RT_t_opt.iloc[1, 0]

                    P_HPP_UP_t0 = P_HPP_UP_t1
                    P_HPP_DW_t0 = P_HPP_DW_t1

            elif not callable(self.ems_models.get("BMOpt")) and callable(
                self.ems_models.get("RDOpt")
            ):
                RDOpt = self.ems_models.get("RDOpt")
                dynamic_inputs["RDOpt_mFRREAM_enabler"] = False
                for i in range(0, 24):
                    # RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*DI_num:i*DI_num+2], HA_wind_forecast.values[i*DI_num+2:(i+2)*DI_num], Wind_measurement.values[(i+2)*DI_num:] + 0.8*(DA_wind_forecast.values[(i+2)*DI_num:] - Wind_measurement.values[(i+2)*DI_num:])])
                    # RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*DI_num:i*DI_num+2], HA_solar_forecast.values[i*DI_num+2:(i+2)*DI_num], Solar_measurement.values[(i+2)*DI_num:] + 0.8*(DA_solar_forecast.values[(i+2)*DI_num:] - Solar_measurement.values[(i+2)*DI_num:])])

                    # BM_up_price_forecast_settle = BM_up_price_forecast.squeeze().repeat(SI_num)
                    # BM_up_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))
                    # BM_dw_price_forecast_settle = BM_dw_price_forecast.squeeze().repeat(SI_num)
                    # BM_dw_price_forecast_settle.index = range(T_SI + int(exten_num/SIDI_num))

                    # BM_up_price_cleared_settle = BM_up_price_cleared.squeeze().repeat(SI_num)
                    # BM_up_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))
                    # BM_dw_price_cleared_settle = BM_dw_price_cleared.squeeze().repeat(SI_num)
                    # BM_dw_price_cleared_settle.index = range(T_SI + int(exten_num/SIDI_num))

                    exist_imbalance = 0
                    dynamic_inputs["SoC0"] = SoC0
                    dynamic_inputs["exist_imbalance"] = exist_imbalance
                    dynamic_inputs["Current_DI"] = i * DI_num
                    P_HPP_RD_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt = RDOpt(
                        parameter_dict, simulation_dict, dynamic_inputs, verbose=False
                    )

                    # Run RTSim

                    (
                        E_HPP_RT_t_opt,
                        P_HPP_RT_t_opt,
                        P_dis_RT_t_opt,
                        P_cha_RT_t_opt,
                        SoC_RT_t_opt,
                        RES_RT_cur_t_opt,
                        P_W_RT_t_opt,
                        P_S_RT_t_opt,
                    ) = utils.RTSim(
                        DI,
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
                        P_HPP_RT_ref,
                        i * DI_num,
                        P_HPP_UP_t0,
                        P_HPP_DW_t0,
                    )

                    SoC_ts.append({"SoC": SoC0})
                    P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                    P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                    RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                    P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                    P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                    P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1, 0]

                    exist_imbalance = (
                        P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[i * DI_num, 0]
                    ) * DI

                    if DI == 1 / 4:
                        residual_imbalance.append({"energy_imbalance": exist_imbalance})
                        exist_imbalance = 0

                    SoC0 = SoC_RT_t_opt.iloc[1, 0]

                    for j in range(1, DI_num):

                        # RD_wind_forecast1 = pd.Series(np.r_[RT_wind_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_wind_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Wind_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_wind_forecast.values[(i+2)*int(1/DI):] - Wind_measurement.values[(i+2)*int(1/DI):])])
                        # RD_solar_forecast1 = pd.Series(np.r_[RT_solar_forecast.values[i*int(1/DI)+j:i*int(1/DI)+j+2], HA_solar_forecast.values[i*int(1/DI)+j+2:(i+2)*int(1/DI)], Solar_measurement.values[(i+2)*int(1/DI):] + 0.8*(DA_solar_forecast.values[(i+2)*int(1/DI):] - Solar_measurement.values[(i+2)*int(1/DI):])])

                        RT_interval = i * DI_num + j
                        # Run RDOpt

                        dynamic_inputs["SoC0"] = SoC0
                        dynamic_inputs["exist_imbalance"] = exist_imbalance
                        dynamic_inputs["Current_DI"] = RT_interval
                        P_HPP_RD_t_opt, P_HPP_UP_k_opt, P_HPP_DW_k_opt = RDOpt(
                            parameter_dict,
                            simulation_dict,
                            dynamic_inputs,
                            verbose=False,
                        )
                        # P_HPP_RD_t_opt = P_HPP_HA_t_opt
                        # P_dis_RD_t_opt = P_dis_HA_t_opt
                        # P_cha_RD_t_opt = P_cha_HA_t_opt

                        # Run RTSim
                        (
                            E_HPP_RT_t_opt,
                            P_HPP_RT_t_opt,
                            P_dis_RT_t_opt,
                            P_cha_RT_t_opt,
                            SoC_RT_t_opt,
                            RES_RT_cur_t_opt,
                            P_W_RT_t_opt,
                            P_S_RT_t_opt,
                        ) = utils.RTSim(
                            DI,
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
                            P_HPP_RT_ref,
                            RT_interval,
                            P_HPP_UP_t0,
                            P_HPP_DW_t0,
                        )
                        SoC_ts.append({"SoC": SoC0})
                        P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                        P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                        RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                        P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                        P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                        if RT_interval < T - 1:
                            P_HPP_RT_ref = P_HPP_RD_t_opt.iloc[1, 0]

                        if RT_interval % SIDI_num == SIDI_num - 1:
                            exist_imbalance = (
                                exist_imbalance
                                + (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[RT_interval, 0])
                                * DI
                            )
                            residual_imbalance.append(
                                {"energy_imbalance": exist_imbalance}
                            )
                            exist_imbalance = 0
                        else:
                            exist_imbalance = (
                                exist_imbalance
                                + (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[RT_interval, 0])
                                * DI
                            )

                        SoC0 = SoC_RT_t_opt.iloc[1, 0]
            else:
                for i in range(0, 24):
                    exist_imbalance = 0
                    for j in range(0, DI_num):
                        RT_interval = i * DI_num + j
                        # run RTSim

                        P_HPP_RT_ref = P_HPP_SM_t_opt.iloc[RT_interval, 0]

                        (
                            E_HPP_RT_t_opt,
                            P_HPP_RT_t_opt,
                            P_dis_RT_t_opt,
                            P_cha_RT_t_opt,
                            SoC_RT_t_opt,
                            RES_RT_cur_t_opt,
                            P_W_RT_t_opt,
                            P_S_RT_t_opt,
                        ) = utils.RTSim(
                            DI,
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
                            P_HPP_RT_ref,
                            RT_interval,
                            P_HPP_UP_t0,
                            P_HPP_DW_t0,
                        )

                        SoC_ts.append({"SoC": SoC0})
                        P_HPP_RT_ts.append({"RT": P_HPP_RT_t_opt})
                        P_HPP_RT_refs.append({"Ref": P_HPP_RT_ref})
                        RES_RT_cur_ts.append({"RES_cur": RES_RT_cur_t_opt})
                        P_dis_RT_ts.append({"dis_RT": P_dis_RT_t_opt})
                        P_cha_RT_ts.append({"cha_RT": P_cha_RT_t_opt})

                        exist_imbalance = (
                            exist_imbalance
                            + (P_HPP_RT_t_opt - P_HPP_SM_t_opt.iloc[RT_interval, 0])
                            * DI
                        )
                        residual_imbalance.append({"energy_imbalance": exist_imbalance})
                        SoC0 = SoC_RT_t_opt.iloc[1, 0]

            residual_imbalance = pd.DataFrame(residual_imbalance)
            P_HPP_RT_ts = pd.DataFrame(P_HPP_RT_ts)
            P_HPP_RT_refs = pd.DataFrame(P_HPP_RT_refs)
            P_dis_RT_ts = pd.DataFrame(P_dis_RT_ts)
            P_cha_RT_ts = pd.DataFrame(P_cha_RT_ts)
            RES_RT_cur_ts = pd.DataFrame(RES_RT_cur_ts)

            (
                SM_revenue,
                reg_revenue,
                im_revenue,
                BM_revenue,
                im_special_revenue_DK1,
            ) = utils.Revenue_calculation(
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
            )

            # SoC_all = pd.read_excel('results_run.xlsx', sheet_name = 'SoC', nrows=(day_num-1)*T, engine='openpyxl')
            SoC_all = pd.read_csv(out_dir + "SoC.csv")
            SoC_ts = pd.DataFrame(SoC_ts)
            if SoC_all.empty:
                SoC_all = SoC_ts
            else:
                SoC_all = pd.concat([SoC_all, SoC_ts])

            SoC_all = SoC_all.values.tolist()

            SoC_for_rainflow = SoC_all
            SoC_for_rainflow = [SoC_for_rainflow[i][0] for i in range(int(day_num * T))]

            ld, nld, ld1, nld1, rf_DoD, rf_SoC, rf_count, nld_t, cycles = (
                DegCal.Deg_Model(SoC_for_rainflow, Ini_nld, pre_nld, ld1, nld1, day_num)
            )

            Deg_cost = (nld - pre_nld) / replace_percent * EBESS * capital_cost

            if day_num == 1:
                Deg_cost_by_cycle = (
                    cycles.iloc[0, 0] / total_cycles * EBESS * capital_cost
                )
            else:
                Deg = pd.read_csv(out_dir + "Degradation.csv")
                cycle_of_day = Deg.iloc[-1, 2] - Deg.iloc[-2, 2]
                Deg_cost_by_cycle = cycle_of_day / total_cycles * EBESS * capital_cost

            P_HPP_RT_ts.index = range(T)
            P_HPP_RT_refs.index = range(T)
            P_dis_RT_ts.index = range(T)
            P_cha_RT_ts.index = range(T)

            """
            P_HPP_SM_t_opt, Spot market schedule, power [MW]
            P_dis_SM_t_opt, Battery discharge schedule in spot market, power [MW]
            P_cha_SM_t_opt, 
            P_w_SM_t_opt, 
            P_HPP_RT_ts, Final HPP output, power [MW] 
            P_HPP_RT_refs, 
            P_dis_RT_ts, Final discharge operation of battery, power [MW]
            P_cha_RT_ts: Final charge operation of battery, power [MW]
            """
            output_schedule = pd.concat(
                [
                    P_HPP_SM_t_opt,
                    P_dis_SM_t_opt,
                    P_cha_SM_t_opt,
                    P_w_SM_t_opt,
                    P_HPP_RT_ts,
                    P_HPP_RT_refs,
                    P_dis_RT_ts,
                    P_cha_RT_ts,
                ],
                axis=1,
            )
            output_revenue = pd.DataFrame(
                [
                    SM_revenue,
                    reg_revenue,
                    im_revenue,
                    im_special_revenue_DK1,
                    Deg_cost,
                    Deg_cost_by_cycle,
                ]
            ).T
            output_revenue.columns = [
                "SM_revenue",
                "reg_revenue",
                "im_revenue",
                "im_special_revenue_DK1",
                "Deg_cost",
                "Deg_cost_by_cycle",
            ]
            output_bids = pd.concat([P_HPP_UP_bid_ts, P_HPP_DW_bid_ts], axis=1)
            output_act_signal = pd.concat(
                [
                    pd.DataFrame(s_UP_t, columns=["signal_up"]),
                    pd.DataFrame(s_DW_t, columns=["signal_down"]),
                ],
                axis=1,
            )
            # output_time = pd.concat([pd.DataFrame([run_time], columns=['time-1']), pd.DataFrame([run_time2], columns=['time0'])], axis=1)
            # output_time = pd.concat([pd.DataFrame([run_time], columns=['time-1']), pd.DataFrame([run_time2], columns=['time0'])], axis=1)
            # output_bounds = pd.concat([pd.DataFrame(UBs, columns=['UB']), pd.DataFrame(LBs, columns=['LB'])], axis=1)
            if day_num == 1:
                output_deg = pd.concat(
                    [
                        pd.DataFrame([Ini_nld, nld], columns=["nld"]),
                        pd.DataFrame([0, ld], columns=["ld"]),
                        pd.DataFrame([0, cycles.iloc[0, 0]], columns=["cycles"]),
                    ],
                    axis=1,
                )
            else:
                output_deg = pd.concat(
                    [
                        pd.DataFrame([nld], columns=["nld"]),
                        pd.DataFrame([ld], columns=["ld"]),
                        cycles,
                    ],
                    axis=1,
                )

            output_schedule.to_csv(
                out_dir + "schedule.csv", mode="a", index=False, header=False
            )
            output_bids.to_csv(
                out_dir + "reg_bids.csv", mode="a", index=False, header=False
            )
            output_act_signal.to_csv(
                out_dir + "act_signal.csv", mode="a", index=False, header=False
            )
            output_deg.to_csv(
                out_dir + "Degradation.csv", mode="a", index=False, header=False
            )
            SoC_ts.to_csv(out_dir + "SoC.csv", mode="a", index=False, header=False)
            residual_imbalance.to_csv(
                out_dir + "energy_imbalance.csv", mode="a", index=False, header=False
            )
            RES_RT_cur_ts.to_csv(
                out_dir + "curtailment.csv", mode="a", index=False, header=False
            )
            output_revenue.to_csv(
                out_dir + "revenue.csv", mode="a", index=False, header=False
            )

            Pdis_all = pd.read_csv(out_dir + "schedule.csv", usecols=[3])
            Pcha_all = pd.read_csv(out_dir + "schedule.csv", usecols=[4])
            nld_all = pd.read_csv(out_dir + "Degradation.csv", usecols=[0])
            ad_all = pd.read_csv(out_dir + "slope.csv", usecols=[0])
            ad = DegCal.slope_update(
                Pdis_all, Pcha_all, nld_all, day_num, 7, T, DI, ad_all
            )

            pd.DataFrame([ad], columns=["slope"]).to_csv(
                out_dir + "slope.csv", mode="a", index=False, header=False
            )
            if nld > 0.2:
                break

            pre_nld = nld

            P_HPP_SM_t_opt_list.append(P_HPP_SM_t_opt.values.ravel())
            SM_price_cleared_list.append(SM_price_cleared.values)
            BM_dw_price_cleared_list.append(BM_dw_price_cleared.values)
            BM_up_price_cleared_list.append(BM_up_price_cleared.values)
            P_HPP_RT_ts_list.append(P_HPP_RT_ts.values.ravel())
            P_HPP_RT_refs_list.append(P_HPP_RT_refs.values.ravel())
            P_HPP_UP_bid_ts_list.append(P_HPP_UP_bid_ts.values.ravel())
            P_HPP_DW_bid_ts_list.append(P_HPP_DW_bid_ts.values.ravel())
            s_UP_t_list.append(s_UP_t)
            s_DW_t_list.append(s_DW_t)
            residual_imbalance_list.append(residual_imbalance.values.ravel())
            RES_RT_cur_ts_list.append(RES_RT_cur_ts.values.ravel())
            P_dis_RT_ts_list.append(P_dis_RT_ts.values.ravel())
            P_cha_RT_ts_list.append(P_cha_RT_ts.values.ravel())
            SoC_ts_list.append(pd.DataFrame(SoC_ts).values.ravel())

            day_num = day_num + 1
            if day_num > simulation_dict["number_of_run_day"]:
                print(P_grid_limit)
                break

        pbar.close()
        # return P_HPP_RT_ts, P_HPP_SM_k_opt, P_HPP_RT_refs, P_HPP_UP_bid_ts, P_HPP_DW_bid_ts, RES_RT_cur_ts, P_cha_RT_ts, P_dis_RT_ts, SoC_ts
        # residual_imbalance = pd.DataFrame(residual_imbalance)
        # P_HPP_RT_ts = pd.DataFrame(P_HPP_RT_ts)
        # P_HPP_RT_refs = pd.DataFrame(P_HPP_RT_refs)
        # P_dis_RT_ts = pd.DataFrame(P_dis_RT_ts)
        # P_cha_RT_ts = pd.DataFrame(P_cha_RT_ts)
        # RES_RT_cur_ts = pd.DataFrame(RES_RT_cur_ts)

        return (
            np.ravel(P_HPP_SM_t_opt_list),
            np.ravel(SM_price_cleared_list),
            np.ravel(BM_dw_price_cleared_list),
            np.ravel(BM_up_price_cleared_list),
            np.ravel(P_HPP_RT_ts_list),
            np.ravel(P_HPP_RT_refs_list),
            np.ravel(P_HPP_UP_bid_ts_list),
            np.ravel(P_HPP_DW_bid_ts_list),
            np.ravel(s_UP_t_list),
            np.ravel(s_DW_t_list),
            np.ravel(residual_imbalance_list),
            np.ravel(RES_RT_cur_ts_list),
            np.ravel(P_dis_RT_ts_list),
            np.ravel(P_cha_RT_ts_list),
            np.ravel(SoC_ts_list),
        )
