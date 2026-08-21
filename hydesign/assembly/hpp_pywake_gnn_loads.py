# %%
import os

import numpy as np

# import openmdao.api as om
import pandas as pd
from costmodels.models.variable_opex import lifetime_aware_model
from py_wake.examples.data.dtu10mw_surrogate import DTU10MW_1WT_Surrogate
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.battery_degradation import (
    battery_degradation_comp,
    battery_loss_in_capacity_due_to_temp_comp,
)
from hydesign.costs.costs import (
    battery_cost_comp,
    pvp_cost_comp,
    shared_cost_comp,
    wpp_cost_comp,
)
from hydesign.ems.ems import ems_long_term_operation_comp

# from hydesign.ems.ems import ems_comp, expand_to_lifetime
from hydesign.ems.ems_variable_opex import ems_comp

# from hydesign.ems.ems import expand_to_lifetime
from hydesign.finance.finance import finance_comp
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.pv.pv import pvp_comp, pvp_with_degradation_comp
from hydesign.reliability import (
    battery_with_reliability_comp,
    pvp_with_reliability_comp,
    wpp_with_reliability_comp,
)
from hydesign.weather.weather import ABL_comp
from hydesign.wind.wind import (  # get_pywake_farm_pc,; get_wind_ts_degradation_2d,
    get_rotor_d,
)


class hpp_model(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
        #  _OPEX = 1.42,
        n_wt,
        # farm,
        # farm_load,
        **kwargs,
    ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        MDF_fn : Optional wind Marginal Displacement Factor time-series file
        emissions_value : Optional EMS shadow value [EUR/tCO2e]
        yaw_mode : EMS yaw policy: ``reference``, ``controlled``, or ``smart``
        ems_batch_size : Number of time steps per EMS optimization batch
        """
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        sim_pars = self.sim_pars
        N_time = self.N_time
        wpp_efficiency = self.wpp_efficiency
        life_y = self.life_y
        # wind_deg_yr = self.wind_deg_yr
        # wind_deg = self.wind_deg

        N_time = sim_pars["N_time"]
        wpp_efficiency = sim_pars["wpp_efficiency"]
        life_y = sim_pars["life_y"]
        input_ts_fn = sim_pars["input_ts_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]
        intervals_per_hour = sim_pars["intervals_per_hour"]
        farm = sim_pars["farm"]
        farm_load = sim_pars["farm_load"]
        x = sim_pars["x"]
        y = sim_pars["y"]
        tilt = sim_pars["tilt"]
        time_stamp = sim_pars["time_stamp"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]

        altitude = sim_pars["altitude"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        input_ts_fn = sim_pars["input_ts_fn"]
        weeks_per_season_per_year = sim_pars["weeks_per_season_per_year"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]

        ems_type = sim_pars["ems_type"]
        reliability_ts_battery = sim_pars["reliability_ts_battery"]
        reliability_ts_trans = sim_pars["reliability_ts_trans"]
        reliability_ts_wind = sim_pars["reliability_ts_wind"]
        reliability_ts_pv = sim_pars["reliability_ts_pv"]
        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]
        G_MW = sim_pars["G_MW"]

        life_h = 365 * 24 * life_y
        # life_intervals = life_h * intervals_per_hour
        # pc, pc_ws, pc_wd, pc_yaw, pc_tilt = get_pywake_farm_pc(farm, x, y)
        load_sensors = [
            "Blade_root_edgewise_M_y",
            "Blade_root_flapwise_M_x",
            "Tower_top_tilt_M_x",
            "Tower_top_yaw_M_z",
        ]
        load_sensors_used = [
            "Blade_root_flapwise_M_x",
            "Blade_root_flapwise_M_x",
            "Blade_root_flapwise_M_x",
            "Tower_top_yaw_M_z",
        ]
        n_sensors = len(load_sensors_used)
        if "CAPEX_ref" in sim_pars:
            CAPEX_ref = sim_pars["CAPEX_ref"]
        else:
            CAPEX_ref = 4 * G_MW  # million euro
        _LIFETIME = life_y  # years
        _base_capex_fraction = 0.2
        _base_downtime = 90  # days
        _OPEX = 1.42  # % of CAPEX per year
        components = [
            {
                "name": "Main bearing",
                "beta": 2.5,  # shape parameter
                "C_replace": 0.45,
                "T_replace_days": 90,
                "number": 1,
                "failed_after_design_life": 0.25,
                "cost_fraction_of_capex": 3 * _base_capex_fraction,
                "downtime": 3 * _base_downtime,
            },
            {
                "name": "Blades",
                "beta": 2.5,
                "C_replace": 0.45,
                "T_replace_days": 90,
                "number": 3,
                "failed_after_design_life": 0.1,
                "cost_fraction_of_capex": 1 * _base_capex_fraction,
                "downtime": 1 * _base_downtime,
            },
            {
                "name": "Pitch bearing",
                "beta": 2.5,
                "C_replace": 0.15,
                "T_replace_days": 30,
                "number": 3,
                "failed_after_design_life": 0.1,
                "cost_fraction_of_capex": 1 * _base_capex_fraction,
                "downtime": 1 * _base_downtime,
            },
            {
                "name": "Yaw bearing",
                "beta": 2.5,
                "C_replace": 0.3,
                "T_replace_days": 60,
                "number": 0,
                "failed_after_design_life": 0.1,
                "cost_fraction_of_capex": 3 * _base_capex_fraction,
                "downtime": 3 * _base_downtime,
            },
        ]

        def pywake(wst, wd, yaw, **kwargs):
            if np.allclose(yaw, np.zeros_like(yaw)):
                no_yaw = True
                print("Running PyWake with no yaw control...")
            else:
                no_yaw = False
            ws = wst
            N_ws = len(ws)
            wds = np.arange(0, 360, 2)
            wss = np.arange(3, 25, 1)  # we don't really need all this range

            slice_to_repeat = yaw[:, 0, :][:, np.newaxis, :]
            yaws_extended = np.concatenate([yaw, slice_to_repeat], axis=1)
            wds_extended = np.arange(0, 362, 2)

            yaw_t = np.zeros((n_wt, N_ws))
            if not no_yaw:
                for turbine in range(n_wt):
                    interp_func = RegularGridInterpolator(
                        (wds_extended, wss),
                        yaws_extended[turbine, :, :],
                        bounds_error=False,
                        fill_value=0,
                    )
                    points = np.array([wd, ws]).T  # (x, y) pairs
                    values = interp_func(
                        points,
                    )
                    yaw_t[turbine, :] = values
            sim_res_load_ref = farm_load(
                x=x,
                y=y,
                yaw=np.zeros((n_wt, N_ws)),
                tilt=tilt,
                wd=wd,
                ws=ws,
                time=time_stamp,
                TI=sim_pars["TI"],
            )
            sim_res_load_ref["duration"] = 3600
            if not no_yaw:
                sim_res_load = farm_load(
                    x=x,
                    y=y,
                    yaw=yaw_t,
                    tilt=tilt,
                    wd=wd,
                    ws=ws,
                    time=time_stamp,
                    TI=sim_pars["TI"],
                )
                # sim_res_load['duration'] = 3600 * 24 * 365 * 1
                sim_res_load["duration"] = 3600
            else:
                sim_res_load = sim_res_load_ref
            DEL = sim_res_load.loads(method="OneWT")["DEL"].values
            DEL_ref = sim_res_load_ref.loads(method="OneWT")["DEL"].values
            wholer_exponents = sim_res_load.loads(method="OneWT")["LDEL"].m.values
            LDELs = (
                DEL
                * (3600 * 24 * 365 * 20 / 10**7)
                ** wholer_exponents[:, np.newaxis, np.newaxis]
            )
            LDEL_refs = (
                DEL_ref
                * (3600 * 24 * 365 * 20 / 10**7)
                ** wholer_exponents[:, np.newaxis, np.newaxis]
            )
            # LDEL = sim_res_load.loads(lifetime_years=25, method="OneWT")["LDEL"].values
            # LDEL_ref = sim_res_load_ref.loads(lifetime_years=25, method="OneWT")["LDEL"].values

            M = np.zeros((N_time, n_wt))
            print("Computing M matrix for variable OPEX calculation...")
            gamma = (LDELs / LDEL_refs) ** wholer_exponents[
                :, np.newaxis, np.newaxis
            ]  # (n_sensors, n_wt, N_time)
            # for t in tqdm(range(N_time)):
            # LDEL = LDELs[:,:,t]
            # LDEL_ref = LDEL_refs[:,:,t]
            # print('LDEL:', LDEL)
            # print('LDEL_ref:', LDEL_ref)
            # print('')
            # gamma_ref = np.ones_like(gamma)
            gamma_used = np.asarray(
                [
                    gamma[load_sensors.index(sensor), :, :]
                    for sensor in load_sensors_used
                ]
            )
            gamma_ref_used = np.ones_like(gamma_used)
            for turbine in tqdm(range(n_wt)):
                _gamma_controls = gamma_used[:, turbine, :]  # (n_sensors, N_time)
                _gamma_refs = gamma_ref_used[:, turbine, :]
                _CAPEX = CAPEX_ref / n_wt  # MEUR
                _, _, _, Total_ONM_cost_ref, Total_ONM_cost_control = (
                    lifetime_aware_model(
                        _gamma_refs,
                        _gamma_controls,
                        components,
                        _LIFETIME,
                        _OPEX,
                        _CAPEX,
                    )
                )
                # print('Total_ONM_cost_ref:', Total_ONM_cost_ref)
                M[:, turbine] = (
                    (Total_ONM_cost_control - Total_ONM_cost_ref) * _CAPEX / (365 * 24)
                )
            M = (
                M.sum((1)) * 1e6
            )  # hourly ONM costs for all turbines for control based on ref (positive if control is more expensive) in EURO
            print("Done computing M matrix for variable OPEX calculation...")

            sim_res_ref = farm(
                x=x,
                y=y,
                yaw=np.zeros_like(yaw_t),
                tilt=tilt,
                wd=wd,
                ws=ws,
                time=time_stamp,
                TI=sim_pars["TI"],
            )
            if not no_yaw:
                sim_res = farm(
                    x=x,
                    y=y,
                    yaw=yaw_t,
                    tilt=tilt,
                    wd=wd,
                    ws=ws,
                    time=time_stamp,
                    TI=sim_pars["TI"],
                )
            else:
                sim_res = sim_res_ref
            # https://docs.nrel.gov/docs/fy25osti/91775.pdf CAPEX = 5.411 dollar/MW = 4.65 euro/MW
            # https://guidetoanoffshorewindfarm.com/wind-farm-costs/ CAPEX = 3.47 pound/MW = 3.96 euro/MW

            # sim_res["duration"] = 0

            wind_t = sim_res.Power.sum("wt").values * wpp_efficiency * 1e-6  # W -> MW
            wind_t_ref = (
                sim_res_ref.Power.sum("wt").values * wpp_efficiency * 1e-6
            )  # W -> MW
            # wind_t_ext_deg_raw = expand_to_lifetime(
            #     wpp_efficiency
            #     * get_wind_ts_degradation_2d(
            #         ws=pc_ws,
            #         wd=pc_wd,
            #         pc=pc,
            #         ws_ts=ws,
            #         wd_ts=wd,
            #         yr=wind_deg_yr,
            #         wind_deg=wind_deg,
            #         life=ws.size,
            #     ),
            #     life_y=life_y,
            #     intervals_per_hour=intervals_per_hour,
            # )

            # return [wind_t, wind_t_ext_deg_raw]
            return [wind_t, wind_t_ref, yaw_t, M, wholer_exponents, DEL, DEL_ref]

        def py_wake_actual(
            wind_t,
            wind_t_ref,
            yaw_t,
            lambda_t,
            wholer_exponents,
            DEL,
            DEL_ref,
            **kwargs,
        ):
            yaw_actual = yaw_t * lambda_t[np.newaxis, :]
            wind_t_actual = wind_t * lambda_t + wind_t_ref * (1 - lambda_t)
            DEL_actual = (
                DEL * lambda_t[np.newaxis, np.newaxis, :]
                + DEL_ref * (1 - lambda_t)[np.newaxis, np.newaxis, :]
            )
            LDEL = (
                (
                    (DEL_actual / DEL_actual.mean())
                    ** wholer_exponents[:, np.newaxis, np.newaxis]
                ).sum((2))
                * 3600
                * 24
                * 365
                * 20
                / 10**7
            ) ** (1 / wholer_exponents[:, np.newaxis]) * DEL_actual.mean()
            LDEL_ref = (
                (
                    (DEL_ref / DEL_ref.mean())
                    ** wholer_exponents[:, np.newaxis, np.newaxis]
                ).sum((2))
                * 3600
                * 24
                * 365
                * 20
                / 10**7
            ) ** (1 / wholer_exponents[:, np.newaxis]) * DEL_ref.mean()
            gamma = (LDEL / LDEL_ref) ** wholer_exponents[
                :, np.newaxis
            ]  # (n_sensors, n_wt)
            gamma_used = np.asarray(
                [gamma[load_sensors.index(sensor), :] for sensor in load_sensors_used]
            )
            gamma_ref_used = np.ones_like(gamma_used)
            M = np.zeros((n_wt))
            for turbine in tqdm(range(n_wt)):
                _gamma_controls = gamma_used[:, turbine]  # (n_sensors)
                _gamma_refs = gamma_ref_used[:, turbine]
                _CAPEX = CAPEX_ref / n_wt  # MEUR
                _, Total_ONM_cost_ref, Total_ONM_cost_control, _, _ = (
                    lifetime_aware_model(
                        _gamma_refs,
                        _gamma_controls,
                        components,
                        _LIFETIME,
                        _OPEX,
                        _CAPEX,
                    )
                )
                M[turbine] = (Total_ONM_cost_control - Total_ONM_cost_ref) * _CAPEX
            M_actual = (
                M.sum() * 1e6
            )  # yearly ONM costs for all turbines for control based on ref (positive if control is more expensive) in EURO
            return [wind_t_actual, yaw_actual, DEL_actual, LDEL, LDEL_ref, M_actual]

        comps = [
            (
                "abl",
                ABL_comp(weather_fn=input_ts_fn, N_time=N_time, interpolate_wd=True),
            ),
            (
                "pywake_comp",
                ComponentWrapper(
                    [
                        ("wst", {"shape": [N_time], "units": "m/s"}),
                        ("wd", {"shape": [N_time], "units": "deg"}),
                        ("yaw", {"shape": (n_wt, 180, 22)}),
                    ],
                    [
                        ("wind_t", {"shape": [N_time], "units": "MW"}),
                        ("wind_t_ref", {"shape": [N_time], "units": "MW"}),
                        ("yaw_t", {"shape": (n_wt, N_time)}),
                        ("M", {"shape": [N_time]}),
                        ("wholer_exponents", {"shape": (n_sensors)}),
                        ("DEL", {"shape": (n_sensors, n_wt, N_time)}),
                        ("DEL_ref", {"shape": (n_sensors, n_wt, N_time)}),
                    ],
                    pywake,
                    partial_options=[{"dependent": False, "val": 0}],
                ),
            ),
            (
                "pvp",
                pvp_comp(
                    weather_fn=input_ts_fn,
                    N_time=N_time,
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    tracking=sim_pars["tracking"],
                ),
            ),
            (
                "ems",
                ems_comp(
                    N_time=N_time,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    life_y=life_y,
                    ems_type=ems_type,
                    yaw_mode=sim_pars.get("yaw_mode", "smart"),
                    batch_size=sim_pars.get("ems_batch_size", 110),
                ),
            ),
            (
                "pywake_actual_comp",
                ComponentWrapper(
                    [
                        ("wind_t", {"shape": [N_time], "units": "MW"}),
                        ("wind_t_ref", {"shape": [N_time], "units": "MW"}),
                        ("yaw_t", {"shape": (n_wt, N_time)}),
                        ("lambda_t", {"shape": [N_time]}),
                        ("M", {"shape": [N_time]}),
                        ("wholer_exponents", {"shape": (n_sensors)}),
                        ("DEL", {"shape": (n_sensors, n_wt, N_time)}),
                        ("DEL_ref", {"shape": (n_sensors, n_wt, N_time)}),
                    ],
                    [
                        ("wind_t_actual", {"shape": [N_time], "units": "MW"}),
                        ("yaw_actual", {"shape": (n_wt, N_time)}),
                        ("DEL_actual", {"shape": (n_sensors, n_wt, N_time)}),
                        ("LDEL", {"shape": (n_sensors, n_wt)}),
                        ("LDEL_ref", {"shape": (n_sensors, n_wt)}),
                        ("M_actual",),
                    ],
                    py_wake_actual,
                    partial_options=[{"dependent": False, "val": 0}],
                ),
            ),
            (
                "battery_degradation",
                battery_degradation_comp(
                    weather_fn=input_ts_fn,  # for extracting temperature
                    num_batteries=max_num_batteries_allowed,
                    life_y=life_y,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "battery_loss_in_capacity_due_to_temp",
                battery_loss_in_capacity_due_to_temp_comp(
                    weather_fn=input_ts_fn,  # for extracting temperature
                    life_y=life_y,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "pvp_with_degradation",
                pvp_with_degradation_comp(
                    life_y=life_y,
                    pv_deg_yr=sim_pars["pv_deg_yr"],
                    pv_deg=sim_pars["pv_deg"],
                ),
            ),
            (
                "battery_with_reliability",
                battery_with_reliability_comp(
                    life_y=life_y,
                    reliability_ts_battery=reliability_ts_battery,
                    reliability_ts_trans=reliability_ts_trans,
                ),
            ),
            (
                "wpp_with_reliability",
                wpp_with_reliability_comp(
                    life_y=life_y,
                    reliability_ts_wind=reliability_ts_wind,
                    reliability_ts_trans=reliability_ts_trans,
                ),
                {"wind_t": "wind_t_ext"},
            ),
            (
                "pvp_with_reliability",
                pvp_with_reliability_comp(
                    life_y=life_y,
                    reliability_ts_pv=reliability_ts_pv,
                    reliability_ts_trans=reliability_ts_trans,
                ),
                {"solar_t": "solar_t_ext_deg"},
            ),
            (
                "ems_long_term_operation",
                ems_long_term_operation_comp(N_time=N_time, life_y=life_y),
                {
                    "SoH": "SoH_all",
                    "wind_t_ext_deg": "wind_t_rel",
                    "solar_t_ext_deg": "solar_t_rel",
                    "b_t": "b_t_rel",
                },  # (<parent key>, <child key>)
            ),
            (
                "wpp_cost",
                wpp_cost_comp(
                    wind_turbine_cost=sim_pars["wind_turbine_cost"],
                    wind_civil_works_cost=sim_pars["wind_civil_works_cost"],
                    wind_fixed_onm_cost=sim_pars["wind_fixed_onm_cost"],
                    wind_variable_onm_cost=sim_pars["wind_variable_onm_cost"],
                    d_ref=sim_pars["d_ref"],
                    hh_ref=sim_pars["hh_ref"],
                    p_rated_ref=sim_pars["p_rated_ref"],
                    N_time=N_time,
                ),
            ),
            (
                "pvp_cost",
                pvp_cost_comp(
                    solar_PV_cost=sim_pars["solar_PV_cost"],
                    solar_hardware_installation_cost=sim_pars[
                        "solar_hardware_installation_cost"
                    ],
                    solar_inverter_cost=sim_pars["solar_inverter_cost"],
                    solar_fixed_onm_cost=sim_pars["solar_fixed_onm_cost"],
                ),
            ),
            (
                "battery_cost",
                battery_cost_comp(
                    battery_energy_cost=sim_pars["battery_energy_cost"],
                    battery_power_cost=sim_pars["battery_power_cost"],
                    battery_BOP_installation_commissioning_cost=sim_pars[
                        "battery_BOP_installation_commissioning_cost"
                    ],
                    battery_control_system_cost=sim_pars["battery_control_system_cost"],
                    battery_energy_onm_cost=sim_pars["battery_energy_onm_cost"],
                    # N_life = N_life,
                    life_y=life_y,
                    battery_price_reduction_per_year=battery_price_reduction_per_year,
                ),
            ),
            (
                "shared_cost",
                shared_cost_comp(
                    hpp_BOS_soft_cost=sim_pars["hpp_BOS_soft_cost"],
                    hpp_grid_connection_cost=sim_pars["hpp_grid_connection_cost"],
                    land_cost=sim_pars["land_cost"],
                ),
            ),
            (
                "finance",
                finance_comp(
                    N_time=N_time,
                    # Depreciation curve
                    depreciation_yr=sim_pars["depreciation_yr"],
                    depreciation=sim_pars["depreciation"],
                    # Inflation curve
                    inflation_yr=sim_pars["inflation_yr"],
                    inflation=sim_pars["inflation"],
                    ref_yr_inflation=sim_pars["ref_yr_inflation"],
                    # Early paying or CAPEX Phasing
                    phasing_yr=sim_pars["phasing_yr"],
                    phasing_CAPEX=sim_pars["phasing_CAPEX"],
                    life_y=life_y,
                    # save_finance_ts = sim_pars['save_finance_ts'],
                    # work_dir = work_dir,
                    # time_str = sim_pars['time_str']
                ),
                {
                    "CAPEX_el": "CAPEX_sh",
                    "OPEX_el": "OPEX_sh",
                    "penalty_t": "penalty_t_with_deg",
                },  # {<input-key-name in model> (that corresponds to): <output-key-name from prior component>}
            ),
        ]

        prob = self.get_prob(comps)
        prob.model.options["auto_order"] = True
        prob.setup()

        # Additional parameters
        prob.set_val("price_t", self.price)
        MDF_fn = sim_pars.get("MDF_fn")
        if MDF_fn is not None:
            mdf = pd.read_csv(MDF_fn, index_col=0)
            mdf.index = pd.to_datetime(mdf.index, dayfirst=True)
            weather_index = pd.read_csv(
                input_ts_fn, index_col=0, parse_dates=True, usecols=[0]
            ).index
            mdf_t = mdf.reindex(weather_index).iloc[:, 0].to_numpy()
            if len(mdf_t) != N_time or np.isnan(mdf_t).any():
                raise ValueError(
                    "MDF time series must cover every HPP weather time step"
                )
            prob.set_val("marginal_displacement_factor_t", mdf_t)
        prob.set_val("emissions_value", sim_pars.get("emissions_value", 0.0))
        prob.set_val("G_MW", G_MW)
        prob.set_val(
            "battery_depth_of_discharge", sim_pars["battery_depth_of_discharge"]
        )
        prob.set_val("battery_charge_efficiency", sim_pars["battery_charge_efficiency"])
        prob.set_val("peak_hr_quantile", sim_pars["peak_hr_quantile"])
        prob.set_val(
            "n_full_power_hours_expected_per_day_at_peak_price",
            sim_pars["n_full_power_hours_expected_per_day_at_peak_price"],
        )
        prob.set_val("min_LoH", sim_pars["min_LoH"])
        prob.set_val("wind_WACC", sim_pars["wind_WACC"])
        prob.set_val("solar_WACC", sim_pars["solar_WACC"])
        prob.set_val("battery_WACC", sim_pars["battery_WACC"])
        prob.set_val("tax_rate", sim_pars["tax_rate"])
        prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])

        self.prob = prob

        self.list_out_vars = [
            "NPV_over_CAPEX",
            "NPV [MEuro]",
            "IRR",
            "LCOE [Euro/MWh]",
            "Revenues [MEuro]",
            "CAPEX [MEuro]",
            "OPEX [MEuro]",
            "Wind CAPEX [MEuro]",
            "Wind OPEX [MEuro]",
            "PV CAPEX [MEuro]",
            "PV OPEX [MEuro]",
            "Batt CAPEX [MEuro]",
            "Batt OPEX [MEuro]",
            "Shared CAPEX [MEuro]",
            "Shared OPEX [MEuro]",
            "penalty lifetime [MEuro]",
            "AEP [GWh]",
            "GUF",
            "grid [MW]",
            "wind [MW]",
            "solar [MW]",
            "Battery Energy [MWh]",
            "Battery Power [MW]",
            "Total curtailment [GWh]",
            "Total curtailment with deg [GWh]",
            "Awpp [km2]",
            "Apvp [km2]",
            "Plant area [km2]",
            "Rotor diam [m]",
            "Hub height [m]",
            "Number of batteries used in lifetime",
            "Break-even PPA price [Euro/MWh]",
            "Capacity factor wind [-]",
        ]

        self.list_vars = [
            "clearance [m]",
            "sp [W/m2]",
            "p_rated [MW]",
            "Nwt",
            "wind_MW_per_km2 [MW/km2]",
            "solar_MW [MW]",
            "surface_tilt [deg]",
            "surface_azimuth [deg]",
            "DC_AC_ratio",
            "b_P [MW]",
            "b_E_h [h]",
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            # 'yaw [deg]',
        ]

    def evaluate(
        self,
        # Wind plant design
        clearance,
        sp,
        p_rated,
        Nwt,
        wind_MW_per_km2,
        # PV plant design
        solar_MW,
        surface_tilt,
        surface_azimuth,
        DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P,
        b_E_h,
        cost_of_battery_P_fluct_in_peak_price_ratio,
        # Wind turbine control
        yaw,
    ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        clearance : Distance from the ground to the tip of the blade [m]
        sp : Specific power of the turbine [W/m2]
        p_rated : Rated powe of the turbine [MW]
        Nwt : Number of wind turbines
        wind_MW_per_km2 : Wind power installation density [MW/km2]
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        self.num_batteries : Number of allowed replacements of the battery
        """
        self.inputs = [
            clearance,
            sp,
            p_rated,
            Nwt,
            wind_MW_per_km2,
            # PV plant design
            solar_MW,
            surface_tilt,
            surface_azimuth,
            DC_AC_ratio,
            # Energy storage & EMS price constrains
            b_P,
            b_E_h,
            cost_of_battery_P_fluct_in_peak_price_ratio,
            # Wind turbine control
            # yaw,
        ]
        prob = self.prob

        d = get_rotor_d(p_rated * 1e6 / sp)
        hh = (d / 2) + clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2
        # Awpp = Awpp + 1e-10*(Awpp==0)
        b_E = b_E_h * b_P

        # pass design variables
        prob.set_val("hh", hh)
        prob.set_val("d", d)
        prob.set_val("p_rated", p_rated)
        prob.set_val("Nwt", Nwt)
        prob.set_val("Awpp", Awpp)

        prob.set_val("surface_tilt", surface_tilt)
        prob.set_val("surface_azimuth", surface_azimuth)
        prob.set_val("DC_AC_ratio", DC_AC_ratio)
        prob.set_val("solar_MW", solar_MW)

        prob.set_val("b_P", b_P)
        prob.set_val("b_E", b_E)
        prob.set_val(
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            cost_of_battery_P_fluct_in_peak_price_ratio,
        )

        prob.set_val("yaw", yaw)

        prob.run_model()

        self.prob = prob

        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = (
                prob["wind_t_ext"].mean() / p_rated / Nwt
            )  # Capacity factor of wind only

        outputs = np.hstack(
            [
                prob["NPV_over_CAPEX"],
                prob["NPV"] / 1e6,
                prob["IRR"],
                prob["LCOE"],
                prob["revenues"] / 1e6,
                prob["CAPEX"] / 1e6,
                prob["OPEX"] / 1e6,
                prob["CAPEX_w"] / 1e6,
                prob["OPEX_w"] / 1e6,
                prob["CAPEX_s"] / 1e6,
                prob["OPEX_s"] / 1e6,
                prob["CAPEX_b"] / 1e6,
                prob["OPEX_b"] / 1e6,
                prob["CAPEX_sh"] / 1e6,
                prob["OPEX_sh"] / 1e6,
                prob["penalty_lifetime"] / 1e6,
                prob["mean_AEP"] / 1e3,  # [GWh]
                # Grid Utilization factor
                prob["mean_AEP"] / (self.sim_pars["G_MW"] * 365 * 24),
                self.sim_pars["G_MW"],
                wind_MW,
                solar_MW,
                b_E,
                b_P,
                prob["total_curtailment"] / 1e3,  # [GWh]
                prob["total_curtailment_with_deg"] / 1e3,  # [GWh]
                Awpp,
                prob["Apvp"],
                max(Awpp, prob["Apvp"]),
                d,
                hh,
                prob["n_batteries"] * (b_P > 0),
                prob["break_even_PPA_price"],
                cf_wind,
            ]
        )
        self.outputs = outputs
        return outputs


if __name__ == "__main__":
    import sys
    import time

    import numpy as np
    import openmdao.api as om
    import xarray as xr
    from design_friendly.utils.easy import easy_yaw_gnn
    from design_friendly.utils.get_flowmodel import get_flowmodel
    from design_friendly.utils.iea22s import IEA22s
    from design_friendly.utils.sites import Hornsrev1Site
    from py_wake import NOJ
    from py_wake.deficit_models.gaussian import ZongGaussianDeficit
    from py_wake.deficit_models.utils import ct2a_mom1d
    from py_wake.deflection_models import JimenezWakeDeflection
    from py_wake.deflection_models.jimenez import JimenezWakeDeflection
    from py_wake.rotor_avg_models import CGIRotorAvg, GaussianOverlapAvgModel
    from py_wake.site._site import UniformSite
    from py_wake.superposition_models import LinearSum, SqrMaxSum, WeightedSum
    from py_wake.turbulence_models import CrespoHernandez
    from py_wake.turbulence_models.stf import STF2017TurbulenceModel
    from py_wake.wind_farm_models import PropagateDownwind
    from scipy.spatial import ConvexHull
    from topfarm.utils import regular_generic_layout

    from hydesign.examples import examples_filepath

    # arg_var = int(sys.argv[1]) - 1 # job-array starts from 1, python from 0
    # sample, loc = np.unravel_index(arg_var, (1000, 3))
    # print('')
    # print('sample, loc:', sample, loc)
    # print('')
    # output_file_name = os.path.join('data', f'results_master_sample_{arg_var:04d}.csv')
    # ds = xr.open_dataset("samples.nc")
    """
<xarray.Dataset> Size: 212kB
Dimensions:         (sample: 1000, loc: 3)
Coordinates:
  * sample          (sample) int64 8kB 0 1 2 3 4 5 6 ... 994 995 996 997 998 999
  * loc             (loc) int64 24B 0 1 2
Data variables:
    TI              (sample, loc) float64 24kB 0.0834 0.1441 ... 0.0337 0.1058
    n_wt            (sample, loc) int32 12kB 41 144 76 141 137 ... 146 85 193 44
    spacing         (sample, loc) float64 24kB 4.502 6.322 2.001 ... 3.011 5.175
    master_samples  (sample, loc) int64 24kB 0 1000 2000 1 ... 999 1999 2999
    G_MW            (sample, loc) int64 24kB 902 3168 1672 ... 1870 4246 968
    sx              (sample, loc) float64 24kB 1.279e+03 1.795e+03 ... 1.47e+03
    sy              (sample, loc) float64 24kB 1.279e+03 1.795e+03 ... 1.47e+03
    area            (sample, loc) float64 24kB 4.823e+07 3.901e+08 ... 7.235e+07
    b_P             (sample, loc) float64 24kB 188.1 1.141e+03 ... 357.7 256.1
"""

    start = time.time()
    loc = 0  # 0: Denmark, 1: France, 2: Germany
    sample = 0  # sample index
    names = ["Denmark_good_wind", "France_good_wind", "Germany_good_wind"]
    name = names[loc]
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site["longitude"].values[0]
    latitude = ex_site["latitude"].values[0]
    altitude = ex_site["altitude"].values[0]

    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]

    # this is a smoother version of PyWake IEA22 that works better with wake steering optimization
    wt = IEA22s()
    wds = np.arange(0, 360, 2)
    wss = np.arange(3, 25, 1)  # we don't really need all this range
    # (x, y), site = Hornsrev1Site(
    #     scale_D=wt.diameter()  # scale up the layout based on turbine diameter ratio
    # )
    TI = 0.06  # site.local_wind().TI_ilk.ravel()
    # TI = ds.TI.values[sample, loc]  # site.local_wind().TI_ilk.ravel()

    life_y = 25
    intervals_per_hour = 1
    n_wt = 5
    # n_wt = ds.n_wt.values[sample, loc]
    print("n_wt:", n_wt)
    print("")
    d = 284
    hh = 170
    RP = 22.0
    G_MW = int(n_wt * RP * 1.0)  # grid connection size
    # G_MW = ds.G_MW.values[sample, loc]
    sx = 4 * d
    sy = 5 * d
    # sx = ds.sx.values[sample, loc]
    # sy = ds.sy.values[sample, loc]
    x, y = regular_generic_layout(n_wt, sx, sy, stagger=0, rotation=0)

    hull = ConvexHull(list(np.asarray([x, y]).T))
    area = hull.volume
    # area = ds.area.values[sample, loc]

    N_ws = 365 * 24 * intervals_per_hour
    time_stamp = np.arange(N_ws) / 6 / 24
    farm = get_flowmodel(wt=wt)

    yaws = easy_yaw_gnn(x, y, wd=wds, ws=wss, TI=TI)
    # yaws = np.zeros((n_wt, len(wss), len(wds)))
    yaw = yaws
    tilt = np.zeros(N_ws)

    # b_P = ds.b_P.values[sample, loc]
    # wind_MW_per_km2 = ds.wind_MW_per_km2.values[sample, loc]
    wind_MW_per_km2 = n_wt * RP / (area * 10**-6)
    b_P = 27
    # b_P = ds.b_P.values[sample, loc]
    # wind_MW_per_km2 = ds.wind_MW_per_km2.values[sample, loc]
    wt_load = DTU10MW_1WT_Surrogate()
    farm_load = get_flowmodel(wt=wt_load)
    hpp = hpp_model(
        n_wt=n_wt,
        farm=farm,
        farm_load=farm_load,
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        sim_pars_fn=sim_pars_fn,
        input_ts_fn=input_ts_fn,
        intervals_per_hour=intervals_per_hour,
        # farm=farm,
        x=x,
        y=y,
        tilt=tilt,
        time_stamp=time_stamp,
        G_MW=G_MW,
        TI=TI,
    )

    # Wind plant design
    x = dict(
        clearance=hh - d / 2,
        sp=RP * 10**6 / (np.pi * d**2 / 4),
        p_rated=RP,
        Nwt=n_wt,
        wind_MW_per_km2=wind_MW_per_km2,
        # PV plant design
        solar_MW=0,
        surface_tilt=28.125,
        surface_azimuth=191.250,
        DC_AC_ratio=1.479,
        # Energy storage & EMS price constrains
        b_P=b_P,
        b_E_h=4,
        cost_of_battery_P_fluct_in_peak_price_ratio=8.750,
        # Wind turbine control
        yaw=yaw,
    )

    outs = hpp.evaluate(**x)

    hpp.print_design(list(x.values()), outs)
    # hpp.evaluation_in_csv(output_file_name)

    end = time.time()
    print("exec. time [min]:", (end - start) / 60)

    # om.n2(hpp.prob)

    # print(hpp.prob.model.list_outputs())
    # print(hpp.prob.model.list_inputs())
