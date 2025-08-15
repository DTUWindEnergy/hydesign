# %%
import numpy as np
import openmdao.api as om
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.battery_degradation import battery_degradation as battery_degradation_pp
from hydesign.battery_degradation import (
    battery_loss_in_capacity_due_to_temp as battery_loss_in_capacity_due_to_temp_pp,
)
from hydesign.costs.costs import battery_cost as battery_cost_pp
from hydesign.costs.costs import pvp_cost as pvp_cost_pp
from hydesign.costs.costs import shared_cost as shared_cost_pp
from hydesign.costs.costs import wpp_cost as wpp_cost_pp
from hydesign.ems.ems import ems as ems_pp
from hydesign.ems.ems import ems_long_term_operation as ems_long_term_operation_pp
from hydesign.ems.ems import (
    expand_to_lifetime,
)
from hydesign.finance.finance import finance as finance_pp
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.pv.pv import pvp as pvp_pp
from hydesign.pv.pv import pvp_with_degradation as pvp_with_degradation_pp
from hydesign.weather.weather import ABL as ABL_pp
from hydesign.wind.wind import get_rotor_d


class hpp_model(hpp_base):
    """HPP design evaluator"""

    def __init__(self, sim_pars_fn, **kwargs):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        """
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        sim_pars = self.sim_pars

        N_time = sim_pars["N_time"]
        wpp_efficiency = sim_pars["wpp_efficiency"]
        life_y = sim_pars["life_y"]
        price_t = sim_pars["price"]
        input_ts_fn = sim_pars["input_ts_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]
        G_MW = sim_pars["G_MW"]
        battery_depth_of_discharge = sim_pars["battery_depth_of_discharge"]
        battery_charge_efficiency = sim_pars["battery_charge_efficiency"]
        peak_hr_quantile = sim_pars["peak_hr_quantile"]
        n_full_power_hours_expected_per_day_at_peak_price = sim_pars[
            "n_full_power_hours_expected_per_day_at_peak_price"
        ]
        min_LoH = sim_pars["min_LoH"]
        wind_WACC = sim_pars["wind_WACC"]
        solar_WACC = sim_pars["solar_WACC"]
        battery_WACC = sim_pars["battery_WACC"]
        tax_rate = sim_pars["tax_rate"]
        land_use_per_solar_MW = sim_pars["land_use_per_solar_MW"]
        intervals_per_hour = sim_pars["intervals_per_hour"]
        farm = sim_pars["farm"]
        x = sim_pars["x"]
        y = sim_pars["y"]
        tilt = sim_pars["tilt"]
        time_stamp = sim_pars["time_stamp"]
        num_batteries = sim_pars["max_num_batteries_allowed"]
        n_loads = sim_pars["n_loads"]

        life_h = 365 * 24 * life_y
        life_intervals = life_h * intervals_per_hour

        ABL = ABL_pp(weather_fn=input_ts_fn, N_time=N_time, interpolate_wd=True)

        PVP = pvp_pp(
            weather_fn=input_ts_fn,
            N_time=N_time,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            tracking=sim_pars["tracking"],
        )
        PVP_deg = pvp_with_degradation_pp(
            life_y=life_y,
            intervals_per_hour=intervals_per_hour,
            pv_deg_yr=sim_pars["pv_deg_yr"],
            pv_deg=sim_pars["pv_deg"],
        )
        EMS = ems_pp(
            N_time=N_time, life_y=life_y, intervals_per_hour=intervals_per_hour
        )

        Batt_deg = battery_degradation_pp(
            weather_fn=input_ts_fn,
            num_batteries=num_batteries,
            life_y=life_y,
            intervals_per_hour=intervals_per_hour,
        )

        Batt_loss = battery_loss_in_capacity_due_to_temp_pp(
            weather_fn=input_ts_fn,
            num_batteries=num_batteries,
            life_y=life_y,
            intervals_per_hour=intervals_per_hour,
        )
        EMS_long_term = ems_long_term_operation_pp(
            N_time=N_time,
            num_batteries=num_batteries,
            life_y=life_y,
            intervals_per_hour=intervals_per_hour,
        )

        Wind_cost = wpp_cost_pp(
            wind_turbine_cost=sim_pars["wind_turbine_cost"],
            wind_civil_works_cost=sim_pars["wind_civil_works_cost"],
            wind_fixed_onm_cost=sim_pars["wind_fixed_onm_cost"],
            wind_variable_onm_cost=sim_pars["wind_variable_onm_cost"],
            d_ref=sim_pars["d_ref"],
            hh_ref=sim_pars["hh_ref"],
            p_rated_ref=sim_pars["p_rated_ref"],
            N_time=N_time,
        )

        Solar_cost = pvp_cost_pp(
            solar_PV_cost=sim_pars["solar_PV_cost"],
            solar_hardware_installation_cost=sim_pars[
                "solar_hardware_installation_cost"
            ],
            solar_inverter_cost=sim_pars["solar_inverter_cost"],
            solar_fixed_onm_cost=sim_pars["solar_fixed_onm_cost"],
        )

        Batt_cost = battery_cost_pp(
            battery_energy_cost=sim_pars["battery_energy_cost"],
            battery_power_cost=sim_pars["battery_power_cost"],
            battery_BOP_installation_commissioning_cost=sim_pars[
                "battery_BOP_installation_commissioning_cost"
            ],
            battery_control_system_cost=sim_pars["battery_control_system_cost"],
            battery_energy_onm_cost=sim_pars["battery_energy_onm_cost"],
            life_y=life_y,
            battery_price_reduction_per_year=battery_price_reduction_per_year,
        )

        Shared_cost = shared_cost_pp(
            hpp_BOS_soft_cost=sim_pars["hpp_BOS_soft_cost"],
            hpp_grid_connection_cost=sim_pars["hpp_grid_connection_cost"],
            land_cost=sim_pars["land_cost"],
        )

        Finance = finance_pp(
            N_time=N_time,
            depreciation_yr=sim_pars["depreciation_yr"],
            depreciation=sim_pars["depreciation"],
            inflation_yr=sim_pars["inflation_yr"],
            inflation=sim_pars["inflation"],
            ref_yr_inflation=sim_pars["ref_yr_inflation"],
            phasing_yr=sim_pars["phasing_yr"],
            phasing_CAPEX=sim_pars["phasing_CAPEX"],
            life_y=life_y,
        )

        def abl(hh, **kwargs):
            ws, wd = ABL.compute(hh)
            return [ws, wd]

        def pywake_ref(ws, wd, **kwargs):
            sim_res = farm(
                x=x,
                y=y,
                yaw=np.zeros_like(wd),
                tilt=tilt,
                wd=wd,
                ws=ws,
                time=time_stamp,
                Alpha=0.12,
            )
            sim_res["duration"] = 0

            loads_ref_ext = expand_to_lifetime(
                sim_res.loads(normalize_probabilities=False, method="OneWT")[
                    "DEL"
                ].values,
                life_y=life_y,
                intervals_per_hour=intervals_per_hour,
                axis=2,
            )
            return loads_ref_ext

        def pywake(ws, wd, yaw, loads_ref_ext, **kwargs):
            sim_res = farm(
                x=x, y=y, yaw=yaw, tilt=tilt, wd=wd, ws=ws, time=time_stamp, Alpha=0.12
            )
            sim_res["duration"] = 0

            loads = sim_res.loads(normalize_probabilities=False, method="OneWT")[
                "DEL"
            ].values
            wind_t = sim_res.Power.sum("wt").values * wpp_efficiency * 1e-6  # W -> MW
            wind_t_ext = expand_to_lifetime(
                wind_t, life_y=life_y, intervals_per_hour=intervals_per_hour
            )
            loads_rel_ext = (
                expand_to_lifetime(
                    loads, life_y=life_y, intervals_per_hour=intervals_per_hour, axis=2
                )
                / loads_ref_ext
            )
            return [wind_t, loads_rel_ext, wind_t_ext]

        def pv(surface_tilt, surface_azimuth, solar_MW, DC_AC_ratio, **kwargs):
            solar_t, Apvp = PVP.compute(
                surface_tilt,
                surface_azimuth,
                solar_MW,
                land_use_per_solar_MW,
                DC_AC_ratio,
            )
            solar_t_ext = expand_to_lifetime(
                solar_t, life_y=life_y, intervals_per_hour=intervals_per_hour
            )
            solar_t_ext_deg = PVP_deg.compute(solar_t_ext)
            return [solar_t, solar_t_ext, solar_t_ext_deg, Apvp]

        def ems(
            wind_t,
            solar_t,
            b_P,
            b_E,
            cost_of_battery_P_fluct_in_peak_price_ratio,
            **kwargs,
        ):
            (
                price_t_ext,  # wind_t_ext, solar_t_ext,
                hpp_t,
                hpp_curt_t,
                b_t,
                b_E_SOC_t,
                penalty_t,
            ) = EMS.compute(
                wind_t,
                solar_t,
                price_t,
                b_P,
                b_E,
                G_MW,
                battery_depth_of_discharge,
                battery_charge_efficiency,
                peak_hr_quantile,
                cost_of_battery_P_fluct_in_peak_price_ratio,
                n_full_power_hours_expected_per_day_at_peak_price,
            )
            return [
                price_t_ext,
                hpp_t,
                hpp_curt_t,
                b_t,
                b_E_SOC_t,
                penalty_t,
            ]  # wind_t_ext, solar_t_ext,

        def battery_degradation(b_E_SOC_t, **kwargs):
            SoH, n_batteries = Batt_deg.compute(b_E_SOC_t, min_LoH)
            return [SoH, n_batteries]

        def battery_loss(SoH, **kwargs):
            SoH_all = Batt_loss.compute(SoH)
            return SoH_all

        def ems_long_term(
            SoH_all,
            solar_t_ext_deg,
            wind_t_ext,
            solar_t_ext,
            price_t_ext,
            b_E,
            hpp_curt_t,
            b_t,
            b_E_SOC_t,
            **kwargs,
        ):
            wind_t_ext_deg = wind_t_ext.copy()
            (
                hpp_t_with_deg,
                hpp_curt_t_with_deg,
                b_t_with_deg,
                b_E_SOC_t_with_deg,
                penalty_t_with_deg,
                total_curtailment_with_deg,
                total_curtailment,
            ) = EMS_long_term.compute(
                SoH_all,
                wind_t_ext_deg,
                solar_t_ext_deg,
                wind_t_ext,
                solar_t_ext,
                price_t_ext,
                b_E,
                G_MW,
                battery_depth_of_discharge,
                battery_charge_efficiency,
                hpp_curt_t,
                b_t,
                b_E_SOC_t,
                peak_hr_quantile,
                n_full_power_hours_expected_per_day_at_peak_price,
            )
            return [
                hpp_t_with_deg,
                hpp_curt_t_with_deg,
                b_t_with_deg,
                b_E_SOC_t_with_deg,
                penalty_t_with_deg,
                total_curtailment_with_deg,
                total_curtailment,
            ]

        def wind_cost(Nwt, hh, d, p_rated, wind_t, **kwargs):
            CAPEX_w, OPEX_w = Wind_cost.compute(Nwt, hh, d, p_rated, wind_t)
            return [CAPEX_w, OPEX_w]

        def solar_cost(solar_MW, DC_AC_ratio, **kwargs):
            CAPEX_s, OPEX_s = Solar_cost.compute(solar_MW, DC_AC_ratio)
            return [CAPEX_s, OPEX_s]

        def batt_cost(b_E, b_P, SoH, **kwargs):
            CAPEX_b, OPEX_b = Batt_cost.compute(b_E, b_P, SoH)
            return [CAPEX_b, OPEX_b]

        def shared_cost(Awpp, Apvp, **kwargs):
            CAPEX_sh, OPEX_sh = Shared_cost.compute(G_MW, Awpp, Apvp)
            return [CAPEX_sh, OPEX_sh]

        def finance(
            hpp_t_with_deg,
            penalty_t_with_deg,
            price_t_ext,
            CAPEX_w,
            CAPEX_s,
            CAPEX_b,
            CAPEX_sh,
            OPEX_w,
            OPEX_s,
            OPEX_b,
            OPEX_sh,
            **kwargs,
        ):
            (
                CAPEX,
                OPEX,
                revenues_mean,
                NPV,
                IRR,
                NPV_over_CAPEX,
                LCOE,
                mean_AEP,
                penalty_lifetime,
                break_even_PPA_price,
            ) = Finance.compute(
                hpp_t_with_deg,
                penalty_t_with_deg,
                price_t_ext,
                CAPEX_w,
                CAPEX_s,
                CAPEX_b,
                CAPEX_sh,
                OPEX_w,
                OPEX_s,
                OPEX_b,
                OPEX_sh,
                wind_WACC,
                solar_WACC,
                battery_WACC,
                tax_rate,
            )
            return [
                CAPEX,
                OPEX,
                revenues_mean,
                NPV,
                IRR,
                NPV_over_CAPEX,
                LCOE,
                mean_AEP,
                penalty_lifetime,
                break_even_PPA_price,
            ]

        comps = dict(
            abl_comp=ComponentWrapper(
                [
                    ("hh",),
                ],
                [
                    ("ws", {"shape": [N_time]}),
                    ("wd", {"shape": [N_time]}),
                ],
                abl,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            pywake_ref_comp=ComponentWrapper(
                [
                    ("ws", {"shape": [N_time]}),
                    ("wd", {"shape": [N_time]}),
                ],
                [
                    ("loads_ref_ext", {"shape": [n_loads, n_wt, life_intervals]}),
                ],
                pywake_ref,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            pywake_comp=ComponentWrapper(
                [
                    ("ws", {"shape": [N_time]}),
                    ("wd", {"shape": [N_time]}),
                    ("yaw", {"shape": [N_time]}),
                    ("loads_ref_ext", {"shape": [n_loads, n_wt, life_intervals]}),
                ],
                [
                    ("wind_t", {"shape": [N_time]}),
                    ("loads_rel_ext", {"shape": [n_loads, n_wt, life_intervals]}),
                    ("wind_t_ext", {"shape": [life_intervals]}),
                ],
                pywake,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            pv_comp=ComponentWrapper(
                [
                    ("surface_tilt",),
                    ("surface_azimuth",),
                    ("solar_MW",),
                    ("DC_AC_ratio",),
                ],
                [
                    ("solar_t", {"shape": [N_time]}),
                    ("solar_t_ext", {"shape": [life_intervals]}),
                    ("solar_t_ext_deg", {"shape": [life_intervals]}),
                    ("Apvp",),
                ],
                pv,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            ems_comp=ComponentWrapper(
                [
                    ("wind_t", {"shape": [N_time]}),
                    ("solar_t", {"shape": [N_time]}),
                    ("b_P",),
                    ("b_E",),
                    ("cost_of_battery_P_fluct_in_peak_price_ratio",),
                ],
                [
                    ("price_t_ext", {"shape": [life_intervals]}),
                    ("hpp_t", {"shape": [life_intervals]}),
                    ("hpp_curt_t", {"shape": [life_intervals]}),
                    ("b_t", {"shape": [life_intervals]}),
                    ("b_E_SOC_t", {"shape": [life_intervals + 1]}),
                    ("penalty_t", {"shape": [life_intervals]}),
                ],
                ems,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            battery_degradation_comp=ComponentWrapper(
                [
                    ("b_E_SOC_t", {"shape": [life_intervals + 1]}),
                ],
                [
                    ("SoH", {"shape": [life_intervals]}),
                    ("n_batteries",),
                ],
                battery_degradation,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            battery_loss_comp=ComponentWrapper(
                [("SoH", {"shape": [life_intervals]})],
                [
                    ("SoH_all", {"shape": [life_intervals]}),
                ],
                battery_loss,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            ems_long_term_comp=ComponentWrapper(
                [
                    ("SoH_all", {"shape": [life_intervals]}),
                    # ('wind_t_ext_deg', {'shape': [life_intervals]}),
                    ("solar_t_ext_deg", {"shape": [life_intervals]}),
                    ("wind_t_ext", {"shape": [life_intervals]}),
                    ("solar_t_ext", {"shape": [life_intervals]}),
                    ("price_t_ext", {"shape": [life_intervals]}),
                    ("b_E",),
                    ("hpp_curt_t", {"shape": [life_intervals]}),
                    ("b_t", {"shape": [life_intervals]}),
                    ("b_E_SOC_t", {"shape": [life_intervals + 1]}),
                ],
                [
                    ("hpp_t_with_deg", {"shape": [life_intervals]}),
                    ("hpp_curt_t_with_deg", {"shape": [life_intervals]}),
                    ("b_t_with_deg", {"shape": [life_intervals]}),
                    ("b_E_SOC_t_with_deg", {"shape": [life_intervals + 1]}),
                    ("penalty_t_with_deg", {"shape": [life_intervals]}),
                    ("total_curtailment_with_deg",),
                    ("total_curtailment",),
                ],
                ems_long_term,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            wind_cost_comp=ComponentWrapper(
                [
                    ("Nwt",),
                    ("hh",),
                    ("d",),
                    ("p_rated",),
                    ("wind_t", {"shape": [N_time]}),
                ],
                [
                    ("CAPEX_w",),
                    ("OPEX_w",),
                ],
                wind_cost,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            solar_cost_comp=ComponentWrapper(
                [
                    ("solar_MW",),
                    ("DC_AC_ratio",),
                ],
                [
                    ("CAPEX_s",),
                    ("OPEX_s",),
                ],
                solar_cost,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            batt_cost_comp=ComponentWrapper(
                [
                    ("b_E",),
                    ("b_P",),
                    ("SoH", {"shape": [life_intervals]}),
                ],
                [
                    ("CAPEX_b",),
                    ("OPEX_b",),
                ],
                batt_cost,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            shared_cost_comp=ComponentWrapper(
                [
                    ("Awpp",),
                    ("Apvp",),
                ],
                [
                    ("CAPEX_sh",),
                    ("OPEX_sh",),
                ],
                shared_cost,
                partial_options=[{"dependent": False, "val": 0}],
            ),
            finance_comp=ComponentWrapper(
                [
                    ("hpp_t_with_deg", {"shape": [life_intervals]}),
                    ("penalty_t_with_deg", {"shape": [life_intervals]}),
                    ("price_t_ext", {"shape": [life_intervals]}),
                    ("CAPEX_w",),
                    ("CAPEX_s",),
                    ("CAPEX_b",),
                    ("CAPEX_sh",),
                    ("OPEX_w",),
                    ("OPEX_s",),
                    ("OPEX_b",),
                    ("OPEX_sh",),
                ],
                [
                    ("CAPEX",),
                    ("OPEX",),
                    ("revenues_mean",),
                    ("NPV",),
                    ("IRR",),
                    ("NPV_over_CAPEX",),
                    ("LCOE",),
                    ("mean_AEP",),
                    ("penalty_lifetime",),
                    ("break_even_PPA_price",),
                ],
                finance,
                partial_options=[{"dependent": False, "val": 0}],
            ),
        )

        prob = om.Problem(reports=None)
        for k, v in comps.items():
            prob.model.add_subsystem(k, v, promotes=["*"])
        prob.model.options["auto_order"] = True
        prob.setup()
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
            yaw,
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
                prob["revenues_mean"] / 1e6,
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

    import time

    from py_wake import NOJ
    from py_wake.deflection_models import JimenezWakeDeflection
    from py_wake.examples.data.dtu10mw_surrogate import DTU10MW_1WT_Surrogate
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site
    from py_wake.turbulence_models.stf import STF2017TurbulenceModel
    from topfarm.utils import regular_generic_layout

    from hydesign.examples import examples_filepath

    name = "France_good_wind"
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site["longitude"].values[0]
    latitude = ex_site["latitude"].values[0]
    altitude = ex_site["altitude"].values[0]

    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]

    life_y = 25
    intervals_per_hour = 1
    n_wt = 10
    n_loads = 4
    wt = DTU10MW_1WT_Surrogate()
    d = wt.diameter()
    site = Hornsrev1Site()
    sx = 4 * d
    sy = 5 * d
    x, y = regular_generic_layout(n_wt, sx, sy, stagger=0, rotation=0)
    N_ws = 365 * 24 * intervals_per_hour
    time_stamp = np.arange(N_ws) / 6 / 24
    farm = NOJ(
        site,
        wt,
        turbulenceModel=STF2017TurbulenceModel(),
        deflectionModel=JimenezWakeDeflection(),
    )

    yaw = 30 * np.sin(np.arange(N_ws) / 100)
    tilt = np.zeros(N_ws)

    hpp = hpp_model(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        sim_pars_fn=sim_pars_fn,
        input_ts_fn=input_ts_fn,
        intervals_per_hour=intervals_per_hour,
        farm=farm,
        x=x,
        y=y,
        tilt=tilt,
        time_stamp=time_stamp,
        n_loads=n_loads,
    )

    start = time.time()

    # Wind plant design
    x = dict(
        clearance=55.0,
        sp=257.0,
        p_rated=10.0,
        Nwt=n_wt,
        wind_MW_per_km2=5.917,
        # PV plant design
        solar_MW=75,
        surface_tilt=28.125,
        surface_azimuth=191.250,
        DC_AC_ratio=1.479,
        # Energy storage & EMS price constrains
        b_P=27,
        b_E_h=4,
        cost_of_battery_P_fluct_in_peak_price_ratio=8.750,
        # Wind turbine control
        yaw=yaw,
    )

    outs = hpp.evaluate(**x)

    hpp.print_design(list(x.values()), outs)

    end = time.time()
    print("exec. time [min]:", (end - start) / 60)

    sensors = wt.loadFunction.output_keys
    sensor_no = 0
    sensor = sensors[sensor_no]
    turb = 0
    import matplotlib.pyplot as plt

    plt.plot(hpp.prob["loads_rel_ext"][sensor_no, turb, :])
