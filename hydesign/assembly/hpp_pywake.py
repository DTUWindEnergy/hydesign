# %%
import numpy as np
import openmdao.api as om
import pandas as pd

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
from hydesign.ems.ems import ems_comp, ems_long_term_operation_comp, expand_to_lifetime
from hydesign.finance.finance import finance_comp
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.pv.pv import pvp_comp, pvp_with_degradation_comp
from hydesign.reliability import (
    battery_with_reliability_comp,
    pvp_with_reliability_comp,
    wpp_with_reliability_comp,
)
from hydesign.weather.weather import ABL_comp
from hydesign.wind.wind import (
    get_pywake_farm_pc,
    get_rotor_d,
    get_wind_ts_degradation_2d,
)


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
        N_time = self.N_time
        wpp_efficiency = self.wpp_efficiency
        life_y = self.life_y
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg

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

        life_h = 365 * 24 * life_y
        life_intervals = life_h * intervals_per_hour
        pc, pc_ws, pc_wd, pc_yaw, pc_tilt = get_pywake_farm_pc(farm, x, y)

        def pywake(ws, wd, yaw, **kwargs):
            sim_res = farm(x=x, y=y, yaw=yaw, tilt=tilt, wd=wd, ws=ws, time=time_stamp)
            sim_res["duration"] = 0

            wind_t = sim_res.Power.sum("wt").values * wpp_efficiency * 1e-6  # W -> MW
            wind_t_ext_deg_raw = expand_to_lifetime(
                wpp_efficiency
                * get_wind_ts_degradation_2d(
                    ws=pc_ws,
                    wd=pc_wd,
                    pc=pc,
                    ws_ts=ws,
                    wd_ts=wd,
                    yr=wind_deg_yr,
                    wind_deg=wind_deg,
                    life=ws.size,
                ),
                life_y=life_y,
                intervals_per_hour=intervals_per_hour,
            )

            return [wind_t, wind_t_ext_deg_raw]

        comps = [
            (
                "abl",
                ABL_comp(weather_fn=input_ts_fn, N_time=N_time),
            ),
            (
                "pywake_comp",
                ComponentWrapper(
                    [
                        ("ws", {"shape": [N_time]}),
                        ("wd", {"shape": [N_time]}),
                        ("yaw", {"shape": [N_time]}),
                    ],
                    [
                        ("wind_t", {"shape": [N_time], "units": "MW"}),
                        ("wind_t_ext_deg_raw", {"shape": [life_intervals]}),
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
                {"wind_t": "wind_t_ext_deg"},
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

    import time

    from py_wake import NOJ
    from py_wake.deflection_models import JimenezWakeDeflection
    from py_wake.examples.data.hornsrev1 import Hornsrev1Site, HornsrevV80
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
    n_wt = 30
    wt = HornsrevV80()
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
        solar_MW=0,
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
