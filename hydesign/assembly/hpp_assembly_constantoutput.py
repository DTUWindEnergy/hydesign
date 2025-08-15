import numpy as np

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.battery_degradation import battery_degradation_comp as battery_degradation
from hydesign.battery_degradation import (
    battery_loss_in_capacity_due_to_temp_comp as battery_loss_in_capacity_due_to_temp,
)
from hydesign.costs.costs import battery_cost_comp as battery_cost
from hydesign.costs.costs import pvp_cost_comp as pvp_cost
from hydesign.costs.costs import shared_cost_comp as shared_cost
from hydesign.costs.costs import wpp_cost_comp as wpp_cost
from hydesign.ems.ems_constant_output import ems_constantoutput_comp as ems
from hydesign.ems.ems_constant_output import (
    ems_long_term_operation_comp as ems_long_term_operation,
)
from hydesign.finance.finance import finance_comp as finance
from hydesign.pv.pv import pvp_comp as pvp
from hydesign.pv.pv import pvp_with_degradation_comp as pvp_with_degradation
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.wind.wind import genericWake_surrogate_comp as genericWake_surrogate
from hydesign.wind.wind import genericWT_surrogate_comp as genericWT_surrogate
from hydesign.wind.wind import (
    get_rotor_d,
)
from hydesign.wind.wind import wpp_comp as wpp
from hydesign.wind.wind import wpp_with_degradation_comp as wpp_with_degradation


class hpp_model_constant_output(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
        battery_deg=False,
        load_min=3,  # MW
        load_min_penalty=100,
        **kwargs,
    ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        battery_deg : boolean - Include battery degradation
        load_min : Minimum load that should be met
        load_min_penalty : Penalty for not meeting minimum load
        """
        self.load_min = load_min
        self.load_min_penalty = load_min_penalty
        self.battery_deg = battery_deg

        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        life_y = self.life_y
        price = self.price

        input_ts_fn = sim_pars["input_ts_fn"]
        genWT_fn = sim_pars["genWT_fn"]
        genWake_fn = sim_pars["genWake_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        weeks_per_season_per_year = sim_pars["weeks_per_season_per_year"]
        ems_type = sim_pars["ems_type"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]
        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]

        comps = [
            (
                "abl",
                ABL(weather_fn=input_ts_fn, N_time=N_time),
            ),
            (
                "genericWT",
                genericWT_surrogate(genWT_fn=genWT_fn, N_ws=N_ws),
            ),
            (
                "genericWake",
                genericWake_surrogate(genWake_fn=genWake_fn, N_ws=N_ws),
            ),
            (
                "wpp",
                wpp(
                    N_time=N_time,
                    N_ws=N_ws,
                    wpp_efficiency=wpp_efficiency,
                ),
            ),
            (
                "pvp",
                pvp(
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
                ems(
                    N_time=N_time,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    ems_type=ems_type,
                ),
            ),
            (
                "battery_degradation",
                battery_degradation(
                    weather_fn=input_ts_fn,  # for extracting temperature
                    num_batteries=max_num_batteries_allowed,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    battery_deg=battery_deg,
                ),
            ),
            (
                "battery_loss_in_capacity_due_to_temp",
                battery_loss_in_capacity_due_to_temp(
                    weather_fn=input_ts_fn,  # for extracting temperature
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    battery_deg=battery_deg,
                ),
            ),
            (
                "wpp_with_degradation",
                wpp_with_degradation(
                    N_time=N_time,
                    N_ws=N_ws,
                    wpp_efficiency=wpp_efficiency,
                    wind_deg_yr=wind_deg_yr,
                    wind_deg=wind_deg,
                    share_WT_deg_types=share_WT_deg_types,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "pvp_with_degradation",
                pvp_with_degradation(
                    pv_deg_yr=sim_pars["pv_deg_yr"],
                    pv_deg=sim_pars["pv_deg"],
                ),
            ),
            (
                "ems_long_term_operation",
                ems_long_term_operation(
                    N_time=N_time,
                    ems_type="constant_output",
                    load_min_penalty_factor=load_min_penalty,
                ),
                {"SoH": "SoH_all"},  # (<parent key>, <child key>)
            ),
            (
                "wpp_cost",
                wpp_cost(
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
                pvp_cost(
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
                battery_cost(
                    battery_energy_cost=sim_pars["battery_energy_cost"],
                    battery_power_cost=sim_pars["battery_power_cost"],
                    battery_BOP_installation_commissioning_cost=sim_pars[
                        "battery_BOP_installation_commissioning_cost"
                    ],
                    battery_control_system_cost=sim_pars["battery_control_system_cost"],
                    battery_energy_onm_cost=sim_pars["battery_energy_onm_cost"],
                    life_y=life_y,
                    battery_price_reduction_per_year=battery_price_reduction_per_year,
                ),
            ),
            (
                "shared_cost",
                shared_cost(
                    hpp_BOS_soft_cost=sim_pars["hpp_BOS_soft_cost"],
                    hpp_grid_connection_cost=sim_pars["hpp_grid_connection_cost"],
                    land_cost=sim_pars["land_cost"],
                ),
            ),
            (
                "finance",
                finance(
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
                ),
                {
                    "CAPEX_el": "CAPEX_sh",
                    "OPEX_el": "OPEX_sh",
                    "penalty_t": "penalty_t_with_deg",
                },  # {<input-key-name in model> (that corresponds to): <output-key-name from prior component>}
            ),
        ]

        prob = self.get_prob(comps)

        prob.setup()

        # Additional parameters
        prob.set_val("price_t", price)
        prob.set_val("G_MW", sim_pars["G_MW"])
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
        prob.set_val("load_min", load_min)

        self.prob = prob

        self.list_out_vars = [
            "NPV_over_CAPEX",
            "NPV [MEuro]",
            "IRR",
            "LCOE [Euro/MWh]",
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
            solar_MW,
            surface_tilt,
            surface_azimuth,
            DC_AC_ratio,
            b_P,
            b_E_h,
            cost_of_battery_P_fluct_in_peak_price_ratio,
        ]
        prob = self.prob

        d = get_rotor_d(p_rated * 1e6 / sp)
        hh = (d / 2) + clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2
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

        prob.run_model()

        self.prob = prob

        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = (
                prob.get_val("wind_t_ext_deg").mean() / p_rated / Nwt
            )  # Capacity factor of wind only

        outputs = np.hstack(
            [
                prob["NPV_over_CAPEX"],
                prob["NPV"] / 1e6,
                prob["IRR"],
                prob["LCOE"],
                prob["CAPEX"] / 1e6,
                prob["OPEX"] / 1e6,
                prob.get_val("CAPEX_w") / 1e6,
                prob.get_val("OPEX_w") / 1e6,
                prob.get_val("CAPEX_s") / 1e6,
                prob.get_val("OPEX_s") / 1e6,
                prob.get_val("CAPEX_b") / 1e6,
                prob.get_val("OPEX_b") / 1e6,
                prob.get_val("CAPEX_sh") / 1e6,
                prob.get_val("OPEX_sh") / 1e6,
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
                Awpp,
                prob.get_val("Apvp"),
                max(Awpp, prob.get_val("Apvp")),
                d,
                hh,
                prob.get_val("n_batteries") * (b_P > 0),
                prob["break_even_PPA_price"],
                cf_wind,
            ]
        )
        self.outputs = outputs
        return outputs


if __name__ == "__main__":
    import time

    import pandas as pd

    from hydesign.examples import examples_filepath

    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    name = "France_good_wind"
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site["longitude"].values[0]
    latitude = ex_site["latitude"].values[0]
    altitude = ex_site["altitude"].values[0]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]
    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0].replace(
        ".yml", "_benchmark.yml"
    )
    H2_demand_fn = examples_filepath + ex_site["H2_demand_col"].values[0]

    PPA = 40  # Euro/MWh
    hpp = hpp_model_constant_output(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        num_batteries=10,
        work_dir="./",
        sim_pars_fn=sim_pars_fn,
        input_ts_fn=input_ts_fn,
        ppa_price=PPA,
        load_min=3,  # MW
        load_min_penalty=100,  # MEuro
        battery_deg=True,
    )
    x = [35.0, 300.0, 10.0, 3, 7.0, 30, 25.0, 180.0, 1.0, 30, 7, 10.0]

    start = time.time()

    outs = hpp.evaluate(*x)

    hpp.print_design()

    end = time.time()
    print("exec. time [min]:", (end - start) / 60)
