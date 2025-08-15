import os

# basic libraries
import numpy as np
import openmdao.api as om
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.battery_degradation import battery_degradation_comp as battery_degradation
from hydesign.battery_degradation import (
    battery_loss_in_capacity_due_to_temp_comp as battery_loss_in_capacity_due_to_temp,
)
from hydesign.costs.costs import battery_cost_comp as battery_cost
from hydesign.costs.costs import pvp_cost_comp as pvp_cost
from hydesign.costs.costs import wpp_cost_comp as wpp_cost
from hydesign.costs.costs_hybridized_pv import (
    decommissioning_cost_comp as decommissioning_cost,
)
from hydesign.costs.costs_hybridized_pv import shared_cost_comp as shared_cost
from hydesign.ems.ems import ems_comp as ems
from hydesign.ems.ems import ems_long_term_operation_comp as ems_long_term_operation
from hydesign.finance.finance_hybridized_pv import finance_comp as finance
from hydesign.pv.pv import pvp_with_degradation_comp as pvp_with_degradation
from hydesign.pv.pv_hybridization import (
    existing_pvp_comp as existing_pvp,  # , existing_pvp_with_degradation
)
from hydesign.utils import hybridization_shifted_comp as hybridization_shifted
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.wind.wind import genericWake_surrogate_comp as genericWake_surrogate
from hydesign.wind.wind import (
    genericWT_surrogate_comp as genericWT_surrogate,  # , get_rotor_area
)
from hydesign.wind.wind import (
    get_rotor_d,
)
from hydesign.wind.wind import wpp_comp as wpp
from hydesign.wind.wind_hybridization import (
    wpp_with_degradation_comp as wpp_with_degradation,
)


class hpp_model(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
        N_limit=15,
        **kwargs,
    ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        N_limit: NA
        """
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
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

        inverter_eff_fn = os.path.join(
            os.path.dirname(sim_pars_fn), "Inverter_efficiency_curves.csv"
        )
        df = pd.read_csv(inverter_eff_fn)
        inverter_eff_curve_name = sim_pars["inverter_eff_curve_name"]
        col_no = df.columns.get_loc(inverter_eff_curve_name + "_power")
        my_df = df.iloc[:, col_no : col_no + 2].dropna()
        inverter_eff_curve = my_df.values.astype(float)
        life_h = (self.life_y + N_limit) * 365 * 24
        self.life_h = life_h

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
                "existing_pvp",
                existing_pvp(
                    weather_fn=input_ts_fn,
                    N_time=N_time,
                    latitude=latitude,
                    longitude=longitude,
                    altitude=altitude,
                    inverter_eff_curve=inverter_eff_curve,
                    pdc0=sim_pars["pdc0"],
                    v_mp=sim_pars["v_mp"],
                    i_mp=sim_pars["i_mp"],
                    v_oc=sim_pars["v_oc"],
                    i_sc=sim_pars["i_sc"],
                    alpha_sc_spec=sim_pars["alpha_sc_spec"],
                    beta_voc_spec=sim_pars["beta_voc_spec"],
                    gamma_pdc=sim_pars["gamma_pdc"],
                    cells_in_series=sim_pars["cells_in_series"],
                    temp_ref=sim_pars["temp_ref"],
                    celltype=sim_pars["celltype"],
                    panel=sim_pars["panel"],
                    tracking=sim_pars["tracking"],
                    pac0_inv=sim_pars["pac0_inv"],
                    eta_max=sim_pars["eta_max"],
                    eta_euro=sim_pars["eta_euro"],
                    modules_per_string=sim_pars["modules_per_string"],
                    strings_per_inverter=sim_pars["strings_per_inverter"],
                    number_of_inverters=sim_pars["number_of_inverters"],
                    soiling=sim_pars["soiling"],
                    shading=sim_pars["shading"],
                    snow=sim_pars["snow"],
                    mismatch=sim_pars["mismatch"],
                    wiring=sim_pars["wiring"],
                    connections=sim_pars["connections"],
                    lid=sim_pars["lid"],
                    nameplate_rating=sim_pars["nameplate_rating"],
                    age=sim_pars["age"],
                    availability=sim_pars["availability"],
                ),
            ),
            (
                "ems",
                ems(
                    life_y=self.life_y + N_limit,
                    N_time=N_time,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    ems_type=ems_type,
                ),
            ),
            (
                "battery_degradation",
                battery_degradation(
                    life_y=self.life_y + N_limit,
                    weather_fn=input_ts_fn,  # for extracting temperature
                    num_batteries=max_num_batteries_allowed,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "battery_loss_in_capacity_due_to_temp",
                battery_loss_in_capacity_due_to_temp(
                    life_y=self.life_y + N_limit,
                    weather_fn=input_ts_fn,  # for extracting temperature
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "hybridization_shifted",
                hybridization_shifted(
                    N_limit=N_limit,
                    life_y=life_y,
                    N_time=N_time,
                    life_h=life_h,
                ),
                {"SoH": "SoH_all"},
            ),
            (
                "wpp_with_degradation",
                wpp_with_degradation(
                    life_h=life_h,
                    N_limit=N_limit,
                    life_y=life_y,
                    N_time=N_time,
                    N_ws=N_ws,
                    wpp_efficiency=wpp_efficiency,
                    wind_deg=wind_deg,
                    share_WT_deg_types=share_WT_deg_types,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "existing_pvp_with_degradation",
                pvp_with_degradation(
                    life_y=self.life_y + N_limit,
                    pv_deg=sim_pars["pv_deg"],
                    pv_deg_yr=sim_pars["pv_deg_yr"],
                ),
            ),
            (
                "ems_long_term_operation",
                ems_long_term_operation(
                    N_time=N_time,
                    life_y=self.life_y + N_limit,
                ),
                {
                    "SoH": "SoH_shifted",
                },
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
                    life_y=self.life_y + N_limit,
                    battery_price_reduction_per_year=sim_pars[
                        "battery_price_reduction_per_year"
                    ],
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
                "decommissioning_cost",
                decommissioning_cost(
                    decommissioning_cost_w=sim_pars["decommissioning_cost_w"],
                    decommissioning_cost_s=sim_pars["decommissioning_cost_s"],
                ),
            ),
            (
                "finance",
                finance(
                    N_limit=N_limit,
                    life_y=life_y,
                    life_h=life_h,
                    N_time=N_time,
                    # Depreciation curve
                    depreciation_yr=sim_pars["depreciation_yr"],
                    depreciation=sim_pars["depreciation"],
                    depre_rate=sim_pars["depre_rate"],
                    # Inflation curve
                    inflation_yr=sim_pars["inflation_yr"],
                    inflation=sim_pars["inflation"],
                    ref_yr_inflation=sim_pars["ref_yr_inflation"],
                ),
                {
                    "CAPEX_el_w": "CAPEX_sh_w",
                    "CAPEX_el_s": "CAPEX_sh_s",
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
        prob.set_val("hpp_WACC", sim_pars["hpp_WACC"])
        prob.set_val("tax_rate", sim_pars["tax_rate"])
        prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])

        prob.set_val("surface_tilt", sim_pars["surface_tilt"])
        prob.set_val("surface_azimuth", sim_pars["surface_azimuth"])
        prob.set_val("DC_AC_ratio", sim_pars["DC_AC_ratio"])

        self.prob = prob

        self.list_out_vars = [
            "NPV_over_CAPEX",
            "NPV [MEuro]",
            "IRR",
            "LCOE [Euro/MWh]",
            "COVE [-]",
            "CAPEX [MEuro]",
            "OPEX [MEuro]",
            "Wind CAPEX [MEuro]",
            "Wind OPEX [MEuro]",
            "PV CAPEX [MEuro]",
            "PV OPEX [MEuro]",
            "Batt CAPEX [MEuro]",
            "Batt OPEX [MEuro]",
            "Shared CAPEX [MEuro]",
            "Shared OPEX W [MEuro]",
            "Shared OPEX S [MEuro]",
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
            "b_P [MW]",
            "b_E_h [h]",
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            "delta_life [years]",
        ]

    def evaluate(
        self,
        # Wind plant design
        clearance,
        sp,
        p_rated,
        Nwt,
        wind_MW_per_km2,
        # Energy storage & EMS price constrains
        b_P,
        b_E_h,
        cost_of_battery_P_fluct_in_peak_price_ratio,
        # Time desig
        delta_life,
    ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]
        delta_life : Distance in years between the starting operations of the wind plant and of the PV+batteries

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['COVE'] : Cost of Valued Energy
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
            b_P,
            b_E_h,
            cost_of_battery_P_fluct_in_peak_price_ratio,
            delta_life,
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

        prob.set_val("b_P", b_P)
        prob.set_val("b_E", b_E)
        prob.set_val(
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            cost_of_battery_P_fluct_in_peak_price_ratio,
        )

        prob.set_val("delta_life", delta_life)

        prob.run_model()

        self.prob = prob

        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = (
                prob.get_val("wpp_with_degradation.wind_t_ext_deg").mean()
                / p_rated
                / Nwt
            )  # Capacity factor of wind only

        outputs = np.hstack(
            [
                prob["NPV_over_CAPEX"],
                prob["NPV"] / 1e6,
                prob["IRR"],
                prob["LCOE"],
                prob["COVE"],
                prob["CAPEX"] / 1e6,
                prob["OPEX"] / 1e6,
                prob["CAPEX_w"] / 1e6,
                prob["OPEX_w"] / 1e6,
                prob["CAPEX_s"] / 1e6,
                prob["OPEX_s"] / 1e6,
                prob["CAPEX_b"] / 1e6,
                prob["OPEX_b"] / 1e6,
                prob["CAPEX_sh_w"] / 1e6,
                prob["CAPEX_sh_s"] / 1e6,
                prob["OPEX_sh"] / 1e6,
                prob["penalty_lifetime"] / 1e6,
                prob["mean_AEP"] / 1e3,  # [GWh]
                # Grid Utilization factor
                prob["mean_AEP"] / (self.sim_pars["G_MW"] * 365 * 24),
                self.sim_pars["G_MW"],
                wind_MW,
                prob["solar_MW"],
                b_E,
                b_P,
                prob["total_curtailment"] / 1e3,  # [GWh]
                Awpp,
                prob["Apvp"],
                max(Awpp, prob["shared_cost.Apvp"]),
                d,
                hh,
                prob["n_batteries"] * (b_P > 0),
                prob["break_even_PPA_price"],
                cf_wind,
            ]
        )
        self.outputs = outputs
        return outputs
