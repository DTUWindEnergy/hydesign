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
from hydesign.costs.costs_hybridized_wind import (
    decommissioning_cost_comp as decommissioning_cost,
)
from hydesign.costs.costs_hybridized_wind import shared_cost_comp as shared_cost
from hydesign.ems.ems import ems_comp as ems
from hydesign.ems.ems import ems_long_term_operation_comp as ems_long_term_operation
from hydesign.finance.finance_hybridized_wind import finance_comp as finance
from hydesign.pv.pv import pvp_comp as pvp
from hydesign.pv.pv_hybridization import (
    pvp_with_degradation_comp as pvp_with_degradation,
)
from hydesign.utils import hybridization_shifted_comp as hybridization_shifted
from hydesign.weather.weather_wind_hybridization import ABL_WD_comp as ABL_WD
from hydesign.wind.wind_hybridization import (
    existing_wpp_comp as existing_wpp,  # , genericWT_surrogate, genericWake_surrogate, get_rotor_area, get_rotor_d
)
from hydesign.wind.wind_hybridization import (
    existing_wpp_with_degradation_comp as existing_wpp_with_degradation,
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
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        life_y = self.life_y
        price = self.price

        input_ts_fn = sim_pars["input_ts_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        weeks_per_season_per_year = sim_pars["weeks_per_season_per_year"]
        ems_type = sim_pars["ems_type"]
        max_num_batteries_allowed = sim_pars["max_num_batteries_allowed"]

        existing_wpp_power_curve_xr_fn = os.path.join(
            os.path.dirname(sim_pars_fn), sim_pars["existing_wpp_power_curve_xr_fn"]
        )
        life_h = (self.life_y + N_limit) * 365 * 24
        self.life_h = life_h

        comps = [
            (
                "abl_wd",
                ABL_WD(weather_fn=input_ts_fn, N_time=N_time),
            ),
            (
                "existing_wpp",
                existing_wpp(
                    N_time=N_time,
                    wpp_efficiency=wpp_efficiency,
                    existing_wpp_power_curve_xr_fn=existing_wpp_power_curve_xr_fn,
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
                    life_y=life_y + N_limit,
                    N_time=N_time,
                    weeks_per_season_per_year=weeks_per_season_per_year,
                    ems_type=ems_type,
                ),
            ),
            (
                "battery_degradation",
                battery_degradation(
                    life_y=life_y + N_limit,
                    weather_fn=input_ts_fn,  # for extracting temperature
                    num_batteries=max_num_batteries_allowed,
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
                "battery_loss_in_capacity_due_to_temp",
                battery_loss_in_capacity_due_to_temp(
                    life_y=life_y + N_limit,
                    weather_fn=input_ts_fn,  # for extracting temperature
                    weeks_per_season_per_year=weeks_per_season_per_year,
                ),
            ),
            (
                "existing_wpp_with_degradation",
                existing_wpp_with_degradation(
                    life_h=life_h,
                    N_time=N_time,
                    existing_wpp_power_curve_xr_fn=existing_wpp_power_curve_xr_fn,
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
                    life_y=life_y,
                    N_limit=N_limit,
                    life_h=life_h,
                    pv_deg=sim_pars["pv_deg"],
                ),
            ),
            (
                "ems_long_term_operation",
                ems_long_term_operation(
                    N_time=N_time,
                    life_y=life_y + N_limit,
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
                    # pv_reduction=pv_reduction,
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
                    life_y=life_y + N_limit,
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
                    # Early paying or CAPEX Phasing
                ),
                {
                    "CAPEX_el_w": "CAPEX_sh_w",
                    "CAPEX_el_s": "CAPEX_sh_s",
                    "penalty_t": "penalty_t_with_deg",
                    "OPEX_el": "OPEX_sh",
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

        prob.set_val("hh", sim_pars["hh"])
        prob.set_val("d", sim_pars["d"])
        prob.set_val("p_rated", sim_pars["p_rated"])
        prob.set_val("Nwt", sim_pars["Nwt"])
        prob.set_val("Awpp", sim_pars["Awpp"])

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
            "Shared CAPEX W [MEuro]",
            "Shared CAPEX S [MEuro]",
            "Shared OPEX [MEuro]",
            "penalty lifetime [MEuro]",
            "AEP per year [GWh]",
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
            "solar_MW [MW]",
            "surface_tilt [deg]",
            "surface_azimuth [deg]",
            "DC_AC_ratio",
            "b_P [MW]",
            "b_E_h [h]",
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            "delta_life [years]",
        ]

    def evaluate(
        self,
        # PV plant design
        solar_MW,
        surface_tilt,
        surface_azimuth,
        DC_AC_ratio,
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
            solar_MW,
            surface_tilt,
            surface_azimuth,
            DC_AC_ratio,
            b_P,
            b_E_h,
            cost_of_battery_P_fluct_in_peak_price_ratio,
            delta_life,
        ]

        prob = self.prob

        b_E = b_E_h * b_P

        # pass design variables
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

        prob.set_val("delta_life", delta_life)

        prob.run_model()

        self.prob = prob

        hh = prob["hh"]
        d = prob["d"]
        p_rated = prob["p_rated"]
        Nwt = prob["Nwt"]
        Awpp = prob["Awpp"]

        wind_MW = p_rated * Nwt

        if Nwt == 0:
            cf_wind = np.nan
        else:
            wind_t_ext_degg = prob["wind_t_ext_deg"]
            cf_wind = (
                np.mean(wind_t_ext_degg[wind_t_ext_degg != 0]) / wind_MW
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
                solar_MW,
                b_E,
                b_P,
                prob["total_curtailment"] / 1e3,  # [GWh]
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
