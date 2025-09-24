import os
import sys

import numpy as np
import openmdao.api as om
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.costs.costs_P2MeOH import (
    P2MeOH_cost,
    battery_cost,
    pvp_cost,
    shared_cost,
    wpp_cost,
)
from hydesign.ems.ems_P2MeOH_bidirectional import (
    ems_P2MeOH_bidirectional_comp as ems_P2MeOH_bidirectional,
)
from hydesign.finance.finance_P2MeOH_bidirectional import (
    finance_P2MeOH_bidirectional_comp as finance_P2MeOH_bidirectional,
)
from hydesign.pv.pv import pvp_comp as pvp  # , pvp_with_degradation
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.wind.wind import (
    genericWake_surrogate_comp as genericWake_surrogate,  # , wpp_with_degradation, get_rotor_area
)
from hydesign.wind.wind import genericWT_surrogate_comp as genericWT_surrogate
from hydesign.wind.wind import (
    get_rotor_d,
)
from hydesign.wind.wind import wpp_comp as wpp

# Add the parent directory to the Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


# sys.path.append('C:/Users/maelg/OneDrive/Documents/DTU/Master_Thesis/hydesign')


class hpp_model_P2MeOH_bidirectional(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
        MeOH_demand_fn=None,  # If input_ts_fn is given it should include MeOH_demand column.
        **kwargs,
    ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        MeOH_demand_fn : MeOH demand time series file path
        """
        defaults = {
            "electrolyzer_eff_curve_name": "SOEC load",
            "electrolyzer_eff_curve_type": "efficiency",
            "batch_size": 24 * 1,  # hours, should be a multiple of 24
        }
        hpp_base.__init__(self, sim_pars_fn=sim_pars_fn, defaults=defaults, **kwargs)

        self.life_h = self.sim_pars.get("life_h", 219000)  # 25 y in h
        self.life_y = self.sim_pars.get("life_y", int(self.life_h / self.N_time))
        self.load_min_penalty_factor = self.sim_pars.get("load_min_penalty_factor", 1e6)

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        life_h = self.life_h
        life_y = self.life_y
        price = self.price
        load_min_penalty_factor = self.load_min_penalty_factor

        input_ts_fn = sim_pars["input_ts_fn"]
        genWT_fn = sim_pars["genWT_fn"]
        genWake_fn = sim_pars["genWake_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        ems_type = sim_pars["ems_type"]
        batch_size = sim_pars["batch_size"]

        weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
        MeOH_demand_data = pd.read_csv(
            MeOH_demand_fn, index_col=0, parse_dates=True
        ).loc[weather.index, :]
        electrolyzer_eff_curve_type = sim_pars["electrolyzer_eff_curve_type"]
        eff_curve = [[0, 0], [0.02, 0], [0.05, 0.89], [0.37, 0.89], [1, 0.69]]
        eff_curve_prod_no_eff = [
            [0, 0],
            [0.02, 0],
            [0.05, 1.5],
            [0.37, 11.1],
            [1, 30.01],
        ]  # To calculate the power purchased from the grid that supplies the SOEC. The efficiency is a constant in this case.

        MeOH_demand_scheme = sim_pars["MeOH_demand_scheme"]

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
                "ems_P2MeOH_bidirectional",
                ems_P2MeOH_bidirectional(
                    N_time=N_time,
                    eff_curve=eff_curve,
                    eff_curve_prod_no_eff=eff_curve_prod_no_eff,
                    MeOH_demand_scheme=MeOH_demand_scheme,
                    life_h=life_h,
                    ems_type=ems_type,
                    load_min_penalty_factor=load_min_penalty_factor,
                    electrolyzer_eff_curve_type=electrolyzer_eff_curve_type,
                    batch_size=batch_size,
                ),
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
                    life_h=life_h,
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
                "P2MeOH_cost",
                P2MeOH_cost(
                    SOEC_capex_cost=sim_pars["SOEC_capex_cost"],
                    SOEC_opex_cost=sim_pars["SOEC_opex_cost"],
                    SOEC_power_electronics_cost=sim_pars["SOEC_power_electronics_cost"],
                    water_cost=sim_pars["water_cost"],
                    water_treatment_cost=sim_pars["water_treatment_cost"],
                    water_consumption=sim_pars["water_consumption"],
                    heater_capex_cost=sim_pars["heater_capex_cost"],
                    heater_opex_cost=sim_pars["heater_opex_cost"],
                    DAC_capex_cost=sim_pars["DAC_capex_cost"],
                    DAC_opex_cost=sim_pars["DAC_opex_cost"],
                    MeOH_reactor_capex_cost=sim_pars["MeOH_reactor_capex_cost"],
                    MeOH_reactor_opex_cost=sim_pars["MeOH_reactor_opex_cost"],
                    MeOH_tank_capex_cost=sim_pars["MeOH_tank_capex_cost"],
                    MeOH_tank_opex_cost=sim_pars["MeOH_tank_opex_cost"],
                    N_time=N_time,
                    MeOH_demand_scheme=MeOH_demand_scheme,
                    life_h=life_h,
                ),
            ),
            (
                "finance_P2MeOH_bidirectional",
                finance_P2MeOH_bidirectional(
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
                    life_h=life_h,
                ),
            ),
        ]

        prob = self.get_prob(comps)

        prob.setup()

        # Additional parameters
        prob.set_val("elec_spot_price_t", price)
        prob.set_val("elec_grid_price_t", 0.85 * np.max(price))
        prob.set_val("m_MeOH_demand_t", MeOH_demand_data["MeOH_demand"])
        prob.set_val("hpp_grid_connection", sim_pars["hpp_grid_connection"])
        prob.set_val(
            "battery_depth_of_discharge", sim_pars["battery_depth_of_discharge"]
        )
        prob.set_val(
            "battery_charging_efficiency", sim_pars["battery_charging_efficiency"]
        )
        prob.set_val(
            "battery_price_reduction_per_year",
            sim_pars["battery_price_reduction_per_year"],
        )
        prob.set_val("peak_hr_quantile", sim_pars["peak_hr_quantile"])
        prob.set_val(
            "n_full_power_hours_expected_per_day_at_peak_price",
            sim_pars["n_full_power_hours_expected_per_day_at_peak_price"],
        )
        prob.set_val("wind_WACC", sim_pars["wind_WACC"])
        prob.set_val("solar_WACC", sim_pars["solar_WACC"])
        prob.set_val("battery_WACC", sim_pars["battery_WACC"])
        prob.set_val("P2MeOH_WACC", sim_pars["P2MeOH_WACC"])
        prob.set_val("tax_rate", sim_pars["tax_rate"])
        prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])
        prob.set_val("price_green_MeOH", sim_pars["price_green_MeOH"])
        prob.set_val("price_grid_MeOH", sim_pars["price_grid_MeOH"])
        prob.set_val("lhv", sim_pars["lhv"])
        # prob.set_val('heater_efficiency', sim_pars['heater_efficiency'])
        prob.set_val("psi_DAC_MWhkg", sim_pars["psi_DAC_MWhkg"])
        prob.set_val("phi_DAC_MWhkg", sim_pars["phi_DAC_MWhkg"])
        prob.set_val("M_CO2", sim_pars["M_CO2"])
        prob.set_val("M_H2", sim_pars["M_H2"])
        prob.set_val("M_H2O", sim_pars["M_H2O"])
        prob.set_val("M_CH3OH", sim_pars["M_CH3OH"])
        prob.set_val("w_comp_reactor_MWhkg", sim_pars["w_comp_reactor_MWhkg"])
        prob.set_val("H2O_yield", sim_pars["H2O_yield"])
        prob.set_val("MeOH_yield", sim_pars["MeOH_yield"])
        prob.set_val("m_MeOH_tank_flow_max_kg", sim_pars["m_MeOH_tank_flow_max_kg"])
        prob.set_val(
            "charging_efficiency_MeOH_tank", sim_pars["charging_efficiency_MeOH_tank"]
        )
        prob.set_val("w", sim_pars["w"])
        prob.set_val("penalty_factor_MeOH", sim_pars["penalty_factor_MeOH"])

        self.prob = prob

        self.list_out_vars = [
            "NPV_over_CAPEX",
            "NPV",
            "IRR",
            "LCOE",
            "LCOgreenMeOH",
            "LCOgridMeOH",
            "Revenue",
            "CAPEX",
            "OPEX",
            "penalty lifetime",
            "Mean Annual Electricity Sold [GWh]",
            "mean_Power2Grid",
            "GUF",
            "annual_green_MeOH",
            "annual_grid_MeOH",
            "annual_P_SOEC_green",
            "annual_P_purch_grid",
            "hpp_grid_connection",
            "wind_MW",
            "solar_MW",
            "P_SOEC_MW",
            # 'P_heater_MW',
            "P_DAC_MW",
            "P_reactor_MW",
            "m_MeOH_tank_max_kg",
            "E_batt_MWh_t",
            "P_batt_MW",
            "total_curtailment",
            "Awpp",
            "Apvp",
            "d",
            "hh",
            "break_even_PPA_price",
            "break_even_green_MeOH_price",
            "cf_wind",
            "AEP [GWh]",
        ]

        self.list_vars = [
            "clearance",
            "sp",
            "p_rated",
            "Nwt",
            "wind_MW_per_km2",
            "solar_MW",
            "surface_tilt",
            "surface_azimuth",
            "DC_AC_ratio",
            "P_batt_MW",
            "b_E_h",
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            "P_SOEC_MW",
            # 'P_heater_MW',
            # 'P_DAC_MW',
            # 'P_reactor_MW',
            "m_MeOH_tank_max_kg",
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
        P_batt_MW,
        b_E_h,
        cost_of_battery_P_fluct_in_peak_price_ratio,
        # PtMeOH plant design
        P_SOEC_MW,
        # P_heater_MW, P_DAC_MW, P_reactor_MW,
        # Methanol storage capacity
        m_MeOH_tank_max_kg,
    ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        clearance : Distance from the ground to the tip of the blade [m]
        sp : Specific power of the turbine [MW/m2]
        p_rated : Rated powe of the turbine [MW]
        Nwt : Number of wind turbines
        wind_MW_per_km2 : Wind power installation density [MW/km2]
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        P_batt_MW : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]
        P_SOEC_MW: SOEC capacity [MW]
        P_heater_MW: Heater capacity [MW]
        P_DAC_MW: DAC capacity [MW]
        P_reactor_MW: Reactor capacity [MW]
        m_MeOH_tank_max_kg: Methanol storgae capacity [kg]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['LCOgreenMeOH'] : Levelized cost of green methanol
        prob['LCOgridMeOH'] : Levelized cost of grid methanol
        prob['Revenue'] : Revenue of HPP
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['mean_Power2Grid']: Power to grid
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        prob['annual_green_MeOH']: Annual green methanol production
        prob['annual_grid_MeOH']: Annual grid methanol production
        prob['annual_P_SOEC_green']: Annual green power consumed by the SOEC
        prob['annual_P_purch_grid']: Annual grid power purchased
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        P_SOEC_MW: SOEC capacity [MW]
        P_heater_MW: Heater capacity [MW]
        P_DAC_MW: DAC capacity [MW]
        P_reactor_MW: Reactor capacity [MW]
        m_MeOH_tank_max_kg: MeOH tank capacity [kg]
        P_batt_MW : Battery power [MW]
        E_batt_MWh_t : Battery energy [MWh]
        prob['total_curtailment']/1e3 : Total curtailed power [GW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        num_batteries : Number of batteries
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
            P_batt_MW,
            b_E_h,
            cost_of_battery_P_fluct_in_peak_price_ratio,
            P_SOEC_MW,
            m_MeOH_tank_max_kg,
        ]

        prob = self.prob

        d = get_rotor_d(p_rated * 1e6 / sp)
        hh = (d / 2) + clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2
        E_batt_MWh_t = b_E_h * P_batt_MW

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
        prob.set_val("P_SOEC_MW", P_SOEC_MW)
        prob.set_val("m_MeOH_tank_max_kg", m_MeOH_tank_max_kg)

        prob.set_val("P_batt_MW", P_batt_MW)
        prob.set_val("E_batt_MWh_t", E_batt_MWh_t)
        prob.set_val(
            "ems_P2MeOH_bidirectional.cost_of_battery_P_fluct_in_peak_price_ratio",
            cost_of_battery_P_fluct_in_peak_price_ratio,
        )

        prob.run_model()

        self.prob = prob

        eff_cor = 0.69

        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = (
                prob.get_val("wpp.wind_t").mean() / p_rated / Nwt
            )  # Capacity factor of wind only
        AEP = (
            (prob["wind_t"].mean() + prob["solar_t"].mean()) * 1e-3 * 24 * 365
        )  # Annual energy production [MWh]
        outputs = [
            prob["NPV_over_CAPEX"],
            prob["NPV"] / 1e6,
            prob["IRR"],
            prob["LCOE"],
            prob["LCOgreenMeOH"],
            prob["LCOgridMeOH"],
            prob["Revenue"] / 1e6,
            prob["CAPEX"] / 1e6,
            prob["OPEX"] / 1e6,
            prob["penalty_lifetime"] / 1e6,
            np.mean(prob["P_HPP_t"]) * 24 * 365 / 1e3,  # [GWh]
            prob["mean_Power2Grid"] / 1e3,  # GWh
            prob["mean_Power2Grid"] / (self.sim_pars["hpp_grid_connection"] * 365 * 24),
            prob["annual_green_MeOH"] / 1e3,  # in tons
            prob["annual_grid_MeOH"] / 1e3,  # in tons
            prob["annual_P_SOEC_green"] / 1e3,  # in GWh
            prob["annual_P_purch_grid"] / 1e3,  # in GWh
            self.sim_pars["hpp_grid_connection"],
            wind_MW,
            solar_MW,
            P_SOEC_MW,
            # P_heater_MW,
            P_SOEC_MW
            * self.sim_pars["psi_DAC_MWhkg"]
            * self.sim_pars["M_CO2"]
            * eff_cor
            / (3 * self.sim_pars["M_H2"] * self.sim_pars["lhv"]),
            P_SOEC_MW
            * self.sim_pars["w_comp_reactor_MWhkg"]
            * self.sim_pars["MeOH_yield"]
            * self.sim_pars["M_CH3OH"]
            * eff_cor
            / (3 * self.sim_pars["M_H2"] * self.sim_pars["lhv"]),
            m_MeOH_tank_max_kg,
            E_batt_MWh_t,
            P_batt_MW,
            prob["total_curtailment"] / 1e3,  # [GWh]
            Awpp,
            prob.get_val("shared_cost.Apvp"),
            d,
            hh,
            prob["break_even_PPA_price"],
            prob["break_even_green_MeOH_price"],
            cf_wind,
            AEP,
        ]
        self.outputs = outputs
        return outputs


if __name__ == "__main__":
    import time

    from hydesign.examples import examples_filepath

    name = "France_good_wind_MeOH"
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site["longitude"].values[0]
    latitude = ex_site["latitude"].values[0]
    altitude = ex_site["altitude"].values[0]

    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]
    MeOH_demand_fn = examples_filepath + "Europe/MeOH_demand.csv"

    hpp = hpp_model_P2MeOH_bidirectional(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        sim_pars_fn=sim_pars_fn,
        input_ts_fn=input_ts_fn,
        MeOH_demand_fn=MeOH_demand_fn,
        batch_size=24 * 1,
    )

    start = time.time()

    clearance = 10
    sp = 303
    p_rated = 5
    Nwt = 48
    wind_MW_per_km2 = 3
    solar_MW = 80
    surface_tilt = 50
    surface_azimuth = 210
    DC_AC_ratio = 1.5
    P_batt_MW = 50
    b_E_h = 3
    cost_of_battery_P_fluct_in_peak_price_ratio = 5
    P_SOEC_MW = 100
    # P_heater_MW = 0
    # P_DAC_MW = 5.6
    # P_reactor_MW = 7.3
    m_MeOH_tank_max_kg = 1800000

    x = [
        clearance,
        sp,
        p_rated,
        Nwt,
        wind_MW_per_km2,
        solar_MW,
        surface_tilt,
        surface_azimuth,
        DC_AC_ratio,
        P_batt_MW,
        b_E_h,
        cost_of_battery_P_fluct_in_peak_price_ratio,
        P_SOEC_MW,
        m_MeOH_tank_max_kg,
    ]

    outs = hpp.evaluate(*x)

    hpp.print_design()

    end = time.time()
    print("exec. time [min]:", (end - start) / 60)

    print(hpp.prob["NPV_over_CAPEX"])
