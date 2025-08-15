import os

import numpy as np
import openmdao.api as om
import pandas as pd

from hydesign.assembly.hpp_assembly_P2X import hpp_model_P2X
from hydesign.costs.costs import battery_cost_comp as battery_cost
from hydesign.costs.costs import ptg_cost_comp as ptg_cost
from hydesign.costs.costs import pvp_cost_comp as pvp_cost
from hydesign.costs.costs import shared_cost_comp as shared_cost
from hydesign.costs.costs import wpp_cost_comp as wpp_cost
from hydesign.ems.ems_P2X_bidirectional import ems_P2X_bidirectional_comp as ems
from hydesign.finance.finance_P2X_bidirectional import (
    finance_P2X_bidirectional_comp as finance,
)
from hydesign.pv.pv import pvp_comp as pvp
from hydesign.weather.weather import ABL_comp as ABL
from hydesign.wind.wind import genericWake_surrogate_comp as genericWake_surrogate
from hydesign.wind.wind import (
    genericWT_surrogate_comp as genericWT_surrogate,  # , wpp_with_degradation, get_rotor_area
)
from hydesign.wind.wind import (
    get_rotor_d,
)
from hydesign.wind.wind import wpp_comp as wpp


class hpp_model_P2X_bidirectional(hpp_model_P2X):
    """HPP design evaluator"""

    def __init__(self, sim_pars_fn, **kwargs):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP
        """
        hpp_model_P2X.__init__(self, sim_pars_fn=sim_pars_fn, **kwargs)

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        life_y = self.life_y
        price = self.price

        input_ts_fn = sim_pars["input_ts_fn"]
        genWT_fn = sim_pars["genWT_fn"]
        genWake_fn = sim_pars["genWake_fn"]
        latitude = sim_pars["latitude"]
        longitude = sim_pars["longitude"]
        altitude = sim_pars["altitude"]
        ems_type = sim_pars["ems_type"]
        H2_demand = sim_pars["H2_demand"]

        electrolyzer_eff_fn = os.path.join(
            os.path.dirname(sim_pars_fn), "Electrolyzer_efficiency_curves.csv"
        )
        df = pd.read_csv(electrolyzer_eff_fn)
        electrolyzer_eff_curve_name = sim_pars["electrolyzer_eff_curve_name"]
        col_no = df.columns.get_loc(electrolyzer_eff_curve_name)
        my_df = df.iloc[:, col_no : col_no + 2].dropna()
        eff_curve = my_df[1:].values.astype(float)
        electrolyzer_eff_curve_type = sim_pars["electrolyzer_eff_curve_type"]

        # model = om.Group()
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
                    eff_curve=eff_curve,
                    # life_h = life_h,
                    ems_type=ems_type,
                    electrolyzer_eff_curve_type=electrolyzer_eff_curve_type,
                    price_H2=sim_pars["price_H2"],
                    storage_eff=sim_pars["storage_eff"],
                    hhv=sim_pars["hhv"],
                    penalty_factor_H2=sim_pars["penalty_factor_H2"],
                    min_power_standby=sim_pars["min_power_standby"],
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
                "ptg_cost",
                ptg_cost(
                    electrolyzer_capex_cost=sim_pars["electrolyzer_capex_cost"],
                    electrolyzer_opex_cost=sim_pars["electrolyzer_opex_cost"],
                    electrolyzer_power_electronics_cost=sim_pars[
                        "electrolyzer_power_electronics_cost"
                    ],
                    water_cost=sim_pars["water_cost"],
                    water_treatment_cost=sim_pars["water_treatment_cost"],
                    water_consumption=sim_pars["water_consumption"],
                    storage_capex_cost=sim_pars["H2_storage_capex_cost"],
                    storage_opex_cost=sim_pars["H2_storage_opex_cost"],
                    transportation_cost=sim_pars["H2_transportation_cost"],
                    transportation_distance=sim_pars["H2_transportation_distance"],
                    N_time=N_time,
                ),
            ),
            (
                "finance",
                finance(
                    N_time=N_time,
                    depreciation_yr=sim_pars["depreciation_yr"],
                    depreciation=sim_pars["depreciation"],
                    inflation_yr=sim_pars["inflation_yr"],
                    inflation=sim_pars["inflation"],
                    ref_yr_inflation=sim_pars["ref_yr_inflation"],
                    phasing_yr=sim_pars["phasing_yr"],
                    phasing_CAPEX=sim_pars["phasing_CAPEX"],
                    price_H2=sim_pars["price_H2"],
                    wind_WACC=sim_pars["wind_WACC"],
                    solar_WACC=sim_pars["solar_WACC"],
                    battery_WACC=sim_pars["battery_WACC"],
                    ptg_WACC=sim_pars["ptg_WACC"],
                    tax_rate=sim_pars["tax_rate"],
                ),
                {
                    "CAPEX_el": "CAPEX_sh",
                    "OPEX_el": "OPEX_sh",
                },
            ),
        ]

        # model.connect('shared_cost.CAPEX_sh', 'finance.CAPEX_el')
        # model.connect('shared_cost.OPEX_sh', 'finance.OPEX_el')

        prob = self.get_prob(comps)

        prob.setup()

        # Additional parameters
        prob.set_val("price_t", price)
        prob.set_val("m_H2_demand_t", H2_demand)
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
        prob.set_val("land_use_per_solar_MW", sim_pars["land_use_per_solar_MW"])

        self.prob = prob

        self.list_out_vars = [
            "NPV_over_CAPEX",
            "NPV [MEuro]",
            "IRR",
            "LCOE [Euro/MWh]",
            "LCOH [Euro/kg]",
            "Revenue [MEuro]",
            "CAPEX [MEuro]",
            "OPEX [MEuro]",
            "penalty lifetime [MEuro]",
            "AEP [GWh]",
            "annual_Power2Grid [GWh]",
            "GUF",
            "annual_H2 [tons]",
            "annual_P_ptg [GWh]",
            "annual_P_ptg_H2 [GWh]",
            "grid [MW]",
            "wind [MW]",
            "solar [MW]",
            "PtG [MW]",
            "HSS [kg]",
            "Battery Energy [MWh]",
            "Battery Power [MW]",
            "Total curtailment [GWh]",
            "Awpp [km2]",
            "Apvp [km2]",
            "Rotor diam [m]",
            "Hub height [m]",
            "Number of batteries used in lifetime",
            "Break-even H2 price [Euro/kg]",
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
            "ptg_MW [MW]",
            "HSS_kg [kg]",
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
        # PtG plant design
        ptg_MW,
        # Hydrogen storage capacity
        HSS_kg,
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
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]
        ptg_MW: Electrolyzer capacity [MW]
        HSS_kg: Hydrogen storgae capacity [kg]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['LCOH'] : Levelized cost of hydrogen
        prob['Revenue'] : Revenue of HPP
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['AEP']: Annual energy production
        prob['mean_Power2Grid']: Power to grid
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        prob['annual_H2']: Annual H2 production
        prob['annual_P_ptg']: Annual power converted to hydrogen
        prob['annual_P_ptg_H2']: Annual power from grid converted to hydrogen
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        ptg_MW: Electrolyzer capacity [MW]
        HSS_kg: Hydrogen storgae capacity [kg]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
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
            b_P,
            b_E_h,
            cost_of_battery_P_fluct_in_peak_price_ratio,
            ptg_MW,
            HSS_kg,
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
        prob.set_val("ptg_MW", ptg_MW)
        prob.set_val("HSS_kg", HSS_kg)

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
                prob.get_val("wind_t").mean() / p_rated / Nwt
            )  # Capacity factor of wind only

        outputs = np.hstack(
            [
                prob["NPV_over_CAPEX"],
                prob["NPV"] / 1e6,
                prob["IRR"],
                prob["LCOE"],
                prob["LCOH"],
                prob["Revenue"] / 1e6,
                prob["CAPEX"] / 1e6,
                prob["OPEX"] / 1e6,
                prob["penalty_lifetime"] / 1e6,  # 9
                prob["mean_AEP"] / 1e3,  # [GWh]
                prob["mean_Power2Grid"] / 1e3,  # GWh
                # Grid Utilization factor
                prob["mean_AEP"] / (self.sim_pars["G_MW"] * 365 * 24),
                prob["annual_H2"] / 1e3,  # in tons
                prob["annual_P_ptg"] / 1e3,  # in GWh
                prob["annual_P_ptg_H2"] / 1e3,  # in GWh               #15
                self.sim_pars["G_MW"],
                wind_MW,
                solar_MW,
                ptg_MW,
                HSS_kg,
                b_E,
                b_P,
                prob["total_curtailment"] / 1e3,  # [GWh]
                Awpp,
                prob.get_val("Apvp"),
                d,
                hh,
                1 * (b_P > 0),
                prob["break_even_H2_price"],  # 29
                prob["break_even_PPA_price"],  # 30
                cf_wind,
            ]
        )
        self.outputs = outputs
        return outputs


if __name__ == "__main__":
    import time

    from hydesign.examples import examples_filepath

    name = "Denmark_good_wind"
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site["longitude"].values[0]
    latitude = ex_site["latitude"].values[0]
    altitude = ex_site["altitude"].values[0]

    sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
    input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]
    H2_demand_fn = examples_filepath + ex_site["H2_demand_col"].values[0]

    hpp = hpp_model_P2X_bidirectional(
        latitude=latitude,
        longitude=longitude,
        altitude=altitude,
        max_num_batteries_allowed=10,
        work_dir="./",
        sim_pars_fn=sim_pars_fn,
        input_ts_fn=input_ts_fn,
        H2_demand_fn=H2_demand_fn,
        electrolyzer_eff_curve_name="Alkaline electrolyzer H2 production",
        penalty_factor_H2=0.5,
    )

    x = [50, 300, 10, 40, 10, 0, 45, 180, 1.5, 40, 4, 5, 250, 5000]

    start = time.time()

    outs = hpp.evaluate(*x)

    hpp.print_design()

    end = time.time()
    print("exec. time [min]:", (end - start) / 60)
