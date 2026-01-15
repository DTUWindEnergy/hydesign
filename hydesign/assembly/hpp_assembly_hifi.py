# basic libraries
import os

import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.costs.costs import battery_cost_comp as battery_cost
from hydesign.costs.costs import pvp_cost_comp as pvp_cost
from hydesign.costs.costs import shared_cost_comp as shared_cost
from hydesign.costs.costs import wpp_cost_comp as wpp_cost
from hydesign.ems.ems_hifi import ems_comp as ems
from hydesign.examples import examples_filepath
from hydesign.finance.finance_hifi_ems import finance_comp as finance


class hpp_model(hpp_base):
    """HPP design evaluator"""

    def __init__(self, sim_pars_fn, **kwargs):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        latitude : Latitude at chosen location
        longitude : Longitude at chosen location
        altitude : Altitude at chosen location, if not provided, elevation is calculated using elevation map datasets
        sims_pars_fn : Case study input values of the HPP
        work_dir : Working directory path
        max_num_batteries_allowed : Maximum number of batteries allowed including start and replacements
        weeks_per_season_per_year: Number of weeks per season to select from the input data, to reduce computation time. Default is `None` which uses all the input time series
        seed: seed number for week selection
        ems_type : Energy management system optimization type: cplex solver or rule based
        inputs_ts_fn : User provided weather timeseries, if not provided, the weather data is calculated using ERA5 datasets
        price_fn : Price timeseries
        era5_zarr : Location of wind speed renalysis
        ratio_gwa_era5 : Location of mean wind speed correction factor
        era5_ghi_zarr : Location of GHI renalysis
        elevation_fn : Location of GHI renalysis
        genWT_fn : Wind turbine power curve look-up tables
        genWake_fn : Wind turbine wake look-up tables
        """
        hpp_base.__init__(
            self,
            sim_pars_fn=sim_pars_fn,
            defaults={
                "input_ts_fn": os.path.join(
                    examples_filepath, "HiFiEMS_inputs/Weather/input_ts_DA.csv"
                ),
            },
            **kwargs,
        )

        N_time = self.N_time
        sim_pars = self.sim_pars

        data_dir = examples_filepath
        if "data_dir" in sim_pars:
            if sim_pars["data_dir"] is not None:
                data_dir = sim_pars["data_dir"]

        Wind_data = pd.read_csv(os.path.join(data_dir, sim_pars["wind_fn"]))
        Solar_data = pd.read_csv(os.path.join(data_dir, sim_pars["solar_fn"]))
        Market_data = pd.read_csv(os.path.join(data_dir, sim_pars["market_fn"]))

        battery_price_reduction_per_year = sim_pars["battery_price_reduction_per_year"]

        BM_model = sim_pars["BM_model"]
        RD_model = sim_pars["RD_model"]
        SMOpt = sim_pars["SMOpt"]
        BMOpt = sim_pars["BMOpt"]
        RDOpt = sim_pars["RDOpt"]

        parameter_keys = [
            "hpp_grid_connection",
            "wind_capacity",
            "solar_capacity",
            "battery_energy_capacity",
            "battery_power_capacity",
            "battery_minimum_SoC",
            "battery_maximum_SoC",
            "battery_initial_SoC",
            "battery_hour_discharge_efficiency",
            "battery_hour_charge_efficiency",
            "battery_self_discharge_efficiency",
            "battery_initial_degradation",
            "battery_marginal_degradation_cost",
            "battery_capital_cost",
            "degradation_in_optimization",
            "max_up_bid",
            "max_dw_bid",
            "min_up_bid",
            "min_dw_bid",
            "dispatch_interval",
            "settlement_interval",
            "offer_interval",
            "imbalance_fee",
            "deviation",
        ]
        simulation_keys = [
            "wind_as_component",
            "solar_as_component",
            "battery_as_component",
            "start_date",
            "number_of_run_day",
            "out_dir",
            "wind_df",
            "solar_df",
            "market_df",
            "DA_wind",
            "HA_wind",
            "FMA_wind",
            "DA_solar",
            "HA_solar",
            "FMA_solar",
            "SP",
            "RP",
            "BP",
            "history_wind_fn",
            "history_solar_fn",
            "history_market_fn",
            "N_Samples",
            "epsilon",
            "epsilon1",
            "wind_error_ub",
            "wind_error_lb",
            "Cp",
            "number_of_scenario",
            "probability",
        ]

        parameter_dict = {k: v for k, v in sim_pars.items() if k in parameter_keys}
        simulation_dict = {k: v for k, v in sim_pars.items() if k in simulation_keys}

        parameter_dict.update(
            {
                "battery_initial_degradation": 0,  # hpp battery degradation parameters
                "degradation_in_optimization": 0,
            }
        )

        simulation_dict.update(
            {
                "wind_df": Wind_data,
                "solar_df": Solar_data,
                "market_df": Market_data,
                "history_wind_fn": os.path.join(data_dir, sim_pars["history_wind_fn"]),
                "history_solar_fn": os.path.join(
                    data_dir, sim_pars["history_solar_fn"]
                ),
                "history_market_fn": os.path.join(
                    data_dir, sim_pars["history_market_fn"]
                ),
            }
        )

        comps = [
            (
                "ems",
                ems(
                    parameter_dict=parameter_dict,
                    simulation_dict=simulation_dict,
                    N_time=N_time,
                    SMOpt=SMOpt,
                    BMOpt=BMOpt,
                    RDOpt=RDOpt,
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
                    intervals_per_hour=4,
                ),
                {"wind_t": "wind_t_rt"},
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
                    intervals_per_hour=4,
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
                {
                    "Apvp": "Apvp_rt",
                },
            ),
            (
                "finance",
                finance(
                    parameter_dict=parameter_dict,
                    depreciation_yr=sim_pars["depreciation_yr"],
                    depreciation=sim_pars["depreciation"],
                    inflation_yr=sim_pars["inflation_yr"],
                    inflation=sim_pars["inflation"],
                    ref_yr_inflation=sim_pars["ref_yr_inflation"],
                    phasing_yr=sim_pars["phasing_yr"],
                    phasing_CAPEX=sim_pars["phasing_CAPEX"],
                ),
                {
                    "CAPEX_el": "CAPEX_sh",
                    "OPEX_el": "OPEX_sh",
                },
            ),
        ]

        prob = self.get_prob(comps)

        prob.setup()

        # Additional parameters
        prob.set_val("G_MW", sim_pars["G_MW"])
        prob.set_val(
            "battery_depth_of_discharge", sim_pars["battery_depth_of_discharge"]
        )
        prob.set_val("wind_WACC", sim_pars["wind_WACC"])
        prob.set_val("solar_WACC", sim_pars["solar_WACC"])
        prob.set_val("battery_WACC", sim_pars["battery_WACC"])
        prob.set_val("tax_rate", sim_pars["tax_rate"])

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
            "Mean Annual Electricity Sold [GWh]",
            "GUF",
            "grid [MW]",
            "wind [MW]",
            "solar [MW]",
            "Battery Energy [MWh]",
            "Battery Power [MW]",
            "Awpp [km2]",
            "Apvp [km2]",
            "Plant area [km2]",
            "Break-even PPA price [Euro/MWh]",
        ]

        self.list_vars = [
            "clearance [m]",
            "sp [W/m2]",
            "p_rated [MW]",
            "Nwt",
            "wind_MW_per_km2 [MW/km2]",
            "wind_MW [MW]",
            "solar_MW [MW]",
            "b_P [MW]",
            "b_E_h [h]",
        ]

    def evaluate(
        self,
        wind_MW,
        solar_MW,
        b_P,
        b_E_h,
        **kwargs,
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

        prob = self.prob

        # assumed values:
        wind_MW_per_km2 = 6  # [MW/km2]

        Awpp = wind_MW / wind_MW_per_km2
        b_E = b_E_h * b_P

        # pass design variables
        p_rated_ref = self.sim_pars["p_rated_ref"]
        Nwt = int(wind_MW / p_rated_ref)
        p_rated = wind_MW / Nwt
        d = self.sim_pars["d_ref"]
        hh = self.sim_pars["hh_ref"]
        clearance = hh - d / 2
        sp = p_rated * 1e6 / (np.pi * (d / 2) ** 2)  # [W/m2]

        prob.set_val("hh", hh)
        prob.set_val("d", d)
        prob.set_val("Nwt", Nwt)
        prob.set_val("p_rated", p_rated)
        prob.set_val("Awpp", Awpp)

        prob.set_val("solar_MW", solar_MW)

        prob.set_val("b_P", b_P)
        prob.set_val("b_E", b_E)
        prob.set_val("wind_MW", wind_MW)

        self.inputs = [
            clearance,
            sp,
            p_rated,
            Nwt,
            wind_MW_per_km2,
            wind_MW,
            solar_MW,
            b_P,
            b_E_h,
        ]

        prob.run_model()

        self.prob = prob

        outputs = np.hstack(
            [
                prob["NPV_over_CAPEX"],
                prob["NPV"] / 1e6,
                prob["IRR"],
                prob["LCOE"],
                prob["revenues"] / 1e6,
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
                prob["mean_AEP"] / 1e3,  # [GWh]
                prob["mean_AEP"] / (self.sim_pars["G_MW"] * 365 * 24),
                self.sim_pars["G_MW"],
                wind_MW,
                solar_MW,
                b_E,
                b_P,
                Awpp,
                prob.get_val("Apvp_rt"),
                max(Awpp, prob.get_val("Apvp_rt")),
                prob["break_even_PPA_price"],
            ]
        )
        self.outputs = outputs
        return outputs


if __name__ == "__main__":
    sim_pars_fn = os.path.join(examples_filepath, "Europe/hpp_pars_HiFiEMS.yml")
    hpp = hpp_model(
        sim_pars_fn=sim_pars_fn,
    )
    inputs = dict(
        wind_MW=120,
        solar_MW=10,
        b_P=40,
        b_E_h=3,
    )

    res = hpp.evaluate(**inputs)
    hpp.print_design()
