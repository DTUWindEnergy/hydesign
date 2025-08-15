import numpy as np
import openmdao.api as om

# Wisdem
from hydesign.nrel_csm_wrapper import wt_cost
from hydesign.openmdao_wrapper import ComponentWrapper


class wpp_cost:
    """Pure Python Wind power plant cost model is used to assess the overall wind plant cost. It is based on the The NREL Cost and Scaling model [1].
    It estimates the total capital expenditure costs and operational and maintenance costs, as a function of the installed capacity, the cost of the
    turbine, intallation costs and O&M costs.
    [1] Dykes, K., et al. 2014. Sensitivity analysis of wind plant performance to key turbine design parameters: a systems engineering approach. Tech. rep. National Renewable Energy Laboratory
    """

    def __init__(
        self,
        wind_turbine_cost,
        wind_civil_works_cost,
        wind_fixed_onm_cost,
        wind_variable_onm_cost,
        d_ref,
        hh_ref,
        p_rated_ref,
        N_time,
        intervals_per_hour=1,
    ):
        """Initialization of the wind power plant cost model

        Parameters
        ----------
        wind_turbine_cost : Wind turbine cost [Euro/MW]
        wind_civil_works_cost : Wind civil works cost [Euro/MW]
        wind_fixed_onm_cost : Wind fixed onm (operation and maintenance) cost [Euro/MW/year]
        wind_variable_onm_cost : Wind variable onm cost [EUR/MWh_e]
        d_ref : Reference diameter of the cost model [m]
        hh_ref : Reference hub height of the cost model [m]
        p_rated_ref : Reference turbine power of the cost model [MW]
        N_time : Length of the representative data

        """
        self.wind_turbine_cost = wind_turbine_cost
        self.wind_civil_works_cost = wind_civil_works_cost
        self.wind_fixed_onm_cost = wind_fixed_onm_cost
        self.wind_variable_onm_cost = wind_variable_onm_cost
        self.d_ref = d_ref
        self.hh_ref = hh_ref
        self.p_rated_ref = p_rated_ref
        self.N_time = N_time
        self.intervals_per_hour = intervals_per_hour

    def compute(self, Nwt, hh, d, p_rated, wind_t, **kwargs):
        """Computing the CAPEX and OPEX of the wind power plant.

        Parameters
        ----------
        Nwt : Number of wind turbines
        Awpp : Land use area of WPP [km**2]
        hh : Turbine's hub height [m]
        d : Turbine's diameter [m]
        p_rated : Turbine's rated power [MW]
        wind_t : WPP power time series [MW]

        Returns
        -------
        CAPEX_w : CAPEX of the wind power plant [Eur]
        OPEX_w : OPEX of the wind power plant [Eur/year]
        """
        wind_turbine_cost = self.wind_turbine_cost
        wind_civil_works_cost = self.wind_civil_works_cost
        wind_fixed_onm_cost = self.wind_fixed_onm_cost
        wind_variable_onm_cost = self.wind_variable_onm_cost

        d_ref = self.d_ref
        hh_ref = self.hh_ref
        p_rated_ref = self.p_rated_ref

        WT_cost_ref = (
            wt_cost(
                rotor_diameter=d_ref,
                turbine_class=1,
                blade_has_carbon=False,
                blade_number=3,
                machine_rating=p_rated_ref * 1e3,  # kW
                hub_height=hh_ref,
                bearing_number=2,
                crane=True,
            )
            * 1e-6
        )

        WT_cost = (
            wt_cost(
                rotor_diameter=d,
                turbine_class=1,
                blade_has_carbon=False,
                blade_number=3,
                machine_rating=p_rated * 1e3,  # kW
                hub_height=hh,
                bearing_number=2,
                crane=True,
            )
            * 1e-6
        )
        scale = (WT_cost / p_rated) / (WT_cost_ref / p_rated_ref)
        mean_aep_wind = wind_t.mean() * 365 * 24 * self.intervals_per_hour

        CAPEX_w = scale * (wind_turbine_cost + wind_civil_works_cost) * (Nwt * p_rated)
        OPEX_w = (
            wind_fixed_onm_cost * (Nwt * p_rated)
            + mean_aep_wind * wind_variable_onm_cost * p_rated / p_rated_ref
        )
        return CAPEX_w, OPEX_w


class wpp_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = wpp_cost(**insta_inp)
        super().__init__(
            inputs=[
                ("Nwt", {"desc": "Number of wind turbines"}),
                ("hh", {"units": "m", "desc": "Turbine's hub height"}),
                ("d", {"units": "m", "desc": "Turbine's diameter"}),
                ("p_rated", {"units": "MW", "desc": "Turbine's rated power"}),
                (
                    "wind_t",
                    {
                        "units": "MW",
                        "desc": "WPP power time series",
                        "shape": [model.N_time],
                    },
                ),
            ],
            outputs=[
                ("CAPEX_w", {"desc": "CAPEX of the wind power plant"}),
                ("OPEX_w", {"desc": "OPEX of the wind power plant"}),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class pvp_cost:
    """Pure Python PV plant cost model is used to calculate the overall PV plant cost. The cost model estimates the total solar capital expenditure costs
    and  operational and maintenance costs as a function of the installed solar capacity and the PV cost per MW installation costs (extracted from the danish energy agency data catalogue).
    """

    def __init__(
        self,
        solar_PV_cost,
        solar_hardware_installation_cost,
        solar_inverter_cost,
        solar_fixed_onm_cost,
    ):
        """Initialization of the PV power plant cost model

        Parameters
        ----------
        solar_PV_cost : PV panels cost [Euro/MW]
        solar_hardware_installation_cost : Solar panels civil works cost [Euro/MW]
        solar_fixed_onm_cost : Solar fixed onm (operation and maintenance) cost [Euro/MW/year]

        """
        self.solar_PV_cost = solar_PV_cost
        self.solar_hardware_installation_cost = solar_hardware_installation_cost
        self.solar_inverter_cost = solar_inverter_cost
        self.solar_fixed_onm_cost = solar_fixed_onm_cost

    def compute(self, solar_MW, DC_AC_ratio, **kwargs):
        """Computing the CAPEX and OPEX of the solar power plant.

        Parameters
        ----------
        solar_MW : AC nominal capacity of the PV plant [MW]
        DC_AC_ratio: Ratio of DC power rating with respect AC rating of the PV plant

        Returns
        -------
        CAPEX_s : CAPEX of the PV power plant [Eur]
        OPEX_s : OPEX of the PV power plant [Eur/year]
        """

        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_inverter_cost = self.solar_inverter_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost
        CAPEX_s = (
            solar_PV_cost + solar_hardware_installation_cost
        ) * solar_MW * DC_AC_ratio + solar_inverter_cost * solar_MW
        OPEX_s = solar_fixed_onm_cost * solar_MW * DC_AC_ratio
        return CAPEX_s, OPEX_s

    def compute_partials(self, solar_MW, DC_AC_ratio):
        DC_AC_ratio_tech_ref = 1.25
        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_inverter_cost = self.solar_inverter_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost

        d_CAPEX_s_d_solar_MW = (
            solar_PV_cost + solar_hardware_installation_cost
        ) * DC_AC_ratio + solar_inverter_cost * DC_AC_ratio_tech_ref / DC_AC_ratio
        d_CAPEX_s_d_DC_AC_ratio = (
            solar_PV_cost
            + solar_hardware_installation_cost
            + solar_inverter_cost * DC_AC_ratio_tech_ref / (-(DC_AC_ratio**2))
        ) * solar_MW
        d_OPEX_s_d_solar_MW = solar_fixed_onm_cost * DC_AC_ratio
        d_OPEX_s_d_DC_AC_ratio = solar_fixed_onm_cost * solar_MW
        return (
            d_CAPEX_s_d_solar_MW,
            d_CAPEX_s_d_DC_AC_ratio,
            d_OPEX_s_d_solar_MW,
            d_OPEX_s_d_DC_AC_ratio,
        )


class pvp_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = pvp_cost(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "solar_MW",
                    {"units": "MW", "desc": "AC nominal capacity of the PV plant"},
                ),
                (
                    "DC_AC_ratio",
                    {
                        "desc": "Ratio of DC power rating with respect AC rating of the PV plant"
                    },
                ),
            ],
            outputs=[
                ("CAPEX_s", {"desc": "CAPEX of the PV power plant"}),
                ("OPEX_s", {"desc": "OPEX of the PV power plant"}),
            ],
            function=model.compute,
            # partial_function=model.compute_partials,
            partial_options=[{"dependent": False, "val": 0}],
        )


class battery_cost:
    """Pure Python Battery cost model calculates the storage unit costs. It uses technology costs extracted from the danish energy agency data catalogue."""

    def __init__(
        self,
        battery_energy_cost,
        battery_power_cost,
        battery_BOP_installation_commissioning_cost,
        battery_control_system_cost,
        battery_energy_onm_cost,
        life_y=25,
        intervals_per_hour=1,
        battery_price_reduction_per_year=0.1,
    ):
        """Initialization of the battery cost model

        Parameters
        ----------
        battery_energy_cost : Battery energy cost [Euro/MWh]
        battery_power_cost : Battery power cost [Euro/MW]
        battery_BOP_installation_commissioning_cost : Battery installation and commissioning cost [Euro/MW]
        battery_control_system_cost : Battery system controt cost [Euro/MW]
        battery_energy_onm_cost : Battery operation and maintenance cost [Euro/MW]
        num_batteries : Number of battery replacement in the lifetime of the plant
        N_life : Lifetime of the plant in years
        life_h : Total number of hours in the lifetime of the plant


        """
        self.battery_energy_cost = battery_energy_cost
        self.battery_power_cost = battery_power_cost
        self.battery_BOP_installation_commissioning_cost = (
            battery_BOP_installation_commissioning_cost
        )
        self.battery_control_system_cost = battery_control_system_cost
        self.battery_energy_onm_cost = battery_energy_onm_cost
        self.life_h = life_y * 365 * 24
        self.yearly_intervals = 365 * 24 * intervals_per_hour
        self.life_intervals = life_y * self.yearly_intervals
        self.battery_price_reduction_per_year = battery_price_reduction_per_year

    def compute(self, b_E, b_P, SoH, **kwargs):
        """Computing the CAPEX and OPEX of battery.

        Parameters
        ----------
        b_P : Battery power capacity [MW]
        b_E : Battery energy storage capacity [MWh]
        ii_time : Indices on the lifetime time series (Hydesign operates in each range at constant battery health)
        SoH : Battery state of health at discretization levels

        Returns
        -------
        CAPEX_b : CAPEX of the storage unit [Eur]
        OPEX_b : OPEX of the storage unit [Eur/year]
        """

        life_intervals = self.life_intervals
        age = np.arange(life_intervals) / self.yearly_intervals

        battery_price_reduction_per_year = self.battery_price_reduction_per_year

        battery_energy_cost = self.battery_energy_cost
        battery_power_cost = self.battery_power_cost
        battery_BOP_installation_commissioning_cost = (
            self.battery_BOP_installation_commissioning_cost
        )
        battery_control_system_cost = self.battery_control_system_cost
        battery_energy_onm_cost = self.battery_energy_onm_cost

        ii_battery_change = np.where((SoH > 0.99) & (np.append(1, np.diff(SoH)) > 0))[0]
        year_new_battery = np.unique(np.floor(age[ii_battery_change]))

        factor = 1.0 - battery_price_reduction_per_year
        N_beq = np.sum([factor**iy for iy in year_new_battery])

        CAPEX_b = (
            N_beq * (battery_energy_cost * b_E)
            + (
                battery_power_cost
                + battery_BOP_installation_commissioning_cost
                + battery_control_system_cost
            )
            * b_P
        )

        OPEX_b = battery_energy_onm_cost * b_E

        CAPEX_b = CAPEX_b
        OPEX_b = OPEX_b
        return CAPEX_b, OPEX_b


class battery_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = battery_cost(**insta_inp)
        super().__init__(
            inputs=[
                ("b_P", {"units": "MW", "desc": "Battery power capacity"}),
                ("b_E", {"desc": "Battery energy storage capacity"}),
                (
                    "SoH",
                    {
                        "desc": "Battery state of health at discretization levels",
                        "shape": [model.life_intervals],
                    },
                ),
            ],
            outputs=[
                ("CAPEX_b", {"desc": "CAPEX of the storage unit"}),
                ("OPEX_b", {"desc": "OPEX of the storage unit"}),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class shared_cost:
    """Pure Python Electrical infrastructure and land rent cost model"""

    def __init__(self, hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost):
        """Initialization of the shared costs model

        Parameters
        ----------
        hpp_BOS_soft_cost : Balancing of system cost [Euro/MW]
        hpp_grid_connection_cost : Grid connection cost [Euro/MW]
        land_cost : Land rent cost [Euro/km**2]
        """
        self.hpp_BOS_soft_cost = hpp_BOS_soft_cost
        self.hpp_grid_connection_cost = hpp_grid_connection_cost
        self.land_cost = land_cost

    def compute(self, G_MW, Awpp, Apvp, **kwargs):
        """Computing the CAPEX and OPEX of the shared land and infrastructure.

        Parameters
        ----------
        G_MW : Grid capacity [MW]
        Awpp : Land use area of the wind power plant [km**2]
        Apvp : Land use area of the solar power plant [km**2]

        Returns
        -------
        CAPEX_sh : CAPEX electrical infrastructure/ land rent [Eur]
        OPEX_sh : OPEX electrical infrastructure/ land rent [Eur/year]
        """
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        if Awpp >= Apvp:
            land_rent = land_cost * Awpp
        else:
            land_rent = land_cost * Apvp
        CAPEX_sh = (hpp_BOS_soft_cost + hpp_grid_connection_cost) * G_MW + land_rent
        OPEX_sh = 0
        return CAPEX_sh, OPEX_sh

    def compute_partials(self, Awpp, Apvp):
        partials = {}
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        partials["CAPEX_sh", "G_MW"] = hpp_BOS_soft_cost + hpp_grid_connection_cost
        if Awpp >= Apvp:
            partials["CAPEX_sh", "Awpp"] = land_cost
            partials["CAPEX_sh", "Apvp"] = 0
        else:
            partials["CAPEX_sh", "Awpp"] = 0
            partials["CAPEX_sh", "Apvp"] = land_cost
        partials["OPEX_sh", "G_MW"] = 0
        partials["OPEX_sh", "Awpp"] = 0
        partials["OPEX_sh", "Apvp"] = 0
        return partials


class shared_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = shared_cost(**insta_inp)
        super().__init__(
            inputs=[
                ("G_MW", {"units": "MW", "desc": "Grid capacity"}),
                ("Awpp", {"units": "km**2", "desc": "Land use area of WPP"}),
                ("Apvp", {"units": "km**2", "desc": "Land use area of SP"}),
            ],
            outputs=[
                ("CAPEX_sh", {"desc": "CAPEX electrical infrastructure/ land rent"}),
                ("OPEX_sh", {"desc": "OPEX electrical infrastructure/ land rent"}),
            ],
            function=model.compute,
            partial_function=model.compute_partials,
            partial_options=[{"dependent": False, "val": 0}],
        )


class ptg_cost:
    """Pure Python Power to H2 plant cost model is used to calculate the overall H2 plant cost. The cost model includes cost of electrolyzer
    and compressor for storing Hydrogen (data extracted from the danish energy agency data catalogue and IRENA reports).
    """

    def __init__(
        self,
        electrolyzer_capex_cost,
        electrolyzer_opex_cost,
        electrolyzer_power_electronics_cost,
        water_cost,
        water_treatment_cost,
        water_consumption,
        storage_capex_cost,
        storage_opex_cost,
        transportation_cost,
        transportation_distance,
        N_time,
        life_y=25,
        intervals_per_hour=1,
    ):

        self.electrolyzer_capex_cost = electrolyzer_capex_cost
        self.electrolyzer_opex_cost = electrolyzer_opex_cost
        self.electrolyzer_power_electronics_cost = electrolyzer_power_electronics_cost
        self.water_cost = water_cost
        self.water_treatment_cost = water_treatment_cost
        self.water_consumption = water_consumption
        self.storage_capex_cost = storage_capex_cost
        self.storage_opex_cost = storage_opex_cost
        self.transportation_cost = transportation_cost
        self.transportation_distance = transportation_distance
        self.N_time = N_time
        self.life_h = 365 * 24 * life_y * intervals_per_hour
        self.yearly_intervals = 365 * 24 * intervals_per_hour
        self.life_intervals = self.yearly_intervals * life_y

    def compute(self, ptg_MW, HSS_kg, m_H2_offtake_t):

        electrolyzer_capex_cost = self.electrolyzer_capex_cost
        electrolyzer_opex_cost = self.electrolyzer_opex_cost
        electrolyzer_power_electronics_cost = self.electrolyzer_power_electronics_cost
        water_cost = self.water_cost
        water_treatment_cost = self.water_treatment_cost
        water_consumption = self.water_consumption
        storage_capex_cost = self.storage_capex_cost
        storage_opex_cost = self.storage_opex_cost
        transportation_cost = self.transportation_cost
        transportation_distance = self.transportation_distance

        CAPEX_ptg = (
            ptg_MW * (electrolyzer_capex_cost + electrolyzer_power_electronics_cost)
            + storage_capex_cost * HSS_kg
            + (
                m_H2_offtake_t.mean()
                * self.yearly_intervals
                * transportation_cost
                * transportation_distance
            )
        )
        OPEX_ptg = ptg_MW * (electrolyzer_opex_cost) + storage_opex_cost * HSS_kg
        water_consumption_cost = (
            m_H2_offtake_t.mean()
            * self.yearly_intervals
            * water_consumption
            * (water_cost + water_treatment_cost)
            / 1000
        )  # annual mean water consumption to produce hydrogen over an year
        return CAPEX_ptg, OPEX_ptg, water_consumption_cost


class ptg_cost_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = ptg_cost(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "ptg_MW",
                    {
                        "units": "MW",
                        "desc": "Installed capacity for the power to gas plant",
                    },
                ),
                (
                    "HSS_kg",
                    {"units": "kg", "desc": "Installed capacity of Hydrogen storage"},
                ),
                (
                    "m_H2_offtake_t",
                    {
                        "units": "kg",
                        "desc": "Offtake hydrogen",
                        "shape": [model.life_intervals],
                    },
                ),
            ],
            outputs=[
                ("CAPEX_ptg", {"desc": "CAPEX of the power to gas plant"}),
                ("OPEX_ptg", {"desc": "OPEX of the power to gas plant"}),
                (
                    "water_consumption_cost",
                    {"desc": "Annual water consumption and treatment cost"},
                ),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
