# %%
import glob
import os
import time

# basic libraries
import numpy as np
import numpy_financial as npf

# import seaborn as sns
import openmdao.api as om
import pandas as pd
import scipy as sp
import xarray as xr
import yaml
from numpy import newaxis as na
from scipy import stats
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter

# Wisdem
from hydesign.nrel_csm_wrapper import wt_cost


class wpp_cost(om.ExplicitComponent):
    """Wind power plant cost model is used to assess the overall wind plant cost. It is based on the The NREL Cost and Scaling model [1].
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
        super().__init__()
        self.wind_turbine_cost = wind_turbine_cost
        self.wind_civil_works_cost = wind_civil_works_cost
        self.wind_fixed_onm_cost = wind_fixed_onm_cost
        self.wind_variable_onm_cost = wind_variable_onm_cost
        self.d_ref = d_ref
        self.hh_ref = hh_ref
        self.p_rated_ref = p_rated_ref
        self.N_time = N_time

    def setup(self):
        # self.add_discrete_input(
        self.add_input("Nwt", desc="Number of wind turbines")
        self.add_input("Awpp", desc="Land use area of WPP", units="km**2")

        self.add_input("hh", desc="Turbine's hub height", units="m")
        self.add_input("d", desc="Turbine's diameter", units="m")
        self.add_input("p_rated", desc="Turbine's rated power", units="MW")
        self.add_input(
            "wind_t", desc="WPP power time series", units="MW", shape=[self.N_time]
        )

        self.add_output("CAPEX_w", desc="CAPEX wpp")
        self.add_output("OPEX_w", desc="OPEX wpp")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):  # , discrete_inputs, discrete_outputs):
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

        # Nwt = discrete_inputs['Nwt']
        Nwt = inputs["Nwt"]
        # Awpp = inputs['Awpp']
        hh = inputs["hh"]
        d = inputs["d"]
        p_rated = inputs["p_rated"]
        wind_t = inputs["wind_t"]
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
        mean_aep_wind = wind_t.mean() * 365 * 24

        outputs["CAPEX_w"] = (
            scale * (wind_turbine_cost + wind_civil_works_cost) * (Nwt * p_rated)
        )
        outputs["OPEX_w"] = (
            wind_fixed_onm_cost * (Nwt * p_rated)
            + mean_aep_wind * wind_variable_onm_cost * p_rated / p_rated_ref
        )


class pvp_cost(om.ExplicitComponent):
    """PV plant cost model is used to calculate the overall PV plant cost. The cost model estimates the total solar capital expenditure costs
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
        super().__init__()
        self.solar_PV_cost = solar_PV_cost
        self.solar_hardware_installation_cost = solar_hardware_installation_cost
        self.solar_inverter_cost = solar_inverter_cost
        self.solar_fixed_onm_cost = solar_fixed_onm_cost
        # self.setup()

    def setup(self):
        self.add_input("solar_MW", desc="Solar PV plant installed capacity", units="MW")
        self.add_input("DC_AC_ratio", desc="DC/AC PV ratio")

        self.add_output("CAPEX_s", desc="CAPEX solar pvp")
        self.add_output("OPEX_s", desc="OPEX solar pvp")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
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

        solar_MW = inputs["solar_MW"]
        DC_AC_ratio = inputs["DC_AC_ratio"]
        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_inverter_cost = self.solar_inverter_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost

        outputs["CAPEX_s"] = (
            solar_PV_cost + solar_hardware_installation_cost
        ) * solar_MW * DC_AC_ratio + solar_inverter_cost * solar_MW
        outputs["OPEX_s"] = solar_fixed_onm_cost * solar_MW * DC_AC_ratio

    def compute_partials(self, inputs, partials):
        solar_MW = inputs["solar_MW"]
        DC_AC_ratio = inputs["DC_AC_ratio"]
        DC_AC_ratio_tech_ref = 1.25
        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_inverter_cost = self.solar_inverter_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost

        partials["CAPEX_s", "solar_MW"] = (
            solar_PV_cost + solar_hardware_installation_cost
        ) * DC_AC_ratio + solar_inverter_cost * DC_AC_ratio_tech_ref / DC_AC_ratio
        partials["CAPEX_s", "DC_AC_ratio"] = (
            solar_PV_cost
            + solar_hardware_installation_cost
            + solar_inverter_cost * DC_AC_ratio_tech_ref / (-(DC_AC_ratio**2))
        ) * solar_MW
        partials["OPEX_s", "solar_MW"] = solar_fixed_onm_cost * DC_AC_ratio
        partials["OPEX_s", "DC_AC_ratio"] = solar_fixed_onm_cost * solar_MW


class battery_cost(om.ExplicitComponent):
    """Battery cost model calculates the storage unit costs. It uses technology costs extracted from the danish energy agency data catalogue."""

    def __init__(
        self,
        battery_energy_cost,
        battery_power_cost,
        battery_BOP_installation_commissioning_cost,
        battery_control_system_cost,
        battery_energy_onm_cost,
        life_y=25,
        life_h=25 * 365 * 24,
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
        life_y : Lifetime of the plant in years
        life_h : Total number of hours in the lifetime of the plant


        """
        super().__init__()
        self.battery_energy_cost = battery_energy_cost
        self.battery_power_cost = battery_power_cost
        self.battery_BOP_installation_commissioning_cost = (
            battery_BOP_installation_commissioning_cost
        )
        self.battery_control_system_cost = battery_control_system_cost
        self.battery_energy_onm_cost = battery_energy_onm_cost
        self.life_y = life_y
        self.life_h = life_h

    def setup(self):
        self.add_input("P_batt_MW", desc="Battery power capacity", units="MW")
        self.add_input("E_batt_MWh_t", desc="Battery energy storage capacity")
        self.add_input(
            "SoH",
            desc="Battery state of health at discretization levels",
            shape=[self.life_h],
        )
        self.add_input(
            "battery_price_reduction_per_year",
            desc="Factor of battery price reduction per year",
        )

        self.add_output("CAPEX_b", desc="CAPEX battery")
        self.add_output("OPEX_b", desc="OPEX battery")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Computing the CAPEX and OPEX of battery.

        Parameters
        ----------
        P_batt_MW : Battery power capacity [MW]
        E_batt_MWh_t : Battery energy storage capacity [MWh]
        ii_time : Indices on the lifetime time series (Hydesign operates in each range at constant battery health)
        SoH : Battery state of health at discretization levels
        battery_price_reduction_per_year : Factor of battery price reduction per year

        Returns
        -------
        CAPEX_b : CAPEX of the storage unit [Eur]
        OPEX_b : OPEX of the storage unit [Eur/year]
        """

        life_y = self.life_y
        life_h = self.life_h
        age = np.arange(life_h) / (24 * 365)

        E_batt_MWh_t = inputs["E_batt_MWh_t"]
        P_batt_MW = inputs["P_batt_MW"]
        SoH = inputs["SoH"]
        battery_price_reduction_per_year = inputs["battery_price_reduction_per_year"]

        battery_energy_cost = self.battery_energy_cost
        battery_power_cost = self.battery_power_cost
        battery_BOP_installation_commissioning_cost = (
            self.battery_BOP_installation_commissioning_cost
        )
        battery_control_system_cost = self.battery_control_system_cost
        battery_energy_onm_cost = self.battery_energy_onm_cost

        ii_battery_change = np.where((SoH > 0.99) & (np.append(1, np.diff(SoH)) > 0))[0]
        year_new_battery = np.unique(np.floor(age[ii_battery_change]))

        battery_price_reduction_per_year = 0.1
        factor = 1.0 - battery_price_reduction_per_year
        N_beq = np.sum([factor**iy for iy in year_new_battery])

        CAPEX_b = (
            N_beq
            * (
                battery_energy_cost
                + battery_BOP_installation_commissioning_cost
                + battery_control_system_cost
            )
            * E_batt_MWh_t
            + battery_power_cost * P_batt_MW
        )

        OPEX_b = battery_energy_onm_cost * P_batt_MW

        outputs["CAPEX_b"] = CAPEX_b
        outputs["OPEX_b"] = OPEX_b


class shared_cost(om.ExplicitComponent):
    """Electrical infrastructure and land rent cost model"""

    def __init__(self, hpp_BOS_soft_cost, hpp_grid_connection_cost, land_cost):
        """Initialization of the shared costs model

        Parameters
        ----------
        hpp_BOS_soft_cost : Balancing of system cost [Euro/MW]
        hpp_grid_connection_cost : Grid connection cost [Euro/MW]
        land_cost : Land rent cost [Euro/km**2]
        """
        super().__init__()
        self.hpp_BOS_soft_cost = hpp_BOS_soft_cost
        self.hpp_grid_connection_cost = hpp_grid_connection_cost
        self.land_cost = land_cost

    def setup(self):
        self.add_input("hpp_grid_connection", desc="Grid capacity", units="MW")
        self.add_input("Awpp", desc="Land use area of WPP", units="km**2")
        self.add_input("Apvp", desc="Land use area of SP", units="km**2")

        self.add_output("CAPEX_el", desc="CAPEX electrical infrastructure/ land rent")
        self.add_output("OPEX_el", desc="OPEX electrical infrastructure/ land rent")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Computing the CAPEX and OPEX of the shared land and infrastructure.

        Parameters
        ----------
        hpp_grid_connection : Grid capacity [MW]
        Awpp : Land use area of the wind power plant [km**2]
        Apvp : Land use area of the solar power plant [km**2]

        Returns
        -------
        CAPEX_el : CAPEX electrical infrastructure/ land rent [Eur]
        OPEX_el : OPEX electrical infrastructure/ land rent [Eur/year]
        """
        hpp_grid_connection = inputs["hpp_grid_connection"]
        Awpp = inputs["Awpp"]
        Apvp = inputs["Apvp"]
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        if Awpp >= Apvp:
            land_rent = land_cost * Awpp
        else:
            land_rent = land_cost * Apvp

        # print(land_rent)
        outputs["CAPEX_el"] = (
            hpp_BOS_soft_cost + hpp_grid_connection_cost
        ) * hpp_grid_connection + land_rent
        outputs["OPEX_el"] = 0

    def compute_partials(self, inputs, partials):
        hpp_grid_connection = inputs["hpp_grid_connection"]
        Awpp = inputs["Awpp"]
        Apvp = inputs["Apvp"]
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        partials["CAPEX_el", "hpp_grid_connection"] = (
            hpp_BOS_soft_cost + hpp_grid_connection_cost
        )
        if Awpp >= Apvp:
            partials["CAPEX_el", "Apvp"] = 0
        else:
            partials["CAPEX_el", "Awpp"] = 0
            partials["CAPEX_el", "Apvp"] = land_cost
        partials["OPEX_el", "hpp_grid_connection"] = 0
        partials["OPEX_el", "Awpp"] = 0
        partials["OPEX_el", "Apvp"] = 0


class P2MeOH_cost(om.ExplicitComponent):
    """Power to MeOH plant cost model is used to calculate the overall MeOH plant cost. The cost model includes cost of electrolyzer,
    heater, DAC, MeOH reactor, and MeOH tank to store green MeOH (data extracted from the danish energy agency data catalogue, IRENA reports, and other scientific articles)
    """

    def __init__(
        self,
        SOEC_capex_cost,
        SOEC_opex_cost,
        SOEC_power_electronics_cost,
        water_cost,
        water_treatment_cost,
        water_consumption,
        heater_capex_cost,
        heater_opex_cost,
        DAC_capex_cost,
        DAC_opex_cost,
        MeOH_reactor_capex_cost,
        MeOH_reactor_opex_cost,
        MeOH_tank_capex_cost,
        MeOH_tank_opex_cost,
        N_time,
        MeOH_demand_scheme,
        life_h=25 * 365 * 24,
    ):

        super().__init__()
        self.SOEC_capex_cost = SOEC_capex_cost
        self.SOEC_opex_cost = SOEC_opex_cost
        self.SOEC_power_electronics_cost = SOEC_power_electronics_cost
        self.water_cost = water_cost
        self.water_treatment_cost = water_treatment_cost
        self.water_consumption = water_consumption
        self.heater_capex_cost = heater_capex_cost
        self.heater_opex_cost = heater_opex_cost
        self.DAC_capex_cost = DAC_capex_cost
        self.DAC_opex_cost = DAC_opex_cost
        self.MeOH_reactor_capex_cost = MeOH_reactor_capex_cost
        self.MeOH_reactor_opex_cost = MeOH_reactor_opex_cost
        self.MeOH_tank_capex_cost = MeOH_tank_capex_cost
        self.MeOH_tank_opex_cost = MeOH_tank_opex_cost
        # self.transportation_cost = transportation_cost
        # self.transportation_distance = transportation_distance
        self.N_time = N_time
        self.MeOH_demand_scheme = MeOH_demand_scheme
        self.life_h = life_h

    def setup(self):

        self.add_input("P_SOEC_MW", desc="Installed capacity for the SOEC", units="MW")

        # self.add_input('P_heater_MW',
        #                desc = "Installed capacity for the heater",
        #                units = 'MW')

        # self.add_input('P_DAC_MW',
        #                desc = "Installed power capacity for the DAC",
        #                units = 'MW')

        # self.add_input('P_reactor_MW',
        #                desc = "Installed capacity for the MeOH reactor",
        #                units = 'MW')

        self.add_input(
            "m_MeOH_tank_max_kg",
            desc="Installed capacity of Methanol storage",
            units="kg",
        )

        self.add_input(
            "m_H2O_t",
            desc="Water produced from MeOH synthesis",
            units="kg",
            shape=[self.life_h],
        )

        self.add_input("psi_DAC_MWhkg", desc="DAC power consumption", units="MW/kg")

        self.add_input("M_CO2", desc="Molar mass of CO2")

        self.add_input("M_H2", desc="Molar mass of H2")

        self.add_input("lhv", desc="Low heat value.")

        self.add_input(
            "w_comp_reactor_MWhkg", desc="Reactor power consumption", units="MW/kg"
        )

        self.add_input("MeOH_yield", desc="MeOH production yield")

        self.add_input("M_CH3OH", desc="CH3OH molar mass")

        if self.MeOH_demand_scheme == "fixed":

            self.add_input(
                "m_MeOH_demand_t_ext",
                desc="Methanol demand",
                units="kg",
                shape=[self.life_h],
            )

        if self.MeOH_demand_scheme == "infinite":

            self.add_input(
                "m_green_MeOH_dist_t",
                desc="Green MeOH distributed",
                units="kg",
                shape=[self.life_h],
            )

        # Creating outputs:
        self.add_output("CAPEX_P2MeOH", desc="CAPEX P2MeOH")
        self.add_output("OPEX_SOEC", desc="OPEX SOEC")
        self.add_output(
            "water_consumption_cost",
            desc="Annual water consumption and treatment cost",
        )
        # self.add_output('OPEX_heater',
        #                 desc = "OPEX heater")
        self.add_output("OPEX_DAC", desc="OPEX DAC")
        self.add_output("OPEX_reactor", desc="OPEX reactor")
        self.add_output("OPEX_MeOH_tank", desc="OPEX MeOH_tank")

    def compute(self, inputs, outputs):

        P_SOEC_MW = inputs["P_SOEC_MW"]
        # P_heater_MW = inputs['P_heater_MW']
        # P_DAC_MW = inputs['P_DAC_MW']
        # P_reactor_MW = inputs['P_reactor_MW']
        m_MeOH_tank_max_kg = inputs["m_MeOH_tank_max_kg"]
        m_H2O_t = inputs["m_H2O_t"]
        psi_DAC_MWhkg = inputs["psi_DAC_MWhkg"]
        M_CO2 = inputs["M_CO2"]
        eff_cor = 0.69
        M_H2 = inputs["M_H2"]
        lhv = inputs["lhv"]
        w_comp_reactor_MWhkg = inputs["w_comp_reactor_MWhkg"]
        MeOH_yield = inputs["MeOH_yield"]
        M_CH3OH = inputs["M_CH3OH"]

        if self.MeOH_demand_scheme == "fixed":

            m_MeOH_demand_t_ext = inputs["m_MeOH_demand_t_ext"]

        if self.MeOH_demand_scheme == "infinite":

            m_green_MeOH_dist_t = inputs["m_green_MeOH_dist_t"]

        SOEC_capex_cost = self.SOEC_capex_cost
        SOEC_opex_cost = self.SOEC_opex_cost
        SOEC_power_electronics_cost = self.SOEC_power_electronics_cost
        water_cost = self.water_cost
        water_treatment_cost = self.water_treatment_cost
        water_consumption = self.water_consumption
        heater_capex_cost = self.heater_capex_cost
        heater_opex_cost = self.heater_opex_cost
        DAC_capex_cost = self.DAC_capex_cost
        DAC_opex_cost = self.DAC_opex_cost
        MeOH_reactor_capex_cost = self.MeOH_reactor_capex_cost
        MeOH_reactor_opex_cost = self.MeOH_reactor_opex_cost
        MeOH_tank_capex_cost = self.MeOH_tank_capex_cost
        MeOH_tank_opex_cost = self.MeOH_tank_opex_cost
        # transportation_cost = self.transportation_cost
        # transportation_distance = self.transportation_distance

        outputs["CAPEX_P2MeOH"] = (
            P_SOEC_MW * (SOEC_capex_cost + SOEC_power_electronics_cost)
            + P_SOEC_MW
            * psi_DAC_MWhkg
            * M_CO2
            * eff_cor
            / (3 * M_H2 * lhv)
            * DAC_capex_cost
            + P_SOEC_MW
            * w_comp_reactor_MWhkg
            * MeOH_yield
            * M_CH3OH
            * eff_cor
            / (3 * M_H2 * lhv)
            * MeOH_reactor_capex_cost
            + m_MeOH_tank_max_kg * MeOH_tank_capex_cost
        )
        # + P_heater_MW * heater_capex_cost
        outputs["OPEX_SOEC"] = P_SOEC_MW * SOEC_opex_cost

        if self.MeOH_demand_scheme == "fixed":

            outputs["water_consumption_cost"] = (
                (
                    m_MeOH_demand_t_ext.mean() * 365 * 24 * water_consumption
                    - m_H2O_t.mean() * 365 * 24
                )
                * (water_cost + water_treatment_cost)
                / 1000
            )  # annual mean water consumption to produce hydrogen over an year

        if self.MeOH_demand_scheme == "infinite":

            outputs["water_consumption_cost"] = (
                (
                    m_green_MeOH_dist_t.mean() * 365 * 24 * water_consumption
                    - m_H2O_t.mean() * 365 * 24
                )
                * (water_cost + water_treatment_cost)
                / 1000
            )  # annual mean water consumption to produce hydrogen over an year

        # outputs['OPEX_heater'] = P_heater_MW * heater_opex_cost
        outputs["OPEX_DAC"] = (
            P_SOEC_MW
            * psi_DAC_MWhkg
            * M_CO2
            * eff_cor
            / (3 * M_H2 * lhv)
            * DAC_opex_cost
        )
        outputs["OPEX_reactor"] = (
            P_SOEC_MW
            * w_comp_reactor_MWhkg
            * MeOH_yield
            * M_CH3OH
            * eff_cor
            / (3 * M_H2 * lhv)
            * MeOH_reactor_opex_cost
        )
        outputs["OPEX_MeOH_tank"] = m_MeOH_tank_max_kg * MeOH_tank_opex_cost
        # outputs['transportation_total_cost'] = m_MeOH_demand_t.mean()*365*24 * transportation_cost * transportation_distance
