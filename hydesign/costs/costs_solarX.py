# %%

# Import necessary libraries
import os
import time
import numpy as np
from numpy import newaxis as na
import openmdao.api as om
import scipy as sp


class sf_cost(om.ExplicitComponent):
    """
    Solar Field (SF) Cost Model - Calculates the capital and operational expenses
    for a solar field based on the input parameters.
    """
    def __init__(self, heliostat_cost_per_m2, sf_opex_cost_per_m2):
        super().__init__()
        # Set the cost parameters
        self.heliostat_cost_per_m2 = heliostat_cost_per_m2
        self.sf_opex_cost_per_m2 = sf_opex_cost_per_m2

    def setup(self):
        # Define inputs and outputs for the component
        # inputs
        self.add_input('sf_area', desc="Installed capacity of the solar field", units='m**2')
        # outputs
        self.add_output('CAPEX_sf', desc="CAPEX of solar field (mirrors and controllers)")
        self.add_output('OPEX_sf', desc="OPEX of solar field (mirrors and controllers)")

    def compute(self, inputs, outputs):
        # Calculate capital and operational costs based on area of solar field
        sf_area = inputs['sf_area']
        capex_sf = self.heliostat_cost_per_m2 * sf_area
        opex_sf = self.sf_opex_cost_per_m2 * sf_area
        outputs['CAPEX_sf'] = capex_sf
        outputs['OPEX_sf'] = opex_sf


class cpv_cost(om.ExplicitComponent):
    """
    Concentrated Photovoltaic (cpv) Cost Model - Calculates CAPEX and OPEX
    for cpv systems based on installation and equipment costs.
    """
    def __init__(self, cpv_cost_per_m2, inverter_cost_per_MW_DC, cpv_fixed_opex_cost_per_m2):
        super().__init__()
        # Set cost parameters for cpv system
        self.cpv_cost_per_m2 = cpv_cost_per_m2
        self.inverter_cost_per_MW_DC = inverter_cost_per_MW_DC
        self.cpv_fixed_opex_cost_per_m2 = cpv_fixed_opex_cost_per_m2

    def setup(self):
        # Define inputs and outputs
        # inputs
        self.add_input('cpv_inverter_mw', desc="rated power of the cpv inverter", units='MW',)
        self.add_input('area_cpv_receiver_m2', desc="Area of cpv receivers", units='m**2')
        # outputs
        self.add_output('CAPEX_cpv', desc="CAPEX of cpv system")
        self.add_output('OPEX_cpv', desc="OPEX of cpv system")

    def compute(self, inputs, outputs):
        # Compute CAPEX and OPEX for cpv system
        cpv_inverter_mw = inputs['cpv_inverter_mw']
        cpv_m2 = inputs['area_cpv_receiver_m2']

        capex_cpv = self.cpv_cost_per_m2 * cpv_m2 + (self.inverter_cost_per_MW_DC * cpv_inverter_mw)
        opex_cpv = self.cpv_fixed_opex_cost_per_m2 * cpv_m2
        outputs['CAPEX_cpv'] = capex_cpv
        outputs['OPEX_cpv'] = opex_cpv


class cst_cost(om.ExplicitComponent):
    """
    Concentrated Solar Thermal (cst) Cost Model - Calculates the capital and
    operational expenses for cst systems based on collector, molten salt tank,
    and turbine-related costs.
    """
    def __init__(self,
                 cst_th_collector_cost_per_m2,
                 ms_installation_cost_per_m3,
                 steam_turbine_cost_per_MW,
                 heat_exchnager_cost_per_MW,# MWt, kg/h steam
                 fixed_opex_per_MW):
        super().__init__()
        # Set cost parameters for cst system components
        self.cst_th_collector_cost_per_m2 = cst_th_collector_cost_per_m2
        self.ms_installation_cost_per_m3 = ms_installation_cost_per_m3
        self.steam_turbine_cost_per_MW = steam_turbine_cost_per_MW
        self.heat_exchnager_cost_per_MW = heat_exchnager_cost_per_MW
        self.fixed_opex_per_MW = fixed_opex_per_MW

    def setup(self):
        # Define inputs and outputs for cst component
        # inputs
        self.add_input('area_cst_receiver_m2', val=1, desc="Area of heat receiver on the tower", units='m**2')
        self.add_input('v_molten_salt_tank_m3', val=1, desc="Volume of the molten salt storage", units='m**3')
        self.add_input('p_rated_st', desc="Steam turbine power capacity", units="MW")
        self.add_input('heat_exchanger_capacity', desc='Heat exchnager power capacity', units='MW')

        # outputs
        self.add_output('CAPEX_cst', desc="CAPEX of cst system ")
        self.add_output('OPEX_cst', desc="OPEX of cst system ")

    def compute(self, inputs, outputs):
        # Calculate CAPEX and OPEX for cst system
        area_cst_receiver_m2 = inputs['area_cst_receiver_m2']
        v_molten_salt_tank_m3 = inputs['v_molten_salt_tank_m3']
        p_rated_st = inputs['p_rated_st']
        heat_exchanger_capacity = inputs['heat_exchanger_capacity']

        # CAPEX and OPEX calculations
        capex_receiver = self.cst_th_collector_cost_per_m2 * area_cst_receiver_m2
        capex_molten_salt = self.ms_installation_cost_per_m3 * v_molten_salt_tank_m3
        capex_heat_exchanger = heat_exchanger_capacity * self.heat_exchnager_cost_per_MW
        capex_turbine = p_rated_st * self.steam_turbine_cost_per_MW
        capex_cst = capex_receiver + capex_molten_salt + capex_heat_exchanger + capex_turbine

        opex_cst = self.fixed_opex_per_MW * p_rated_st

        # outputs
        outputs['CAPEX_cst'] = capex_cst
        outputs['OPEX_cst'] = opex_cst


class H2Cost(om.ExplicitComponent):
    """
    Hydrogen Production Cost Model - Calculates the capital and operational expenses
    for H2 production based on reactor, installation, and maintenance costs.
    """
    def __init__(self,
                 reactor_cost_per_m2,  # Waiting for Luc input
                 maximum_h2_production_reactor_kg_per_m2,
                 el_heater_cost_kg_per_h,  # Waiting for Luc input
                 pipe_pump_valves_cost_kg_per_h,  # Waiting for Luc input
                 psa_cost_kg_per_h,
                 carbon_capture_cost_kg_per_h,
                 dni_installation_cost_kg_per_h,
                 el_installation_cost_kg_per_h,
                 maintenance_cost_kg_per_h,
                 life_h,
                 carbon_capture=False,
                 ):

        super().__init__()
        self.life_h = int(life_h)
        self.carbon_capture = carbon_capture
        self.maximum_h2_production_reactor_kg_per_m2 = maximum_h2_production_reactor_kg_per_m2
        # capex
        self.reactor_cost_per_m2 = reactor_cost_per_m2
        self.el_heater_cost_kg_per_h = el_heater_cost_kg_per_h  # only for electrical reactors
        self.pipe_pump_valves_cost_kg_per_h = pipe_pump_valves_cost_kg_per_h
        self.psa_cost_kg_per_h = psa_cost_kg_per_h
        self.carbon_capture_cost_kg_per_h = carbon_capture_cost_kg_per_h

        # installation
        self.dni_installation_cost_kg_per_h = dni_installation_cost_kg_per_h
        self.el_installation_cost_kg_per_h = el_installation_cost_kg_per_h

        # opex
        self.maintenance_cost_kg_per_h = maintenance_cost_kg_per_h

    def setup(self):
        # Define inputs for the openmdao model
        # inputs
        self.add_input('area_el_reactor_biogas_h2', desc='Area of the biogas_h2 electrical reactor', units='m**2')
        self.add_input('area_dni_reactor_biogas_h2', desc='Area of the biogas_h2 dni reactor', units='m**2')
        self.add_input('biogas_t_ext', desc="Biogas consumption time series", units='kg/h', shape=[self.life_h])
        self.add_input('water_t_ext', desc="Water consumption time series", units='kg/h', shape=[self.life_h])
        self.add_input('co2_t_ext', desc="CO2 consumption time series", units='kg/h', shape=[self.life_h])
        self.add_input('p_biogas_h2_t', desc="electricity consumption time series", shape=[self.life_h], units='MW')
        self.add_input('price_el_t_ext', desc="electricity price time series", shape=[self.life_h])
        self.add_input('price_biogas_t_ext', desc="electricity price time series", shape=[self.life_h])
        self.add_input('price_water_t_ext', desc="electricity price time series", shape=[self.life_h])
        self.add_input('price_co2_t_ext', desc="electricity price time series", shape=[self.life_h])

        # outputs
        self.add_output('CAPEX_h2', desc="CAPEX of H2 Production")
        self.add_output('OPEX_h2', desc="OPEX of H2 Production")
        self.add_output('OPEX_el', desc="OPEX costs for electricity")

    def compute(self, inputs, outputs):
        # Calculate CAPEX based on capacity and component costs
        # load data
        area_el_reactor_biogas_h2 = inputs['area_el_reactor_biogas_h2']
        area_dni_reactor_biogas_h2 = inputs['area_dni_reactor_biogas_h2']
        water_t_ext = inputs['water_t_ext']
        biogas_t_ext = inputs['biogas_t_ext']
        co2_t_ext = inputs['co2_t_ext']
        p_biogas_h2_t = inputs['p_biogas_h2_t']
        price_el_t = inputs['price_el_t_ext']
        price_biogas_t = inputs['price_biogas_t_ext']
        price_water_t = inputs['price_water_t_ext']
        price_co2_t = inputs['price_co2_t_ext']
        carbon_capture = self.carbon_capture # indicator of the presence of carbon capture

        el_h2_kg_h = area_el_reactor_biogas_h2 * self.maximum_h2_production_reactor_kg_per_m2
        dni_h2_kg_h = area_dni_reactor_biogas_h2 * self.maximum_h2_production_reactor_kg_per_m2

        # Total area and capacity for H2 reactors
        overall_h2_receiver_area = area_el_reactor_biogas_h2 + area_dni_reactor_biogas_h2
        overall_h2_kg_per_h = el_h2_kg_h + dni_h2_kg_h

        # Reactor and component costs
        reactor_cost = self.reactor_cost_per_m2 * overall_h2_receiver_area
        pipe_pump_valves_cost = self.pipe_pump_valves_cost_kg_per_h * overall_h2_kg_per_h
        psa_cost = self.psa_cost_kg_per_h * overall_h2_kg_per_h

        # cost for capturing the carbon
        if carbon_capture:
            carbon_capture_cost = self.carbon_capture_cost_kg_per_h * overall_h2_kg_per_h
        else:
            carbon_capture_cost = 0

        # Electrical heater and installation costs
        el_heater_cost = el_h2_kg_h * self.el_heater_cost_kg_per_h
        installation_cost = el_h2_kg_h * self.el_installation_cost_kg_per_h + dni_h2_kg_h * self.dni_installation_cost_kg_per_h

        # Total CAPEX
        capex = reactor_cost + pipe_pump_valves_cost + psa_cost + carbon_capture_cost + el_heater_cost + installation_cost
        outputs['CAPEX_h2'] = capex

        # OPEX calculations for maintenance, consumed water, biogas, CO2
        maintenance_cost = overall_h2_kg_per_h * self.maintenance_cost_kg_per_h
        water_cost = sum(water_t_ext * price_water_t)
        biogas_cost = sum(biogas_t_ext * price_biogas_t)
        co2_cost = sum(co2_t_ext * price_co2_t)
        outputs['OPEX_h2'] = water_cost + co2_cost + biogas_cost + maintenance_cost

        # OPEX calculations for consumed electricity
        outputs['OPEX_el'] = sum(p_biogas_h2_t * price_el_t)


class shared_cost(om.ExplicitComponent):
    """
    Shared Cost Model - Calculates costs for electrical infrastructure, land rental, and tower.
    """

    def __init__(self,
                 grid_connection_cost_per_mw,
                 grid_h2_connection_cost_per_kg_h,
                 grid_thermal_connection_cost_per_mwt,
                 land_cost_m2,
                 BOS_soft_cost,
                 tower_cost_per_m,
                 ):
        super().__init__()
        # Set cost parameters
        self.BOS_soft_cost = BOS_soft_cost
        self.grid_connection_cost_per_mw = grid_connection_cost_per_mw
        self.grid_h2_connection_cost_per_kg_h = grid_h2_connection_cost_per_kg_h
        self.grid_thermal_connection_cost_per_mwt = grid_thermal_connection_cost_per_mwt
        self.land_cost_m2 = land_cost_m2
        self.tower_cost_per_m = tower_cost_per_m

    def setup(self):
        # Define inputs and outputs for shared costs
        # inputs
        self.add_input('grid_el_capacity',
                       desc="Grid electrical capacity",
                       units='MW')
        self.add_input('grid_heat_capacity',
                       desc="Grid Heat connection capacity",
                       units='MW')
        self.add_input('grid_h2_capacity',
                       desc="Grid Hydrogen capacity",
                       units='kg/h')
        self.add_input('sf_area',
                       desc="Land use area of SolarX",
                       units='m**2')
        self.add_input('tower_height', val=1,
                       desc="Total height of the tower",
                       units='m')

        # outputs
        self.add_output('CAPEX_sh', desc="shared CAPEX")
        self.add_output('OPEX_sh', desc="shared OPEX")

    def compute(self, inputs, outputs):
        # Calculate shared CAPEX and OPEX costs
        grid_el_capacity = inputs['grid_el_capacity']
        grid_heat_capacity = inputs['grid_heat_capacity']
        grid_h2_capacity = inputs['grid_h2_capacity']
        sf_area = inputs['sf_area']
        tower_height = inputs['tower_height']

        # Land and grid connection costs
        land_cost_m2 = self.land_cost_m2
        BOS_soft_cost = self.BOS_soft_cost
        grid_connection_cost_per_mw = self.grid_connection_cost_per_mw

        land_rent = land_cost_m2 * sf_area
        CAPEX_tower = self.tower_cost_per_m * tower_height

        outputs['CAPEX_sh'] = BOS_soft_cost * sf_area + grid_connection_cost_per_mw * grid_el_capacity + grid_heat_capacity * self.grid_thermal_connection_cost_per_mwt + grid_h2_capacity * self.grid_h2_connection_cost_per_kg_h + land_rent + CAPEX_tower
        outputs['OPEX_sh'] = 0

# class ptg_cost(om.ExplicitComponent):
#     """Power to H2 plant cost model is used to calculate the overall H2 plant cost. The cost model includes cost of electrolyzer
#      and compressor for storing Hydrogen (data extracted from the danish energy agency data catalogue and IRENA reports).
#     """
#     def __init__(self,
#                  electrolyzer_capex_cost,
#                  electrolyzer_opex_cost,
#                  electrolyzer_power_electronics_cost,
#                  water_cost,
#                  water_treatment_cost,
#                  water_t_ext,
#                  storage_capex_cost,
#                  storage_opex_cost,
#                  transportation_cost,
#                  transportation_distance,
#                  N_time,
#                  life_h = 25*365*24,):
#
#         super().__init__()
#         self.electrolyzer_capex_cost = electrolyzer_capex_cost
#         self.electrolyzer_opex_cost = electrolyzer_opex_cost
#         self.electrolyzer_power_electronics_cost = electrolyzer_power_electronics_cost
#         self.water_cost = water_cost
#         self.water_treatment_cost = water_treatment_cost
#         self.water_t_ext = water_t_ext
#         self.storage_capex_cost = storage_capex_cost
#         self.storage_opex_cost = storage_opex_cost
#         self.transportation_cost = transportation_cost
#         self.transportation_distance = transportation_distance
#         self.N_time = N_time
#         self.life_h = life_h
#
#     def setup(self):
#
#         self.add_input('ptg_MW',
#                         desc = "Installed capacity for the power to gas plant",
#                         units = 'MW')
#         self.add_input('m_H2_t',
#                         desc = "Produced hydrogen",
#                         units = "kg",
#                         shape=[self.life_h])
#         self.add_input('HSS_kg',
#                         desc = "Installed capacity of Hydrogen storage",
#                         units = 'kg')
#         self.add_input('m_H2_demand_t_ext',
#                         desc = "Hydrogen demand",
#                         units = "kg",
#                         shape=[self.life_h])
#         self.add_input('m_H2_offtake_t',
#                         desc = "Offtake hydrogen",
#                         units = "kg",
#                         shape=[self.life_h])
#
#
#         #Creating outputs:
#         self.add_output('CAPEX_ptg',
#                         desc = "CAPEX power to gas")
#         self.add_output('OPEX_ptg',
#                         desc = "OPEX power to gas")
#         self.add_output('water_t_ext_cost',
#                         desc = "Annual water consumption and treatment cost",
#                         )
#
#     def compute(self, inputs, outputs):
#
#
#         ptg_MW = inputs['ptg_MW']
#         m_H2_t = inputs['m_H2_t']
#         HSS_kg = inputs['HSS_kg']
#         m_H2_demand_t = inputs['m_H2_demand_t_ext']
#         m_H2_offtake_t = inputs['m_H2_offtake_t']
#
#         electrolyzer_capex_cost = self.electrolyzer_capex_cost
#         electrolyzer_opex_cost = self.electrolyzer_opex_cost
#         electrolyzer_power_electronics_cost = self.electrolyzer_power_electronics_cost
#         water_cost = self.water_cost
#         water_treatment_cost = self.water_treatment_cost
#         water_t_ext = self.water_t_ext
#         storage_capex_cost = self.storage_capex_cost
#         storage_opex_cost = self.storage_opex_cost
#         transportation_cost = self.transportation_cost
#         transportation_distance = self.transportation_distance
#
#         outputs['CAPEX_ptg'] = ptg_MW * (electrolyzer_capex_cost + electrolyzer_power_electronics_cost) + storage_capex_cost * HSS_kg + \
#                                (m_H2_offtake_t.mean()*365*24 * transportation_cost * transportation_distance)
#         outputs['OPEX_ptg'] = ptg_MW * (electrolyzer_opex_cost) + storage_opex_cost * HSS_kg
#         outputs['water_t_ext_cost'] = (m_H2_offtake_t.mean()*365*24 * water_t_ext * (water_cost + water_treatment_cost)/1000) # annual mean water consumption to produce hydrogen over an year


       