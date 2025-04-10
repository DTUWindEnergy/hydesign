# %%
# import glob
# import os
# import time

# basic libraries
import numpy as np
# from numpy import newaxis as na
# import numpy_financial as npf
# import pandas as pd
# import seaborn as sns
import openmdao.api as om
# import yaml
# import scipy as sp
# from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
# from scipy import stats
# import xarray as xr

#Wisdem
from hydesign.nrel_csm_wrapper import wt_cost


class wpp_cost(om.ExplicitComponent):
    """Wind power plant cost model is used to assess the overall wind plant cost. It is based on the The NREL Cost and Scaling model [1].
    It estimates the total capital expenditure costs and operational and maintenance costs, as a function of the installed capacity, the cost of the
    turbine, intallation costs and O&M costs.
    [1] Dykes, K., et al. 2014. Sensitivity analysis of wind plant performance to key turbine design parameters: a systems engineering approach. Tech. rep. National Renewable Energy Laboratory"""

    def __init__(self,
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
        super().__init__()
        self.wind_turbine_cost = wind_turbine_cost
        self.wind_civil_works_cost = wind_civil_works_cost
        self.wind_fixed_onm_cost = wind_fixed_onm_cost
        self.wind_variable_onm_cost = wind_variable_onm_cost
        self.d_ref = d_ref
        self.hh_ref = hh_ref
        self.p_rated_ref = p_rated_ref
        self.N_time= N_time
        self.intervals_per_hour = intervals_per_hour

    def setup(self):
        #self.add_discrete_input(
        self.add_input(
            'Nwt',
            val=1,
            desc="Number of wind turbines")
        self.add_input('Awpp',
                       desc="Land use area of WPP",
                       units='km**2')

        self.add_input('hh',
                       desc="Turbine's hub height",
                       units='m')
        self.add_input('d',
                       desc="Turbine's diameter",
                       units='m')
        self.add_input('p_rated',
                       desc="Turbine's rated power",
                       units='MW')
        self.add_input('wind_t',
                       desc="WPP power time series",
                       units='MW',
                       shape=[self.N_time])

        self.add_output('CAPEX_w',
                        desc="CAPEX wpp")
        self.add_output('OPEX_w',
                        desc="OPEX wpp")

    def setup_partials(self):
        self.declare_partials('*', '*', dependent=False, val=0)

    def compute_partials(self, inputs, partials):
        pass        

    def compute(self, inputs, outputs):#, discrete_inputs, discrete_outputs):
        """ Computing the CAPEX and OPEX of the wind power plant.

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
        
        #Nwt = discrete_inputs['Nwt']
        Nwt = inputs['Nwt'][0]
        # Awpp = inputs['Awpp'][0]
        hh = inputs['hh'][0]
        d = inputs['d'][0]
        p_rated = inputs['p_rated'][0]
        wind_t= inputs['wind_t']
        wind_turbine_cost = self.wind_turbine_cost
        wind_civil_works_cost = self.wind_civil_works_cost
        wind_fixed_onm_cost = self.wind_fixed_onm_cost
        wind_variable_onm_cost= self.wind_variable_onm_cost
        
        d_ref = self.d_ref
        hh_ref = self.hh_ref
        p_rated_ref = self.p_rated_ref
        
        WT_cost_ref = wt_cost(
            rotor_diameter = d_ref,
            turbine_class = 1,
            blade_has_carbon = False,
            blade_number = 3    ,
            machine_rating = p_rated_ref*1e3, #kW
            hub_height = hh_ref,
            bearing_number = 2,
            crane = True,  
            )*1e-6
        
        WT_cost = wt_cost(
            rotor_diameter = d,
            turbine_class = 1,
            blade_has_carbon = False,
            blade_number = 3    ,
            machine_rating = p_rated*1e3, #kW
            hub_height = hh,
            bearing_number = 2,
            crane = True,  
            )*1e-6
        scale = (WT_cost/p_rated)/(WT_cost_ref/p_rated_ref)
        mean_aep_wind = wind_t.mean()*365*24*self.intervals_per_hour
        
        #print(WT_cost)
        #print(WT_cost_ref)
        #print(scale)
        #print(wind_turbine_cost)
        #print(wind_civil_works_cost)
    
        outputs['CAPEX_w'] = scale*(
            wind_turbine_cost + \
            wind_civil_works_cost) * (Nwt * p_rated)
        outputs['OPEX_w'] = wind_fixed_onm_cost * (Nwt * p_rated) + \
                            mean_aep_wind * wind_variable_onm_cost * p_rated/p_rated_ref





class pvp_cost(om.ExplicitComponent):
    """PV plant cost model is used to calculate the overall PV plant cost. The cost model estimates the total solar capital expenditure costs
    and  operational and maintenance costs as a function of the installed solar capacity and the PV cost per MW installation costs (extracted from the danish energy agency data catalogue).
    """

    def __init__(self,
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

    def setup(self):
        self.add_input(
            'solar_MW',
            val=1,
            desc="Solar PV plant installed capacity",
            units='MW')
        self.add_input(
            'DC_AC_ratio',
            desc="DC/AC PV ratio")

        self.add_output('CAPEX_s',
                        desc="CAPEX solar pvp")
        self.add_output('OPEX_s',
                        desc="OPEX solar pvp")

    def setup_partials(self):
        self.declare_partials('*', '*', dependent=False, val=0)

    # def compute_partials(self, inputs, partials):
    #     pass        

    def compute(self, inputs, outputs):
        """ Computing the CAPEX and OPEX of the solar power plant.

        Parameters
        ----------
        solar_MW : AC nominal capacity of the PV plant [MW]
        DC_AC_ratio: Ratio of DC power rating with respect AC rating of the PV plant

        Returns
        -------
        CAPEX_s : CAPEX of the PV power plant [Eur]
        OPEX_s : OPEX of the PV power plant [Eur/year]
        """
        
        solar_MW = inputs['solar_MW'][0]
        DC_AC_ratio = inputs['DC_AC_ratio'][0]
        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_inverter_cost= self.solar_inverter_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost
        
        outputs['CAPEX_s'] = (solar_PV_cost + solar_hardware_installation_cost ) * solar_MW * DC_AC_ratio + \
                              solar_inverter_cost * solar_MW
        outputs['OPEX_s'] = solar_fixed_onm_cost * solar_MW * DC_AC_ratio

    def compute_partials(self, inputs, partials):
        solar_MW = inputs['solar_MW'][0]
        DC_AC_ratio = inputs['DC_AC_ratio'][0]
        DC_AC_ratio_tech_ref =  1.25
        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_inverter_cost = self.solar_inverter_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost

        partials['CAPEX_s', 'solar_MW'] = (solar_PV_cost + solar_hardware_installation_cost ) * DC_AC_ratio + \
             solar_inverter_cost * DC_AC_ratio_tech_ref/DC_AC_ratio 
        partials['CAPEX_s', 'DC_AC_ratio'] = (solar_PV_cost + \
            solar_hardware_installation_cost + solar_inverter_cost * DC_AC_ratio_tech_ref/(- DC_AC_ratio ** 2)) * solar_MW
        partials['OPEX_s', 'solar_MW'] = solar_fixed_onm_cost * DC_AC_ratio
        partials['OPEX_s', 'DC_AC_ratio'] = solar_fixed_onm_cost * solar_MW


class battery_cost(om.ExplicitComponent):
    """Battery cost model calculates the storage unit costs. It uses technology costs extracted from the danish energy agency data catalogue."""

    def __init__(self,
                 battery_energy_cost,
                 battery_power_cost,
                 battery_BOP_installation_commissioning_cost,
                 battery_control_system_cost,
                 battery_energy_onm_cost,
                 life_y = 25,
                 # life_h = 25*365*24
                 intervals_per_hour = 1,
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
        super().__init__()
        self.battery_energy_cost = battery_energy_cost
        self.battery_power_cost = battery_power_cost
        self.battery_BOP_installation_commissioning_cost = battery_BOP_installation_commissioning_cost
        self.battery_control_system_cost = battery_control_system_cost
        self.battery_energy_onm_cost = battery_energy_onm_cost
        # self.N_life = life_y
        self.life_h = life_y * 365 * 24
        self.yearly_intervals = 365 * 24 * intervals_per_hour
        self.life_intervals = life_y * self.yearly_intervals
        self.battery_price_reduction_per_year = battery_price_reduction_per_year


    def setup(self):
        self.add_input('b_P',
                       desc="Battery power capacity",
                       units='MW')
        self.add_input('b_E',
                       desc="Battery energy storage capacity")
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_intervals])
        self.add_output('CAPEX_b',
                        desc="CAPEX battery")
        self.add_output('OPEX_b',
                        desc="OPEX battery")

    def setup_partials(self):
        self.declare_partials('*', '*', dependent=False, val=0)

    def compute_partials(self, inputs, partials):
        pass        

    def compute(self, inputs, outputs):
        """ Computing the CAPEX and OPEX of battery.

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
        
        # N_life = self.N_life
        life_intervals = self.life_intervals
        age = np.arange(life_intervals)/self.yearly_intervals
        
        b_E = inputs['b_E'][0]
        b_P = inputs['b_P'][0]
        SoH = inputs['SoH']
        battery_price_reduction_per_year = self.battery_price_reduction_per_year

        battery_energy_cost = self.battery_energy_cost
        battery_power_cost = self.battery_power_cost
        battery_BOP_installation_commissioning_cost = self.battery_BOP_installation_commissioning_cost
        battery_control_system_cost = self.battery_control_system_cost
        battery_energy_onm_cost = self.battery_energy_onm_cost
        

        ii_battery_change = np.where( (SoH>0.99) & ( np.append(1, np.diff(SoH)) > 0) )[0]
        year_new_battery = np.unique(np.floor(age[ii_battery_change]))
        
        factor = 1.0 - battery_price_reduction_per_year
        N_beq = np.sum([factor**iy for iy in year_new_battery])

        CAPEX_b = N_beq*(battery_energy_cost * b_E) + \
                (battery_power_cost + \
                 battery_BOP_installation_commissioning_cost + \
                 battery_control_system_cost) * b_P
        
        OPEX_b  = battery_energy_onm_cost * b_E

        outputs['CAPEX_b'] = CAPEX_b
        outputs['OPEX_b'] = OPEX_b

class shared_cost(om.ExplicitComponent):
    """Electrical infrastructure and land rent cost model"""

    def __init__(self,
                 hpp_BOS_soft_cost,
                 hpp_grid_connection_cost,
                 land_cost
                 ):
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
        self.add_input('G_MW',
                       desc="Grid capacity",
                       units='MW')
        self.add_input('Awpp',
                       desc="Land use area of WPP",
                       units='km**2')
        self.add_input('Apvp',
                        desc="Land use area of SP",
                        units='km**2')

        self.add_output('CAPEX_sh',
                        desc="CAPEX electrical infrastructure/ land rent")
        self.add_output('OPEX_sh',
                        desc="OPEX electrical infrastructure/ land rent")

    def setup_partials(self):
        self.declare_partials('*', '*', dependent=False, val=0)

    # def compute_partials(self, inputs, partials):
    #     pass        

    def compute(self, inputs, outputs):
        """ Computing the CAPEX and OPEX of the shared land and infrastructure.

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
        G_MW = inputs['G_MW'][0]
        Awpp = inputs['Awpp'][0]
        Apvp = inputs['Apvp'][0]
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost
        
        if (Awpp>=Apvp):
            land_rent = land_cost * Awpp
        else:
            land_rent= land_cost * Apvp
            
        #print(land_rent)
        outputs['CAPEX_sh'] = (
            hpp_BOS_soft_cost + hpp_grid_connection_cost) * G_MW + land_rent
        outputs['OPEX_sh'] = 0

    def compute_partials(self, inputs, partials):
        # G_MW = inputs['G_MW']
        Awpp = inputs['Awpp'][0]
        Apvp = inputs['Apvp'][0]
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        partials['CAPEX_sh', 'G_MW'] = hpp_BOS_soft_cost + hpp_grid_connection_cost
        if (Awpp>=Apvp):
            partials['CAPEX_sh', 'Awpp'] = land_cost
            partials['CAPEX_sh', 'Apvp'] = 0
        else:
            partials['CAPEX_sh', 'Awpp'] = 0
            partials['CAPEX_sh', 'Apvp'] = land_cost
        partials['OPEX_sh', 'G_MW'] = 0
        partials['OPEX_sh', 'Awpp'] = 0
        partials['OPEX_sh', 'Apvp'] = 0

class ptg_cost(om.ExplicitComponent):
    """Power to H2 plant cost model is used to calculate the overall H2 plant cost. The cost model includes cost of electrolyzer
     and compressor for storing Hydrogen (data extracted from the danish energy agency data catalogue and IRENA reports).
    """
    def __init__(self,
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
                 life_y = 25,
                 intervals_per_hour = 1,
                 ):

        super().__init__()
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

    def setup(self):

        self.add_input('ptg_MW',
                        desc = "Installed capacity for the power to gas plant",
                        units = 'MW')
        self.add_input('m_H2_t',
                        desc = "Produced hydrogen",
                        units = "kg",
                        shape=[self.life_intervals])
        self.add_input('HSS_kg',
                        desc = "Installed capacity of Hydrogen storage",
                        units = 'kg')
        self.add_input('m_H2_demand_t_ext',
                        desc = "Hydrogen demand",
                        units = "kg",
                        shape=[self.life_intervals])
        self.add_input('m_H2_offtake_t',
                        desc = "Offtake hydrogen",
                        units = "kg",
                        shape=[self.life_intervals])

        
        #Creating outputs:
        self.add_output('CAPEX_ptg',
                        desc = "CAPEX power to gas")
        self.add_output('OPEX_ptg',
                        desc = "OPEX power to gas") 
        self.add_output('water_consumption_cost',
                        desc = "Annual water consumption and treatment cost",
                        )

    def compute(self, inputs, outputs):

       
        ptg_MW = inputs['ptg_MW'][0]
        # m_H2_t = inputs['m_H2_t']
        HSS_kg = inputs['HSS_kg'][0]
        # m_H2_demand_t = inputs['m_H2_demand_t_ext']
        m_H2_offtake_t = inputs['m_H2_offtake_t']

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

        outputs['CAPEX_ptg'] = ptg_MW * (electrolyzer_capex_cost + electrolyzer_power_electronics_cost) + storage_capex_cost * HSS_kg + \
                               (m_H2_offtake_t.mean()*self.yearly_intervals * transportation_cost * transportation_distance) 
        outputs['OPEX_ptg'] = ptg_MW * (electrolyzer_opex_cost) + storage_opex_cost * HSS_kg
        outputs['water_consumption_cost'] = (m_H2_offtake_t.mean()*self.yearly_intervals * water_consumption * (water_cost + water_treatment_cost)/1000) # annual mean water consumption to produce hydrogen over an year


       