# %%
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
# import seaborn as sns
import openmdao.api as om
import yaml
import scipy as sp
from statsmodels.distributions.empirical_distribution import ECDF, monotone_fn_inverter
from scipy import stats
import xarray as xr

#Wisdem
from hydesign.nrel_csm_tcc_2015 import get_WT_cost_wisdem


class wpp_cost(om.ExplicitComponent):
    """Wind power plant cost model is used to assess the overall wind plant cost. It is based on the The NREL Cost and Scaling model *.
    It estimates the total capital expenditure costs and operational and maintenance costs, as a function of the installed capacity, the cost of the
    turbine, intallation costs and O&M costs.
    * Dykes, K., et al. 2014. Sensitivity analysis of wind plant performance to key turbine design parameters: a systems engineering approach. Tech. rep. National Renewable Energy Laboratory"""

    def __init__(self,
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
        self.N_time= N_time

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
        self.declare_partials('*', '*', method='fd')

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
        Nwt = inputs['Nwt']
        Awpp = inputs['Awpp']
        hh = inputs['hh']
        d = inputs['d']
        p_rated = inputs['p_rated']
        wind_t= inputs['wind_t']
        wind_turbine_cost = self.wind_turbine_cost
        wind_civil_works_cost = self.wind_civil_works_cost
        wind_fixed_onm_cost = self.wind_fixed_onm_cost
        wind_variable_onm_cost= self.wind_variable_onm_cost
        # costs scales as a function of diameter and hh with respect reference values 
        # https://www.nrel.gov/docs/fy07osti/40566.pdf
        # # Cheap
        # d_ref = 120
        # hh_ref = 100
        
        # # Medium
        #d_ref = 100
        #hh_ref = 80
        
        # Expensive
        # d_ref = 80
        # hh_ref = 60
        
        d_ref = self.d_ref
        hh_ref = self.hh_ref
        p_rated_ref = self.p_rated_ref
        
        WT_cost_ref = get_WT_cost_wisdem(
            rotor_diameter = d_ref,
            turbine_class = 1,
            blade_has_carbon = False,
            blade_number = 3    ,
            machine_rating = p_rated_ref*1e3, #kW
            hub_height = hh_ref,
            bearing_number = 2,
            crane = True,  
            verbosity = False
            )
        
        WT_cost = get_WT_cost_wisdem(
            rotor_diameter = d,
            turbine_class = 1,
            blade_has_carbon = False,
            blade_number = 3    ,
            machine_rating = p_rated*1e3, #kW
            hub_height = hh,
            bearing_number = 2,
            crane = True,  
            verbosity = False
            )
        scale = (WT_cost/p_rated)/(WT_cost_ref/p_rated_ref)
        mean_aep_wind = wind_t.mean()*365*24
        
        #print(WT_cost)
        #print(WT_cost_ref)
        #print(scale)
        #print(wind_turbine_cost)
        #print(wind_civil_works_cost)
    
        outputs['CAPEX_w'] = scale*(
            wind_turbine_cost + \
            wind_civil_works_cost) * (Nwt * p_rated)
        outputs['OPEX_w'] = wind_fixed_onm_cost * (Nwt * p_rated) + mean_aep_wind * wind_variable_onm_cost * p_rated_ref/p_rated





class pvp_cost(om.ExplicitComponent):
    """PV plant cost model is used to calculate the overall PV plant cost. The cost model estimates the total solar capital expenditure costs
    and  operational and maintenance costs as a function of the installed solar capacity and the PV cost per MW installation costs (extracted from the danish energy agency data catalogue).
    """

    def __init__(self,
                 solar_PV_cost,
                 solar_hardware_installation_cost,
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
        self.solar_fixed_onm_cost = solar_fixed_onm_cost

    def setup(self):
        self.add_input('solar_MW',
                       val=1,
                       desc="Solar PV plant installed capacity",
                       units='MW')

        self.add_output('CAPEX_s',
                        desc="CAPEX solar pvp")
        self.add_output('OPEX_s',
                        desc="OPEX solar pvp")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """ Computing the CAPEX and OPEX of the solar power plant.

        Parameters
        ----------
        solar_MW : AC nominal capacity of the PV plant [MW]

        Returns
        -------
        CAPEX_s : CAPEX of the PV power plant [Eur]
        OPEX_s : OPEX of the PV power plant [Eur/year]
        """
        
        solar_MW = inputs['solar_MW']

        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost

        outputs['CAPEX_s'] = (
            solar_PV_cost + solar_hardware_installation_cost) * solar_MW
        outputs['OPEX_s'] = solar_fixed_onm_cost * solar_MW

    def compute_partials(self, inputs, partials):
        solar_MW = inputs['solar_MW']

        solar_PV_cost = self.solar_PV_cost
        solar_hardware_installation_cost = self.solar_hardware_installation_cost
        solar_fixed_onm_cost = self.solar_fixed_onm_cost

        partials['CAPEX_s', 'solar_MW'] = solar_PV_cost + \
            solar_hardware_installation_cost 
        partials['OPEX_s', 'solar_MW'] = solar_fixed_onm_cost


class battery_cost(om.ExplicitComponent):
    """Battery cost model calculates the storage unit costs. It uses technology costs extracted from the danish energy agency data catalogue."""

    def __init__(self,
                 battery_energy_cost,
                 battery_power_cost,
                 battery_BOP_installation_commissioning_cost,
                 battery_control_system_cost,
                 battery_energy_onm_cost,
                 num_batteries = 1,
                 n_steps_in_LoH = 30,
                 N_life = 25,
                 life_h = 25*365*24):
        """Initialization of the battery cost model

        Parameters
        ----------
        battery_energy_cost : Battery energy cost [Euro/MWh]
        battery_power_cost : Battery power cost [Euro/MW]
        battery_BOP_installation_commissioning_cost : Battery installation and commissioning cost [Euro/MW]
        battery_control_system_cost : Battery system controt cost [Euro/MW]
        battery_energy_onm_cost : Battery operation and maintenance cost [Euro/MW]
        num_batteries : Number of battery replacement in the lifetime of the plant
        n_steps_in_LoH : Number of discretization steps in battery level of health
        N_life : Lifetime of the plant in years
        life_h : Total number of hours in the lifetime of the plant


        """ 
        super().__init__()
        self.battery_energy_cost = battery_energy_cost
        self.battery_power_cost = battery_power_cost
        self.battery_BOP_installation_commissioning_cost = battery_BOP_installation_commissioning_cost
        self.battery_control_system_cost = battery_control_system_cost
        self.battery_energy_onm_cost = battery_energy_onm_cost
        self.N_life = N_life
        self.life_h = life_h
        self.num_batteries = num_batteries
        self.n_steps_in_LoH = n_steps_in_LoH


    def setup(self):
        self.add_input('b_P',
                       desc="Battery power capacity",
                       units='MW')
        self.add_input('b_E',
                       desc="Battery energy storage capacity")
        self.add_input(
            'ii_time',
            desc="indices on the lifetime timeseries."+
                " Hydesign operates in each range at constant battery health",
            shape=[self.n_steps_in_LoH*self.num_batteries + 1 ])
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.n_steps_in_LoH*self.num_batteries + 1])
        self.add_input('battery_price_reduction_per_year',
                       desc="Factor of battery price reduction per year")

        self.add_output('CAPEX_b',
                        desc="CAPEX battery")
        self.add_output('OPEX_b',
                        desc="OPEX battery")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """ Computing the CAPEX and OPEX of battery.

        Parameters
        ----------
        b_P : Battery power capacity [MW]
        b_E : Battery energy storage capacity [MWh]
        ii_time : Indices on the lifetime time series (Hydesign operates in each range at constant battery health)
        SoH : Battery state of health at discretization levels 
        battery_price_reduction_per_year : Factor of battery price reduction per year

        Returns
        -------
        CAPEX_b : CAPEX of the storage unit [Eur]
        OPEX_b : OPEX of the storage unit [Eur/year]
        """
        
        N_life = self.N_life
        life_h = self.life_h
        
        b_E = inputs['b_E']
        b_P = inputs['b_P']
        ii_time = inputs['ii_time']
        SoH = inputs['SoH']
        battery_price_reduction_per_year = inputs['battery_price_reduction_per_year']

        battery_energy_cost = self.battery_energy_cost
        battery_power_cost = self.battery_power_cost
        battery_BOP_installation_commissioning_cost = self.battery_BOP_installation_commissioning_cost
        battery_control_system_cost = self.battery_control_system_cost
        battery_energy_onm_cost = self.battery_energy_onm_cost
        
        ii_battery_change = ii_time[np.where(SoH>=0.999)[0]]
        
        
        factor = 1.0 - battery_price_reduction_per_year[0]
        N_b_equi = np.sum([factor**(N_life*ii/life_h) for ii in ii_battery_change])
        
        outputs['CAPEX_b'] = N_b_equi * (battery_energy_cost * b_E) + \
                (battery_power_cost + \
                 battery_BOP_installation_commissioning_cost + \
                 battery_control_system_cost) * b_P
        outputs['OPEX_b'] = battery_energy_onm_cost * b_E


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
        self.declare_partials('*', '*', method='fd')

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
        G_MW = inputs['G_MW']
        Awpp = inputs['Awpp']
        Apvp = inputs['Apvp']
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost
        
        #land_use_per_solar_MW = 0.01226 #[km2] # add source
        #Apvp = solar_MW * land_use_per_solar_MW
        #print(Awpp)
        #print(Apvp)
        if (Awpp>=Apvp):
            land_rent = land_cost * Awpp
        else:
            land_rent= land_cost * Apvp
        #print(land_rent)
        outputs['CAPEX_sh'] = (
            hpp_BOS_soft_cost + hpp_grid_connection_cost) * G_MW + land_rent
        outputs['OPEX_sh'] = 0

    def compute_partials(self, inputs, partials):
        G_MW = inputs['G_MW']
        Awpp = inputs['Awpp']
        Apvp = inputs['Apvp']
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
       