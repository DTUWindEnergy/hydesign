# import glob
# import os
# import time

# basic libraries
# import numpy as np
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
# import matplotlib.pyplot as plt

# Wisdem
# from hydesign.nrel_csm_wrapper import wt_cost


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
        self.add_input('p_rated',
                       desc="Power rated",
                       units='MW')
        self.add_input('Nwt',
                       desc="Number of wind turbines",
                       )
        self.add_input('solar_MW',
                       desc="Solar capacity",
                       units='MW')
        self.add_input('Awpp',
                       desc="Land use area of WPP",
                       units='km**2')
        self.add_input('Apvp',
                       desc="Land use area of SP",
                       units='km**2')
        self.add_output('CAPEX_sh_w',
                        desc="CAPEX electrical infrastructure/ land rent for wind")
        self.add_output('CAPEX_sh_s',
                        desc="CAPEX electrical infrastructure/ land rent for PV and batteries")
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
        CAPEX_sh_w : CAPEX electrical infrastructure/ land rent for the wind stand-alone[Eur]
        CAPEX_sh_s : CAPEX electrical infrastructure/ land rent for the added pv [Eur]
        OPEX_sh : OPEX electrical infrastructure/ land rent [Eur/year]
        """
        Nwt = inputs['Nwt']
        p_rated = inputs['p_rated']
        # solar_MW = inputs['solar_MW']
        # G_MW = inputs['G_MW']
        Awpp = inputs['Awpp']
        Apvp = inputs['Apvp']
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        #if (Awpp >= Apvp):
        #    land_rent = land_cost * Awpp
        #else:

        land_rent_wind = land_cost * Awpp
        land_rent_pv = land_cost * Apvp

        outputs['CAPEX_sh_w'] = (hpp_BOS_soft_cost + hpp_grid_connection_cost) * p_rated * Nwt + land_rent_wind  # MODIFICA!

        if (Apvp > Awpp):
            outputs['CAPEX_sh_s'] =  land_rent_pv - land_rent_wind # (hpp_BOS_soft_cost  + hpp_grid_connection_cost) * (G_MW-solar_MW)  # We don't include the land ofthe PV, because they can be occupy the same space of the wt
        else:
            outputs['CAPEX_sh_s'] = 0

        outputs['OPEX_sh'] = 0


    def compute_partials(self, inputs, partials):
        # G_MW = inputs['G_MW']
        Awpp = inputs['Awpp']
        Apvp = inputs['Apvp']
        land_cost = self.land_cost
        hpp_BOS_soft_cost = self.hpp_BOS_soft_cost
        hpp_grid_connection_cost = self.hpp_grid_connection_cost

        partials['CAPEX_sh', 'G_MW'] = hpp_BOS_soft_cost + hpp_grid_connection_cost
        if (Awpp >= Apvp):
            partials['CAPEX_sh', 'Awpp'] = land_cost
            partials['CAPEX_sh', 'Apvp'] = 0
        else:
            partials['CAPEX_sh', 'Awpp'] = 0
            partials['CAPEX_sh', 'Apvp'] = land_cost
        partials['OPEX_sh', 'G_MW'] = 0
        partials['OPEX_sh', 'Awpp'] = 0
        partials['OPEX_sh', 'Apvp'] = 0


class decommissioning_cost(om.ExplicitComponent):
    """Decommissioning cost model"""

    def __init__(self,
                 decommissioning_cost_w,
                 decommissioning_cost_s,
                 ):
        """Initialization of the decommissioning costs model

        Parameters
        ----------
        decommissioning_cost_w : Decommissioning cost of the wind turbines [Euro/turbine]
        decommissioning_cost_s : Decommissioning cost of the PV [Euro/MW]

        """
        super().__init__()
        self.decommissioning_cost_w = decommissioning_cost_w
        self.decommissioning_cost_s = decommissioning_cost_s

    def setup(self):

        self.add_input('CAPEX_w',
                        desc="CAPEX wpp")
        self.add_input('solar_MW',
                       desc="Solar capacity",
                       units='MW')
        self.add_output('decommissioning_cost_tot_w',
                        desc="Decommissioning cost of the entire wind plant")
        self.add_output('decommissioning_cost_tot_s',
                        desc="Decommissioning cost of the entire PV plant")

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """ Computing the decommissioning costs of the entire wind plant and PV plant.

        Parameters
        ----------
        Nwt : Number of wind turbines
        solar_MW : AC nominal capacity of the PV plant [MW]

        Returns
        -------
        decommissioning_cost_tot_w : Decommissioning cost of the entire wind plant [Eur]
        decommissioning_cost_tot_s : Decommissioning cost of the entire PV plant [Eur]
        """

        CAPEX_w = inputs['CAPEX_w']
        solar_MW = inputs['solar_MW']

        decommissioning_cost_w = self.decommissioning_cost_w
        decommissioning_cost_s = self.decommissioning_cost_s

        outputs['decommissioning_cost_tot_w'] = decommissioning_cost_w * CAPEX_w
        outputs['decommissioning_cost_tot_s'] = decommissioning_cost_s * solar_MW


