# %%
import time

# basic libraries
import numpy as np
from scipy.interpolate import interp1d
import openmdao.api as om

# Define a class for cpv (Concentrated Photovoltaics) component within OpenMDAO framework
class cpv(om.ExplicitComponent):
    def __init__(self,
                 N_time,
                 cpv_efficiency,
                 p_max_cpv_mw_per_m2,
                 ):
        # Initialize the superclass and assign instance variables
        super().__init__()
        self.N_time = N_time
        self.cpv_efficiency = cpv_efficiency
        self.p_max_cpv_mw_per_m2 = p_max_cpv_mw_per_m2

    def setup(self):
        # Define inputs and outputs to the component
        # inputs
        self.add_input(
            'max_solar_flux_cpv_t',
            val=0,
            desc="maximum solar flux on cpv reciever time series",
            shape=[self.N_time],
            units='MW'
        )
        self.add_input('area_cpv_receiver_m2', desc="area_cpv_receiver", units='m**2')
        self.add_input('cpv_dc_ac_ratio',
                       desc="ratio of the dc_ac inverter respect to the rated power of installed cpv",
                       units='MW')

        # outputs
        self.add_output(
            'p_cpv_max_dni_t',
            desc="maximum cpv power time series (Gross value, i.e. assuming all flux_sf_t goes to cpv)",
            units='MW',
            shape=[self.N_time]
        )

        self.add_output(
            'cpv_inverter_mw',
            desc="rated power of the cpv inverter",
            units='MW',
        )
        self.add_output(
            'cpv_rated_mw',
            desc="rated power of the cpv reciever",
            units='MW',
        )

    def compute(self, inputs, outputs):
        area_cpv_receiver_m2 = inputs['area_cpv_receiver_m2']
        cpv_dc_ac_ratio = inputs['cpv_dc_ac_ratio']

        # Compute cpv efficiency based on the cell temperature
        cpv_efficiency = self.cpv_efficiency

        # Calculate cpv power output for each time step
        p_cpv_max_dni_t = cpv_efficiency * inputs['max_solar_flux_cpv_t']

        cpv_rated_mw = area_cpv_receiver_m2 * self.p_max_cpv_mw_per_m2
        cpv_inverter_mw = cpv_dc_ac_ratio * cpv_rated_mw

        # Set output to calculated power time series
        outputs['p_cpv_max_dni_t'] = p_cpv_max_dni_t
        outputs['cpv_inverter_mw'] = cpv_inverter_mw
        outputs['cpv_rated_mw'] = cpv_rated_mw
