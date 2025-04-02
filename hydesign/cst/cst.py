# %%
import time
# basic libraries
import numpy as np
import openmdao.api as om
from scipy.interpolate import RegularGridInterpolator

# Define a class for CST (Concentrated Solar Thermal) component within OpenMDAO framework
class cst(om.ExplicitComponent):
    def __init__(self,
                 N_time,
                 cst_ms_receiver_efficiency_table,
                 wind_speed,
                 # Molten salt
                 Hot_molten_salt_storage_temperature,  # °C
                 Cold_molten_salt_storage_temperature,  # °C
                 hot_molten_salt_density,  # kg/m3
                 Cold_molten_salt_density,  # kg/m3
                 Hot_molten_salt_specific_q,  # kJ/kg/K
                 Cold_molten_salt_specific_q,  # kJ/kg/K
                 heat_exchanger_efficiency,
                 steam_turbine_efficiency,
                 flow_ms_max_cst_receiver_per_m2,
                 ):
        super().__init__()
        self.N_time = N_time
        self.cst_ms_receiver_efficiency_table = cst_ms_receiver_efficiency_table
        self.wind_speed = wind_speed
        self.Hot_molten_salt_storage_temperature = Hot_molten_salt_storage_temperature  # °C
        self.Cold_molten_salt_storage_temperature = Cold_molten_salt_storage_temperature  # °C
        self.hot_molten_salt_density = hot_molten_salt_density  # kg/m3
        self.Cold_molten_salt_density = Cold_molten_salt_density  # kg/m3
        self.Hot_molten_salt_specific_q = Hot_molten_salt_specific_q  # kJ/kg/K
        self.Cold_molten_salt_specific_q = Cold_molten_salt_specific_q  # kJ/kg/K
        self.heat_exchanger_efficiency=heat_exchanger_efficiency
        self.steam_turbine_efficiency=steam_turbine_efficiency
        self.flow_ms_max_cst_receiver_per_m2 = flow_ms_max_cst_receiver_per_m2

    def setup(self):
        # inputs
        self.add_input('area_cst_receiver_m2',
                       desc="area of the cst receiver on the tower",
                       units='m**2',)
        self.add_input(
            'max_solar_flux_cst_t',
            val=0,
            desc="Maximum solar flux towards cst reciever timeseries",
            shape=[self.N_time],
            units='MW'
        )

        # outputs
        self.add_output(
            'flow_ms_max_t',
            desc="Flow of molten salt time series (Gross value, i.e. asuuing all flux_sf_t goes to CSP)",
            units='kg/h',
            shape=[self.N_time]
        )
        self.add_output(
            'delta_q_hot_cold_ms_per_kg',
            desc="Heat (kJ) difference between hot and cold molten salt per kg",
            units='kJ/kg'
        )
        self.add_output(
            'flow_ms_max_cst_receiver_capacity',
            desc="Capacity of the reciever for molten salt flow",
            units='kg/h',
        )


    def compute(self, inputs, outputs):
        area_cst_receiver_m2 = inputs['area_cst_receiver_m2']

        # Retrieve molten salt parameters
        Hot_molten_salt_storage_temperature = self.Hot_molten_salt_storage_temperature  # °C
        Cold_molten_salt_storage_temperature = self.Cold_molten_salt_storage_temperature  # °C
        hot_molten_salt_density = self.hot_molten_salt_density  # kg/m3
        Cold_molten_salt_density = self.Cold_molten_salt_density  # kg/m3
        Hot_molten_salt_specific_q = self.Hot_molten_salt_specific_q  # kJ/kg/K
        Cold_molten_salt_specific_q = self.Cold_molten_salt_specific_q  # kJ/kg/K

        # Calculate energy content per kg for hot and cold molten salt
        q_hot_ms_per_kg = Hot_molten_salt_specific_q * Hot_molten_salt_storage_temperature # [kJ/kg]=[kJ/kg/K]*[K] and T_ref = 0
        q_cold_ms_per_kg = Cold_molten_salt_specific_q * Cold_molten_salt_storage_temperature # kJ/kg and  T_ref = 0
        delta_q_hot_cold_ms_per_kg = (q_hot_ms_per_kg - q_cold_ms_per_kg) # kJ / kg

        # Calculate CST receiver efficiency based on flux and wind speed
        flux_values = self.cst_ms_receiver_efficiency_table['flux_values']
        wind_speed_values = self.cst_ms_receiver_efficiency_table['wind_speed_values']
        efficiency_values = np.array(self.cst_ms_receiver_efficiency_table['efficiency_values'])

        # Interpolator for efficiency based on given flux and wind speed
        efficiency_interpolator = RegularGridInterpolator((flux_values, wind_speed_values), efficiency_values, bounds_error=False, fill_value=None)

        # Compute received heat on the CST receiver (MW)
        flux_cst_t = inputs['max_solar_flux_cst_t']
        cst_efficiency = efficiency_interpolator((flux_cst_t, self.wind_speed))
        q_cst_receiver = cst_efficiency * flux_cst_t # MW

        # Calculate molten salt flow rate (kg/s)
        flow_ms_max_t = 3.6e6 * q_cst_receiver / delta_q_hot_cold_ms_per_kg # [MW]/[kJ/kg]=[MW]/[kWh/(3600*kg)]= 3.6e6 * [kg/h]

        flow_ms_max_cst_receiver_capacity = self.flow_ms_max_cst_receiver_per_m2 * area_cst_receiver_m2

        # Set computed outputs
        outputs['delta_q_hot_cold_ms_per_kg'] = delta_q_hot_cold_ms_per_kg
        outputs['flow_ms_max_t'] = flow_ms_max_t
        outputs['flow_ms_max_cst_receiver_capacity'] = flow_ms_max_cst_receiver_capacity