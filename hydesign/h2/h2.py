# %%
# Import essential libraries
import time
import numpy as np
import openmdao.api as om
import numpy as np

# Define a class for Biogas to H2 production within OpenMDAO framework
class BiogasH2(om.ExplicitComponent):
    def __init__(self,
                 N_time,
                 heat_mwht_per_kg_h2,
                 biogas_h2_mass_ratio,
                 water_h2_mass_ratio,
                 co2_h2_mass_ratio,
                 ):
        """
        Initializes the BiogasH2 component with necessary parameters and settings.
        """
        super().__init__()
        self.N_time = N_time
        self.biogas_h2_mass_ratio = biogas_h2_mass_ratio
        self.co2_h2_mass_ratio = co2_h2_mass_ratio
        self.water_h2_mass_ratio = water_h2_mass_ratio
        self.heat_mwht_per_kg_h2 = heat_mwht_per_kg_h2

    def setup(self):
        # Define inputs and outputs to the component
        # Inputs
        self.add_input('max_solar_flux_biogas_h2_t', desc="Solar flux from mirrors (MW/m²)", units='MW', shape=[self.N_time])

        # Outputs
        self.add_output('biogas_h2_mass_ratio', desc="biogas_h2_mass_ratio")
        self.add_output('water_h2_mass_ratio', desc="water_h2_mass_ratio")
        self.add_output('co2_h2_mass_ratio', desc="co2_h2_mass_ratio")
        self.add_output('heat_mwht_per_kg_h2', desc="Heat required for producing 1 kg of H2", units='MW*h/kg')
        self.add_output('max_solar_flux_dni_reactor_biogas_h2_t', desc="timeseries of the maximum solar flux on dni reactor of the biogas_h2 module", units='MW', shape=[self.N_time])

    def compute(self, inputs, outputs):
        # Retrieve parameters and input values
        N_time = self.N_time
        biogas_h2_mass_ratio = self.biogas_h2_mass_ratio
        co2_h2_mass_ratio = self.co2_h2_mass_ratio
        water_h2_mass_ratio = self.water_h2_mass_ratio
        heat_mwht_per_kg_h2 = self.heat_mwht_per_kg_h2

        # Solar-driven flux for biogas-to-H2 reaction
        max_solar_flux_dni_reactor_biogas_h2_t = inputs['max_solar_flux_biogas_h2_t']

        # Set outputs to calculated or predefined values
        outputs['biogas_h2_mass_ratio'] = biogas_h2_mass_ratio
        outputs['water_h2_mass_ratio'] = water_h2_mass_ratio
        outputs['heat_mwht_per_kg_h2'] = heat_mwht_per_kg_h2 # neglecting the amount of energy for puryfing the syngas
        outputs['co2_h2_mass_ratio'] = co2_h2_mass_ratio
        outputs['max_solar_flux_dni_reactor_biogas_h2_t'] = max_solar_flux_dni_reactor_biogas_h2_t
