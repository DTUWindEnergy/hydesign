# %%
# Import essential libraries
# import time
# import numpy as np
# import openmdao.api as om
import numpy as np

from hydesign.openmdao_wrapper import ComponentWrapper


# Define a class for Biogas to H2 production within OpenMDAO framework
class BiogasH2:
    def __init__(
        self,
        N_time,
        heat_mwht_per_kg_h2,
        biogas_h2_mass_ratio,
        water_h2_mass_ratio,
        co2_h2_mass_ratio,
    ):
        """
        Initializes the BiogasH2 component with necessary parameters and settings.
        """
        # super().__init__()
        self.N_time = N_time
        self.biogas_h2_mass_ratio = biogas_h2_mass_ratio
        self.co2_h2_mass_ratio = co2_h2_mass_ratio
        self.water_h2_mass_ratio = water_h2_mass_ratio
        self.heat_mwht_per_kg_h2 = heat_mwht_per_kg_h2

        # def setup(self):
        # Define inputs and outputs to the component
        # Inputs
        self.inputs = [
            (
                "max_solar_flux_biogas_h2_t",
                dict(
                    desc="Solar flux from mirrors (MW/m²)",
                    units="MW",
                    shape=[self.N_time],
                ),
            ),
        ]
        # Outputs
        self.outputs = [
            ("biogas_h2_mass_ratio", dict(desc="biogas_h2_mass_ratio")),
            ("water_h2_mass_ratio", dict(desc="water_h2_mass_ratio")),
            ("co2_h2_mass_ratio", dict(desc="co2_h2_mass_ratio")),
            (
                "heat_mwht_per_kg_h2",
                dict(desc="Heat required for producing 1 kg of H2", units="MW*h/kg"),
            ),
            (
                "max_solar_flux_dni_reactor_biogas_h2_t",
                dict(
                    desc="timeseries of the maximum solar flux on dni reactor of the biogas_h2 module",
                    units="MW",
                    shape=[self.N_time],
                ),
            ),
        ]

    def compute(self, **inputs):
        outputs = {}
        # Retrieve parameters and input values
        N_time = self.N_time
        biogas_h2_mass_ratio = self.biogas_h2_mass_ratio
        co2_h2_mass_ratio = self.co2_h2_mass_ratio
        water_h2_mass_ratio = self.water_h2_mass_ratio
        heat_mwht_per_kg_h2 = self.heat_mwht_per_kg_h2

        # Solar-driven flux for biogas-to-H2 reaction
        max_solar_flux_dni_reactor_biogas_h2_t = inputs["max_solar_flux_biogas_h2_t"]

        # Set outputs to calculated or predefined values
        outputs["biogas_h2_mass_ratio"] = biogas_h2_mass_ratio
        outputs["water_h2_mass_ratio"] = water_h2_mass_ratio
        outputs["heat_mwht_per_kg_h2"] = (
            heat_mwht_per_kg_h2  # neglecting the amount of energy for puryfing the syngas
        )
        outputs["co2_h2_mass_ratio"] = co2_h2_mass_ratio
        outputs["max_solar_flux_dni_reactor_biogas_h2_t"] = (
            max_solar_flux_dni_reactor_biogas_h2_t
        )
        out_keys = [
            "biogas_h2_mass_ratio",
            "water_h2_mass_ratio",
            "co2_h2_mass_ratio",
            "heat_mwht_per_kg_h2",
            "max_solar_flux_dni_reactor_biogas_h2_t",
        ]
        return [outputs[key] for key in out_keys]


class BiogasH2_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = BiogasH2(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
