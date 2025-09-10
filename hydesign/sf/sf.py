# Import necessary libraries
import math

import numpy as np

# from numpy import newaxis as na
import pandas as pd

# import xarray as xr
# import openmdao.api as om
import pvlib
from scipy.interpolate import RegularGridInterpolator  # , interp1d

from hydesign.openmdao_wrapper import ComponentWrapper


# Solar Field (sf) Class using OpenMDAO for explicit components
class sf:
    def __init__(
        self,
        N_time,  # Number of time steps
        sf_azimuth_altitude_efficiency_table_cpv,  # Efficiency table for the solar field - cpv
        sf_azimuth_altitude_efficiency_table_cst,  # Efficiency table for the solar field - cst
        sf_azimuth_altitude_efficiency_table_h2,  # Efficiency table for the solar field - h2
        latitude,  # Latitude of the site
        longitude,  # Longitude of the site
        altitude,  # Altitude of the site
        dni,  # Direct normal irradiance time series
    ):
        """
        Initializes the Solar Field component with the provided inputs.
        Uses solar data, efficiency tables, and receiver heights to compute
        maximum fluxes for cpv, cst, and H2 receivers.

        Parameters
        ----------
        N_time: int
            Number of time steps (typically the length of the DNI series).
        sf_azimuth_altitude_efficiency_table: dict
            Efficiency data based on solar position (altitude, azimuth).
        latitude: float
            Geographical latitude of the site.
        longitude: float
            Geographical longitude of the site.
        altitude: float
            Altitude of the site above sea level.
        dni: pd.Series
            Direct normal irradiance (DNI) time series.
        dni_receivers_height_efficiency_table: dict
            Efficiency vs. height for solar receivers.
        """
        # super().__init__()
        self.N_time = N_time
        self.sf_azimuth_altitude_efficiency_table_cpv = (
            sf_azimuth_altitude_efficiency_table_cpv
        )
        self.sf_azimuth_altitude_efficiency_table_cst = (
            sf_azimuth_altitude_efficiency_table_cst
        )
        self.sf_azimuth_altitude_efficiency_table_h2 = (
            sf_azimuth_altitude_efficiency_table_h2
        )
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.dni = dni

        # def setup(self):
        """
        Sets up the inputs and outputs for the solar field component in OpenMDAO.
        Inputs are solar field area and receiver heights; outputs are maximum fluxes and AOI.
        """
        # Inputs
        self.inputs = [
            (
                "sf_area",
                dict(desc="Area of the solar field in square meters", units="m**2"),
            ),
            ("tower_diameter", dict(desc="Diameter of the tower", units="m")),
            ("tower_height", dict(desc="Heigh of the tower", units="m")),
            ("area_cpv_receiver_m2", dict(desc="area_cpv_receiver", units="m**2")),
            ("area_cst_receiver_m2", dict(desc="area_cst_receiver", units="m**2")),
            (
                "area_dni_reactor_biogas_h2",
                dict(desc="area_dni_reactor_biogas_h2", units="m**2"),
            ),
        ]
        # Outputs
        self.outputs = [
            (
                "max_solar_flux_cpv_t",
                dict(
                    val=0,
                    desc="maximum solar flux on cpv reciever time series",
                    shape=[self.N_time],
                    units="MW",
                ),
            ),
            (
                "max_solar_flux_cst_t",
                dict(
                    val=0,
                    desc="maximum solar flux on cst reciever time series",
                    shape=[self.N_time],
                    units="MW",
                ),
            ),
            (
                "max_solar_flux_biogas_h2_t",
                dict(
                    val=0,
                    desc="maximum solar flux on biogas_h2 reciever time series",
                    shape=[self.N_time],
                    units="MW",
                ),
            ),
        ]

    def compute(self, **inputs):
        """
        Computes the solar flux based on solar position, efficiency tables, and receiver heights.
        Uses interpolation to calculate fluxes on cpv, cst, and H2 receivers.

        Parameters
        ----------
        inputs: dict
            Dictionary of input values including solar field area and receiver heights.
        outputs: dict
            Dictionary to store computed maximum fluxes for cpv, cst, and H2 receivers.
        """
        outputs = {}
        # Extract geographical and DNI data
        latitude = self.latitude
        longitude = self.longitude
        altitude = self.altitude
        dni = self.dni
        times = dni.index  # UTC time index for solar position calculation

        # Iterate over each receiver and check if height is within tower limits
        sf_area = inputs["sf_area"]
        tower_height = inputs["tower_height"]
        tower_diameter = inputs["tower_diameter"]
        area_cpv_receiver_m2 = inputs["area_cpv_receiver_m2"]
        area_cst_receiver_m2 = inputs["area_cst_receiver_m2"]
        area_dni_reactor_biogas_h2 = inputs["area_dni_reactor_biogas_h2"]

        concentration_ratio_cpv = (
            sf_area / area_cpv_receiver_m2 if area_cpv_receiver_m2 != 0 else np.inf
        )
        concentration_ratio_cst = (
            sf_area / area_cst_receiver_m2 if area_cst_receiver_m2 != 0 else np.inf
        )
        concentration_ratio_h2 = (
            sf_area / area_dni_reactor_biogas_h2
            if area_dni_reactor_biogas_h2 != 0
            else np.inf
        )

        cpv_receiver_height = tower_height - 0.5 * area_cpv_receiver_m2 / (
            math.pi * tower_diameter
        )
        h2_receiver_height = tower_height - (
            area_cpv_receiver_m2 + 0.5 * area_dni_reactor_biogas_h2
        ) / (math.pi * tower_diameter)
        cst_receiver_height = tower_height - (
            area_cpv_receiver_m2
            + area_dni_reactor_biogas_h2
            + 0.5 * area_cst_receiver_m2
        ) / (math.pi * tower_diameter)

        # Calculate solar position (sun's altitude and azimuth)
        solar_position = pvlib.solarposition.get_solarposition(
            times, latitude, longitude, altitude
        )
        sun_altitude = solar_position["apparent_elevation"].clip(
            lower=0
        )  # Clip negative altitudes to 0
        sun_azimuth = solar_position["azimuth"]

        # Convert the list to a Pandas Series for further calculations

        if area_cpv_receiver_m2 > 0:
            max_solar_flux_cpv_t = self.calculate_flux_sf(
                self.sf_azimuth_altitude_efficiency_table_cpv,
                cpv_receiver_height,
                sf_area,
                concentration_ratio_cpv,
                sun_altitude,
                sun_azimuth,
                dni,
            )

        else:
            max_solar_flux_cpv_t = 0

        if area_cst_receiver_m2 > 0:
            max_solar_flux_cst_t = self.calculate_flux_sf(
                self.sf_azimuth_altitude_efficiency_table_cst,
                cst_receiver_height,
                sf_area,
                concentration_ratio_cst,
                sun_altitude,
                sun_azimuth,
                dni,
            )
        else:
            max_solar_flux_cst_t = 0

        if area_dni_reactor_biogas_h2 > 0:
            max_solar_flux_biogas_h2_t = self.calculate_flux_sf(
                self.sf_azimuth_altitude_efficiency_table_h2,
                h2_receiver_height,
                sf_area,
                concentration_ratio_h2,
                sun_altitude,
                sun_azimuth,
                dni,
            )
        else:
            max_solar_flux_biogas_h2_t = 0

        # Assign computed fluxes to the outputs
        outputs["max_solar_flux_cpv_t"] = max_solar_flux_cpv_t  # MW for cpv
        outputs["max_solar_flux_cst_t"] = max_solar_flux_cst_t  # MW for cst
        outputs["max_solar_flux_biogas_h2_t"] = (
            max_solar_flux_biogas_h2_t  # MW for biogas to H2
        )
        out_keys = [
            "max_solar_flux_cpv_t",
            "max_solar_flux_cst_t",
            "max_solar_flux_biogas_h2_t",
        ]
        return [outputs[key] for key in out_keys]

    def calculate_flux_sf(
        self,
        sf_azimuth_altitude_efficiency_table,
        tower_height,
        sf_area,
        concentration_ratio,
        sun_altitude,
        sun_azimuth,
        dni,
    ):
        """
        Calculate the effective solar flux for a CPV system using efficiency interpolation.

        Parameters:
        - sf_azimuth_altitude_efficiency_table (dict): Efficiency table with azimuth, altitude, tower heights, and sf areas.
        - tower_height (float): The tower height to be used for interpolation.
        - sf_area (float): The solar field area to calculate the flux.
        - sun_altitude (array-like): Array of solar altitude angles.
        - sun_azimuth (array-like): Array of solar azimuth angles.
        - dni (pd.Series): Direct Normal Irradiance (DNI) values with time-based index.

        Returns:
        - pd.Series: Effective flux values (flux_sf_t) with the same index as the `dni`.
        """
        # Extract data for interpolation
        tower_heights = sf_azimuth_altitude_efficiency_table["tower_height"]
        sf_areas = sf_azimuth_altitude_efficiency_table["sf_area"]
        concentration_ratios = sf_azimuth_altitude_efficiency_table[
            "concentration_ratio"
        ]
        azimuth_values = sf_azimuth_altitude_efficiency_table["azimuth"]
        altitude_values = sf_azimuth_altitude_efficiency_table["altitude"]
        efficiency_data = np.array(sf_azimuth_altitude_efficiency_table["efficiency"])

        # Check if 0° or 360° is already present
        azimuth_values = np.array(azimuth_values)
        if 0 not in azimuth_values or 360 not in azimuth_values:
            # Find the index of the azimuth value closest to 0 or 360
            closest_idx = np.argmin(
                np.minimum(np.abs(azimuth_values - 0), np.abs(azimuth_values - 360))
            )

            # Prepare the efficiency slice to duplicate
            duplicate_slice = efficiency_data[..., closest_idx : closest_idx + 1]

            # Add 0° at the beginning and 360° at the end
            if 0 not in azimuth_values:
                azimuth_values = np.concatenate(([0], azimuth_values))
                efficiency_data = np.concatenate(
                    [duplicate_slice, efficiency_data], axis=4
                )
            if 360 not in azimuth_values:
                azimuth_values = np.concatenate((azimuth_values, [360]))
                efficiency_data = np.concatenate(
                    [efficiency_data, duplicate_slice], axis=4
                )

        # Interpolator for 5D efficiency data: (tower_height, sf_area, concentration_ratio, altitude, azimuth)
        efficiency_interpolator = RegularGridInterpolator(
            (
                tower_heights,
                sf_areas,
                concentration_ratios,
                altitude_values,
                azimuth_values,
            ),
            efficiency_data,
            bounds_error=False,  # Allow extrapolation
            fill_value=None,  # Extrapolated values return None
        )

        # Calculate effective flux for the solar field
        effective_flux_sf = []
        for alt, azi, dni_value in zip(sun_altitude, sun_azimuth, dni):
            # Interpolate efficiency for the given tower height, sf area, altitude, and azimuth
            efficiency = efficiency_interpolator(
                (tower_height, sf_area, concentration_ratio, alt, azi)
            )
            if efficiency is None:
                efficiency = 0  # Default to 0 if extrapolation fails
            effective_flux_sf.append(dni_value * efficiency * sf_area)  # Calculate flux

        # Convert the list to a Pandas Series for further calculations
        flux_sf_t = pd.Series(effective_flux_sf, index=dni.index)

        return flux_sf_t


class sf_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = sf(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
