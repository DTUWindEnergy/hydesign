# Import necessary libraries
import math
import numpy as np
from numpy import newaxis as na
import pandas as pd
from scipy.interpolate import RegularGridInterpolator, interp1d
import xarray as xr
import openmdao.api as om
import pvlib

# Solar Field (sf) Class using OpenMDAO for explicit components
class sf(om.ExplicitComponent):
    def __init__(self,
                 N_time,  # Number of time steps
                 sf_azimuth_altitude_efficiency_table,  # Efficiency table for the solar field
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
        super().__init__()
        self.N_time = N_time
        self.sf_azimuth_altitude_efficiency_table = sf_azimuth_altitude_efficiency_table
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.dni = dni

    def setup(self):
        """
        Sets up the inputs and outputs for the solar field component in OpenMDAO.
        Inputs are solar field area and receiver heights; outputs are maximum fluxes and AOI.
        """
        # Inputs
        self.add_input('sf_area', desc="Area of the solar field in square meters", units='m**2')
        self.add_input('tower_diameter', desc="Diameter of the tower", units='m')
        self.add_input('tower_height', desc="Heigh of the tower", units='m')
        self.add_input('area_cpv_receiver_m2', desc="area_cpv_receiver", units='m**2')
        self.add_input('area_cst_receiver_m2', desc="area_cst_receiver", units='m**2')
        self.add_input('area_dni_reactor_biogas_h2', desc="area_dni_reactor_biogas_h2", units='m**2')

        # Outputs
        self.add_output('flux_sf_t', desc="solar flux from solar field", shape=[self.N_time], units='MW')
        self.add_output('max_solar_flux_cpv_t', val=0, desc="maximum solar flux on cpv reciever time series", shape=[self.N_time], units='MW')
        self.add_output('max_solar_flux_cst_t', val=0, desc="maximum solar flux on cst reciever time series", shape=[self.N_time], units='MW')
        self.add_output('max_solar_flux_biogas_h2_t', val=0, desc="maximum solar flux on biogas_h2 reciever time series", shape=[self.N_time], units='MW')

    def compute(self, inputs, outputs):
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
        # Extract geographical and DNI data
        latitude = self.latitude
        longitude = self.longitude
        altitude = self.altitude
        dni = self.dni
        times = dni.index  # UTC time index for solar position calculation

        # Iterate over each receiver and check if height is within tower limits
        sf_area = inputs['sf_area']
        tower_height = inputs['tower_height']
        tower_diameter = inputs['tower_diameter']
        area_cpv_receiver_m2 = inputs['area_cpv_receiver_m2']
        area_cst_receiver_m2 = inputs['area_cst_receiver_m2']
        area_dni_reactor_biogas_h2 = inputs['area_dni_reactor_biogas_h2']

        cpv_receiver_height = tower_height - 0.5 * area_cpv_receiver_m2 / (math.pi * tower_diameter)
        h2_receiver_height = tower_height - (area_cpv_receiver_m2 + 0.5 * area_dni_reactor_biogas_h2) / (math.pi * tower_diameter)
        cst_receiver_height = tower_height - (area_cpv_receiver_m2 + area_dni_reactor_biogas_h2 + 0.5 * area_cst_receiver_m2) / (math.pi * tower_diameter)

        # Calculate solar position (sun's altitude and azimuth)
        solar_position = pvlib.solarposition.get_solarposition(times, latitude, longitude, altitude)
        sun_altitude = solar_position['apparent_elevation'].clip(lower=0)  # Clip negative altitudes to 0
        sun_azimuth = solar_position['azimuth']

        # Convert the list to a Pandas Series for further calculations
        flux_sf_cpv_t = self.calculate_flux_sf(
                self.sf_azimuth_altitude_efficiency_table,
                cpv_receiver_height,
                sf_area,
                sun_altitude,
                sun_azimuth,
                dni
        )

        flux_sf_cst_t = self.calculate_flux_sf(
                self.sf_azimuth_altitude_efficiency_table,
                cst_receiver_height,
                sf_area,
                sun_altitude,
                sun_azimuth,
                dni
        )
        flux_sf_biogas_h2_t = self.calculate_flux_sf(
                self.sf_azimuth_altitude_efficiency_table,
                h2_receiver_height,
                sf_area,
                sun_altitude,
                sun_azimuth,
                dni
        )

        # Compute maximum flux for cpv, cst, and H2 receivers
        max_solar_flux_cpv_t = flux_sf_cpv_t
        max_solar_flux_cst_t = flux_sf_cst_t
        max_solar_flux_biogas_h2_t = flux_sf_biogas_h2_t

        # Assign computed fluxes to the outputs
        outputs['flux_sf_t'] = flux_sf_cpv_t
        outputs['max_solar_flux_cpv_t'] = max_solar_flux_cpv_t  # MW for cpv
        outputs['max_solar_flux_cst_t'] = max_solar_flux_cst_t  # MW for cst
        outputs['max_solar_flux_biogas_h2_t'] = max_solar_flux_biogas_h2_t  # MW for biogas to H2


    def calculate_flux_sf(self,
                          sf_azimuth_altitude_efficiency_table,
                          tower_height,
                          sf_area,
                          sun_altitude,
                          sun_azimuth,
                          dni
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
        tower_heights = sf_azimuth_altitude_efficiency_table['tower_height']
        sf_areas = sf_azimuth_altitude_efficiency_table['sf_area']
        azimuth_values = sf_azimuth_altitude_efficiency_table['azimuth']
        altitude_values = sf_azimuth_altitude_efficiency_table['altitude']
        efficiency_data = np.array(sf_azimuth_altitude_efficiency_table['efficiency'])

        # Ensure 360° azimuth is included in the efficiency table for interpolation
        if 360 not in azimuth_values:
            azimuth_values = np.append(azimuth_values, 360)
            efficiency_data = np.concatenate(
                [efficiency_data, efficiency_data[:, :, :, 0:1]], axis=3
            )  # Duplicate 0° column as 360°

        # Interpolator for 4D efficiency data: (tower_height, sf_area, altitude, azimuth)
        efficiency_interpolator = RegularGridInterpolator(
            (tower_heights, sf_areas, altitude_values, azimuth_values),
            efficiency_data,
            bounds_error=False,  # Allow extrapolation
            fill_value=None  # Extrapolated values return None
        )

        # Calculate effective flux for the solar field
        effective_flux_sf = []
        for alt, azi, dni_value in zip(sun_altitude, sun_azimuth, dni):
            # Interpolate efficiency for the given tower height, sf area, altitude, and azimuth
            efficiency = efficiency_interpolator((tower_height, sf_area, alt, azi))
            if efficiency is None:
                efficiency = 0  # Default to 0 if extrapolation fails
            effective_flux_sf.append(dni_value * efficiency * sf_area)  # Calculate flux

        # Convert the list to a Pandas Series for further calculations
        flux_sf_t = pd.Series(effective_flux_sf, index=dni.index)

        return flux_sf_t


