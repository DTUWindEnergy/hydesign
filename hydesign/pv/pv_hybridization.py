# %%
import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import pandas as pd
import xarray as xr
import openmdao.api as om
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# pvlib imports
import pvlib
from pvlib import pvsystem, tools, irradiance, atmosphere
from pvlib.location import Location
from pvlib.modelchain import ModelChain
from pvlib import temperature
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS


class pvp_with_degradation(om.ExplicitComponent):
    """
    PV degradation model providing the PV degradation time series throughout the lifetime of the plant
    """

    def __init__(
            self,
            N_limit,
            life_y,
            life_h,
            pv_deg=[0, 25 * 1 / 100],
    ):
        """Initialization of the PV degradation model

        Parameters
        ----------
        life_h : lifetime of the plant

        """
        super().__init__()

        self.N_limit = N_limit
        self.life_y = life_y
        self.life_h = life_h

        # PV degradation curve
        self.pv_deg = pv_deg

    def setup(self):
        self.add_input('delta_life',
                       desc="Years between the starting of operations of the existing plant and the new plant",
                       val=1)

        self.add_input(
            'solar_t_ext',
            desc="PVP power time series",
            units='MW',
            shape=[self.life_h])

        self.add_output(
            'solar_t_ext_deg',
            desc="PVP power time series with degradation",
            units='MW',
            shape=[self.life_h])

    def compute(self, inputs, outputs):
        N_limit = self.N_limit
        life_y = self.life_y
        pv_deg = self.pv_deg

        delta_life = inputs['delta_life'][0]
        solar_t_ext = inputs['solar_t_ext']

        t_over_year = np.arange(self.life_h) / (365 * 24)
        pv_deg_yr = [0, int(delta_life), int(delta_life) + 0.0001, int(delta_life) + 25, int(delta_life) + 25.0001,
                     int(life_y) + int(+N_limit)]
        degradation = np.interp(t_over_year, pv_deg_yr, pv_deg)

        outputs['solar_t_ext_deg'] = (1 - degradation) * solar_t_ext


class existing_pvp(om.ExplicitComponent):
    """
    PV power plant model : It computes the solar power output during the lifetime of the plant using data from the datasheet

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    inverter_eff_curve: inverter efficiency curve that is imported.
    pdc0 : Module nominal power
    v_mp : Module maximum power voltage
    i_mp : Module maximum power current
    v_oc : Module open circuit voltage
    i_sc : Module short circuit current
    alpha_sc_spec : short circuit temperature coefficient
    beta_voc_spec : open circuit temperature coefficient
    gamma_pdc : power temperature coefficient
    cells_in_series : number of cells in series
    temp_ref : reference temperature (25°)
    celltype : type of cell (mono/poly Si)
    panel : type of panel (monofacial/bifacial)
    tracking : fixed/single-axis

    pac0_inv : Inverter AC nominal power
    eta_max : maximum inverter efficiency
    eta_euro : european inverter efficiency

    modules_per_string : number of modules per string
    strings_per_inverter : number of strings per inverter
    number_of_inverters : number of inverters

    soiling : losses due to soiling (%)
    shading : losses due to shading (%)
    snow : losses due to snow (%)
    mismatch : lossesdue to mismatch (%)
    wiring : losses due to wiring (%)
    connections : losses due to connections (%)
    lid : losses due to lid (%)
    nameplate_rating : losses due to nameplate rating (%)
    age : losses due to age
    availability : losses due to availability (%)
    """

    def __init__(self,
                 weather_fn,
                 N_time,
                 latitude,
                 longitude,
                 altitude,
                 inverter_eff_curve,

                 pdc0,
                 v_mp,
                 i_mp,
                 v_oc,
                 i_sc,
                 alpha_sc_spec,
                 beta_voc_spec,
                 gamma_pdc,
                 cells_in_series,
                 temp_ref,
                 celltype,
                 panel,
                 tracking,

                 pac0_inv,
                 eta_max,
                 eta_euro,

                 modules_per_string,
                 strings_per_inverter,
                 number_of_inverters,

                 soiling,
                 shading,
                 snow,
                 mismatch,
                 wiring,
                 connections,
                 lid,
                 nameplate_rating,
                 age,
                 availability,
                 ):
        """Initialization of the PV power plant model

        Parameters
        ----------
        weather_fn : Weather timeseries
        N_time : Length of the representative data
        latitude : Latitude at chosen location
        longitude : Longitude at chosen location
        altitude : Altitude at chosen location


        """
        super().__init__()
        self.weather_fn = weather_fn
        self.N_time = N_time
        self.latitude = latitude
        self.longitude = longitude
        self.altitude = altitude
        self.inverter_eff_curve = inverter_eff_curve

        self.pdc0 = pdc0
        self.v_mp = v_mp
        self.i_mp = i_mp
        self.i_sc = i_sc
        self.v_oc = v_oc
        self.alpha_sc_spec = alpha_sc_spec
        self.beta_voc_spec = beta_voc_spec
        self.gamma_pdc = gamma_pdc
        self.cells_in_series = cells_in_series
        self.temp_ref = temp_ref
        self.celltype = celltype
        self.panel = panel
        self.tracking = tracking

        self.pac0_inv = pac0_inv
        self.eta_max = eta_max
        self.eta_euro = eta_euro

        self.modules_per_string = modules_per_string
        self.strings_per_inverter = strings_per_inverter
        self.number_of_inverters = number_of_inverters

        self.soiling = soiling
        self.shading =shading
        self.snow = snow
        self.mismatch = mismatch
        self.wiring = wiring
        self.connections = connections
        self.lid = lid
        self.nameplate_rating = nameplate_rating
        self.age = age
        self.availability = availability

        pvloc = Location(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            name='Plant')

        weather = pd.read_csv(
            weather_fn,
            index_col=0,
            parse_dates=True)

        weather['temp_air'] = weather['temp_air_1'] - 273.15  # Celcium
        weather['wind_speed'] = weather['WS_1']

        self.weather = weather
        self.pvloc = pvloc

    def setup(self):
        self.add_input(
            'surface_tilt',
            val=20,
            desc="Solar PV tilt angle in degs")

        self.add_input(
            'surface_azimuth',
            val=180,
            desc="Solar PV azimuth angle in degs, 180=south facing")

        self.add_input(
            'DC_AC_ratio',
            desc="DC/AC PV ratio")

        self.add_input(
            'land_use_per_solar_MW',
            val=1,
            desc="Solar land use per solar MW",
            units='km**2/MW')

        self.add_output(
            'solar_t',
            desc="PV power time series",
            units='MW',
            shape=[self.N_time])

        self.add_output(
            'solar_MW',
            val=1,
            desc="Solar PV plant installed capacity",
            units='MW')

        self.add_output(
            'Apvp',
            desc="Land use area of WPP",
            units='km**2')

    def compute(self, inputs, outputs):
        surface_tilt = inputs['surface_tilt']
        surface_azimuth = inputs['surface_azimuth']
        land_use_per_solar_MW = inputs['land_use_per_solar_MW'][0]
        DC_AC_ratio = inputs['DC_AC_ratio']

        # Definition of some missing parameters for the module
        alpha_sc = self.alpha_sc_spec * self.i_sc  # cannot be calculated here
        beta_voc = self.beta_voc_spec * self.v_oc  # cannot be calculated here

        modules_per_string = self.modules_per_string
        strings_per_inverter = self.strings_per_inverter
        number_of_inverters = self.number_of_inverters

        # existing_inverter_efficiency_curve = self.inverter_eff_curve

        # Creation of a dcitionary with the module parameters to pass it to the function
        module_parameters = {
            'pdc0': self.pdc0,
            'v_mp': self.v_mp,
            'i_mp': self.i_mp,
            'v_oc': self.v_oc,
            'i_sc': self.i_sc,
            'alpha_sc': alpha_sc,
            'beta_voc': beta_voc,
            'gamma_pdc': self.gamma_pdc,
            'cells_in_series': self.cells_in_series,
            'temp_ref': self.temp_ref,
            'celltype': self.celltype,
            'panel': self.panel,
            'tracking': self.tracking,
        }

        # Creation of a dictionary with the inverter parameters to pass it to the function
        inverter_parameters = {
            'pac0_inv': self.pac0_inv,
            'eta_max': self.eta_max,
            'eta_euro': self.eta_euro,
        }

        # Creation of a dictionary with the losses parameters to pass it to the function
        losses_parameters = {
            'soiling': self.soiling,
            'shading': self.shading,
            'snow': self.snow,
            'mismatch': self.mismatch,
            'wiring': self.wiring,
            'connections': self.connections,
            'lid': self.lid,
            'nameplate_rating': self.nameplate_rating,
            'age': self.age,
            'availability': self.availability,
        }

        solar_t, solar_MW = get_solar_time_series_existing_pv(

            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth,
            land_use_per_solar_MW=land_use_per_solar_MW,
            DC_AC_ratio=DC_AC_ratio,

            module_parameters=module_parameters,
            inverter_parameters=inverter_parameters,
            losses_parameters=losses_parameters,
            existing_inverter_efficiency_curve=self.inverter_eff_curve,

            modules_per_string=modules_per_string,
            strings_per_inverter=strings_per_inverter,
            number_of_inverters=number_of_inverters,

            pvloc=self.pvloc,
            weather=self.weather)

        Apvp = solar_MW * land_use_per_solar_MW  # We need it in km**2 for the cost model
        outputs['solar_MW'] = solar_MW
        outputs['solar_t'] = solar_t
        outputs['Apvp'] = Apvp


# class existing_pvp_with_degradation(om.ExplicitComponent):
#     """
#     PV degradation model providing the PV degradation time series throughout the lifetime of the plant
#     """

#     def __init__(
#             self,
#             life_h,
#             pv_deg_yr,
#             pv_deg=[0, 25 * 1 / 100],
#     ):
#         """Initialization of the PV degradation model

#         Parameters
#         ----------
#         life_h : lifetime of the plant

#         """
#         super().__init__()
#         self.life_h = life_h

#         # PV degradation curve
#         self.pv_deg_yr = pv_deg_yr
#         self.pv_deg = pv_deg

#     def setup(self):
#         self.add_input(
#             'solar_t_ext',
#             desc="PVP power time series",
#             units='MW',
#             shape=[self.life_h])

#         self.add_output(
#             'solar_t_ext_deg',
#             desc="PVP power time series with degradation",
#             units='MW',
#             shape=[self.life_h])

#     def compute(self, inputs, outputs):
#         solar_t_ext = inputs['solar_t_ext']
#         t_over_year = np.arange(self.life_h) / (365 * 24)
#         degradation = np.interp(t_over_year, self.pv_deg_yr, self.pv_deg)

#         outputs['solar_t_ext_deg'] = (1 - degradation) * solar_t_ext
    
class pvp_degradation_linear(om.ExplicitComponent):
    """
    PV degradation model providing the PV degradation time series throughout the lifetime of the plant, 
    considering a fixed linear degradation of the PV panels
    """
    def __init__(self, life_h):
        """Initialization of the PV degradation model

        Parameters
        ----------
        life_h : lifetime of the plant

        """ 
        super().__init__()
        self.life_h = life_h
        
    def setup(self):
        self.add_input('pv_deg_per_year', desc="PV degradation per year", val=0.5 / 100)
        self.add_output('SoH_pv', desc="PV state of health time series", shape=[self.life_h])   

    def compute(self, inputs, outputs):
        pv_deg_per_year = inputs['pv_deg_per_year']
        outputs['SoH_pv'] = get_linear_solar_degradation(pv_deg_per_year, self.life_h)   

class shadow(om.ExplicitComponent):
    """pv loss model due to shadows of wt"""

    # TODO implement degradation model in pcw
    # 1. Add input for:
    #    - turbine locations x_wt, y_wt in lat long
    #    - Pv locations
    #    - Altitude at the site
    # 2. Compute sun poisition:
    #    - sun position
    #    - simple wt shadow model to estimate covered area
    # 3. Estimate efficiency_t due to shadows

    def __init__(self, N_time):
        super().__init__()
        self.N_time = N_time

    def setup(self):
        self.add_input('solar_deg_t',
                       desc="PV power time series with degradation",
                       units='W',
                       shape=[self.N_time])
        self.add_output(
            'solar_deg_shad_t',
            desc="PV power time series with degradation and shadow losses",
            units='W',
            shape=[
                self.N_time])

    # def setup_partials(self):
    #    self.declare_partials('*', '*',  method='fd')

    def compute(self, inputs, outputs):

        solar_deg_t = inputs['solar_deg_t']
        outputs['solar_deg_shad_t'] = solar_deg_t




# -----------------------------------------------------------------------
# Auxiliar functions 
# -----------------------------------------------------------------------

def get_solar_time_series(
        surface_tilt,
        surface_azimuth,
        solar_MW,
        land_use_per_solar_MW,
        DC_AC_ratio,
        tracking,
        pvloc,
        weather):
    """ Computing the output power time series of the PV plant

    Parameters
    ----------
    surface_tilt : surface tilt of the PV panels
    surface_azimuth : azimuth of the PV panels
    DC_AC_ratio : DC-AC ratio of the PV converter
    solar_MW : AC nominal capacity of the PV power plant

    Returns
    -------
    solar_t : PV power time series
    """

    # Sandia
    sandia_modules = pvsystem.retrieve_sam('SandiaMod')
    module_name = 'Canadian_Solar_CS5P_220M___2009_'
    module = sandia_modules[module_name]
    module['aoi_model'] = irradiance.aoi

    # 2. Inverter
    # -------------
    inverters = pvsystem.retrieve_sam('cecinverter')
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

    temp_model = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    if tracking == 'single_axis':

        mount = pvsystem.SingleAxisTrackerMount(
            axis_tilt=float(surface_tilt),
            axis_azimuth=float(surface_azimuth),
            max_angle=90.0,
            backtrack=True,
            gcr=0.2857142857142857,
            cross_axis_tilt=0.0,
            # module_height = 1
        )
        array = pvsystem.Array(
            mount=mount,
            module_parameters=module,
            temperature_model_parameters=temp_model)
        system = pvsystem.PVSystem(
            arrays=[array],
            inverter_parameters=inverter,
        )
    else:
        system = pvsystem.PVSystem(
            module_parameters=module,
            inverter_parameters=inverter,
            temperature_model_parameters=temp_model,
            surface_tilt=surface_tilt,
            surface_azimuth=surface_azimuth)

    mc = ModelChain(system, pvloc)

    # Run solar with the WRF weather
    mc.run_model(weather)

    DC_AC_ratio_ref = inverter.Pdco / inverter.Paco
    Paco = inverter.Paco * DC_AC_ratio_ref / DC_AC_ratio
    solar_t = (mc.results.ac / Paco)

    solar_t[solar_t > 1] = 1
    solar_t[solar_t < 0] = 0
    return solar_MW * solar_t.fillna(0.0)


def get_solar_time_series_existing_pv(
        surface_tilt,
        surface_azimuth,
        land_use_per_solar_MW,
        DC_AC_ratio,

        module_parameters,
        inverter_parameters,
        losses_parameters,
        existing_inverter_efficiency_curve,

        modules_per_string,
        strings_per_inverter,
        number_of_inverters,

        pvloc,
        weather,
        gcr=0.2857142857142857,
        plot=False
        ):

    """ Computing the output power time series of the existing PV plant

    Parameters
    ----------


    Returns
    -------
    solar_t : PV power time series 
    """

    times = weather.index

    # Get solar position data
    solar_position = pvloc.get_solarposition(times)

    # Option to get tilt and azimuth for single-axis or fixed
    max_phi = 60
    if module_parameters['tracking'] == 'single_axis':
        # Determine the rotation angle of a single-axis tracker when given particular solar zenith and azimuth angles.
        sat = pvlib.tracking.singleaxis(apparent_zenith=solar_position['apparent_zenith'],
                                        apparent_azimuth=solar_position['azimuth'],
                                        max_angle=max_phi,
                                        backtrack=True,
                                        gcr=gcr)
        surface_tilt_val = sat['surface_tilt'].fillna(0)
        surface_azimuth_val = sat['surface_azimuth'].fillna(0)
    else:
        surface_tilt_val = surface_tilt
        surface_azimuth_val = surface_azimuth

    # Calculation with pvfactors of both front and rear-side absorbed irradiance
    # axis_azimuth = 180
    # pvrow_height = 1
    # pvrow_width = 1.1
    # albedo = 0.2

    # Option to get the transposed irradiance on the tilted panel for a bifacial or a monofacial
    if module_parameters['panel'] == 'bifacial':
        # total_irradiance_bif = pvfactors_timeseries(solar_position['azimuth'],
        #                                             solar_position['apparent_zenith'],
        #                                             surface_azimuth_val,
        #                                             surface_tilt_val,
        #                                             axis_azimuth,
        #                                             weather.index,
        #                                             weather['dni'],
        #                                             weatther['dhi'],
        #                                             gcr,
        #                                             pvrow_height,
        #                                             pvrow_width,
        #                                             albedo,
        #                                             n_pvrows=2,
        #                                             index_observed_pvrow=1
        #                                             )
        # total_irradiance_bif = pd.concat(irrad, axis=1)

        # # using bifaciality factor and pvfactors results, create effective irradiance
        # bifaciality = 0.75
        # effective_irrad_bif = irrad['total_abs_front'] + (irrad['total_abs_back'] * bifaciality)
        # irrad = effective_irrad_bif
        raise Warning('bifacial panel is not implemented')
        irrad = None

    else:
        total_irradiance = irradiance.get_total_irradiance(
            surface_tilt=surface_tilt_val,
            surface_azimuth=surface_azimuth_val,
            solar_zenith=solar_position['apparent_zenith'],
            solar_azimuth=solar_position['azimuth'],
            dni=weather['dni'],
            ghi=weather['ghi'],
            dhi=weather['dhi'],
            dni_extra=irradiance.get_extra_radiation(times),
        )
        irrad = total_irradiance['poa_global']

    # Temperature modelling
    temp_cell = temperature.faiman(irrad, weather['temp_air'], weather['wind_speed'])

    # Calculatio of the CEC parameters of the module in reference consitions
    I_L_ref, I_o_ref, R_s, R_sh_ref, a_ref, Adjust = pvlib.ivtools.sdm.fit_cec_sam(module_parameters['celltype'],
                                                                                   v_mp=module_parameters['v_mp'],
                                                                                   i_mp=module_parameters['i_mp'],
                                                                                   v_oc=module_parameters['v_oc'],
                                                                                   i_sc=module_parameters['i_sc'],
                                                                                   alpha_sc=module_parameters['alpha_sc'],
                                                                                   beta_voc=module_parameters['beta_voc'],
                                                                                   gamma_pmp=module_parameters['gamma_pdc'] * 100,
                                                                                   cells_in_series=module_parameters['cells_in_series'],
                                                                                   temp_ref=module_parameters['temp_ref'])

    # Calculation of the Single Diode Model currents
    cec_param = pvlib.pvsystem.calcparams_cec(irrad,  # It returns the CEC parameters
                                              temp_cell,
                                              module_parameters['alpha_sc'],
                                              a_ref,
                                              I_L_ref,
                                              I_o_ref,
                                              R_sh_ref,
                                              R_s,
                                              Adjust)

    # Maximum Power Point parameteres
    mpp = pvlib.pvsystem.max_power_point(*cec_param, method='Newton')

    # Assembling of the system and scaling of power, current, voltage
    system = pvlib.pvsystem.PVSystem(modules_per_string=modules_per_string, strings_per_inverter=strings_per_inverter)

    dc_scaled_values = system.scale_voltage_current_power(mpp)

    dc_losses = pvlib.pvsystem.pvwatts_losses(soiling=losses_parameters['soiling'], shading=losses_parameters['shading'], snow=losses_parameters['snow'], mismatch=losses_parameters['mismatch'], wiring=losses_parameters['wiring'], connections=losses_parameters['connections'],
                                              lid=losses_parameters['lid'], nameplate_rating=losses_parameters['nameplate_rating'], age=losses_parameters['age'], availability=losses_parameters['availability']) / 100  # [%]

    dc_scaled_no_inv = dc_scaled_values.p_mp * (1 - dc_losses)


    # Interpolation of the efficiency curve
    # existing_inverter_efficiency_curve.columns = ['Pdc0','eff']
    efficiency_curve = np.interp(dc_scaled_no_inv, existing_inverter_efficiency_curve[:, 0], existing_inverter_efficiency_curve[:, 1])
    ac_scaled_no_inv = pvlib.inverter.pvwatts(pdc=dc_scaled_no_inv,
                                       pdc0=inverter_parameters['pac0_inv']/inverter_parameters['eta_max'],
                                       eta_inv_nom=efficiency_curve,
                                       eta_inv_ref=0.9637)

    dc_scaled = dc_scaled_no_inv * number_of_inverters
    ac_scaled = ac_scaled_no_inv * number_of_inverters

    solar_MW = inverter_parameters['pac0_inv'] * number_of_inverters/ 1000000 # [MW]
    solar_t = ac_scaled/1000000
    
    if plot:
        plt.figure(figsize=[10,5])
        plt.plot(dc_scaled/1000000, label='DC')
        plt.plot(ac_scaled/1000000, label='AC')
        plt.axhline(y=7.48, color='r', linestyle='--', label='Grid capacity')
        plt.legend()
        plt.title('DC and AC power along the year')
        plt.xlabel('Time [Hours]')
        plt.ylabel('Power [MW]')
        plt.show()
    
        plt.figure(figsize=[10,5])
        plt.plot(dc_scaled[24*180:24*187]/1000000, label='DC')
        plt.plot(ac_scaled[24*180:24*187]/1000000, label='AC')
        plt.legend()
        plt.title('DC and AC power along one week')
        plt.xlabel('Time [Hours]')
        plt.ylabel('Power [MW]')
        plt.gca().xaxis.set_major_locator(mdates.DayLocator())
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
        plt.show()



    return solar_t, solar_MW


def get_linear_solar_degradation(pv_deg_per_year, life_h):
    """ 
    Computes the PV degradation

    Parameters
    ----------
    pv_deg_per_year : fixed yearly degradation of PV panels
    life_h : lifetime of the plant in hours

    Returns
    -------
    SoH_pv : degradation of the PV plant throughout the lifetime
    """
    t_over_year = np.arange(life_h)/(365*24)
    degradation = pv_deg_per_year * t_over_year

    y = 1 - degradation
    if len(y[y < 0]) > 0:
        y[y < 0] = 0  # No replacement of PV panels
    return y