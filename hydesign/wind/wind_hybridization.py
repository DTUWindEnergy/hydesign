# %%

# basic libraries
import numpy as np
# import scipy as sp
import xarray as xr
import openmdao.api as om

# import statistics as st

# from hydesign.look_up_tables import lut_filepath
from hydesign.ems.ems import expand_to_lifetime
from hydesign.wind.wind import get_wind_ts, get_Dws, get_shifted_pc


class wpp_with_degradation(om.ExplicitComponent):
    """
    Wind power plant model

    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    life_h : lifetime in hours
    N_ws : number of points in the power curves
    wpp_efficiency : WPP efficiency
    wind_deg_yr : year list for providing WT degradation curve
    wind_deg : degradation losses at yr
    share_WT_deg_types : share ratio between two degradation mechanism (0: only shift in power curve, 1: degradation as a loss factor )
    ws : Power curve wind speed list
    pcw : Wake affected power curve
    wst : wind speed time series at the hub height

    Returns
    -------
    wind_t_ext_deg : power time series with degradation extended through lifetime

    """

    def __init__(
        self,
        N_limit,
        life_y,
        N_time,
        life_h,
        N_ws = 51,
        wpp_efficiency = 0.95,
        wind_deg=[0, 25 * 1 / 100],
        share_WT_deg_types=0.5,
        weeks_per_season_per_year = None,
        ):
        super().__init__()
        self.N_limit = N_limit
        self.life_y = life_y
        self.N_time = N_time
        self.life_h = life_h
        # number of points in the power curves
        self.N_ws = N_ws
        self.wpp_efficiency = wpp_efficiency
        
        # number of elements in WT degradation curve
        self.wind_deg = wind_deg
        self.share_WT_deg_types = share_WT_deg_types

        # In case data is provided as weeks per season
        self.weeks_per_season_per_year = weeks_per_season_per_year
        
    def setup(self):
        self.add_input('delta_life',
                       desc="Years between the starting of operations of the existing plant and the new plant",
                       val=1)
        self.add_input('ws',
                       desc="Turbine's ws",
                       units='m/s',
                       shape=[self.N_ws])
        self.add_input('pcw',
                       desc="Wake affected power curve",
                       shape=[self.N_ws])
        self.add_input('wst',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.N_time])

        self.add_output('wind_t_ext_deg',
                        desc="power time series with degradation",
                        units='MW',
                        shape=[self.life_h])


    def compute(self, inputs, outputs):
        
        ws = inputs['ws']
        pcw = inputs['pcw']
        wst = inputs['wst']
        delta_life = inputs['delta_life'][0]

        N_limit = self.N_limit
        life_y = self.life_y

        wind_deg_yr = [0, int(delta_life), int(delta_life) + 0.0001, int(delta_life) + 25, int(delta_life) + 25.0001,
                     int(life_y) + int(+N_limit)]

        wst_ext = expand_to_lifetime(
            wst, life = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        
        outputs['wind_t_ext_deg'] = self.wpp_efficiency*get_wind_ts_degradation(
            ws = ws, 
            pc = pcw, 
            ws_ts = wst_ext, 
            yr = wind_deg_yr,
            wind_deg=self.wind_deg, 
            life_h = self.life_h, 
            share = self.share_WT_deg_types)


class existing_wpp(om.ExplicitComponent):
    """
    Wind power plant model for an existing layout

    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    existing_wpp_power_curve_xr_fn: File name of a netcdf xarray. 
            
    The xarray should include 'P_no_wake' as function of 'ws' and 'wake_losses' as a function of 'ws' and 'wd'.
    Note that the wd must include both 0 and 360, and a large WS (for interpolation). 
    Resolution of ws and wd is flexible.
    
    <xarray.Dataset>
    Dimensions:          (ws: 53, wd: 361)
    Coordinates:
      * ws               (ws) float64 0.0 0.5 1.0 1.5 2.0 ... 24.5 25.0 25.0 100.0
      * wd               (wd) float64 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0 360.0
    Data variables:
        wake_losses_eff  (ws, wd) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
        P_no_wake        (ws) float64 0.0 0.0 0.0 0.0 0.0 ... 100.0 100.0 0.0 0.0
            
    wpp_efficiency : WPP efficiency
    wst : wind speed time series at the hub height
    wdt : wind direction time series at the hub height

    Returns
    -------
    wind_t_ext_deg : power time series with degradation extended through lifetime

    """

    def __init__(
        self, 
        N_time,
        existing_wpp_power_curve_xr_fn, 
        wpp_efficiency = 0.95,
        ):
        
        super().__init__()
        self.N_time = N_time
        self.wpp_efficiency = wpp_efficiency

        self.existing_wpp_power_curve_xr_fn = existing_wpp_power_curve_xr_fn
        
        
    def setup(self):
        self.add_input('wst',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.N_time])
        self.add_input('wdt',
                       desc="wd time series at the hub height",
                       units='deg',
                       shape=[self.N_time])

        self.add_output('wind_t',
                        desc="power time series at the hub height",
                        units='MW',
                        shape=[self.N_time])


    def compute(self, inputs, outputs):

        N_time = self.N_time
        # wpp_efficiency = self.wpp_efficiency
        
        wst = inputs['wst']
        wdt = inputs['wdt']

        # Calculation of the mode of wdt
        # mode_wdt = st.mode(wdt)

        existing_wpp_power_curve_xr = xr.open_dataset(self.existing_wpp_power_curve_xr_fn, engine='h5netcdf')

        xr_time = xr.Dataset()
        xr_time['wst'] = xr.DataArray( 
            data = wst,
            dims = ['t'],
            coords = {'t':np.arange(N_time)})
        xr_time['wdt'] = xr.DataArray( 
            data = wdt,
            dims = ['t'],
            coords = {'t':np.arange(N_time)})

        wake_losses_eff_t = existing_wpp_power_curve_xr.wake_losses_eff.interp(ws=xr_time.wst, wd=xr_time.wdt).values


        # ws = existing_wpp_power_curve_xr.ws
        # pc = existing_wpp_power_curve_xr.P_no_wake
        # wake_ratio = existing_wpp_power_curve_xr.wake_losses_eff.sel(wd=mode_wdt, method='nearest') #For the average wind direction
        # pcw = pc * wake_ratio


        wind_t_no_wake = get_wind_ts(
            ws = existing_wpp_power_curve_xr.ws.values,
            pcw = existing_wpp_power_curve_xr.P_no_wake.values,
            wst = wst,
            wpp_efficiency = self.wpp_efficiency,
        )
        
        outputs['wind_t'] = wake_losses_eff_t * wind_t_no_wake

class existing_wpp_with_degradation(om.ExplicitComponent):
    """
    

    Wind power plant model for an existing layout
    Provides the wind power time series using wake affected power curve and the wind speed time series.

    Parameters
    ----------
    N_time : Number of time-steps in weather simulation
    life_h : lifetime in hours
    existing_wpp_power_curve_xr_fn : File name of a netcdf xarray. 

    The xarray should include 'P_no_wake' as function of 'ws' and 'wake_losses' as a function of 'ws' and 'wd'.
    Note that the wd must include both 0 and 360, and a large WS (for interpolation). 
    Resolution of ws and wd is flexible.

    <xarray.Dataset>
    Dimensions:          (ws: 53, wd: 361)
    Coordinates:
      * ws               (ws) float64 0.0 0.5 1.0 1.5 2.0 ... 24.5 25.0 25.0 100.0
      * wd               (wd) float64 0.0 1.0 2.0 3.0 ... 357.0 358.0 359.0 360.0
    Data variables:
        wake_losses_eff  (ws, wd) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
        P_no_wake        (ws) float64 0.0 0.0 0.0 0.0 0.0 ... 100.0 100.0 0.0 0.0           

    wpp_efficiency : WPP efficiency
    wind_deg_yr : year list for providing WT degradation curve
    wind_deg : degradation losses at yr
    share_WT_deg_types : share ratio between two degradation mechanism (0: only shift in power curve, 1: degradation as a loss factor )
    ws : Power curve wind speed list
    pcw : Wake affected power curve
    wst : wind speed time series at the hub height

    Returns
    -------
    wind_t_ext_deg : power time series with degradation extended through lifetime

    """

    def __init__(
        self,
        life_h,
        N_time,
        existing_wpp_power_curve_xr_fn, 
        wpp_efficiency = 0.95,
        wind_deg_yr = [0, 25],
        wind_deg = [0, 25*1/100],
        share_WT_deg_types = 0.5,
        weeks_per_season_per_year = None,
        ):
        
        super().__init__()
        self.life_h = life_h
        self.N_time = N_time
        self.wpp_efficiency = wpp_efficiency

        self.existing_wpp_power_curve_xr_fn = existing_wpp_power_curve_xr_fn
        
        # number of elements in WT degradation curve
        self.wind_deg_yr = wind_deg_yr
        self.wind_deg = wind_deg
        self.share_WT_deg_types = share_WT_deg_types

        # In case data is provided as weeks per season
        self.weeks_per_season_per_year = weeks_per_season_per_year
        
    def setup(self):

        self.add_input('wst',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.N_time])
        self.add_input('wdt',
                       desc="wd time series at the hub height",
                       units='deg',
                       shape=[self.N_time])

        self.add_output('wst_ext',
                       desc="ws time series at the hub height",
                       units='m/s',
                       shape=[self.life_h])
        self.add_output('wdt_ext',
                       desc="wd time series at the hub height",
                       units='deg',
                       shape=[self.life_h] )
        self.add_output('wind_t_ext_deg',
                        desc="power time series with degradation",
                        units='MW',
                        shape=[self.life_h])


    def compute(self, inputs, outputs):

        # N_time = self.N_time
        life_h = self.life_h
        wpp_efficiency = self.wpp_efficiency

        existing_wpp_power_curve_xr = xr.open_dataset(self.existing_wpp_power_curve_xr_fn, engine='h5netcdf')
        
        # number of elements in WT degradation curve
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types

        # In case data is provided as weeks per season
        weeks_per_season_per_year = self.weeks_per_season_per_year
        
        wst = inputs['wst']
        wst_ext = expand_to_lifetime(
            wst, life = life_h, weeks_per_season_per_year = weeks_per_season_per_year)

        wdt = inputs['wdt']
        wdt_ext = expand_to_lifetime(
            wdt, life = life_h, weeks_per_season_per_year = weeks_per_season_per_year)

        xr_time = xr.Dataset()
        xr_time['wst'] = xr.DataArray( 
            data=wst_ext, 
            dims = ['t'],
            coords = {'t':np.arange(life_h)})
        xr_time['wdt'] = xr.DataArray( 
            data=wdt_ext, 
            dims = ['t'],
            coords = {'t':np.arange(life_h)})

        wake_losses_eff_t = existing_wpp_power_curve_xr.wake_losses_eff.interp(ws=xr_time.wst, wd=xr_time.wdt).values
        wake_losses_eff_t_ext = expand_to_lifetime(
            wake_losses_eff_t, life = life_h, weeks_per_season_per_year = weeks_per_season_per_year)

        ws = existing_wpp_power_curve_xr.ws.values
        pcw = existing_wpp_power_curve_xr.P_no_wake.values
        
        wind_t_ext_deg_no_wake = wpp_efficiency*get_wind_ts_degradation(
            ws = ws, 
            pc = pcw, 
            ws_ts = wst_ext, 
            yr = wind_deg_yr, 
            wind_deg = wind_deg, 
            life_h = life_h, 
            share = share_WT_deg_types)
        
        wind_t_ext_deg = wake_losses_eff_t_ext*wind_t_ext_deg_no_wake

        outputs['wst_ext'] = wst_ext
        outputs['wdt_ext'] = wdt_ext
        outputs['wind_t_ext_deg'] = wind_t_ext_deg


# -----------------------------------------------------------------------
# Auxiliar functions 
# -----------------------------------------------------------------------        

def get_wind_ts_degradation(ws, pc, ws_ts, yr, wind_deg, life_h, share=0.5):
    """
    

    Parameters
    ----------
    ws : array-like
        wind speed.
    pc : array-like
        power curve.
    ws_ts : array-like
        wind speed time series.
    yr : array-like
        year.
    wind_deg : array-like
        degradation values for each year in the yr array.
    life_h : int
        lifetime in hours.
    share : float, optional
        share ratio between two degradation mechanism (0: only shift in power curve, 1: degradation as a loss factor ) . The default is 0.5.

    Returns
    -------
    p_ts_deg_partial_factor : array
        power time series after degradation.

    """
    
    t_over_year = np.arange(life_h)/(365*24)
    #degradation = wind_deg_per_year * t_over_year
    degradation = np.interp(t_over_year, yr, wind_deg)

    p_ts = get_wind_ts(ws=ws, pcw=pc, wst=ws_ts, wpp_efficiency=1)
    Dws = get_Dws(ws, pc, ws_ts,wind_deg_end=degradation[-1])
    pcdeg = get_shifted_pc(ws,pc,Dws=Dws)
    p_ts_fulldeg = get_wind_ts(ws=ws, pcw=pcdeg, wst=ws_ts, wpp_efficiency=1)

    # blend variable for pc shift over time
    if np.max(wind_deg) <= 0:
        alpha = 0
    else:
        alpha = degradation/np.max(degradation)

    # degradation in CF as a results of a shift in ws on power curve
    p_ts_deg = (1-alpha)*p_ts + alpha*p_ts_fulldeg
    # degradation in CF as a factor or losses
    p_ts_deg_factor = (1-degradation)*p_ts

    p_ts_deg_partial_factor = (1-share)*p_ts_deg + share*p_ts_deg_factor

    # ws shift cannot handle large degradations. (degradation >0.8)
    p_ts_deg_partial_factor[np.where(degradation>0.8)[0]] = p_ts_deg_factor[np.where(degradation>0.8)[0]]
    
    return p_ts_deg_partial_factor

