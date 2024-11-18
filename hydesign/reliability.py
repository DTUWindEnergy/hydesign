# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:08:01 2024

@author: mikf
"""
import openmdao.api as om
import numpy as np
import chaospy as cp
import pandas as pd
import xarray as xr


class battery_with_reliability(om.ExplicitComponent):
    def __init__(
        self,
        life_y = 25,
        intervals_per_hour = 1,
        reliability_ts_battery=None,
        reliability_ts_trans=None,
        ):
        """
        Parameters
        ----------
        life_y : int, optional
            lifetime in years. The default is 25.
        intervals_per_hour : int, optional
            intervals per hour. The default is 1.
        reliability_ts_battery : array-like, optional
            battery reliability time series. The default is None.
        reliability_ts_trans : array-like, optional
            transformer reliability time series. The default is None.

        Returns
        -------
        None.
        """

        super().__init__()
        self.life_intervals = life_y * 365 * 24 * intervals_per_hour
        self.reliability_ts_battery = reliability_ts_battery
        self.reliability_ts_trans = reliability_ts_trans
        
    def setup(self):
        self.add_input(
            'b_t',
            desc="Battery charge/discharge power time series w/o reliability",
            units='MW',
            shape=[self.life_intervals])

        self.add_output(
            'b_t_rel',
            desc="Battery charge/discharge power time series with reliability",
            units='MW',
            shape=[self.life_intervals])

    def compute(self, inputs, outputs):
        if ((self.reliability_ts_battery is None) or (self.reliability_ts_trans is None)):
            outputs['b_t_rel'] = inputs['b_t']
            return
        outputs['b_t_rel'] = inputs['b_t'] * self.reliability_ts_battery[:self.life_intervals] * self.reliability_ts_trans[:self.life_intervals]
    

class wpp_with_reliability(om.ExplicitComponent):
    def __init__(
        self, 
        life_y = 25,
        intervals_per_hour = 1,
        reliability_ts_wind=None,
        reliability_ts_trans=None,
        ):
        """
        Parameters
        ----------
        life_y : int, optional
            lifetime in years. The default is 25.
        intervals_per_hour : int, optional
            intervals per hour. The default is 1.
        reliability_ts_wind : array-like, optional
            wind farm reliability time series. The default is None.
        reliability_ts_trans : array-like, optional
            transformer reliability time series. The default is None.

        Returns
        -------
        None.
        """

        super().__init__()
        self.life_intervals = life_y * 365 * 24 * intervals_per_hour
        self.reliability_ts_wind = reliability_ts_wind
        self.reliability_ts_trans = reliability_ts_trans
        
    def setup(self):
        self.add_input(
            'wind_t',
            desc="WPP power time series w/o reliability",
            units='MW',
            shape=[self.life_intervals])
        self.add_output(
            'wind_t_rel',
            desc="WPP power time series with reliability",
            units='MW',
            shape=[self.life_intervals])

    def compute(self, inputs, outputs):
        if ((self.reliability_ts_wind is None) or (self.reliability_ts_trans is None)):
            outputs['wind_t_rel'] = inputs['wind_t']
            return
        outputs['wind_t_rel'] = inputs['wind_t'] * self.reliability_ts_wind[:self.life_intervals] * self.reliability_ts_trans[:self.life_intervals]
    

class pvp_with_reliability(om.ExplicitComponent):
    def __init__(
        self, 
        life_y = 25,
        intervals_per_hour = 1,
        reliability_ts_pv=None,
        reliability_ts_trans=None,
        ):
        """
        Parameters
        ----------
        life_y : int, optional
            lifetime in years. The default is 25.
        intervals_per_hour : int, optional
            intervals per hour. The default is 1.
        reliability_ts_pv : array-like, optional
            solar farm reliability time series. The default is None.
        reliability_ts_trans : array-like, optional
            transformer reliability time series. The default is None.

        Returns
        -------
        None.
        """

        super().__init__()
        self.life_intervals = life_y * 365 * 24 * intervals_per_hour
        self.reliability_ts_pv = reliability_ts_pv
        self.reliability_ts_trans = reliability_ts_trans
        
    def setup(self):
        self.add_input(
            'solar_t',
            desc="PVP power time series w/o reliability",
            units='MW',
            shape=[self.life_intervals])
        self.add_output(
            'solar_t_rel',
            desc="PVP power time series with reliability",
            units='MW',
            shape=[self.life_intervals])

    def compute(self, inputs, outputs):
        if ((self.reliability_ts_pv is None) or (self.reliability_ts_trans is None)):
            outputs['solar_t_rel'] = inputs['solar_t']
            return
        outputs['solar_t_rel'] = inputs['solar_t'] * self.reliability_ts_pv[:self.life_intervals] * self.reliability_ts_trans[:self.life_intervals]


def availability_data_set(pdf_TTF, pdf_TTR, N_components, seed, ts_start, ts_end, ts_freq, sampling_const, component_name, **kwargs):
    """
    Parameters
    ----------
    pdf_TTF : probability distribution
        probability distribution of time to failure
    pdf_TTR : probability distribution
        probability distribution of time to repair
    N_components : INTEGER
        Number of the components for which availability time-series will be estimated.
    seed : INTEGER
        It is a constant that is introduced to ensure the reproducibility of the random sampling. 
    ts_start : str
        time series start time in the format: '2030-01-01 00:00'
    ts_end : str
        time series end time in the format: '2030-01-01 00:00'
    ts_freq : str
        time series frequency in the format: '1h'
    sampling_const : INTEGER
        It is another constant that is introduced to estimate the required samples for well-converged result.
    component_name : STRING
        Name of the component, so it can be printed on the plot figure.

    Returns
    -------
    availability_ds : Dataset
        xarray dataset with the time to failure and time to repair indices (not the actual timeseries). The reasoning for omiting saving the full time series ond only the indices where failure or "back online" occurs is that this only takes up a fraction of the space, and the construction of the timeseries is very fast when the sampling has already been pre processed.
    """

    ts_indices = pd.date_range(ts_start, ts_end, freq=ts_freq)
    N_components = int(N_components)
    N_ts = len(ts_indices)                                                  # get the length of time series
    pdf_TTF_plant = cp.Iid(pdf_TTF, N_components)               # get pdf of TTF & TTR of all components using Independent identical distributed vector of random variables
    pdf_TTR_plant = cp.Iid(pdf_TTR, N_components)
    pdf_plant = cp.J(pdf_TTF_plant,pdf_TTR_plant)                           # Joining the all pdfs of TTF and TTR together
    np.random.seed(seed)                                                    # Now set the seed for a specific realization
    N_sample = int(sampling_const * N_ts / pdf_TTR.mom(1))      # int(1e5) # number of failures and repairs sampled
    sample = pdf_plant.sample(size=N_sample, rule='R').T                    # Gnenerating the sampling of overall joined function of pdfs of TTF & TTR
    TTF = sample[:,:N_components]                                           # collecting the failure sample in i_FT & downtime sample in i_BO
    TTR = sample[:,N_components:]
    i_FT = np.cumsum(TTF, axis=0) + np.vstack([np.zeros(N_components), np.cumsum(TTR, axis=0)[:-1, :]])
    i_BO = i_FT + TTR
    i_FT = i_FT.astype(np.int64)                                                 # Round up i_FT & i_BO so they are in hours
    i_BO = i_BO.astype(np.int64)
    cond = np.any(i_FT<N_sample,axis=1)                                     # stack sample events to availability or, non-availability condition
    i_FT = i_FT[cond,:]
    i_BO = i_BO[cond,:]
    N_sample_needed = i_FT.shape[0]                                         # Updated number of failures to fill up the N_ts
    ds = xr.Dataset({'TTF_indices': (["sample", "component"], i_FT),
                     'TTR_indices': (["sample", "component"], i_BO),},
                    coords={"sample": range(N_sample_needed),
                            "component": range(N_components),
                            'N_components': N_components,
                            'ts_start': ts_start,
                            'ts_end': ts_end,
                            'ts_freq': ts_freq,
                            'N_sample': N_sample,
                            'N_sample_needed': N_sample_needed,
                            'component_name': component_name,
                            'seed': seed,},)
    return ds


def generate_availability_ensamble(ts_start='2030-01-01 00:00',
                                   ts_end='2054-12-31 23:00',
                                   ts_freq='1h',
                                   seeds=range(100),
                                   component_name='WT',
                                   file_name=None,
                                   MTTF = 1.10e4,
                                   MTTR = 1.10e2,
                                   N_components = 200, 
                                   sampling_const = 50.4,
                                   pdf = cp.Exponential,
                                   ):
    """
    Parameters
    ----------
    ts_start : str
        time series start time in the format: '2030-01-01 00:00'
    ts_end : str
        time series end time in the format: '2030-01-01 00:00'
    ts_freq : str
        time series frequency in the format: '1h'
    seeds : array-like
        Iterable of seeds to run
    component_name : STRING
        Name of the component, so it can be printed on the plot figure.
    file_name : STRING
        Name to save the netcdf dataset to.
    MTTF : float
        mean time to fail
    MTTR : float
        mean time to repair
    N_components : INTEGER
        Number of the components for which availability time-series will be estimated.
    sampling_const : INTEGER
        It is another constant that is introduced to estimate the required samples for well-converged 
    pdf : probability distribution method
        probability distribution method result.

    Returns
    -------
    availability_ds : Dataset
        xarray dataset with the time to failure and time to repair indices (not the actual timeseries) for all seeds

    for data sets with number of components >= 400 it will look like this:
    <xarray.Dataset> Size: 129kB
    Dimensions:          (sample: 20, component: 200, seed: 2)
    Coordinates:
      * sample           (sample) int32 80B 0 1 2 3 4 5 6 7 ... 13 14 15 16 17 18 19
      * component        (component) int32 800B 0 1 2 3 4 5 ... 195 196 197 198 199
        N_components     int32 4B 200
        ts_start         <U16 64B '2030-01-01 00:00'
        ts_end           <U16 64B '2054-12-31 23:00'
        ts_freq          <U2 8B '1h'
        N_sample         int32 4B 100407
        N_sample_needed  (seed) int32 8B 20 17
        component_name   <U2 8B 'WT'
      * seed             (seed) int32 8B 0 1
    Data variables:
        TTF_indices      (seed, sample, component) float64 64kB 8.754e+03 ... nan
        TTR_indices      (seed, sample, component) float64 64kB 8.914e+03 ... nan     
    for data sets with number of components > 400 it will look like this:
    <xarray.Dataset> Size: 258kB
    Dimensions:          (sample: 10, component: 400, batch_no: 2, seed: 2)
    Coordinates:
      * sample           (sample) int32 40B 0 1 2 3 4 5 6 7 8 9
      * component        (component) int32 2kB 0 1 2 3 4 5 ... 395 396 397 398 399
      * batch_no         (batch_no) int32 8B 0 1
        N_components     int32 4B 400
        ts_start         <U16 64B '2030-01-01 00:00'
        ts_end           <U16 64B '2054-12-31 23:00'
        ts_freq          <U2 8B '1h'
        N_sample         int32 4B 100524
        N_sample_needed  (seed, batch_no) int32 16B 10 8 8 7
        component_name   <U2 8B 'PV'
      * seed             (seed) int32 8B 0 1
    Data variables:
        TTF_indices      (seed, batch_no, sample, component) float64 128kB 3.742e...
        TTR_indices      (seed, batch_no, sample, component) float64 128kB 3.809e...
        final_seed       (seed, batch_no) int32 16B 1000 2000 1001 2001            
    """

    dss = []
    for seed in seeds:
        if N_components > 400:
            batches = int(np.ceil(N_components / 400))
            dss_batch = []
            for batch in range(batches):
                final_seed = seed + 1000 * (batch + 1)
                ds_batch = availability_data_set(pdf_TTF=pdf(scale=MTTF),
                                           pdf_TTR=pdf(scale=MTTR),
                                           N_components=400,
                                           seed=final_seed,
                                           ts_start=ts_start,
                                           ts_end=ts_end,
                                           ts_freq=ts_freq,
                                           sampling_const=sampling_const,
                                           component_name=component_name)
                ds_batch['batch_no'] = batch
                ds_batch['final_seed'] = final_seed
                ds_batch['seed'] = seed
                dss_batch.append(ds_batch)
            ds = xr.concat(dss_batch, 'batch_no')
        else:
            ds = availability_data_set(pdf_TTF=pdf(scale=MTTF),
                                       pdf_TTR=pdf(scale=MTTR),
                                       N_components=N_components,
                                       seed=seed,
                                       ts_start=ts_start,
                                       ts_end=ts_end,
                                       ts_freq=ts_freq,
                                       sampling_const=sampling_const,
                                       component_name=component_name)
        dss.append(ds)
    ds_out = xr.concat(dss, 'seed')
    if file_name is not None:
        ds_out.to_netcdf(file_name)
    return ds_out
            
    
if __name__ == '__main__':
    N_seeds = 2
    inputs = {
        "WT": {"MTTF": 1.10e4,
                "MTTR": 1.10e2,
                "N_components": 200,              
                "sampling_const": 50.4,},
        "PV": {"MTTF": 3.53e4,
               "MTTR": 2.18e3,
               "N_components": 400*2,              
               "sampling_const": 1000},
        "inverter": {"MTTF": 3.01e4,
                     "MTTR": 7.20e2,
                     "N_components": 20,       
                     "sampling_const": 328.8},
        "transformer": {"MTTF": 1.77e4,
                        "MTTR": 6*30*24,
                        "N_components": 1,    
                        "sampling_const": 1972},
        "BESS": {"MTTF": 7.09e4,
                 "MTTR": 1.68e2,
                 "N_components": 1,           
                 "sampling_const": 76.8}
        }
    for k, v in inputs.items():
        ds_out = generate_availability_ensamble(component_name=k,
                                                MTTF = v['MTTF'],
                                                MTTR = v['MTTR'],
                                                N_components = v['N_components'],
                                                sampling_const = v['sampling_const'],
                                                seeds=range(N_seeds),
                                                )
        print(k)
        print(ds_out)

  