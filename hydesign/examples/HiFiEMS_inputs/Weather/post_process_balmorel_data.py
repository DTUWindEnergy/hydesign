# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:15:16 2024

@author: mikf
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce

plt.close('all')

dir_path = 'C:\Sandbox\Repo\TOPFARM\hydesign\hydesign\examples\HiFiEMS_inputs\Weather\Balmorel_output'

#%% SOLAR - splitting GHI_DA into DHI_DA and DNI_DA and re-sampling

dhi_fn = os.path.join(dir_path, "DHI.csv")
dni_fn = os.path.join(dir_path, "DNI.csv")
ghi_fn = os.path.join(dir_path, "GHI.csv")
ghi_da_fn = os.path.join(dir_path, "GHI_DA_draft.csv")

new_freq = '15min'
n_intervals = 4

index = pd.read_csv(dhi_fn)['time'].values
old_index = pd.date_range(index[0], periods = 365 * 24, freq='1h')
new_index = pd.date_range(index[0], periods = 365 * 24 * n_intervals, freq=new_freq)

dhi = pd.read_csv(dhi_fn)['loc1'].values
dni = pd.read_csv(dni_fn)['loc1'].values
ghi = pd.read_csv(ghi_fn)['loc1'].values
ghi_da = pd.read_csv(ghi_da_fn)['loc1'].values

theta = np.nan_to_num(np.arccos((ghi - dhi) / dni))
split = np.nan_to_num(dni * np.cos(theta) / dhi)

dhi_da = ghi_da / (1 + split)
dni_da = (ghi_da - dhi_da) / np.cos(theta)
# dni_da2 = split * dhi_da / np.cos(theta)

df = pd.DataFrame({'GHI': ghi, 'DNI': dni, 'DHI': dhi}, index=old_index)
# df.to_csv("C:\Sandbox\Repo\TOPFARM\hydesign\hydesign\examples\HiFiEMS_inputs\weater_input\derived_output\solar.csv")

df2 = pd.DataFrame({'GHI': ghi_da, 'DNI': dni_da, 'DHI': dhi_da}, index=old_index)
# df2.to_csv("C:\Sandbox\Repo\TOPFARM\hydesign\hydesign\examples\HiFiEMS_inputs\weater_input\derived_output\solar_da.csv")

df_15min = df.resample(new_freq).interpolate('linear')
df_15min = pd.concat([df_15min] + (n_intervals - 1) * [df_15min[-1:]])  # resampling does not fill the last hour
df_15min.set_index(new_index, inplace=True)

df_15min_da = df2.resample(new_freq).interpolate('linear')
df_15min_da = pd.concat([df_15min_da] + (n_intervals - 1) * [df_15min_da[-1:]])  # resampling does not fill the last hour
df_15min_da.set_index(new_index, inplace=True)

#%% TEMP - No temperature information so copying from another example and resampling

temp = pd.read_csv("C:\Sandbox\Repo\TOPFARM\hydesign\hydesign\examples\Europe\GWA2\input_ts_Denmark_good_wind.csv")['temp_air_1'].values[:365*24]
df_temp = pd.DataFrame({'temp_air_1': temp}, index=old_index)
df_temp_15min = df_temp.resample(new_freq).interpolate('linear')
df_temp_15min = pd.concat([df_temp_15min] + (n_intervals - 1) * [df_temp_15min[-1:]])  # resampling does not fill the last hour
df_temp_15min.set_index(new_index, inplace=True)


#%% WIND - Down-sampling from 5 to 15 min.

WS_10m_fn = os.path.join(dir_path, 'WS_10m.csv')
WS_50m_fn = os.path.join(dir_path, 'WS_50m.csv')
WS_100m_fn = os.path.join(dir_path, 'WS_100m.csv')
WS_150m_fn = os.path.join(dir_path, 'WS_150m.csv')
WS_200m_fn = os.path.join(dir_path, 'WS_200m.csv')

WS_DA_10m_fn = os.path.join(dir_path, 'WS_DA_10m.csv')
WS_DA_50m_fn = os.path.join(dir_path, 'WS_DA_50m.csv')
WS_DA_100m_fn = os.path.join(dir_path, 'WS_DA_100m.csv')
WS_DA_150m_fn = os.path.join(dir_path, 'WS_DA_150m.csv')
WS_DA_200m_fn = os.path.join(dir_path, 'WS_DA_200m.csv')

WS_10m = pd.read_csv(WS_10m_fn)['loc1'].values
WS_50m = pd.read_csv(WS_50m_fn)['loc1'].values
WS_100m = pd.read_csv(WS_100m_fn)['loc1'].values
WS_150m = pd.read_csv(WS_150m_fn)['loc1'].values
WS_200m = pd.read_csv(WS_200m_fn)['loc1'].values

WS_10m_DA = pd.read_csv(WS_DA_10m_fn)['loc1'].values
WS_50m_DA = pd.read_csv(WS_DA_50m_fn)['loc1'].values
WS_100m_DA = pd.read_csv(WS_DA_100m_fn)['loc1'].values
WS_150m_DA = pd.read_csv(WS_DA_150m_fn)['loc1'].values
WS_200m_DA = pd.read_csv(WS_DA_200m_fn)['loc1'].values

new_freq = '15min'
n_intervals = 4

index = pd.read_csv(WS_10m_fn)['time'].values
old_index = pd.date_range(index[0], periods = 365 * 24 * 12, freq='5min')

index_DA = pd.read_csv(WS_DA_10m_fn)['time'].values
old_index_DA = pd.date_range(index_DA[0], periods = 365 * 24, freq='1h')

new_index = pd.date_range(index[0], periods = 365 * 24 * n_intervals, freq=new_freq)

df_wind = pd.DataFrame({'WS_10': WS_10m, 'WS_50': WS_50m, 'WS_100': WS_100m, 'WS_150': WS_150m, 'WS_200': WS_200m, }, index=old_index)
df_wind_DA = pd.DataFrame({'WS_10': WS_10m_DA, 'WS_50': WS_50m_DA, 'WS_100': WS_100m_DA, 'WS_150': WS_150m_DA, 'WS_200': WS_200m_DA, }, index=old_index_DA)

df_wind_15min = df_wind.resample(new_freq).interpolate('linear')
df_wind_15min.set_index(new_index, inplace=True)
# df_wind_15min = pd.concat([df_wind_15min] + (n_intervals - 1) * [df_wind_15min[-1:]])  # resampling does not fill the last hour

df_wind_15min_DA = df_wind_DA.resample(new_freq).interpolate('linear')
df_wind_15min_DA = pd.concat([df_wind_15min_DA] + (n_intervals - 1) * [df_wind_15min_DA[-1:]])  # resampling does not fill the last hour
df_wind_15min_DA.set_index(new_index, inplace=True)

#%% COLLECT ALL

df_merged = pd.concat([df_wind_15min, df_temp_15min, df_15min], axis=1, sort=False)
df_merged_DA = pd.concat([df_wind_15min_DA, df_temp_15min, df_15min_da], axis=1, sort=False)


#%% No price available so set this to zero
df_merged['Price'] = 0
df_merged_DA['Price'] = 0

#%% Constructing real time weather by assuming the HA forecast will be accurate

df_RT = df_merged.copy()
df_RT[:] = np.roll(df_merged.to_numpy(), 1, axis=0)

#%% Save and plot
df_merged_DA.to_csv('input_ts_HA.csv')
df_merged_DA.to_csv('input_ts_DA.csv')
df_merged.to_csv('input_ts_RT.csv')
df_RT.to_csv('input_ts_Measurement.csv')

plt.figure()
plt.plot(df.index, df.DNI, marker='<')
plt.plot(df_15min.index, df_15min.DNI, marker='x')

plt.figure()
plt.plot(df_wind.index, df_wind.WS_150, marker='<')
plt.plot(df_wind_15min.index, df_wind_15min.WS_150, marker='x')

plt.figure()
plt.plot(df_merged.index, df_merged.WS_50)
plt.plot(df_RT.index, df_RT.WS_50)
