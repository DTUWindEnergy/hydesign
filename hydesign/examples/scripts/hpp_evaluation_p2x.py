# -*- coding: utf-8 -*-
"""HPP_evaluation_P2X.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1z4fy2dYldx_9Uft1YwqgXd2fyGinq3NU

## Evaluating the performance of a hybrid power plant with P2X using HyDesign

In this notebook we will evaluate a hybrid power plant design in a specific location.

A hybrid power plant design consists on selecting the following parameters:

**Wind Plant design:**

1. Number of wind turbines in the wind plant [-] (`Nwt`)
2. Wind power installation density [MW/km2] (`wind_MW_per_km2`): This parameter controls how closely spaced are the turbines, which in turns affect how much wake losses are present.

**PV Plant design:**

3. Solar plant power capacity [MW] (`solar_MW`)

**Battery Storage design:**

4. Battery power [MW] (`b_P`)
5. Battery energy capacity in hours [MWh] (`b_E_h `): Battery storage capacity in hours of full battery power (`b_E = b_E_h * b_P `).
6. Cost of battery power fluctuations in peak price ratio [-] (`cost_of_batt_degr`): This parameter controls how much penalty is given to do ramps in battery power in the HPP operation.

**Electrolyzer design:**

7. Electrolyzer capacity [MW] (`ptg_MW`)
8. H2 storage capacity [kg] (`HSS_kg`)

##
**Imports**

Install hydesign if needed.
Import basic libraries.
Import HPP model assembly class.
Import the examples file path.
"""

# # Detect if running in Kaggle
# import os
# if os.environ.get('KAGGLE_KERNEL_RUN_TYPE'):
#     mypaths = !python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"
#     mypath = mypaths[0]
#     !pip install trash-cli
#     !trash $mypath/numpy*
#     !pip install --upgrade numpy
#     !pip install finitediff
#     import os
#     os.kill(os.getpid(), 9)

# # Install hydesign if needed
# import importlib
# if not importlib.util.find_spec("hydesign"):
#     !pip install git+https://gitlab.windenergy.dtu.dk/TOPFARM/hydesign.git

import os
import time
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hydesign.hpp_assembly_P2X import hpp_model_P2X
from hydesign.examples import examples_filepath

"""##
**Specifying the site**

Hydesign, provides example data from several sites in India and Europe.

The site coordinates (longitude, latitude, and altitude) are given in `examples_sites.csv`.
"""

examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0)
examples_sites

"""##
**Select a site to run**
"""

name = 'Denmark_good_wind'

ex_site = examples_sites.loc[examples_sites.name == name]

longitude = ex_site['longitude'].values[0]
latitude = ex_site['latitude'].values[0]
altitude = ex_site['altitude'].values[0]

# Weather data and Price data
input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]

input_ts = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)

required_cols = [col for col in input_ts.columns if 'WD' not in col]
input_ts = input_ts.loc[:,required_cols]
input_ts

# Hydrogen demand data, when H2 offtake is infinite -> make H2_demand values very high (1e6) in H2_demand.csv file and penalty_H2 as '0' in hpp_pars.yml file
H2_demand_fn = examples_filepath+ex_site['H2_demand_col'].values[0]

H2_demand_ts = pd.read_csv(H2_demand_fn, index_col=0, parse_dates=True)
H2_demand_ts

# Input data of technology's cost
sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]

with open(sim_pars_fn) as file:
    sim_pars = yaml.load(file, Loader=yaml.FullLoader)

print(sim_pars_fn)
sim_pars

"""##
**Initializing the HPP model**

Initialize the HPP model (hpp_model class) with the coordinates and the necessary input files.
"""

hpp = hpp_model_P2X(
        latitude,
        longitude,
        altitude,
        num_batteries = 3,
        work_dir = './',
        sim_pars_fn = sim_pars_fn,
        input_ts_fn = input_ts_fn,
        H2_demand_fn = H2_demand_fn,
)

"""##
### Evaluating the HPP model
"""

start = time.time()

clearance = 10
sp = 360
p_rated = 4
Nwt = 90
wind_MW_per_km2 = 5
solar_MW = 80
surface_tilt = 50
surface_azimuth = 210
DC_AC_ratio = 1.5
b_P = 40
b_E_h  = 4
cost_of_batt_degr = 10
ptg_MW = 200
HSS_kg = 3000

x = [clearance, sp, p_rated, Nwt, wind_MW_per_km2, solar_MW, \
surface_tilt, surface_azimuth, DC_AC_ratio, b_P, b_E_h , cost_of_batt_degr, ptg_MW, HSS_kg]

outs = hpp.evaluate(*x)

hpp.print_design(x, outs)

end = time.time()
print(f'exec. time [min]:', (end - start)/60 )

"""##
### Plot the HPP operation
"""

b_E_SOC_t = hpp.prob.get_val('ems_P2X.b_E_SOC_t')
b_t = hpp.prob.get_val('ems_P2X.b_t')
price_t = hpp.prob.get_val('ems_P2X.price_t')

wind_t = hpp.prob.get_val('ems_P2X.wind_t')
solar_t = hpp.prob.get_val('ems_P2X.solar_t')
hpp_t = hpp.prob.get_val('ems_P2X.hpp_t')
hpp_curt_t = hpp.prob.get_val('ems_P2X.hpp_curt_t')
P_ptg_t = hpp.prob.get_val('ems_P2X.P_ptg_t')
P_ptg_grid_t = hpp.prob.get_val('ems_P2X.P_ptg_grid_t')
grid_MW = hpp.prob.get_val('ems_P2X.G_MW')

n_days_plot = 7 #14
plt.figure(figsize=[12,4])
plt.plot(price_t[:24*n_days_plot], label='price')
# plt.plot(b_E_SOC_t[:24*n_days_plot], label='SoC [MWh]')
# plt.plot(b_t[:24*n_days_plot], label='Battery P [MW]')
plt.xlabel('time [hours]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=3, fancybox=0, shadow=0)

plt.figure(figsize=[12,4])
plt.plot(wind_t[:24*n_days_plot], label='wind', color='blue')
plt.plot(solar_t[:24*n_days_plot], label='PV', color='cyan')
plt.plot(hpp_t[:24*n_days_plot], label='HPP', color='purple')
plt.plot(hpp_curt_t[:24*n_days_plot], label='HPP curtailed', color='red')
plt.plot(b_t[:24*n_days_plot], label='Battery P [MW]', color='grey')
plt.plot(P_ptg_t[:24*n_days_plot], label='PtG_green', color='green')
plt.plot(P_ptg_grid_t[:24*n_days_plot], label='PtG_grid', color='orange')
plt.axhline(grid_MW, label='Grid MW', color='black')
plt.xlabel('time [hours]')
plt.ylabel('Power [MW]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=0, shadow=0)


m_H2_t = hpp.prob.get_val('ems_P2X.m_H2_t')
m_H2_demand_t = hpp.prob.get_val('ems_P2X.m_H2_demand_t_ext')
m_H2_offtake_t = hpp.prob.get_val('ems_P2X.m_H2_offtake_t')
m_H2_storage_t = hpp.prob.get_val('ems_P2X.m_H2_storage_t')
m_H2_grid_t = hpp.prob.get_val('ems_P2X.m_H2_grid_t')
LoS_H2_t = hpp.prob.get_val('ems_P2X.LoS_H2_t')

plt.figure(figsize=[12,4])
plt.plot(m_H2_t[:24*n_days_plot], label='Green H2 produced', color='green')
plt.plot(m_H2_offtake_t[:24*n_days_plot], label='H2 offtake', color='orange')
plt.plot(m_H2_demand_t[:24*n_days_plot], label='H2 demand', color='black')
plt.plot(m_H2_storage_t[:24*n_days_plot], label='H2 storage', color='purple')
plt.plot(m_H2_grid_t[:24*n_days_plot], label='H2 from grid', color='red')
plt.plot(LoS_H2_t[:24*n_days_plot], label='LoS', color='grey')
plt.xlabel('time [hours]')
plt.ylabel('Hydrogen [kg]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=0, shadow=0)

# Create a figure and a grid of subplots
plt.figure(figsize=[16,6])
fig, axes = plt.subplots(nrows=3, ncols=1)

# Now you can use the axes to create individual plots
axes[0].plot(m_H2_offtake_t[:24*n_days_plot])
axes[0].set_title('H2 offtake')

axes[1].plot(m_H2_t[:24*n_days_plot])
axes[1].set_title('H2 produced')

axes[2].plot(LoS_H2_t[:24*n_days_plot])
axes[2].set_title('LoS_H2')

# Adjust layout and display the plots
plt.tight_layout()
plt.show()
