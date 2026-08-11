# -*- coding: utf-8 -*-

import time

import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath

examples_sites = pd.read_csv(
    f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
)
name = "Denmark_hybridization_solar_Langelinie"

ex_site = examples_sites.loc[examples_sites.name == name]

longitude = ex_site["longitude"].values[0]
latitude = ex_site["latitude"].values[0]
altitude = ex_site["altitude"].values[0]

sim_pars_fn = examples_filepath + ex_site["sim_pars_fn"].values[0]
input_ts_fn = examples_filepath + ex_site["input_ts_fn"].values[0]

hpp = hpp_model(
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    max_num_batteries_allowed=10,
    work_dir="./",
    sim_pars_fn=sim_pars_fn,
    input_ts_fn=input_ts_fn,
    technologies=[
        {
            "name": "solar",
            "deployment_year": 0,
            "lifetime": 25,
            "capex_phasing": {"years": [0], "shares": [1]},
        },
        {
            "name": "wind",
            "deployment_year": 5,
            "lifetime": 25,
            "capex_phasing": {"years": [0], "shares": [1]},
        },
        {
            "name": "battery",
            "deployment_year": 5,
            "lifetime": 25,
            "capex_phasing": {"years": [0], "shares": [1]},
        },
    ],
)

solar_MW = 7.48
surface_tilt = 25
surface_azimuth = 180
DC_AC_ratio = 1.32
clearance = 50
sp = 301
p_rated = 2
Nwt = 3
wind_MW_per_km2 = 10
b_P = 10  # MW
b_E_h = 3  # hours
cost_of_battery_P_fluct_in_peak_price_ratio = 0

x = [
    # Wind plant design
    clearance,
    sp,
    p_rated,
    Nwt,
    wind_MW_per_km2,
    # PV plant design
    solar_MW,
    surface_tilt,
    surface_azimuth,
    DC_AC_ratio,
    # Energy storage & EMS price constrains
    b_P,
    b_E_h,
    cost_of_battery_P_fluct_in_peak_price_ratio,
]

"""##
### Evaluating the HPP model
"""

start = time.time()

outs = hpp.evaluate(*x)

hpp.print_design(x, outs)

end = time.time()

print("exec. time [min]:", (end - start) / 60)
