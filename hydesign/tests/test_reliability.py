# -*- coding: utf-8 -*-
"""
Created on Sun Sep  1 16:22:15 2024

@author: mikf
"""
import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.examples import examples_filepath
from hydesign.assembly.hpp_assembly_reliability import ReliabilityModel

def test_reliability():
    name = 'France_good_wind'
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    ex_site = examples_sites.loc[examples_sites.name == name]

    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]

    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]

    n_reliability_seed = 2
    
    wt_rated_power_MW = 4
    surface_tilt_deg = 35
    surface_azimuth_deg = 180
    DC_AC_ratio = 1.5
    Nwt = 50
    wind_MW_per_km2 = 7
    solar_MW = 100
    b_P = 20
    b_E_h  = 3
    cost_of_batt_degr = 5
    clearance = 20
    sp = 350
    
    inverter_size = 1000 #  [kW]
    panel_size = 500 # [W]
    x = [# Wind plant design
        clearance, sp, wt_rated_power_MW, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt_deg, surface_azimuth_deg, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_batt_degr,
        # Reliability inputs
        inverter_size, panel_size,
        ]

    RM = ReliabilityModel(hpp_model,
               latitude=latitude,
               longitude=longitude,
               altitude=altitude,
               num_batteries = 10,
               sim_pars_fn = sim_pars_fn,
               input_ts_fn = input_ts_fn,
               n_reliability_seed=n_reliability_seed,
               )
    
    outs = RM.evaluate(*x)
    ref = np.array([3.31385425e-01, 8.38383438e+01, 8.54866443e-02, 4.51844300e+01,
                    1.93811075e+01, 2.52993455e+02, 3.77653071e+00, 1.53565703e+02,
                    3.10153071e+00, 3.35000000e+01, 6.75000000e-01, 6.37432351e+00,
                    0.00000000e+00, 5.95534286e+01, 0.00000000e+00, 0.00000000e+00,
                    5.21068783e+02, 1.98275793e-01, 3.00000000e+02, 2.00000000e+02,
                    1.00000000e+02, 6.00000000e+01, 2.00000000e+01, 0.00000000e+00,
                    0.00000000e+00, 2.85714286e+01, 1.22600000e+00, 2.85714286e+01,
                    1.20628807e+02, 8.03144035e+01, 2.00000000e+00, 2.89525776e+01,
                    2.68837974e-01, 1.00000000e+03, 5.00000000e+02, 1.00000000e+02,
                    1.00000000e+03, 2.00000000e+03, 5.00000000e+02])
    np.testing.assert_allclose(outs, ref, rtol=1.3e-6)
    

