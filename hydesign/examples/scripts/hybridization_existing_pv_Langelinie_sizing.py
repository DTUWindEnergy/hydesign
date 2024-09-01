if __name__ == '__main__':
    from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
    import os

    # import numpy as np
    # from numpy import newaxis as na
    # import numpy_financial as npf
    import pandas as pd

    # import seaborn as sns
    # import openmdao.api as om
    # import yaml
    # import scipy as sp
    # from scipy import stats
    # import xarray as xr
    from hydesign.assembly.hpp_assembly_hybridization_pv import hpp_model
    from hydesign.examples import examples_filepath

    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    name = 'Denmark_hybridization_solar_Langelinie'
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    # GWA 2
    # input_ts_fn = '/Users/martagrossi/hydesign/hydesign/examples/Europe/GWA2/input_ts_Denmark_hybridization_solar_Langelinie.csv'
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]

    # Inputs for the PV model
    # existing_inverter_efficiency_curve_fn = '/Users/martagrossi/hydesign/hydesign/PV validation/eff_curve_M88H_710.csv'
    # sim_pars_fn = examples_filepath + 'hybridization_existing_pv/hpp_pars_Langelinie.yml'

    # # Replace 'example_existing_wpp_power_curve.nc' with the actual file path
    # file_path = 'eff_curve_M88H_710.csv'

    # existing_inverter_efficiency_curve = pd.read_csv(existing_inverter_efficiency_curve_fn)
    # existing_inverter_efficiency_curve

    n_procs = int(os.cpu_count())
    print(n_procs)


    inputs = {
        'name': 'Denmark_hybridization_solar_Langelinie',
        'longitude': longitude,
        'latitude': latitude,
        'altitude': altitude,
        'input_ts_fn': input_ts_fn,
        'sim_pars_fn': sim_pars_fn,
        # 'existing_wpp_power_curve_xr_fn': existing_inverter_efficiency_curve_fn,
        # 'batt_reduction': 0,

        'opt_var': "NPV_over_CAPEX",
        'num_batteries': 10,
        'n_procs': n_procs - 1,
        'n_doe': 20,
        'n_clusters': 5,
        'n_seed': 0,
        'max_iter': 10,
        'final_design_fn': 'hydesign_design_0.csv',
        'npred': 3e4,
        'tol': 1e-6,
        'n_comp': 3,
        'min_conv_iter': 3,
        'work_dir': './',
        'hpp_model': hpp_model,
        #'PPA_price': 40,
    'variables': {
        'clearance [m]':
        {'var_type':'design',
         'limits':[10, 60],
         'types':'int'
         },
            # {'var_type': 'fixed',
            #  'value': 35
            #  },
        'sp [W/m2]':
        # {'var_type':'design',
        # 'limits':[200, 359],
        # 'types':'int'
        # },
            {'var_type': 'fixed',
             'value': 300
             },
        'p_rated [MW]':
            {'var_type': 'design',
             'limits': [1, 10],
             'types': 'int'
             },
        # {'var_type':'fixed',
        #  'value': 6
        # },
        'Nwt':
            {'var_type': 'design',
             'limits': [0, 400],
             'types': 'int'
             },
        # {'var_type':'fixed',
        #   'value': 200
        #   },
        'wind_MW_per_km2 [MW/km2]':
        # {'var_type':'design',
        #  'limits':[5, 9],
        #  'types':'float'
        #  },
            {'var_type': 'fixed',
             'value': 7
             },
        'b_P [MW]':
            {'var_type': 'design',
             'limits': [0, 100],
             'types': 'int'
             },
        # {'var_type':'fixed',
        #  'value': 50
        #  },
        'b_E_h [h]':
            # {'var_type': 'design',
            #  'limits': [1, 10],
            #  'types': 'int'
            #  },
        {'var_type':'fixed',
         'value': 6
         },
        'cost_of_battery_P_fluct_in_peak_price_ratio':
            {'var_type': 'design',
             'limits': [0, 20],
             'types': 'float'
             },
        #         {'var_type':'fixed',
        #           'value': 10},
    }}

    EGOD = EfficientGlobalOptimizationDriver(**inputs)
    EGOD.run()
    result = EGOD.result

