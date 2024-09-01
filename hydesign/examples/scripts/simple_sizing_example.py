def main():
    if __name__ == '__main__':
        from hydesign.assembly.hpp_assembly import hpp_model
        from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
        from hydesign.examples import examples_filepath
        import pandas as pd
        
        example = 4
        examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
        ex_site = examples_sites.iloc[example]

        # Simple example to size wind only with a single core to run test machines and colab
        
        inputs = {
            'name': ex_site['name'],
            'longitude': ex_site['longitude'],
            'latitude': ex_site['latitude'],
            'altitude': ex_site['altitude'],
            'input_ts_fn': examples_filepath+ex_site['input_ts_fn'],
            'sim_pars_fn': examples_filepath+ex_site['sim_pars_fn'],
    
            'opt_var': "NPV_over_CAPEX",
            'num_batteries': 1,
            'n_procs': 1,
            'n_doe': 8,
            'n_clusters': 1,
            'n_seed': 0,
            'max_iter': 2,
            'final_design_fn': 'hydesign_design_0.csv',
            'npred': 3e4,
            'tol': 1e-6,
            'min_conv_iter': 2,
            'work_dir': './',
            'hpp_model': hpp_model,
        'variables': {
            'clearance [m]':
                {'var_type':'design',
                  'limits':[10, 60],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #   'value': 35
                #   },
             'sp [W/m2]':
                {'var_type':'design',
                 'limits':[200, 359],
                 'types':'int'
                 },
            'p_rated [MW]':
                {'var_type':'design',
                  'limits':[1, 10],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #  'value': 6
                 # },
            'Nwt':
                {'var_type':'design',
                  'limits':[0, 400],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #   'value': 200
                #   },
            'wind_MW_per_km2 [MW/km2]':
                {'var_type':'design',
                  'limits':[5, 9],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #   'value': 7
                #   },
            'solar_MW [MW]':
                # {'var_type':'design',
                #   'limits':[0, 400],
                #   'types':'int'
                #   },
                {'var_type':'fixed',
                  'value': 200
                  },
            'surface_tilt [deg]':
                # {'var_type':'design',
                #   'limits':[0, 50],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 25
                  },
            'surface_azimuth [deg]':
                # {'var_type':'design',
                #   'limits':[150, 210],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 180
                  },
            'DC_AC_ratio':
                # {'var_type':'design',
                #   'limits':[1, 2.0],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value':1.0,
                  },
            'b_P [MW]':
                # {'var_type':'design',
                #   'limits':[0, 100],
                #   'types':'int'
                #   },
                {'var_type':'fixed',
                  'value': 50
                  },
            'b_E_h [h]':
                # {'var_type':'design',
                #   'limits':[1, 10],
                #   'types':'int'
                #   },
                {'var_type':'fixed',
                  'value': 6
                  },
            'cost_of_battery_P_fluct_in_peak_price_ratio':
                # {'var_type':'design',
                #   'limits':[0, 20],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 10},
            }}
        EGOD = EfficientGlobalOptimizationDriver(**inputs)
        EGOD.run()
        result = EGOD.result

main()