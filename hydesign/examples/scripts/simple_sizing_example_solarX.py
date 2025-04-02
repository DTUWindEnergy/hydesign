def main():
    if __name__ == '__main__':
        from hydesign.assembly.hpp_assembly_solarX import hpp_model_solarX as hpp_model
        from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver
        # import yaml
        import pandas as pd
        
        from hydesign.examples import examples_filepath

        examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
        
        # Select specific site by name (e.g., 'France_good_wind')
        name = 'Denmark_good_solar'
        ex_site = examples_sites.loc[examples_sites.name == name]
        
        # Extract geographical information of the selected site
        longitude = ex_site['longitude'].values[0]
        latitude = ex_site['latitude'].values[0]
        altitude = ex_site['altitude'].values[0]

        #input_ts_fn = examples_filepath + "solarX/input_ts_tests.csv"
        input_ts_fn = examples_filepath + "solarX/input_ts_Denmark_good_solar.csv"
        
        sim_pars_fn = examples_filepath + "solarX/hpp_pars.yml"
        batch_size = 30 * 24
        
        inputs = {
            'example': None,
            'name': name,
            'longitude': longitude,
            'latitude': latitude,
            'altitude': altitude,
            'input_ts_fn': input_ts_fn,
            'sim_pars_fn': sim_pars_fn,
    
            'opt_var': "NPV_over_CAPEX",
            'num_batteries': 1,
            'n_procs': 3,
            'n_doe': 40,
            'n_clusters': 4,
            'n_seed': 0,
            'max_iter': 15,
            'final_design_fn': 'hydesign_design_0.csv',
            'npred': 3e4,
            'tol': 1e-6,
            'min_conv_iter': 2,
            'work_dir': './',
            'hpp_model': hpp_model,
            'batch_size': batch_size,
            }
    
        inputs['variables'] = {
            # sf
            'sf_area':
                {'var_type':'design',
                  'limits':[1e4, 1e6],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 1e4
                #  },
            'tower_height':
                # {'var_type':'design',
                #   'limits':[11, 30],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                 'value': 100 # can only be 20, 25 or 30
                 },

            # cpv
            'area_cpv_receiver_m2':
                {'var_type':'design',
                  'limits':[0, 10],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 10
                # },

            # cst
            'heat_exchanger_capacity':
                {'var_type':'design',
                  'limits':[0, 50],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 40
                # },
            'p_rated_st':
                {'var_type':'design',
                  'limits':[0, 20],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 10
                # },
            'v_molten_salt_tank_m3':
                {'var_type':'design',
                  'limits':[0, 1e3],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 1e3
                # },
            'area_cst_receiver_m2':
                {'var_type':'design',
                  'limits':[0, 10],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 10
                # },

            # bigas_h2
            'area_dni_reactor_biogas_h2':
                {'var_type':'design',
                  'limits':[0, 10],
                  'types':'float'
                  },
                # {'var_type':'fixed',
                #  'value': 10
                # },
            'area_el_reactor_biogas_h2':
                {'var_type':'design',
                  'limits':[0, 10],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #  'value': 10
                # },

                  }

        EGOD = EfficientGlobalOptimizationDriver(**inputs)
        EGOD.run()
        # result = EGOD.result

main()