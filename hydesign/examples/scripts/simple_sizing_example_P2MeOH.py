def main():
    if __name__ == "__main__":
        import pandas as pd

        from hydesign.assembly.hpp_assembly_P2MeOH_bidirectional_infinite_demand import (
            hpp_model_P2MeOH_bidirectional_infinite_demand as hpp_model,
        )
        from hydesign.examples import examples_filepath
        from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver

        example = 9
        examples_sites = pd.read_csv(
            f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
        )
        ex_site = examples_sites.iloc[example]

        # Simple example to size wind only with a single core to run test machines and colab

        inputs = {
            "name": ex_site["name"],
            "longitude": ex_site["longitude"],
            "latitude": ex_site["latitude"],
            "altitude": ex_site["altitude"],
            "input_ts_fn": examples_filepath + ex_site["input_ts_fn"],
            "sim_pars_fn": examples_filepath + ex_site["sim_pars_fn"],
            "input_HA_ts_fn": examples_filepath + str(ex_site["input_HA_ts_fn"]),
            "MeOH_demand_fn": examples_filepath + ex_site["MeOH_demand_col"],
            # 'price_up_ts_fn': examples_filepath+str(ex_site['price_up_ts']),
            # 'price_dwn_ts_fn': examples_filepath+str(ex_site['price_dwn_ts']),
            "price_col": ex_site["price_col"],
            "opt_var": "NPV_over_CAPEX",
            "num_batteries": 1,
            "n_procs": 4,
            "n_doe": 32,
            "n_clusters": 1,
            "n_seed": 0,
            "max_iter": 5,
            "final_design_fn": "hydesign_design_9.csv",
            "npred": 3e4,
            "tol": 1e-6,
            "min_conv_iter": 2,
            "work_dir": "./",
            "hpp_model": hpp_model,
            "variables": {
                "clearance":
                # {'var_type':'design',
                #   'limits':[10, 60],
                #   'types':'int'
                #   },
                {"var_type": "fixed", "value": 10},
                "sp":
                # {'var_type':'design',
                #  'limits':[200, 360],
                #  'types':'int'
                #  },
                {"var_type": "fixed", "value": 360},
                "p_rated":
                # {'var_type':'design',
                #   'limits':[1, 10],
                #   'types':'int'
                #   },
                {"var_type": "fixed", "value": 5},
                "Nwt": {"var_type": "design", "limits": [0, 400], "types": "int"},
                # {'var_type':'fixed',
                #   'value': 48
                #   },
                "wind_MW_per_km2":
                # {'var_type':'design',
                #   'limits':[5, 9],
                #   'types':'float'
                #   },
                {"var_type": "fixed", "value": 5},
                "solar_MW": {"var_type": "design", "limits": [0, 400], "types": "int"},
                # {'var_type':'fixed',
                #   'value': 100
                # },
                "surface_tilt":
                # {'var_type':'design',
                #   'limits':[0, 50],
                #   'types':'float'
                #   },
                {"var_type": "fixed", "value": 50},
                "surface_azimuth":
                # {'var_type':'design',
                #   'limits':[150, 210],
                #   'types':'float'
                #   },
                {"var_type": "fixed", "value": 210},
                "DC_AC_ratio":
                # {'var_type':'design',
                #   'limits':[1, 2.0],
                #   'types':'float'
                #   },
                {
                    "var_type": "fixed",
                    "value": 1.5,
                },
                "P_batt_MW": {"var_type": "design", "limits": [0, 100], "types": "int"},
                # {'var_type':'fixed',
                #   'value': 50
                #   },
                "b_E_h": {"var_type": "design", "limits": [1, 3], "types": "int"},
                # {'var_type':'fixed',
                #   'value': 3
                #   },
                "cost_of_battery_P_fluct_in_peak_price_ratio":
                # {'var_type':'design',
                #   'limits':[0, 20],
                #   'types':'float'
                #   },
                {"var_type": "fixed", "value": 5},
                "P_SOEC_MW": {"var_type": "design", "limits": [1, 200], "types": "int"},
                # {'var_type':'fixed',
                #   'value': 100
                # },
                "m_MeOH_tank_max_kg":
                # {'var_type':'design',
                #   'limits':[0, 3000000],
                #   'types':'int'
                #   },
                {"var_type": "fixed", "value": 0},
            },
        }

        EGOD = EfficientGlobalOptimizationDriver(**inputs)
        EGOD.run()
        result = EGOD.result


main()
