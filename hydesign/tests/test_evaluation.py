import numpy as np
import pandas as pd
import pytest
import pickle
import os

from hydesign.tests.test_files import tfp
from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.assembly.hpp_assembly_P2X import hpp_model_P2X
from hydesign.assembly.hpp_assembly_constantoutput import hpp_model_constant_output
from hydesign.assembly.hpp_assembly_P2X_bidrectional import hpp_model_P2X_bidirectional
from hydesign.assembly.hpp_assembly_BM import hpp_model as hpp_model_BM
from hydesign.examples import examples_filepath


def run_evaluation(out_name = 'France_good_wind_design.csv',
                   name = 'France_good_wind',
                   design_name = 'Design 1',
                   p2x = False,
                   tmp_name = 'test_eval_design_1',
                   PPA=None,
                   constant_load=False,
                   load_min=3,
                   p2x_bidirectional=False,
                   BM=False,
                   ):
    output_df = pd.read_csv(
        tfp+out_name,
        index_col=0, 
        parse_dates = True)
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    H2_demand_fn = examples_filepath+ex_site['H2_demand_col'].values[0]
    if BM:
        input_HA_ts_fn = examples_filepath+ex_site['input_HA_ts_fn'].values[0]
        price_up_ts_fn = examples_filepath+ex_site['price_up_ts'].values[0]
        price_dwn_ts_fn = examples_filepath+ex_site['price_dwn_ts'].values[0]
        price_col = ex_site['price_col'].values[0]

        hpp = hpp_model_BM(            
            latitude,
            longitude,
            altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            input_HA_ts_fn = input_HA_ts_fn,
            price_up_ts_fn = price_up_ts_fn,
            price_dwn_ts_fn = price_dwn_ts_fn,
            price_col = price_col,
)
    elif p2x_bidirectional:
        hpp = hpp_model_P2X_bidirectional(
            latitude,
            longitude,
            altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            H2_demand_fn = H2_demand_fn,
            electrolyzer_eff_curve_name = 'Alkaline electrolyzer H2 production',
            )
    elif constant_load:
        hpp = hpp_model_constant_output(
            latitude,
            longitude,
            altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            load_min=load_min,)
                
    elif PPA is not None:
        hpp = hpp_model(
            latitude,
            longitude,
            altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            ppa_price=PPA,)
    elif not p2x:
        hpp = hpp_model(
            latitude,
            longitude,
            altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn)
    else:
        hpp = hpp_model_P2X(
            latitude,
            longitude,
            altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            H2_demand_fn=H2_demand_fn) 
    clearance = output_df.loc['clearance [m]',design_name]
    sp = output_df.loc['sp [W/m2]',design_name]
    p_rated = output_df.loc['p_rated [MW]',design_name]
    Nwt = output_df.loc['Nwt',design_name]
    wind_MW_per_km2 = output_df.loc['wind_MW_per_km2 [MW/km2]',design_name]
    solar_MW = output_df.loc['solar_MW [MW]',design_name]
    surface_tilt = output_df.loc['surface_tilt [deg]',design_name]
    surface_azimuth = output_df.loc['surface_azimuth [deg]',design_name]
    solar_DCAC = output_df.loc['DC_AC_ratio',design_name]
    b_P = output_df.loc['b_P [MW]',design_name]
    b_E_h  = output_df.loc['b_E_h [h]',design_name]
    cost_of_batt_degr = output_df.loc['cost_of_battery_P_fluct_in_peak_price_ratio',design_name]

    x = [clearance, sp, p_rated, Nwt, wind_MW_per_km2, \
    solar_MW, surface_tilt, surface_azimuth, solar_DCAC, \
    b_P, b_E_h , cost_of_batt_degr]
    if p2x:
        ptg_MW = output_df.loc['PTG [MW]','Design 1']
        HSS_kg = output_df.loc['HSS [kg]','Design 1']
        x.extend([ptg_MW, HSS_kg])
    if p2x_bidirectional:
        x=[50, 300, 10, 40, 10, 0, 45, 180, 1.5, 40, 4, 5, 250, 5000]
    if BM:
        x = [10.0, 350.0, 5.0, 70, 7.0, 0.0, 25.0, 180.0, 1.0, 1.0, 4.0, 10.0]
    outs = hpp.evaluate(*x)
    hpp.evaluation_in_csv(os.path.join(tfp + 'tmp', tmp_name), longitude, latitude, altitude, x, outs)
    return outs

def update_test(out_name='France_good_wind_design.csv',
                name = 'France_good_wind',
                design_name = 'Design 1',
                p2x = False,
                tmp_name = 'test_eval_design_1',
                PPA=None,
                constant_load=False,
                load_min=3,
                p2x_bidirectional=False,
                BM=False,
                ):
    output_df = pd.read_csv(
        tfp+out_name,
        index_col=0, 
        parse_dates = True)
    run_evaluation(out_name, name, design_name, p2x, tmp_name, PPA, constant_load, load_min, p2x_bidirectional, BM)
    eval_df = pd.read_csv(os.path.join(tfp + 'tmp', tmp_name + '.csv'))
    output_df[design_name] = eval_df.T[0]
    output_df.to_csv(tfp+out_name)
    
def load_evaluation(out_name='France_good_wind_design.csv',
                    design_name = 'Design 1',
                    p2x = False,
                    ):
    output_df = pd.read_csv(
        tfp+out_name,
        index_col=0, 
        parse_dates = True)
    if not p2x:
        load_file = np.array(output_df.iloc[15:][design_name])
    else:
        load_file = np.array(output_df.iloc[17:][design_name])
    return load_file
    
    
# ------------------------------------------------------------------------------------------------
# design 1

def run_evaluation_design_1():
    return run_evaluation(out_name = 'France_good_wind_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 1',
                       p2x = False,
                       tmp_name = 'test_eval_design_1',
                       )

def update_test_design_1():
    update_test(out_name='France_good_wind_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    p2x = False,
                    tmp_name = 'test_eval_design_1',
                    )
    

def load_evaluation_design_1():
    return load_evaluation(out_name='France_good_wind_design.csv',
                        design_name = 'Design 1',
                        p2x = False,)

def test_evaluation_design_1():
    evaluation_metrics = run_evaluation_design_1()
    loaded_metrics = load_evaluation_design_1()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)
        
# ------------------------------------------------------------------------------------------------
# design 2

def run_evaluation_design_2():
    return run_evaluation(out_name = 'France_good_wind_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 2',
                       p2x = False,
                       tmp_name = 'test_eval_design_2',
                       )

def update_test_design_2():
    update_test(out_name='France_good_wind_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 2',
                    p2x = False,
                    tmp_name = 'test_eval_design_2',
                    )
    

def load_evaluation_design_2():
    return load_evaluation(out_name='France_good_wind_design.csv',
                        design_name = 'Design 2',
                        p2x = False,)


def test_evaluation_design_2():
    evaluation_metrics = run_evaluation_design_2()
    loaded_metrics = load_evaluation_design_2()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)
        
# ------------------------------------------------------------------------------------------------

# # # design 3

def run_evaluation_design_3():
    return run_evaluation(out_name = 'France_good_wind_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 3',
                       p2x = False,
                       tmp_name = 'test_eval_design_3',
                       )

def update_test_design_3():
    update_test(out_name='France_good_wind_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 3',
                    p2x = False,
                    tmp_name = 'test_eval_design_3',
                    )
    

def load_evaluation_design_3():
    return load_evaluation(out_name='France_good_wind_design.csv',
                        design_name = 'Design 3',
                        p2x = False,)


def test_evaluation_design_3():
    evaluation_metrics = run_evaluation_design_3()
    loaded_metrics = load_evaluation_design_3()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)

# ------------------------------------------------------------------------------------------------
# design 1_P2X

def run_evaluation_design_1_P2X():
    return run_evaluation(out_name = 'Evaluation_test_P2X.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 1',
                       p2x = True,
                       tmp_name = 'test_eval_design_1_P2X',
                       )

def update_test_design_1_P2X():
    update_test(out_name='Evaluation_test_P2X.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    p2x = True,
                    tmp_name = 'test_eval_design_1_P2X',
                    )
    

def load_evaluation_design_1_P2X():
    return load_evaluation(out_name='Evaluation_test_P2X.csv',
                        design_name = 'Design 1',
                        p2x = True,)

def test_evaluation_design_1_P2X():
    evaluation_metrics = run_evaluation_design_1_P2X()
    loaded_metrics = load_evaluation_design_1_P2X()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)
        
# ------------------------------------------------------------------------------------------------
# design 2_P2X

def run_evaluation_design_2_P2X():
    return run_evaluation(out_name = 'Evaluation_test_P2X.csv',
                       name = 'Indian_site_good_wind',
                       design_name = 'Design 2',
                       p2x = True,
                       tmp_name = 'test_eval_design_2_P2X',
                       )

def update_test_design_2_P2X():
    update_test(out_name='Evaluation_test_P2X.csv',
                    name = 'Indian_site_good_wind',
                    design_name = 'Design 2',
                    p2x = True,
                    tmp_name = 'test_eval_design_2_P2X',
                    )
    

def load_evaluation_design_2_P2X():
    return load_evaluation(out_name='Evaluation_test_P2X.csv',
                        design_name = 'Design 2',
                        p2x = True,)

def test_evaluation_design_2_P2X():
    evaluation_metrics = run_evaluation_design_2_P2X()
    loaded_metrics = load_evaluation_design_2_P2X()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)
        
# ------------------------------------------------------------------------------------------------

def run_evaluation_design_3_P2X():
    return run_evaluation(out_name = 'Evaluation_test_P2X.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 3',
                       p2x = True,
                       tmp_name = 'test_eval_design_3_P2X',
                       )

def update_test_design_3_P2X():
    update_test(out_name='Evaluation_test_P2X.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 3',
                    p2x = True,
                    tmp_name = 'test_eval_design_3_P2X',
                    )
    

def load_evaluation_design_3_P2X():
    return load_evaluation(out_name='Evaluation_test_P2X.csv',
                        design_name = 'Design 3',
                        p2x = True,)


def test_evaluation_design_3_P2X():
    evaluation_metrics = run_evaluation_design_3_P2X()
    loaded_metrics = load_evaluation_design_3_P2X()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)


# ------------------------------------------------------------------------------------------------
# PPA 1

def run_evaluation_PPA():
    return run_evaluation(out_name = 'PPA_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 1',
                       p2x = False,
                       tmp_name = 'test_eval_PPA',
                       PPA=21.4,
                       )

def update_test_PPA():
    update_test(out_name='PPA_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    p2x = False,
                    tmp_name = 'test_eval_PPA',
                    PPA=21.4,
                    )
    

def load_evaluation_PPA():
    return load_evaluation(out_name='PPA_design.csv',
                        design_name = 'Design 1',
                        p2x = False)

def test_evaluation_PPA():
    evaluation_metrics = run_evaluation_PPA()
    loaded_metrics = load_evaluation_PPA()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)

        
# PPA 2

def run_evaluation_PPA2():
    return run_evaluation(out_name = 'PPA_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 2',
                       p2x = False,
                       tmp_name = 'test_eval_PPA2',
                       PPA=41.4,
                       )

def update_test_PPA2():
    update_test(out_name='PPA_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 2',
                    p2x = False,
                    tmp_name = 'test_eval_PPA2',
                    PPA=41.4,
                    )
    

def load_evaluation_PPA2():
    return load_evaluation(out_name='PPA_design.csv',
                        design_name = 'Design 2',
                        p2x = False)

def test_evaluation_PPA2():
    evaluation_metrics = run_evaluation_PPA2()
    loaded_metrics = load_evaluation_PPA2()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)

# ------------------------------------------------------------------------------------------------
# constant load 1

def run_evaluation_constant_load():
    return run_evaluation(out_name = 'constant_load_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 1',
                       p2x = False,
                       tmp_name = 'test_eval_constant_load',
                       constant_load=True,
                       load_min = 3,
                       )

def update_test_constant_load():
    update_test(out_name='constant_load_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    p2x = False,
                    tmp_name = 'test_eval_constant_load',
                    constant_load=True,
                    load_min = 3,
                    )
    

def load_evaluation_constant_load():
    return load_evaluation(out_name='constant_load_design.csv',
                        design_name = 'Design 1',
                        p2x = False,
)

def test_evaluation_constant_load():
    evaluation_metrics = run_evaluation_constant_load()
    loaded_metrics = load_evaluation_constant_load()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=1e-04)

# ------------------------------------------------------------------------------------------------
# P2X bidirectional

def run_evaluation_P2X_bidirectional():
    return run_evaluation(out_name = 'Evaluation_test_P2X_bidirectional.csv',
                       name = 'Denmark_good_wind',
                       design_name = 'Design 1',
                       p2x = True,
                       p2x_bidirectional = True,
                       tmp_name = 'test_eval_design_P2X_bidirectional',
                       )

def update_test_P2X_bidirectional():
    update_test(out_name='Evaluation_test_P2X_bidirectional.csv',
                    name = 'Denmark_good_wind',
                    design_name = 'Design 1',
                    p2x = True,
                    p2x_bidirectional = True,
                    tmp_name = 'test_eval_P2X_bidirectional',
                    )
    

def load_evaluation_P2X_bidirectional():
    return load_evaluation(out_name='Evaluation_test_P2X_bidirectional.csv',
                        design_name = 'Design 1',
                        p2x = True,)

def test_evaluation_P2X_bidirectional():
    evaluation_metrics = run_evaluation_P2X_bidirectional()
    loaded_metrics = load_evaluation_P2X_bidirectional()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=2e-04)

# ------------------------------------------------------------------------------------------------
# BM

def run_evaluation_BM():
    return run_evaluation(out_name = 'Evaluation_test_BM.csv',
                       name = 'Denmark_good_wind_BM',
                       design_name = 'Design 1',
                       BM = True,
                       tmp_name = 'test_eval_design_BM',
                       )

def update_test_BM():
    update_test(out_name='Evaluation_test_BM.csv',
                    name = 'Denmark_good_wind_BM',
                    design_name = 'Design 1',
                    BM = True,
                    tmp_name = 'test_eval_design_BM',
                    )
    

def load_evaluation_BM():
    return load_evaluation(out_name='Evaluation_test_BM.csv',
                        design_name = 'Design 1',
                        )

def test_evaluation_BM():
    evaluation_metrics = run_evaluation_BM()
    loaded_metrics = load_evaluation_BM()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=2e-04)

        
# # ------------------------------------------------------------------------------------------------
# update_test_design_1()
# update_test_design_2()
# update_test_design_3()
# update_test_design_1_P2X()
# update_test_design_2_P2X()
# update_test_design_3_P2X()
# update_test_PPA()
# update_test_PPA2()
# update_test_constant_load()
# update_test_P2X_bidirectional()
# update_test_BM()
