import numpy as np
import pandas as pd
import os

from hydesign.tests.test_files import tfp
from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.assembly.hpp_assembly_P2X import hpp_model_P2X
from hydesign.assembly.hpp_assembly_constantoutput import hpp_model_constant_output
from hydesign.assembly.hpp_assembly_P2X_bidrectional import hpp_model_P2X_bidirectional
from hydesign.assembly.hpp_assembly_BM import hpp_model as hpp_model_BM
from hydesign.assembly.hpp_assembly_hifi_dems import hpp_model as hpp_model_hifi_ems
from hydesign.assembly.hpp_assembly_solarX import hpp_model_solarX
from hydesign.examples import examples_filepath


def run_evaluation(out_name, name, design_name, tmp_name, case, **kwargs):
    output_df = pd.read_csv(tfp+out_name, index_col=0, sep=';')
    examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
    ex_site = examples_sites.loc[examples_sites.name == name]
    longitude = ex_site['longitude'].values[0]
    latitude = ex_site['latitude'].values[0]
    altitude = ex_site['altitude'].values[0]
    input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
    sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
    H2_demand_fn = examples_filepath+ex_site['H2_demand_col'].values[0]

    if case not in ['HiFiEMS', 'SolarX']:
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
    elif case in ['SolarX']:
        sf_area = output_df.loc['sf_area', design_name]
        tower_height = output_df.loc['tower_height', design_name]
        area_cpv_receiver_m2 = output_df.loc['area_cpv_receiver_m2', design_name]
        heat_exchanger_capacity = output_df.loc['heat_exchanger_capacity', design_name]
        p_rated_st = output_df.loc['p_rated_st', design_name]
        v_molten_salt_tank_m3 = output_df.loc['v_molten_salt_tank_m3', design_name]
        area_cst_receiver_m2 = output_df.loc['area_cst_receiver_m2', design_name]
        area_dni_reactor_biogas_h2 = output_df.loc['area_dni_reactor_biogas_h2', design_name]
        area_el_reactor_biogas_h2 = output_df.loc['area_el_reactor_biogas_h2', design_name]
        x = [
            # sizing variables
            # sf
            sf_area,
            tower_height,

            # cpv
            area_cpv_receiver_m2,

            # cst
            heat_exchanger_capacity,
            p_rated_st,
            v_molten_salt_tank_m3,
            area_cst_receiver_m2,

            # bigas_h2
            area_dni_reactor_biogas_h2,
            area_el_reactor_biogas_h2,
        ]
    else:
        x = None
    
    if case=='base':
        hpp = hpp_model(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn)
        
    elif case=='bm':
        input_HA_ts_fn = examples_filepath+ex_site['input_HA_ts_fn'].values[0]
        price_up_ts_fn = examples_filepath+ex_site['price_up_ts'].values[0]
        price_dwn_ts_fn = examples_filepath+ex_site['price_dwn_ts'].values[0]
        price_col = ex_site['price_col'].values[0]

        hpp = hpp_model_BM(            
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            input_HA_ts_fn = input_HA_ts_fn,
            price_up_ts_fn = price_up_ts_fn,
            price_dwn_ts_fn = price_dwn_ts_fn,
            price_col = price_col,)
        x = [10.0, 350.0, 5.0, 70, 7.0, 0.0, 25.0, 180.0, 1.0, 1.0, 4.0, 10.0]
        
    elif case=='p2x_bidirectional':
        hpp = hpp_model_P2X_bidirectional(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            H2_demand_fn = H2_demand_fn,
            electrolyzer_eff_curve_name = 'Alkaline electrolyzer H2 production',
            penalty_factor_H2=0.5,
            )
        x=[50, 300, 10, 40, 10, 0, 45, 180, 1.5, 40, 4, 5, 250, 5000]
        
    elif case=='constant_output':
        hpp = hpp_model_constant_output(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            load_min=kwargs['load_min'],)
                
    elif case=='ppa':
        hpp = hpp_model(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            ppa_price=kwargs['PPA'],)
        
    elif case=='p2x':
        hpp = hpp_model_P2X(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            max_num_batteries_allowed = 10,
            work_dir = './',
            sim_pars_fn = sim_pars_fn,
            input_ts_fn = input_ts_fn,
            H2_demand_fn=H2_demand_fn) 

        ptg_MW = output_df.loc['ptg_MW [MW]','Design 1']
        HSS_kg = output_df.loc['HSS_kg [kg]','Design 1']
        x.extend([ptg_MW, HSS_kg])

        
    elif case=='HiFiEMS':
        sim_pars_fn = os.path.join(examples_filepath, 'Europe/hpp_pars_HiFiEMS.yml')
        hpp = hpp_model_hifi_ems(sim_pars_fn=sim_pars_fn,
                        input_ts_da=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_DA.csv'),
                        input_ts_ha=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_HA.csv'),
                        input_ts_rt=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_RT.csv'),
                        market_fn=os.path.join(examples_filepath, 'HiFiEMS_inputs/Market/Market2021.csv'),)
        x = [20, 350, 10, 12, 6, 10, 25, 180, 1.5, 40, 3,]

    elif case=='SolarX':
        batch_size = 1 * 24
        sim_pars_fn = examples_filepath + "solarX/hpp_pars.yml"
        hpp = hpp_model_solarX(
            latitude, longitude, altitude,  # Geographical data for the site
            work_dir='./',  # Directory for saving outputs
            sim_pars_fn=sim_pars_fn,  # Simulation parameters
            input_ts_fn=input_ts_fn,  # Input time series (weather, prices, etc.)
            batch_size=batch_size,)
        x = [10000.0, 100, 10, 10, 10, 1000, 10, 5, 5]

    else:
        print(f'case type not implemented: {case}')
        
    outs = hpp.evaluate(*x)
    tmp_path = tfp + 'tmp'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
    hpp.evaluation_in_csv(os.path.join(tmp_path, tmp_name), longitude, latitude, altitude, x, outs)
    return outs

def update_test(out_name, name, design_name, tmp_name, case, **kwargs):
    output_df = pd.read_csv(
        tfp+out_name,
        index_col=0, 
        sep=';')
    run_evaluation(out_name, name, design_name, tmp_name, case, **kwargs)
    eval_df = pd.read_csv(os.path.join(tfp + 'tmp', tmp_name + '.csv'))
    output_df[design_name] = eval_df.T[0]
    output_df.to_csv(tfp+out_name, sep=';')
    
def load_evaluation(out_name,design_name,case,):
    output_df = pd.read_csv(tfp+out_name, index_col=0, sep=';')
    if case in ['p2x', 'p2x_bidirectional']:
        load_file = np.array(output_df.iloc[17:][design_name])
    elif case in ['HiFiEMS']:
        load_file = np.array(output_df.iloc[14:][design_name])
    elif case in ['SolarX']:
        load_file = np.array(output_df.iloc[12:][design_name])
    else:
        load_file = np.array(output_df.iloc[15:][design_name])
    return load_file
    
    
# ------------------------------------------------------------------------------------------------
# design 1

def run_evaluation_design_1():
    return run_evaluation(out_name = 'France_good_wind_design.csv',
                       name = 'France_good_wind',
                       design_name = 'Design 1',
                       case = 'base',
                       tmp_name = 'test_eval_design_1',
                       )

def update_test_design_1():
    update_test(out_name='France_good_wind_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    case = 'base',
                    tmp_name = 'test_eval_design_1',
                    )
    

def load_evaluation_design_1():
    return load_evaluation(out_name='France_good_wind_design.csv',
                           design_name = 'Design 1',
                           case = 'base',
                           )

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
                       case = 'base',
                       tmp_name = 'test_eval_design_2',
                       )

def update_test_design_2():
    update_test(out_name='France_good_wind_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 2',
                    case = 'base',
                    tmp_name = 'test_eval_design_2',
                    )
    

def load_evaluation_design_2():
    return load_evaluation(out_name='France_good_wind_design.csv',
                           design_name = 'Design 2',
                           case = 'base',
                           )


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
                       case = 'base',
                       tmp_name = 'test_eval_design_3',
                       )

def update_test_design_3():
    update_test(out_name='France_good_wind_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 3',
                    case = 'base',
                    tmp_name = 'test_eval_design_3',
                    )
    

def load_evaluation_design_3():
    return load_evaluation(out_name='France_good_wind_design.csv',
                           design_name = 'Design 3',
                           case = 'base',
                           )


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
                       case = 'p2x',
                       tmp_name = 'test_eval_design_1_P2X',
                       )

def update_test_design_1_P2X():
    update_test(out_name='Evaluation_test_P2X.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    case = 'p2x',
                    tmp_name = 'test_eval_design_1_P2X',
                    )
    

def load_evaluation_design_1_P2X():
    return load_evaluation(out_name='Evaluation_test_P2X.csv',
                           design_name = 'Design 1',
                           case = 'p2x',
                           )

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
                       case = 'p2x',
                       tmp_name = 'test_eval_design_2_P2X',
                       )

def update_test_design_2_P2X():
    update_test(out_name='Evaluation_test_P2X.csv',
                    name = 'Indian_site_good_wind',
                    design_name = 'Design 2',
                    case = 'p2x',
                    tmp_name = 'test_eval_design_2_P2X',
                    )
    

def load_evaluation_design_2_P2X():
    return load_evaluation(out_name='Evaluation_test_P2X.csv',
                           design_name = 'Design 2',
                           case = 'p2x',
                           )

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
                       case = 'p2x',
                       tmp_name = 'test_eval_design_3_P2X',
                       )

def update_test_design_3_P2X():
    update_test(out_name='Evaluation_test_P2X.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 3',
                    case = 'p2x',
                    tmp_name = 'test_eval_design_3_P2X',
                    )
    

def load_evaluation_design_3_P2X():
    return load_evaluation(out_name='Evaluation_test_P2X.csv',
                           design_name = 'Design 3',
                           case = 'p2x',
                           )


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
                       case = 'ppa',
                       tmp_name = 'test_eval_PPA',
                       PPA=21.4,
                       )

def update_test_PPA():
    update_test(out_name='PPA_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    case = 'ppa',
                    tmp_name = 'test_eval_PPA',
                    PPA=21.4,
                    )
    

def load_evaluation_PPA():
    return load_evaluation(out_name='PPA_design.csv',
                           design_name = 'Design 1',
                           case = 'ppa',
                           )

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
                       case = 'ppa',
                       tmp_name = 'test_eval_PPA2',
                       PPA=41.4,
                       )

def update_test_PPA2():
    update_test(out_name='PPA_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 2',
                    case = 'ppa',
                    tmp_name = 'test_eval_PPA2',
                    PPA=41.4,
                    )
    

def load_evaluation_PPA2():
    return load_evaluation(out_name='PPA_design.csv',
                           design_name = 'Design 2',
                           case = 'ppa',
                           )

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
                       case = 'constant_output',
                       tmp_name = 'test_eval_constant_load',
                       load_min = 3,
                       )

def update_test_constant_load():
    update_test(out_name='constant_load_design.csv',
                    name = 'France_good_wind',
                    design_name = 'Design 1',
                    case = 'constant_output',
                    tmp_name = 'test_eval_constant_load',
                    load_min = 3,
                    )
    

def load_evaluation_constant_load():
    return load_evaluation(out_name='constant_load_design.csv',
                           design_name = 'Design 1',
                           case = 'constant_output',
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
                       case = 'p2x_bidirectional',
                       tmp_name = 'test_eval_design_P2X_bidirectional',
                       )

def update_test_P2X_bidirectional():
    update_test(out_name='Evaluation_test_P2X_bidirectional.csv',
                    name = 'Denmark_good_wind',
                    design_name = 'Design 1',
                    case = 'p2x_bidirectional',
                    tmp_name = 'test_eval_P2X_bidirectional',
                    )
    

def load_evaluation_P2X_bidirectional():
    return load_evaluation(out_name='Evaluation_test_P2X_bidirectional.csv',
                           design_name = 'Design 1',
                           case = 'p2x_bidirectional',
                           )

def test_evaluation_P2X_bidirectional():
    evaluation_metrics = run_evaluation_P2X_bidirectional()
    loaded_metrics = load_evaluation_P2X_bidirectional()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=3e-04)

# ------------------------------------------------------------------------------------------------
# BM

def run_evaluation_BM():
    return run_evaluation(out_name = 'Evaluation_test_BM.csv',
                       name = 'Denmark_good_wind_BM',
                       design_name = 'Design 1',
                       case = 'bm',
                       tmp_name = 'test_eval_design_BM',
                       )

def update_test_BM():
    update_test(out_name='Evaluation_test_BM.csv',
                    name = 'Denmark_good_wind_BM',
                    design_name = 'Design 1',
                    case = 'bm',
                    tmp_name = 'test_eval_design_BM',
                    )
    

def load_evaluation_BM():
    return load_evaluation(out_name='Evaluation_test_BM.csv',
                        design_name = 'Design 1',
                        case = 'bm',
                        )

def test_evaluation_BM():
    evaluation_metrics = run_evaluation_BM()
    loaded_metrics = load_evaluation_BM()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i], rtol=6e-03)

# ------------------------------------------------------------------------------------------------
# HiFiEMS

def run_evaluation_HiFiEMS():
    return run_evaluation(out_name = 'Evaluation_test_HiFiEMS.csv',
                       name = 'Denmark_good_wind',
                       design_name = 'Design 1',
                       case = 'HiFiEMS',
                       tmp_name = 'test_eval_design_HiFiEMS',
                       )

def update_test_HiFiEMS():
    update_test(out_name='Evaluation_test_HiFiEMS.csv',
                    name = 'Denmark_good_wind',
                    design_name = 'Design 1',
                    case = 'HiFiEMS',
                    tmp_name = 'test_eval_design_HiFiEMS',
                    )
    

def load_evaluation_HiFiEMS():
    return load_evaluation(out_name='Evaluation_test_HiFiEMS.csv',
                           design_name = 'Design 1',
                           case = 'HiFiEMS',
                           )

def test_evaluation_HiFiEMS():
    evaluation_metrics = run_evaluation_HiFiEMS()
    loaded_metrics = load_evaluation_HiFiEMS()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i])


# ------------------------------------------------------------------------------------------------
# SolarX

def run_evaluation_SolarX():
    return run_evaluation(out_name='Evaluation_test_SolarX.csv',
                          name='Denmark_good_solar',
                          design_name='Design 1',
                          case='SolarX',
                          tmp_name='test_eval_design_SolarX',
                          )


def update_test_SolarX():
    update_test(out_name='Evaluation_test_SolarX.csv',
                name='Denmark_good_solar',
                design_name='Design 1',
                case='SolarX',
                tmp_name='test_eval_design_SolarX',
                )


def load_evaluation_SolarX():
    return load_evaluation(out_name='Evaluation_test_SolarX.csv',
                           design_name='Design 1',
                           case='SolarX',
                           )


def test_evaluation_SolarX():
    evaluation_metrics = run_evaluation_SolarX()
    loaded_metrics = load_evaluation_SolarX()
    for i in range(len(loaded_metrics)):
        np.testing.assert_allclose(evaluation_metrics[i], loaded_metrics[i])
        
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
# update_test_HiFiEMS()
# update_test_SolarX()

# test_evaluation_design_1()
# test_evaluation_design_2()
# test_evaluation_design_3()
# test_evaluation_design_1_P2X()
# test_evaluation_design_2_P2X()
# test_evaluation_design_3_P2X()
# test_evaluation_PPA()
# test_evaluation_PPA2()
# test_evaluation_constant_load()
# test_evaluation_P2X_bidirectional()
# test_evaluation_BM()
# test_evaluation_HiFiEMS()
# test_evaluation_SolarX()
