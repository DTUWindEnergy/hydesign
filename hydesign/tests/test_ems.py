# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import numpy as np
# import pandas as pd
# import pytest
import pickle

from hydesign.tests.test_files import tfp
from hydesign.ems.ems import ems_cplex, operation_solar_batt_deg

# ------------------------------------------------------------------------------------------------
def run_ems():
    with open(tfp+'ems_input_ems.pickle', 'rb') as f:
        input_ems = pickle.load(f)
    ems_out = ems_cplex(**input_ems)
    return ems_out

def load_ems():
    with open(tfp+'ems_output_ems.pickle','rb') as f:
        ems_out = pickle.load(f)
    return ems_out

def update_test_ems():
    ems_out = run_ems()
    with open(tfp+'ems_output_ems.pickle','wb') as f:
        pickle.dump(ems_out, f)
    

def test_ems():
    ems_out = run_ems()
    ems_out_data = load_ems()
    # keys = ['P_HPP_ts', 'P_curtailment_ts', 'P_charge_discharge_ts', 'E_SOC_ts', 'penalty_ts']
    for i in range(len(ems_out)):
        np.testing.assert_allclose(ems_out[i], ems_out_data[i])
        #print(np.allclose(ems_out[i], ems_out_data[i]))
# ------------------------------------------------------------------------------------------------
def run_operation_with_deg():
    with open(tfp+'ems_input_ems_longterm.pickle', 'rb') as f:
        input_ems_long = pickle.load(f)
    out_operation_with_deg = operation_solar_batt_deg(**input_ems_long)
    return out_operation_with_deg

def load_operation_with_deg():
    with open(tfp+'ems_output_ems_longterm.pickle','rb') as f:
        out_operation_with_deg = pickle.load(f)
    return out_operation_with_deg

def test_operation_with_deg():
    out_operation_with_deg = run_operation_with_deg()
    out_operation_with_deg_data = load_operation_with_deg()
    for i in range(len(out_operation_with_deg)):
        np.testing.assert_allclose(out_operation_with_deg[i], out_operation_with_deg_data[i])
        #print(np.allclose(out_operation_with_deg[i], out_operation_with_deg_data[i]))

