# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import pickle

import numpy as np

from hydesign.ems.ems import operation_solar_batt_deg
from hydesign.ems_modular.ems_core import ems_core
from hydesign.tests.test_files import tfp

# import pandas as pd
# import pytest


# ------------------------------------------------------------------------------------------------
def run_ems():
    with open(tfp + "ems_input_ems.pickle", "rb") as f:
        input_ems = pickle.load(f)
    outputs_ems_core = ems_core(**input_ems)
    ems_out = (
        outputs_ems_core["P_HPP_ts"],
        outputs_ems_core["P_curtailment_ts"],
        outputs_ems_core["P_charge_discharge_ts"],
        outputs_ems_core["E_SOC_ts"],
        outputs_ems_core["penalty_ts"],
    )
    return ems_out


def load_ems():
    with open(tfp + "ems_output_ems.pickle", "rb") as f:
        ems_out = pickle.load(f)
    return ems_out


def update_test_ems():
    ems_out = run_ems()
    with open(tfp + "ems_output_ems.pickle", "wb") as f:
        pickle.dump(ems_out, f)


def test_ems():
    ems_out = run_ems()
    ems_out_data = load_ems()

    for i in range(len(ems_out)):
        if i == 3:
            continue
        sum_out = np.sum(ems_out[i])
        sum_expected = np.sum(ems_out_data[i])
        np.testing.assert_allclose(
            sum_out,
            sum_expected,
            rtol=1e-8,
            atol=1e-10,
            err_msg=f"Sum mismatch for item {i}: got {sum_out}, expected {sum_expected}",
        )


# ------------------------------------------------------------------------------------------------
def run_operation_with_deg():
    with open(tfp + "ems_input_ems_longterm.pickle", "rb") as f:
        input_ems_long = pickle.load(f)
    out_operation_with_deg = operation_solar_batt_deg(**input_ems_long)
    return out_operation_with_deg


def load_operation_with_deg():
    with open(tfp + "ems_output_ems_longterm.pickle", "rb") as f:
        out_operation_with_deg = pickle.load(f)
    return out_operation_with_deg


def test_operation_with_deg():
    out_operation_with_deg = run_operation_with_deg()
    out_operation_with_deg_data = load_operation_with_deg()
    for i in range(len(out_operation_with_deg)):
        np.testing.assert_allclose(
            out_operation_with_deg[i], out_operation_with_deg_data[i]
        )
        # print(np.allclose(out_operation_with_deg[i], out_operation_with_deg_data[i]))
