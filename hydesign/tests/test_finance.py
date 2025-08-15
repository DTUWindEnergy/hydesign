# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from hydesign.finance.finance import calculate_NPV_IRR, calculate_WACC
from hydesign.tests.test_files import tfp


# ------------------------------------------------------------------------------------------------
def run_WACC():
    with open(tfp + "finance_input_WACC.pickle", "rb") as f:
        input_WACC = pickle.load(f)
    WACC_out = calculate_WACC(**input_WACC)
    return WACC_out


def load_WACC():
    with open(tfp + "finance_output_WACC.pickle", "rb") as f:
        WACC_out = pickle.load(f)
    return WACC_out


def test_WACC():
    WACC_out = run_WACC()
    WACC_out_data = load_WACC()
    for i in range(len(WACC_out)):
        np.testing.assert_allclose(WACC_out[i], WACC_out_data[i])
        # print(np.allclose(WACC_out[i], WACC_out_data[i]))


# ------------------------------------------------------------------------------------------------
def run_NPV():
    with open(tfp + "finance_input_NPV.pickle", "rb") as f:
        input_NPV = pickle.load(f)
    NPV_out = calculate_NPV_IRR(**input_NPV)
    return NPV_out


def load_NPV():
    with open(tfp + "finance_output_NPV.pickle", "rb") as f:
        NPV_out = pickle.load(f)
    return NPV_out


def test_NPV():
    NPV_out = run_NPV()
    NPV_out_data = load_NPV()
    for i in range(len(NPV_out)):
        np.testing.assert_allclose(NPV_out[i], NPV_out_data[i])
        # print(np.allclose(NPV_out[i], NPV_out_data[i]))


def update_NPV():
    NPV_out = run_NPV()
    with open(tfp + "finance_output_NPV.pickle", "wb") as f:
        pickle.dump(NPV_out, f)


# update_NPV()
