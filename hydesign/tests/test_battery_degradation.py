# -*- coding: utf-8 -*-
"""
Created on 24/01/2023

@author: jumu
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from hydesign.battery_degradation import (
    Linear_Degfun,
    RFcount,
    battery_replacement,
    degradation,
)
from hydesign.tests.test_files import tfp


# ------------------------------------------------------------------------------------------------
def run_RFcount():
    np.random.seed(0)
    SoC = np.random.rand(24 * 365 * 25)
    RF_out_ = RFcount(SoC)
    return RF_out_[:-1]


def load_RFcount():
    with open(tfp + "battery_degradation_RF_count.pickle", "rb") as f:
        RF_out_ = pickle.load(f)
    return RF_out_


def test_RFcount():
    RF_out_ = run_RFcount()
    RF_out_data = load_RFcount()
    for i in range(len(RF_out_)):
        np.testing.assert_allclose(RF_out_[i], RF_out_data[i])
        # print(np.allclose(RF_out_[i], RF_out_data[i]))


# ------------------------------------------------------------------------------------------------
def run_Linear_Degfun():
    RF_out_data = load_RFcount()

    rf_DoD = RF_out_data[0]
    rf_SoC = RF_out_data[1]
    rf_count = RF_out_data[2]
    rf_i_start = RF_out_data[3]
    avr_tem = 25

    LLoC_hist = Linear_Degfun(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem)
    return LLoC_hist


def load_Linear_Degfun():
    with open(tfp + "battery_degradation_Linear_Degfun.pickle", "rb") as f:
        LLoC_hist = pickle.load(f)
    return LLoC_hist


def test_Linear_Degfun():
    LLoC_hist = run_Linear_Degfun()
    LLoC_hist_data = load_Linear_Degfun()
    np.testing.assert_allclose(LLoC_hist, LLoC_hist_data)
    # print(np.allclose(LLoC_hist, LLoC_hist_data))


# ------------------------------------------------------------------------------------------------
def run_degradation():
    RF_out_data = load_RFcount()

    rf_DoD = RF_out_data[0]
    rf_SoC = RF_out_data[1]
    rf_count = RF_out_data[2]
    rf_i_start = RF_out_data[3]
    avr_tem = 25

    deg_out = degradation(rf_DoD, rf_SoC, rf_count, rf_i_start, avr_tem, LLoC_0=0)
    return deg_out


def load_degradation():
    with open(tfp + "battery_degradation_degradation.pickle", "rb") as f:
        deg_out = pickle.load(f)
    return deg_out


def test_degradation():
    deg_out = run_degradation()
    deg_outdata = load_degradation()
    for i in range(len(deg_out)):
        np.testing.assert_allclose(deg_out[i], deg_outdata[i])


# ------------------------------------------------------------------------------------------------
def run_battery_replacement():
    RF_out_data = load_RFcount()

    rf_DoD = RF_out_data[0]
    rf_SoC = RF_out_data[1]
    rf_count = RF_out_data[2]
    rf_i_start = RF_out_data[3]
    avr_tem = 25
    min_LoH = 0.7
    n_steps_in_LoH = 30
    num_batteries = 3

    br_out = battery_replacement(
        rf_DoD,
        rf_SoC,
        rf_count,
        rf_i_start,
        avr_tem,
        min_LoH,
        n_steps_in_LoH,
        num_batteries,
    )
    return br_out


def load_battery_replacement():
    with open(tfp + "battery_degradation_battery_replacement.pickle", "rb") as f:
        br_out = pickle.load(f)
    return br_out


def test_battery_replacement():
    br_out = run_battery_replacement()
    br_out_data = load_battery_replacement()
    for i in range(len(br_out)):
        np.testing.assert_allclose(br_out[i], br_out_data[i])
