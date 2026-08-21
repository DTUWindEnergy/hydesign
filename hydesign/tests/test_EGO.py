# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 14:43:04 2022

@author: mikf
"""
import pickle

import numpy as np
import pandas as pd
import pytest

from hydesign import Parallel_EGO
from hydesign.Parallel_EGO import smt_minor  # opt_sm,
from hydesign.Parallel_EGO import (
    EI,
    KB,
    LCB,
    KStd,
    eval_sm,
    get_candiate_points,
    get_mixint_context,
    get_sm,
    smt_major,
)
from hydesign.tests.test_files import tfp

# import smt
# smt_version = smt.__version__.split('.')
# major, minor = smt_version[:2]


def get_test_sm():
    nt = 100
    data = pd.read_csv(
        tfp + "test_data.csv", sep=";", nrows=nt
    )  # y=A2^2+B2^3+SIN(C2)+ATAN(D2)+1/E2+PI()*F2+EXP(G2/10)+H2^4+LN(I2)
    x = data[[f"x{int(i + 1)}" for i in range(9)]].values
    y = data["y2"].values.reshape(nt, 1)
    return get_sm(x, y)


def save_sm():
    sm = get_test_sm()
    with open(tfp + f"sm_{smt_major}_{smt_minor}.pkl", "wb") as f:
        pickle.dump(sm, f)


def load_sm():
    with open(tfp + f"sm_{smt_major}_{smt_minor}.pkl", "rb") as f:
        sm = pickle.load(f)
    return sm


sm = load_sm()
# sm = get_test_sm()
fmins = [1, 10, 50, 500, 10000]
point = np.array([10, 1, 0, 142, 2, 10, 0, 0, 1]).reshape(
    (1, 9)
)  # analytical: 135.4796807


def get_data1():
    df = pd.read_csv(
        tfp + f"test_surrogate_models_{smt_major}_{smt_minor}.csv", sep=";"
    )
    return df


def generate_data1():
    df = pd.DataFrame(
        {
            "LCB": float(LCB(sm, point)[0]),
            "EI": float(EI(sm, point)[0]),
            "KStd": float(KStd(sm, point)[0]),
            "KB": float(KB(sm, point)[0]),
        },
        index=[0],
    )
    df.to_csv(
        tfp + f"test_surrogate_models_{smt_major}_{smt_minor}.csv", sep=";", index=False
    )


def get_data2(fmin):
    data2 = pd.read_csv(
        tfp + f"sm_pred_test_data_{smt_major}_{smt_minor}.csv", sep=";"
    ).values
    n = fmins.index(fmin)
    a = data2[n * 5 : n * 5 + 5, :9]
    b = data2[n * 5 : n * 5 + 5, 9]
    return a, b


def generate_data2():
    new_data = np.zeros((25, 10))
    for n, fmin in enumerate(fmins):
        a, b = eval_sm(sm, mixint, npred=5, fmin=fmin)
        new_data[n * 5 : n * 5 + 5, :9] = a
        new_data[n * 5 : n * 5 + 5, 9] = b.ravel()
    df = pd.DataFrame(new_data, columns=[f"x{i + 1}" for i in range(9)] + ["y"])
    df.to_csv(
        tfp + f"sm_pred_test_data_{smt_major}_{smt_minor}.csv", sep=";", index=False
    )


df = get_data1()


def test_LCB():
    res = LCB(sm, point)
    np.testing.assert_allclose(float(res[0][0]), float(df.LCB.iloc[0]))


def test_EI():
    res = EI(sm, point)
    np.testing.assert_allclose(float(res[0][0]), float(df.EI.iloc[0]))


def test_KStd():
    res = KStd(sm, point)
    np.testing.assert_allclose(float(res[0][0]), float(df.KStd.iloc[0]), rtol=1.3e-7)


def test_KB():
    res = KB(sm, point)
    np.testing.assert_allclose(float(res[0][0]), float(df.KB.iloc[0]))


variables = {
    "a": {"var_type": "design", "limits": [4, 22], "types": "int"},
    "b": {"var_type": "design", "limits": [0.1, 4.5], "types": "float"},
    "c": {"var_type": "design", "limits": [-0.65, 1.75], "types": "float"},
    "d": {"var_type": "design", "limits": [8, 270], "types": "int"},
    "e": {"var_type": "design", "limits": [1, 4.9], "types": "float"},
    "f": {"var_type": "design", "limits": [-22, 43], "types": "int"},
    "g": {"var_type": "design", "limits": [-10, 9], "types": "float"},
    "h": {"var_type": "design", "limits": [-10, 9], "types": "float"},
    "i": {"var_type": "design", "limits": [1.1, 1.34], "types": "float"},
}
mixint = get_mixint_context(variables, seed=0)


@pytest.mark.parametrize("fmin", fmins)
def test_eval_sm(fmin):
    a, b = eval_sm(sm, mixint, npred=5, fmin=fmin)
    a_ref, b_ref = get_data2(fmin)
    np.testing.assert_allclose(a, a_ref)
    np.testing.assert_allclose(b.ravel(), b_ref, rtol=1e-6)


def test_get_candidate_points():
    xpred, ypred_LB = eval_sm(sm, mixint, npred=5, fmin=1e3)
    print(xpred, ypred_LB)
    xnew = get_candiate_points(xpred, ypred_LB, n_clusters=1, quantile=1e-4)
    np.testing.assert_allclose(
        xnew, np.array([[9, 4.06, 1.51, 86, 2.95, -3, -4.3, -0.5, 1.316]])
    )


def test_design_variable_helpers_prepare_model_inputs():
    """The EGO helpers round design values and insert fixed model values."""
    variables = {
        "turbines": {
            "var_type": "design",
            "limits": [1, 5],
            "types": "int",
        },
        "spacing": {
            "var_type": "design",
            "limits": [0.0, 1.0],
            "types": "resolution",
            "resolution": 0.25,
        },
        "capacity": {"var_type": "fixed", "value": 10.0},
    }
    values = np.array([[1.6, 0.62]])

    design, fixed = Parallel_EGO.get_design_vars(variables)
    rounded = Parallel_EGO.cast_to_mixint(values.copy(), variables)
    expanded = Parallel_EGO.expand_x_for_model_eval(
        rounded,
        {
            "list_vars": ["turbines", "capacity", "spacing"],
            "variables": variables,
            "design_vars": design,
            "fixed_vars": fixed,
        },
    )

    assert design == ["turbines", "spacing"]
    assert fixed == ["capacity"]
    np.testing.assert_array_equal(Parallel_EGO.get_limits(variables), [[1, 5], [0, 1]])
    assert Parallel_EGO.get_xtypes(variables) == ["int", "resolution"]
    np.testing.assert_allclose(expanded, [[2.0, 10.0, 0.5]])


def test_candidate_helpers_create_distinct_nearby_points():
    point = np.array([[0.2, 0.8]])

    extremes = Parallel_EGO.extreme_around_point(point)
    nearby = Parallel_EGO.perturbe_around_point(point, step=0.1)
    unique_x, unique_y = Parallel_EGO.concat_to_existing(
        point,
        np.array([[1.0]]),
        np.vstack([point, nearby[:1]]),
        np.array([[1.0], [2.0]]),
    )

    np.testing.assert_allclose(
        extremes, [[0.0, 0.8], [0.2, 0.0], [1.0, 0.8], [0.2, 1.0]]
    )
    np.testing.assert_allclose(nearby, [[0.3, 0.8], [0.2, 0.9], [0.1, 0.8], [0.2, 0.7]])
    assert len(unique_x) == len(unique_y) == 2


def test_surrogate_optimizer_finds_a_quadratic_minimum():
    class QuadraticSurrogate:
        def predict_values(self, values):
            return np.sum((values - 0.4) ** 2, axis=1, keepdims=True)

        def predict_derivatives(self, values, kx):
            return 2 * (values[:, [kx]] - 0.4)

    mixint = type("MixInt", (), {"get_unfolded_dimension": lambda self: 2})()
    optimum = Parallel_EGO.opt_sm(QuadraticSurrogate(), mixint, np.array([[0.1, 0.9]]))

    np.testing.assert_allclose(optimum, [[0.4, 0.4]], atol=1e-5)


# generate_data1()
# generate_data2()
