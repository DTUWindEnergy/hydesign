import tempfile

import chaospy as cp
import numpy as np
import openmdao.api as om
import pandas as pd

from hydesign.ems.ems import expand_to_lifetime
from hydesign.openmdao_wrapper import ComponentWrapper
from hydesign.pv.pv_hybridization import (
    pvp_with_degradation_comp as pvp_with_degradation,
)
from hydesign.reliability_utils import generate_availability_ensamble
from hydesign.utils import (
    get_weights,
)
from hydesign.utils import hybridization_shifted_comp as hybridization_shifted
from hydesign.utils import (
    sample_mean,
)
from hydesign.weather.weather import interpolate_WS_loglog
from hydesign.weather.weather_wind_hybridization import ABL_WD_comp as ABL_WD
from hydesign.weather.weather_wind_hybridization import (
    interpolate_WD_linear,
)
from hydesign.wind.wind_hybridization import (
    get_wind_ts_degradation,
)
from hydesign.wind.wind_hybridization import (
    wpp_with_degradation_comp as wpp_with_degradation,
)


def test_get_weights():
    grid = np.array([-1.0, 0.0, 1.0])
    weights = get_weights(grid, 0.0, 1)
    # first derivative central difference [-0.5, 0, 0.5]
    assert np.allclose(weights[:, 1], [-0.5, 0.0, 0.5])


def test_hybridization_shifted_and_sample_mean():
    life_y = 1
    N_limit = 2
    life_h = (life_y + N_limit) * 365 * 24
    comp = hybridization_shifted(
        N_limit=N_limit, life_y=life_y, N_time=0, life_h=life_h
    )
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()
    SoH = np.ones(life_h)
    prob.set_val("comp.delta_life", 1)
    prob.set_val("comp.SoH", SoH)
    prob.run_model()
    shifted = prob.get_val("comp.SoH_shifted")
    expected = np.concatenate((np.zeros(365 * 24), SoH[: 365 * 24], np.zeros(365 * 24)))
    assert np.array_equal(shifted, expected)
    assert np.allclose(sample_mean(np.vstack([shifted, shifted])), shifted)


def test_component_wrapper_gradients():
    def func(x, y):
        return x**2 + y

    def grad(x, y):
        return [2 * x, 1.0]

    comp = ComponentWrapper(
        inputs=[("x", {"val": 0.0}), ("y", {"val": 0.0})],
        outputs=[("f", {"val": 0.0})],
        function=func,
        gradient_function=grad,
    )
    prob = om.Problem()
    prob.model.add_subsystem("comp", comp)
    prob.setup()
    prob.set_val("comp.x", 3.0)
    prob.set_val("comp.y", 2.0)
    prob.run_model()
    assert prob.get_val("comp.f") == 11.0
    partials = prob.check_partials(method="fd", out_stream=None)
    df_dx = partials["comp"][("f", "x")]["J_fwd"][0][0]
    df_dy = partials["comp"][("f", "y")]["J_fwd"][0][0]
    assert np.isclose(df_dx, 6.0, atol=1e-6)
    assert np.isclose(df_dy, 1.0, atol=1e-6)


def test_weather_interpolation_and_abl_wd(tmp_path):
    index = pd.date_range("2020-01-01", periods=2, freq="h")
    weather = pd.DataFrame(
        {
            "WS_10": [5, 5],
            "WS_50": [10, 10],
            "WD_10": [0, 90],
            "WD_50": [90, 180],
        },
        index=index,
    )
    fn = tmp_path / "weather.csv"
    weather.to_csv(fn)
    interp = interpolate_WD_linear(weather, 30)
    assert np.allclose(interp["WD"].values, [45.0, 135.0])
    ws_interp = interpolate_WS_loglog(weather, 30)
    abl = ABL_WD(weather_fn=str(fn), N_time=2)
    prob = om.Problem()
    prob.model.add_subsystem("abl", abl)
    prob.setup()
    prob.set_val("abl.hh", 30.0)
    prob.run_model()
    assert np.allclose(prob.get_val("abl.wst"), ws_interp.WS.values)
    assert np.allclose(prob.get_val("abl.wdt"), interp.WD.values)


def test_pv_and_wind_hybridization(tmp_path):
    life_y = 25
    N_limit = 1
    life_h = (life_y + N_limit) * 8760
    pv_deg = [0, 0, 0.2, 0.5, 0.5, 0.5]

    pv_comp = pvp_with_degradation(
        N_limit=N_limit, life_y=life_y, life_h=life_h, pv_deg=pv_deg
    )
    prob_pv = om.Problem()
    prob_pv.model.add_subsystem("pv", pv_comp)
    prob_pv.setup()
    prob_pv.set_val("pv.delta_life", 1)
    solar = np.ones(life_h)
    prob_pv.set_val("pv.solar_t_ext", solar)
    prob_pv.run_model()
    t_over_year = np.arange(life_h) / (365 * 24)
    pv_deg_yr = [0, 1, 1.0001, 26, 26.0001, 26]
    expected = (1 - np.interp(t_over_year, pv_deg_yr, pv_deg)) * solar
    assert np.allclose(prob_pv.get_val("pv.solar_t_ext_deg"), expected)

    ws = np.array([0, 5, 10, 15, 20])
    pcw = np.array([0, 0.5, 1, 1, 1])
    wst = np.array([6, 7])
    wind_deg = [0, 0.2, 0.5, 0.5, 0.5, 0.5]
    wind_comp = wpp_with_degradation(
        N_limit=N_limit,
        life_y=life_y,
        N_time=2,
        life_h=life_h,
        N_ws=len(ws),
        wpp_efficiency=0.9,
        wind_deg=wind_deg,
        share_WT_deg_types=0.5,
    )
    prob_wind = om.Problem()
    prob_wind.model.add_subsystem("wind", wind_comp)
    prob_wind.setup()
    prob_wind.set_val("wind.delta_life", 1)
    prob_wind.set_val("wind.ws", ws)
    prob_wind.set_val("wind.pcw", pcw)
    prob_wind.set_val("wind.wst", wst)
    prob_wind.run_model()
    wind_deg_yr = [0, 1, 1.0001, 26, 26.0001, 26]
    wst_ext = expand_to_lifetime(wst, life=life_h)
    expected_wind = 0.9 * get_wind_ts_degradation(
        ws, pcw, wst_ext, wind_deg_yr, wind_deg, life_h, share=0.5
    )
    assert np.allclose(prob_wind.get_val("wind.wind_t_ext_deg"), expected_wind)


def test_generate_availability_ensamble_small():
    ds = generate_availability_ensamble(
        ts_start="2030-01-01 00:00",
        ts_end="2030-01-01 05:00",
        ts_freq="1h",
        seeds=[0, 1],
        component_name="WT",
        MTTF=1,
        MTTR=1,
        N_components=2,
        sampling_const=1,
        pdf=cp.Exponential,
    )
    assert ds.dims["seed"] == 2
    assert ds.dims["component"] == 2
    assert ds.TTF_indices.shape == ds.TTR_indices.shape
