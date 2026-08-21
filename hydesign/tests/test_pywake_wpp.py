from types import SimpleNamespace

import numpy as np
import pytest

from hydesign.wind import pywake_wpp


class ArrayResult:
    """Small stand-in for the xarray values returned by PyWake."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def sum(self, _dimension):
        return self

    def to_numpy(self):
        return self.values


class FarmResult:
    def __init__(self, power, yaw):
        self.Power = ArrayResult(power)
        self.yaw = ArrayResult(yaw)


def test_pywake_wpp_computes_smart_and_actual_operation(monkeypatch):
    """Run the main wake-steering and lifetime-cost workflow with tiny fakes."""
    n_turbines, n_hours = 2, 3

    monkeypatch.setattr(pywake_wpp, "_design_friendly_import_error", None)
    monkeypatch.setattr(pywake_wpp, "_wind_farm_loads_import_error", None)
    monkeypatch.setattr(pywake_wpp, "LOAD_PATHS", ".")
    model = SimpleNamespace(metadata={"Wohler_exponent": np.array(4.0)})
    monkeypatch.setattr(
        pywake_wpp.TensorFlowModel,
        "load_h5",
        lambda **_kwargs: model,
    )
    monkeypatch.setattr(
        pywake_wpp,
        "easy_yaw_gnn",
        lambda *_args, **_kwargs: np.full((n_turbines, n_hours), 5.0),
    )

    def fake_loads(_surrogates, simulation, *_args, **_kwargs):
        scale = 1.1 if np.any(simulation.yaw.values) else 1.0
        return ArrayResult(np.full((n_turbines, n_hours, 1), scale))

    monkeypatch.setattr(pywake_wpp, "predict_loads_rotor_average", fake_loads)
    monkeypatch.setattr(
        pywake_wpp,
        "lifetime_aware_model",
        lambda *_args: (0.0, 1.0, 1.1, 1.0, 1.1),
    )

    def farm(**kwargs):
        controlled = np.any(kwargs["yaw"])
        power = np.full(n_hours, 2.2e6 if controlled else 2.0e6)
        return FarmResult(power, kwargs["yaw"])

    wpp = pywake_wpp.PyWakeWPP(
        n_wt=n_turbines,
        x=np.array([0.0, 500.0]),
        y=np.zeros(n_turbines),
        tilt=np.zeros((n_turbines, n_hours)),
        time_stamp=np.arange(n_hours),
        sim_pars={"TI": 0.08},
        load_sensors=["tower"],
        load_sensors_used=["tower"],
        components=[{"name": "tower"}],
        _LIFETIME=25,
        wpp_efficiency=1.0,
        farm=farm,
        N_time=n_hours,
        G_MW=2.0,
    )

    wind, reference_wind, yaw, hourly_opex, loads, reference_loads = wpp.compute(
        wst=np.full(n_hours, 10.0),
        wd=np.full(n_hours, 270.0),
    )
    actual = wpp.compute_actual(
        wind_t=wind,
        wind_t_ref=reference_wind,
        yaw_t=yaw,
        lambda_t=np.array([1.0, 0.0, 1.0]),
        DEL=loads,
        DEL_ref=reference_loads,
    )

    np.testing.assert_allclose(wind, 2.2)
    np.testing.assert_allclose(reference_wind, 2.0)
    np.testing.assert_allclose(actual[0], [2.2, 2.0, 2.2])
    np.testing.assert_allclose(actual[1][:, 1], 0.0)
    assert hourly_opex.shape == (n_hours,)
    assert np.isfinite(actual[-1])


def test_pywake_wpp_explains_how_to_install_optional_dependencies(monkeypatch):
    missing_dependency = ModuleNotFoundError("No module named 'design_friendly'")
    monkeypatch.setattr(
        pywake_wpp,
        "_design_friendly_import_error",
        missing_dependency,
    )

    with pytest.raises(ImportError, match=r"hydesign\[gnn-loads\]"):
        pywake_wpp._require_optional_dependencies()
