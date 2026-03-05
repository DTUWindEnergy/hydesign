import importlib.util
import shutil
from pathlib import Path

import numpy as np

from hydesign.costs.costs import battery_cost, pvp_cost, shared_cost, wpp_cost
from hydesign.project_builder import build_baseline_wsb_costs_project


def _load_generated_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("generated_costs_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _project_root(project_name: str) -> Path:
    return (Path.cwd() / "hydesign" / "combined_projects" / project_name).resolve()


def _assert_tuple_allclose(actual, expected) -> None:
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item, expected_item)


def test_build_baseline_wsb_costs_project_creates_expected_file():
    project_name = "pytest_baseline_wsb_costs"
    output_dir = _project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        built_dir = build_baseline_wsb_costs_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        assert built_dir == output_dir
        assert (output_dir / "costs.py").exists()
        assert (output_dir / "__init__.py").exists()
        assert (output_dir / "manifest.yml").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_generated_wsb_costs_matches_original_classes():
    project_name = "pytest_baseline_wsb_costs_parity"
    output_dir = _project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_wsb_costs_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = _load_generated_module(output_dir / "costs.py")

        wind_kwargs = {
            "wind_turbine_cost": 1.2e6,
            "wind_civil_works_cost": 2.1e5,
            "wind_fixed_onm_cost": 3.1e4,
            "wind_variable_onm_cost": 4.0,
            "d_ref": 120.0,
            "hh_ref": 100.0,
            "p_rated_ref": 5.0,
            "N_time": 48,
        }
        wind_inputs = {
            "Nwt": 12,
            "hh": 110.0,
            "d": 130.0,
            "p_rated": 6.0,
            "wind_t": np.linspace(5.0, 50.0, 48),
        }
        orig_wind = wpp_cost(**wind_kwargs).compute(**wind_inputs)
        gen_wind = generated.wpp_cost(**wind_kwargs).compute(**wind_inputs)
        _assert_tuple_allclose(gen_wind, orig_wind)

        solar_kwargs = {
            "solar_PV_cost": 2.5e5,
            "solar_hardware_installation_cost": 1.4e5,
            "solar_inverter_cost": 8.5e4,
            "solar_fixed_onm_cost": 1.1e4,
        }
        solar_inputs = {"solar_MW": 80.0, "DC_AC_ratio": 1.25}
        orig_solar = pvp_cost(**solar_kwargs).compute(**solar_inputs)
        gen_solar = generated.pvp_cost(**solar_kwargs).compute(**solar_inputs)
        _assert_tuple_allclose(gen_solar, orig_solar)

        battery_kwargs = {
            "battery_energy_cost": 1.8e5,
            "battery_power_cost": 9.0e4,
            "battery_BOP_installation_commissioning_cost": 2.0e4,
            "battery_control_system_cost": 1.0e4,
            "battery_energy_onm_cost": 5.0e3,
            "life_y": 2,
        }
        soh = np.ones(2 * 365 * 24)
        soh[365 * 24 :] = 1.0
        battery_inputs = {"b_E": 240.0, "b_P": 60.0, "SoH": soh}
        orig_battery = battery_cost(**battery_kwargs).compute(**battery_inputs)
        gen_battery = generated.battery_cost(**battery_kwargs).compute(**battery_inputs)
        _assert_tuple_allclose(gen_battery, orig_battery)

        shared_kwargs = {
            "hpp_BOS_soft_cost": 7.0e4,
            "hpp_grid_connection_cost": 1.2e5,
            "land_cost": 6.0e4,
        }
        shared_inputs = {"G_MW": 120.0, "Awpp": 14.0, "Apvp": 10.0}
        orig_shared = shared_cost(**shared_kwargs).compute(**shared_inputs)
        gen_shared = generated.shared_cost(**shared_kwargs).compute(**shared_inputs)
        _assert_tuple_allclose(gen_shared, orig_shared)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
