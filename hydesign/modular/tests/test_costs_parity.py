import shutil

import numpy as np

from hydesign.costs.costs import battery_cost, pvp_cost, shared_cost, wpp_cost
from hydesign.modular import build_baseline_wsb_costs_project

from .helpers import (
    assert_sequence_allclose,
    load_generated_file_module,
    project_root,
)


def test_modular_costs_builder_creates_expected_file():
    project_name = "pytest_modular_costs_files"
    output_dir = project_root(project_name)
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


def test_modular_costs_matches_original_classes():
    project_name = "pytest_modular_costs_parity"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_wsb_costs_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = load_generated_file_module(project_name, "costs")

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
        assert_sequence_allclose(
            generated.wpp_cost(**wind_kwargs).compute(**wind_inputs),
            wpp_cost(**wind_kwargs).compute(**wind_inputs),
        )

        solar_kwargs = {
            "solar_PV_cost": 2.5e5,
            "solar_hardware_installation_cost": 1.4e5,
            "solar_inverter_cost": 8.5e4,
            "solar_fixed_onm_cost": 1.1e4,
        }
        solar_inputs = {"solar_MW": 80.0, "DC_AC_ratio": 1.25}
        assert_sequence_allclose(
            generated.pvp_cost(**solar_kwargs).compute(**solar_inputs),
            pvp_cost(**solar_kwargs).compute(**solar_inputs),
        )

        battery_kwargs = {
            "battery_energy_cost": 1.8e5,
            "battery_power_cost": 9.0e4,
            "battery_BOP_installation_commissioning_cost": 2.0e4,
            "battery_control_system_cost": 1.0e4,
            "battery_energy_onm_cost": 5.0e3,
            "life_y": 2,
        }
        soh = np.ones(2 * 365 * 24)
        battery_inputs = {"b_E": 240.0, "b_P": 60.0, "SoH": soh}
        assert_sequence_allclose(
            generated.battery_cost(**battery_kwargs).compute(**battery_inputs),
            battery_cost(**battery_kwargs).compute(**battery_inputs),
        )

        shared_kwargs = {
            "hpp_BOS_soft_cost": 7.0e4,
            "hpp_grid_connection_cost": 1.2e5,
            "land_cost": 6.0e4,
        }
        shared_inputs = {"G_MW": 120.0, "Awpp": 14.0, "Apvp": 10.0}
        assert_sequence_allclose(
            generated.shared_cost(**shared_kwargs).compute(**shared_inputs),
            shared_cost(**shared_kwargs).compute(**shared_inputs),
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
