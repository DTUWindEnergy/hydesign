import importlib
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hydesign.examples import examples_filepath
from hydesign.tests.test_files import tfp

DEFAULT_RTOL = 1e-7
DEFAULT_ATOL = 1e-7


def project_root(project_name: str) -> Path:
    return (Path.cwd() / "hydesign" / "modular" / "projects" / project_name).resolve()


def drop_generated_modules(project_name: str) -> None:
    prefix = f"hydesign.modular.projects.{project_name}"
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(prefix + "."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def load_generated_file_module(project_name: str, module_basename: str):
    module_path = project_root(project_name) / f"{module_basename}.py"
    module_name = f"generated_{project_name}_{module_basename}".replace("-", "_")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_generated_project_module(project_name: str, module_basename: str):
    drop_generated_modules(project_name)
    return importlib.import_module(
        f"hydesign.modular.projects.{project_name}.{module_basename}"
    )


def assert_sequence_allclose(
    actual,
    expected,
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> None:
    assert len(actual) == len(expected)
    for actual_item, expected_item in zip(actual, expected):
        np.testing.assert_allclose(actual_item, expected_item, rtol=rtol, atol=atol)


def baseline_design_1_inputs():
    examples_sites = pd.read_csv(
        f"{examples_filepath}examples_sites.csv", index_col=0, sep=";"
    )
    ex_site = examples_sites.loc[examples_sites.name == "France_good_wind"]
    output_df = pd.read_csv(tfp + "France_good_wind_design.csv", index_col=0, sep=";")
    design = output_df["Design 1"]
    x = [
        design["clearance [m]"],
        design["sp [W/m2]"],
        design["p_rated [MW]"],
        design["Nwt"],
        design["wind_MW_per_km2 [MW/km2]"],
        design["solar_MW [MW]"],
        design["surface_tilt [deg]"],
        design["surface_azimuth [deg]"],
        design["DC_AC_ratio"],
        design["b_P [MW]"],
        design["b_E_h [h]"],
        design["cost_of_battery_P_fluct_in_peak_price_ratio"],
    ]
    return {
        "latitude": ex_site["latitude"].values[0],
        "longitude": ex_site["longitude"].values[0],
        "altitude": ex_site["altitude"].values[0],
        "sim_pars_fn": examples_filepath + ex_site["sim_pars_fn"].values[0],
        "input_ts_fn": examples_filepath + ex_site["input_ts_fn"].values[0],
        "x": x,
    }
