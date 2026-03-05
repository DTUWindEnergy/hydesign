import importlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from hydesign.assembly.hpp_assembly import hpp_model as original_hpp_model
from hydesign.examples import examples_filepath
from hydesign.project_builder import build_baseline_wsb_assembly_project
from hydesign.tests.test_files import tfp


def _project_root(project_name: str) -> Path:
    return (Path.cwd() / "hydesign" / "combined_projects" / project_name).resolve()


def _drop_generated_modules(project_name: str) -> None:
    prefix = f"hydesign.combined_projects.{project_name}"
    for module_name in list(sys.modules):
        if module_name == prefix or module_name.startswith(prefix + "."):
            sys.modules.pop(module_name, None)
    importlib.invalidate_caches()


def _load_generated_assembly_module(project_name: str):
    _drop_generated_modules(project_name)
    return importlib.import_module(
        f"hydesign.combined_projects.{project_name}.assembly"
    )


def _baseline_design_1_inputs():
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


def test_build_baseline_wsb_assembly_project_creates_expected_files():
    project_name = "pytest_baseline_wsb_assembly"
    output_dir = _project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        built_dir = build_baseline_wsb_assembly_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        assert built_dir == output_dir
        assert (output_dir / "assembly.py").exists()
        assert (output_dir / "weather.py").exists()
        assert (output_dir / "wind.py").exists()
        assert (output_dir / "pv.py").exists()
        assert (output_dir / "ems.py").exists()
        assert (output_dir / "ems_long_term.py").exists()
        assert (output_dir / "battery_degradation.py").exists()
        assert (output_dir / "reliability.py").exists()
        assert (output_dir / "costs.py").exists()
        assert (output_dir / "finance.py").exists()
        assert (output_dir / "__init__.py").exists()
        manifest = json.loads((output_dir / "manifest.yml").read_text(encoding="utf-8"))
        assert set(manifest["artifacts"]) == {
            "weather",
            "wind",
            "pv",
            "ems",
            "ems_long_term",
            "battery_degradation",
            "reliability",
            "costs",
            "finance",
            "assembly",
        }
        assert "hydesign/assembly/hpp_assembly.py" in manifest["source_hashes"]
    finally:
        _drop_generated_modules(project_name)
        shutil.rmtree(output_dir, ignore_errors=True)


def test_generated_wsb_assembly_matches_original_design_1():
    project_name = "pytest_baseline_wsb_assembly_parity"
    output_dir = _project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_wsb_assembly_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = _load_generated_assembly_module(project_name)
        inputs = _baseline_design_1_inputs()
        original_work_dir = str(output_dir / "orig") + os.sep
        generated_work_dir = str(output_dir / "generated") + os.sep

        original_hpp = original_hpp_model(
            latitude=inputs["latitude"],
            longitude=inputs["longitude"],
            altitude=inputs["altitude"],
            max_num_batteries_allowed=10,
            work_dir=original_work_dir,
            sim_pars_fn=inputs["sim_pars_fn"],
            input_ts_fn=inputs["input_ts_fn"],
        )
        generated_hpp = generated.hpp_model(
            latitude=inputs["latitude"],
            longitude=inputs["longitude"],
            altitude=inputs["altitude"],
            max_num_batteries_allowed=10,
            work_dir=generated_work_dir,
            sim_pars_fn=inputs["sim_pars_fn"],
            input_ts_fn=inputs["input_ts_fn"],
        )

        original_outputs = original_hpp.evaluate(*inputs["x"])
        generated_outputs = generated_hpp.evaluate(*inputs["x"])
        np.testing.assert_allclose(
            generated_outputs, original_outputs, rtol=1e-8, atol=1e-8
        )
    finally:
        _drop_generated_modules(project_name)
        shutil.rmtree(output_dir, ignore_errors=True)
