import os
import shutil

import numpy as np

from hydesign.assembly.hpp_assembly import hpp_model as original_hpp_model
from hydesign.modular import build_baseline_wsb_assembly_project

from .helpers import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    baseline_design_1_inputs,
    drop_generated_modules,
    load_generated_project_module,
    project_root,
)


def test_modular_assembly_builder_creates_expected_files():
    project_name = "pytest_modular_assembly_files"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        built_dir = build_baseline_wsb_assembly_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        assert built_dir == output_dir
        for filename in (
            "assembly.py",
            "weather.py",
            "wind.py",
            "pv.py",
            "ems.py",
            "ems_long_term.py",
            "battery_degradation.py",
            "reliability.py",
            "costs.py",
            "finance.py",
            "__init__.py",
            "manifest.yml",
        ):
            assert (output_dir / filename).exists()
    finally:
        drop_generated_modules(project_name)
        shutil.rmtree(output_dir, ignore_errors=True)


def test_modular_assembly_matches_original_design_1():
    project_name = "pytest_modular_assembly_parity"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_wsb_assembly_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = load_generated_project_module(project_name, "assembly")
        inputs = baseline_design_1_inputs()
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

        np.testing.assert_allclose(
            generated_hpp.evaluate(*inputs["x"]),
            original_hpp.evaluate(*inputs["x"]),
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
        )
    finally:
        drop_generated_modules(project_name)
        shutil.rmtree(output_dir, ignore_errors=True)
