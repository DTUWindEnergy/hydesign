import pickle
import shutil

from hydesign.modular import build_baseline_cplex_ems_project
from hydesign.tests.test_files import tfp

from .helpers import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    load_generated_file_module,
    project_root,
)


def test_modular_ems_builder_creates_expected_files():
    project_name = "pytest_modular_ems_files"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        built_dir = build_baseline_cplex_ems_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        assert built_dir == output_dir
        assert (output_dir / "ems.py").exists()
        assert (output_dir / "__init__.py").exists()
        assert (output_dir / "manifest.yml").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_modular_ems_matches_original_reference_fixture():
    project_name = "pytest_modular_ems_parity"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_cplex_ems_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = load_generated_file_module(project_name, "ems")

        with open(tfp + "ems_input_ems.pickle", "rb") as stream:
            inputs = pickle.load(stream)
        with open(tfp + "ems_output_ems.pickle", "rb") as stream:
            expected = pickle.load(stream)

        actual = generated.ems_cplex(**inputs)
        for actual_item, expected_item in zip(actual, expected):
            import numpy as np

            np.testing.assert_allclose(
                actual_item,
                expected_item,
                rtol=DEFAULT_RTOL,
                atol=DEFAULT_ATOL,
            )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
