import importlib.util
import pickle
import shutil
from pathlib import Path

import numpy as np

from hydesign.project_builder import build_baseline_cplex_ems_project
from hydesign.tests.test_files import tfp


def _load_generated_module(module_path: Path):
    spec = importlib.util.spec_from_file_location("generated_ems_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _project_root(project_name: str) -> Path:
    return (Path.cwd() / "hydesign" / "combined_projects" / project_name).resolve()


def test_build_baseline_cplex_ems_project_creates_expected_files():
    project_name = "pytest_baseline_wsb_files"
    output_dir = _project_root(project_name)
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


def test_generated_baseline_ems_matches_reference_fixture():
    project_name = "pytest_baseline_wsb_parity"
    output_dir = _project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_cplex_ems_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = _load_generated_module(output_dir / "ems.py")

        with open(tfp + "ems_input_ems.pickle", "rb") as stream:
            inputs = pickle.load(stream)
        with open(tfp + "ems_output_ems.pickle", "rb") as stream:
            expected = pickle.load(stream)

        actual = generated.ems_cplex(**inputs)
        for actual_item, expected_item in zip(actual, expected):
            np.testing.assert_allclose(actual_item, expected_item)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
