import pickle
import shutil

import numpy as np

from hydesign.finance.finance import calculate_NPV_IRR as original_calculate_NPV_IRR
from hydesign.finance.finance import calculate_WACC as original_calculate_WACC
from hydesign.finance.finance import finance as original_finance
from hydesign.modular import build_baseline_wsb_finance_project
from hydesign.tests.test_files import tfp

from .helpers import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    assert_sequence_allclose,
    load_generated_file_module,
    project_root,
)


def test_modular_finance_builder_creates_expected_file():
    project_name = "pytest_modular_finance_files"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        built_dir = build_baseline_wsb_finance_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        assert built_dir == output_dir
        assert (output_dir / "finance.py").exists()
        assert (output_dir / "__init__.py").exists()
        assert (output_dir / "manifest.yml").exists()
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def test_modular_finance_helpers_and_compute_match_original():
    project_name = "pytest_modular_finance_parity"
    output_dir = project_root(project_name)
    shutil.rmtree(output_dir, ignore_errors=True)
    try:
        build_baseline_wsb_finance_project(
            project_name=project_name,
            output_root=output_dir.parent,
            regenerate=True,
        )
        generated = load_generated_file_module(project_name, "finance")

        with open(tfp + "finance_input_WACC.pickle", "rb") as stream:
            input_wacc = pickle.load(stream)
        with open(tfp + "finance_input_NPV.pickle", "rb") as stream:
            input_npv = pickle.load(stream)

        np.testing.assert_allclose(
            generated.calculate_WACC(**input_wacc),
            original_calculate_WACC(**input_wacc),
            rtol=DEFAULT_RTOL,
            atol=DEFAULT_ATOL,
        )
        assert_sequence_allclose(
            generated.calculate_NPV_IRR(**input_npv),
            original_calculate_NPV_IRR(**input_npv),
        )

        finance_kwargs = {
            "N_time": 8760,
            "depreciation_yr": np.array([0, 1]),
            "depreciation": np.array([0.0, 1.0]),
            "inflation_yr": np.array([0, 1]),
            "inflation": np.array([0.02, 0.02]),
            "ref_yr_inflation": 0,
            "phasing_yr": np.array([-1, 0]),
            "phasing_CAPEX": np.array([0.3, 0.7]),
            "life_y": 1,
        }
        price = np.full(8760, 55.0)
        hpp = np.full(8760, 80.0)
        penalty = np.zeros(8760)
        finance_inputs = {
            "price_t_ext": price,
            "hpp_t_with_deg": hpp,
            "penalty_t": penalty,
            "CAPEX_w": 8.0e7,
            "CAPEX_s": 3.0e7,
            "CAPEX_b": 1.5e7,
            "CAPEX_el": 1.0e7,
            "OPEX_w": 3.0e6,
            "OPEX_s": 7.0e5,
            "OPEX_b": 3.5e5,
            "OPEX_el": 0.0,
            "wind_WACC": 0.06,
            "solar_WACC": 0.05,
            "battery_WACC": 0.07,
            "tax_rate": 0.22,
        }
        assert_sequence_allclose(
            generated.finance(**finance_kwargs).compute(**finance_inputs),
            original_finance(**finance_kwargs).compute(**finance_inputs),
        )
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
