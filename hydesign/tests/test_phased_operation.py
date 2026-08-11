import numpy as np

from hydesign.finance.finance import _year_activity
from hydesign.pv.pv import pvp_with_degradation
from hydesign.utils import operation_shifted, shift_project_profile


def test_shift_project_profile_respects_project_window():
    component_profile = np.arange(2 * 365 * 24)
    shifted = shift_project_profile(
        profile=component_profile,
        operation_start_year=1,
        component_life_y=2,
        project_life_y=4,
    )

    assert shifted.shape == (4 * 365 * 24,)
    assert np.all(shifted[: 365 * 24] == 0)
    assert np.array_equal(
        shifted[365 * 24 : 3 * 365 * 24],
        component_profile,
    )
    assert np.all(shifted[3 * 365 * 24 :] == 0)


def test_operation_shifted_uses_the_operation_year():
    model = operation_shifted(
        operation_start_year=1,
        component_life_y=1,
        project_life_y=2,
    )

    shifted = model.compute(np.ones(365 * 24))

    assert np.all(shifted[: 365 * 24] == 0)
    assert np.all(shifted[365 * 24 :] == 1)


def test_pv_degradation_supports_delayed_operation():
    model = pvp_with_degradation(
        life_y=2,
        project_life_y=4,
        operation_start_year=1,
        pv_deg_yr=[0, 2],
        pv_deg=[0, 0],
    )
    solar_t_ext = np.ones(4 * 365 * 24)
    solar_t_ext_deg = model.compute(solar_t_ext)

    assert solar_t_ext_deg.shape == (4 * 365 * 24,)
    assert np.allclose(solar_t_ext_deg[: 365 * 24], 0)
    assert np.allclose(solar_t_ext_deg[365 * 24 : 3 * 365 * 24], 1)
    assert np.allclose(solar_t_ext_deg[3 * 365 * 24 :], 0)


def test_year_activity_handles_delayed_commissioning():
    activity = _year_activity(start_year=2, component_life_y=3, project_life_y=6)
    assert np.array_equal(activity, np.array([0, 0, 1, 1, 1, 0]))
