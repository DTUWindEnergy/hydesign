import numpy as np
import openmdao.api as om
import pandas as pd

from hydesign.costs import costs
from hydesign.ems import ems_variable_opex
from hydesign.ems.ems_variable_opex import ems_comp, ems_cplex_parts
from hydesign.weather.weather import interpolate_WD


def test_smart_yaw_dispatch_accounts_for_variable_opex():
    """Use yaw only where its extra power is worth more than its OPEX."""
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    controlled_wind = pd.Series(12.0, index=index)
    reference_wind = pd.Series(10.0, index=index)
    price = pd.Series(5.0, index=index)
    variable_opex = pd.Series(np.r_[np.full(12, 5.0), np.full(12, 15.0)], index=index)
    zeros = pd.Series(0.0, index=index)

    hpp, curtailment, battery, state_of_charge, penalty, smart_yaw = ems_cplex_parts(
        wind_ts=controlled_wind,
        wind_ts_ref=reference_wind,
        M=variable_opex,
        solar_ts=zeros,
        price_ts=price,
        P_batt_MW=0.0,
        E_batt_MWh_t=zeros,
        hpp_grid_connection=20.0,
        battery_depth_of_discharge=0.8,
        charge_efficiency=0.95,
        n_full_power_hours_expected_per_day_at_peak_price=0,
    )

    np.testing.assert_array_equal(smart_yaw, np.r_[np.ones(12), np.zeros(12)])
    np.testing.assert_allclose(hpp, np.r_[np.full(12, 12.0), np.full(12, 10.0)])
    np.testing.assert_allclose(curtailment, 0.0)
    np.testing.assert_allclose(battery, 0.0)
    np.testing.assert_allclose(state_of_charge, 0.0)
    np.testing.assert_allclose(penalty, 0.0)


def test_smart_yaw_dispatch_can_value_emissions_displacement():
    """A high MDF can justify wake steering that revenue alone would reject."""
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    controlled_wind = pd.Series(12.0, index=index)
    reference_wind = pd.Series(10.0, index=index)
    variable_opex = pd.Series(75.0, index=index)
    price = pd.Series(0.0, index=index)
    mdf = pd.Series(np.r_[np.full(12, 500.0), np.zeros(12)], index=index)
    zeros = pd.Series(0.0, index=index)

    *_, smart_yaw = ems_cplex_parts(
        wind_ts=controlled_wind,
        wind_ts_ref=reference_wind,
        M=variable_opex,
        solar_ts=zeros,
        price_ts=price,
        P_batt_MW=0.0,
        E_batt_MWh_t=zeros,
        hpp_grid_connection=20.0,
        battery_depth_of_discharge=0.8,
        charge_efficiency=0.95,
        n_full_power_hours_expected_per_day_at_peak_price=0,
        marginal_displacement_factor_ts=mdf,
        emissions_value=100.0,
    )

    np.testing.assert_array_equal(smart_yaw, np.r_[np.ones(12), np.zeros(12)])


def test_yaw_modes_can_be_fixed_for_policy_comparison():
    index = pd.date_range("2025-01-01", periods=24, freq="h")
    controlled_wind = pd.Series(12.0, index=index)
    reference_wind = pd.Series(10.0, index=index)
    zeros = pd.Series(0.0, index=index)

    def dispatch(yaw_mode):
        return ems_cplex_parts(
            wind_ts=controlled_wind,
            wind_ts_ref=reference_wind,
            M=pd.Series(15.0, index=index),
            solar_ts=zeros,
            price_ts=pd.Series(5.0, index=index),
            P_batt_MW=0.0,
            E_batt_MWh_t=zeros,
            hpp_grid_connection=20.0,
            battery_depth_of_discharge=0.8,
            charge_efficiency=0.95,
            n_full_power_hours_expected_per_day_at_peak_price=0,
            yaw_mode=yaw_mode,
        )

    reference = dispatch("reference")
    controlled = dispatch("controlled")
    smart = dispatch("smart")

    np.testing.assert_array_equal(reference[-1], np.zeros(24))
    np.testing.assert_array_equal(controlled[-1], np.ones(24))
    np.testing.assert_array_equal(smart[-1], np.zeros(24))
    np.testing.assert_allclose(reference[0], 10.0)
    np.testing.assert_allclose(controlled[0], 12.0)


def test_emissions_inputs_are_available_on_the_hpp_ems_component():
    """The OpenMDAO component exposes MDF and its objective weight."""
    problem = om.Problem(reports=False)
    problem.model.add_subsystem("ems", ems_comp(N_time=24, life_y=1), promotes=["*"])
    problem.setup()

    np.testing.assert_array_equal(
        problem.get_val("marginal_displacement_factor_t"), np.zeros(24)
    )
    assert problem.get_val("emissions_value").item() == 0.0


def test_hpp_ems_component_forwards_yaw_mode_and_batch_size(monkeypatch):
    """The full HPP assembly configures its yaw policy and solver batches."""
    received = {}

    def fake_ems_cplex(**kwargs):
        received["yaw_mode"] = kwargs["yaw_mode"]
        received["batch_size"] = kwargs["batch_size"]
        n_time = len(kwargs["wind_ts"])
        zeros = np.zeros(n_time)
        return zeros, zeros, zeros, np.zeros(n_time + 1), zeros, zeros

    monkeypatch.setattr("hydesign.ems.ems_variable_opex.ems_cplex", fake_ems_cplex)
    component = ems_comp(N_time=24, life_y=1, yaw_mode="reference", batch_size=72)
    component.function(
        wind_t=np.ones(24),
        wind_t_ref=np.ones(24),
        M=np.zeros(24),
        solar_t=np.zeros(24),
        price_t=np.ones(24),
        b_P=0.0,
        b_E=0.0,
        G_MW=1.0,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
        peak_hr_quantile=0.9,
        cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
        n_full_power_hours_expected_per_day_at_peak_price=0.0,
    )

    assert received["yaw_mode"] == "reference"
    assert received["batch_size"] == 72


def test_wind_opex_includes_actual_yaw_control_cost(monkeypatch):
    monkeypatch.setattr(costs, "wt_cost", lambda **kwargs: 1_000_000.0)
    model = costs.wpp_cost(
        wind_turbine_cost=1_000_000.0,
        wind_civil_works_cost=0.0,
        wind_fixed_onm_cost=10_000.0,
        wind_variable_onm_cost=2.0,
        d_ref=100.0,
        hh_ref=100.0,
        p_rated_ref=10.0,
        N_time=24,
    )
    inputs = dict(
        Nwt=2,
        hh=100.0,
        d=100.0,
        p_rated=10.0,
        wind_t=np.full(24, 5.0),
    )

    _, baseline_opex = model.compute(**inputs)
    _, controlled_opex = model.compute(**inputs, M_actual=-25_000.0)

    assert controlled_opex - baseline_opex == -25_000.0


def test_wind_direction_interpolation_is_repeatable():
    weather = pd.DataFrame(
        {"WD_10": [350.0, 10.0], "WD_100": [10.0, 30.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="h"),
    )
    original = weather.copy()

    first = interpolate_WD(weather, hh=50.0)
    second = interpolate_WD(weather, hh=50.0)

    np.testing.assert_allclose(first, second)
    pd.testing.assert_frame_equal(weather, original)


def test_batched_dispatch_combines_each_solver_batch(monkeypatch):
    """The public EMS helper joins independently solved time batches."""
    calls = []

    def fake_dispatch(**kwargs):
        wind = kwargs["wind_ts"].to_numpy()
        calls.append(wind)
        zeros = np.zeros(wind.size)
        return wind, zeros, zeros, np.zeros(wind.size + 1), zeros, np.ones(wind.size)

    monkeypatch.setattr(ems_variable_opex, "ems_cplex_parts", fake_dispatch)
    index = pd.date_range("2025-01-01", periods=5, freq="h")
    wind = pd.Series(np.arange(5.0), index=index)
    zeros = pd.Series(0.0, index=index)

    hpp, *_, yaw = ems_variable_opex.ems_cplex(
        wind_ts=wind,
        wind_ts_ref=wind,
        M=zeros,
        solar_ts=zeros,
        price_ts=zeros,
        P_batt_MW=0.0,
        E_batt_MWh_t=zeros,
        hpp_grid_connection=10.0,
        battery_depth_of_discharge=0.8,
        charge_efficiency=0.95,
        batch_size=2,
    )

    assert [len(batch) for batch in calls] == [2, 2, 1]
    np.testing.assert_array_equal(hpp, wind)
    np.testing.assert_array_equal(yaw, np.ones(5))


def test_degraded_operation_follows_the_planned_battery_dispatch():
    """Exercise the normal lifetime-operation path with one day of data."""
    n_hours = 24
    wind = np.linspace(4.0, 6.0, n_hours)
    solar = np.clip(np.sin(np.linspace(-np.pi / 2, 3 * np.pi / 2, n_hours)), 0, None)
    battery = np.tile([-1.0, 1.0], n_hours // 2)
    state_of_charge = np.full(n_hours + 1, 5.0)
    health = np.full(n_hours, 0.9)

    result = ems_variable_opex.operation_solar_batt_deg(
        wind_t_deg=wind * 0.95,
        solar_t_deg=solar * 0.9,
        batt_degradation=health,
        wind_t=wind,
        solar_t=solar,
        hpp_curt_t=np.zeros(n_hours),
        b_t=battery,
        b_E_SOC_t=state_of_charge,
        G_MW=10.0,
        b_E=10.0,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
        price_ts=np.linspace(20.0, 80.0, n_hours),
    )

    hpp, curtailment, actual_battery, actual_soc, penalty = result
    assert all(
        len(values) == n_hours for values in (hpp, curtailment, actual_battery, penalty)
    )
    assert len(actual_soc) == n_hours + 1
    assert np.all((actual_soc >= 1.8) & (actual_soc <= 9.0))


def test_long_term_operation_reports_total_curtailment():
    """The long-term model exposes both degraded and planned totals."""
    n_hours = 24
    model = ems_variable_opex.ems_long_term_operation(N_time=n_hours, life_y=1)
    result = model.compute(
        SoH=np.ones(n_hours),
        wind_t_ext_deg=np.full(n_hours, 5.0),
        solar_t_ext_deg=np.zeros(n_hours),
        wind_t_ext=np.full(n_hours, 5.0),
        solar_t_ext=np.zeros(n_hours),
        price_t_ext=np.linspace(20.0, 80.0, n_hours),
        b_P=np.array([1.0]),
        b_E=np.array([10.0]),
        G_MW=np.array([4.0]),
        battery_depth_of_discharge=np.array([0.8]),
        battery_charge_efficiency=np.array([0.95]),
        hpp_curt_t=np.ones(n_hours),
        b_t=np.zeros(n_hours),
        b_E_SOC_t=np.full(n_hours + 1, 5.0),
        peak_hr_quantile=0.9,
        n_full_power_hours_expected_per_day_at_peak_price=3,
    )

    assert result[-1] == n_hours
    assert result[-2] >= 0


def test_rule_based_constant_output_is_daily_and_finite():
    """The simple constant-output policy produces one stable daily schedule."""
    n_hours = 24
    wind = np.linspace(4.0, 6.0, n_hours)
    result = ems_variable_opex.operation_constant_output(
        wind_t_deg=wind * 0.95,
        solar_t_deg=np.zeros(n_hours),
        batt_degradation=np.ones(n_hours),
        wind_t=wind,
        solar_t=np.zeros(n_hours),
        hpp_curt_t=np.zeros(n_hours),
        b_t=np.tile([-0.5, 0.5], n_hours // 2),
        b_E_SOC_t=np.full(n_hours + 1, 5.0),
        G_MW=10.0,
        b_E=10.0,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
        load_min=3.0,
    )

    hpp, curtailment, battery, state_of_charge, penalty = result
    assert np.ptp(hpp) == 0.0
    assert all(np.all(np.isfinite(values)) for values in result)
    assert len(curtailment) == len(battery) == len(penalty) == n_hours
    assert len(state_of_charge) == n_hours + 1


def test_constant_output_dispatch_keeps_a_steady_wind_profile():
    """The constant-output CPLEX workflow preserves already steady generation."""
    n_hours = 48
    index = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    wind = pd.Series(5.0, index=index)
    zeros = pd.Series(0.0, index=index)

    hpp, curtailment, battery, state_of_charge, penalty = (
        ems_variable_opex.ems_cplex_constantoutput(
            wind_ts=wind,
            solar_ts=zeros,
            price_ts=pd.Series(50.0, index=index),
            P_batt_MW=0.0,
            E_batt_MWh_t=zeros,
            hpp_grid_connection=10.0,
            battery_depth_of_discharge=0.8,
            charge_efficiency=0.95,
            batch_size=24,
            load_min=3.0,
        )
    )

    np.testing.assert_allclose(hpp, 5.0)
    np.testing.assert_allclose(curtailment, 0.0, atol=1e-7)
    np.testing.assert_allclose(battery, 0.0, atol=1e-7)
    assert len(state_of_charge) == n_hours + 1
    assert len(penalty) == n_hours


def test_pyomo_dispatch_combines_solver_batches(monkeypatch):
    """The legacy Pyomo wrapper combines its normal multi-day batches."""

    def fake_dispatch(**kwargs):
        wind = kwargs["wind_ts"].to_numpy()
        zeros = np.zeros(wind.size)
        return wind, zeros, zeros, zeros, zeros

    monkeypatch.setattr(
        ems_variable_opex,
        "ems_Wind_Solar_Battery_Pyomo_parts",
        fake_dispatch,
    )
    n_hours = 48
    index = pd.date_range("2025-01-01", periods=n_hours, freq="h")
    wind = pd.Series(5.0, index=index)
    zeros = pd.Series(0.0, index=index)

    hpp, curtailment, battery, state_of_charge, penalty = (
        ems_variable_opex.ems_Wind_Solar_Battery_Pyomo(
            wind_ts=wind,
            solar_ts=zeros,
            price_ts=zeros,
            P_batt_MW=0.0,
            E_batt_MWh_t=zeros,
            hpp_grid_connection=10.0,
            battery_depth_of_discharge=0.8,
            charge_efficiency=0.95,
            batch_size=24,
        )
    )

    np.testing.assert_allclose(hpp, 5.0)
    assert len(curtailment) == len(battery) == len(penalty) == n_hours
    assert len(state_of_charge) == n_hours + 1
