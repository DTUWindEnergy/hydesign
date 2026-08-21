import numpy as np
import pandas as pd

from hydesign.ems import ems as standard_ems
from hydesign.ems import ems_BM as balancing_ems
from hydesign.ems import ems_P2MeOH_bidirectional as methanol_ems
from hydesign.finance import finance_P2MeOH_bidirectional as methanol_finance


def hourly_series(value, hours):
    index = pd.date_range("2025-01-01", periods=hours, freq="h")
    return pd.Series(value, index=index, dtype=float)


def test_standard_ems_keeps_an_already_constant_wind_profile():
    """Cover the complete constant-output dispatch with a simple feasible plant."""
    hours = 48
    wind = hourly_series(5.0, hours)
    zeros = hourly_series(0.0, hours)

    hpp, curtailment, battery, state_of_charge, penalty = (
        standard_ems.ems_cplex_constantoutput(
            wind_ts=wind,
            solar_ts=zeros,
            price_ts=hourly_series(50.0, hours),
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
    assert len(state_of_charge) == hours + 1
    assert len(penalty) == hours


def test_standard_rule_based_operation_makes_daily_constant_output():
    """Exercise the normal degraded-operation policy without a solver."""
    hours = 24
    wind = np.linspace(4.0, 6.0, hours)
    result = standard_ems.operation_constant_output(
        wind_t_deg=wind * 0.95,
        solar_t_deg=np.zeros(hours),
        batt_degradation=np.ones(hours),
        wind_t=wind,
        solar_t=np.zeros(hours),
        hpp_curt_t=np.zeros(hours),
        b_t=np.tile([-0.5, 0.5], hours // 2),
        b_E_SOC_t=np.full(hours + 1, 5.0),
        G_MW=10.0,
        b_E=10.0,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
        load_min=3.0,
    )

    assert np.ptp(result[0]) == 0.0
    assert all(np.all(np.isfinite(values)) for values in result)


def test_balancing_market_dispatch_returns_a_feasible_schedule():
    """Exercise the main OR-Tools balancing-market optimization workflow."""
    hours = 4
    wind = hourly_series(5.0, hours)
    imbalance = pd.Series([-1.0, 1.0, -1.0, 1.0], index=wind.index)

    result = balancing_ems.ems_ORtools_parts(
        wind_ts=wind,
        solar_ts=hourly_series(0.0, hours),
        price_ts=hourly_series(50.0, hours),
        wind_BM_ts=hourly_series(4.5, hours),
        price_up_reg_ts=hourly_series(60.0, hours),
        price_dwn_reg_ts=hourly_series(40.0, hours),
        SO_imbalance_ts=imbalance,
        penalty_BM=70.0,
        bi_directional_status=0,
        P_batt_MW=1.0,
        E_batt_MWh_t=hourly_series(4.0, hours),
        hpp_grid_connection=10.0,
        battery_depth_of_discharge=0.8,
        charge_efficiency=0.95,
    )

    assert len(result) == 13
    assert all(len(values) == hours for values in result[:-2])
    assert len(result[-2]) == hours + 1
    assert len(result[-1]) == hours
    assert all(np.all(np.isfinite(values)) for values in result)


def test_balancing_market_batch_wrappers_join_solver_results(monkeypatch):
    """Cover the normal batching paths independently of each optimizer backend."""
    hours = 48
    wind = hourly_series(5.0, hours)
    zeros = hourly_series(0.0, hours)

    def fake_balancing_dispatch(**kwargs):
        size = len(kwargs["wind_ts"])
        outputs = [np.zeros(size) for _ in range(13)]
        outputs[0] = kwargs["wind_ts"].to_numpy()
        outputs[11] = np.zeros(size + 1)
        return tuple(outputs)

    monkeypatch.setattr(balancing_ems, "ems_ORtools_parts", fake_balancing_dispatch)
    balancing = balancing_ems.ems_ORtools(
        wind_ts=wind,
        solar_ts=zeros,
        price_ts=hourly_series(50.0, hours),
        wind_BM_ts=wind,
        price_up_reg_ts=hourly_series(60.0, hours),
        price_dwn_reg_ts=hourly_series(40.0, hours),
        SO_imbalance_ts=zeros,
        penalty_BM=70.0,
        bi_directional_status=0,
        P_batt_MW=1.0,
        E_batt_MWh_t=hourly_series(4.0, hours),
        hpp_grid_connection=10.0,
        battery_depth_of_discharge=0.8,
        charge_efficiency=0.95,
        batch_size=24,
    )
    np.testing.assert_array_equal(balancing[0], wind)
    assert len(balancing[11]) == hours + 1

    def fake_pyomo_dispatch(**kwargs):
        size = len(kwargs["wind_ts"])
        zeros = np.zeros(size)
        return kwargs["wind_ts"].to_numpy(), zeros, zeros, zeros, zeros

    monkeypatch.setattr(
        balancing_ems,
        "ems_Wind_Solar_Battery_Pyomo_parts",
        fake_pyomo_dispatch,
    )
    pyomo = balancing_ems.ems_Wind_Solar_Battery_Pyomo(
        wind_ts=wind,
        solar_ts=zeros,
        price_ts=zeros,
        P_batt_MW=1.0,
        E_batt_MWh_t=hourly_series(4.0, hours),
        hpp_grid_connection=10.0,
        battery_depth_of_discharge=0.8,
        charge_efficiency=0.95,
        batch_size=24,
    )
    np.testing.assert_array_equal(pyomo[0], wind)
    assert len(pyomo[3]) == hours + 1

    def fake_degraded_operation(**kwargs):
        size = len(kwargs["wind_t_deg"])
        outputs = [np.zeros(size) for _ in range(7)]
        outputs[0] = kwargs["wind_t_deg"].to_numpy()
        outputs[3] = np.zeros(size + 1)
        return tuple(outputs)

    monkeypatch.setattr(
        balancing_ems,
        "operation_wind_batt_deg_parts",
        fake_degraded_operation,
    )
    degraded = balancing_ems.ems_operation_wind_batt_deg(
        wind_t_deg=wind,
        batt_degradation=hourly_series(0.9, hours),
        hpp_curt_t=zeros,
        b_t=zeros,
        b_E_SOC_t=hourly_series(2.0, hours),
        b_P=1.0,
        b_E=hourly_series(4.0, hours),
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
        P_up_reg_t=zeros,
        P_dwn_reg_t=zeros,
        P_up_max_t=zeros,
        P_dwn_max_t=zeros,
        price_up_t=hourly_series(60.0, hours),
        price_dwn_t=hourly_series(40.0, hours),
        hpp_t=wind,
        penalty_BM=70.0,
        batch_size=24,
    )
    np.testing.assert_array_equal(degraded[0], wind)
    assert len(degraded[3]) == hours + 1


def test_balancing_rule_based_operation_tracks_battery_state():
    """Exercise the balancing module's simple degraded-battery policy."""
    hours = 4
    wind = np.linspace(4.0, 5.0, hours)
    result = balancing_ems.operation_rule_base_no_penalty(
        wind_t_deg=wind * 0.95,
        solar_t_deg=np.zeros(hours),
        batt_degradation=np.ones(hours),
        wind_t=wind,
        solar_t=np.zeros(hours),
        hpp_curt_t=np.zeros(hours),
        b_t=np.array([-0.5, 0.5, -0.5, 0.5]),
        b_E_SOC_t=np.full(hours + 1, 2.0),
        G_MW=10.0,
        b_E=4.0,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
    )

    assert all(np.all(np.isfinite(values)) for values in result)
    assert len(result[-1]) == hours + 1


def test_methanol_dispatch_combines_normal_time_batches(monkeypatch):
    """The public methanol EMS joins every output from each solver batch."""

    def fake_dispatch(**kwargs):
        hours = len(kwargs["wind_ts"])
        outputs = [np.zeros(hours) for _ in range(26)]
        outputs[0] = np.asarray(kwargs["wind_ts"], dtype=float)
        outputs[3] = np.zeros(hours + 1)  # battery state of charge
        outputs[21] = np.zeros(hours + 1)  # methanol tank level
        return tuple(outputs)

    monkeypatch.setattr(
        methanol_ems,
        "ems_cplex_parts_P2MeOH_bidirectional",
        fake_dispatch,
    )
    hours = 5
    wind = np.arange(1.0, hours + 1)
    zeros = np.zeros(hours)

    result = methanol_ems.ems_cplex_P2MeOH_bidirectional(
        wind_ts=wind,
        solar_ts=zeros,
        elec_spot_price_ts=np.full(hours, 50.0),
        P_batt_MW=1.0,
        E_batt_MWh_t=4.0,
        hpp_grid_connection=10.0,
        battery_depth_of_discharge=0.8,
        battery_charging_efficiency=0.95,
        P_SOEC_MW=2.0,
        lhv=33.3,
        eff_curve=np.array([0.7]),
        eff_curve_prod_no_eff=np.array([1.0]),
        price_green_MeOH=1.0,
        price_grid_MeOH=1.0,
        elec_grid_price_ts=np.full(hours, 60.0),
        m_MeOH_demand_ts=zeros,
        P_heater_MW=1.0,
        P_DAC_MW=1.0,
        Q_DAC_MW=1.0,
        M_CO2=44.0,
        M_H2=2.0,
        M_H2O=18.0,
        M_CH3OH=32.0,
        w=1.0,
        psi_DAC_MWhkg=1.0,
        phi_DAC_MWhkg=1.0,
        P_reactor_MW=1.0,
        w_comp_reactor_MWhkg=1.0,
        H2O_yield=1.0,
        MeOH_yield=1.0,
        m_MeOH_tank_flow_max_kg=10.0,
        m_MeOH_tank_max_kg=100.0,
        charging_efficiency_MeOH_tank=0.95,
        penalty_factor_MeOH=1.0,
        MeOH_demand_scheme="fixed",
        batch_size=2,
    )

    assert len(result) == 26
    np.testing.assert_array_equal(result[0], wind)
    assert len(result[3]) == len(result[21]) == hours + 1
    assert all(
        len(values) == hours for i, values in enumerate(result) if i not in (3, 21)
    )


def test_methanol_component_expands_dispatch_to_the_project_lifetime(monkeypatch):
    """Exercise the component's main input, dispatch, and lifetime-output flow."""

    def fake_dispatch(**kwargs):
        hours = len(kwargs["wind_ts"])
        outputs = [np.zeros(hours) for _ in range(26)]
        outputs[0] = np.asarray(kwargs["wind_ts"], dtype=float)
        outputs[3] = np.zeros(hours + 1)
        outputs[21] = np.zeros(hours + 1)
        return tuple(outputs)

    monkeypatch.setattr(
        methanol_ems,
        "ems_cplex_P2MeOH_bidirectional",
        fake_dispatch,
    )
    hours = 3
    model = methanol_ems.ems_P2MeOH_bidirectional(
        N_time=hours,
        life_h=hours * 2,
        eff_curve=np.array([[0.0, 0.6], [1.0, 0.7]]),
        eff_curve_prod_no_eff=np.array([[0.0, 0.0], [1.0, 0.02]]),
        MeOH_demand_scheme="fixed",
    )
    inputs = {name: np.array([1.0]) for name, _metadata in model.inputs}
    inputs.update(
        wind_t=np.arange(1.0, hours + 1),
        solar_t=np.zeros(hours),
        elec_spot_price_t=np.full(hours, 50.0),
        elec_grid_price_t=np.full(hours, 60.0),
        m_MeOH_demand_t=np.zeros(hours),
    )

    result = model.compute(**inputs)

    assert len(result) == 32
    np.testing.assert_array_equal(result[0], [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    assert len(result[9]) == len(result[27]) == hours * 2 + 1
    assert result[7] == 0.0


def test_methanol_solver_exports_wind_when_demand_is_zero():
    """Run the complete methanol optimizer for a tiny, directly feasible plant."""
    hours = 3
    wind = np.full(hours, 5.0)
    zeros = np.zeros(hours)

    result = methanol_ems.ems_cplex_parts_P2MeOH_bidirectional(
        wind_ts=wind,
        solar_ts=zeros,
        elec_spot_price_ts=np.full(hours, 50.0),
        P_batt_MW=0.0,
        E_batt_MWh_t=0.0,
        hpp_grid_connection=10.0,
        battery_depth_of_discharge=0.8,
        battery_charging_efficiency=0.95,
        P_SOEC_MW=1.0,
        lhv=33.3,
        eff_curve=np.array([[0.0, 0.6], [1.0, 0.7]]),
        eff_curve_prod_no_eff=np.array([[0.0, 0.0], [1.0, 0.02]]),
        price_green_MeOH=0.0,
        price_grid_MeOH=0.0,
        elec_grid_price_ts=np.full(hours, 60.0),
        m_MeOH_demand_ts=zeros,
        P_heater_MW=0.0,
        Q_DAC_MW=0.0,
        M_CO2=44.0,
        M_H2=2.0,
        M_H2O=18.0,
        M_CH3OH=32.0,
        w=1.0,
        psi_DAC_MWhkg=0.001,
        w_comp_reactor_MWhkg=0.001,
        H2O_yield=1.0,
        MeOH_yield=1.0,
        m_MeOH_tank_flow_max_kg=1.0,
        m_MeOH_tank_max_kg=10.0,
        charging_efficiency_MeOH_tank=0.95,
        penalty_factor_MeOH=1.0,
        MeOH_demand_scheme="fixed",
    )

    np.testing.assert_allclose(result[0], wind)
    np.testing.assert_allclose(result[1], 0.0, atol=1e-7)
    np.testing.assert_allclose(result[24], 0.0, atol=1e-7)


def test_methanol_finance_calculates_the_main_project_metrics(monkeypatch):
    """Run the complete finance calculation for one representative year."""
    monkeypatch.setattr(
        methanol_finance,
        "calculate_break_even_green_MeOH_price_bidirectional",
        lambda **_kwargs: np.array([1.5]),
    )
    monkeypatch.setattr(
        methanol_finance,
        "calculate_break_even_PPA_price_P2MeOH_bidirectional",
        lambda **_kwargs: np.array([45.0]),
    )
    hours = 24
    model = methanol_finance.finance_P2MeOH_bidirectional(
        N_time=hours,
        life_h=hours,
        depreciation_yr=[0, 25],
        depreciation=[0, 1],
        inflation_yr=[-3, 0, 1, 25],
        inflation=[0.1, 0.1, 0.06, 0.06],
        ref_yr_inflation=0,
        phasing_yr=[-1, 0],
        phasing_CAPEX=[1, 1],
    )
    inputs = {
        "P_HPP_t": np.full(hours, 5.0),
        "P_green_reactor_t": np.full(hours, 0.1),
        "P_green_DAC_t": np.full(hours, 0.1),
        "P_green_heater_t": np.zeros(hours),
        "P_SOEC_green_t": np.full(hours, 1.0),
        "P_grid_reactor_t": np.full(hours, 0.1),
        "P_grid_DAC_t": np.full(hours, 0.1),
        "P_grid_heater_t": np.zeros(hours),
        "P_SOEC_grid_t": np.full(hours, 0.5),
        "penalty_t": np.zeros(hours),
        "P_curtailment_t": np.zeros(hours),
        "elec_spot_price_t_ext": np.full(hours, 50.0),
        "elec_grid_price_t_ext": np.full(hours, 60.0),
        "price_green_MeOH": 1.0,
        "price_grid_MeOH": 0.8,
        "P_purch_grid_t": np.full(hours, 0.7),
        "m_grid_MeOH_reactor_t": np.full(hours, 0.5),
        "m_green_MeOH_dist_t": np.full(hours, 1.0),
        "m_MeOH_demand_t_ext": np.full(hours, 1.5),
        "CAPEX_w": 1_000_000.0,
        "OPEX_w": 10_000.0,
        "CAPEX_s": 500_000.0,
        "OPEX_s": 5_000.0,
        "CAPEX_b": 250_000.0,
        "OPEX_b": 2_500.0,
        "CAPEX_sh": 100_000.0,
        "OPEX_sh": 1_000.0,
        "CAPEX_P2MeOH": 750_000.0,
        "OPEX_SOEC": 5_000.0,
        "water_consumption_cost": 1_000.0,
        "OPEX_DAC": 2_000.0,
        "OPEX_reactor": 2_000.0,
        "OPEX_MeOH_tank": 1_000.0,
        "wind_WACC": 0.06,
        "solar_WACC": 0.06,
        "battery_WACC": 0.06,
        "P2MeOH_WACC": 0.08,
        "tax_rate": 0.22,
    }
    outputs = {}

    model.compute(inputs, outputs)

    assert outputs["CAPEX"] == 2_600_000.0
    assert outputs["OPEX"] == 29_500.0
    assert outputs["mean_AEP"] > outputs["mean_Power2Grid"] > 0
    assert outputs["annual_green_MeOH"] > outputs["annual_grid_MeOH"] > 0
    np.testing.assert_array_equal(outputs["break_even_PPA_price"], [45.0])
