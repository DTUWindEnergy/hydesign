import numpy as np

from hydesign.costs.costs import battery_cost, pvp_cost, shared_cost, wpp_cost


def test_disabled_technology_costs_are_zero():
    wind = wpp_cost(1, 1, 1, 1, 1, 1, 1, N_time=1, enabled=False)
    solar = pvp_cost(1, 1, 1, 1, enabled=False)
    battery = battery_cost(1, 1, 1, 1, 1, life_y=1, enabled=False)

    assert wind.compute(1, 1, 1, 1, np.ones(1)) == (0.0, 0.0)
    assert solar.compute(1, 1) == (0.0, 0.0)
    assert battery.compute(1, 1, np.ones(8760)) == (0.0, 0.0)


def test_pv_cost_arithmetic():
    model = pvp_cost(
        solar_PV_cost=1,
        solar_hardware_installation_cost=2,
        solar_inverter_cost=3,
        solar_fixed_onm_cost=4,
    )

    assert model.compute(solar_MW=5, DC_AC_ratio=2) == (45, 40)
    np.testing.assert_allclose(
        model.compute_partials(solar_MW=5, DC_AC_ratio=2),
        [7.875, 10.3125, 8, 20],
    )


def test_shared_cost_uses_only_present_technologies():
    wind_only = shared_cost(2, 3, 10, technologies={"wind"})
    solar_only = shared_cost(2, 3, 10, technologies={"solar"})

    assert wind_only.compute(G_MW=4, Awpp=5, Apvp=20) == (70, 0, 20, 50, 0)
    assert solar_only.compute(G_MW=4, Awpp=5, Apvp=20) == (220, 0, 20, 0, 200)
    assert wind_only.compute_partials(Awpp=5, Apvp=20)["CAPEX_sh", "Awpp"] == 10
    assert solar_only.compute_partials(Awpp=5, Apvp=20)["CAPEX_sh", "Apvp"] == 10
