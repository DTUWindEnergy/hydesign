import os

import numpy as np
import pandas as pd
import pytest
import yaml

import hydesign.ems.ems as ems_module
from hydesign.assembly.hpp_assembly import hpp_model
from hydesign.ems.ems import ems, operation_solar_batt_deg
from hydesign.examples import examples_filepath
from hydesign.finance.finance import build_finance_vectors, finance
from hydesign.technology import (
    normalize_technology_schedule,
    project_lifetime,
    technology_stage_intervals,
)


def test_normalize_technology_schedule_list_and_mapping():
    schedule = normalize_technology_schedule(
        [
            {"name": "wpp", "deployment_year": 0, "lifetime": 10},
            {"name": "bess", "year": 5, "life_y": 4},
            {"name": "pv", "deployment_year": 9, "lifetime": 7},
        ]
    )
    assert set(schedule) == {"wind", "battery", "solar"}
    assert schedule["battery"]["end_year"] == 9
    assert project_lifetime(schedule) == 16

    mapping = normalize_technology_schedule(
        {"wind": {"deployment_year": 2, "lifetime": 3}}
    )
    assert mapping["wind"] == {
        "deployment_year": 2,
        "lifetime": 3,
        "end_year": 5,
        "capex_phasing": {"years": [0], "shares": [1.0]},
    }

    phased = normalize_technology_schedule(
        [
            {
                "name": "battery",
                "deployment_year": 5,
                "lifetime": 4,
                "capex_phasing": {"years": [-2, -1, 0], "shares": [1, 2, 1]},
            }
        ]
    )
    assert phased["battery"]["capex_phasing"] == {
        "years": [-2, -1, 0],
        "shares": [0.25, 0.5, 0.25],
    }


@pytest.mark.parametrize(
    "technologies",
    [
        [],
        [{"name": "wind", "deployment_year": -1, "lifetime": 2}],
        [{"name": "unknown", "deployment_year": 0, "lifetime": 2}],
        [
            {"name": "pv", "deployment_year": 0, "lifetime": 2},
            {"name": "solar", "deployment_year": 1, "lifetime": 2},
        ],
        [
            {
                "name": "wind",
                "deployment_year": 0,
                "lifetime": 2,
                "capex_phasing": {"years": [-1, 0], "shares": [0, 0]},
            }
        ],
        [
            {
                "name": "wind",
                "deployment_year": 0,
                "lifetime": 2,
                "capex_phasing": {"years": [0, 1], "shares": [0.5, 0.5]},
            }
        ],
    ],
)
def test_invalid_technology_schedules(technologies):
    with pytest.raises(ValueError):
        normalize_technology_schedule(technologies)


def test_ems_uses_the_active_technology_stage(monkeypatch):
    schedule = normalize_technology_schedule(
        [
            {"name": "wind", "deployment_year": 0, "lifetime": 3},
            {"name": "battery", "deployment_year": 1, "lifetime": 1},
            {"name": "solar", "deployment_year": 2, "lifetime": 1},
        ]
    )

    calls = []

    def fake_ems(wind_ts, solar_ts, P_batt_MW, E_batt_MWh_t, **kwargs):
        calls.append((wind_ts.max(), solar_ts.max(), P_batt_MW))
        n = len(wind_ts)
        hpp = wind_ts.to_numpy() + solar_ts.to_numpy()
        battery = np.full(n, P_batt_MW)
        soc = np.full(n + 1, float(E_batt_MWh_t.iloc[0]))
        return hpp, np.zeros(n), battery, soc, np.zeros(n)

    monkeypatch.setattr(ems_module, "ems_cplex", fake_ems)
    model = ems(
        N_time=24,
        life_y=3,
        ems_type="cplex",
        technology_schedule=schedule,
    )
    outputs = model.compute(
        wind_t=np.ones(24),
        solar_t=np.full(24, 2.0),
        price_t=np.ones(24),
        b_P=4.0,
        b_E=8.0,
        G_MW=10.0,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.9,
        peak_hr_quantile=0.9,
        cost_of_battery_P_fluct_in_peak_price_ratio=0.0,
        n_full_power_hours_expected_per_day_at_peak_price=0.0,
    )
    _, _, _, hpp, _, battery, soc, _ = outputs
    year = 365 * 24
    np.testing.assert_array_equal(hpp[:year], 1.0)
    np.testing.assert_array_equal(hpp[year : 2 * year], 1.0)
    np.testing.assert_array_equal(hpp[2 * year :], 3.0)
    np.testing.assert_array_equal(battery[:year], 0.0)
    np.testing.assert_array_equal(battery[year : 2 * year], 4.0)
    np.testing.assert_array_equal(battery[2 * year :], 0.0)
    assert len(soc) == 3 * year + 1
    assert calls == [(1.0, 0.0, 0.0), (1.0, 0.0, 4.0), (1.0, 2.0, 0.0)]
    assert technology_stage_intervals(schedule) == [
        {"start_year": 0, "end_year": 1, "technologies": ("wind",)},
        {
            "start_year": 1,
            "end_year": 2,
            "technologies": ("wind", "battery"),
        },
        {
            "start_year": 2,
            "end_year": 3,
            "technologies": ("wind", "solar"),
        },
    ]


def test_finance_vectors_include_incremental_shared_capex_and_phasing():
    schedule = normalize_technology_schedule(
        [
            {"name": "wind", "deployment_year": 0, "lifetime": 12},
            {
                "name": "battery",
                "deployment_year": 5,
                "lifetime": 4,
                "capex_phasing": {"years": [-1, 0], "shares": [0.4, 0.6]},
            },
            {"name": "solar", "deployment_year": 7, "lifetime": 5},
        ]
    )
    vectors = build_finance_vectors(
        technology_schedule=schedule,
        project_life_y=12,
        CAPEX_w=100,
        CAPEX_s=80,
        CAPEX_b=50,
        CAPEX_sh=70,
        OPEX_w=10,
        OPEX_s=8,
        OPEX_b=5,
        OPEX_sh=0,
        CAPEX_sh_fixed=20,
        CAPEX_sh_w_land=30,
        CAPEX_sh_s_land=50,
    )

    assert vectors["capex_wind"][0] == 100
    assert vectors["capex_battery"][4] == 20
    assert vectors["capex_battery"][5] == 30
    assert vectors["capex_solar"][7] == 80
    assert vectors["capex_shared"][0] == 50
    assert vectors["capex_shared"][7] == 20
    assert vectors["shared_events"] == {0: 50, 7: 20}
    np.testing.assert_array_equal(vectors["opex_wind"], np.full(12, 10.0))
    np.testing.assert_array_equal(
        vectors["opex_battery"], [0, 0, 0, 0, 0, 5, 5, 5, 5, 0, 0, 0]
    )
    np.testing.assert_array_equal(
        vectors["opex_solar"], [0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 8]
    )


def test_battery_deployment_initializes_soc_without_predeployment_power():
    zeros = np.zeros(3)
    outputs = operation_solar_batt_deg(
        wind_t_deg=zeros,
        solar_t_deg=zeros,
        batt_degradation=np.array([0.0, 1.0, 1.0]),
        wind_t=zeros,
        solar_t=zeros,
        hpp_curt_t=zeros,
        b_t=zeros,
        b_E_SOC_t=np.array([0.0, 5.0, 5.0, 5.0]),
        G_MW=10,
        b_E=10,
        battery_depth_of_discharge=0.8,
        battery_charge_efficiency=0.95,
        price_ts=np.ones(3),
        n_full_power_hours_expected_per_day_at_peak_price=0,
    )
    _, _, battery_t, soc_t, _ = outputs

    assert battery_t[0] == 0
    assert soc_t[1] == 5
    np.testing.assert_array_equal(battery_t, zeros)


def test_scheduled_finance_places_capex_and_opex_in_active_years():
    schedule = normalize_technology_schedule(
        [
            {"name": "wind", "deployment_year": 0, "lifetime": 2},
            {"name": "battery", "deployment_year": 1, "lifetime": 1},
        ]
    )
    model = finance(
        N_time=365 * 24,
        life_y=2,
        project_life_y=2,
        depreciation_yr=[0, 1],
        depreciation=[0, 1],
        inflation_yr=[-1, 0, 2],
        inflation=[0, 0, 0],
        ref_yr_inflation=0,
        phasing_yr=[0],
        phasing_CAPEX=[1],
        wind_operation_year=0,
        battery_operation_year=1,
        technology_schedule=schedule,
    )
    zeros = np.zeros(2 * 365 * 24)
    outputs = model.compute(
        hpp_t_with_deg=zeros,
        penalty_t=zeros,
        price_t_ext=zeros,
        CAPEX_w=100.0,
        CAPEX_s=0.0,
        CAPEX_b=50.0,
        CAPEX_sh=10.0,
        OPEX_w=10.0,
        OPEX_s=0.0,
        OPEX_b=5.0,
        OPEX_sh=0.0,
        wind_WACC=0.0,
        solar_WACC=0.0,
        battery_WACC=0.0,
        tax_rate=0.0,
        decommissioning_cost_tot_w=0.0,
        decommissioning_cost_tot_s=0.0,
    )
    CAPEX, OPEX, _, NPV = outputs[:4]
    assert CAPEX == 160.0
    assert OPEX == 15.0
    assert NPV == -185.0


def test_base_hpp_accepts_schedule_from_constructor(tmp_path):
    sites = pd.read_csv(f"{examples_filepath}examples_sites.csv", index_col=0, sep=";")
    site = sites.loc[sites.name == "France_good_wind"].iloc[0]
    hpp = hpp_model(
        sim_pars_fn=examples_filepath + site["sim_pars_fn"],
        input_ts_fn=examples_filepath + site["input_ts_fn"],
        latitude=site["latitude"],
        longitude=site["longitude"],
        altitude=site["altitude"],
        verbose=False,
        work_dir=str(tmp_path) + os.sep,
        weeks_per_season_per_year=1,
        technologies=[
            {"name": "wind", "deployment_year": 0, "lifetime": 2},
            {"name": "battery", "deployment_year": 1, "lifetime": 1},
            {"name": "solar", "deployment_year": 1, "lifetime": 2},
        ],
    )
    assert hpp.project_life_y == 3
    assert hpp.prob.get_val("ems.wind_t_ext").shape == (3 * 365 * 24,)
    outputs = hpp.evaluate(20, 350, 4, 5, 7, 10, 35, 180, 1.2, 2, 2, 0)
    assert np.all(np.isfinite(outputs))


def test_base_hpp_accepts_schedule_from_yaml_and_omits_technologies(tmp_path):
    sites = pd.read_csv(f"{examples_filepath}examples_sites.csv", index_col=0, sep=";")
    site = sites.loc[sites.name == "France_good_wind"].iloc[0]
    source_config = examples_filepath + site["sim_pars_fn"]
    with open(source_config) as stream:
        sim_pars = yaml.safe_load(stream)
    sim_pars["technologies"] = [{"name": "wind", "deployment_year": 0, "lifetime": 2}]
    config = tmp_path / "sim_pars.yml"
    config.write_text(yaml.safe_dump(sim_pars), encoding="utf-8")

    hpp = hpp_model(
        sim_pars_fn=str(config),
        input_ts_fn=examples_filepath + site["input_ts_fn"],
        latitude=site["latitude"],
        longitude=site["longitude"],
        altitude=site["altitude"],
        verbose=False,
        work_dir=str(tmp_path) + os.sep,
        weeks_per_season_per_year=1,
    )
    outputs = hpp.evaluate(20, 350, 4, 5, 7, 10, 35, 180, 1.2, 2, 2, 0)

    assert hpp.project_life_y == 2
    assert hpp.prob.get_val("pvp_cost.CAPEX_s").item() == 0
    assert hpp.prob.get_val("battery_cost.CAPEX_b").item() == 0
    assert np.all(np.isfinite(outputs))
