# Auto-generated combined EMS module
# Combined from original EMS source of truth: hydesign/ems/ems.py:13-193, hydesign/ems/ems.py:604-927, hydesign/ems/ems.py:66-117, hydesign/ems/ems.py:865-869, hydesign/ems/ems.py:807-919

import numpy as np
import openmdao.api as om
import pandas as pd
from docplex.mp.model import Model

from hydesign.openmdao_wrapper import ComponentWrapper


class ems:
    """Generated baseline EMS for wind + solar + battery using the original CPLEX logic."""

    def __init__(
        self,
        N_time,
        life_y=25,
        intervals_per_hour=1,
        weeks_per_season_per_year=None,
        ems_type="cplex",
    ):
        self.weeks_per_season_per_year = weeks_per_season_per_year
        self.N_time = int(N_time)
        self.ems_type = ems_type
        self.life_y = life_y
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.intervals_per_hour = intervals_per_hour

    def compute(
        self,
        wind_t,
        solar_t,
        price_t,
        b_P,
        b_E,
        G_MW,
        battery_depth_of_discharge,
        battery_charge_efficiency,
        peak_hr_quantile,
        cost_of_battery_P_fluct_in_peak_price_ratio,
        n_full_power_hours_expected_per_day_at_peak_price,
        **kwargs,
    ):
        if self.ems_type != "cplex":
            raise Warning("This generated EMS only supports cplex")

        peak_hr_quantile = np.asarray(peak_hr_quantile).item()
        n_full_power_hours_expected_per_day_at_peak_price = np.asarray(
            n_full_power_hours_expected_per_day_at_peak_price
        ).item()
        battery_depth_of_discharge = np.asarray(battery_depth_of_discharge).item()
        battery_charge_efficiency = np.asarray(battery_charge_efficiency).item()
        cost_of_battery_P_fluct_in_peak_price_ratio = np.asarray(
            cost_of_battery_P_fluct_in_peak_price_ratio
        ).item()

        WSPr_df = pd.DataFrame(
            index=pd.date_range(
                start="01-01-1991 00:00", periods=len(wind_t), freq="1h"
            )
        )
        WSPr_df["price_t"] = price_t
        WSPr_df["wind_t"] = wind_t
        WSPr_df["solar_t"] = solar_t
        WSPr_df["E_batt_MWh_t"] = np.asarray(b_E).item()

        P_HPP_ts, P_curtailment_ts, P_charge_discharge_ts, E_SOC_ts, penalty_ts = (
            ems_cplex(
                wind_ts=WSPr_df.wind_t,
                solar_ts=WSPr_df.solar_t,
                price_ts=WSPr_df.price_t,
                P_batt_MW=np.asarray(b_P).item(),
                E_batt_MWh_t=WSPr_df.E_batt_MWh_t,
                hpp_grid_connection=np.asarray(G_MW).item(),
                battery_depth_of_discharge=float(battery_depth_of_discharge),
                charge_efficiency=float(battery_charge_efficiency),
                peak_hr_quantile=peak_hr_quantile,
                cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio,
                n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price,
            )
        )

        outputs = {}
        outputs["price_t_ext"] = expand_to_lifetime(
            price_t,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["hpp_t"] = expand_to_lifetime(
            P_HPP_ts,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["hpp_curt_t"] = expand_to_lifetime(
            P_curtailment_ts,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["penalty_t"] = expand_to_lifetime(
            penalty_ts,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["wind_t_ext"] = expand_to_lifetime(
            wind_t,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["solar_t_ext"] = expand_to_lifetime(
            solar_t,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["b_t"] = expand_to_lifetime(
            P_charge_discharge_ts,
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
        )
        outputs["b_E_SOC_t"] = expand_to_lifetime(
            E_SOC_ts[:-1],
            life_y=self.life_y,
            intervals_per_hour=self.intervals_per_hour,
            weeks_per_season_per_year=self.weeks_per_season_per_year,
            additional_intervals=1,
        )
        return (
            outputs["wind_t_ext"],
            outputs["solar_t_ext"],
            outputs["price_t_ext"],
            outputs["hpp_t"],
            outputs["hpp_curt_t"],
            outputs["b_t"],
            outputs["b_E_SOC_t"],
            outputs["penalty_t"],
        )


class ems_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = ems(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "wind_t",
                    {
                        "units": "MW",
                        "desc": "WPP power time series",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "solar_t",
                    {
                        "units": "MW",
                        "desc": "PVP power time series",
                        "shape": [model.N_time],
                    },
                ),
                (
                    "price_t",
                    {"desc": "Electricity price time series", "shape": [model.N_time]},
                ),
                ("b_P", {"units": "MW", "desc": "Battery power capacity"}),
                ("b_E", {"desc": "Battery energy storage capacity"}),
                ("G_MW", {"units": "MW", "desc": "Grid capacity"}),
                (
                    "battery_depth_of_discharge",
                    {"units": "MW", "desc": "battery depth of discharge"},
                ),
                ("battery_charge_efficiency", {"desc": "battery charge efficiency"}),
                (
                    "peak_hr_quantile",
                    {
                        "desc": "Quantile of price tim sereis to define peak price hours (above this quantile).\nOnly used for peak production penalty and for cost of battery degradation."
                    },
                ),
                (
                    "cost_of_battery_P_fluct_in_peak_price_ratio",
                    {
                        "desc": "cost of battery power fluctuations computed as a peak price ratio."
                    },
                ),
                (
                    "n_full_power_hours_expected_per_day_at_peak_price",
                    {
                        "desc": "Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached."
                    },
                ),
            ],
            outputs=[
                (
                    "wind_t_ext",
                    {
                        "units": "MW",
                        "desc": "WPP power time series",
                        "shape": [model.life_h],
                    },
                ),
                (
                    "solar_t_ext",
                    {
                        "units": "MW",
                        "desc": "PVP power time series",
                        "shape": [model.life_h],
                    },
                ),
                (
                    "price_t_ext",
                    {"desc": "Electricity price time series", "shape": [model.life_h]},
                ),
                (
                    "hpp_t",
                    {
                        "units": "MW",
                        "desc": "HPP power time series",
                        "shape": [model.life_h],
                    },
                ),
                (
                    "hpp_curt_t",
                    {
                        "units": "MW",
                        "desc": "HPP curtailed power time series",
                        "shape": [model.life_h],
                    },
                ),
                (
                    "b_t",
                    {
                        "units": "MW",
                        "desc": "Battery charge/discharge power time series",
                        "shape": [model.life_h],
                    },
                ),
                (
                    "b_E_SOC_t",
                    {
                        "desc": "Battery energy SOC time series",
                        "shape": [model.life_h + 1],
                    },
                ),
                (
                    "penalty_t",
                    {
                        "desc": "penalty for not reaching expected energy productin at peak hours",
                        "shape": [model.life_h],
                    },
                ),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


def expand_to_lifetime(
    x,
    life_y=25,
    intervals_per_hour=1,
    weeks_per_season_per_year=None,
    axis=0,
    additional_intervals=0,
    life=None,
):
    if life is None:
        life = life_y * 365 * 24 * intervals_per_hour
    life = life + additional_intervals
    if weeks_per_season_per_year == None:
        len_x = np.size(x, axis=axis)
        N_repeats = int(np.ceil(life / len_x))
    else:
        x_ext = np.array([])
        for x_batch in split_in_batch(
            x, weeks_per_season_per_year * 7 * 24 * intervals_per_hour
        ):
            x_ext = np.append(
                x_ext, np.tile(x_batch, 20)[: 24 * 7 * 13 * intervals_per_hour]
            )
        x_ext = np.append(x_ext, x_batch[-24 * intervals_per_hour :])
        x = x_ext
        N_repeats = int(np.ceil(life / (365 * 24 * intervals_per_hour)))
    return np.tile(x, N_repeats)[:life]


def ems_cplex(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.5,
    n_full_power_hours_expected_per_day_at_peak_price=3,
    batch_size=110,
):
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    batches = batches_all
    P_HPP_ts = np.zeros(len(price_ts))
    P_curtailment_ts = np.zeros(len(price_ts))
    penalty_ts = np.zeros(len(price_ts))
    P_charge_discharge_ts = np.zeros(len(price_ts))
    E_SOC_ts = np.zeros(len(price_ts) + 1)

    for batch in batches:
        price_ts_sel = price_ts.iloc[batch]
        wind_ts_sel = wind_ts.iloc[batch]
        solar_ts_sel = solar_ts.iloc[batch]
        E_batt_MWh_t_sel = E_batt_MWh_t.iloc[batch]
        (
            P_HPP_ts_batch,
            P_curtailment_ts_batch,
            P_charge_discharge_ts_batch,
            E_SOC_ts_batch,
            penalty_batch,
        ) = ems_cplex_parts(
            wind_ts=wind_ts_sel,
            solar_ts=solar_ts_sel,
            price_ts=price_ts_sel,
            P_batt_MW=P_batt_MW,
            E_batt_MWh_t=E_batt_MWh_t_sel,
            hpp_grid_connection=hpp_grid_connection,
            battery_depth_of_discharge=battery_depth_of_discharge,
            charge_efficiency=charge_efficiency,
            peak_hr_quantile=peak_hr_quantile,
            cost_of_battery_P_fluct_in_peak_price_ratio=cost_of_battery_P_fluct_in_peak_price_ratio,
            n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price,
        )
        P_HPP_ts[batch] = P_HPP_ts_batch
        P_curtailment_ts[batch] = P_curtailment_ts_batch
        penalty_ts[batch] = penalty_batch
        P_charge_discharge_ts[batch] = P_charge_discharge_ts_batch
        E_SOC_ts[batch] = E_SOC_ts_batch[:-1]

    E_SOC_ts[-1] = E_SOC_ts[0]
    return (
        P_HPP_ts,
        P_curtailment_ts,
        P_charge_discharge_ts,
        E_SOC_ts,
        penalty_ts,
    )


def ems_cplex_parts(
    wind_ts,
    solar_ts,
    price_ts,
    P_batt_MW,
    E_batt_MWh_t,
    hpp_grid_connection,
    battery_depth_of_discharge,
    charge_efficiency,
    peak_hr_quantile=0.9,
    cost_of_battery_P_fluct_in_peak_price_ratio=0.5,
    n_full_power_hours_expected_per_day_at_peak_price=3,
):
    N_t = len(price_ts.index)
    N_days = N_t / 24
    e_peak_day_expected = (
        n_full_power_hours_expected_per_day_at_peak_price * hpp_grid_connection
    )
    e_peak_period_expected = e_peak_day_expected * N_days
    price_peak = np.quantile(price_ts.values, peak_hr_quantile)
    peak_hours_index = np.where(price_ts.values >= price_peak)[0]

    price_ts_to_max = price_peak - price_ts
    price_ts_to_max.loc[price_ts_to_max < 0] = 0
    price_ts_to_max.iloc[:-1] = (
        0.5 * price_ts_to_max.iloc[:-1].values + 0.5 * price_ts_to_max.iloc[1:].values
    )

    mdl = Model(name="EMS")
    mdl.context.cplex_parameters.threads = 1
    mdl.context.cplex_parameters.emphasis.mip = 1

    time = price_ts.index
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta("1hour")]))

    P_HPP_t = mdl.continuous_var_dict(
        time, lb=0, ub=hpp_grid_connection, name="HPP power output"
    )
    P_curtailment_t = mdl.continuous_var_dict(time, lb=0, name="Curtailment")
    penalty = mdl.continuous_var(name="penalty", lb=-1e12)
    e_penalty = mdl.continuous_var(name="e_penalty", lb=-1e12)
    P_charge_discharge_t = mdl.continuous_var_dict(
        time,
        lb=-P_batt_MW / charge_efficiency,
        ub=P_batt_MW * charge_efficiency,
        name="Battery power",
    )
    E_SOC_t = mdl.continuous_var_dict(SOCtime, lb=0, name="Energy level")
    fabs = mdl.piecewise(-1, [(0, 0)], 1)

    mdl.maximize(
        mdl.sum(price_ts[t] * P_HPP_t[t] for t in time)
        - penalty
        - mdl.sum(
            fabs(
                P_charge_discharge_t[t + pd.Timedelta("1hour")]
                - P_charge_discharge_t[t]
            )
            * cost_of_battery_P_fluct_in_peak_price_ratio
            * price_ts_to_max[t]
            for t in time[:-1]
        )
    )

    mdl.add_constraint(
        e_penalty
        == (
            e_peak_period_expected - mdl.sum(P_HPP_t[time[i]] for i in peak_hours_index)
        )
    )
    f1 = mdl.piecewise(0, [(0, 0)], 1)
    mdl.add_constraint(E_SOC_t[SOCtime[0]] == 0.5 * E_batt_MWh_t[time[0]])
    mdl.add_constraint(E_SOC_t[SOCtime[-1]] == 0.5 * E_batt_MWh_t[time[0]])
    f2 = mdl.piecewise(charge_efficiency, [(0, 0)], 1 / charge_efficiency)
    mdl.add_constraint(penalty == price_peak * f1(e_penalty))

    for t in time:
        tt = t + pd.Timedelta("1hour")
        dt = 1
        mdl.add_constraint(
            P_HPP_t[t]
            == -P_curtailment_t[t] + wind_ts[t] + solar_ts[t] + P_charge_discharge_t[t]
        )
        mdl.add_constraint(E_SOC_t[tt] == E_SOC_t[t] - f2(P_charge_discharge_t[t]) * dt)
        mdl.add_constraint(
            E_SOC_t[t] >= (1 - battery_depth_of_discharge) * E_batt_MWh_t[t]
        )
        mdl.add_constraint(E_SOC_t[t] <= E_batt_MWh_t[t])
        mdl.add_constraint(P_charge_discharge_t[t] <= P_batt_MW * charge_efficiency)
        mdl.add_constraint(P_charge_discharge_t[t] >= -P_batt_MW / charge_efficiency)

    sol = mdl.solve(log_output=False)

    P_HPP_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_HPP_t), orient="index"
    ).loc[:, 0]
    P_curtailment_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_curtailment_t), orient="index"
    ).loc[:, 0]
    P_charge_discharge_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(P_charge_discharge_t), orient="index"
    ).loc[:, 0]
    E_SOC_ts_df = pd.DataFrame.from_dict(
        sol.get_value_dict(E_SOC_t), orient="index"
    ).loc[:, 0]

    penalty_2 = sol.get_value(penalty)
    penalty_ts = np.ones(N_t) * (penalty_2 / N_t)

    mdl.end()

    P_HPP_ts = P_HPP_ts_df.reindex(time, fill_value=0).values
    P_curtailment_ts = P_curtailment_ts_df.reindex(time, fill_value=0).values
    P_charge_discharge_ts = P_charge_discharge_ts_df.reindex(time, fill_value=0).values
    E_SOC_ts = E_SOC_ts_df.reindex(SOCtime, fill_value=0).values

    if len(P_HPP_ts_df) < len(wind_ts):
        P_HPP_ts = wind_ts.values + solar_ts.values + P_charge_discharge_ts

    pass
    return (
        P_HPP_ts,
        P_curtailment_ts,
        P_charge_discharge_ts,
        E_SOC_ts,
        penalty_ts,
    )


def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N : end]]
        counter += 1
    return batch
