# Auto-generated combined EMS long-term operation module
# Combined from original EMS source of truth: hydesign/ems/ems.py:313-596, hydesign/ems/ems.py:1606-1752

import numpy as np
import openmdao.api as om
import pandas as pd
from docplex.mp.model import Model

from hydesign.openmdao_wrapper import ComponentWrapper


class ems_long_term_operation:
    """Long term operation EMS. Predicts the operation of the plant throughout the entire lifetime, taking into account the battery
    and PV degradations.

    Parameters
    ----------
    ii_time : indices on the liftime timeseries. Hydesign operates in each range at constant battery health.
    SoH : Battery state of health at discretization levels
    wind_t_ext_deg : WPP power time series with degradation [MW]
    solar_t_ext_deg : WPP power time series with degradation [MW]
    wind_t_ext : WPP power time series [MW]
    solar_t_ext : PVP power time series [MW]
    price_t_ext : Electricity price time series
    b_P : Battery power capacity
    b_E : Battery energy storage capacity
    G_MW : Grid capacity
    battery_depth_of_discharge : battery depth of discharge
    battery_charge_efficiency : battery charge efficiency
    hpp_curt_t : HPP curtailed power time series
    b_t : Battery charge/discharge power time series
    b_E_SOC_t : Battery energy SOC time series
    peak_hr_quantile : Quantile of price time series to define peak price hours
    n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached

    Returns
    -------
    hpp_t_with_deg : HPP power time series
    hpp_curt_t_with_deg : HPP curtailed power time series
    b_t_with_deg : Battery charge/discharge power time series
    b_E_SOC_t_with_deg : Battery energy SOC time series
    penalty_t_with_deg : penalty for not reaching expected energy production at peak hours
    total_curtailment : total curtailment in the lifetime
    """

    def __init__(
        self,
        N_time,
        num_batteries=1,
        life_y=25,
        intervals_per_hour=1,
        ems_type="energy_penalty",
        load_min_penalty_factor=1e6,
    ):

        # super().__init__()
        self.N_time = N_time
        self.intervals_per_hour = intervals_per_hour
        self.life_y = life_y
        self.life_h = 365 * 24 * life_y
        self.life_intervals = self.life_h * intervals_per_hour
        self.ems_type = ems_type
        self.load_min_penalty_factor = load_min_penalty_factor

    def compute(
        self,
        SoH,
        wind_t_ext_deg,
        solar_t_ext_deg,
        wind_t_ext,
        solar_t_ext,
        price_t_ext,
        b_P,
        b_E,
        G_MW,
        battery_depth_of_discharge,
        battery_charge_efficiency,
        hpp_curt_t,
        b_t,
        b_E_SOC_t,
        peak_hr_quantile,
        n_full_power_hours_expected_per_day_at_peak_price,
        load_min=None,
        **kwargs,
    ):

        peak_hr_quantile = np.asarray(peak_hr_quantile).item()
        n_full_power_hours_expected_per_day_at_peak_price = np.asarray(
            n_full_power_hours_expected_per_day_at_peak_price
        ).item()

        if self.ems_type == "energy_penalty":
            ems_longterm = operation_solar_batt_deg
        else:
            raise Warning("This class should only be used for energy_penalty")

        if self.intervals_per_hour == 1:
            freq = "1h"
        else:
            freq = f"{60 / self.intervals_per_hour:.0f}min"

        args = dict(
            wind_t_deg=wind_t_ext_deg,
            solar_t_deg=solar_t_ext_deg,
            batt_degradation=SoH,
            wind_t=wind_t_ext,
            solar_t=solar_t_ext,
            hpp_curt_t=hpp_curt_t,
            b_t=b_t,
            b_E_SOC_t=b_E_SOC_t,
            G_MW=G_MW[0],
            b_E=b_E[0],
            battery_depth_of_discharge=battery_depth_of_discharge[0],
            battery_charge_efficiency=battery_charge_efficiency[0],
            b_E_SOC_0=None,
            price_ts=price_t_ext,
            peak_hr_quantile=peak_hr_quantile,
            n_full_power_hours_expected_per_day_at_peak_price=n_full_power_hours_expected_per_day_at_peak_price,
            freq=freq,
        )

        Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat, penalty_t_with_deg = ems_longterm(
            **args
        )

        outputs = {}
        outputs["hpp_t_with_deg"] = Hpp_deg
        outputs["hpp_curt_t_with_deg"] = P_curt_deg
        outputs["b_t_with_deg"] = b_t_sat
        outputs["b_E_SOC_t_with_deg"] = b_E_SOC_t_sat
        outputs["penalty_t_with_deg"] = penalty_t_with_deg
        outputs["total_curtailment_with_deg"] = P_curt_deg.sum()
        outputs["total_curtailment"] = hpp_curt_t.sum()
        return (
            outputs["hpp_t_with_deg"],
            outputs["hpp_curt_t_with_deg"],
            outputs["b_t_with_deg"],
            outputs["b_E_SOC_t_with_deg"],
            outputs["penalty_t_with_deg"],
            outputs["total_curtailment_with_deg"],
            outputs["total_curtailment"],
        )


class ems_long_term_operation_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = ems_long_term_operation(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "SoH",
                    {
                        "desc": "Battery state of health at discretization levels",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "wind_t_ext_deg",
                    {
                        "units": "MW",
                        "desc": "Wind time series including degradation",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "solar_t_ext_deg",
                    {
                        "units": "MW",
                        "desc": "PV time series including degradation",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "wind_t_ext",
                    {
                        "units": "MW",
                        "desc": "WPP power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "solar_t_ext",
                    {
                        "units": "MW",
                        "desc": "PVP power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "price_t_ext",
                    {
                        "desc": "Electricity price time series",
                        "shape": [model.life_intervals],
                    },
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
                    "hpp_curt_t",
                    {
                        "units": "MW",
                        "desc": "HPP curtailed power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "b_t",
                    {
                        "units": "MW",
                        "desc": "Battery charge/discharge power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "b_E_SOC_t",
                    {
                        "desc": "Battery energy SOC time series",
                        "shape": [model.life_intervals + 1],
                    },
                ),
                (
                    "peak_hr_quantile",
                    {
                        "desc": "Quantile of price time series to define peak price hours (above this quantile).\n"
                        + "Only used for peak production penalty and for cost of battery degradation."
                    },
                ),
                (
                    "n_full_power_hours_expected_per_day_at_peak_price",
                    {
                        "desc": "Pnealty occurs if number of full power hours expected per day at peak price are not reached."
                    },
                ),
                ("load_min", {"desc": "Minimum electrical load to meet"}),
            ],
            outputs=[
                (
                    "hpp_t_with_deg",
                    {
                        "units": "MW",
                        "desc": "HPP power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "hpp_curt_t_with_deg",
                    {
                        "units": "MW",
                        "desc": "HPP curtailed power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "b_t_with_deg",
                    {
                        "units": "MW",
                        "desc": "Battery charge/discharge power time series",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "b_E_SOC_t_with_deg",
                    {
                        "desc": "Battery energy SOC time series",
                        "shape": [model.life_intervals + 1],
                    },
                ),
                (
                    "penalty_t_with_deg",
                    {
                        "desc": "penalty for not reaching expected energy production at peak hours",
                        "shape": [model.life_intervals],
                    },
                ),
                (
                    "total_curtailment_with_deg",
                    {
                        "units": "GW*h",
                        "desc": "total curtailment in the lifetime after degradation",
                    },
                ),
                (
                    "total_curtailment",
                    {"units": "GW*h", "desc": "total curtailment in the lifetime"},
                ),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


def operation_solar_batt_deg(
    wind_t_deg,
    solar_t_deg,
    batt_degradation,
    wind_t,  #  this method is not using wind_t
    solar_t,  #  this method is not using solar_t
    hpp_curt_t,
    b_t,
    b_E_SOC_t,
    G_MW,
    b_E,
    battery_depth_of_discharge,
    battery_charge_efficiency,
    price_ts,
    b_E_SOC_0=None,
    peak_hr_quantile=0.9,
    n_full_power_hours_expected_per_day_at_peak_price=3,
    freq="1h",
):
    """EMS operation for degraded PV and battery based on an existing EMS.

    Parameters
    ----------
    wind_t_deg: Wind time series including degradation
    solar_t_deg: PV time series including degradation
    batt_degradation: Battery degradation as health factor [0=dead,1=new]
    wind_t: WPP power time series
    solar_t: PVP power time series
    hpp_curt_t: HPP curtailment time series results form an EMS planed without degradation
    b_t: HPP battery power (charge/discharge) time series results form an EMS planed without degradation
    b_E_SOC_t: HPP battery state of charge (SoC) time series results form an EMS planed without degradation
    b_E_SOC_0: Initial charge status of the actual operation
    price_ts : price time series
    G_MW : grid connection
    E_batt_MWh_t : battery energy capacity time series
    battery_depth_of_discharge : battery depth of discharge
    battery_charge_efficiency : battery charge efficiency
    peak_hr_quantile : quantile of price time series to define peak price hours
    cost_of_battery_P_fluct_in_peak_price_ratio : cost of battery power fluctuations computed as a peak price ratio
    n_full_power_hours_expected_per_day_at_peak_price : Penalty occurs if number of full power hours expected per day at peak price are not reached

    Returns
    -------
    Hpp_deg : HPP power time series
    P_curt_deg : HPP curtailed power time series
    b_t_sat : Battery charge/discharge power time series
    b_E_SOC_t_sat : Battery energy SOC time series
    penalty_ts : penalty for not reaching expected energy production at peak hours
    """
    B_p = np.max(np.abs(b_t))

    # wind_solar_t = solar_t + wind_t
    wind_solar_t_deg = solar_t_deg + wind_t_deg
    # P_deg_t_sat_loss = (solar_t - solar_t_deg) + (wind_t - wind_t_deg)

    # P_loss =  np.maximum( 0 , P_deg_t_sat_loss  - hpp_curt_t)
    b_t_less_sol = b_t.copy()
    dt = 1

    # Reduction in power to battery due to reduction of solar
    for i in range(len(b_t)):
        if b_t[i] < 0:
            # Try to keep the ratio of b_t[i] / wind_solar_t[i]  SoC to the maximum
            # b_t_less_sol[i] = ( b_t[i] / wind_solar_t[i] ) * ( wind_solar_t_deg[i] )

            # Hajar's method
            # b_t_less_sol[i] = np.minimum(b_t[i] + P_loss[i],0)

            # Try to follow SoC to the maximum
            if -b_t[i] > wind_solar_t_deg[i]:
                b_t_less_sol[i] = -wind_solar_t_deg[i]

            b_t_less_sol[i] = np.clip(b_t_less_sol[i], -B_p, B_p)

    # Initialize the SoC
    b_E_SOC_t_sat = b_E_SOC_t.copy()
    if b_E_SOC_0 == None:
        try:
            b_E_SOC_t_sat[0] = b_E_SOC_t[0]
        except:
            raise ("len(b_E_SOC_t):", len(b_E_SOC_t))
    else:
        b_E_SOC_t_sat[0] = b_E_SOC_0

    # Update the SoC
    for i in range(len(b_t_less_sol)):
        if b_t_less_sol[i] < 0:  # charging
            b_E_SOC_t_sat[i + 1] = (
                b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt * battery_charge_efficiency
            )
        if b_t_less_sol[i] >= 0:  # discharging
            b_E_SOC_t_sat[i + 1] = (
                b_E_SOC_t_sat[i] - b_t_less_sol[i] * dt / battery_charge_efficiency
            )

        b_E_SOC_t_sat[i + 1] = np.clip(
            b_E_SOC_t_sat[i + 1],
            (1 - battery_depth_of_discharge) * b_E * batt_degradation[i],
            b_E * batt_degradation[i],
        )

    # Recompute the battery power
    b_t_sat = b_t.copy()
    for i in range(len(b_t_sat)):
        if b_t[i] < 0:
            b_t_sat[i] = (
                (b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i + 1]) / battery_charge_efficiency
            ) / dt
        elif b_t[i] >= 0:
            b_t_sat[i] = (
                (b_E_SOC_t_sat[i] - b_E_SOC_t_sat[i + 1]) * battery_charge_efficiency
            ) / dt

    Hpp_deg = np.minimum(wind_t_deg + solar_t_deg + b_t_sat, G_MW)
    P_curt_deg = np.maximum(wind_t_deg + solar_t_deg + b_t_sat - G_MW, 0)

    # Compute penalty per week
    H_df = pd.DataFrame(
        np.copy(Hpp_deg),
        columns=["hpp_t_with_deg"],
        index=pd.date_range(
            start="01-01-1991 00:00",
            periods=len(Hpp_deg),
            freq=freq,
        ),
    )
    H_df["price_t_ext"] = price_ts
    price_peak = float(np.quantile(H_df["price_t_ext"], peak_hr_quantile))
    H_df["peak"] = H_df["price_t_ext"] >= price_peak
    H_df.loc[~H_df["peak"], "hpp_t_with_deg"] = 0

    N_days = 7
    H_df_week = H_df.resample(f"{N_days}d").sum()

    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price * G_MW
    e_peak_period_expected = e_peak_day_expected * N_days

    H_df_week["e_peak_period_expected"] = e_peak_period_expected
    H_df_week["e_penalty"] = (
        H_df_week["e_peak_period_expected"] - H_df_week["hpp_t_with_deg"]
    )
    H_df_week["penalty"] = price_peak * np.maximum(0, H_df_week["e_penalty"])

    penalty_df = H_df_week["penalty"].reindex_like(H_df).fillna(0.0)
    penalty_ts = penalty_df.values.flatten()

    return Hpp_deg, P_curt_deg, b_t_sat, b_E_SOC_t_sat, penalty_ts
