# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:08:01 2024

@author: mikf
"""
from hydesign.openmdao_wrapper import ComponentWrapper


class battery_with_reliability:
    def __init__(
        self,
        life_y=25,
        intervals_per_hour=1,
        reliability_ts_battery=None,
        reliability_ts_trans=None,
    ):
        """Initialize the component with optional reliability time series.

        Parameters
        ----------
        life_y : int, optional
            Lifetime of the plant in years. Default is ``25``.
        intervals_per_hour : int, optional
            Number of simulation steps per hour. Default is ``1``.
        reliability_ts_battery : array-like, optional
            Time series describing battery availability.
        reliability_ts_trans : array-like, optional
            Time series describing transformer availability.
        """

        self.life_intervals = life_y * 365 * 24 * intervals_per_hour
        self.reliability_ts_battery = reliability_ts_battery
        self.reliability_ts_trans = reliability_ts_trans

    def compute(self, b_t, **kwargs):
        if (self.reliability_ts_battery is None) or (self.reliability_ts_trans is None):
            b_t_rel = b_t
            return b_t_rel
        b_t_rel = (
            b_t
            * self.reliability_ts_battery[: self.life_intervals]
            * self.reliability_ts_trans[: self.life_intervals]
        )
        return b_t_rel


class battery_with_reliability_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = battery_with_reliability(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "b_t",
                    {
                        "shape": [model.life_intervals],
                        "units": "MW",
                    },
                )
            ],
            outputs=[
                (
                    "b_t_rel",
                    {
                        "shape": [model.life_intervals],
                        "units": "MW",
                    },
                )
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class wpp_with_reliability:
    def __init__(
        self,
        life_y=25,
        intervals_per_hour=1,
        reliability_ts_wind=None,
        reliability_ts_trans=None,
    ):
        """Initialize the component with optional wind farm reliability data.

        Parameters
        ----------
        life_y : int, optional
            Lifetime of the plant in years. Default is ``25``.
        intervals_per_hour : int, optional
            Number of simulation steps per hour. Default is ``1``.
        reliability_ts_wind : array-like, optional
            Time series describing wind farm availability.
        reliability_ts_trans : array-like, optional
            Time series describing transformer availability.
        """

        self.life_intervals = life_y * 365 * 24 * intervals_per_hour
        self.reliability_ts_wind = reliability_ts_wind
        self.reliability_ts_trans = reliability_ts_trans

    def compute(self, wind_t, **kwargs):
        if (self.reliability_ts_wind is None) or (self.reliability_ts_trans is None):
            wind_t_rel = wind_t
            return wind_t_rel
        wind_t_rel = (
            wind_t
            * self.reliability_ts_wind[: self.life_intervals]
            * self.reliability_ts_trans[: self.life_intervals]
        )
        return wind_t_rel


class wpp_with_reliability_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = wpp_with_reliability(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "wind_t",
                    {
                        "shape": [model.life_intervals],
                        "units": "MW",
                    },
                )
            ],
            outputs=[
                (
                    "wind_t_rel",
                    {
                        "shape": [model.life_intervals],
                        "units": "MW",
                    },
                )
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class pvp_with_reliability:
    def __init__(
        self,
        life_y=25,
        intervals_per_hour=1,
        reliability_ts_pv=None,
        reliability_ts_trans=None,
    ):
        """Initialize the component with optional PV plant reliability data.

        Parameters
        ----------
        life_y : int, optional
            Lifetime of the plant in years. Default is ``25``.
        intervals_per_hour : int, optional
            Number of simulation steps per hour. Default is ``1``.
        reliability_ts_pv : array-like, optional
            Time series describing PV availability.
        reliability_ts_trans : array-like, optional
            Time series describing transformer availability.
        """

        self.life_intervals = life_y * 365 * 24 * intervals_per_hour
        self.reliability_ts_pv = reliability_ts_pv
        self.reliability_ts_trans = reliability_ts_trans

    def compute(self, solar_t, **kwargs):
        if (self.reliability_ts_pv is None) or (self.reliability_ts_trans is None):
            solar_t_rel = solar_t
            return solar_t_rel
        solar_t_rel = (
            solar_t
            * self.reliability_ts_pv[: self.life_intervals]
            * self.reliability_ts_trans[: self.life_intervals]
        )
        return solar_t_rel


class pvp_with_reliability_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = pvp_with_reliability(**insta_inp)
        super().__init__(
            inputs=[
                (
                    "solar_t",
                    {
                        "shape": [model.life_intervals],
                        "units": "MW",
                    },
                )
            ],
            outputs=[
                (
                    "solar_t_rel",
                    {
                        "shape": [model.life_intervals],
                        "units": "MW",
                    },
                )
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )
