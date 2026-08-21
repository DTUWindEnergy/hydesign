"""Shared utilities for energy-management-system time series."""

import numpy as np


def _expand_seasonal_weeks(x, weeks_per_season_per_year, intervals_per_hour=1):
    """Build non-leap years by repeating representative meteorological weeks."""
    x = np.asarray(x).ravel()
    season_h = weeks_per_season_per_year * 7 * 24 * intervals_per_hour
    year_input_h = 4 * season_h
    if len(x) % year_input_h:
        raise ValueError(
            "Seasonal input must contain equally sized DJF, MAM, JJA, and SON blocks"
        )

    day_h = 24 * intervals_per_hour

    def repeat_to(block, length):
        return np.tile(block, int(np.ceil(length / len(block))))[:length]

    years = []
    for start in range(0, len(x), year_input_h):
        djf, mam, jja, son = np.split(x[start : start + year_input_h], 4)
        # Keep winter continuous across the December-to-January year boundary.
        winter = repeat_to(djf, 90 * day_h)
        years.append(
            np.concatenate(
                [
                    winter[31 * day_h :],  # January-February (59 days)
                    repeat_to(mam, 92 * day_h),  # March-May
                    repeat_to(jja, 92 * day_h),  # June-August
                    repeat_to(son, 91 * day_h),  # September-November
                    winter[: 31 * day_h],  # December
                ]
            )
        )
    return np.concatenate(years)


def expand_to_lifetime(
    x,
    life_y=25,
    intervals_per_hour=1,
    weeks_per_season_per_year=None,
    axis=0,
    additional_intervals=0,
    life=None,
    life_h=None,
):
    """Repeat a time series to a requested project lifetime.

    When representative seasonal weeks are supplied, they are first expanded
    over their corresponding meteorological months in a 365-day year.

    ``life`` and ``life_h`` are equivalent explicit lifetime lengths;
    ``life_h`` is retained for compatibility with BM and SolarX callers.
    """
    if life is not None and life_h is not None:
        raise ValueError("Specify either life or life_h, not both")
    if life is None:
        life = life_h
    if life is None:
        life = life_y * 365 * 24 * intervals_per_hour
    life = life + additional_intervals

    if weeks_per_season_per_year is None:
        len_x = np.size(x, axis=axis)
        repeats = int(np.ceil(life / len_x))
    else:
        x = _expand_seasonal_weeks(
            x,
            weeks_per_season_per_year,
            intervals_per_hour=intervals_per_hour,
        )
        repeats = int(np.ceil(life / (365 * 24 * intervals_per_hour)))

    return np.tile(x, repeats)[:life]
