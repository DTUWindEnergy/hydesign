# -*- coding: utf-8 -*-
import numpy as np
import openmdao.api as om

from hydesign.openmdao_wrapper import ComponentWrapper


def shift_project_profile(
    profile,
    operation_start_year,
    component_life_y,
    project_life_y,
    intervals_per_hour=1,
):
    """Shift a component-lifetime profile into the full project lifetime.

    Parameters
    ----------
    profile : ndarray
        Profile defined over the component lifetime.
    operation_start_year : int or float
        Commissioning year counted from project year 0.
    component_life_y : int
        Component lifetime in years.
    project_life_y : int
        Project lifetime in years.
    intervals_per_hour : int, optional
        Number of intervals per hour.

    Returns
    -------
    ndarray
        Project-lifetime array with zeros outside the component operating window.
    """

    start_year = int(np.asarray(operation_start_year).item())
    intervals_per_year = 365 * 24 * intervals_per_hour
    component_intervals = int(component_life_y * intervals_per_year)
    project_intervals = int(project_life_y * intervals_per_year)
    start_idx = start_year * intervals_per_year
    component_start_idx = 0
    if start_idx < 0:
        component_start_idx = -start_idx
        start_idx = 0

    end_idx = min(
        start_idx + (component_intervals - component_start_idx), project_intervals
    )

    shifted = np.zeros(project_intervals)
    if end_idx <= start_idx:
        return shifted

    component_profile = np.asarray(profile)[:component_intervals]
    shifted[start_idx:end_idx] = component_profile[
        component_start_idx : component_start_idx + (end_idx - start_idx)
    ]
    return shifted


def get_weights(grid, xtgt, maxorder):
    """Return finite-difference weights on an arbitrary grid.

    Parameters
    ----------
    grid : array-like
        Grid points where the function is sampled.
    xtgt : float
        Target location at which the derivative is approximated.
    maxorder : int
        Highest derivative order to compute.

    Returns
    -------
    numpy.ndarray
        Weight matrix with shape ``(len(grid), maxorder + 1)``.

    Notes
    -----
    Based on Fornberg's method for generating finite-difference formulas:
        @article{fornberg_generation_1988,
         title={Generation of finite difference formulas on arbitrarily spaced grids},
         author={Fornberg, Bengt},
         journal={Mathematics of computation},
         volume={51},
         number={184},
         pages={699--706},
         year={1988}
         doi={10.1090/S0025-5718-1988-0935077-0}
         }

    """
    x = grid
    z = xtgt
    m = maxorder

    #    nd: Number of data points - 1
    nd = len(x) - 1

    c = np.zeros((nd + 1, m + 1))
    c1 = 1.0
    c4 = x[0] - z
    c[0, 0] = 1.0
    for i in range(1, nd + 1):
        mn = min(i, m)
        c2 = 1.0
        c5 = c4
        c4 = x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[i, k] = c1 * (k * c[i - 1, k - 1] - c5 * c[i - 1, k]) / c2
                c[i, 0] = -c1 * c5 * c[i - 1, 0] / c2
            for k in range(mn, 0, -1):
                c[j, k] = (c4 * c[j, k] - k * c[j, k - 1]) / c3
            c[j, 0] = c4 * c[j, 0] / c3
        c1 = c2
    return c


class hybridization_shifted:
    def __init__(
        self,
        N_limit,
        life_y,
        N_time,
        life_h,
    ):
        """
        The hybridization_shifted model is used to shift the battery activity, in order to make them work starting from the chosen year (delta_life)

        Parameters
        ----------
        N_limit : int
            maximum number of years after start of operation at which the plant can be hybridized.
        life_y : int
            life time in years.
        life_h : int
            life time in hours.

        Returns
        -------
        None.

        """
        self.N_limit = N_limit
        self.life_y = life_y
        self.life_h = life_h
        self.N_time = N_time

    def compute(self, delta_life, SoH, **kwargs):
        project_life_y = self.life_y + self.N_limit
        return shift_project_profile(
            profile=SoH,
            operation_start_year=delta_life,
            component_life_y=self.life_y,
            project_life_y=project_life_y,
        )


class hybridization_shifted_comp(ComponentWrapper):
    def __init__(self, N_limit, life_y, N_time, life_h):
        model = hybridization_shifted(N_limit, life_y, N_time, life_h)
        super().__init__(
            inputs=[
                (
                    "delta_life",
                    {
                        "desc": "Years between the starting of operations of the existing plant and the new plant"
                    },
                ),
                (
                    "SoH",
                    {
                        "desc": "Battery state of health at discretization levels",
                        "shape": [life_h],
                    },
                ),
            ],
            outputs=[
                (
                    "SoH_shifted",
                    {
                        "desc": "Battery state of health at discretization levels shifted of delta_life",
                        "shape": [life_h],
                    },
                )
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


class operation_shifted:
    def __init__(self, operation_start_year, component_life_y, project_life_y):
        self.operation_start_year = operation_start_year
        self.component_life_y = component_life_y
        self.project_life_y = project_life_y
        self.project_life_h = int(project_life_y * 365 * 24)

    def compute(self, profile, **kwargs):
        return shift_project_profile(
            profile=profile,
            operation_start_year=self.operation_start_year,
            component_life_y=self.component_life_y,
            project_life_y=self.project_life_y,
        )


class operation_shifted_comp(ComponentWrapper):
    def __init__(self, operation_start_year, component_life_y, project_life_y):
        model = operation_shifted(
            operation_start_year=operation_start_year,
            component_life_y=component_life_y,
            project_life_y=project_life_y,
        )
        super().__init__(
            inputs=[
                (
                    "profile",
                    {
                        "desc": "Component lifetime profile",
                        "shape": [model.project_life_h],
                    },
                )
            ],
            outputs=[
                (
                    "profile_shifted",
                    {
                        "desc": "Project lifetime profile shifted to the commissioning year",
                        "shape": [model.project_life_h],
                    },
                )
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


def sample_mean(outs):
    """Return the mean of all samples along the first axis.

    Parameters
    ----------
    outs : ndarray
        Array of samples with shape ``(n_samples, ...)``.

    Returns
    -------
    ndarray
        Mean value over ``axis=0``.
    """
    return np.mean(outs, axis=0)
