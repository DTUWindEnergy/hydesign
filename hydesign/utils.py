# -*- coding: utf-8 -*-
import numpy as np
import openmdao.api as om

from hydesign.openmdao_wrapper import ComponentWrapper


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

        N_limit = self.N_limit
        life_y = self.life_y
        # life_h = self.life_h

        # SoH = inputs['SoH']
        delta_life = int(delta_life)

        SoH_shifted = np.concatenate(
            (
                np.zeros(delta_life * 365 * 24),
                SoH[0 : life_y * 365 * 24],
                np.zeros((N_limit - delta_life) * 365 * 24),
            )
        )
        return SoH_shifted


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
