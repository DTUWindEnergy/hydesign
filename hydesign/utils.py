# -*- coding: utf-8 -*-
import numpy as np
import openmdao.api as om

def get_weights(grid, xtgt, maxorder):
    """

    Populates weights for finite difference formulas for derivatives of various orders.

    Parameters
    ----------
    grid : array-like
        Target x value where derivatives are approximated
    xtgt : int
        Array of x values
    maxorder : int
        Maximum order of derivative

    Returns:
    c : array-like
        Array of weights

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
    nd = len(x) -1 
    
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


class hybridization_shifted(om.ExplicitComponent):
    def __init__(self,
            N_limit,
            life_y,
            N_time,
            life_h,):
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
        super().__init__()
        self.N_limit = N_limit
        self.life_y = life_y
        self.life_h = life_h
        self.N_time = N_time

    def setup(self):
        self.add_input('delta_life',
            desc="Years between the starting of operations of the existing plant and the new plant",
            val=1)
        self.add_input(
            'SoH',
            desc="Battery state of health at discretization levels",
            shape=[self.life_h])

        # -------------------------------------------------------

        self.add_output(
            'SoH_shifted',
            desc="Battery state of health at discretization levels shifted of delta_life",
            shape=[self.life_h])

    def compute(self, inputs, outputs):

        N_limit = self.N_limit
        life_y = self.life_y
        # life_h = self.life_h

        SoH = inputs['SoH']
        delta_life = int(inputs['delta_life'][0])

        outputs['SoH_shifted'] = np.concatenate((np.zeros(delta_life * 365 * 24), SoH[0:life_y * 365 * 24], np.zeros((N_limit-delta_life) * 365 * 24)))

def sample_mean(outs):
    return np.mean(outs, axis=0)
