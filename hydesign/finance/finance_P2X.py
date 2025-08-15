# import glob
# import os
# import time

# basic libraries
import numpy as np

# from numpy import newaxis as na
# import numpy_financial as npf
import pandas as pd

# import seaborn as sns
# import openmdao.api as om
# import yaml
import scipy as sp

from hydesign.finance.finance import (
    calculate_CAPEX_phasing,
    calculate_NPV_IRR,
    calculate_WACC,
    get_inflation_index,
)
from hydesign.openmdao_wrapper import ComponentWrapper


class finance_P2X:
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant with P2X.
    It considers different weighted average costs of capital (WACC) for wind, PV, battery and P2X. The model calculates
    the yearly cashflow as a function of the average revenue over the year, the tax rate and WACC after tax
    ( = weighted sum of the wind, solar, battery, P2X and electrical infrastracture WACC). Net present value (NPV)
    and levelized cost of energy (LCOE) is then be calculated using the calculates WACC as the discount rate, as well
    as the internal rate of return (IRR).
    """

    def __init__(
        self,
        N_time,
        # Depreciation curve
        depreciation_yr,
        depreciation,
        # Inflation curve
        inflation_yr,
        inflation,
        ref_yr_inflation,
        # Early paying or CAPEX Phasing
        phasing_yr,
        phasing_CAPEX,
        life_y=25,
        price_H2=None,
        penalty_factor_H2=None,
        wind_WACC=None,
        solar_WACC=None,
        battery_WACC=None,
        ptg_WACC=None,
        tax_rate=None,
    ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """
        # super().__init__()
        self.N_time = int(N_time)
        self.life_h = int(life_y * 365 * 24)

        # Depreciation curve
        self.depreciation_yr = depreciation_yr
        self.depreciation = depreciation

        # Inflation curve
        self.inflation_yr = inflation_yr
        self.inflation = inflation
        self.ref_yr_inflation = ref_yr_inflation

        # Early paying or CAPEX Phasing
        self.phasing_yr = phasing_yr
        self.phasing_CAPEX = phasing_CAPEX

        self.price_H2 = price_H2
        self.penalty_factor_H2 = penalty_factor_H2
        self.wind_WACC = wind_WACC
        self.solar_WACC = solar_WACC
        self.battery_WACC = battery_WACC
        self.ptg_WACC = ptg_WACC
        self.tax_rate = tax_rate

        # def setup(self):
        self.inputs = [
            (
                "price_t_ext",
                dict(desc="Electricity price time series", shape=[self.life_h]),
            ),
            (
                "hpp_t",
                dict(desc="HPP power time series", units="MW", shape=[self.life_h]),
            ),
            (
                "penalty_t",
                dict(
                    desc="penalty for not reaching expected energy production at peak hours",
                    shape=[self.life_h],
                ),
            ),
            (
                "hpp_curt_t",
                dict(
                    desc="HPP curtailed power time series",
                    units="MW",
                    shape=[self.life_h],
                ),
            ),
            ("m_H2_t", dict(desc="Produced Hydrogen", units="kg", shape=[self.life_h])),
            (
                "m_H2_offtake_t",
                dict(desc="Produced Hydrogen", units="kg", shape=[self.life_h]),
            ),
            (
                "m_H2_demand_t_ext",
                dict(
                    desc="Hydrogen demand times series", units="kg", shape=[self.life_h]
                ),
            ),
            (
                "P_ptg_t",
                dict(
                    desc="Electrolyzer power consumption time series",
                    units="MW",
                    shape=[self.life_h],
                ),
            ),
            ("CAPEX_w", dict(desc="CAPEX wpp")),
            ("OPEX_w", dict(desc="OPEX wpp")),
            ("CAPEX_s", dict(desc="CAPEX solar pvp")),
            ("OPEX_s", dict(desc="OPEX solar pvp")),
            ("CAPEX_b", dict(desc="CAPEX battery")),
            ("OPEX_b", dict(desc="OPEX battery")),
            ("CAPEX_el", dict(desc="CAPEX electrical infrastructure")),
            ("OPEX_el", dict(desc="OPEX electrical infrastructure")),
            ("CAPEX_ptg", dict(desc="CAPEX ptg plant")),
            ("OPEX_ptg", dict(desc="OPEX ptg plant")),
            (
                "water_consumption_cost",
                dict(desc="Water usage and purification for the electrolysis"),
            ),
        ]
        self.outputs = [
            ("CAPEX", dict(desc="CAPEX")),
            ("OPEX", dict(desc="OPEX")),
            ("NPV", dict(desc="NPV")),
            ("IRR", dict(desc="IRR")),
            ("NPV_over_CAPEX", dict(desc="NPV/CAPEX")),
            ("mean_AEP", dict(desc="mean AEP")),
            ("annual_H2", dict(desc="Annual H2 production")),
            ("LCOE", dict(desc="LCOE")),
            ("LCOH", dict(desc="LCOH")),
            ("Revenue", dict(desc="Revenue")),
            ("annual_P_ptg", dict(desc="annual_P_ptg")),
            ("penalty_lifetime", dict(desc="penalty_lifetime")),
            (
                "break_even_H2_price",
                dict(
                    desc="price of hydrogen that results in NPV=0 with the given hybrid power plant configuration and operation",
                    val=0,
                ),
            ),
            (
                "break_even_PPA_price",
                dict(
                    desc="PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation",
                    val=0,
                ),
            ),
        ]

    # def setup_partials(self):
    #     self.declare_partials('*', '*', method='fd')

    def compute(self, **inputs):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        price_t_ext : Electricity price time series [Eur]
        hpp_t_with_deg : HPP power time series [MW]
        penalty_t : penalty for not reaching expected energy productin at peak hours [Eur]
        hpp_curt_t : HPP curtailed power time series
        m_H2_t: Produced Hydrogen
        m_H2_offtake_t: Hydrogen offtake time series
        m_H2_demand_t_ext: Hydrogen demand times series
        penalty_factor_H2: Penalty for not meeting hydrogen demand in an hour
        P_ptg_t: Electrolyzer power consumption time series
        price_H2: H2 price
        CAPEX_w : CAPEX of the wind power plant
        OPEX_w : OPEX of the wind power plant
        CAPEX_s : CAPEX of the solar power plant
        OPEX_s : OPEX of solar power plant
        CAPEX_b : CAPEX of the battery
        OPEX_b : OPEX of the battery
        CAPEX_ptg : CAPEX of P2G plant
        OPEX_ptg : OPEX of P2G plant
        CAPEX_sh :  CAPEX of the shared electrical infrastracture
        OPEX_sh : OPEX of the shared electrical infrastracture
        wind_WACC : After tax WACC for onshore WT
        solar_WACC : After tax WACC for solar PV
        battery_WACC: After tax WACC for stationary storge li-ion batteries
        ptg_WACC: After tax WACC for power to gas plant
        tax_rate : Corporate tax rate

        Returns
        -------
        CAPEX : Total capital expenditure costs of the HPP
        OPEX : Operational and maintenance costs of the HPP
        NPV : Net present value
        IRR : Internal rate of return
        NPV_over_CAPEX : NPV over CAPEX
        mean_AEP : Mean annual energy production
        mean_Power2Grid: Mean annual power to grid
        annual_H2: Annual H2 production
        LCOE : Levelized cost of energy
        LCOH : Levelized cost of hydrogen
        Revenue: revenue of the HPP owner
        penalty_lifetime : total penalty
        annual_P_ptg: Mean annual power to electrolyzer to produce hydrogen
        """
        outputs = {}
        N_time = self.N_time
        life_h = self.life_h

        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation

        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation

        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX

        df = pd.DataFrame()

        df["hpp_t"] = inputs["hpp_t"]
        df["m_H2_t"] = inputs["m_H2_t"]
        df["m_H2_offtake_t"] = inputs["m_H2_offtake_t"]
        df["m_H2_demand_t_ext"] = inputs["m_H2_demand_t_ext"]
        df["P_ptg_t"] = inputs["P_ptg_t"]
        df["hpp_curt_t"] = inputs["hpp_curt_t"]
        price_H2 = self.price_H2
        penalty_factor_H2 = self.penalty_factor_H2
        price_t = inputs["price_t_ext"]
        df["penalty_t"] = inputs["penalty_t"]

        df["i_year"] = np.hstack(
            [np.array([ii] * N_time) for ii in range(int(np.ceil(life_h / N_time)))]
        )[:life_h]

        # Compute yearly revenues and cashflow
        revenues = calculate_revenues_P2X(price_H2, price_t, penalty_factor_H2, df)
        CAPEX = (
            inputs["CAPEX_w"]
            + inputs["CAPEX_s"]
            + inputs["CAPEX_b"]
            + inputs["CAPEX_el"]
            + inputs["CAPEX_ptg"]
        )
        OPEX = (
            inputs["OPEX_w"]
            + inputs["OPEX_s"]
            + inputs["OPEX_b"]
            + inputs["OPEX_el"]
            + inputs["OPEX_ptg"]
            + inputs["water_consumption_cost"]
        )

        CAPEX_LCOE = (
            inputs["CAPEX_w"]
            + inputs["CAPEX_s"]
            + inputs["CAPEX_b"]
            + inputs["CAPEX_el"]
        )
        OPEX_LCOE = (
            inputs["OPEX_w"] + inputs["OPEX_s"] + inputs["OPEX_b"] + inputs["OPEX_el"]
        )

        outputs["CAPEX"] = CAPEX
        outputs["OPEX"] = OPEX

        # Discount rates
        WACC_after_tax = calculate_WACC_P2X(
            inputs["CAPEX_w"],
            inputs["CAPEX_s"],
            inputs["CAPEX_b"],
            inputs["CAPEX_el"],
            inputs["CAPEX_ptg"],
            self.wind_WACC,
            self.solar_WACC,
            self.battery_WACC,
            self.ptg_WACC,
        )
        WACC_after_tax_LCOE = calculate_WACC(
            inputs["CAPEX_w"],
            inputs["CAPEX_s"],
            inputs["CAPEX_b"],
            inputs["CAPEX_el"],
            self.wind_WACC,
            self.solar_WACC,
            self.battery_WACC,
        )

        # Apply CAPEX phasing using the inflation index for all years before the start of the project (t=0).
        inflation_index_phasing = get_inflation_index(
            yr=phasing_yr,
            inflation_yr=inflation_yr,
            inflation=inflation,
            ref_yr_inflation=ref_yr_inflation,
        )
        CAPEX_eq = calculate_CAPEX_phasing(
            CAPEX=CAPEX,
            phasing_yr=phasing_yr,
            phasing_CAPEX=phasing_CAPEX,
            discount_rate=WACC_after_tax,
            inflation_index=inflation_index_phasing,
        )

        # len of revenues = years of life
        iy = (
            np.arange(len(revenues)) + 1
        )  # Plus becasue the year zero is added externally in the NPV and IRR calculations

        # Compute inflation, all cahsflow are in nominal prices
        inflation_index = get_inflation_index(
            yr=np.arange(
                len(revenues) + 1
            ),  # It includes t=0, to compute the reference
            inflation_yr=inflation_yr,
            inflation=inflation,
            ref_yr_inflation=ref_yr_inflation,
        )

        # We need to add Development costs
        DEVEX = 0

        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t=revenues.values.flatten(),
            investment_cost=CAPEX_eq,  # Include phasing
            maintenance_cost_per_year=OPEX,
            tax_rate=self.tax_rate,
            discount_rate=WACC_after_tax,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )

        outputs["NPV"] = NPV
        outputs["IRR"] = IRR
        outputs["NPV_over_CAPEX"] = NPV / CAPEX

        # LCOE calculation
        hpp_discount_factor_LCOE = WACC_after_tax_LCOE
        level_costs = (
            np.sum(OPEX_LCOE / (1 + hpp_discount_factor_LCOE) ** iy) + CAPEX_LCOE
        )
        AEP_per_year = df.groupby("i_year").hpp_t.mean() * 365 * 24
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor_LCOE) ** iy)
        mean_AEP_per_year = np.mean(AEP_per_year)

        P_ptg_per_year = df.groupby("i_year").P_ptg_t.mean() * 365 * 24
        mean_P_ptg_per_year = np.mean(P_ptg_per_year)
        level_P_ptg = np.sum(P_ptg_per_year / (1 + hpp_discount_factor_LCOE) ** iy)

        # Power2Grid_per_year = df.groupby('i_year').hpp_t.mean()*365*24
        # mean_Power2Grid_per_year = np.mean(Power2Grid_per_year)
        level_energy = level_AEP + level_P_ptg
        if level_energy > 0:
            LCOE = level_costs / (level_energy)  # in Euro/MWh
        else:
            LCOE = 1e6
        outputs["LCOE"] = LCOE

        # LCOH calculation using LCOE
        OPEX_ptg = inputs["OPEX_ptg"] + inputs["water_consumption_cost"]
        CAPEX_ptg = inputs["CAPEX_ptg"]
        hpp_discount_factor_H2 = self.ptg_WACC
        OPEX_ptg_el = LCOE * np.sum(
            inputs["P_ptg_t"]
        )  # operational cost for the electrilcity consumed to produce hydrogen
        level_costs_H2 = (
            np.sum(OPEX_ptg / (1 + hpp_discount_factor_H2) ** iy)
            + OPEX_ptg_el
            + CAPEX_ptg
        )
        AHP_per_year = df.groupby("i_year").m_H2_t.mean() * 365 * 24
        level_AHP = np.sum(AHP_per_year / (1 + hpp_discount_factor_H2) ** iy)

        mean_AHP_per_year = np.mean(AHP_per_year)
        if level_AHP > 0:
            outputs["LCOH"] = level_costs_H2 / (level_AHP)  # in Euro/kg
        else:
            outputs["LCOH"] = 1e6

        break_even_H2_price = calculate_break_even_H2_price(
            penalty_factor_H2=penalty_factor_H2,
            df=df,
            CAPEX=CAPEX_eq,
            OPEX=OPEX,
            tax_rate=self.tax_rate,
            discount_rate=WACC_after_tax,
            price_el=price_t,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            DEVEX=DEVEX,
            inflation_index=inflation_index,
        )

        break_even_PPA_price = np.maximum(
            0,
            calculate_break_even_PPA_price_P2X(
                penalty_factor_H2=penalty_factor_H2,
                df=df,
                CAPEX=CAPEX_eq,
                OPEX=OPEX,
                tax_rate=self.tax_rate,
                discount_rate=WACC_after_tax,
                price_H2=price_H2,
                depreciation_yr=depreciation_yr,
                depreciation=depreciation,
                DEVEX=DEVEX,
                inflation_index=inflation_index,
            ),
        )

        outputs["Revenue"] = np.sum(revenues.values.flatten())
        outputs["annual_P_ptg"] = mean_P_ptg_per_year
        outputs["mean_AEP"] = mean_AEP_per_year
        # outputs['mean_Power2Grid'] = mean_Power2Grid_per_year
        outputs["annual_H2"] = mean_AHP_per_year
        outputs["penalty_lifetime"] = df["penalty_t"].sum()
        outputs["break_even_H2_price"] = break_even_H2_price
        outputs["break_even_PPA_price"] = break_even_PPA_price
        out_keys = [
            "CAPEX",
            "OPEX",
            "NPV",
            "IRR",
            "NPV_over_CAPEX",
            "mean_AEP",
            "annual_H2",
            "LCOE",
            "LCOH",
            "Revenue",
            "annual_P_ptg",
            "penalty_lifetime",
            "break_even_H2_price",
            "break_even_PPA_price",
        ]
        return [outputs[key] for key in out_keys]


class finance_P2X_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = finance_P2X(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------


def calculate_WACC_P2X(
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_el,
    CAPEX_ptg,
    wind_WACC,
    solar_WACC,
    battery_WACC,
    ptg_WACC,
):
    """This function returns the weighted average cost of capital after tax, using solar, wind, electrolyzer and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.

    Parameters
    ----------
    CAPEX_w : CAPEX of the wind power plant
    CAPEX_s : CAPEX of the solar power plant
    CAPEX_b : CAPEX of the battery
    CAPEX_el : CAPEX of the shared electrical costs
    wind_WACC : After tax WACC for onshore WT
    solar_WACC : After tax WACC for solar PV
    battery_WACC : After tax WACC for stationary storge li-ion batteries
    ptg_WACC : After tax WACC for power to gas plant

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Weighted average cost of capital
    WACC_after_tax = (
        CAPEX_w * wind_WACC
        + CAPEX_s * solar_WACC
        + CAPEX_b * battery_WACC
        + CAPEX_ptg * ptg_WACC
        + CAPEX_el * (wind_WACC + solar_WACC + battery_WACC + ptg_WACC) / 4
    ) / (CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el + CAPEX_ptg)
    return WACC_after_tax


def calculate_revenues_P2X(price_H2, price_el, penalty_factor_H2, df):
    df["revenue"] = (
        df["hpp_t"] * np.broadcast_to(price_el, df["hpp_t"].shape)
        + df["m_H2_offtake_t"] * price_H2
        - penalty_factor_H2 * (df["m_H2_demand_t_ext"] - df["m_H2_offtake_t"])
        - df["penalty_t"]
    )
    return df.groupby("i_year").revenue.mean() * 365 * 24


def calculate_break_even_PPA_price_P2X(
    penalty_factor_H2,
    df,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    price_H2,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
):
    def fun(price_el):
        revenues = calculate_revenues_P2X(price_H2, price_el, penalty_factor_H2, df)
        NPV, _ = calculate_NPV_IRR(
            Net_revenue_t=revenues.values.flatten(),
            investment_cost=CAPEX,
            maintenance_cost_per_year=OPEX,
            tax_rate=tax_rate,
            discount_rate=discount_rate,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )
        return NPV**2

    out = sp.optimize.minimize(fun=fun, x0=50, method="SLSQP", tol=1e-10)
    return out.x


def calculate_break_even_H2_price(
    penalty_factor_H2,
    df,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    price_el,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
):
    def fun(price_H2):
        revenues = calculate_revenues_P2X(price_H2, price_el, penalty_factor_H2, df)
        NPV, _ = calculate_NPV_IRR(
            Net_revenue_t=revenues.values.flatten(),
            investment_cost=CAPEX,
            maintenance_cost_per_year=OPEX,
            tax_rate=tax_rate,
            discount_rate=discount_rate,
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            development_cost=DEVEX,
            inflation_index=inflation_index,
        )
        return NPV**2

    out = sp.optimize.minimize(fun=fun, x0=4, method="SLSQP", tol=1e-10)
    return out.x
