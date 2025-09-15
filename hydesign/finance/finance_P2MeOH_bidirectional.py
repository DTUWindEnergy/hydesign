import glob
import os
import time

# basic libraries
import numpy as np
import numpy_financial as npf

# import seaborn as sns
import openmdao.api as om
import pandas as pd
import scipy as sp
import yaml
from numpy import newaxis as na

from hydesign.finance.finance import (
    calculate_CAPEX_phasing,
    calculate_NPV_IRR,
    calculate_WACC,
    get_inflation_index,
)
from hydesign.openmdao_wrapper import ComponentWrapper, OpenMDAOModel2Python


class finance_P2MeOH_bidirectional(OpenMDAOModel2Python):
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant with P2MeOH.
    It considers different weighted average costs of capital (WACC) for wind, PV, battery and P2MeOH. The model calculates
    the yearly cashflow as a function of the average revenue over the year, the tax rate and WACC after tax
    ( = weighted sum of the wind, solar, battery, P2MeOH and electrical infrastracture WACC). Net present value (NPV)
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
        life_h=25 * 365 * 24,
    ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """
        self.N_time = int(N_time)
        self.life_h = int(life_h)

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
        super().__init__()

    def setup(self):

        self.add_input(
            "P_HPP_t", desc="HPP power time series", units="MW", shape=[self.life_h]
        )

        self.add_input(
            "P_green_reactor_t",
            desc="Reactor green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_green_DAC_t",
            desc="DAC green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_green_heater_t",
            desc="Heater green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_SOEC_green_t",
            desc="SOEC green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_grid_reactor_t",
            desc="Reactor green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_grid_DAC_t",
            desc="DAC green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_grid_heater_t",
            desc="Heater green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "P_SOEC_grid_t",
            desc="SOEC green power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "penalty_t",
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h],
        )

        self.add_input(
            "P_curtailment_t",
            desc="HPP curtailed power time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "elec_spot_price_t_ext",
            desc="Electricity price time series",
            shape=[self.life_h],
        )

        self.add_input(
            "elec_grid_price_t_ext",
            desc="Grid electricity purchase price time series",
            shape=[self.life_h],
        )

        self.add_input("price_green_MeOH", desc="Price of green methanol")

        self.add_input("price_grid_MeOH", desc="Price of grid methanol")

        self.add_input(
            "P_purch_grid_t",
            desc="Power purchased from the grid time series",
            units="MW",
            shape=[self.life_h],
        )

        self.add_input(
            "m_grid_MeOH_reactor_t",
            desc="Grid MeOH production time series",
            units="kg",
            shape=[self.life_h],
        )

        self.add_input(
            "m_green_MeOH_dist_t",
            desc="Green MeOH distribution time series",
            units="kg",
            shape=[self.life_h],
        )

        self.add_input(
            "m_MeOH_demand_t_ext",
            desc="Methanol demand times series",
            units="kg",
            shape=[self.life_h],
        )

        self.add_input("CAPEX_w", desc="CAPEX wpp")
        self.add_input("OPEX_w", desc="OPEX wpp")

        self.add_input("CAPEX_s", desc="CAPEX solar pvp")
        self.add_input("OPEX_s", desc="OPEX solar pvp")

        self.add_input("CAPEX_b", desc="CAPEX battery")
        self.add_input("OPEX_b", desc="OPEX battery")

        self.add_input("CAPEX_el", desc="CAPEX electrical infrastructure")
        self.add_input("OPEX_el", desc="OPEX electrical infrastructure")

        self.add_input("CAPEX_P2MeOH", desc="CAPEX P2MeOH")
        self.add_input("OPEX_SOEC", desc="OPEX SOEC")
        self.add_input(
            "water_consumption_cost",
            desc="Annual water consumption and treatment cost",
        )
        # self.add_input('OPEX_heater',
        #                 desc = "OPEX heater")
        self.add_input("OPEX_DAC", desc="OPEX DAC")
        self.add_input("OPEX_reactor", desc="OPEX reactor")
        self.add_input("OPEX_MeOH_tank", desc="OPEX MeOH_tank")

        self.add_input("wind_WACC", desc="After tax WACC for onshore WT")

        self.add_input("solar_WACC", desc="After tax WACC for solar PV")

        self.add_input(
            "battery_WACC", desc="After tax WACC for stationary storge li-ion batteries"
        )

        self.add_input(
            "P2MeOH_WACC", desc="After tax WACC for the P2MeOH part of the plant"
        )

        self.add_input("tax_rate", desc="Corporate tax rate")

        ##### OUTPUTs ######

        self.add_output("CAPEX", desc="CAPEX")

        self.add_output("OPEX", desc="OPEX")

        self.add_output("NPV", desc="NPV")

        self.add_output("IRR", desc="IRR")

        self.add_output("NPV_over_CAPEX", desc="NPV/CAPEX")

        self.add_output("mean_Power2Grid", desc="Power to grid")

        self.add_output("mean_AEP", desc="mean AEP")

        self.add_output("annual_green_MeOH", desc="Annual green MeOH production")

        self.add_output("annual_grid_MeOH", desc="Annual grid MeOH production")

        self.add_output("LCOE", desc="LCOE")

        self.add_output("LCOgreenMeOH", desc="LCOgreenMeOH")

        self.add_output("LCOgridMeOH", desc="LCOgridMeOH")

        self.add_output("Revenue", desc="Revenue")

        self.add_output("annual_P_purch_grid", desc="annual_P_purch_grid")

        self.add_output("annual_P_SOEC_green", desc="annual_P_SOEC_green")

        self.add_output("penalty_lifetime", desc="penalty_lifetime")

        self.add_output(
            "break_even_green_MeOH_price",
            desc="price of green methanol that results in NPV=0 with the given hybrid power plant configuration and operation",
            val=0,
        )

        self.add_output(
            "break_even_PPA_price",
            desc="PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation",
            val=0,
        )

    # def setup_partials(self):
    #     self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        elec_spot_price_t_ext : Electricity price time series [Eur]
        elec_grid_price_t_ext : Grid electricity purchase price time series
        P_HPP_t: HPP power time series [MW]
        P_SOEC_green_t: SOEC green power consumption time series [MW]
        P_green_heater_t: Heater green power consumption time series [MW]
        P_green_DAC_t: DAC green power consumption time series [MW]
        P_green_reactor_t: Reactor green power consumption time series [MW]
        P_SOEC_grid_t: SOEC green power consumption time series [MW]
        P_grid_heater_t: Heater green power consumption time series [MW]
        P_grid_DAC_t: DAC green power consumption time series [MW]
        P_grid_reactor_t: Reactor green power consumption time series [MW]
        penalty_t : penalty for not reaching expected energy productin at peak hours [Eur]
        P_curtailment_t : HPP curtailed power time series
        m_grid_MeOH_reactor_t: Grid MeOH production time series
        m_green_MeOH_dist_t: Green MeOH distribution time series
        m_MeOH_demand_t_ext: Hydrogen demand times series
        P_purch_grid_t: Electrolyzer power consumption time series
        price_green_MeOH : green MeOH price
        price_grid_MeOH : grid MeOH price
        CAPEX_w : CAPEX of the wind power plant
        OPEX_w : OPEX of the wind power plant
        CAPEX_s : CAPEX of the solar power plant
        OPEX_s : OPEX of solar power plant
        CAPEX_b : CAPEX of the battery
        OPEX_b : OPEX of the battery
        CAPEX_P2MeOH : CAPEX of P2MeOH part of the plant
        OPEX_SOEC : OPEX of SOEC
        OPEX_heater : OPEX of heater
        OPEX_DAC : OPEX of DAC
        OPEX_reactor : OPEX of reactor
        OPEX_MeOH_tank : OPEX of methanol tank
        CAPEX_el :  CAPEX of the shared electrical infrastracture
        OPEX_el : OPEX of the shared electrical infrastracture
        water_consumption_cost : Annual water consumption and treatment cost
        wind_WACC : After tax WACC for onshore WT
        solar_WACC : After tax WACC for solar PV
        battery_WACC: After tax WACC for stationary storge li-ion batteries
        P2MeOH_WACC : After tax WACC for the P2MeOH part of the plant
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
        annual_MeOH_green: Annual MeOH green distribution
        annual_MeOH_grid: Annual MeOH grid production
        LCOE : Levelized cost of energy
        LCOgreenMeOH : Levelized cost of green Methanol
        LCOgridMeOH : Levelized cost of grid Methanol
        Revenue: revenue of the HPP owner
        penalty_lifetime : total penalty
        annual_P_purch_grid: Mean annual power purchased from the grid
        break_even_green_MeOH_price: Price of green methanol that results in NPV=0 with the given hybrid power plant configuration and operation
        break_even_grid_MeOH_price: Price of grid methanol that results in NPV=0 with the given hybrid power plant configuration and operation
        break_even_PPA_price: PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation
        """

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

        df["P_HPP_t"] = inputs["P_HPP_t"]
        df["m_grid_MeOH_reactor_t"] = inputs["m_grid_MeOH_reactor_t"]
        df["m_green_MeOH_dist_t"] = inputs["m_green_MeOH_dist_t"]
        df["P_purch_grid_t"] = inputs["P_purch_grid_t"]
        df["m_MeOH_demand_t_ext"] = inputs["m_MeOH_demand_t_ext"]
        df["P_SOEC_green_t"] = inputs["P_SOEC_green_t"]
        df["P_green_heater_t"] = inputs["P_green_heater_t"]
        df["P_green_DAC_t"] = inputs["P_green_DAC_t"]
        df["P_green_reactor_t"] = inputs["P_green_reactor_t"]
        df["P_SOEC_grid_t"] = inputs["P_SOEC_grid_t"]
        df["P_grid_heater_t"] = inputs["P_grid_heater_t"]
        df["P_grid_DAC_t"] = inputs["P_grid_DAC_t"]
        df["P_grid_reactor_t"] = inputs["P_grid_reactor_t"]
        df["P_curtailment_t"] = inputs["P_curtailment_t"]
        price_green_MeOH = inputs["price_green_MeOH"]
        price_grid_MeOH = inputs["price_grid_MeOH"]
        elec_spot_price_t_ext = inputs["elec_spot_price_t_ext"]
        elec_grid_price_t_ext = inputs["elec_grid_price_t_ext"]
        df["penalty_t"] = inputs["penalty_t"]

        df["i_year"] = np.hstack(
            [np.array([ii] * N_time) for ii in range(int(np.ceil(life_h / N_time)))]
        )[:life_h]

        revenues = calculate_revenues_P2MeOH_bidirectional(
            price_green_MeOH,
            price_grid_MeOH,
            elec_spot_price_t_ext,
            elec_grid_price_t_ext,
            df,
        )
        CAPEX = (
            inputs["CAPEX_w"]
            + inputs["CAPEX_s"]
            + inputs["CAPEX_b"]
            + inputs["CAPEX_el"]
            + inputs["CAPEX_P2MeOH"]
        )
        OPEX = (
            inputs["OPEX_w"]
            + inputs["OPEX_s"]
            + inputs["OPEX_b"]
            + inputs["OPEX_el"]
            + inputs["OPEX_SOEC"]
            + inputs["OPEX_DAC"]
            + inputs["OPEX_reactor"]
            + inputs["OPEX_MeOH_tank"]
            + inputs["water_consumption_cost"]
        )
        # + inputs['OPEX_heater']

        CAPEX_LCOE = (
            inputs["CAPEX_w"]
            + inputs["CAPEX_s"]
            + inputs["CAPEX_b"]
            + inputs["CAPEX_el"]
        )
        OPEX_LCOE = (
            inputs["OPEX_w"] + inputs["OPEX_s"] + inputs["OPEX_b"] + inputs["OPEX_el"]
        )
        # + inputs['OPEX_heater']

        outputs["CAPEX"] = CAPEX
        outputs["OPEX"] = OPEX

        # Discount rates
        WACC_after_tax = calculate_WACC_P2MeOH(
            inputs["CAPEX_w"],
            inputs["CAPEX_s"],
            inputs["CAPEX_b"],
            inputs["CAPEX_el"],
            inputs["CAPEX_P2MeOH"],
            inputs["wind_WACC"],
            inputs["solar_WACC"],
            inputs["battery_WACC"],
            inputs["P2MeOH_WACC"],
        )
        WACC_after_tax_LCOE = calculate_WACC(
            inputs["CAPEX_w"],
            inputs["CAPEX_s"],
            inputs["CAPEX_b"],
            inputs["CAPEX_el"],
            inputs["wind_WACC"],
            inputs["solar_WACC"],
            inputs["battery_WACC"],
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
            tax_rate=inputs["tax_rate"],
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
        AEP_per_year = (
            df.groupby("i_year").P_HPP_t.mean() * 365 * 24
            + df.groupby("i_year").P_SOEC_green_t.mean() * 365 * 24
            + df.groupby("i_year").P_green_reactor_t.mean() * 365 * 24
            + df.groupby("i_year").P_green_heater_t.mean() * 365 * 24
            + df.groupby("i_year").P_green_DAC_t.mean() * 365 * 24
        )
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor_LCOE) ** iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        Power2Grid_per_year = df.groupby("i_year").P_HPP_t.mean() * 365 * 24
        mean_Power2Grid_per_year = np.mean(Power2Grid_per_year)

        if level_AEP > 0:
            LCOE = level_costs / (level_AEP)  # in Euro/MWh
        else:
            LCOE = 1e6
        outputs["LCOE"] = LCOE

        P_SOEC_green_per_year = df.groupby("i_year").P_SOEC_green_t.mean() * 365 * 24
        mean_P_SOEC_green_per_year = np.mean(P_SOEC_green_per_year)

        P_purch_grid_per_year = df.groupby("i_year").P_purch_grid_t.mean() * 365 * 24
        mean_P_purch_grid_per_year = np.mean(P_purch_grid_per_year)

        # LCOgreenMeOH calculation using LCOE
        OPEX_ptgreenMeOH = (
            inputs["OPEX_SOEC"]
            + inputs["OPEX_DAC"]
            + inputs["OPEX_reactor"]
            + inputs["water_consumption_cost"]
        )
        # + inputs['OPEX_heater']
        CAPEX_ptgreenMeOH = inputs["CAPEX_P2MeOH"]
        hpp_discount_factor_MeOH = inputs["P2MeOH_WACC"]
        OPEX_ptgreenMeOH_el = LCOE * (
            np.sum(inputs["P_SOEC_green_t"])
            + np.sum(inputs["P_green_heater_t"])
            + np.sum(inputs["P_green_DAC_t"])
            + np.sum(inputs["P_green_reactor_t"])
        )  # operational cost for the electrilcity consumed to produce green Methanol
        level_costs_greenMeOH = (
            np.sum(OPEX_ptgreenMeOH / (1 + hpp_discount_factor_MeOH) ** iy)
            + OPEX_ptgreenMeOH_el
            + CAPEX_ptgreenMeOH
        )
        AgreenMeOHP_per_year = (
            df.groupby("i_year").m_green_MeOH_dist_t.mean() * 365 * 24
        )
        level_AgreenMeOHP = np.sum(
            AgreenMeOHP_per_year / (1 + hpp_discount_factor_MeOH) ** iy
        )

        mean_AgreenMeOHP_per_year = np.mean(AgreenMeOHP_per_year)
        if level_AgreenMeOHP > 0:
            outputs["LCOgreenMeOH"] = level_costs_greenMeOH / (
                level_AgreenMeOHP
            )  # in Euro/kg
        else:
            outputs["LCOgreenMeOH"] = 1e6

        # LCOgridMeOH calculation using LCOE
        OPEX_ptgridMeOH = (
            inputs["OPEX_SOEC"]
            + inputs["OPEX_DAC"]
            + inputs["OPEX_reactor"]
            + inputs["water_consumption_cost"]
        )
        # + inputs['OPEX_heater']
        CAPEX_ptgridMeOH = inputs["CAPEX_P2MeOH"]
        OPEX_ptgridMeOH_el = elec_grid_price_t_ext * (
            np.sum(inputs["P_SOEC_grid_t"])
            + np.sum(inputs["P_grid_heater_t"])
            + np.sum(inputs["P_grid_DAC_t"])
            + np.sum(inputs["P_grid_reactor_t"])
        )  # operational cost for the electrilcity consumed to produce green Methanol
        level_costs_gridMeOH = (
            np.sum(OPEX_ptgridMeOH / (1 + hpp_discount_factor_MeOH) ** iy)
            + OPEX_ptgridMeOH_el
            + CAPEX_ptgridMeOH
        )
        AgridMeOHP_per_year = (
            df.groupby("i_year").m_grid_MeOH_reactor_t.mean() * 365 * 24
        )
        level_AgridMeOHP = np.sum(
            AgridMeOHP_per_year / (1 + hpp_discount_factor_MeOH) ** iy
        )

        mean_AgridMeOHP_per_year = np.mean(AgridMeOHP_per_year)
        if level_AgridMeOHP > 0:
            outputs["LCOgridMeOH"] = np.mean(
                level_costs_gridMeOH / (level_AgridMeOHP)
            )  # in Euro/kg
        else:
            outputs["LCOgridMeOH"] = 1e6

        break_even_green_MeOH_price = np.maximum(
            0,
            calculate_break_even_green_MeOH_price_bidirectional(
                df=df,
                CAPEX=CAPEX_eq,
                OPEX=OPEX,
                tax_rate=inputs["tax_rate"],
                discount_rate=WACC_after_tax,
                elec_spot_price_t_ext=elec_spot_price_t_ext,
                elec_grid_price_t_ext=elec_grid_price_t_ext,
                depreciation_yr=depreciation_yr,
                depreciation=depreciation,
                DEVEX=DEVEX,
                inflation_index=inflation_index,
                price_grid_MeOH=price_grid_MeOH,
            ),
        )

        break_even_PPA_price = np.maximum(
            0,
            calculate_break_even_PPA_price_P2MeOH_bidirectional(
                df=df,
                CAPEX=CAPEX_eq,
                OPEX=OPEX,
                tax_rate=inputs["tax_rate"],
                discount_rate=WACC_after_tax,
                price_green_MeOH=price_green_MeOH,
                price_grid_MeOH=price_grid_MeOH,
                elec_grid_price_t_ext=elec_grid_price_t_ext,
                depreciation_yr=depreciation_yr,
                depreciation=depreciation,
                DEVEX=DEVEX,
                inflation_index=inflation_index,
            ),
        )

        outputs["Revenue"] = np.sum(revenues.values.flatten())
        outputs["annual_P_SOEC_green"] = mean_P_SOEC_green_per_year
        outputs["annual_P_purch_grid"] = mean_P_purch_grid_per_year
        outputs["mean_AEP"] = mean_AEP_per_year
        outputs["mean_Power2Grid"] = mean_Power2Grid_per_year
        outputs["annual_green_MeOH"] = mean_AgreenMeOHP_per_year
        outputs["annual_grid_MeOH"] = mean_AgridMeOHP_per_year
        outputs["penalty_lifetime"] = df["penalty_t"].sum()
        outputs["break_even_green_MeOH_price"] = break_even_green_MeOH_price
        outputs["break_even_PPA_price"] = break_even_PPA_price
        return outputs


class finance_P2MeOH_bidirectional_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = finance_P2MeOH_bidirectional(**insta_inp)
        super().__init__(
            inputs=model.inputs,
            outputs=model.outputs,
            function=model.run,
            partial_options=[{"dependent": False, "val": 0}],
        )


# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------


def calculate_WACC_P2MeOH(
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_el,
    CAPEX_P2MeOH,
    wind_WACC,
    solar_WACC,
    battery_WACC,
    P2MeOH_WACC,
):
    """This function returns the weighted average cost of capital after tax, using solar, wind, electrolyzer, heater, DAC,
    reactor, MeOH tank and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.
    """

    # Weighted average cost of capital
    WACC_after_tax = (
        CAPEX_w * wind_WACC
        + CAPEX_s * solar_WACC
        + CAPEX_b * battery_WACC
        + CAPEX_P2MeOH * P2MeOH_WACC
        + CAPEX_el * (wind_WACC + solar_WACC + battery_WACC + P2MeOH_WACC) / 4
    ) / (CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el + CAPEX_P2MeOH)
    return WACC_after_tax


def calculate_revenues_P2MeOH_bidirectional(
    price_green_MeOH, price_grid_MeOH, elec_spot_price_t_ext, elec_grid_price_t_ext, df
):
    df["revenue"] = (
        df["P_HPP_t"] * np.broadcast_to(elec_spot_price_t_ext, df["P_HPP_t"].shape)
        + df["m_green_MeOH_dist_t"] * price_green_MeOH
        + df["m_grid_MeOH_reactor_t"] * price_grid_MeOH
        - df["P_purch_grid_t"]
        * np.broadcast_to(elec_grid_price_t_ext, df["P_HPP_t"].shape)
    )
    return df.groupby("i_year").revenue.mean() * 365 * 24


def calculate_break_even_PPA_price_P2MeOH_bidirectional(
    df,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    price_green_MeOH,
    price_grid_MeOH,
    elec_grid_price_t_ext,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
):
    def fun(elec_spot_price_t_ext):
        revenues = calculate_revenues_P2MeOH_bidirectional(
            price_green_MeOH,
            price_grid_MeOH,
            elec_spot_price_t_ext,
            elec_grid_price_t_ext,
            df,
        )
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


def calculate_break_even_green_MeOH_price_bidirectional(
    df,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    elec_spot_price_t_ext,
    elec_grid_price_t_ext,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
    price_grid_MeOH,
):
    def fun(price_green_MeOH):
        revenues = calculate_revenues_P2MeOH_bidirectional(
            price_green_MeOH,
            price_grid_MeOH,
            elec_spot_price_t_ext,
            elec_grid_price_t_ext,
            df,
        )
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
