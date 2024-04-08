import numpy as np
import pandas as pd
import openmdao.api as om
import scipy as sp

from hydesign.finance.finance import calculate_WACC, get_inflation_index, calculate_CAPEX_phasing, calculate_NPV_IRR

class finance(om.ExplicitComponent):
    """Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant.
    It considers different weighted average costs of capital (WACC) for wind, PV and battery. The model calculates
    the yearly cashflow as a function of the average revenue over the year, the tax rate and WACC after tax
    ( = weighted sum of the wind, solar, battery, and electrical infrastracture WACC). Net present value (NPV)
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
        
        life_h = 25*365*24,
        ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """ 
        super().__init__()
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

    def setup(self):
        self.add_input('price_t_ext',
                       desc="Electricity price time series",
                       shape=[self.life_h])
        
        self.add_input('hpp_t',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('hpp_up_reg_t',
                       desc="HPP up regulation power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('hpp_dwn_reg_t',
                       desc="HPP down regulation power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('price_up_reg_t_ext',
                       desc="Up regulation price time series",
                       shape=[self.life_h])
        
        self.add_input('price_dwn_reg_t_ext',
                       desc="Down regulation price time series",
                       shape=[self.life_h])
        
        self.add_input('penalty_t',
                        desc="penalty for not reaching expected energy productin at peak hours",
                        shape=[self.life_h])

        self.add_input('hpp_t_deg',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('hpp_up_reg_t_deg',
                       desc="HPP up regulation power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('hpp_dwn_reg_t_deg',
                       desc="HPP down regulation power time series",
                       units='MW',
                       shape=[self.life_h])

        self.add_input('penalty_t_deg',
                        desc="penalty for not reaching expected energy productin at peak hours",
                        shape=[self.life_h])
        

        self.add_input('CAPEX_w',
                       desc="CAPEX wpp")
        self.add_input('OPEX_w',
                       desc="OPEX wpp")

        self.add_input('CAPEX_s',
                       desc="CAPEX solar pvp")
        self.add_input('OPEX_s',
                       desc="OPEX solar pvp")

        self.add_input('CAPEX_b',
                       desc="CAPEX battery")
        self.add_input('OPEX_b',
                       desc="OPEX battery")

        self.add_input('CAPEX_el',
                       desc="CAPEX electrical infrastructure")
        self.add_input('OPEX_el',
                       desc="OPEX electrical infrastructure")

        self.add_input('wind_WACC',
                       desc="After tax WACC for onshore WT")
        
        self.add_input('solar_WACC',
                       desc="After tax WACC for solar PV")
        
        self.add_input('battery_WACC',
                       desc="After tax WACC for stationary storge li-ion batteries")
        
        self.add_input('tax_rate',
                       desc="Corporate tax rate")
        
        self.add_output('CAPEX',
                        desc="CAPEX")
        
        self.add_output('revenues_without_deg',
                        desc="Revenue without deg")
        self.add_output('revenues',
                        desc="Revenue with deg")
        
        self.add_output('OPEX',
                        desc="OPEX")
        
        self.add_output('NPV',
                        desc="NPV")
        
        self.add_output('IRR',
                        desc="IRR")
        
        self.add_output('NPV_over_CAPEX',
                        desc="NPV/CAPEX")
        
        self.add_output('mean_AEP',
                        desc="mean AEP")
        
        self.add_output('LCOE',
                        desc="LCOE")
        
        self.add_output('penalty_lifetime',
                        desc="penalty_lifetime")

        self.add_output('break_even_PPA_price',
                        desc='PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation',
                        val=0)

    def setup_partials(self):
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """ Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        price_t_ext : Electricity price time series [Eur]
        hpp_t_with_deg : HPP power time series [MW]
        penalty_t : penalty for not reaching expected energy productin at peak hours [Eur]
        CAPEX_w : CAPEX of the wind power plant
        OPEX_w : OPEX of the wind power plant
        CAPEX_s : CAPEX of the solar power plant
        OPEX_s : OPEX of solar power plant   
        CAPEX_b : CAPEX of the battery
        OPEX_b : OPEX of the battery
        CAPEX_sh :  CAPEX of the shared electrical infrastracture
        OPEX_sh : OPEX of the shared electrical infrastracture
        wind_WACC : After tax WACC for onshore WT
        solar_WACC : After tax WACC for solar PV
        battery_WACC: After tax WACC for stationary storge li-ion batteries
        tax_rate : Corporate tax rate

        Returns
        -------
        CAPEX : Total capital expenditure costs of the HPP
        OPEX : Operational and maintenance costs of the HPP
        Revenue: Total revenue
        NPV : Net present value
        IRR : Internal rate of return
        NPV_over_CAPEX : NPV over CAPEX
        mean_AEP : Mean annual energy production
        LCOE : Levelized cost of energy
        penalty_lifetime : total penalty
        """
        
        N_time = self.N_time
        life_h = self.life_h
        life_yr = int(np.ceil(life_h/N_time))

        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation

        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation

        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX
        
        df = pd.DataFrame()
        
        df['hpp_t'] = inputs['hpp_t']
        df['price_t'] = inputs['price_t_ext']
        df['hpp_up_reg_t'] = inputs['hpp_up_reg_t']
        df['hpp_dwn_reg_t'] = inputs['hpp_dwn_reg_t']
        df['price_up_reg_t'] = inputs['price_up_reg_t_ext']
        df['price_dwn_reg_t'] = inputs['price_dwn_reg_t_ext']
        df['penalty_t'] = inputs['penalty_t']
        df['hpp_up_reg_t_deg'] = inputs['hpp_up_reg_t_deg']
        df['hpp_dwn_reg_t_deg'] = inputs['hpp_dwn_reg_t_deg']
        df['hpp_t_deg'] = inputs['hpp_t_deg']
        df['penalty_t_deg'] = inputs['penalty_t_deg']
        
        df['i_year'] = np.hstack([np.array([ii]*N_time) for ii in range(life_yr)])[:life_h]

        # Compute yearly revenues and cashflow
        revenues_without_deg = calculate_revenues_without_deg(df)
        revenues = calculate_revenues(df)
        CAPEX = inputs['CAPEX_w'] + inputs['CAPEX_s'] + \
            inputs['CAPEX_b'] + inputs['CAPEX_el']
        OPEX = inputs['OPEX_w'] + inputs['OPEX_s'] + \
            inputs['OPEX_b'] + inputs['OPEX_el']

        # Discount rate
        hpp_discount_factor = calculate_WACC(
            inputs['CAPEX_w'],
            inputs['CAPEX_s'],
            inputs['CAPEX_b'],
            inputs['CAPEX_el'],
            inputs['wind_WACC'],
            inputs['solar_WACC'],
            inputs['battery_WACC'],
            )
        
        # Apply CAPEX phasing using the inflation index for all years before the start of the project (t=0). 
        inflation_index_phasing = get_inflation_index(
            yr = phasing_yr,
            inflation_yr = inflation_yr, 
            inflation = inflation,
            ref_yr_inflation = ref_yr_inflation,
        )
        CAPEX_eq = calculate_CAPEX_phasing(
            CAPEX = CAPEX,
            phasing_yr = phasing_yr,
            phasing_CAPEX = phasing_CAPEX,
            discount_rate = hpp_discount_factor,
            inflation_index = inflation_index_phasing,
            )

        # len of revenues = years of life
        iy = np.arange(len(revenues)) + 1 # Plus becasue the year zero is added externally in the NPV and IRR calculations
        
        # Compute inflation, all cahsflow are in nominal prices
        inflation_index = get_inflation_index(
                yr = np.arange(len(revenues)+1), # It includes t=0, to compute the reference
                inflation_yr = inflation_yr, 
                inflation = inflation,
                ref_yr_inflation = ref_yr_inflation,
        )
        
        revenues_without_deg = revenues_without_deg.values.flatten()
        revenues = revenues.values.flatten()
        outputs['CAPEX'] = CAPEX
        outputs['OPEX'] = OPEX
        outputs['revenues_without_deg'] = revenues_without_deg.mean()
        outputs['revenues'] = revenues.mean()

        # We need to add DEVEX
        DEVEX = 0

        # Calculate the 
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t = revenues,
            investment_cost = CAPEX_eq, # includes early paying of CAPEX, CAPEX-phasing
            maintenance_cost_per_year = OPEX,
            tax_rate = inputs['tax_rate'],
            discount_rate = hpp_discount_factor,
            depreciation_yr = depreciation_yr,
            depreciation = depreciation,
            development_cost = DEVEX, 
            inflation_index = inflation_index,
        )
        
        break_even_PPA_price = np.maximum(
            0, 
            calculate_break_even_PPA_price(
                df = df, 
                CAPEX = CAPEX_eq, 
                OPEX = OPEX, 
                tax_rate = inputs['tax_rate'],
                discount_rate = hpp_discount_factor,
                depreciation_yr = depreciation_yr, 
                depreciation = depreciation, 
                DEVEX = DEVEX,
                inflation_index = inflation_index,
                ))

        outputs['NPV'] = NPV
        outputs['IRR'] = IRR
        outputs['NPV_over_CAPEX'] = NPV / CAPEX

        level_costs = np.sum(OPEX / (1 + hpp_discount_factor)**iy) + CAPEX
        AEP_per_year = df.groupby('i_year').hpp_t.mean()*365*24 + df.groupby('i_year').hpp_up_reg_t.mean()*365*24 - df.groupby('i_year').hpp_dwn_reg_t.mean()*365*24
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor)**iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        if level_AEP > 0:
            outputs['LCOE'] = level_costs / (level_AEP) # in Euro/MWh
        else:
            outputs['LCOE'] = 1e6

        outputs['mean_AEP'] = mean_AEP_per_year
        
        outputs['penalty_lifetime'] = df['penalty_t'].sum()
        outputs['break_even_PPA_price'] = break_even_PPA_price

# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------

def calculate_revenues_without_deg(df):
    df['revenue_without_deg'] = df['hpp_t'] * df['price_t'] + df['hpp_up_reg_t'] * df['price_up_reg_t'] - df['hpp_dwn_reg_t'] * df['price_dwn_reg_t'] - df['penalty_t']
    return df.groupby('i_year').revenue_without_deg.mean()*365*24

def calculate_revenues(df):
    df['revenue'] = df['hpp_t_deg'] * df['price_t'] + df['hpp_up_reg_t_deg'] * df['price_up_reg_t'] - df['hpp_dwn_reg_t_deg'] * df['price_dwn_reg_t'] - df['penalty_t_deg'] - df['penalty_t']
    return df.groupby('i_year').revenue.mean()*365*24

def calculate_break_even_PPA_price(df, CAPEX, OPEX, tax_rate, discount_rate,
                                   depreciation_yr, depreciation, DEVEX, inflation_index):
    def fun(price_el):
        revenues = calculate_revenues(df)
        NPV, _ = calculate_NPV_IRR(
            Net_revenue_t = revenues.values.flatten(),
            investment_cost = CAPEX,
            maintenance_cost_per_year = OPEX,
            tax_rate = tax_rate,
            discount_rate = discount_rate,
            depreciation_yr = depreciation_yr,
            depreciation = depreciation,
            development_cost = DEVEX,
            inflation_index = inflation_index,
        )
        return NPV ** 2
    out = sp.optimize.minimize(
        fun=fun, 
        x0=50, 
        method='SLSQP',
        tol=1e-10)
    return out.x


