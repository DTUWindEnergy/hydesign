import glob
import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
# import seaborn as sns
import openmdao.api as om
import yaml
import scipy as sp

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
        
        life_y = 25,
        save_finance_ts=False,
        work_dir='./',
        time_str='',
        ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """ 
        super().__init__()
        self.N_time = int(N_time)
        self.life_y = life_y
        self.life_h = int(life_y*365*24)

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
        
        self.save_finance_ts = save_finance_ts
        
        self.work_dir = work_dir
        self.time_str = time_str

    def setup(self):
        self.add_input('price_t_ext',
                       desc="Electricity price time series",
                       shape=[self.life_h])
        
        self.add_input('hpp_t_with_deg',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        
        self.add_input('penalty_t',
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
        self.add_output('revenues',
                        desc="Revenues")
        
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
        
        df['hpp_t'] = inputs['hpp_t_with_deg']
        # df['price_t'] = inputs['price_t_ext']
        df['penalty_t'] = inputs['penalty_t']
        # df['revenue'] = df['hpp_t'] * df['price_t'] - df['penalty_t']
        
        df['i_year'] = np.hstack([np.array([ii]*N_time) for ii in range(life_yr)])[:life_h]

        # Compute yearly revenues and cashflow
        revenues = calculate_revenues(inputs['price_t_ext'], df)
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
        
        revenues = revenues.values.flatten()
        outputs['CAPEX'] = CAPEX
        outputs['OPEX'] = OPEX
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
        AEP_per_year = df.groupby('i_year').hpp_t.mean()*365*24
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor)**iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        if level_AEP > 0:
            outputs['LCOE'] = level_costs / (level_AEP) # in Euro/MWh
        else:
            outputs['LCOE'] = 1e6

        outputs['mean_AEP'] = mean_AEP_per_year
        
        outputs['penalty_lifetime'] = df['penalty_t'].sum()
        outputs['break_even_PPA_price'] = break_even_PPA_price
        
        if self.save_finance_ts:
            df_out = pd.DataFrame()
            df_out['production'] = np.insert(np.sum(np.split(df['hpp_t'].values, 25), axis=1), 0, 0)
            df_out['revenue'] = np.insert(np.sum(np.split(df['hpp_t'].values*inputs['price_t_ext'], 25), axis=1), 0, 0)
            df_out['price_eq'] = df_out['revenue'] / df_out['production']
            df_out['actual_mean_price'] = np.insert(np.mean(np.split(inputs['price_t_ext'], 25),axis=1), 0, 0)
            df_out['year'] = np.arange(25 + 1)
            df_out['inflation_index'] = inflation_index
            df_out.to_csv(os.path.join(self.work_dir, f'finance_ts_{self.time_str}.csv'), sep=';')
            
            df_out2 = pd.DataFrame()
            df_out2['CAPEX'] = CAPEX
            df_out2['OPEX'] = OPEX
            df_out2['LCOE'] = outputs['LCOE']
            df_out2['NPV'] = [NPV]
            df_out2['WACC'] = hpp_discount_factor
            df_out2['TAX_rate'] = inputs['tax_rate']
            df_out2['break_even_price'] = break_even_PPA_price
            df_out2.to_csv(os.path.join(self.work_dir, f'finance_vals_{self.time_str}.csv'), sep=';')
        


# -----------------------------------------------------------------------
# Auxiliar functions for financial modelling
# -----------------------------------------------------------------------

def calculate_NPV_IRR(
    Net_revenue_t,
    investment_cost,
    maintenance_cost_per_year,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    development_cost,
    inflation_index
):
    """ A function to estimate the yearly cashflow using the net revenue time series, and the yearly OPEX costs.
    It then calculates the NPV and IRR using the yearly cashlow, the CAPEX, the WACC after tax, and the tax rate.

    Parameters
    ----------
    Net_revenue_t : Net revenue time series
    investment_cost : Capital costs
    maintenance_cost_per_year : yearly operation and maintenance costs
    tax_rate : tax rate
    discount_rate : Discount rate
    depreciation_yr : Depreciation curve (x-axis) time in years
    depreciation : Depreciation curve at the given times 
    development_cost : DEVEX
    inflation_index : Yearly Inflation index time-sereis

    Returns
    -------
    NPV : Net present value
    IRR : Internal rate of return
    """

    yr = np.arange(len(Net_revenue_t)+1) # extra year to start at 0 and end at end of lifetime.
    depre = np.interp(yr, depreciation_yr, depreciation)

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index[1:]

    # EBIT taxable income
    depreciation_on_each_year = np.diff(investment_cost*depre)
    EBIT = EBITDA - depreciation_on_each_year
    
    # Taxes
    Taxes = EBIT*tax_rate
    
    Net_income = EBITDA - Taxes
    Cashflow = np.insert(Net_income, 0, -investment_cost-development_cost)
    NPV = npf.npv(discount_rate, Cashflow)
    if NPV > 0:
        IRR = npf.irr(Cashflow)    
    else:
        IRR = 0
    return NPV, IRR

def calculate_WACC(
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_el,
    wind_WACC,
    solar_WACC,
    battery_WACC,
    ):
    """ This function returns the weighted average cost of capital after tax, using solar, wind, and battery
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

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Weighted average cost of capital 
    WACC_after_tax = \
        ( CAPEX_w * wind_WACC + \
          CAPEX_s * solar_WACC + \
          CAPEX_b * battery_WACC + \
          CAPEX_el * (wind_WACC + solar_WACC + battery_WACC)/3 ) / \
        ( CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_el )
    return WACC_after_tax


def calculate_revenues(price_el, df):
    df['revenue'] = df['hpp_t'] * np.broadcast_to(price_el, df['hpp_t'].shape) - df['penalty_t']
    return df.groupby('i_year').revenue.mean()*365*24

   

def calculate_break_even_PPA_price(df, CAPEX, OPEX, tax_rate, discount_rate,
                                   depreciation_yr, depreciation, DEVEX, inflation_index):
    def fun(price_el):
        revenues = calculate_revenues(price_el, df)
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


def calculate_CAPEX_phasing(
    CAPEX,
    phasing_yr,
    phasing_CAPEX,
    discount_rate,
    inflation_index,
    ):

    """ This function calulates the equivalent net present value CAPEX given a early paying "phasing" approach.

    Parameters
    ----------
    CAPEX : CAPEX 
    phasing_yr : Yearly early paying of CAPEX curve. x-axis, time in years. 
    phasing_CAPEX : Yearly early paying of CAPEX curve. Shares will be normalized to sum the CAPEX. 
    discount_rate : Discount rate for present value calculation
    inflation_index : Inflation index time series at the phasing_yr years. Accounts for inflation.

    Returns
    -------
    CAPEX_eq : Present value equivalent CAPEX
    """
    
    phasing_CAPEX = inflation_index*CAPEX*phasing_CAPEX/np.sum(phasing_CAPEX)
    CAPEX_eq = np.sum([phasing_CAPEX[ii]/( 1 + discount_rate)**yr for ii,yr in enumerate(phasing_yr)])
    
    return CAPEX_eq

def get_inflation_index(
    yr,
    inflation_yr, 
    inflation,
    ref_yr_inflation = 0
):
    """ This function calulates the inflation index time series.

    Parameters
    ----------
    yr : Years for eavaluation of the  inflation index
    inflation_yr : Yearly inflation curve. x-axis, time in years. To be used in interpolation.
    inflation : Yearly inflation curve.  To be used in interpolation.
    ref_yr_inflation : Referenece year, at which the inflation index takes value of 1.

    Returns
    -------
    inflation_index : inflation index time series at yr
    """
    infl = np.interp(yr, inflation_yr, inflation)
    
    
    ind_ref = np.where(np.array(yr)==ref_yr_inflation)[0]
    inflation_index = np.cumprod(1+np.array(infl))
    inflation_index = inflation_index/inflation_index[ind_ref]
    
    return inflation_index
