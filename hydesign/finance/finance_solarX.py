import os
import time

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
import openmdao.api as om
import scipy as sp
import openpyxl

class finance_solarX(om.ExplicitComponent):
    """
    Financial model to evaluate the profitability of a hybrid power plant (HPP).

    This model calculates annual cash flows, using parameters like Weighted Average Cost of Capital (WACC),
    tax rates, depreciation schedules, and inflation indices. Financial metrics like Net Present Value (NPV),
    Internal Rate of Return (IRR), Levelized Cost of Energy (LCOE), and other performance indicators are computed.
    """

    def __init__(
        self, 
        N_time,
        life_h,

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
        ):
        """
        Initializes the HPP financial model with parameters for depreciation, inflation, and CAPEX phasing.

        Parameters
        ----------
        N_time : int
            Number of hours in the representative dataset.
        life_h : int, optional
            Plant's operational lifetime in hours (default is 25 years).
        depreciation_yr : array-like
            Years for which depreciation is calculated.
        depreciation : array-like
            Values of depreciation at specified years.
        inflation_yr : array-like
            Years for which inflation rates are provided.
        inflation : array-like
            Inflation values corresponding to specified years.
        ref_yr_inflation : int
            Reference year for the inflation index.
        phasing_yr : array-like
            Years for CAPEX phasing.
        phasing_CAPEX : array-like
            CAPEX values associated with each phasing year.
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
        """
        Defines inputs and outputs for the financial model in OpenMDAO.

        Inputs represent various financial and operational parameters like CAPEX, OPEX, and prices.
        Outputs include computed financial metrics such as CAPEX, OPEX, NPV, IRR, and break-even PPA prices.
        """
        # inputs
        # hpp
        self.add_input('hpp_t_ext',
                       desc="HPP power time series",
                       units='MW',
                       shape=[self.life_h])
        self.add_input('hpp_curt_t_ext',
                       desc="HPP curtailment power time series",
                       units='MW',
                       shape=[self.life_h])

        # cpv
        self.add_input('cpv_t_ext',
                       desc="cpv power time series",
                       units='MW',
                       shape=[self.life_h])


        # cst
        self.add_input('p_st_t_ext',
                       desc="CSP power time series",
                       units='MW',
                       shape=[self.life_h])
        self.add_input('q_t_ext',
                       desc="produced heat time series",
                       units='MJ',
                       shape=[self.life_h])


        # biogas_h2
        self.add_input('h2_t_ext',
                       desc="H2 production time series",
                       units='kg/h',
                       shape=[self.life_h])
        self.add_input('biogas_t_ext',
                       desc="Consumed biogas time series",
                       units='kg/h',
                       shape=[self.life_h])

        # prices
        self.add_input('price_el_t_ext',
                       desc="Electricity price time series",
                       shape=[self.life_h])
        self.add_input('price_h2_t_ext',
                       desc="Hydrogen price time series",
                       shape=[self.life_h])
    
        self.add_input('price_biogas_t_ext',
                       desc="Biogas price time series",
                       shape=[self.life_h])

        # capex and opex
        self.add_input('CAPEX_sf',
                       desc="CAPEX solar field")
        self.add_input('OPEX_sf',
                       desc="OPEX solar field")
        self.add_input('CAPEX_cpv',
                       desc="CAPEX cpv")
        self.add_input('OPEX_cpv',
                       desc="OPEX cpv")
        self.add_input('CAPEX_cst',
                       desc="CAPEX CST")
        self.add_input('OPEX_cst',
                       desc="OPEX CSP")
        self.add_input('CAPEX_h2',
                       desc="CAPEX Biogas_H2")
        self.add_input('OPEX_h2',
                       desc="OPEX H2")
        self.add_input('CAPEX_sh',
                       desc="CAPEX electrical infrastructure")
        self.add_input('OPEX_sh',
                       desc="OPEX electrical infrastructure")

        # other
        self.add_input('penalty_t_ext',
                        desc="penalty for not reaching expected energy production at peak hours",
                        shape=[self.life_h])
        self.add_input('penalty_q_t_ext',
                        desc="penalty for not reaching expected heat production",
                        shape=[self.life_h])
        self.add_input('discount_rate',
                       desc="discount rate")
        self.add_input('tax_rate',
                       desc="Corporate tax rate")

        # outputs
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
        self.add_output('mean_AH2P',
                        desc="mean annual H2 production")
        self.add_output('LCOE',
                        desc="LCOE")
        self.add_output('revenues',
                        desc="Revenues")
        self.add_output('penalty_lifetime',
                        desc="penalty_lifetime")
        self.add_output('break_even_PPA_price',
                        desc='PPA price of electricity that results in NPV=0 with the given hybrid power plant configuration and operation')
        self.add_output('break_even_PPA_price_h2',
                        desc='PPA price of hydrogen that results in NPV=0 with the given hybrid power plant configuration and operation')
        self.add_output('break_even_PPA_price_q',
                        desc='PPA price of heat that results in NPV=0 with the given hybrid power plant configuration and operation')
        self.add_output('lcove',
                        desc="cost of valued energy")

    def compute(self, inputs, outputs):
        """Computes financial metrics based on revenue, CAPEX, OPEX, and other financial parameters.

        The method calculates cash flow using inflation-adjusted revenues, CAPEX phasing, and
        depreciation. It computes financial indicators like NPV, IRR, LCOE, and additional metrics for
        profitability evaluation.

        Parameters
        ----------
        inputs : dict
            Dictionary of input values, such as CAPEX, OPEX, price time series, and tax rate.
        outputs : dict
            Dictionary of computed financial outputs, including CAPEX, OPEX, NPV, IRR, etc.
        """

        # Extract inputs and setup time-based parameters
        N_time = self.N_time
        life_h = self.life_h
        life_yr = int(np.ceil(life_h/N_time))

        # Preparing data for financial calculations including revenue and depreciation
        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation
        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation
        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX
        df = pd.DataFrame()
        df['hpp_t'] = inputs['hpp_t_ext']
        df['p_cpv_t'] = inputs['cpv_t_ext']
        df['p_st_t'] = inputs['p_st_t_ext']
        df['h2_t'] = inputs['h2_t_ext']
        df['q_t'] = inputs['q_t_ext']
        df['biogas_t'] = inputs['biogas_t_ext']
        df['penalty_t'] = inputs['penalty_t_ext']
        df['penalty_q_t'] = inputs['penalty_q_t_ext']
        df['i_year'] = np.hstack([np.array([ii]*N_time) for ii in range(life_yr)])[:life_h]
        df['price_el_t'] = inputs['price_el_t_ext']
        df['price_h2_t'] = inputs['price_h2_t_ext']
        df['price_biogas_t'] = inputs['price_biogas_t_ext']

        # Calculate yearly revenues and cash flow
        revenues = calculate_revenues(df)
        CAPEX = inputs['CAPEX_sf'] + inputs['CAPEX_cpv'] + \
            inputs['CAPEX_cst'] + inputs['CAPEX_sh'] + inputs['CAPEX_h2']
        OPEX = inputs['OPEX_sf'] * life_yr + inputs['OPEX_cpv'] * life_yr + \
            inputs['OPEX_cst'] * life_yr + inputs['OPEX_sh'] * life_yr + inputs['OPEX_h2']
        
        # Calculate present value of CAPEX based on phasing and inflation adjustment
        inflation_index_phasing = get_inflation_index(
            yr = phasing_yr,
            inflation_yr = inflation_yr, 
            inflation = inflation,
            ref_yr_inflation = ref_yr_inflation,)

        CAPEX_eq = calculate_CAPEX_phasing(
            CAPEX = CAPEX,
            phasing_yr = phasing_yr,
            phasing_CAPEX = phasing_CAPEX,
            discount_rate = inputs['discount_rate'], # hpp_discount_factor,
            inflation_index = inflation_index_phasing,)

        # Final calculations for NPV, IRR, and LCOE based on revenues and annualized costs
        # len of revenues = years of life
        iy = np.arange(len(revenues)) + 1 # Plus becasue the year zero is added externally in the NPV and IRR calculations
        
        # Compute inflation, all cahsflow are in nominal prices
        inflation_index = get_inflation_index(
                yr = np.arange(len(revenues)+1), # It includes t=0, to compute the reference
                inflation_yr = inflation_yr, 
                inflation = inflation,
                ref_yr_inflation = ref_yr_inflation,)
        
        revenues = revenues.values.flatten()
        outputs['CAPEX'] = CAPEX
        outputs['OPEX'] = OPEX
        outputs['revenues'] = revenues.sum()

        # DEVEX
        DEVEX = 0

        opex_per_year = OPEX / life_yr

        # Calculate the 
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t = revenues,
            investment_cost = CAPEX_eq, # includes early paying of CAPEX, CAPEX-phasing
            maintenance_cost_per_year = opex_per_year,
            tax_rate = inputs['tax_rate'],
            discount_rate = inputs['discount_rate'], #hpp_discount_factor,
            depreciation_yr = depreciation_yr,
            depreciation = depreciation,
            development_cost = DEVEX, 
            inflation_index = inflation_index,
        )

        break_even_price_el, break_even_price_h2, break_even_price_q = calculate_break_even_PPA_price(
            df=df,
            CAPEX=CAPEX_eq,
            OPEX=OPEX,
            tax_rate=inputs['tax_rate'],
            discount_rate=inputs['discount_rate'],
            depreciation_yr=depreciation_yr,
            depreciation=depreciation,
            DEVEX=DEVEX,
            inflation_index=inflation_index
        )

        # Ensure that the break-even prices are not negative
        break_even_PPA_price_el = np.maximum(0, break_even_price_el)
        break_even_PPA_price_h2 = np.maximum(0, break_even_price_h2)
        break_even_PPA_price_q = np.maximum(0, break_even_price_q)

        outputs['NPV'] = NPV
        outputs['IRR'] = IRR
        outputs['NPV_over_CAPEX'] = NPV / CAPEX

        level_costs = np.sum(opex_per_year / (1 + inputs['discount_rate'])**iy) + CAPEX
        AEP_per_year = df.groupby('i_year').hpp_t.mean()*365*24
        level_AEP = np.sum(AEP_per_year / (1 + inputs['discount_rate'])**iy)
        mean_AEP_per_year = np.mean(AEP_per_year)

        # Excluding H2 cost and consumpsion for LCOE
        CAPEX_no_h2 = inputs['CAPEX_sf'] + inputs['CAPEX_cpv'] + \
                inputs['CAPEX_cst'] + inputs['CAPEX_sh']
        OPEX_no_h2 = inputs['OPEX_sf'] * life_yr + inputs['OPEX_cpv'] * life_yr + \
               inputs['OPEX_cst'] * life_yr + inputs['OPEX_sh'] * life_yr
        opex_no_h2_per_year = OPEX_no_h2 / life_yr
        level_costs_no_h2 = np.sum(opex_no_h2_per_year / (1 + inputs['discount_rate']) ** iy) + CAPEX_no_h2
        AEP_per_year_no_h2 = (df.groupby('i_year').p_cpv_t.mean() + df.groupby('i_year').p_st_t.mean() )* 365 * 24
        level_AEP_no_h2 = np.sum(AEP_per_year_no_h2 / (1 + inputs['discount_rate'])**iy)

        if level_AEP > 0:
            outputs['LCOE'] = level_costs_no_h2 / (level_AEP_no_h2) # in Euro/MWh
        else:
            outputs['LCOE'] = 1e6

        outputs['mean_AEP'] = mean_AEP_per_year
        
        outputs['penalty_lifetime'] = df['penalty_t'].sum()
        outputs['break_even_PPA_price'] = break_even_PPA_price_el
        outputs['break_even_PPA_price_h2'] = break_even_PPA_price_h2

        # AH2P
        AH2P_per_year = df.groupby('i_year').h2_t.mean() * 365 * 24
        mean_AH2P_per_year = np.mean(AH2P_per_year)
        outputs['mean_AH2P'] = mean_AH2P_per_year

        # lcove
        revenues_discount = np.sum(revenues / (1 + inputs['discount_rate'])**iy)
        lcove = level_costs / revenues_discount
        outputs['lcove'] = lcove

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


def calculate_revenues(df):
    # Extract prices from df (assuming prices are stored in columns)
    price_el_t = df['price_el_t']  # Price of electricity
    price_h2_t = df['price_h2_t']  # Price of hydrogen
    price_biogas_t = df['price_biogas_t']  # Price of biogas (input cost)

    # Calculate the contribution of each price to the revenue
    revenue_el = df['hpp_t'] * price_el_t  # Revenue from electricity
    revenue_h2 = df['h2_t'] * price_h2_t  # Revenue from hydrogen
    #revenue_biogas = df['biogas_t'] * price_biogas_t  # Cost (negative revenue) from biogas

    # Calculate the total revenue for each time step
    df['revenue'] = revenue_el + revenue_h2 - df['penalty_t'] - df['penalty_q_t']  # Adjust with penalties

    # Return the yearly mean revenue, scaled to annual hours (365 * 24)
    return df.groupby('i_year')['revenue'].mean() * 365 * 24

def calculate_break_even_PPA_price(df, CAPEX, OPEX, tax_rate, discount_rate,
                               depreciation_yr, depreciation, DEVEX, inflation_index,
                               initial_price_el=50, initial_price_h2=50, initial_price_q=50):

    # Generalized objective function for any price (electricity, hydrogen, or heat)
    def fun(price, price_type):
        # Update the price in the DataFrame depending on the price_type
        if price_type == 'el':
            df['price_el_t'] = np.broadcast_to(price, df['price_el_t'].shape) # Update price for electricity
        elif price_type == 'h2':
            df['price_h2_t'] = np.broadcast_to(price, df['price_h2_t'].shape)  # Update price for hydrogen

        # Calculate the total revenue with the updated price
        revenues = calculate_revenues(df)

        # Calculate NPV with the updated revenue
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
        return NPV ** 2  # Minimize the square of NPV

    # Function to calculate break-even price for any price type
    def calculate_break_even_for_price(price_type, initial_price):
        out = sp.optimize.minimize(
            fun=fun,
            x0=initial_price,  # Initial guess for the given price
            args=(price_type,),  # Pass price_type to fun (i.e., 'el', 'h2', 'q')
            method='SLSQP',
            tol=1e-10
        )
        return out.x

    # Calculate the break-even price for electricity
    break_even_price_el = calculate_break_even_for_price('el', initial_price_el)

    # Calculate the break-even price for hydrogen
    break_even_price_h2 = calculate_break_even_for_price('h2', initial_price_h2)

    # Calculate the break-even price for heat
    break_even_price_q = calculate_break_even_for_price('q', initial_price_q)

    return break_even_price_el, break_even_price_h2, break_even_price_q

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

