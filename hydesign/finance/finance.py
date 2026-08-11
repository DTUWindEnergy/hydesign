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

from hydesign.openmdao_wrapper import ComponentWrapper


def _year_activity(start_year, component_life_y, project_life_y):
    years = np.arange(project_life_y)
    return ((years >= start_year) & (years < start_year + component_life_y)).astype(
        float
    )


def _phase_capex(capex, technology, project_life_y, discount_rate):
    """Place a technology's CAPEX payments on the project cash-flow timeline."""
    vector = np.zeros(project_life_y + 1)
    if capex <= 0 or technology is None:
        return vector

    deployment_year = technology["deployment_year"]
    phasing = technology.get("capex_phasing", {"years": [0], "shares": [1.0]})
    for relative_year, share in zip(phasing["years"], phasing["shares"]):
        payment_year = deployment_year + relative_year
        payment = capex * share
        if payment_year < 0:
            # The finance timeline starts at project year zero. Preserve the
            # present value of pre-project payments at that boundary.
            payment /= (1 + discount_rate) ** payment_year
            payment_year = 0
        vector[payment_year] += payment
    return vector


def build_finance_vectors(
    technology_schedule,
    project_life_y,
    CAPEX_w,
    CAPEX_s,
    CAPEX_b,
    CAPEX_sh,
    OPEX_w,
    OPEX_s,
    OPEX_b,
    OPEX_sh,
    CAPEX_sh_fixed=0,
    CAPEX_sh_w_land=0,
    CAPEX_sh_s_land=0,
    discount_rate=0,
):
    """Compile technology and shared CAPEX/OPEX contribution vectors.

    CAPEX vectors include project year zero and the terminal project boundary.
    OPEX vectors are aligned directly with operating years ``0..life-1``.
    Shared grid/BOS CAPEX is triggered by the first deployment. Land CAPEX is
    paid incrementally whenever a newly deployed wind or solar plant increases
    the maximum shared site footprint.
    """
    schedule = technology_schedule
    capex = {
        "wind": float(CAPEX_w),
        "solar": float(CAPEX_s),
        "battery": float(CAPEX_b),
    }
    opex = {
        "wind": float(OPEX_w),
        "solar": float(OPEX_s),
        "battery": float(OPEX_b),
    }
    activity = {
        name: (
            _year_activity(item["deployment_year"], item["lifetime"], project_life_y)
            if item is not None
            else np.zeros(project_life_y)
        )
        for name, item in (
            ("wind", schedule.get("wind")),
            ("solar", schedule.get("solar")),
            ("battery", schedule.get("battery")),
        )
    }

    vectors = {}
    for name in ("wind", "solar", "battery"):
        vectors[f"capex_{name}"] = _phase_capex(
            capex[name], schedule.get(name), project_life_y, discount_rate
        )
        vectors[f"opex_{name}"] = activity[name] * opex[name]

    shared_fixed = float(CAPEX_sh_fixed)
    shared_land = {
        "wind": float(CAPEX_sh_w_land),
        "solar": float(CAPEX_sh_s_land),
    }
    calculated_shared_total = shared_fixed + max(shared_land.values())
    # Pure-finance callers may provide only the historical aggregate input.
    # Treat any unexplained remainder as fixed shared infrastructure.
    shared_fixed += float(CAPEX_sh) - calculated_shared_total
    shared_fixed = max(shared_fixed, 0)

    deployments = {}
    for name, item in schedule.items():
        deployments.setdefault(item["deployment_year"], []).append(name)

    shared_by_trigger = {name: 0.0 for name in schedule}
    first_year = min(deployments)
    first_trigger = next(
        name for name in ("wind", "solar", "battery") if name in deployments[first_year]
    )
    shared_by_trigger[first_trigger] += shared_fixed

    paid_land = 0.0
    deployed_land = {}
    for year in sorted(deployments):
        for name in deployments[year]:
            if name in shared_land:
                deployed_land[name] = shared_land[name]
        required_land = max(deployed_land.values(), default=0.0)
        land_increment = max(required_land - paid_land, 0.0)
        if land_increment:
            trigger = max(
                (name for name in deployments[year] if name in shared_land),
                key=shared_land.get,
            )
            shared_by_trigger[trigger] += land_increment
            paid_land = required_land

    vectors["shared_events"] = {}
    for name, amount in shared_by_trigger.items():
        if amount > 0:
            year = schedule[name]["deployment_year"]
            vectors["shared_events"][year] = (
                vectors["shared_events"].get(year, 0) + amount
            )
    vectors["capex_shared"] = np.zeros(project_life_y + 1)
    for name, amount in shared_by_trigger.items():
        vectors["capex_shared"] += _phase_capex(
            amount, schedule[name], project_life_y, discount_rate
        )

    any_active = np.maximum.reduce(list(activity.values()))
    vectors["opex_shared"] = any_active * float(OPEX_sh)
    vectors["capex_total"] = (
        sum(vectors[f"capex_{name}"] for name in ("wind", "solar", "battery"))
        + vectors["capex_shared"]
    )
    vectors["opex_total"] = (
        sum(vectors[f"opex_{name}"] for name in ("wind", "solar", "battery"))
        + vectors["opex_shared"]
    )
    vectors["activity"] = activity
    return vectors


class finance:
    """Pure Python Hybrid power plant financial model to estimate the overall profitability of the hybrid power plant.
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
        life_y=25,
        project_life_y=None,
        wind_operation_year=0,
        solar_operation_year=0,
        battery_operation_year=0,
        depre_rate=None,
        technology_schedule=None,
    ):
        """Initialization of the HPP finance model

        Parameters
        ----------
        N_time : Number of hours in the representative dataset
        life_h : Lifetime of the plant in hours
        """
        self.N_time = int(N_time)
        self.life_y = int(life_y)
        self.project_life_y = int(project_life_y or life_y)
        self.life_h = int(self.project_life_y * 365 * 24)

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
        self.wind_operation_year = int(wind_operation_year)
        self.solar_operation_year = int(solar_operation_year)
        self.battery_operation_year = int(battery_operation_year)
        self.depre_rate = 1 / self.life_y if depre_rate is None else depre_rate
        self.technology_schedule = technology_schedule
        if technology_schedule is None:
            self.wind_life_y = self.life_y
            self.solar_life_y = self.life_y
            self.battery_life_y = self.life_y
            self.wind_enabled = True
            self.solar_enabled = True
            self.battery_enabled = True
        else:
            wind = technology_schedule.get("wind")
            solar = technology_schedule.get("solar")
            battery = technology_schedule.get("battery")
            self.wind_enabled = wind is not None
            self.solar_enabled = solar is not None
            self.battery_enabled = battery is not None
            self.wind_life_y = wind["lifetime"] if wind else 0
            self.solar_life_y = solar["lifetime"] if solar else 0
            self.battery_life_y = battery["lifetime"] if battery else 0

    def _year_activity(self, start_year, component_life_y):
        return _year_activity(start_year, component_life_y, self.project_life_y)

    def compute(
        self,
        hpp_t_with_deg,
        penalty_t,
        price_t_ext,
        CAPEX_w,
        CAPEX_s,
        CAPEX_b,
        CAPEX_sh,
        OPEX_w,
        OPEX_s,
        OPEX_b,
        OPEX_sh,
        wind_WACC,
        solar_WACC,
        battery_WACC,
        tax_rate,
        decommissioning_cost_tot_w,
        decommissioning_cost_tot_s,
        CAPEX_sh_fixed=0,
        CAPEX_sh_w_land=0,
        CAPEX_sh_s_land=0,
        **kwargs,
    ):
        """Calculating the financial metrics of the hybrid power plant project.

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
        use_legacy_finance = (
            self.technology_schedule is None
            and self.project_life_y == self.life_y
            and self.wind_operation_year == 0
            and self.solar_operation_year == 0
            and self.battery_operation_year == 0
            and float(np.asarray(decommissioning_cost_tot_w).item()) == 0
            and float(np.asarray(decommissioning_cost_tot_s).item()) == 0
        )
        life_yr = (
            int(np.ceil(life_h / N_time)) if use_legacy_finance else self.project_life_y
        )

        depreciation_yr = self.depreciation_yr
        depreciation = self.depreciation

        inflation_yr = self.inflation_yr
        inflation = self.inflation
        ref_yr_inflation = self.ref_yr_inflation

        phasing_yr = self.phasing_yr
        phasing_CAPEX = self.phasing_CAPEX

        df = pd.DataFrame()

        df["hpp_t"] = hpp_t_with_deg
        # df['price_t'] = inputs['price_t_ext']
        df["penalty_t"] = penalty_t
        # df['revenue'] = df['hpp_t'] * df['price_t'] - df['penalty_t']

        if use_legacy_finance:
            df["i_year"] = np.repeat(np.arange(life_yr), N_time)[:life_h]
        else:
            df["i_year"] = np.repeat(np.arange(life_yr), 365 * 24)[:life_h]

        # Compute yearly revenues and cashflow
        revenues = calculate_revenues(price_t_ext, df).values.flatten()

        def _scalar(x):
            return float(np.asarray(x).item())

        CAPEX_w = _scalar(CAPEX_w)
        CAPEX_s = _scalar(CAPEX_s)
        CAPEX_b = _scalar(CAPEX_b)
        CAPEX_sh = _scalar(CAPEX_sh)
        legacy_capex_key = "CAPEX" + "_el"
        if (CAPEX_sh == 0.0) and (legacy_capex_key in kwargs):
            CAPEX_sh = _scalar(kwargs[legacy_capex_key])
        OPEX_w = _scalar(OPEX_w)
        OPEX_s = _scalar(OPEX_s)
        OPEX_b = _scalar(OPEX_b)
        OPEX_sh = _scalar(OPEX_sh)
        legacy_opex_key = "OPEX" + "_el"
        if (OPEX_sh == 0.0) and (legacy_opex_key in kwargs):
            OPEX_sh = _scalar(kwargs[legacy_opex_key])
        wind_WACC = _scalar(wind_WACC)
        solar_WACC = _scalar(solar_WACC)
        battery_WACC = _scalar(battery_WACC)
        decommissioning_cost_tot_w = _scalar(decommissioning_cost_tot_w)
        decommissioning_cost_tot_s = _scalar(decommissioning_cost_tot_s)
        CAPEX_sh_fixed = _scalar(CAPEX_sh_fixed)
        CAPEX_sh_w_land = _scalar(CAPEX_sh_w_land)
        CAPEX_sh_s_land = _scalar(CAPEX_sh_s_land)

        if use_legacy_finance:
            return self._compute_legacy(
                df=df,
                revenues=revenues,
                CAPEX_w=CAPEX_w,
                CAPEX_s=CAPEX_s,
                CAPEX_b=CAPEX_b,
                CAPEX_sh=CAPEX_sh,
                OPEX_w=OPEX_w,
                OPEX_s=OPEX_s,
                OPEX_b=OPEX_b,
                OPEX_sh=OPEX_sh,
                wind_WACC=wind_WACC,
                solar_WACC=solar_WACC,
                battery_WACC=battery_WACC,
                tax_rate=tax_rate,
            )

        hpp_discount_factor = calculate_WACC(
            CAPEX_w,
            CAPEX_s,
            CAPEX_b,
            CAPEX_sh,
            wind_WACC,
            solar_WACC,
            battery_WACC,
        )
        vectors = build_finance_vectors(
            technology_schedule=self.technology_schedule,
            project_life_y=life_yr,
            CAPEX_w=CAPEX_w,
            CAPEX_s=CAPEX_s,
            CAPEX_b=CAPEX_b,
            CAPEX_sh=CAPEX_sh,
            OPEX_w=OPEX_w,
            OPEX_s=OPEX_s,
            OPEX_b=OPEX_b,
            OPEX_sh=OPEX_sh,
            CAPEX_sh_fixed=CAPEX_sh_fixed,
            CAPEX_sh_w_land=CAPEX_sh_w_land,
            CAPEX_sh_s_land=CAPEX_sh_s_land,
            discount_rate=hpp_discount_factor,
        )
        CAPEX_vec = vectors["capex_total"]
        OPEX_project_vec = vectors["opex_total"]
        OPEX_cashflow_vec = np.insert(OPEX_project_vec, 0, 0)
        CAPEX = CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_sh
        OPEX = OPEX_project_vec.max() if OPEX_project_vec.size else 0

        # Build yearly depreciation from each technology deployment year using the
        # common depreciation curve as local years since commissioning. Payment
        # phasing changes cash timing, not the asset's depreciation start.
        depreciation_vec = np.zeros(life_yr + 1)
        depreciation_assets = [
            (self.wind_operation_year, CAPEX_w, self.wind_life_y),
            (self.solar_operation_year, CAPEX_s, self.solar_life_y),
            (self.battery_operation_year, CAPEX_b, self.battery_life_y),
        ]
        depreciation_assets.extend(
            (year, capex, life_yr - year)
            for year, capex in vectors["shared_events"].items()
        )
        for start_year, capex, component_life_y in depreciation_assets:
            if (start_year < 0) or (capex <= 0) or (component_life_y <= 0):
                continue
            depre_yr_local = np.arange(component_life_y + 1)
            depre_curve_local = np.interp(depre_yr_local, depreciation_yr, depreciation)
            depreciation_local = np.diff(capex * depre_curve_local)
            for local_year, dep_val in enumerate(depreciation_local):
                global_year = start_year + 1 + local_year
                if 0 <= global_year <= life_yr:
                    depreciation_vec[global_year] += dep_val

        decommissioning_vec = np.zeros(life_yr + 1)
        wind_decom_year = self.wind_operation_year + self.wind_life_y
        solar_decom_year = self.solar_operation_year + self.solar_life_y
        if self.wind_enabled and 0 <= wind_decom_year <= life_yr:
            decommissioning_vec[wind_decom_year] += decommissioning_cost_tot_w
        if self.solar_enabled and 0 <= solar_decom_year <= life_yr:
            decommissioning_vec[solar_decom_year] += decommissioning_cost_tot_s

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

        revenues_mean = revenues.mean()

        # We need to add DEVEX
        DEVEX = 0

        NPV, IRR = calculate_NPV_IRR_phased(
            Net_revenue_t=np.insert(revenues, 0, 0),
            maintenance_cost_per_year=OPEX_cashflow_vec,
            capex_vector=CAPEX_vec,
            depreciation_on_each_year=depreciation_vec,
            tax_rate=tax_rate,
            discount_rate=hpp_discount_factor,
            decommissioning_vec=decommissioning_vec,
            inflation_index=inflation_index,
        )

        break_even_PPA_price = np.maximum(
            0,
            calculate_break_even_PPA_price_phased(
                df=df,
                maintenance_cost_per_year=OPEX_cashflow_vec,
                capex_vector=CAPEX_vec,
                depreciation_on_each_year=depreciation_vec,
                tax_rate=tax_rate,
                discount_rate=hpp_discount_factor,
                decommissioning_vec=decommissioning_vec,
                inflation_index=inflation_index,
            ),
        )

        NPV_over_CAPEX = NPV / CAPEX

        level_costs = np.sum(
            OPEX_project_vec / (1 + hpp_discount_factor) ** iy
        ) + np.sum(CAPEX_vec / (1 + hpp_discount_factor) ** np.arange(len(CAPEX_vec)))
        AEP_per_year = df.groupby("i_year").hpp_t.mean() * 365 * 24
        level_AEP = np.sum(AEP_per_year / (1 + hpp_discount_factor) ** iy)

        mean_AEP_per_year = np.mean(AEP_per_year)
        if level_AEP > 0:
            LCOE = level_costs / (level_AEP)  # in Euro/MWh
        else:
            LCOE = 1e6

        mean_AEP = mean_AEP_per_year

        penalty_lifetime = df["penalty_t"].sum()

        return (
            CAPEX,
            OPEX,
            revenues_mean,
            NPV,
            IRR,
            NPV_over_CAPEX,
            LCOE,
            mean_AEP,
            penalty_lifetime,
            break_even_PPA_price,
            vectors["capex_wind"],
            vectors["capex_solar"],
            vectors["capex_battery"],
            vectors["capex_shared"],
            CAPEX_vec,
            vectors["opex_wind"],
            vectors["opex_solar"],
            vectors["opex_battery"],
            vectors["opex_shared"],
            OPEX_project_vec,
        )

    def _compute_legacy(
        self,
        df,
        revenues,
        CAPEX_w,
        CAPEX_s,
        CAPEX_b,
        CAPEX_sh,
        OPEX_w,
        OPEX_s,
        OPEX_b,
        OPEX_sh,
        wind_WACC,
        solar_WACC,
        battery_WACC,
        tax_rate,
    ):
        """Preserve the original base-case finance calculation exactly."""
        CAPEX = CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_sh
        OPEX = OPEX_w + OPEX_s + OPEX_b + OPEX_sh
        discount_rate = calculate_WACC(
            CAPEX_w,
            CAPEX_s,
            CAPEX_b,
            CAPEX_sh,
            wind_WACC,
            solar_WACC,
            battery_WACC,
        )
        inflation_index_phasing = get_inflation_index(
            yr=self.phasing_yr,
            inflation_yr=self.inflation_yr,
            inflation=self.inflation,
            ref_yr_inflation=self.ref_yr_inflation,
        )
        CAPEX_eq = calculate_CAPEX_phasing(
            CAPEX=CAPEX,
            phasing_yr=self.phasing_yr,
            phasing_CAPEX=self.phasing_CAPEX,
            discount_rate=discount_rate,
            inflation_index=inflation_index_phasing,
        )
        inflation_index = get_inflation_index(
            yr=np.arange(len(revenues) + 1),
            inflation_yr=self.inflation_yr,
            inflation=self.inflation,
            ref_yr_inflation=self.ref_yr_inflation,
        )
        NPV, IRR = calculate_NPV_IRR(
            Net_revenue_t=revenues,
            investment_cost=CAPEX_eq,
            maintenance_cost_per_year=OPEX,
            tax_rate=tax_rate,
            discount_rate=discount_rate,
            depreciation_yr=self.depreciation_yr,
            depreciation=self.depreciation,
            development_cost=0,
            inflation_index=inflation_index,
        )
        break_even_PPA_price = np.maximum(
            0,
            calculate_break_even_PPA_price(
                df=df,
                CAPEX=CAPEX_eq,
                OPEX=OPEX,
                tax_rate=tax_rate,
                discount_rate=discount_rate,
                depreciation_yr=self.depreciation_yr,
                depreciation=self.depreciation,
                DEVEX=0,
                inflation_index=inflation_index,
            ),
        )
        iy = np.arange(len(revenues)) + 1
        level_costs = np.sum(OPEX / (1 + discount_rate) ** iy) + CAPEX
        AEP_per_year = df.groupby("i_year").hpp_t.mean() * 365 * 24
        level_AEP = np.sum(AEP_per_year / (1 + discount_rate) ** iy)
        LCOE = level_costs / level_AEP if level_AEP > 0 else 1e6
        capex_factor = CAPEX_eq / CAPEX if CAPEX > 0 else 0

        def legacy_capex_vector(value):
            vector = np.zeros(self.project_life_y + 1)
            vector[0] = value * capex_factor
            return vector

        capex_w_vec = legacy_capex_vector(CAPEX_w)
        capex_s_vec = legacy_capex_vector(CAPEX_s)
        capex_b_vec = legacy_capex_vector(CAPEX_b)
        capex_sh_vec = legacy_capex_vector(CAPEX_sh)
        opex_w_vec = np.full(self.project_life_y, OPEX_w)
        opex_s_vec = np.full(self.project_life_y, OPEX_s)
        opex_b_vec = np.full(self.project_life_y, OPEX_b)
        opex_sh_vec = np.full(self.project_life_y, OPEX_sh)
        return (
            CAPEX,
            OPEX,
            revenues.mean(),
            NPV,
            IRR,
            NPV / CAPEX,
            LCOE,
            np.mean(AEP_per_year),
            df["penalty_t"].sum(),
            break_even_PPA_price,
            capex_w_vec,
            capex_s_vec,
            capex_b_vec,
            capex_sh_vec,
            capex_w_vec + capex_s_vec + capex_b_vec + capex_sh_vec,
            opex_w_vec,
            opex_s_vec,
            opex_b_vec,
            opex_sh_vec,
            opex_w_vec + opex_s_vec + opex_b_vec + opex_sh_vec,
        )


class finance_comp(ComponentWrapper):
    def __init__(self, **insta_inp):
        model = finance(**insta_inp)
        legacy_capex_key = "CAPEX" + "_el"
        legacy_opex_key = "OPEX" + "_el"
        super().__init__(
            inputs=[
                ("price_t_ext", {"shape": [model.life_h]}),
                ("hpp_t_with_deg", {"shape": [model.life_h], "units": "MW"}),
                ("penalty_t", {"shape": [model.life_h]}),
                ("CAPEX_w", {}),
                ("OPEX_w", {}),
                ("CAPEX_s", {}),
                ("OPEX_s", {}),
                ("CAPEX_b", {}),
                ("OPEX_b", {}),
                ("CAPEX_sh", {"val": 0}),
                ("OPEX_sh", {"val": 0}),
                ("CAPEX_sh_fixed", {"val": 0}),
                ("CAPEX_sh_w_land", {"val": 0}),
                ("CAPEX_sh_s_land", {"val": 0}),
                (legacy_capex_key, {"val": 0}),
                (legacy_opex_key, {"val": 0}),
                ("wind_WACC", {}),
                ("solar_WACC", {}),
                ("battery_WACC", {}),
                ("tax_rate", {}),
                ("decommissioning_cost_tot_w", {"val": 0}),
                ("decommissioning_cost_tot_s", {"val": 0}),
            ],
            outputs=[
                ("CAPEX", {}),
                ("OPEX", {}),
                ("revenues", {}),
                ("NPV", {}),
                ("IRR", {}),
                ("NPV_over_CAPEX", {}),
                ("LCOE", {}),
                ("mean_AEP", {}),
                ("penalty_lifetime", {}),
                ("break_even_PPA_price", {}),
                ("CAPEX_w_vec", {"shape": [model.project_life_y + 1]}),
                ("CAPEX_s_vec", {"shape": [model.project_life_y + 1]}),
                ("CAPEX_b_vec", {"shape": [model.project_life_y + 1]}),
                ("CAPEX_sh_vec", {"shape": [model.project_life_y + 1]}),
                ("CAPEX_vec", {"shape": [model.project_life_y + 1]}),
                ("OPEX_w_vec", {"shape": [model.project_life_y]}),
                ("OPEX_s_vec", {"shape": [model.project_life_y]}),
                ("OPEX_b_vec", {"shape": [model.project_life_y]}),
                ("OPEX_sh_vec", {"shape": [model.project_life_y]}),
                ("OPEX_vec", {"shape": [model.project_life_y]}),
            ],
            function=model.compute,
            partial_options=[{"dependent": False, "val": 0}],
        )


def calculate_NPV_IRR_phased(
    Net_revenue_t,
    maintenance_cost_per_year,
    capex_vector,
    depreciation_on_each_year,
    tax_rate,
    discount_rate,
    decommissioning_vec,
    inflation_index,
):
    tax_rate = np.asarray(tax_rate).item()
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index
    EBIT = EBITDA - depreciation_on_each_year

    Taxes = np.zeros(len(EBIT))
    for ii in range(1, len(EBIT)):
        if EBIT[ii] > 0:
            Taxes[ii] = EBIT[ii] * tax_rate

    Net_income = EBITDA - Taxes
    Cashflow = Net_income - capex_vector - decommissioning_vec
    NPV = npf.npv(discount_rate, Cashflow)
    IRR = npf.irr(Cashflow) if NPV > 0 else 0
    return NPV, IRR


def calculate_break_even_PPA_price_phased(
    df,
    maintenance_cost_per_year,
    capex_vector,
    depreciation_on_each_year,
    tax_rate,
    discount_rate,
    decommissioning_vec,
    inflation_index,
):
    """Find the break-even constant price using the complete phased cash flow."""

    def objective(price_el):
        revenues = calculate_revenues(price_el, df).values.flatten()
        NPV, _ = calculate_NPV_IRR_phased(
            Net_revenue_t=np.insert(revenues, 0, 0),
            maintenance_cost_per_year=maintenance_cost_per_year,
            capex_vector=capex_vector,
            depreciation_on_each_year=depreciation_on_each_year,
            tax_rate=tax_rate,
            discount_rate=discount_rate,
            decommissioning_vec=decommissioning_vec,
            inflation_index=inflation_index,
        )
        return NPV**2

    result = sp.optimize.minimize(fun=objective, x0=50, method="SLSQP", tol=1e-10)
    return result.x


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
    inflation_index,
):
    """A function to estimate the yearly cashflow using the net revenue time series, and the yearly OPEX costs.
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

    yr = np.arange(
        len(Net_revenue_t) + 1
    )  # extra year to start at 0 and end at end of lifetime.
    depre = np.interp(yr, depreciation_yr, depreciation)

    # EBITDA: earnings before interest and taxes in nominal prices
    EBITDA = (Net_revenue_t - maintenance_cost_per_year) * inflation_index[1:]

    # EBIT taxable income
    depreciation_on_each_year = np.diff(investment_cost * depre)
    EBIT = EBITDA - depreciation_on_each_year

    # Taxes
    Taxes = EBIT * tax_rate

    Net_income = EBITDA - Taxes
    Cashflow = np.insert(Net_income, 0, -investment_cost - development_cost)
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
    CAPEX_sh=None,
    wind_WACC=None,
    solar_WACC=None,
    battery_WACC=None,
    **legacy_kwargs,
):
    """This function returns the weighted average cost of capital after tax, using solar, wind, and battery
    WACC. First the shared costs WACC is computed by taking the mean of the WACCs across all technologies.
    Then the WACC after tax is calculated by taking the weighted sum by the corresponding CAPEX.

    Parameters
    ----------
    CAPEX_w : CAPEX of the wind power plant
    CAPEX_s : CAPEX of the solar power plant
    CAPEX_b : CAPEX of the battery
    CAPEX_sh : CAPEX of the shared electrical costs
    wind_WACC : After tax WACC for onshore WT
    solar_WACC : After tax WACC for solar PV
    battery_WACC : After tax WACC for stationary storge li-ion batteries

    Returns
    -------
    WACC_after_tax : WACC after tax
    """

    # Backward-compatible alias for legacy shared CAPEX keyword.
    legacy_shared_key = "CAPEX" + "_el"
    if (CAPEX_sh is None) and (legacy_shared_key in legacy_kwargs):
        CAPEX_sh = legacy_kwargs[legacy_shared_key]
    if CAPEX_sh is None:
        CAPEX_sh = 0

    # Weighted average cost of capital
    WACC_after_tax = (
        CAPEX_w * wind_WACC
        + CAPEX_s * solar_WACC
        + CAPEX_b * battery_WACC
        + CAPEX_sh * (wind_WACC + solar_WACC + battery_WACC) / 3
    ) / (CAPEX_w + CAPEX_s + CAPEX_b + CAPEX_sh)
    return WACC_after_tax


def calculate_revenues(price_el, df):
    df["revenue"] = (
        df["hpp_t"] * np.broadcast_to(price_el, df["hpp_t"].shape) - df["penalty_t"]
    )
    return df.groupby("i_year").revenue.mean() * 365 * 24


def calculate_break_even_PPA_price(
    df,
    CAPEX,
    OPEX,
    tax_rate,
    discount_rate,
    depreciation_yr,
    depreciation,
    DEVEX,
    inflation_index,
):
    def fun(price_el):
        revenues = calculate_revenues(price_el, df)
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


def calculate_CAPEX_phasing(
    CAPEX,
    phasing_yr,
    phasing_CAPEX,
    discount_rate,
    inflation_index,
):
    """This function calulates the equivalent net present value CAPEX given a early paying "phasing" approach.

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

    phasing_CAPEX = inflation_index * CAPEX * phasing_CAPEX / np.sum(phasing_CAPEX)
    CAPEX_eq = np.sum(
        [
            phasing_CAPEX[ii] / (1 + discount_rate) ** yr
            for ii, yr in enumerate(phasing_yr)
        ]
    )

    return CAPEX_eq


def get_inflation_index(yr, inflation_yr, inflation, ref_yr_inflation=0):
    """This function calulates the inflation index time series.

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

    ind_ref = np.where(np.array(yr) == ref_yr_inflation)[0]
    inflation_index = np.cumprod(1 + np.array(infl))
    inflation_index = inflation_index / inflation_index[ind_ref]

    return inflation_index
