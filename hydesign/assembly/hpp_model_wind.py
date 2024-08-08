import glob
import os
import time
import ast

import numpy as np
import pandas as pd
import xarray as xr

from .hpp_model_abc import hpp_model_abc
from .assembly_utils import mkdir
from hydesign.look_up_tables import lut_filepath
from hydesign.weather.weather import extract_weather_for_HPP, ABL, select_years
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, wpp_with_degradation, get_rotor_area, get_rotor_d
from hydesign.pv.pv import pvp, pvp_with_degradation
from hydesign.ems.ems import ems, ems_long_term_operation
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance.finance import finance

class hpp_model_wind(hpp_model_abc):
    def __init__(self, excel_data, sim_pars, verbose, weather, N_time, genWT_fn, N_ws, genWake_fn, price, **kwargs):
        self.sim_pars = sim_pars
        self.weather = weather
        self.N_time = N_time
        self.genWT_fn = genWT_fn
        self.N_ws = N_ws
        self.genWake_fn = genWake_fn
        self.price = price


    def add_subsystems(self, model):
        sim_pars = self.sim_pars
        weather = self.weather
        N_time = self.N_time
        genWT_fn = self.genWT_fn
        N_ws = self.N_ws
        genWake_fn = self.genWake_fn


        latitude = sim_pars.get('latitude')
        longitude = float(sim_pars.get('longitude'))
        altitude = sim_pars.get('altitude')
        wpp_efficiency = sim_pars['wpp_efficiency']
        wind_deg = sim_pars.get('wind_deg', [0, 0.25])
        wind_deg_yr = sim_pars.get('wind_deg_yr', [0, 25])
        share_WT_deg_types = sim_pars.get('share_WT_deg_types', 0.5)
        N_life = sim_pars['N_life']
        life_h = N_life * 365 * 24
        max_num_batteries_allowed = sim_pars.get('max_num_batteries_allowed', 3)
        factor_battery_cost = sim_pars.get('factor_battery_cost', 1)
        ems_type = sim_pars.get('ems_type', 'cplex')
        weeks_per_season_per_year = sim_pars.get('weeks_per_season_per_year')


        return model

    def connect_subsytems(self, model):
        model.connect('genericWT.ws', 'genericWake.ws')
        model.connect('genericWT.pc', 'genericWake.pc')
        model.connect('genericWT.ct', 'genericWake.ct')
        model.connect('genericWT.ws', 'wpp.ws')
        model.connect('genericWake.pcw', 'wpp.pcw')
        model.connect('abl.wst', 'wpp.wst')
        model.connect('genericWT.ws', 'wpp_with_degradation.ws')
        model.connect('genericWake.pcw', 'wpp_with_degradation.pcw')
        model.connect('abl.wst', 'wpp_with_degradation.wst')
        model.connect('wpp_with_degradation.wind_t_ext_deg', 'ems_long_term_operation.wind_t_ext_deg')
        model.connect('wpp.wind_t', 'ems.wind_t')
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('wpp.wind_t', 'wpp_cost.wind_t')
        model.connect('wpp_cost.CAPEX_w', 'finance.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance.OPEX_w')

        return model

    def set_prob_values(self, prob):
        sim_pars = self.sim_pars
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])

    def list_vars_and_outvars(self, list_vars, list_out_vars):
        added_list_out_vars = [
            'Hub height [m]',
            'Rotor diam [m]',
            'Wind CAPEX [MEuro]',
            'Wind OPEX [MEuro]',
            'Capacity factor wind [-]',
            'wind [MW]',
            'Awpp [km2]',
        ]

        added_list_vars = [
            'clearance [m]',
            'sp [W/m2]',
            'p_rated [MW]',
            'Nwt',
            'wind_MW_per_km2 [MW/km2]',
        ]

        list_vars = list_vars + added_list_vars
        list_out_vars = list_out_vars + added_list_out_vars

        return list_vars, list_out_vars

    def evalute_set_and_get_prob_values(self, model):
        pass

    def evalute_outputs(self, model):
        pass

    def evaluation_in_csv_design_df(self, model):
        pass
