# %%
import glob
import os
import time
import ast

# basic libraries
import numpy as np
from numpy import newaxis as na
import numpy_financial as npf
import pandas as pd
# import seaborn as sns
import openmdao.api as om
import yaml
import scipy as sp
from scipy import stats
import xarray as xr

from hydesign.weather.weather import extract_weather_for_HPP, ABL, select_years
from hydesign.wind.wind import get_rotor_area, get_rotor_d
from hydesign.ems.ems import ems, ems_long_term_operation
from hydesign.ems.ems_builder import ems_builder, ems_long_term_operation_builder
from hydesign.look_up_tables import lut_filepath
from .hpp_model_base import hpp_model_base
from .hpp_model_wind import hpp_model_wind
from .hpp_model_solar import hpp_model_solar
from .hpp_model_bess import hpp_model_bess

from hydesign.ems.ems_wind import ems_wind_abc, ems_wind_long_term_operation
from hydesign.ems.ems_solar import ems_solar_abc, ems_solar_long_term_operation
from hydesign.ems.ems_bess import ems_bess_abc, ems_bess_long_term_operation
from hydesign.ems.ems_base import ems_base_abc, ems_base_long_term_operation


class hpp_model:
    """HPP design evaluator"""

    def __init__(
        self,
        excel_data,
        verbose = True,
        **kwargs
        ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        latitude : Latitude at chosen location
        longitude : Longitude at chosen location
        altitude : Altitude at chosen location, if not provided, elevation is calculated using elevation map datasets
        sims_pars_fn : Case study input values of the HPP
        work_dir : Working directory path
        max_num_batteries_allowed : Maximum number of batteries allowed including start and replacements
        weeks_per_season_per_year: Number of weeks per season to select from the input data, to reduce computation time. Default is `None` which uses all the input time series
        seed: seed number for week selection
        ems_type : Energy management system optimization type: cplex solver or rule based
        inputs_ts_fn : User provided weather timeseries, if not provided, the weather data is calculated using ERA5 datasets
        price_fn : Price timeseries
        era5_zarr : Location of wind speed renalysis
        ratio_gwa_era5 : Location of mean wind speed correction factor
        era5_ghi_zarr : Location of GHI renalysis
        elevation_fn : Location of GHI renalysis
        genWT_fn : Wind turbine power curve look-up tables
        genWake_fn : Wind turbine wake look-up tables
        """
        excel_data_main = excel_data['dashboard']

        # Reset the header to the 6th row (index 5) and reassign the DataFrame
        excel_data_main.columns = excel_data_main.iloc[5]
        excel_data_main = excel_data_main[6:].reset_index(drop=True)

        # Drop rows where 'Parameter' is NaN
        excel_data_main.dropna(subset=['Parameter'], inplace=True)

        # Extract 'Parameter' and 'Value' columns as a dictionary
        sim_pars = dict(zip(excel_data_main['Parameter'], excel_data_main['Value']))

        # Convert specific sim_pars to np.array using ast.literal_eval
        for key in ['wind_deg', 'wind_deg_yr', 'pv_deg', 'pv_deg_yr', 'inflation', 'inflation_yr', 'phasing_yr', 'phasing_CAPEX', 'depreciation_yr',
                    'depreciation', 'technology_types']:
            if key in sim_pars:
                sim_pars[key] = np.array(ast.literal_eval(sim_pars[key]))

        # Overwrite parameters with values from kwargs if provided
        for key in sim_pars.keys():
            if key in kwargs:
                sim_pars[key] = kwargs[key]

        # sort the technology_types based on the default order
        sorted_tech_types = ['wind', 'solar', 'bess', 'p2x', 'sm', 'bm']
        sim_pars['technology_types'] = [tech_type for tech_type in sorted_tech_types if tech_type in sim_pars['technology_types']]

        # Extract sim_pars from the dictionary
        latitude = sim_pars.get('latitude')
        longitude = float(sim_pars.get('longitude'))
        altitude = sim_pars.get('altitude')
        work_dir = sim_pars.get('work_dir', './')
        max_num_batteries_allowed = sim_pars.get('max_num_batteries_allowed', 3)
        factor_battery_cost = sim_pars.get('factor_battery_cost', 1)
        ems_type = sim_pars.get('ems_type', 'cplex')
        weeks_per_season_per_year = sim_pars.get('weeks_per_season_per_year')
        seed = sim_pars.get('seed', 0)
        genWT_fn = sim_pars.get('genWT_fn', lut_filepath + './genWT_v3.nc')
        genWake_fn = sim_pars.get('genWake_fn', lut_filepath + './genWake_v3.nc')
        verbose = verbose
        name = sim_pars.get('name', '')
        ppa_price = sim_pars.get('ppa_price')
        ems_type = sim_pars.get('ems_type', 'cplex')
        technology_types = sim_pars.get('technology_types')

        work_dir = mkdir(work_dir)

        # Extract additional data from separate sheets if available
        input_ts_data = excel_data.get('input_ts_DA')

        if altitude is None:
            elevation_fn = sim_pars['elevation_fn']  # Altitude map for extracting altitude
            elevation_ds = xr.open_dataset(elevation_fn)
            altitude = elevation_ds['elev'].interp(
                latitude=latitude,
                longitude=longitude,
                kwargs={"fill_value": 0.0}
            ).values
        if verbose:
            print('\nFixed parameters on the site')
            print('-------------------------------')
            print('longitude =', longitude)
            print('latitude =', latitude)
            print('altitude =', altitude)

        # Parameters of the simulation
        year_start = sim_pars.get('year_start', sim_pars['year'])
        year_end = sim_pars.get('year_end', sim_pars['year'])
        N_life = sim_pars['N_life']
        life_h = N_life * 365 * 24
        G_MW = sim_pars['G_MW']
        battery_depth_of_discharge = sim_pars['battery_depth_of_discharge']
        battery_charge_efficiency = sim_pars['battery_charge_efficiency']
        min_LoH = sim_pars['min_LoH']
        wpp_efficiency = sim_pars['wpp_efficiency']
        land_use_per_solar_MW = sim_pars['land_use_per_solar_MW']

        wind_deg = sim_pars.get('wind_deg', [0, 0.25])
        wind_deg_yr = sim_pars.get('wind_deg_yr', [0, 25])
        share_WT_deg_types = sim_pars.get('share_WT_deg_types', 0.5)
        self.ems_type = sim_pars.get('ems_type', 'cplex')

        # Extract weather timeseries
        if input_ts_data.empty:
            # Weather database
            era5_zarr = sim_pars['era5_zarr']  # location of wind speed renalysis
            ratio_gwa_era5 = sim_pars['ratio_gwa_era5']  # location of mean wind speed correction factor
            era5_ghi_zarr = sim_pars['era5_ghi_zarr']  # location of GHI renalysis

            weather = extract_weather_for_HPP(
                longitude=longitude,
                latitude=latitude,
                altitude=altitude,
                era5_zarr=era5_zarr,
                ratio_gwa_era5=ratio_gwa_era5,
                era5_ghi_zarr=era5_ghi_zarr,
                year_start=year_start,
                year_end=year_end
            )

            price_fn = sim_pars.get('price_fn')
            if type(price_fn) is str:
                price = pd.read_csv(price_fn, index_col=0, parse_dates=True)
            else:
                price = price_fn
            try:
                weather['Price'] = price.loc[weather.index].bfill()
            except Exception as e:
                raise Exception('Price timeseries does not match the weather')

            # Check for complete years in the input_ts: Hydesign works with years of 365 days
            N_time = len(weather)
            if np.mod(N_time, 365 * 24) == 0:
                pass
            else:
                N_sel = N_time - np.mod(N_time, 365 * 24)
                weather = weather.iloc[:N_sel]

            input_ts_fn = f'{work_dir}/input_ts{name}.csv'
            if verbose:
                print(f'\ninput_ts_fn extracted and stored in {input_ts_fn}')
            #weather.to_csv(input_ts_fn, index=True)
            N_time = len(weather)

        else:  # User provided weather timeseries
            weather = excel_data['input_ts_DA']
            weather.set_index(input_ts_data.columns[0], inplace=True)
            weather.index = pd.to_datetime(weather.index)
            N_time = len(weather)

        # Check for complete years in the input_ts
        if np.mod(N_time, 365 * 24) == 0:
            pass
        else:
            N_sel = N_time - np.mod(N_time, 365 * 24)
            weather = weather.iloc[:N_sel]

            if verbose:
                print(f'\ninput_ts_fn length is not a complete number of years (hyDesign handles years as 365 days).')
                print(f'The file has been modified accordingly')
            N_time = len(weather)

        # Assign PPA to the full input_ts
        if ppa_price is None:
            price = weather['Price']
        else:
            price = ppa_price * np.ones_like(weather['Price'])

        # Randomly sample the weather to generate representative years
        if weeks_per_season_per_year is not None:
            weather = select_years(
                weather,
                seed=seed,
                weeks_per_season_per_year=weeks_per_season_per_year,
            )
            N_time = len(weather)
            input_ts_fn = f'{work_dir}/input_ts_sel.csv'
            if verbose:
                print(f'\n\nSelected input time series based on {weeks_per_season_per_year} weeks per season are stored in {input_ts_fn}')

        # extract number of ws in the look-up tables
        with xr.open_dataset(genWT_fn) as ds:
            # number of points in the power curves
            N_ws = len(ds.ws.values)

        # run hpp models
        hpp_model_map = {
            'base': hpp_model_base,
            'wind': hpp_model_wind,
            'solar': hpp_model_solar,
            'bess': hpp_model_bess,
        }

        self.ems_class_map = {
            'base': ems_base_abc,
            'wind': ems_wind_abc,
            'solar': ems_solar_abc,
            'bess': ems_bess_abc
        }

        self.ems_long_class_map = {
            'base': ems_base_long_term_operation,
            'wind': ems_wind_long_term_operation,
            'solar': ems_solar_long_term_operation,
            'bess': ems_bess_long_term_operation
        }
        self.ems_long_type = 'energy_penalty'

        technology_types_and_base = ['base'] + ['wind', 'solar', 'bess'] #+ technology_types
        self.technology_types_and_base = technology_types_and_base
        hpp_model_instances = {}
        for tech_type in technology_types_and_base:
            hpp_model_instances[tech_type] = hpp_model_map[tech_type](excel_data, sim_pars, verbose, weather, N_time, genWT_fn, N_ws, genWake_fn, price, **kwargs)

        model = om.Group()

        list_subsystems = {}
        for tech_type in ['base']: # technology_types_and_base:
            list_subsystems = hpp_model_instances[tech_type].add_subsystems(list_subsystems)

        # default order of subsystems
        list_subsystems_default = ['abl_dict', 'genericWT_dict', 'genericWake_dict', 'wpp_dict', 'pvp_dict', 'ems_dict', 'battery_degradation_dict',
                           'wpp_with_degradation_dict',
                           'pvp_with_degradation_dict',
                           'battery_loss_in_capacity_due_to_temp_dict',
                           'ems_long_term_operation_dict', 'wpp_cost_dict', 'pvp_cost_dict', 'battery_cost_dict',
                           'shared_cost_dict', 'finance_dict']

        list_subsystems_sorted = [list_subsystems[sub] for sub in list_subsystems_default if list_subsystems.get(sub, False)]

        for sub in list_subsystems_sorted:
            model.add_subsystem(sub['name'], sub['component'], promotes_inputs=sub['promotes_inputs'], promotes_outputs=sub['promotes_outputs'])

        for tech_type in technology_types_and_base:
            model = hpp_model_instances[tech_type].connect_subsytems(model)

        prob = om.Problem(model, reports=None)

        prob.setup()

        for tech_type in technology_types_and_base:
            hpp_model_instances[tech_type].set_prob_values(prob)

        self.sim_pars = sim_pars
        self.prob = prob

        list_vars = []
        list_out_vars = []
        for tech_type in technology_types_and_base:
            list_vars, list_out_vars = hpp_model_instances[tech_type].list_vars_and_outvars(list_vars, list_out_vars)

        self.list_vars = list_vars
        self.list_out_vars = list_out_vars

        self.available_ems_types = {'cplex': 'ems_cplex'}

    def build(self):
        ems_builder(self)
        ems_long_term_operation_builder(self)


    def evaluate(
        self,
        # Wind plant design
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio
        ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        clearance : Distance from the ground to the tip of the blade [m]
        sp : Specific power of the turbine [W/m2]
        p_rated : Rated powe of the turbine [MW]
        Nwt : Number of wind turbines
        wind_MW_per_km2 : Wind power installation density [MW/km2]
        solar_MW : Solar AC capacity [MW]
        surface_tilt : Surface tilt of the PV panels [deg]
        surface_azimuth : Surface azimuth of the PV panels [deg]
        DC_AC_ratio : DC  AC ratio
        b_P : Battery power [MW]
        b_E_h : Battery storage duration [h]
        cost_of_battery_P_fluct_in_peak_price_ratio : Cost of battery power fluctuations in peak price ratio [Eur]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        self.num_batteries : Number of allowed replacements of the battery
        """

        prob = self.prob

        d = get_rotor_d(p_rated*1e6/sp)
        hh = (d/2)+clearance
        wind_MW = Nwt * p_rated
        Awpp = wind_MW / wind_MW_per_km2
        #Awpp = Awpp + 1e-10*(Awpp==0)
        b_E = b_E_h * b_P

        # pass design variables
        prob.set_val('hh', hh)
        prob.set_val('d', d)
        prob.set_val('p_rated', p_rated)
        prob.set_val('Nwt', Nwt)
        prob.set_val('Awpp', Awpp)

        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)

        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)

        prob.run_model()

        self.prob = prob

        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = prob.get_val('wpp_with_degradation.wind_t_ext_deg').mean() / p_rated / Nwt  # Capacity factor of wind only

        return np.hstack([
            prob['NPV_over_CAPEX'],
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['revenues']/1e6,
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob.get_val('finance.CAPEX_w')/1e6,
            prob.get_val('finance.OPEX_w')/1e6,
            prob.get_val('finance.CAPEX_s')/1e6,
            prob.get_val('finance.OPEX_s')/1e6,
            prob.get_val('finance.CAPEX_b')/1e6,
            prob.get_val('finance.OPEX_b')/1e6,
            prob.get_val('finance.CAPEX_el')/1e6,
            prob.get_val('finance.OPEX_el')/1e6,
            prob['penalty_lifetime']/1e6,
            prob['mean_AEP']/1e3, #[GWh]
            # Grid Utilization factor
            prob['mean_AEP']/(self.sim_pars['G_MW']*365*24),
            self.sim_pars['G_MW'],
            wind_MW,
            solar_MW,
            b_E,
            b_P,
            prob['total_curtailment']/1e3, #[GWh]
            prob['total_curtailment_with_deg']/1e3, #[GWh]
            Awpp,
            prob.get_val('shared_cost.Apvp'),
            max( Awpp , prob.get_val('shared_cost.Apvp') ),
            d,
            hh,
            prob.get_val('battery_degradation.n_batteries') * (b_P>0),
            prob['break_even_PPA_price'],
            cf_wind,
            ])

    def print_design(self, x_opt, outs):
        print()
        print('Design:')
        print('---------------')

        for i_v, var in enumerate(self.list_vars):
                print(f'{var}: {x_opt[i_v]:.3f}')
        print()
        print()
        for i_v, var in enumerate(self.list_out_vars):
            print(f'{var}: {outs[i_v]:.3f}')
        print()


    def evaluation_in_csv(self, name_file ,longitude, latitude, altitude, x_opt, outs ):
        design_df = pd.DataFrame(columns = ['longitude',
                                            'latitude',
                                            'altitude',
                                            'clearance [m]',
                                            'sp [W/m2]',
                                            'p_rated [MW]',
                                            'Nwt',
                                            'wind_MW_per_km2 [MW/km2]',
                                            'solar_MW [MW]',
                                            'surface_tilt [deg]',
                                            'surface_azimuth [deg]',
                                            'DC_AC_ratio',
                                            'b_P [MW]',
                                            'b_E_h [h]',
                                            'cost_of_battery_P_fluct_in_peak_price_ratio',
                                            'NPV_over_CAPEX',
                                            'NPV [MEuro]',
                                            'IRR',
                                            'LCOE [Euro/MWh]',
                                            'Revenues [MEuro]',
                                            'CAPEX [MEuro]',
                                            'OPEX [MEuro]',
                                            'Wind CAPEX [MEuro]',
                                            'Wind OPEX [MEuro]',
                                            'PV CAPEX [MEuro]',
                                            'PV OPEX [MEuro]',
                                            'Batt CAPEX [MEuro]',
                                            'Batt OPEX [MEuro]',
                                            'Shared CAPEX [MEuro]',
                                            'Shared OPEX [MEuro]',
                                            'penalty lifetime [MEuro]',
                                            'AEP [GWh]',
                                            'GUF',
                                            'grid [MW]',
                                            'wind [MW]',
                                            'solar [MW]',
                                            'Battery Energy [MWh]',
                                            'Battery Power [MW]',
                                            'Total curtailment [GWh]',
                                            'Total curtailment with deg [GWh]',
                                            'Awpp [km2]',
                                            'Apvp [km2]',
                                            'Plant area [km2]',
                                            'Rotor diam [m]',
                                            'Hub height [m]',
                                            'Number of batteries used in lifetime',
                                            'Break-even PPA price [Euro/MWh]',
                                            'Capacity factor wind [-]'
                                            ]  , index=range(1))
        design_df.iloc[0] =  [longitude,latitude,altitude] + list(x_opt) + list(outs)
        design_df.to_csv(f'{name_file}.csv')


# -----------------------------------------------------------------------
# Auxiliar functions for ems modelling
# -----------------------------------------------------------------------

def mkdir(dir_):
    if str(dir_).startswith('~'):
        dir_ = str(dir_).replace('~', os.path.expanduser('~'))
    try:
        os.stat(dir_)
    except BaseException:
        try:
            os.mkdir(dir_)
            #Path(dir_).mkdir(parents=True, exist_ok=True)
        except BaseException:
            pass
    return dir_

