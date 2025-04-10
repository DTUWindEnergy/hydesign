# %%

# basic libraries
import numpy as np
import openmdao.api as om
import os

from hydesign.weather.weather import ABL
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, wpp_with_degradation, get_rotor_d  # , get_rotor_area
from hydesign.pv.pv import pvp, pvp_with_degradation
from hydesign.ems.ems_hifi_dems import ems
from hydesign.ems.ems import ems_long_term_operation
from hydesign.battery_degradation import battery_degradation, battery_loss_in_capacity_due_to_temp
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost
from hydesign.finance.finance_hifi_ems import finance
from hydesign.reliability import battery_with_reliability, wpp_with_reliability, pvp_with_reliability
from hydesign.assembly.hpp_assembly import hpp_base
from hydesign.examples import examples_filepath

class hpp_model(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
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
        hpp_base.__init__(self,
                          sim_pars_fn=sim_pars_fn,
                          defaults={'input_ts_fn': os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_DA.csv'),},
                          **kwargs
                          )

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        wind_deg_yr = self.wind_deg_yr
        wind_deg = self.wind_deg
        share_WT_deg_types = self.share_WT_deg_types
        
        if sim_pars['data_dir'] is None:
            data_dir = examples_filepath
        else:
            data_dir = sim_pars['data_dir']

        input_ts_fn_da = os.path.join(data_dir, sim_pars['input_ts_da'])
        input_ts_fn_ha = os.path.join(data_dir, sim_pars['input_ts_ha'])
        input_ts_fn_rt = os.path.join(data_dir, sim_pars['input_ts_rt'])
        input_ts_fn_measurement = os.path.join(data_dir, sim_pars['input_ts_measurement'])
        market_fn = os.path.join(data_dir, sim_pars['market_fn'])
        
        sim_pars['input_ts_da'] = input_ts_fn_da
        sim_pars['input_ts_ha'] = input_ts_fn_ha
        sim_pars['input_ts_rt'] = input_ts_fn_rt
        sim_pars['input_ts_measurement'] = input_ts_fn_measurement
        sim_pars['market_fn'] = market_fn
        
        genWT_fn = sim_pars['genWT_fn']
        genWake_fn = sim_pars['genWake_fn']
        latitude = sim_pars['latitude']
        longitude = sim_pars['longitude']
        altitude = sim_pars['altitude']
        weeks_per_season_per_year = sim_pars['weeks_per_season_per_year']
        max_num_batteries_allowed = sim_pars['max_num_batteries_allowed']
        reliability_ts_battery = sim_pars['reliability_ts_battery']
        reliability_ts_trans = sim_pars['reliability_ts_trans']
        reliability_ts_wind = sim_pars['reliability_ts_wind']
        reliability_ts_pv = sim_pars['reliability_ts_pv']
        battery_price_reduction_per_year = sim_pars['battery_price_reduction_per_year']


        parameter_keys = ['hpp_grid_connection',
                          'wind_capacity',
                          'solar_capacity',
                          'battery_energy_capacity',
                          'battery_power_capacity',
                          'battery_minimum_SoC',
                          'battery_maximum_SoC',
                          'battery_initial_SoC',
                          'battery_hour_discharge_efficiency',
                          'battery_hour_charge_efficiency',
                          'battery_self_discharge_efficiency',
                          'battery_initial_degradation',
                          'battery_marginal_degradation_cost',
                          'battery_capital_cost',
                          'degradation_in_optimization',
                          'max_up_bid',
                          'max_dw_bid',
                          'min_up_bid',
                          'min_dw_bid',
                          'dispatch_interval',
                          'settlement_interval',
                          'imbalance_fee']
        simulation_keys = ['start_date',
                           'number_of_run_day',
                           'out_dir',
                           'SP',
                           'RP',
                           'BP']
        parameter_dict = {k: v for k, v in sim_pars.items() if k in parameter_keys}
        simulation_dict = {k: v for k, v in sim_pars.items() if k in simulation_keys}

        parameter_dict.update({'battery_initial_degradation': 0,  # hpp battery degradation parameters
                               'degradation_in_optimization': 0, }) # 1:yes 0:no
        
        markets = ['da', 'ha', 'rt', 'measurement']
        
        model = om.Group()
        
        for market in markets:
            model.add_subsystem(
                f'abl_{market}', 
                ABL(
                    weather_fn=sim_pars[f'input_ts_{market}'], 
                    N_time=N_time),
                promotes_inputs=['hh']
                )
            model.add_subsystem(
                f'genericWT_{market}', 
                genericWT_surrogate(
                    genWT_fn=genWT_fn,
                    N_ws = N_ws),
                promotes_inputs=[
                   'hh',
                   'd',
                   'p_rated',
                ])
            model.add_subsystem(
                f'genericWake_{market}', 
                genericWake_surrogate(
                    genWake_fn=genWake_fn,
                    N_ws = N_ws),
                promotes_inputs=[
                    'Nwt',
                    'Awpp',
                    'd',
                    'p_rated',
                    ])
            model.add_subsystem(
                f'wpp_{market}', 
                wpp(
                    N_time = N_time,
                    N_ws = N_ws,
                    wpp_efficiency = wpp_efficiency,)
                    )
            model.add_subsystem(
                f'pvp_{market}', 
                pvp(
                    weather_fn = sim_pars[f'input_ts_{market}'], 
                    N_time = N_time,
                    latitude = latitude,
                    longitude = longitude,
                    altitude = altitude,
                    tracking = sim_pars['tracking']
                   ),
                promotes_inputs=[
                    'surface_tilt',
                    'surface_azimuth',
                    'DC_AC_ratio',
                    'solar_MW',
                    'land_use_per_solar_MW',
                    ])

        model.add_subsystem(
            'ems', 
            ems(parameter_dict=parameter_dict,
                simulation_dict=simulation_dict,
                market_fn=market_fn,
                N_time = N_time,
                ),
            promotes_inputs=[
                'G_MW',
                'wind_MW',
                'solar_MW',
                'b_E',
                'battery_depth_of_discharge',
                'b_P',
                ],
            promotes_outputs=[
                'hpp_t',
                'P_HPP_SM_t_opt',
                'SM_price_cleared',
                'BM_dw_price_cleared',
                'BM_up_price_cleared',
                'P_HPP_RT_refs',
                'P_HPP_UP_bid_ts',
                'P_HPP_DW_bid_ts',
                's_UP_t',
                's_DW_t',
                'residual_imbalance',
                'P_HPP_ts',
                'P_curtailment_ts',
                'P_charge_discharge_ts',
                'E_SOC_ts',
                ]
            )
        model.add_subsystem(
            'battery_degradation', 
            battery_degradation(
                weather_fn = input_ts_fn_rt, # for extracting temperature
                num_batteries = max_num_batteries_allowed,
                intervals_per_hour=4,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            promotes_inputs=[
                'min_LoH'
                ])

        model.add_subsystem(
            'battery_loss_in_capacity_due_to_temp', 
            battery_loss_in_capacity_due_to_temp(
                weather_fn = input_ts_fn_rt, # for extracting temperature
                intervals_per_hour=4,
                weeks_per_season_per_year = weeks_per_season_per_year,
            ),
            )

        model.add_subsystem(
            'wpp_with_degradation', 
            wpp_with_degradation(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,
                intervals_per_hour=4,
                wind_deg_yr = wind_deg_yr,
                wind_deg = wind_deg,
                share_WT_deg_types = share_WT_deg_types,
                weeks_per_season_per_year = weeks_per_season_per_year,
                
            )
        )

        model.add_subsystem(
            'pvp_with_degradation', 
            pvp_with_degradation(
                intervals_per_hour=4,
                pv_deg_yr = sim_pars['pv_deg_yr'],
                pv_deg = sim_pars['pv_deg'],
                )
            )

        
        model.add_subsystem(
            'battery_with_reliability', 
            battery_with_reliability(
                intervals_per_hour=4,
                reliability_ts_battery=reliability_ts_battery,
                reliability_ts_trans=reliability_ts_trans,
                ),
            )        
        

        model.add_subsystem(
            'wpp_with_reliability', 
            wpp_with_reliability(
                intervals_per_hour=4,
                reliability_ts_wind=reliability_ts_wind,
                reliability_ts_trans=reliability_ts_trans,
                ),
            )        
        

        model.add_subsystem(
            'pvp_with_reliability', 
            pvp_with_reliability(
                intervals_per_hour=4,
                reliability_ts_pv=reliability_ts_pv,
                reliability_ts_trans=reliability_ts_trans,
                ),
            )        
        
        model.add_subsystem(
            'ems_long_term_operation', 
            ems_long_term_operation(
                N_time = N_time,
                intervals_per_hour=4,
                ),
            promotes_inputs=[
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile',
                'n_full_power_hours_expected_per_day_at_peak_price'
                ],
            promotes_outputs=[
                'total_curtailment',
                'total_curtailment_with_deg'
                ])
        
        model.add_subsystem(
            'wpp_cost',
            wpp_cost(
                wind_turbine_cost=sim_pars['wind_turbine_cost'],
                wind_civil_works_cost=sim_pars['wind_civil_works_cost'],
                wind_fixed_onm_cost=sim_pars['wind_fixed_onm_cost'],
                wind_variable_onm_cost=sim_pars['wind_variable_onm_cost'],
                d_ref=sim_pars['d_ref'],
                hh_ref=sim_pars['hh_ref'],
                p_rated_ref=sim_pars['p_rated_ref'],
                N_time = N_time, 
                intervals_per_hour=4,
                ),
            promotes_inputs=[
                'Nwt',
                'Awpp',
                'hh',
                'd',
                'p_rated'])
        model.add_subsystem(
            'pvp_cost',
            pvp_cost(
                solar_PV_cost=sim_pars['solar_PV_cost'],
                solar_hardware_installation_cost=sim_pars['solar_hardware_installation_cost'],
                solar_inverter_cost=sim_pars['solar_inverter_cost'],
                solar_fixed_onm_cost=sim_pars['solar_fixed_onm_cost'],
            ),
            promotes_inputs=['solar_MW', 'DC_AC_ratio'])

        model.add_subsystem(
            'battery_cost',
            battery_cost(
                battery_energy_cost=sim_pars['battery_energy_cost'],
                battery_power_cost=sim_pars['battery_power_cost'],
                battery_BOP_installation_commissioning_cost=sim_pars['battery_BOP_installation_commissioning_cost'],
                battery_control_system_cost=sim_pars['battery_control_system_cost'],
                battery_energy_onm_cost=sim_pars['battery_energy_onm_cost'],
                intervals_per_hour=4,
                battery_price_reduction_per_year = battery_price_reduction_per_year,
            ),
            promotes_inputs=[
                'b_P',
                'b_E',
                ])

        model.add_subsystem(
            'shared_cost',
            shared_cost(
                hpp_BOS_soft_cost=sim_pars['hpp_BOS_soft_cost'],
                hpp_grid_connection_cost=sim_pars['hpp_grid_connection_cost'],
                land_cost=sim_pars['land_cost'],
            ),
            promotes_inputs=[
                'G_MW',
                'Awpp',
            ])

        model.add_subsystem(
            'finance', 
            finance(
                parameter_dict=parameter_dict,
                depreciation_yr = sim_pars['depreciation_yr'],
                depreciation = sim_pars['depreciation'],
                inflation_yr = sim_pars['inflation_yr'],
                inflation = sim_pars['inflation'],
                ref_yr_inflation = sim_pars['ref_yr_inflation'],
                phasing_yr = sim_pars['phasing_yr'],
                phasing_CAPEX = sim_pars['phasing_CAPEX'],
                ),
            promotes_inputs=['wind_WACC',
                             'solar_WACC', 
                             'battery_WACC',
                             'tax_rate',
                              'P_HPP_SM_t_opt',
                              'SM_price_cleared',
                              'BM_dw_price_cleared',
                              'BM_up_price_cleared',
                              'P_HPP_RT_refs',
                              'P_HPP_UP_bid_ts',
                              'P_HPP_DW_bid_ts',
                              's_UP_t',
                              's_DW_t',
                              'residual_imbalance',
                              'P_HPP_ts',
                              'P_curtailment_ts',
                              'P_charge_discharge_ts',
                              'E_SOC_ts',
                            ],
            promotes_outputs=['NPV',
                              'IRR',
                              'NPV_over_CAPEX',
                              'LCOE',
                              'revenues',
                              'mean_AEP',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX',
                              'break_even_PPA_price',
                              ],
        )
                  
        for market in markets:
            model.connect(f'abl_{market}.wst', f'wpp_{market}.wst')

            model.connect(f'genericWT_{market}.ws', f'genericWake_{market}.ws')
            model.connect(f'genericWT_{market}.ws', f'wpp_{market}.ws')
            model.connect(f'genericWT_{market}.pc', f'genericWake_{market}.pc')
            model.connect(f'genericWT_{market}.ct', f'genericWake_{market}.ct')

            model.connect(f'genericWake_{market}.pcw', f'wpp_{market}.pcw')

            model.connect(f'wpp_{market}.wind_t', f'ems.wind_t_{market}')

            model.connect(f'pvp_{market}.solar_t', f'ems.solar_t_{market}')

        model.connect('abl_rt.wst', 'wpp_with_degradation.wst')
        model.connect('genericWT_rt.ws', 'wpp_with_degradation.ws')
        model.connect('genericWake_rt.pcw', 'wpp_with_degradation.pcw')

        model.connect('wpp_rt.wind_t', 'wpp_cost.wind_t')

        model.connect('pvp_rt.Apvp', 'shared_cost.Apvp')
        

        
        model.connect('ems.b_E_SOC_t', 'battery_degradation.b_E_SOC_t')
        
        model.connect('battery_degradation.SoH', 'battery_loss_in_capacity_due_to_temp.SoH')
        model.connect('battery_loss_in_capacity_due_to_temp.SoH_all', 'ems_long_term_operation.SoH')
        

        model.connect('wpp_with_degradation.wind_t_ext_deg', 'wpp_with_reliability.wind_t')
        model.connect('wpp_with_reliability.wind_t_rel', 'ems_long_term_operation.wind_t_ext_deg')

        model.connect('ems.solar_t_ext','pvp_with_degradation.solar_t_ext')
        model.connect('pvp_with_degradation.solar_t_ext_deg', 'pvp_with_reliability.solar_t')
        model.connect('pvp_with_reliability.solar_t_rel', 'ems_long_term_operation.solar_t_ext_deg')
        
        model.connect('ems.wind_t_ext', 'ems_long_term_operation.wind_t_ext')
        model.connect('ems.solar_t_ext', 'ems_long_term_operation.solar_t_ext')
        model.connect('ems.hpp_curt_t', 'ems_long_term_operation.hpp_curt_t')
        model.connect('ems.b_E_SOC_t', 'ems_long_term_operation.b_E_SOC_t')
        model.connect('ems.b_t', 'battery_with_reliability.b_t')
        model.connect('battery_with_reliability.b_t_rel', 'ems_long_term_operation.b_t')

        
        model.connect('battery_degradation.SoH','battery_cost.SoH')
        
        
        model.connect('wpp_cost.CAPEX_w', 'finance.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance.OPEX_w')

        model.connect('pvp_cost.CAPEX_s', 'finance.CAPEX_s')
        model.connect('pvp_cost.OPEX_s', 'finance.OPEX_s')

        model.connect('battery_cost.CAPEX_b', 'finance.CAPEX_b')
        model.connect('battery_cost.OPEX_b', 'finance.OPEX_b')

        model.connect('shared_cost.CAPEX_sh', 'finance.CAPEX_el')
        model.connect('shared_cost.OPEX_sh', 'finance.OPEX_el')

        model.connect('ems_long_term_operation.hpp_t_with_deg', 'finance.hpp_t_with_deg')
        
        prob = om.Problem(
            model,
            reports=None
        )

        prob.setup()        
        
        # Additional parameters
        prob.set_val('G_MW', sim_pars['G_MW'])
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])      
        prob.set_val('min_LoH', sim_pars['min_LoH'])
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])

        

        self.prob = prob
    
        self.list_out_vars = [
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
            ]

        self.list_vars = [
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
            ]   
    
    
    def evaluate(
        self,
        # Wind plant design
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, **kwargs
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
        prob.set_val('wind_MW', wind_MW)
        
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

if __name__ == '__main__':
    sim_pars_fn = os.path.join(examples_filepath, 'Europe/hpp_pars_HiFiEMS.yml')
    hpp = hpp_model(sim_pars_fn=sim_pars_fn,
                    input_ts_da=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_DA.csv'),
                    input_ts_ha=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_HA.csv'),
                    input_ts_rt=os.path.join(examples_filepath, 'HiFiEMS_inputs/Weather/input_ts_RT.csv'),
                    market_fn=os.path.join(examples_filepath, 'HiFiEMS_inputs/Market/Market2021.csv'),)
    inputs = dict(clearance=20, 
                  sp=350,
                  p_rated=10, 
                  Nwt=12, 
                  wind_MW_per_km2=6,
                  solar_MW=10,
                  surface_tilt=25, 
                  surface_azimuth=180, 
                  DC_AC_ratio=1.5,
                  b_P=40,
                  b_E_h=3,
                  )

    res = hpp.evaluate(**inputs)
    hpp.print_design(list(inputs.values()), res)
    om.n2(hpp.prob)
    
   