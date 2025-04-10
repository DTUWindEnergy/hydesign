# %%
import os
import numpy as np
import pandas as pd
import openmdao.api as om

from hydesign.weather.weather import ABL
from hydesign.wind.wind import genericWT_surrogate, genericWake_surrogate, wpp, get_rotor_d # , wpp_with_degradation, get_rotor_area
from hydesign.pv.pv import pvp #, pvp_with_degradation
from hydesign.ems.ems_P2X_bidirectional import ems_P2X_bidirectional as ems
from hydesign.costs.costs import wpp_cost, pvp_cost, battery_cost, shared_cost, ptg_cost
from hydesign.finance.finance_P2X_bidirectional import finance_P2X_bidirectional as finance
from hydesign.assembly.hpp_assembly import hpp_base


class hpp_model_P2X_bidirectional(hpp_base):
    """HPP design evaluator"""

    def __init__(
        self,
        sim_pars_fn,
        H2_demand_fn = None, # If input_ts_fn is given it should include H2_demand column.
        **kwargs

        ):
        """Initialization of the hybrid power plant evaluator

        Parameters
        ----------
        sims_pars_fn : Case study input values of the HPP 
        H2_demand_fn : H2 demand time series file path
        """
        defaults = {'electrolyzer_eff_curve_name': 'PEM electrolyzer H2 production',
                    'electrolyzer_eff_curve_type': 'production',}
        hpp_base.__init__(self,
                          sim_pars_fn=sim_pars_fn,
                          defaults=defaults,
                          **kwargs
                          )

        N_time = self.N_time
        N_ws = self.N_ws
        wpp_efficiency = self.wpp_efficiency
        sim_pars = self.sim_pars
        # life_h = self.life_h
        # N_life = self.N_life
        life_y = self.life_y
        price = self.price
        
        input_ts_fn = sim_pars['input_ts_fn']
        genWT_fn = sim_pars['genWT_fn']
        genWake_fn = sim_pars['genWake_fn']
        latitude = sim_pars['latitude']
        longitude = sim_pars['longitude']
        altitude = sim_pars['altitude']
        ems_type = sim_pars['ems_type']
        
        weather = pd.read_csv(input_ts_fn, index_col=0, parse_dates=True)
        H2_demand_data = pd.read_csv(H2_demand_fn, index_col=0, parse_dates=True).loc[weather.index,:]
        electrolyzer_eff_fn = os.path.join(os.path.dirname(sim_pars_fn), 'Electrolyzer_efficiency_curves.csv')
        df = pd.read_csv(electrolyzer_eff_fn)
        electrolyzer_eff_curve_name = sim_pars['electrolyzer_eff_curve_name']
        col_no = df.columns.get_loc(electrolyzer_eff_curve_name)
        my_df = df.iloc[:,col_no:col_no+2].dropna()
        eff_curve = my_df[1:].values.astype(float)
        electrolyzer_eff_curve_type = sim_pars['electrolyzer_eff_curve_type']
        
        model = om.Group()
        
        model.add_subsystem(
            'abl', 
            ABL(
                weather_fn=input_ts_fn, 
                N_time=N_time),
            promotes_inputs=['hh']
            )
        model.add_subsystem(
            'genericWT', 
            genericWT_surrogate(
                genWT_fn=genWT_fn,
                N_ws = N_ws),
            promotes_inputs=[
               'hh',
               'd',
               'p_rated',
            ])
        
        model.add_subsystem(
            'genericWake', 
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
            'wpp', 
            wpp(
                N_time = N_time,
                N_ws = N_ws,
                wpp_efficiency = wpp_efficiency,)
                )
        
        model.add_subsystem(
            'pvp', 
            pvp(
                weather_fn = input_ts_fn, 
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
            ems(
                N_time = N_time,
                eff_curve=eff_curve,
                # life_h = life_h, 
                ems_type=ems_type,
                electrolyzer_eff_curve_type=electrolyzer_eff_curve_type,
                ),
            promotes_inputs=[
                'price_t',
                'b_P',
                'b_E',
                'G_MW',
                'battery_depth_of_discharge',
                'battery_charge_efficiency',
                'peak_hr_quantile',
                'cost_of_battery_P_fluct_in_peak_price_ratio',
                'n_full_power_hours_expected_per_day_at_peak_price',
                'price_H2',
                'ptg_MW',
                'HSS_kg',
                'storage_eff',
                'ptg_deg',
                'hhv',
                'm_H2_demand_t',
                'penalty_factor_H2',
                'min_power_standby',
                ],
            promotes_outputs=[
                'total_curtailment'
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
                # N_life = N_life,
                # life_h = life_h
                life_y = life_y,
                battery_price_reduction_per_year = sim_pars['battery_price_reduction_per_year']
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
            'ptg_cost',
            ptg_cost(
                electrolyzer_capex_cost = sim_pars['electrolyzer_capex_cost'],
                electrolyzer_opex_cost = sim_pars['electrolyzer_opex_cost'],
                electrolyzer_power_electronics_cost = sim_pars['electrolyzer_power_electronics_cost'],
                water_cost = sim_pars['water_cost'],
                water_treatment_cost = sim_pars['water_treatment_cost'],
                water_consumption = sim_pars['water_consumption'],
                storage_capex_cost = sim_pars['H2_storage_capex_cost'],
                storage_opex_cost = sim_pars['H2_storage_opex_cost'],
                transportation_cost = sim_pars['H2_transportation_cost'],
                transportation_distance = sim_pars['H2_transportation_distance'],
                N_time = N_time,
                # life_h = life_h,
                ),
            promotes_inputs=[
            'ptg_MW',
            'HSS_kg',
            ])
        model.add_subsystem(
            'finance', 
            finance(
                N_time = N_time, 
                # Depreciation curve
                depreciation_yr = sim_pars['depreciation_yr'],
                depreciation = sim_pars['depreciation'],
                # Inflation curve
                inflation_yr = sim_pars['inflation_yr'],
                inflation = sim_pars['inflation'],
                ref_yr_inflation = sim_pars['ref_yr_inflation'],
                # Early paying or CAPEX Phasing
                phasing_yr = sim_pars['phasing_yr'],
                phasing_CAPEX = sim_pars['phasing_CAPEX'],
                # life_h = life_h
                ),
            promotes_inputs=['price_H2',
                             'wind_WACC',
                             'solar_WACC', 
                             'battery_WACC',
                             'ptg_WACC',
                             'tax_rate',
                             # 'penalty_factor_H2',
                            ],
            promotes_outputs=['NPV',
                              'IRR',
                              'NPV_over_CAPEX',
                              'LCOE',
                              'LCOH',
                              'Revenue',
                              'mean_AEP',
                              'mean_Power2Grid',
                              'annual_H2',
                              'annual_P_ptg',
                              'annual_P_ptg_H2',
                              'penalty_lifetime',
                              'CAPEX',
                              'OPEX',
                              'break_even_H2_price',
                              'break_even_PPA_price',
                              ],
        )
                  
                      
        model.connect('genericWT.ws', 'genericWake.ws')
        model.connect('genericWT.pc', 'genericWake.pc')
        model.connect('genericWT.ct', 'genericWake.ct')
        model.connect('genericWT.ws', 'wpp.ws')

        model.connect('genericWake.pcw', 'wpp.pcw')

        model.connect('abl.wst', 'wpp.wst')
        
        model.connect('wpp.wind_t', 'ems.wind_t')
        model.connect('pvp.solar_t', 'ems.solar_t')
        
        
        model.connect('wpp.wind_t', 'wpp_cost.wind_t')
        
        model.connect('pvp.Apvp', 'shared_cost.Apvp')
        
        model.connect('wpp_cost.CAPEX_w', 'finance.CAPEX_w')
        model.connect('wpp_cost.OPEX_w', 'finance.OPEX_w')

        model.connect('pvp_cost.CAPEX_s', 'finance.CAPEX_s')
        model.connect('pvp_cost.OPEX_s', 'finance.OPEX_s')

        model.connect('battery_cost.CAPEX_b', 'finance.CAPEX_b')
        model.connect('battery_cost.OPEX_b', 'finance.OPEX_b')

        model.connect('shared_cost.CAPEX_sh', 'finance.CAPEX_el')
        model.connect('shared_cost.OPEX_sh', 'finance.OPEX_el')

        model.connect('ptg_cost.CAPEX_ptg', 'finance.CAPEX_ptg')
        model.connect('ptg_cost.OPEX_ptg', 'finance.OPEX_ptg')
        model.connect('ptg_cost.water_consumption_cost', 'finance.water_consumption_cost')

        model.connect('ems.price_t_ext', 'finance.price_t_ext')
        model.connect('ems.hpp_t', 'finance.hpp_t')
        model.connect('ems.penalty_t', 'finance.penalty_t')
        model.connect('ems.hpp_curt_t', 'finance.hpp_curt_t')
        model.connect('ems.m_H2_t', 'finance.m_H2_t')
        model.connect('ems.m_H2_t', 'ptg_cost.m_H2_t' )
        model.connect('ems.P_ptg_t', 'finance.P_ptg_t')
        model.connect('ems.m_H2_demand_t_ext', 'finance.m_H2_demand_t_ext')
        model.connect('ems.m_H2_offtake_t', 'finance.m_H2_offtake_t')
        model.connect('ems.P_ptg_grid_t', 'finance.P_ptg_grid_t')
        model.connect('ems.m_H2_demand_t_ext', 'ptg_cost.m_H2_demand_t_ext')
        model.connect('ems.m_H2_offtake_t', 'ptg_cost.m_H2_offtake_t')
        
        prob = om.Problem(
            model,
            reports=None
        )

        prob.setup()        
        
        # Additional parameters
        prob.set_val('price_t', price)
        prob.set_val('m_H2_demand_t', H2_demand_data['H2_demand'])
        prob.set_val('G_MW', sim_pars['G_MW'])
        #prob.set_val('pv_deg_per_year', sim_pars['pv_deg_per_year'])
        prob.set_val('battery_depth_of_discharge', sim_pars['battery_depth_of_discharge'])
        prob.set_val('battery_charge_efficiency', sim_pars['battery_charge_efficiency'])      
        prob.set_val('peak_hr_quantile',sim_pars['peak_hr_quantile'] )
        prob.set_val('n_full_power_hours_expected_per_day_at_peak_price',
                     sim_pars['n_full_power_hours_expected_per_day_at_peak_price'])        
        #prob.set_val('min_LoH', sim_pars['min_LoH'])
        prob.set_val('wind_WACC', sim_pars['wind_WACC'])
        prob.set_val('solar_WACC', sim_pars['solar_WACC'])
        prob.set_val('battery_WACC', sim_pars['battery_WACC'])
        prob.set_val('ptg_WACC', sim_pars['ptg_WACC'])
        prob.set_val('tax_rate', sim_pars['tax_rate'])
        prob.set_val('land_use_per_solar_MW', sim_pars['land_use_per_solar_MW'])
        prob.set_val('hhv', sim_pars['hhv'])
        prob.set_val('ptg_deg', sim_pars['ptg_deg'])
        prob.set_val('price_H2', sim_pars['price_H2'])
        prob.set_val('penalty_factor_H2', sim_pars['penalty_factor_H2'])
        prob.set_val('storage_eff', sim_pars['storage_eff'])
        prob.set_val('min_power_standby', sim_pars['min_power_standby'])

        self.prob = prob

        self.list_out_vars = [
            'NPV_over_CAPEX',
            'NPV [MEuro]',
            'IRR',
            'LCOE [Euro/MWh]',
            'LCOH [Euro/kg]',
            'Revenue [MEuro]',
            'CAPEX [MEuro]',
            'OPEX [MEuro]',
            'penalty lifetime [MEuro]',
            'AEP [GWh]',
            'annual_Power2Grid [GWh]',
            'GUF',
            'annual_H2 [tons]',
            'annual_P_ptg [GWh]',
            'annual_P_ptg_H2 [GWh]',
            'grid [MW]',
            'wind [MW]',
            'solar [MW]',
            'PtG [MW]',
            'HSS [kg]',
            'Battery Energy [MWh]',
            'Battery Power [MW]',
            'Total curtailment [GWh]',
            'Awpp [km2]',
            'Apvp [km2]',
            'Rotor diam [m]',
            'Hub height [m]',
            'Number of batteries used in lifetime',
            'Break-even H2 price [Euro/kg]',
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
            'cost_of_battery_P_fluct_in_peak_price_ratio',
            'ptg_MW [MW]',
            'HSS_kg [kg]',
            ]   
    
    
    def evaluate(
        self,
        # Wind plant design
        clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        # PV plant design
        solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio,
        # PtG plant design
        ptg_MW,
        # Hydrogen storage capacity
        HSS_kg,
        ):
        """Calculating the financial metrics of the hybrid power plant project.

        Parameters
        ----------
        clearance : Distance from the ground to the tip of the blade [m]
        sp : Specific power of the turbine [MW/m2] 
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
        ptg_MW: Electrolyzer capacity [MW]
        HSS_kg: Hydrogen storgae capacity [kg]

        Returns
        -------
        prob['NPV_over_CAPEX'] : Net present value over the capital expenditures
        prob['NPV'] : Net present value
        prob['IRR'] : Internal rate of return
        prob['LCOE'] : Levelized cost of energy
        prob['LCOH'] : Levelized cost of hydrogen
        prob['Revenue'] : Revenue of HPP
        prob['CAPEX'] : Total capital expenditure costs of the HPP
        prob['OPEX'] : Operational and maintenance costs of the HPP
        prob['penalty_lifetime'] : Lifetime penalty
        prob['AEP']: Annual energy production
        prob['mean_Power2Grid']: Power to grid
        prob['mean_AEP']/(self.sim_pars['G_MW']*365*24) : Grid utilization factor
        prob['annual_H2']: Annual H2 production
        prob['annual_P_ptg']: Annual power converted to hydrogen
        prob['annual_P_ptg_H2']: Annual power from grid converted to hydrogen
        self.sim_pars['G_MW'] : Grid connection [MW]
        wind_MW : Wind power plant installed capacity [MW]
        solar_MW : Solar power plant installed capacity [MW]
        ptg_MW: Electrolyzer capacity [MW]
        HSS_kg: Hydrogen storgae capacity [kg]
        b_E : Battery power [MW]
        b_P : Battery energy [MW]
        prob['total_curtailment']/1e3 : Total curtailed power [GMW]
        d : wind turbine diameter [m]
        hh : hub height of the wind turbine [m]
        num_batteries : Number of batteries
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
        #Apvp = solar_MW * self.sim_pars['land_use_per_solar_MW']
        #prob.set_val('Apvp', Apvp)

        prob.set_val('surface_tilt', surface_tilt)
        prob.set_val('surface_azimuth', surface_azimuth)
        prob.set_val('DC_AC_ratio', DC_AC_ratio)
        prob.set_val('solar_MW', solar_MW)
        prob.set_val('ptg_MW', ptg_MW)
        prob.set_val('HSS_kg', HSS_kg)
        
        prob.set_val('b_P', b_P)
        prob.set_val('b_E', b_E)
        prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio',cost_of_battery_P_fluct_in_peak_price_ratio)        
        
        prob.run_model()
        
        self.prob = prob

        if Nwt == 0:
            cf_wind = np.nan
        else:
            cf_wind = prob.get_val('wpp.wind_t').mean() / p_rated / Nwt  # Capacity factor of wind only

        return np.hstack([
            prob['NPV_over_CAPEX'], 
            prob['NPV']/1e6,
            prob['IRR'],
            prob['LCOE'],
            prob['LCOH'],
            prob['Revenue']/1e6,
            prob['CAPEX']/1e6,
            prob['OPEX']/1e6,
            prob['penalty_lifetime']/1e6,
            prob['mean_AEP']/1e3, #[GWh]
            prob['mean_Power2Grid']/1e3, #GWh
            # Grid Utilization factor
            prob['mean_AEP']/(self.sim_pars['G_MW']*365*24),
            prob['annual_H2']/1e3, # in tons
            prob['annual_P_ptg']/1e3, # in GWh
            prob['annual_P_ptg_H2']/1e3, # in GWh
            self.sim_pars['G_MW'],
            wind_MW,
            solar_MW,
            ptg_MW,
            HSS_kg,
            b_E,
            b_P,
            prob['total_curtailment']/1e3, #[GWh]
            Awpp,
            prob.get_val('shared_cost.Apvp'),
            d,
            hh,
            1 * (b_P>0),
            prob['break_even_H2_price'],
            prob['break_even_PPA_price'],
            cf_wind,
            ])
