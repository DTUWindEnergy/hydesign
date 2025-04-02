# %%

# import necessary libraries
import numpy as np
import pandas as pd
import openmdao.api as om
from docplex.mp.model import Model

# Define EmsSolarX Class
class EmsSolarX(om.ExplicitComponent):
    """
    Energy Management Optimization Model for SolarX.

    This OpenMDAO component models an energy management system optimized for the SolarX power plant, focusing on maximizing revenue
    over time while taking into account production constraints, battery degradation, and penalties for unmet production targets
    during peak hours. The model integrates CPV, hydrogen production, and molten salt heat storage systems.

    Parameters
    ----------
    N_time : int
        Number of simulation time steps.
    steam_turbine_efficiency : float
        Efficiency of the steam turbine for energy conversion.
    steam_specific_heat_capacity : float
        Specific heat capacity of steam in the system.
    hot_tank_efficiency : float
        Efficiency of the molten salt hot storage tank.
    hot_steam_temp_ms : float
        Temperature of hot steam, in degrees Celsius.
    cold_steam_temp_ms : float
        Temperature of cold steam, in degrees Celsius.
    heat_exchanger_efficiency : float
        Efficiency of the heat exchanger between molten salt and steam.
    hot_molten_salt_density : float
        Density of molten salt in the hot storage tank.
    biogas_h2_reactor_dni_to_heat_efficiency : float
        Efficiency of DNI to heat conversion in the biogas hydrogen reactor.
    biogas_h2_reactor_el_to_heat_efficiency : float
        Efficiency of electricity to heat conversion in the biogas hydrogen reactor.
    biogas_h2_reactor_efficiency_curve : dict
        Efficiency curve mapping DNI/electricity input levels to efficiencies.
    maximum_h2_production_reactor_kg_per_m2 : float
        Maximum hydrogen production per square meter in reactor.
    life_h : int
        Lifetime of the model in hours, typically set to 25 years.
    batch_size : int
        Size of batches in the time series data for computation.
    weeks_per_season_per_year : int or None, optional
        Number of weeks per season to simplify annual time series. Default is None (uses full input data).

    Returns
    -------
    dict
        Key output time series over the model's lifetime, covering power output, hydrogen production, molten salt flow,
        and penalty terms, among other outputs:
        - 'p_hpp_t' : ndarray
            Total power output time series of the hybrid power plant.
        - 'p_curtailment_t' : ndarray
            Time series for curtailed power, showing unused power due to grid constraints.
        - 'p_cpv_t' : ndarray
            Power output time series specifically for CPV generation.
        - 'p_st_t' : ndarray
            Power output time series for steam turbine power generation.
        - 'p_biogas_h2_t' : ndarray
            Electricity consumption time series for hydrogen production.
        - 'penalty_t' : ndarray
            Time series representing penalties for failure to meet production targets during peak hours.
        - 'flow_ms_heat_exchanger_t' : ndarray
            Time series of molten salt flow into the heat exchanger.
        - 'flow_ms_q_t' : ndarray
            Time series of molten salt flow used for heat production.
        - 'v_hot_ms_t' : ndarray
            Volume of molten salt stored in the hot tank over time.
        - 'alpha_cpv_t' : ndarray
            Proportion of available flux directed towards CPV.
        - 'alpha_h2_t' : ndarray
            Proportion of available flux directed towards hydrogen production.
        - 'h2_t' : ndarray
            Hydrogen production time series.
        - 'q_t' : ndarray
            Heat output time series.
        - 'biogas_t' : ndarray
            Biogas consumption time series.
        - 'water_t' : ndarray
            Water consumption time series.
        - 'co2_t' : ndarray
            CO2 consumption time series.
        - 'p_st_max_dni_t' : ndarray
            Maximum potential steam turbine power time series, based on DNI input.
        - 'q_max_dni_t' : ndarray
            Maximum potential heat production based on DNI input.
        - 'flow_steam_st_t' : ndarray
            Flow rate of steam directed to the steam turbine over time.
        - 'biogas_h2_procuded_h2_kg_in_dni_reactor_t' : ndarray
            Hydrogen produced in the DNI reactor.
        - 'biogas_h2_procuded_h2_kg_in_el_reactor_t' : ndarray
            Hydrogen produced in the electrical reactor.
    """

    def __init__(
        self, 
        N_time,
        max_el_buy_from_grid_mw,

        # cst
        steam_turbine_efficiency,
        steam_specific_heat_capacity,
        hot_tank_efficiency,
        hot_steam_temp_ms,
        cold_steam_temp_ms,
        heat_exchanger_efficiency,
        hot_molten_salt_density,
        heat_penalty_euro_per_mwht,

        # biogas_h2
        biogas_h2_reactor_dni_to_heat_efficiency,
        biogas_h2_reactor_el_to_heat_efficiency,
        biogas_h2_reactor_efficiency_curve,
        maximum_h2_production_reactor_kg_per_m2,

        # others
        life_h,
        batch_size,
        weeks_per_season_per_year = None,
        ):

        super().__init__()
        self.N_time = int(N_time)
        self.max_el_buy_from_grid_mw = max_el_buy_from_grid_mw

        # cst
        self.steam_turbine_efficiency = steam_turbine_efficiency
        self.hot_tank_efficiency = hot_tank_efficiency
        self.steam_specific_heat_capacity=steam_specific_heat_capacity
        self.hot_steam_temp_ms = hot_steam_temp_ms
        self.cold_steam_temp_ms = cold_steam_temp_ms
        self.heat_exchanger_efficiency = heat_exchanger_efficiency
        self.hot_molten_salt_density = hot_molten_salt_density
        self.heat_penalty_euro_per_mwht = heat_penalty_euro_per_mwht

        # biogas_h2
        self.biogas_h2_reactor_dni_to_heat_efficiency = biogas_h2_reactor_dni_to_heat_efficiency
        self.biogas_h2_reactor_el_to_heat_efficiency = biogas_h2_reactor_el_to_heat_efficiency
        self.biogas_h2_reactor_efficiency_curve = biogas_h2_reactor_efficiency_curve
        self.maximum_h2_production_reactor_kg_per_m2 = maximum_h2_production_reactor_kg_per_m2

        # others
        self.life_h = int(life_h)
        self.batch_size = batch_size
        self.weeks_per_season_per_year = weeks_per_season_per_year
        
    def setup(self):
        # Define input and output variables to the openmdao model
        # inputs
        # grid
        self.add_input(
            'grid_el_capacity',
            desc="Grid capacity",
            units='MW')
        self.add_input(
            'grid_h2_capacity',
            desc="hydrogen capacity of the grid",
            units='kg/h'
        )

        # cpv
        self.add_input(
            'p_cpv_max_dni_t',
            desc="maximum cpv power power time series (assuming all available solar flux goes to cpv)",
            units='MW',
            shape=[self.N_time])
        self.add_input('cpv_inverter_mw',
                       desc="rated power of the cpv inverter",
                       units='MW', )
        self.add_input('cpv_rated_mw',
                       desc="rated power of the cpv reciever",
                       units='MW', )

        # cst
        self.add_input(
            'v_molten_salt_tank_m3',
            desc="Volume of the hot molten salt tank",
            units='m**3')
        self.add_input(
            'flow_ms_max_t',
            desc="maximum flow of molten salt time series (asuuing all available solar flux goes to cst)",
            units='kg/h',
            shape=[self.N_time])
        self.add_input(
            'p_rated_st',
            desc="rated power of the steam turbine",
            units='MW',
        )
        self.add_input(
            'delta_q_hot_cold_ms_per_kg',
            desc="Heat (kJ) difference between hot and cold molten salt per kg",
            units='kJ/kg',
        )
        self.add_input(
            'v_max_hot_ms_percentage',
            desc="Maximim allowable volume of the hot molten salt in percentage",
        )
        self.add_input(
            'v_min_hot_ms_percentage',
            desc="Minimum allowable volume of the hot molten salt in percentage",
        )
        self.add_input(
            'flow_ms_max_cst_receiver_capacity',
            desc="Capacity of the reciever for molten salt flow",
            units='kg/h',
        )
        self.add_input(
            'heat_exchanger_capacity',
            desc="capacity of the steam generator",
            units='MW',
        )

        # biogas_h2
        self.add_input(
            'biogas_h2_mass_ratio',
            desc="amount of biogas required for production of 1kg H2 in biogas_h2 module",
        )
        self.add_input(
            'water_h2_mass_ratio',
            desc="amount of water required for production of 1kg H2 in biogas_h2 module",
        )
        self.add_input(
            'co2_h2_mass_ratio',
            desc="amount of CO2 required for production of 1kg H2 in biogas_h2 module",
        )
        self.add_input('max_solar_flux_dni_reactor_biogas_h2_t',
                       desc="maximum solar flux at dni reactor of the biogas_h2 module",
                       units='MW',
                       shape=[self.N_time])
        self.add_input(
            'heat_mwht_per_kg_h2',
            desc="Required heat for generating 1 kg of hydrogen",
            units = 'MW*h/kg',
        )
        self.add_input(
            'area_dni_reactor_biogas_h2',
            desc="area of dni reactor in biogas_h2 module",
            units='m**2',
        )
        self.add_input(
            'area_el_reactor_biogas_h2',
            desc="area of el reactor in biogas_h2 module",
            units='m**2',
        )

        # prices
        self.add_input(
            'price_el_t',
            desc="electricity price time series",
            shape=[self.N_time])
        self.add_input(
            'price_h2_t',
            desc="Hydrogen price time series",
            shape=[self.N_time])
        self.add_input(
            'price_water_t',
            desc="water price time series",
            shape=[self.N_time])
        self.add_input(
            'price_co2_t',
            desc="co2 price time series",
            shape=[self.N_time])
        self.add_input(
            'price_biogas_t',
            desc="Biogas price time series",
            shape=[self.N_time])
        self.add_input(
            'peak_hr_quantile',
            desc="Quantile of price tim sereis to define peak price hours (above this quantile).\n"+
                 "Only used for peak production penalty and for cost of battery degradation.")
        self.add_input(
            'n_full_power_hours_expected_per_day_at_peak_price',
            desc="Pnealty occurs if nunmber of full power hours expected per day at peak price are not reached.")


        self.add_input(
            'demand_q_t',
            desc="Heat demand time series",
            shape=[self.N_time])

        # outputs
        # hpp
        self.add_output(
            'hpp_t_ext',
            desc="HPP power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'hpp_curt_t_ext',
            desc="HPP curtailed power time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'total_curtailment',
            desc="total HPP curtailed power",
            units='MW',)

        # cpv
        self.add_output(
            'alpha_cpv_t_ext',
            desc="Share of flux_sf directed towards cpv receiver",
            shape=[self.life_h])
        self.add_output(
            'p_cpv_max_dni_t_ext',
            desc="max cpv power time series extended to lifetime",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'p_cpv_t_ext',
            desc="cpv power time series extended to lifetime",
            units='MW',
            shape=[self.life_h])

        # cst
        self.add_output(
            'alpha_cst_t_ext',
            desc="Share of flux_sf directed towards cst receiver",
            shape=[self.life_h])
        self.add_output(
            'flow_ms_heat_exchanger_t_ext',
            desc="flow of molten salt into the heat exchanger",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'flow_ms_q_t_ext',
            desc="flow of molten salt used for heat production",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'p_st_t_ext',
            desc="steam turbine power time series extended to lifetime",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'flow_ms_t_ext',
            desc="flow of molten salt time series extended to life time",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'v_hot_ms_t_ext',
            desc="Volum of the molten salt stored in hot tank",
            units='m**3',
            shape=[self.life_h])
        self.add_output(
            'p_st_max_dni_t_ext',
            desc="maximum steam turbine power with max available solar flux",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'q_max_dni_t_ext',
            desc="maximum heat production with max available solar flux",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'flow_steam_st_t_ext',
            desc="flow of steam into the steam turbine extended to lifetime",
            units='kg/h',
            shape=[self.life_h])

        # biogas_h2
        self.add_output(
            'alpha_h2_t_ext',
            desc="Share of flux_sf directed towards H2 receiver",
            shape=[self.life_h])
        self.add_output(
            'biogas_h2_procuded_h2_kg_in_dni_reactor_t_ext',
            desc="produced h2 in dni reactor of biogas_h2 module extented to lifetime",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'biogas_h2_procuded_h2_kg_in_el_reactor_t_ext',
            desc="produced h2 in el reactor of biogas_h2 module extented to lifetime",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'h2_t_ext',
            desc="H2 production time series",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'max_solar_flux_dni_reactor_biogas_h2_t_ext',
            desc="max solar flux on biogas_h2 dni reactor time series extended to lifetime",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'p_biogas_h2_t_ext',
            desc="consumed electricity in el_reactor of biogas_h2 time series extended to lifetime",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'q_t_ext',
            desc="Output heat time series",
            units='MW',
            shape=[self.life_h])
        self.add_output(
            'biogas_t_ext',
            desc="Required biogas time series",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'water_t_ext',
            desc="Required water time series",
            units='kg/h',
            shape=[self.life_h])
        self.add_output(
            'co2_t_ext',
            desc="Required co2 time series",
            units='kg/h',
            shape=[self.life_h])

        # prices (extended to lifetime)
        self.add_output(
            'price_el_t_ext',
            desc="Electricity price time series",
            shape=[self.life_h])
        self.add_output(
            'price_h2_t_ext',
            desc="Hydrogen price time series",
            shape=[self.life_h])
        self.add_output(
            'demand_q_t_ext',
            desc="Heat price time series",
            shape=[self.life_h])
        self.add_output(
            'price_water_t_ext',
            desc="water price time series",
            shape=[self.life_h])
        self.add_output(
            'price_co2_t_ext',
            desc="co2 price time series",
            shape=[self.life_h])
        self.add_output(
            'price_biogas_t_ext',
            desc="Biogas price time series",
            shape=[self.life_h])

        # others
        self.add_output(
            'penalty_t_ext',
            desc="penalty for not reaching expected energy productin at peak hours",
            shape=[self.life_h])
        self.add_output(
            'penalty_q_t_ext',
            desc="penalty for not reaching expected heat production",
            shape=[self.life_h])

    def compute(self, inputs, outputs):
        # Extract required inputs for computation
        # grid
        max_el_buy_from_grid_mw = self.max_el_buy_from_grid_mw
        grid_el_capacity = inputs['grid_el_capacity']
        grid_h2_capacity = inputs['grid_h2_capacity'][0]

        # cpv
        p_cpv_max_dni_t = inputs['p_cpv_max_dni_t']
        cpv_inverter_mw = inputs['cpv_inverter_mw'][0]
        cpv_rated_mw = inputs['cpv_rated_mw'][0]

        # cst
        v_molten_salt_tank_m3 = inputs['v_molten_salt_tank_m3'][0]
        p_rated_st = inputs['p_rated_st']
        flow_ms_max_t = inputs['flow_ms_max_t']
        flow_ms_max_cst_receiver_capacity = inputs['flow_ms_max_cst_receiver_capacity'][0]
        delta_q_hot_cold_ms_per_kg = inputs['delta_q_hot_cold_ms_per_kg'][0]
        p_rated_st = inputs['p_rated_st'][0]
        heat_exchanger_capacity = inputs['heat_exchanger_capacity'][0]
        v_max_hot_ms_percentage = inputs['v_max_hot_ms_percentage'][0]
        v_min_hot_ms_percentage = inputs['v_min_hot_ms_percentage'][0]
        area_dni_reactor_biogas_h2 = inputs['area_dni_reactor_biogas_h2'][0]
        area_el_reactor_biogas_h2 = inputs['area_el_reactor_biogas_h2'][0]

        # biogas_h2
        biogas_h2_reactor_dni_to_heat_efficiency = self.biogas_h2_reactor_dni_to_heat_efficiency
        biogas_h2_reactor_el_to_heat_efficiency = self.biogas_h2_reactor_el_to_heat_efficiency
        biogas_h2_reactor_efficiency_curve = self.biogas_h2_reactor_efficiency_curve
        maximum_h2_production_reactor_kg_per_m2 = self.maximum_h2_production_reactor_kg_per_m2
        max_solar_flux_dni_reactor_biogas_h2_t = inputs['max_solar_flux_dni_reactor_biogas_h2_t']
        heat_mwht_per_kg_h2 = inputs['heat_mwht_per_kg_h2'][0]
        biogas_h2_mass_ratio = inputs['biogas_h2_mass_ratio'][0]
        water_h2_mass_ratio = inputs['water_h2_mass_ratio'][0]
        co2_h2_mass_ratio = inputs['co2_h2_mass_ratio'][0]

        # prices
        price_el_t = inputs['price_el_t']
        price_h2_t = inputs['price_h2_t']
        demand_q_t = inputs['demand_q_t']
        price_water_t = inputs['price_water_t']
        price_biogas_t = inputs['price_biogas_t']
        price_co2_t = inputs['price_co2_t']
        peak_hr_quantile = inputs['peak_hr_quantile'][0]
        n_full_power_hours_expected_per_day_at_peak_price = inputs['n_full_power_hours_expected_per_day_at_peak_price'][0]

        # Generate synthetic time index for data compatibility in the energy management system (EMS)
        synthetic_time_index_df = pd.DataFrame(
            index=pd.date_range(
                start='01-01-1991 00:00',
                periods=len(price_el_t),
                freq='1h'))
        synthetic_time_index_df['p_cpv_max_dni_t'] = p_cpv_max_dni_t
        synthetic_time_index_df['flow_ms_max_t'] = flow_ms_max_t
        synthetic_time_index_df['max_solar_flux_dni_reactor_biogas_h2_t'] = max_solar_flux_dni_reactor_biogas_h2_t
        synthetic_time_index_df['price_el_t'] = price_el_t
        synthetic_time_index_df['price_h2_t'] = price_h2_t
        synthetic_time_index_df['demand_q_t'] = demand_q_t
        synthetic_time_index_df['price_water_t'] = price_water_t
        synthetic_time_index_df['price_co2_t'] = price_co2_t
        synthetic_time_index_df['price_biogas_t'] = price_biogas_t
        
        # Perform optimization using `ems_cplex_solarX`
        p_hpp_t, p_curtailment_t, p_cpv_t, p_st_t, p_biogas_h2_t, penalty_t, penalty_q_t, flow_ms_heat_exchanger_t, flow_ms_q_t,\
        v_hot_ms_t, alpha_cpv_t, alpha_cst_t, alpha_h2_t, h2_t, q_t, biogas_t, water_t, co2_t, p_st_max_dni_t,\
        q_max_dni_t, flow_steam_st_t, biogas_h2_procuded_h2_kg_in_dni_reactor_t, biogas_h2_procuded_h2_kg_in_el_reactor_t= ems_cplex_solarX(
            # grid
            hpp_grid_connection = grid_el_capacity[0],
            max_el_buy_from_grid_mw = max_el_buy_from_grid_mw,
            grid_h2_capacity=grid_h2_capacity,

            # cpv
            p_cpv_max_dni_t = synthetic_time_index_df.p_cpv_max_dni_t,
            cpv_inverter_mw=cpv_inverter_mw,
            cpv_rated_mw=cpv_rated_mw,

            # cst
            flow_ms_max_t = synthetic_time_index_df.flow_ms_max_t,
            v_molten_salt_tank_m3=v_molten_salt_tank_m3,
            delta_q_hot_cold_ms_per_kg=delta_q_hot_cold_ms_per_kg,
            efficiency_st=self.steam_turbine_efficiency,
            hot_tank_efficiency=self.hot_tank_efficiency,
            steam_specific_heat_capacity=self.steam_specific_heat_capacity,
            hot_steam_temp_ms=self.hot_steam_temp_ms,
            cold_steam_temp_ms=self.cold_steam_temp_ms,
            hot_molten_salt_density=self.hot_molten_salt_density,
            heat_penalty_euro_per_mwht=self.heat_penalty_euro_per_mwht,
            heat_exchanger_efficiency=self.heat_exchanger_efficiency,
            flow_ms_max_cst_receiver_capacity = flow_ms_max_cst_receiver_capacity,
            max_solar_flux_dni_reactor_biogas_h2_t=synthetic_time_index_df.max_solar_flux_dni_reactor_biogas_h2_t,
            p_rated_st=p_rated_st,
            heat_exchanger_capacity=heat_exchanger_capacity,
            v_max_hot_ms_percentage=v_max_hot_ms_percentage,
            v_min_hot_ms_percentage=v_min_hot_ms_percentage,

            # biogas_h2
            biogas_h2_reactor_dni_to_heat_efficiency=biogas_h2_reactor_dni_to_heat_efficiency,
            biogas_h2_reactor_el_to_heat_efficiency=biogas_h2_reactor_el_to_heat_efficiency,
            biogas_h2_reactor_efficiency_curve=biogas_h2_reactor_efficiency_curve,
            area_dni_reactor_biogas_h2=area_dni_reactor_biogas_h2,
            area_el_reactor_biogas_h2=area_el_reactor_biogas_h2,
            maximum_h2_production_reactor_kg_per_m2=maximum_h2_production_reactor_kg_per_m2,
            heat_mwht_per_kg_h2=heat_mwht_per_kg_h2,
            biogas_h2_mass_ratio=biogas_h2_mass_ratio,
            water_h2_mass_ratio=water_h2_mass_ratio,
            co2_h2_mass_ratio=co2_h2_mass_ratio,

            # prices
            price_el_t = synthetic_time_index_df.price_el_t,
            price_h2_t = synthetic_time_index_df.price_h2_t,
            demand_q_t = synthetic_time_index_df.demand_q_t,
            price_water_t = synthetic_time_index_df.price_water_t,
            price_co2_t = synthetic_time_index_df.price_co2_t,
            price_biogas_t = synthetic_time_index_df.price_biogas_t,

            # other
            batch_size=self.batch_size,
            peak_hr_quantile = peak_hr_quantile,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
        )

        # Extend (by repeating them and stacking) all variable to full lifetime
        # hpp
        outputs['hpp_t_ext'] = expand_to_lifetime(
            p_hpp_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        p_curtailment_t_ext = expand_to_lifetime(
            p_curtailment_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['hpp_curt_t_ext'] = p_curtailment_t_ext
        outputs['total_curtailment'] = p_curtailment_t_ext.sum()

        # cpv
        outputs['alpha_cpv_t_ext'] = expand_to_lifetime(
            alpha_cpv_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['p_cpv_max_dni_t_ext'] = expand_to_lifetime(
            p_cpv_max_dni_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['p_cpv_t_ext'] = expand_to_lifetime(
            p_cpv_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)

        # cst
        outputs['alpha_cst_t_ext'] = expand_to_lifetime(
            alpha_cst_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['p_st_t_ext'] = expand_to_lifetime(
            p_st_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['flow_ms_heat_exchanger_t_ext'] = expand_to_lifetime(
            flow_ms_heat_exchanger_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['flow_ms_q_t_ext'] = expand_to_lifetime(
            flow_ms_q_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['v_hot_ms_t_ext'] = expand_to_lifetime(
            v_hot_ms_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['q_t_ext'] = expand_to_lifetime(
            q_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['p_st_max_dni_t_ext'] = expand_to_lifetime(
            p_st_max_dni_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['q_max_dni_t_ext'] = expand_to_lifetime(
            q_max_dni_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['flow_steam_st_t_ext'] = expand_to_lifetime(
            flow_steam_st_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)

        # biogas_h2
        outputs['alpha_h2_t_ext'] = expand_to_lifetime(
            alpha_h2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['biogas_h2_procuded_h2_kg_in_dni_reactor_t_ext'] = expand_to_lifetime(
            biogas_h2_procuded_h2_kg_in_dni_reactor_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['biogas_h2_procuded_h2_kg_in_el_reactor_t_ext'] = expand_to_lifetime(
            biogas_h2_procuded_h2_kg_in_el_reactor_t, life_h=self.life_h, weeks_per_season_per_year=self.weeks_per_season_per_year)
        outputs['h2_t_ext'] = expand_to_lifetime(
            h2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['max_solar_flux_dni_reactor_biogas_h2_t_ext'] = expand_to_lifetime(
            max_solar_flux_dni_reactor_biogas_h2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['water_t_ext'] = expand_to_lifetime(
            water_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['co2_t_ext'] = expand_to_lifetime(
            co2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['biogas_t_ext'] = expand_to_lifetime(
            biogas_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['p_biogas_h2_t_ext'] = expand_to_lifetime(
            p_biogas_h2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)

        # prices
        outputs['price_el_t_ext'] = expand_to_lifetime(
            price_el_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['price_h2_t_ext'] = expand_to_lifetime(
            price_h2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['demand_q_t_ext'] = expand_to_lifetime(
            demand_q_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['price_water_t_ext'] = expand_to_lifetime(
            price_water_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['price_co2_t_ext'] = expand_to_lifetime(
            price_co2_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['price_biogas_t_ext'] = expand_to_lifetime(
            price_biogas_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['penalty_t_ext'] = expand_to_lifetime(
            penalty_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)
        outputs['penalty_q_t_ext'] = expand_to_lifetime(
            penalty_q_t, life_h = self.life_h, weeks_per_season_per_year = self.weeks_per_season_per_year)

# Optimization function for the energy management system
def ems_cplex_solarX(
    # grid
    hpp_grid_connection,
    max_el_buy_from_grid_mw,
    grid_h2_capacity,

    # cpv
    cpv_inverter_mw,
    cpv_rated_mw,

    # cst
    p_cpv_max_dni_t,
    flow_ms_max_t,
    flow_ms_max_cst_receiver_capacity,
    v_molten_salt_tank_m3,
    delta_q_hot_cold_ms_per_kg,
    heat_exchanger_capacity,
    v_max_hot_ms_percentage,
    v_min_hot_ms_percentage,
    efficiency_st,
    hot_tank_efficiency,
    p_rated_st,
    steam_specific_heat_capacity,
    hot_steam_temp_ms,
    cold_steam_temp_ms,
    hot_molten_salt_density,
    heat_penalty_euro_per_mwht,
    heat_exchanger_efficiency,

    # boigas_h2
    max_solar_flux_dni_reactor_biogas_h2_t,
    biogas_h2_reactor_dni_to_heat_efficiency,
    biogas_h2_reactor_el_to_heat_efficiency,
    biogas_h2_reactor_efficiency_curve,
    area_dni_reactor_biogas_h2,
    area_el_reactor_biogas_h2,
    maximum_h2_production_reactor_kg_per_m2,
    heat_mwht_per_kg_h2,
    biogas_h2_mass_ratio,
    water_h2_mass_ratio,
    co2_h2_mass_ratio,

    # prices
    price_el_t,
    price_h2_t,
    demand_q_t,
    price_water_t,
    price_co2_t,
    price_biogas_t,
    peak_hr_quantile,
    n_full_power_hours_expected_per_day_at_peak_price,
    batch_size,
    ):
    """
    Energy Management System (EMS) Optimization for SolarX Plant.

    This function sets up and executes an optimization model to manage energy flows in the SolarX plant, balancing production,
    storage, and consumption across various modules (CPV, hydrogen production, molten salt storage). The model optimizes revenue
    while minimizing penalties for unmet energy demands during peak hours.

    Parameters
    ----------
    hpp_grid_connection : float
        Maximum grid connection capacity in MW.
    grid_h2_capacity : float
        Maximum grid hydrogen capacity in kg/h.
    cpv_inverter_mw : float
        Rated capacity of the CPV inverter in MW.
    cpv_rated_mw : float
        Rated capacity of the CPV receiver in MW.
    p_cpv_max_dni_t : pd.Series
        Maximum CPV power time series based on DNI.
    flow_ms_max_t : pd.Series
        Maximum molten salt flow rate time series in kg/h.
    flow_ms_max_cst_receiver_capacity : float
        Maximum flow capacity of CST receiver in kg/h.
    v_molten_salt_tank_m3 : float
        Volume of the hot molten salt tank in cubic meters.
    delta_q_hot_cold_ms_per_kg : float
        Heat difference (kJ/kg) between hot and cold molten salt.
    heat_exchanger_capacity : float
        Capacity of the steam heat exchanger in MW.
    v_max_hot_ms_percentage : float
        Maximum allowable volume percentage for the hot molten salt tank.
    v_min_hot_ms_percentage : float
        Minimum allowable volume percentage for the hot molten salt tank.
    efficiency_st : float
        Efficiency of the steam turbine.
    hot_tank_efficiency : float
        Efficiency of the hot molten salt storage tank.
    steam_specific_heat_capacity : float
        Specific heat capacity of steam.
    hot_steam_temp_ms : float
        Temperature of hot steam.
    cold_steam_temp_ms : float
        Temperature of cold steam.
    hot_molten_salt_density : float
        Density of molten salt in the hot tank.
    heat_exchanger_efficiency : float
        Efficiency of the heat exchanger.
    max_solar_flux_dni_reactor_biogas_h2_t : pd.Series
        DNI flux for hydrogen production in biogas reactor.
    biogas_h2_reactor_dni_to_heat_efficiency : float
        Efficiency of DNI to heat in biogas reactor.
    biogas_h2_reactor_el_to_heat_efficiency : float
        Efficiency of electricity to heat in biogas reactor.
    biogas_h2_reactor_efficiency_curve : dict
        Efficiency curve data for biogas reactor.
    area_dni_reactor_biogas_h2 : float
        DNI reactor area in biogas hydrogen module.
    area_el_reactor_biogas_h2 : float
        EL reactor area in biogas hydrogen module.
    maximum_h2_production_reactor_kg_per_m2 : float
        Max hydrogen production for DNI reactor.
    heat_mwht_per_kg_h2 : float
        Required heat to produce 1 kg hydrogen.
    biogas_h2_mass_ratio : float
        Biogas needed per kg hydrogen.
    water_h2_mass_ratio : float
        Water needed per kg hydrogen.
    co2_h2_mass_ratio : float
        CO2 needed per kg hydrogen.
    price_el_t : pd.Series
        Electricity price time series.
    price_h2_t : pd.Series
        Hydrogen price time series.
    demand_q_t : pd.Series
        Heat demand time series.
    price_water_t : pd.Series
        Water price time series.
    price_co2_t : pd.Series
        CO2 price time series.
    price_biogas_t : pd.Series
        Biogas price time series.
    peak_hr_quantile : float
        Quantile for defining peak price hours.
    n_full_power_hours_expected_per_day_at_peak_price : int
        Expected full power hours at peak price to avoid penalty.
    batch_size : int
        Batch size for optimization.

    Returns
    -------
    tuple of np.ndarray
        Each array contains time series data for various metrics:
        - p_hpp_t : ndarray
            HPP power time series.
        - p_curtailment_t : ndarray
            HPP curtailed power time series.
        - p_cpv_t : ndarray
            CPV power time series.
        - p_st_t : ndarray
            Steam turbine power time series.
        - p_biogas_h2_t : ndarray
            Power time series for hydrogen production.
        - penalty_t : ndarray
            Penalty for unmet energy at peak hours.
        - flow_ms_heat_exchanger_t : ndarray
            Molten salt flow into heat exchanger.
        - flow_ms_q_t : ndarray
            Molten salt flow for heat production.
        - v_hot_ms_t : ndarray
            Volume of molten salt in hot tank.
        - alpha_cpv_t : ndarray
            Share of solar flux to CPV.
        - alpha_h2_t : ndarray
            Share of solar flux to hydrogen production.
        - h2_t : ndarray
            Hydrogen production time series.
        - q_t : ndarray
            Heat production time series.
        - biogas_t : ndarray
            Biogas consumption.
        - water_t : ndarray
            Water consumption.
        - co2_t : ndarray
            CO2 consumption.
        - p_st_max_dni_t : ndarray
            Maximum steam turbine power.
        - q_max_dni_t : ndarray
            Maximum heat production.
        - flow_steam_st_t : ndarray
            Steam flow in turbine.
        - biogas_h2_procuded_h2_kg_in_dni_reactor_t : ndarray
            H2 produced in DNI reactor.
        - biogas_h2_procuded_h2_kg_in_el_reactor_t : ndarray
            H2 produced in EL reactor.
    """

    # Split data into batches (default: weekly batches)
    batches = split_in_batch(list(range(len(price_el_t))), batch_size)
    
    # Allocate arrays to store results across time steps
    # hpp
    p_hpp_t = np.zeros(len(price_el_t))
    p_curtailment_t = np.zeros(len(price_el_t))

    # cpv
    p_cpv_t = np.zeros(len(price_el_t))
    alpha_cpv_t = np.zeros(len(price_el_t))

    # cst
    alpha_cst_t = np.zeros(len(price_el_t))
    p_st_t = np.zeros(len(price_el_t))
    flow_ms_heat_exchanger_t = np.zeros(len(price_el_t))
    flow_ms_q_t = np.zeros(len(price_el_t))
    v_hot_ms_t = np.zeros(len(price_el_t))
    p_st_max_dni_t = np.zeros(len(price_el_t))
    q_max_dni_t = np.zeros(len(price_el_t))
    flow_steam_st_t = np.zeros(len(price_el_t))
    q_t = np.zeros(len(price_el_t))

    # biogas_h2
    alpha_h2_t = np.zeros(len(price_el_t))
    biogas_h2_procuded_h2_kg_in_dni_reactor_t = np.zeros(len(price_el_t))
    biogas_h2_procuded_h2_kg_in_el_reactor_t = np.zeros(len(price_el_t))
    h2_t = np.zeros(len(price_el_t))
    biogas_t = np.zeros(len(price_el_t))
    water_t = np.zeros(len(price_el_t))
    co2_t = np.zeros(len(price_el_t))
    p_biogas_h2_t = np.zeros(len(price_el_t))

    # other
    penalty_t = np.zeros(len(price_el_t))
    penalty_q_t = np.zeros(len(price_el_t))

    # Loop over each batch and perform optimization
    for ib, batch in enumerate(batches):
        p_cpv_max_dni_t_sel = p_cpv_max_dni_t.iloc[batch]
        flow_ms_max_t_sel = flow_ms_max_t.iloc[batch]
        max_solar_flux_dni_reactor_biogas_h2_t_sel = max_solar_flux_dni_reactor_biogas_h2_t.iloc[batch]
        price_el_t_sel = price_el_t.iloc[batch]
        price_h2_t_sel = price_h2_t.iloc[batch]
        demand_q_t_sel = demand_q_t.iloc[batch]
        price_water_t_sel = price_water_t.iloc[batch]
        price_biogas_t_sel = price_biogas_t.iloc[batch]
        price_co2_t_sel = price_co2_t.iloc[batch]
        
        # Solve for each batch using `ems_cplex_solarX_parts` function
        p_hpp_t_batch, p_curtailment_t_batch, p_cpv_t_batch, p_st_t_batch, p_biogas_h2_t_batch,\
        penalty_t_batch, penalty_q_t_batch, flow_ms_heat_exchanger_t_batch, flow_ms_q_t_batch, v_hot_ms_t_batch,\
        alpha_cpv_t_batch, alpha_cst_t_batch, alpha_h2_t_batch, h2_t_batch, q_t_batch, biogas_t_batch, water_t_batch, co2_t_batch, p_st_max_dni_t_batch,\
        q_max_dni_t_batch,  flow_steam_st_t_batch, biogas_h2_procuded_h2_kg_in_dni_reactor_t_batch, biogas_h2_procuded_h2_kg_in_el_reactor_t_batch = ems_cplex_solarX_parts(
            # grid
            hpp_grid_connection = hpp_grid_connection,
            max_el_buy_from_grid_mw = max_el_buy_from_grid_mw,
            grid_h2_capacity = grid_h2_capacity,

            # cpv
            p_cpv_max_dni_t = p_cpv_max_dni_t_sel,
            cpv_inverter_mw=cpv_inverter_mw,
            cpv_rated_mw=cpv_rated_mw,

            # cst
            flow_ms_max_t = flow_ms_max_t_sel,
            v_molten_salt_tank_m3=v_molten_salt_tank_m3,
            delta_q_hot_cold_ms_per_kg=delta_q_hot_cold_ms_per_kg,
            p_rated_st=p_rated_st,
            efficiency_st = efficiency_st,
            hot_tank_efficiency=hot_tank_efficiency,
            heat_exchanger_capacity = heat_exchanger_capacity,
            v_max_hot_ms_percentage = v_max_hot_ms_percentage,
            v_min_hot_ms_percentage = v_min_hot_ms_percentage,
            flow_ms_max_cst_receiver_capacity=flow_ms_max_cst_receiver_capacity,
            steam_specific_heat_capacity=steam_specific_heat_capacity,
            hot_steam_temp_ms = hot_steam_temp_ms,
            cold_steam_temp_ms = cold_steam_temp_ms,
            hot_molten_salt_density=hot_molten_salt_density,
            heat_penalty_euro_per_mwht=heat_penalty_euro_per_mwht,
            heat_exchanger_efficiency=heat_exchanger_efficiency,

            # biogas_h2
            heat_mwht_per_kg_h2 = heat_mwht_per_kg_h2,
            biogas_h2_reactor_dni_to_heat_efficiency = biogas_h2_reactor_dni_to_heat_efficiency,
            biogas_h2_reactor_el_to_heat_efficiency = biogas_h2_reactor_el_to_heat_efficiency,
            biogas_h2_reactor_efficiency_curve = biogas_h2_reactor_efficiency_curve,
            area_dni_reactor_biogas_h2=area_dni_reactor_biogas_h2,
            area_el_reactor_biogas_h2=area_el_reactor_biogas_h2,
            maximum_h2_production_reactor_kg_per_m2=maximum_h2_production_reactor_kg_per_m2,
            max_solar_flux_dni_reactor_biogas_h2_t=max_solar_flux_dni_reactor_biogas_h2_t_sel,
            biogas_h2_mass_ratio=biogas_h2_mass_ratio,
            water_h2_mass_ratio=water_h2_mass_ratio,
            co2_h2_mass_ratio=co2_h2_mass_ratio,

            # prices
            price_el_t = price_el_t_sel,
            price_h2_t = price_h2_t_sel,
            demand_q_t = demand_q_t_sel,
            price_water_t = price_water_t_sel,
            price_co2_t = price_water_t_sel,
            price_biogas_t = price_biogas_t_sel,
            peak_hr_quantile = peak_hr_quantile,
            n_full_power_hours_expected_per_day_at_peak_price = n_full_power_hours_expected_per_day_at_peak_price,
            batch_size=batch_size,
        )

        # Assign batch results to main time series arrays
        # hpp
        p_hpp_t[batch] = p_hpp_t_batch
        p_curtailment_t[batch] = p_curtailment_t_batch

        # cpv
        alpha_cpv_t[batch] = alpha_cpv_t_batch
        p_cpv_t[batch] = p_cpv_t_batch

        # cst
        alpha_cst_t[batch] = alpha_cst_t_batch
        p_st_t[batch] = p_st_t_batch
        flow_ms_heat_exchanger_t[batch] = flow_ms_heat_exchanger_t_batch
        flow_ms_q_t[batch] = flow_ms_q_t_batch
        v_hot_ms_t[batch] = v_hot_ms_t_batch
        q_t[batch] = q_t_batch
        p_st_max_dni_t[batch] = p_st_max_dni_t_batch
        q_max_dni_t[batch] = q_max_dni_t_batch
        flow_steam_st_t[batch] = flow_steam_st_t_batch

        # biogas_h2
        alpha_h2_t[batch] = alpha_h2_t_batch
        biogas_h2_procuded_h2_kg_in_dni_reactor_t[batch] = biogas_h2_procuded_h2_kg_in_dni_reactor_t_batch
        biogas_h2_procuded_h2_kg_in_el_reactor_t[batch] = biogas_h2_procuded_h2_kg_in_el_reactor_t_batch
        h2_t[batch] = h2_t_batch
        p_biogas_h2_t[batch] = p_biogas_h2_t_batch
        biogas_t[batch] = biogas_t_batch
        water_t[batch] = water_t_batch
        co2_t[batch] = co2_t_batch

        penalty_t[batch] = penalty_t_batch
        penalty_q_t[batch] = penalty_q_t_batch

    return p_hpp_t, p_curtailment_t, p_cpv_t, p_st_t, p_biogas_h2_t, penalty_t, penalty_q_t, flow_ms_heat_exchanger_t, flow_ms_q_t, v_hot_ms_t, alpha_cpv_t, alpha_cst_t, alpha_h2_t,\
           h2_t, q_t, biogas_t, water_t, co2_t, p_st_max_dni_t, q_max_dni_t, flow_steam_st_t, biogas_h2_procuded_h2_kg_in_dni_reactor_t, biogas_h2_procuded_h2_kg_in_el_reactor_t

def ems_cplex_solarX_parts(
        # grid
        hpp_grid_connection,
        max_el_buy_from_grid_mw,
        grid_h2_capacity,

        # cpv
        p_cpv_max_dni_t,
        cpv_inverter_mw,
        cpv_rated_mw,

        # cst
        flow_ms_max_t,
        v_molten_salt_tank_m3,
        delta_q_hot_cold_ms_per_kg,
        p_rated_st,
        efficiency_st,
        hot_tank_efficiency,
        heat_exchanger_capacity,
        v_max_hot_ms_percentage,
        v_min_hot_ms_percentage,
        flow_ms_max_cst_receiver_capacity,
        steam_specific_heat_capacity,
        hot_steam_temp_ms,
        cold_steam_temp_ms,
        hot_molten_salt_density,
        heat_penalty_euro_per_mwht,
        heat_exchanger_efficiency,

        # biogas_h2
        heat_mwht_per_kg_h2,
        biogas_h2_reactor_dni_to_heat_efficiency,
        biogas_h2_reactor_el_to_heat_efficiency,
        biogas_h2_reactor_efficiency_curve,
        area_dni_reactor_biogas_h2,
        area_el_reactor_biogas_h2,
        maximum_h2_production_reactor_kg_per_m2,
        max_solar_flux_dni_reactor_biogas_h2_t,
        biogas_h2_mass_ratio,
        water_h2_mass_ratio,
        co2_h2_mass_ratio,

        # prices
        price_el_t,
        price_h2_t,
        demand_q_t,
        price_water_t,
        price_co2_t,
        price_biogas_t,
        peak_hr_quantile,
        n_full_power_hours_expected_per_day_at_peak_price,
        batch_size,
    ):
    """
    Energy Management System (EMS) Optimization Parts for SolarX Plant.

    This function handles specific optimization calculations and constraints for CPV, hydrogen production, and molten salt storage
    in the SolarX energy plant. It separates parts of the EMS model for improved performance and parallel optimization.

    Parameters
    ----------
    hpp_grid_connection : float
        Maximum grid connection capacity in MW.
    grid_h2_capacity : float
        Maximum grid hydrogen capacity in kg/h.
    cpv_inverter_mw : float
        Rated capacity of the CPV inverter in MW.
    cpv_rated_mw : float
        Rated capacity of the CPV receiver in MW.
    p_cpv_max_dni_t : pd.Series
        Maximum CPV power time series based on DNI.
    flow_ms_max_t : pd.Series
        Maximum molten salt flow rate time series in kg/h.
    v_molten_salt_tank_m3 : float
        Volume of the hot molten salt tank in cubic meters.
    delta_q_hot_cold_ms_per_kg : float
        Heat difference between hot and cold molten salt.
    biogas_h2_mass_ratio : float
        Ratio of biogas required per kg hydrogen.
    water_h2_mass_ratio : float
        Ratio of water required per kg hydrogen.
    co2_h2_mass_ratio : float
        Ratio of CO2 required per kg hydrogen.
    p_rated_st : float
        Rated power for steam turbine.
    efficiency_st : float
        Efficiency of steam turbine.
    heat_exchanger_capacity : float
        Maximum capacity of steam heat exchanger.
    v_max_hot_ms_percentage : float
        Maximum allowable molten salt volume percentage.
    v_min_hot_ms_percentage : float
        Minimum allowable molten salt volume percentage.
    biogas_h2_reactor_dni_to_heat_efficiency : float
        DNI to heat efficiency for hydrogen reactor.
    biogas_h2_reactor_el_to_heat_efficiency : float
        Electricity to heat efficiency for hydrogen reactor.
    biogas_h2_reactor_efficiency_curve : dict
        Efficiency curve for biogas reactor.
    area_dni_reactor_biogas_h2 : float
        Area of the DNI reactor in m².
    area_el_reactor_biogas_h2 : float
        Area of the EL reactor in m².
    maximum_h2_production_reactor_kg_per_m2 : float
        Maximum hydrogen production in DNI reactor.
    heat_mwht_per_kg_h2 : float
        Required heat to produce 1 kg hydrogen.
    ... [remaining parameters]

    Returns
    -------
    tuple of np.ndarray
        Tuple containing arrays for key metrics across the plant:
        - p_hpp_t : ndarray
            HPP power time series.
        - p_curtailment_t : ndarray
            HPP curtailed power.
        - p_cpv_t : ndarray
            CPV power time series.
        - p_st_t : ndarray
            Steam turbine power time series.
        - p_biogas_h2_t : ndarray
            Hydrogen production power time series.
        - penalty_t : ndarray
            Penalty for unmet peak hour requirements.
        - flow_ms_heat_exchanger_t : ndarray
            Molten salt flow in steam generator.
        - flow_ms_q_t : ndarray
            Molten salt flow for heat production.
        - v_hot_ms_t : ndarray
            Volume of hot molten salt.
        - alpha_cpv_t : ndarray
            Share of flux to CPV.
        - alpha_h2_t : ndarray
            Share of flux to H2.
        - h2_t : ndarray
            H2 production time series.
        - q_t : ndarray
            Heat production time series.
        - consumed_biogas_t : ndarray
            Biogas consumption.
        - consumed_water_t : ndarray
            Water consumption.
        - consumed_co2_t : ndarray
            CO2 consumption.
        - p_st_max_dni_t : ndarray
            Maximum power in steam turbine.
        - q_max_dni_t : ndarray
            Maximum heat production.
        - flow_steam_st_t : ndarray
            Steam flow in turbine.
        - biogas_h2_procuded_h2_kg_in_dni_reactor_t : ndarray
            H2 produced in DNI reactor.
        - biogas_h2_procuded_h2_kg_in_el_reactor_t : ndarray
            H2 produced in EL reactor.
    """

    # Calculate penalties for unmet power during peak price hours
    time = price_el_t.index
    N_t = len(price_el_t.index)
    N_days = N_t/24
    e_peak_day_expected = n_full_power_hours_expected_per_day_at_peak_price*hpp_grid_connection 
    e_peak_period_expected = e_peak_day_expected*N_days
    price_peak = np.quantile(price_el_t.values, peak_hr_quantile)
    peak_hours_index = np.where(price_el_t>=price_peak)[0]

    # Calculate maximum and minimum tank volumes for constraints
    v_max_hot = v_max_hot_ms_percentage * v_molten_salt_tank_m3
    v_min_hot = v_min_hot_ms_percentage * v_molten_salt_tank_m3

    # Price information for penalty calculation
    price_el_t_to_max = price_peak - price_el_t
    price_el_t_to_max.loc[price_el_t_to_max<0] = 0
    price_el_t_to_max.iloc[:-1] = 0.5*price_el_t_to_max.iloc[:-1].values + 0.5*price_el_t_to_max.iloc[1:].values

    # Initialize CPLEX model
    mdl = Model(name='EMS')
    mdl.context.cplex_parameters.threads = 1
    mdl.context.cplex_parameters.emphasis.mip = 1
    mdl.context.cplex_parameters.timelimit = 2
    cpx = mdl.get_cplex()
    cpx.parameters.simplex.tolerances.optimality.set(1e-6)

    # time set with an additional time slot for the last soc
    SOCtime = time.append(pd.Index([time[-1] + pd.Timedelta('1hour')]))

    # Define optimization variables
    # Power output and curtailment for the solar plant
    p_hpp_t = mdl.continuous_var_dict(time, lb=-max_el_buy_from_grid_mw, ub=hpp_grid_connection, name='HPP power output')
    p_curtailment_t = mdl.continuous_var_dict(time, lb=0, name='Curtailment')

    # Flux allocation to cpv and H2 modules
    alpha_cpv_t = mdl.continuous_var_dict(time, lb=0, ub=1, name='share of the flux going to cpv module')
    alpha_h2_t = mdl.continuous_var_dict(time, lb=0, ub=1, name='share of the flux going to h2 module')
    alpha_cst_t = mdl.continuous_var_dict(time, lb=0, ub=1, name='share of the flux going to cst module')

    # Molten salt tank volume and flow variables
    v_hot_ms_t = mdl.continuous_var_dict(time, lb=v_min_hot, ub=v_max_hot, name='Volume of hot molten salt')
    flow_ms_t = mdl.continuous_var_dict(time, lb=0, ub=flow_ms_max_cst_receiver_capacity,
                                        name='Flow of molten salt into the hot tank')
    flow_ms_heat_exchanger_t = mdl.continuous_var_dict(time, lb=0, name='Flow of molten salt into the steam generator')
    flow_ms_st_t = mdl.continuous_var_dict(time, lb=0, name='Flow of molten salt used for steam turbine')
    flow_ms_q_t = mdl.continuous_var_dict(time, lb=0, name='Flow of molten salt for q sale')
    flow_steam_st_t = mdl.continuous_var_dict(time, lb=0, name='steam going to steam turbine')
    flow_steam_q_t = mdl.continuous_var_dict(time, lb=0, name='steam going for heat sale')
    p_st_t = mdl.continuous_var_dict(time, lb=0, ub=p_rated_st, name='steam turbine power time series')
    p_cpv_t = mdl.continuous_var_dict(time, lb=0, ub=min(cpv_inverter_mw, cpv_rated_mw), name='cpv power time series')
    q_t = mdl.continuous_var_dict(time, lb=0, name='heat generation time series')
    penalty_q_t = mdl.continuous_var_dict(time, lb=0, name='heat penalty time series')

    # Hydrogen production variables
    biogas_h2_procuded_h2_kg_in_dni_reactor_t = mdl.continuous_var_dict(time, lb=0, ub=area_dni_reactor_biogas_h2 * maximum_h2_production_reactor_kg_per_m2,  name='h2 produced in dni reactor')
    biogas_h2_procuded_h2_kg_in_el_reactor_t = mdl.continuous_var_dict(time, lb=0, ub=area_el_reactor_biogas_h2 * maximum_h2_production_reactor_kg_per_m2, name='h2 produced in el reactor')
    p_biogas_h2_t = mdl.continuous_var_dict(time, lb=0, name='el for H2 production time series')
    biogas_h2_el_reactor_heat_per_m2_t = mdl.continuous_var_dict(time, lb=0, name='heat from el in el_reactor of biogas_h2 time series')
    h2_t = mdl.continuous_var_dict(time, lb=0, ub=grid_h2_capacity, name='h2 production time series from solar')
    consumed_biogas_t = mdl.continuous_var_dict(time, lb=0, name='consumed biogas time series')
    consumed_water_t = mdl.continuous_var_dict(time, lb=0, name='consumed water time series')
    consumed_co2_t = mdl.continuous_var_dict(time, lb=0, name='consumed co2 time series')
    biogas_h2_dni_reactor_effictive_heat_mwt_per_m2_t = mdl.continuous_var_dict(time, name="biogas_h2_dni_reactor_effictive_heat_mwt_per_m2_t")
    biogas_h2_el_reactor_effictive_heat_mwt_per_m2_t = mdl.continuous_var_dict(time, name="biogas_h2_el_reactor_effictive_heat_mwt_per_m2_t")
    # effective heat in biogas_h2 reactors
    biogas_h2_heat_mwt_per_m2_efficiency_curve = biogas_h2_reactor_efficiency_curve['heat_mwt_per_m2']
    biogas_h2_efficiency_values_efficiency_curve = biogas_h2_reactor_efficiency_curve['efficiencies']
    biogas_h2_effictive_heat_mwt_per_m2_efficiency_curve = np.array(biogas_h2_efficiency_values_efficiency_curve) * np.array(biogas_h2_heat_mwt_per_m2_efficiency_curve)
    biogas_h2_reactor_effictive_heat_mwt_per_m2 = mdl.piecewise(
        breaksxy=list(zip(biogas_h2_heat_mwt_per_m2_efficiency_curve, biogas_h2_effictive_heat_mwt_per_m2_efficiency_curve)),
        preslope=0,
        postslope=0,
        name="biogas_h2_reactor_effictive_heat_mwt_per_m2")

    # import matplotlib.pyplot as plt
    # plt.plot(biogas_h2_heat_mwt_per_m2_efficiency_curve, biogas_h2_effictive_heat_mwt_per_m2_efficiency_curve, '-ok')
    # plt.show()


    # Penalty for unmet peak hour requirements
    penalty = mdl.continuous_var(name='penalty', lb=-1e12)
    e_penalty = mdl.continuous_var(name='e_penalty', lb=-1e12)

    # Objective function: maximize revenue and minimize penalties
    mdl.maximize(
        mdl.sum(
            price_el_t[t] * p_hpp_t[t]
            + price_h2_t[t] * h2_t[t]
            - price_biogas_t[t] * consumed_biogas_t[t]
            - price_water_t[t] * consumed_water_t[t]
            - price_co2_t[t] * consumed_co2_t[t]
            - penalty_q_t[t]
            for t in time) - penalty)

    # Constraints
    # Penalty constraint based on unmet energy at peak hours
    mdl.add_constraint(e_penalty == ( e_peak_period_expected - mdl.sum(p_hpp_t[time[i]] for i in peak_hours_index) ))
    f1 = mdl.piecewise(0, [(0,0)], 1)
    mdl.add_constraint( penalty == price_peak*f1(e_penalty) )

    # Apply constraints over each time step
    for t in time:
        # Calculate indices for time steps and deltas
        tt = t + pd.Timedelta('1hour')
        t_1 = t if t==time[0] else t - pd.Timedelta('1hour')
        dt = 1 # delta_t of 1 hour

        # Initialize the molten salt tank volume at the starting timestep
        if t == time[0]:
            mdl.add_constraint(v_hot_ms_t[t] == v_min_hot)
        
        # Power balance equation for the solar plant
        mdl.add_constraint(
            p_hpp_t[t] ==
            + p_cpv_t[t] + p_st_t[t]
            - p_biogas_h2_t[t]
            - p_curtailment_t[t])

        # Constraints for electricity generation in steam turbine
        steam_specific_heat_inlet_outlet = steam_specific_heat_capacity * (hot_steam_temp_ms - cold_steam_temp_ms) # kJ/kg
        mdl.add_constraint(flow_steam_st_t[t] == heat_exchanger_efficiency * delta_q_hot_cold_ms_per_kg * flow_ms_st_t[t] / steam_specific_heat_inlet_outlet)# kg/h: [kJ/kg] * [kg/h] / [kg/kg] = [kJ/h]
        mdl.add_constraint(p_st_t[t] == (efficiency_st * steam_specific_heat_inlet_outlet * flow_steam_st_t[t])/3.6e6) # MW: [kJ/kg] * [kg/h] = [kJ/h] = [MW] / 3.6e6

        mdl.add_constraint(p_cpv_t[t] == alpha_cpv_t[t] * p_cpv_max_dni_t[t])

        # Heat production from molten salt flow
        mdl.add_constraint(flow_ms_heat_exchanger_t[t] <= 3.6e6 * heat_exchanger_capacity / delta_q_hot_cold_ms_per_kg )
        mdl.add_constraint(flow_ms_heat_exchanger_t[t] == flow_ms_st_t[t] + flow_ms_q_t[t])
        mdl.add_constraint(flow_steam_q_t[t] == heat_exchanger_efficiency * delta_q_hot_cold_ms_per_kg * flow_ms_q_t[t] / steam_specific_heat_inlet_outlet)
        mdl.add_constraint(q_t[t] == hot_tank_efficiency * steam_specific_heat_inlet_outlet * flow_steam_q_t[t] / 3.6e6) # MWq: [kJ/kg] * [kg/h] = [kJ/h] = [MW] / 3.6e6
        mdl.add_constraint(penalty_q_t[t] == (demand_q_t[t] - q_t[t]) * heat_penalty_euro_per_mwht)

        # Volume balance for molten salt in the hot tank
        mdl.add_constraint(v_hot_ms_t[t] == v_hot_ms_t[t_1] + (flow_ms_t[t] - flow_ms_heat_exchanger_t[t]) * dt / hot_molten_salt_density)
        mdl.add_constraint(flow_ms_t[t] == alpha_cst_t[t] * flow_ms_max_t[t])

        # Ensure flux sharing limits between cpv and H2 modules
        mdl.add_range(0, alpha_cpv_t[t], 1, "Limits of aplpha cpv")
        mdl.add_range(0, alpha_h2_t[t], 1, "Limits of aplpha h2")
        mdl.add_range(0, alpha_cst_t[t], 1, "Limits of aplpha h2")
        mdl.add_constraint(alpha_cpv_t[t] + alpha_h2_t[t] + alpha_cst_t[t] <= 1, "sum of cpv, cst, and h2 share")

        # Constraints for hydrogen production in DNI and electrical reactors
        # dni
        if area_dni_reactor_biogas_h2 != 0:
            mdl.add_constraint(biogas_h2_dni_reactor_effictive_heat_mwt_per_m2_t[t] == biogas_h2_reactor_effictive_heat_mwt_per_m2(biogas_h2_reactor_dni_to_heat_efficiency * alpha_h2_t[t] * max_solar_flux_dni_reactor_biogas_h2_t[t] / area_dni_reactor_biogas_h2))
        else:
            mdl.add_constraint(biogas_h2_dni_reactor_effictive_heat_mwt_per_m2_t[t] == 0)

        mdl.add_constraint(biogas_h2_procuded_h2_kg_in_dni_reactor_t[t] == biogas_h2_dni_reactor_effictive_heat_mwt_per_m2_t[t] * area_dni_reactor_biogas_h2 / heat_mwht_per_kg_h2)

        # el
        mdl.add_constraint(p_biogas_h2_t[t] == biogas_h2_el_reactor_heat_per_m2_t[t] * area_el_reactor_biogas_h2 / biogas_h2_reactor_el_to_heat_efficiency)
        mdl.add_constraint(biogas_h2_el_reactor_effictive_heat_mwt_per_m2_t[t] == biogas_h2_reactor_effictive_heat_mwt_per_m2(biogas_h2_el_reactor_heat_per_m2_t[t]))
        mdl.add_constraint(biogas_h2_procuded_h2_kg_in_el_reactor_t[t] == biogas_h2_el_reactor_effictive_heat_mwt_per_m2_t[t] * area_el_reactor_biogas_h2 / heat_mwht_per_kg_h2)

        # Total hydrogen production is the sum of DNI and electrical reactor outputs
        mdl.add_constraint(h2_t[t] == biogas_h2_procuded_h2_kg_in_el_reactor_t[t] + biogas_h2_procuded_h2_kg_in_dni_reactor_t[t])

        # Biogas, water, and CO2 consumption for hydrogen production
        mdl.add_constraint(consumed_biogas_t[t] == biogas_h2_mass_ratio * h2_t[t])
        mdl.add_constraint(consumed_water_t[t] == water_h2_mass_ratio * h2_t[t])
        mdl.add_constraint(consumed_co2_t[t] == co2_h2_mass_ratio * h2_t[t])

        # Ensure that molten salt volume at end of time series returns to minimum
        if t == time[-1]:
            mdl.add_constraint(v_hot_ms_t[t] == v_min_hot)

    # Solve the model and retrieve solution values
    sol = mdl.solve(log_output=False)

    # Extract results from the solution
    # hpp
    p_hpp_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(p_hpp_t), orient='index').loc[:,0]
    p_curtailment_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(p_curtailment_t), orient='index').loc[:,0]

    # cpv
    alpha_cpv_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(alpha_cpv_t), orient='index').loc[:,0]
    p_cpv_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(p_cpv_t), orient='index').loc[:,0]

    # cst
    alpha_cst_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(alpha_cst_t), orient='index').loc[:, 0]
    flow_ms_heat_exchanger_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(flow_ms_heat_exchanger_t), orient='index').loc[:,0]
    flow_steam_st_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(flow_steam_st_t), orient='index').loc[:, 0]
    flow_ms_q_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(flow_ms_q_t), orient='index').loc[:,0]
    p_st_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(p_st_t), orient='index').loc[:,0]
    q_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(q_t), orient='index').loc[:, 0]
    penalty_q_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(penalty_q_t), orient='index').loc[:, 0]
    v_hot_ms_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(v_hot_ms_t), orient='index').loc[:,0]

    # biogas_h2
    alpha_h2_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(alpha_h2_t), orient='index').loc[:,0]
    biogas_h2_procuded_h2_kg_in_dni_reactor_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(biogas_h2_procuded_h2_kg_in_dni_reactor_t), orient='index').loc[:, 0]
    biogas_h2_procuded_h2_kg_in_el_reactor_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(biogas_h2_procuded_h2_kg_in_el_reactor_t), orient='index').loc[:, 0]
    h2_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(h2_t), orient='index').loc[:, 0]
    p_biogas_h2_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(p_biogas_h2_t), orient='index').loc[:,0]
    consumed_biogas_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(consumed_biogas_t), orient='index').loc[:, 0]
    consumed_water_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(consumed_water_t), orient='index').loc[:, 0]
    consumed_co2_t_df = pd.DataFrame.from_dict(
        sol.get_value_dict(consumed_co2_t), orient='index').loc[:, 0]

    #make a time series like p_hpp with a constant penalty
    penalty_2 = sol.get_value(penalty)
    penalty_t = np.ones(N_t) * (penalty_2/N_t)

    mdl.end()

    # Cplex sometimes returns missing values :O
    # hpp
    p_hpp_t = p_hpp_t_df.reindex(time,fill_value=0).values
    p_curtailment_t = p_curtailment_t_df.reindex(time,fill_value=0).values

    # cpv
    alpha_cpv_t = alpha_cpv_t_df.reindex(time,fill_value=0).values
    p_cpv_t = p_cpv_t_df.reindex(time,fill_value=0).values

    # cst
    alpha_cst_t = alpha_cst_t_df.reindex(time, fill_value=0).values
    flow_ms_heat_exchanger_t = flow_ms_heat_exchanger_t_df.reindex(time,fill_value=0).values
    flow_steam_st_t = flow_steam_st_t_df.reindex(time,fill_value=0).values
    v_hot_ms_t = v_hot_ms_t_df.reindex(time,fill_value=0).values
    flow_ms_q_t = flow_ms_q_t_df.reindex(time,fill_value=0).values
    p_st_t = p_st_t_df.reindex(time,fill_value=0).values
    q_t = q_t_df.reindex(time,fill_value=0).values
    penalty_q_t = penalty_q_t_df.reindex(time,fill_value=0).values

    # biogas_h2
    alpha_h2_t = alpha_h2_t_df.reindex(time,fill_value=0).values
    biogas_h2_procuded_h2_kg_in_dni_reactor_t = biogas_h2_procuded_h2_kg_in_dni_reactor_t_df.reindex(time,fill_value=0).values
    biogas_h2_procuded_h2_kg_in_el_reactor_t = biogas_h2_procuded_h2_kg_in_el_reactor_t_df.reindex(time,fill_value=0).values
    p_biogas_h2_t = p_biogas_h2_t_df.reindex(time,fill_value=0).values
    h2_t = h2_t_df.reindex(time,fill_value=0).values
    consumed_biogas_t = consumed_biogas_t_df.reindex(time,fill_value=0).values
    consumed_water_t = consumed_water_t_df.reindex(time,fill_value=0).values
    consumed_co2_t = consumed_co2_t_df.reindex(time,fill_value=0).values

    # new outputs
    flow_steam_max_dni_t = np.array([(heat_exchanger_efficiency * delta_q_hot_cold_ms_per_kg * flow_ms_max_t.iloc[t] / steam_specific_heat_inlet_outlet) for t in range(len(flow_ms_max_t))])
    p_st_max_dni_t = np.array([(efficiency_st * steam_specific_heat_inlet_outlet * flow_steam_max_dni_t[t])/3.6e6 for t in range(len(flow_steam_st_t))])
    q_max_dni_t = np.array([(delta_q_hot_cold_ms_per_kg * flow_ms_max_t.iloc[t] / 3.6e6) for t in range(len(flow_ms_max_t))])

    return p_hpp_t, p_curtailment_t, p_cpv_t, p_st_t, p_biogas_h2_t, penalty_t, penalty_q_t, flow_ms_heat_exchanger_t, flow_ms_q_t, v_hot_ms_t, alpha_cpv_t, alpha_cst_t, alpha_h2_t, h2_t, q_t, consumed_biogas_t, consumed_water_t,\
           consumed_co2_t, p_st_max_dni_t, q_max_dni_t, flow_steam_st_t, biogas_h2_procuded_h2_kg_in_dni_reactor_t, biogas_h2_procuded_h2_kg_in_el_reactor_t


# -----------------------------------------------------------------------
# Auxiliar functions for ems modelling
# -----------------------------------------------------------------------
def expand_to_lifetime(x, life_h=25 * 365 * 24, weeks_per_season_per_year=None):
    """
    Expands (by repeating) a given variable to match an expected lifetime length.

    If weeks_per_season_per_year is an int then it will first build a year out of the selected weeks

    Parameters
    ----------
    x: input variable
    life_h: lifetime in hours.
    weeks_per_season_per_year: None or int.

    Returns
    -------
    x_ext: extended variable
    """
    if weeks_per_season_per_year == None:

        # Extend the data to match the expected lifetime
        len_x = len(x)
        N_repeats = int(np.ceil(life_h / len_x))

    else:
        x_ext = np.array([])

        # extend selected weeks into a year: 4 season of 13 weeks + one extra day.
        for x_batch in split_in_batch(x, weeks_per_season_per_year * 7 * 24):
            x_ext = np.append(x_ext, np.tile(x_batch, 20)[:24 * 7 * 13])
        x_ext = np.append(x_ext, x_batch[-24:])

        # extend the constructed year to match the expected lifetime
        x = x_ext
        N_repeats = int(np.ceil(life_h / 365 * 24))

    return np.tile(x, N_repeats)[:life_h]

def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N:end]]
        counter += 1
    return batch




