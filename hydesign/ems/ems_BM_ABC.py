import numpy as np
import pandas as pd
import openmdao.api as om

# Define the sample data
N_time = 24 * 7  # One week of hourly data
life_h = 25 * 365 * 24  # 25 years of hourly data

# Create dummy time series data
time_index = pd.date_range(start='01-01-2024 00:00', periods=N_time, freq='H')
wind_t = np.random.rand(N_time) * 100  # Random wind power generation
solar_t = np.random.rand(N_time) * 50  # Random solar power generation
price_t = np.random.rand(N_time) * 100  # Random electricity prices
wind_BM_t = wind_t * 0.9  # Assume some difference for BM
price_up_reg_t = price_t * 1.1  # Slightly higher prices for up regulation
price_dwn_reg_t = price_t * 0.9  # Slightly lower prices for down regulation
SO_imbalance_t = np.random.randn(N_time) * 10  # Random imbalance values

# Define parameters
b_P = 50  # Battery power capacity [MW]
b_E = 200  # Battery energy storage capacity [MW]
G_MW = 150  # Grid capacity [MW]
penalty_BM = 100  # Penalty value
bi_directional_status = 1  # Allow bi-directional flow
battery_depth_of_discharge = 0.8  # Battery depth of discharge
battery_charge_efficiency = 0.9  # Battery charge efficiency
peak_hr_quantile = 0.9  # 90th percentile for peak hours
cost_of_battery_P_fluct_in_peak_price_ratio = 0.5
n_full_power_hours_expected_per_day_at_peak_price = 3

# Create the EMS component
ems_comp = ems(N_time=N_time, life_h=life_h, ems_type='cplex')

# Add the EMS component to a problem
prob = om.Problem()
prob.model.add_subsystem('ems', ems_comp, promotes=['*'])

# Set the inputs
prob.setup()

prob.set_val('wind_t', wind_t)
prob.set_val('wind_BM_t', wind_BM_t)
prob.set_val('solar_t', solar_t)
prob.set_val('price_t', price_t)
prob.set_val('price_up_reg_t', price_up_reg_t)
prob.set_val('price_dwn_reg_t', price_dwn_reg_t)
prob.set_val('SO_imbalance_t', SO_imbalance_t)
prob.set_val('b_P', b_P)
prob.set_val('penalty_BM', penalty_BM)
prob.set_val('bi_directional_status', bi_directional_status)
prob.set_val('b_E', b_E)
prob.set_val('G_MW', G_MW)
prob.set_val('battery_depth_of_discharge', battery_depth_of_discharge)
prob.set_val('battery_charge_efficiency', battery_charge_efficiency)
prob.set_val('peak_hr_quantile', np.array([peak_hr_quantile]))
prob.set_val('cost_of_battery_P_fluct_in_peak_price_ratio', np.array([cost_of_battery_P_fluct_in_peak_price_ratio]))
prob.set_val('n_full_power_hours_expected_per_day_at_peak_price', np.array([n_full_power_hours_expected_per_day_at_peak_price]))

# Run the model
prob.run_model()

# Retrieve the outputs
wind_t_ext = prob.get_val('wind_t_ext')
solar_t_ext = prob.get_val('solar_t_ext')
price_t_ext = prob.get_val('price_t_ext')
hpp_t = prob.get_val('hpp_t')
b_t = prob.get_val('b_t')
b_E_SOC_t = prob.get_val('b_E_SOC_t')
penalty_t = prob.get_val('penalty_t')
total_curtailment = prob.get_val('total_curtailment')

# Print the outputs
print('wind_t_ext:', wind_t_ext)
print('solar_t_ext:', solar_t_ext)
print('price_t_ext:', price_t_ext)
print('hpp_t:', hpp_t)
print('b_t:', b_t)
print('b_E_SOC_t:', b_E_SOC_t)
print('penalty_t:', penalty_t)
print('total_curtailment:', total_curtailment)

# Create the long term operation component
long_term_op_comp = ems_long_term_operation(N_time=life_h, ems_type='No_degradation')

# Add the long term operation component to a new problem
prob_long_term = om.Problem()
prob_long_term.model.add_subsystem('ems_long_term_op', long_term_op_comp, promotes=['*'])

# Set the inputs
prob_long_term.setup()

SoH = np.linspace(1, 0.8, life_h)  # Dummy State of Health data
prob_long_term.set_val('SoH', SoH)
prob_long_term.set_val('wind_t_ext_deg', wind_t_ext)
prob_long_term.set_val('b_P', b_P)
prob_long_term.set_val('b_E', b_E)
prob_long_term.set_val('battery_depth_of_discharge', battery_depth_of_discharge)
prob_long_term.set_val('battery_charge_efficiency', battery_charge_efficiency)
prob_long_term.set_val('penalty_BM', penalty_BM)
prob_long_term.set_val('hpp_curt_BM_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('b_t', b_t[:life_h])
prob_long_term.set_val('b_BM_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('b_E_SOC_t', b_E_SOC_t[:life_h + 1])
prob_long_term.set_val('P_up_reg_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('P_dwn_reg_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('P_up_max_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('P_dwn_max_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('hpp_t', hpp_t[:life_h])
prob_long_term.set_val('price_up_reg_t', np.zeros(life_h))  # Dummy data
prob_long_term.set_val('price_dwn_reg_t', np.zeros(life_h))  # Dummy data

# Run the model
prob_long_term.run_model()

# Retrieve the outputs
hpp_t_with_deg = prob_long_term.get_val('hpp_t_with_deg')
hpp_curt_t_with_deg = prob_long_term.get_val('hpp_curt_t_with_deg')
b_t_with_deg = prob_long_term.get_val('b_t_with_deg')
b_E_SOC_t_with_deg = prob_long_term.get_val('b_E_SOC_t_with_deg')
penalty_t_with_deg = prob_long_term.get_val('penalty_t_with_deg')
total_curtailment_deg = prob_long_term.get_val('total_curtailment_deg')

# Print the outputs
print('hpp_t_with_deg:', hpp_t_with_deg)
print('hpp_curt_t_with_deg:', hpp_curt_t_with_deg)
print('b_t_with_deg:', b_t_with_deg)
print('b_E_SOC_t_with_deg:', b_E_SOC_t_with_deg)
print('penalty_t_with_deg:', penalty_t_with_deg)
print('total_curtailment_deg:', total_curtailment_deg)
