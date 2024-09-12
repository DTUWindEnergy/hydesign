
import pandas as pd
import numpy as np
from hydesign.assembly.hpp_assembly_constantoutput import hpp_model_constant_output as hpp_model
from hydesign.examples import examples_filepath
import matplotlib.pyplot as plt
import time
#from ems import ems_cplex_parts_constantoutput
# Evaluation
examples_sites = pd.read_csv(f'{examples_filepath}examples_sites.csv', index_col=0, sep=';')
name = 'France_good_wind'
ex_site = examples_sites.loc[examples_sites.name == name]
longitude = ex_site['longitude'].values[0]
latitude = ex_site['latitude'].values[0]
altitude = ex_site['altitude'].values[0]
input_ts_fn = examples_filepath+ex_site['input_ts_fn'].values[0]
sim_pars_fn = examples_filepath+ex_site['sim_pars_fn'].values[0]
H2_demand_fn = examples_filepath+ex_site['H2_demand_col'].values[0]

PPA = 40 # Euro/MWh
hpp = hpp_model(
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,
    num_batteries = 1,
    battery_deg = False,
    work_dir = './',
    sim_pars_fn = sim_pars_fn,
    input_ts_fn = input_ts_fn,
    ppa_price=PPA,)
        # Wind plant design
        #x=[clearance, sp, p_rated, Nwt, wind_MW_per_km2,
        # PV plant design
        #solar_MW,  surface_tilt, surface_azimuth, DC_AC_ratio,
        # Energy storage & EMS price constrains
        # b_P, b_E_h, cost_of_battery_P_fluct_in_peak_price_ratio]

number_of_runs = 1        
x=[35.0, 302.8, 5, 1, 7.0, 150, 25.0, 180.0, 1.0, 30, 0, 10.0]
times_array1 = np.zeros((number_of_runs, 1))
for i in range(number_of_runs):
    start_time = time.time()
    outs = hpp.evaluate(*x)
    times_array1[i] = time.time() - start_time

    wind_t = hpp.prob.get_val('wpp_with_degradation.wind_t_ext_deg')
    solar_t = hpp.prob.get_val('pvp_with_degradation.solar_t_ext_deg')

    hpp_t = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')
    hpp_curt_t =hpp.prob.get_val('ems_long_term_operation.hpp_curt_t')
    b_E_SOC_t = hpp.prob.get_val('ems_long_term_operation.b_E_SOC_t')
    b_t = hpp.prob.get_val('ems_long_term_operation.b_t')
    penalty_t = hpp.prob.get_val('ems_long_term_operation.penalty_t_with_deg')

output_timeseries = pd.concat([pd.DataFrame(wind_t, columns=['Wind Production (MW)']), 
                                pd.DataFrame(solar_t, columns=['Solar Production (MW)']), 
                                pd.DataFrame(hpp_t, columns=['HPP send to grid (MW)']), 
                                pd.DataFrame(hpp_curt_t, columns=['Curtailment (MW)']), 
                                pd.DataFrame(b_E_SOC_t, columns=['SoC (MWh)']), 
                                pd.DataFrame(b_t, columns=['discharge/charge (MW)']),
                                pd.DataFrame(penalty_t, columns=['penalty_t'])], axis=1)
output_timeseries.to_csv('output1.csv')
#output_timeseries.to_excel('output.xlsx', sheet_name='power')
output_financial = pd.DataFrame([])
for i, var in enumerate(hpp.list_out_vars):
    output_financial = pd.concat([output_financial, pd.DataFrame(outs[i], columns=var)],axis=1)
output_financial.to_csv('financial1.csv')

x=[35.0, 302.8, 5, 1, 7.0, 150, 25.0, 180.0, 1.0, 5, 5, 10.0]
times_array2 = np.zeros((number_of_runs, 1))
for i in range(number_of_runs):
    start_time = time.time()
    outs = hpp.evaluate(*x)
    times_array2[i] = time.time() - start_time

    wind_t = hpp.prob.get_val('wpp_with_degradation.wind_t_ext_deg')
    solar_t = hpp.prob.get_val('pvp_with_degradation.solar_t_ext_deg')

    hpp_t = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')
    hpp_curt_t =hpp.prob.get_val('ems_long_term_operation.hpp_curt_t')
    b_E_SOC_t = hpp.prob.get_val('ems_long_term_operation.b_E_SOC_t')
    b_t = hpp.prob.get_val('ems_long_term_operation.b_t')
    penalty_t = hpp.prob.get_val('ems_long_term_operation.penalty_t_with_deg')
output_timeseries = pd.concat([pd.DataFrame(wind_t, columns=['Wind Production (MW)']), 
                                pd.DataFrame(solar_t, columns=['Solar Production (MW)']), 
                                pd.DataFrame(hpp_t, columns=['HPP send to grid (MW)']), 
                                pd.DataFrame(hpp_curt_t, columns=['Curtailment (MW)']), 
                                pd.DataFrame(b_E_SOC_t, columns=['SoC (MWh)']), 
                                pd.DataFrame(b_t, columns=['discharge/charge (MW)'])], axis=1)
output_timeseries.to_csv('output2.csv')

x=[35.0, 302.8, 5, 1, 7.0, 150, 25.0, 180.0, 1.0, 20, 5, 10.0]
times_array3 = np.zeros((number_of_runs, 1))
for i in range(number_of_runs):
    start_time = time.time()
    outs = hpp.evaluate(*x)
    times_array3[i] = time.time() - start_time

    wind_t = hpp.prob.get_val('wpp_with_degradation.wind_t_ext_deg')
    solar_t = hpp.prob.get_val('pvp_with_degradation.solar_t_ext_deg')

    hpp_t = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')
    hpp_curt_t =hpp.prob.get_val('ems_long_term_operation.hpp_curt_t')
    b_E_SOC_t = hpp.prob.get_val('ems_long_term_operation.b_E_SOC_t')
    b_t = hpp.prob.get_val('ems_long_term_operation.b_t')
    penalty_t = hpp.prob.get_val('ems_long_term_operation.penalty_t_with_deg')
output_timeseries = pd.concat([pd.DataFrame(wind_t, columns=['Wind Production (MW)']), 
                                pd.DataFrame(solar_t, columns=['Solar Production (MW)']), 
                                pd.DataFrame(hpp_t, columns=['HPP send to grid (MW)']), 
                                pd.DataFrame(hpp_curt_t, columns=['Curtailment (MW)']), 
                                pd.DataFrame(b_E_SOC_t, columns=['SoC (MWh)']), 
                                pd.DataFrame(b_t, columns=['discharge/charge (MW)'])], axis=1)    
output_timeseries.to_csv('output3.csv')    


run_times = pd.concat([pd.DataFrame(times_array1, columns=['scenario 1']), pd.DataFrame(times_array2, columns=['scenario 2']), pd.DataFrame(times_array3, columns=['scenario 3'])], axis=1)
run_times.to_excel('run_times.xlsx', sheet_name='time')

hpp.print_design(x, outs)

b_E_SOC_t = hpp.prob.get_val('ems.b_E_SOC_t')
b_t = hpp.prob.get_val('ems.b_t')
price_t = hpp.prob.get_val('ems.price_t')

wind_t = hpp.prob.get_val('ems.wind_t')
solar_t = hpp.prob.get_val('ems.solar_t')
hpp_t = hpp.prob.get_val('ems.hpp_t')
hpp_curt_t = hpp.prob.get_val('ems.hpp_curt_t')
grid_MW = hpp.prob.get_val('ems.G_MW')

n_days_plot = 14*4
# plt.figure(figsize=[12,4])
# plt.plot(price_t[:24*n_days_plot], label='price')
# plt.plot(b_E_SOC_t[:24*n_days_plot], label='SoC [MWh]')
# plt.plot(b_t[:24*n_days_plot], label='Battery P [MW]')
# plt.plot()
# plt.xlabel('time [hours]')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#            ncol=3, fancybox=0, shadow=0)
# plt.show()

plt.figure(figsize=[12,4])
# plt.plot(wind_t[:24*n_days_plot], label='wind')
# plt.plot(solar_t[:24*n_days_plot] + wind_t[:24*n_days_plot], label='Generation')
plt.plot(hpp_t[:24*n_days_plot], label='HPP')
# plt.plot(hpp_curt_t[:24*n_days_plot], label='HPP curtailed')
plt.axhline(grid_MW, label='Grid MW', color='k')
plt.axhline(hpp.prob['load_min'], label='Min. Load MW', color='grey')
plt.xlabel('time [hours]')
plt.ylabel('Power [MW]')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
           ncol=5, fancybox=0, shadow=0)
plt.show()

# b_E_SOC_t = hpp.prob.get_val('ems.b_E_SOC_t')
# hpp_t = hpp.prob.get_val('ems.hpp_t')
# hpp_curt_t = hpp.prob.get_val('ems.hpp_curt_t')

# b_E_SOC_t_with_deg = hpp.prob.get_val('ems_long_term_operation.b_E_SOC_t_with_deg')
# hpp_t_with_deg = hpp.prob.get_val('ems_long_term_operation.hpp_t_with_deg')
# hpp_curt_t_with_deg = hpp.prob.get_val('ems_long_term_operation.hpp_curt_t_with_deg')

# price_t_ext = hpp.prob.get_val('ems_long_term_operation.price_t_ext')

# # Plot the HPP operation in the 7th year (with and without degradation)
# n_start = int(24*365*7.2)
# n_days_plot = 14

# plt.figure(figsize=[12,4])

# plt.plot(price_t_ext[n_start:n_start+24*n_days_plot], label='price')

# plt.plot(b_E_SOC_t[n_start:n_start+24*n_days_plot], label='SoC [MWh]')
# plt.plot(b_E_SOC_t_with_deg[n_start:n_start+24*n_days_plot], label='SoC with degradation [MWh]')
# plt.xlabel('time [hours]')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#             ncol=5, fancybox=0, shadow=0)
# plt.show()

# plt.figure(figsize=[12,4])
# plt.plot(hpp_t[n_start:n_start+24*n_days_plot], label='HPP')
# plt.plot(hpp_t_with_deg[n_start:n_start+24*n_days_plot], label='HPP with degradation')

# plt.plot(hpp_curt_t[n_start:n_start+24*n_days_plot], label='HPP curtailed')
# plt.plot(hpp_curt_t_with_deg[n_start:n_start+24*n_days_plot], label='HPP curtailed with degradation')

# plt.axhline(grid_MW, label='Grid MW', color='k')
# plt.axhline(hpp.prob['load_min'], label='Min. Load MW', color='r')
# plt.xlabel('time [hours]')
# plt.ylabel('Power [MW]')
# plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
#             ncol=6, fancybox=0, shadow=0)
# plt.show()


N_life = hpp.sim_pars['N_life']
life_h = N_life*365*24
age = np.arange(life_h)/(24*365)

SoH = np.copy(hpp.prob.get_val('battery_degradation.SoH'))
plt.figure(figsize=[12,3])
plt.plot( age, SoH, label=r'$C_{bfl}=0$')
plt.plot( age, 0.7*np.ones_like(age), label=r'$min(1-L) = 0.7$', color='r',alpha=0.5)
plt.xlabel('age [years]')
plt.ylabel(r'Battery State of Health, $1-L(t)$ [-]')
plt.legend(title='Cost of Battery fluctuations',
            loc='upper center', bbox_to_anchor=(0.5, 1.27),
            ncol=3, fancybox=0, shadow=0)
plt.show()   


# Sizing
""" def main():
    if __name__ == '__main__':
        from hydesign.assembly.hpp_assembly_constantoutput import hpp_model_constant_output as hpp_model
        from hydesign.Parallel_EGO import EfficientGlobalOptimizationDriver

        # Simple example to size wind only with a single core to run test machines and colab
        
        inputs = {
            'name': name,
            'longitude': longitude,
            'latitude': latitude,
            'altitude': altitude,
            'input_ts_fn': input_ts_fn,
            'sim_pars_fn': sim_pars_fn,
    
            'opt_var': "NPV_over_CAPEX",
            'num_batteries': 2,
            'n_procs': 1,
            'n_doe': 8,
            'n_clusters': 1,
            'n_seed': 0,
            'max_iter': 2,
            'final_design_fn': 'hydesign_design_0.csv',
            'npred': 3e4,
            'tol': 1e-6,
            'min_conv_iter': 2,
            'work_dir': './',
            'hpp_model': hpp_model,
            'PPA_price': 40,
        'variables': {
            'clearance [m]':
                #{'var_type':'design',
                #  'limits':[10, 60],
                #  'types':'int'
                #  },
                {'var_type':'fixed',
                   'value': 35
                   },
            'sp [W/m2]':
                #{'var_type':'design',
                # 'limits':[200, 359],
                # 'types':'int'
                # },
                {'var_type':'fixed',
                   'value': 300
                   }, 
            'p_rated [MW]':
                {'var_type':'design',
                  'limits':[1, 10],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #  'value': 6
                 # },
            'Nwt':
                {'var_type':'design',
                  'limits':[0, 400],
                  'types':'int'
                  },
                # {'var_type':'fixed',
                #   'value': 200
                #   },
            'wind_MW_per_km2 [MW/km2]':
                #{'var_type':'design',
                #  'limits':[5, 9],
                #  'types':'float'
                #  },
                 {'var_type':'fixed',
                   'value': 7
                   },
            'solar_MW [MW]':
                {'var_type':'design',
                   'limits':[0, 400],
                   'types':'int'
                  },
                #{'var_type':'fixed',
                #  'value': 20
                #  },
            'surface_tilt [deg]':
                # {'var_type':'design',
                #   'limits':[0, 50],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 25
                  },
            'surface_azimuth [deg]':
                # {'var_type':'design',
                #   'limits':[150, 210],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 180
                  },
            'DC_AC_ratio':
                # {'var_type':'design',
                #   'limits':[1, 2.0],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value':1.0,
                  },
            'b_P [MW]':
                {'var_type':'design',
                   'limits':[0, 100],
                   'types':'int'
                  },
                #{'var_type':'fixed',
                #  'value': 50
                #  },
            'b_E_h [h]':
                {'var_type':'design',
                   'limits':[1, 10],
                   'types':'int'
                  },
                #{'var_type':'fixed',
                #  'value': 6
                #  },
            'cost_of_battery_P_fluct_in_peak_price_ratio':
                # {'var_type':'design',
                #   'limits':[0, 20],
                #   'types':'float'
                #   },
                {'var_type':'fixed',
                  'value': 10},
            }}
        EGOD = EfficientGlobalOptimizationDriver(**inputs)
        EGOD.run()
        result = EGOD.result

main() """