import os
import time

import matplotlib.pyplot as plt

from hydesign.assembly.hpp_assembly_solarX import hpp_model_solarX as hpp_model
from hydesign.examples import examples_filepath

# -----------------------------------------------------------
# Setup site-specific data for Hybrid Power Plant simulation
# -----------------------------------------------------------

# Extract geographical information of the selected site
longitude = -6.03
latitude = 37.380
altitude = 10

# # ######################### weather.to_csv(examples_filepath + "solarX/input_ts_" + name + ".csv")


# input_ts_fn = examples_filepath + "solarX/input_ts_Denmark_good_solar.csv"

# -----------------------------------------------------------
# Load simulation parameters and initialize the HPP model
# -----------------------------------------------------------
input_ts_fn = os.path.join(examples_filepath, "Europe/solarX_Spain.csv")

# Load simulation parameters from a YAML file
sim_pars_fn = os.path.join(examples_filepath, "Europe/hpp_pars_solarX.yml")

# Initialize the Hybrid Power Plant (HPP) model
batch_size = 1 * 24
hpp = hpp_model(
    latitude=latitude,
    longitude=longitude,
    altitude=altitude,  # Geographical data for the site
    work_dir="./",  # Directory for saving outputs
    sim_pars_fn=sim_pars_fn,  # Simulation parameters
    input_ts_fn=input_ts_fn,  # Input time series (weather, prices, etc.)
    batch_size=batch_size,
)

# -----------------------------------------------------------
# Evaluate the HPP model and track execution time
# -----------------------------------------------------------
# sizing variables
# sf
sf_area = 8000  # 45920
tower_height = 50
num_tower = 25

# cpv
area_cpv_receiver_m2 = 4

# cst
heat_exchanger_capacity = 50
p_rated_st = 15
v_molten_salt_tank_m3 = 2300
area_cst_receiver_m2 = 9

# bigas_h2
area_dni_reactor_biogas_h2 = 0  # 4
area_el_reactor_biogas_h2 = 0  # 16

start = time.time()

x = [
    # sizing variables
    # sf
    sf_area,
    tower_height,
    num_tower,
    # cpv
    area_cpv_receiver_m2,
    # cst
    heat_exchanger_capacity,
    p_rated_st,
    v_molten_salt_tank_m3,
    area_cst_receiver_m2,
    # bigas_h2
    area_dni_reactor_biogas_h2,
    area_el_reactor_biogas_h2,
]

print(x)

outs = hpp.evaluate(*x)  # Run the model evaluation

hpp.print_design(x, outs)

end = time.time()
print(f"Execution time [min]:", round((end - start) / 60, 2))

# hpp.evaluation_in_csv(r"C:\Users\amia\OneDrive - Danmarks Tekniske Universitet\1. DTU\00_DevelopmentEngineer\3_SolarX\2025\EMD files\test_restuls.csv", longitude, latitude, altitude, x, outs)

# print(f'Execution time [min]:', round((end - start) / 60, 2))

# fig = hpp.plot_solarX_results(n_hours=7*24, index_hour_start=4344) # first of July: 4344
# plt.show()
