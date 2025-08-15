import openmdao.api as om

from hydesign.nrelcsm.nrel_csm_mass_2015 import nrel_csm_2015


def wt_cost(
    machine_rating,
    rotor_diameter,
    turbine_class,
    hub_height,
    blade_number,
    blade_has_carbon,
    bearing_number,
    crane,
    max_tip_speed=80,
    max_efficiency=0.9,
):

    # 0 ---------- (marker for docs)

    # 1 ---------- (marker for docs)
    # OpenMDAO Problem instance
    prob = om.Problem(reports=False)
    prob.model = nrel_csm_2015()
    prob.setup()
    # 1 ---------- (marker for docs)

    # 2 ---------- (marker for docs)
    # Initialize variables for NREL CSM
    prob["machine_rating"] = machine_rating
    prob["rotor_diameter"] = rotor_diameter
    prob["turbine_class"] = turbine_class
    prob["hub_height"] = hub_height
    prob["blade_number"] = blade_number
    prob["blade_has_carbon"] = blade_has_carbon
    prob["max_tip_speed"] = max_tip_speed
    prob["max_efficiency"] = max_efficiency
    prob["main_bearing_number"] = bearing_number
    prob["crane"] = crane
    # 2 ---------- (marker for docs)

    # 3 ---------- (marker for docs)
    # Evaluate the model
    prob.run_model()
    # 3 ---------- (marker for docs)

    # 4 ---------- (marker for docs)
    # Print all intermediate inputs and outputs to the screen
    # 4 ---------- (marker for docs)
    return prob["turbine_costs.turbine_c.turbine_cost"]
