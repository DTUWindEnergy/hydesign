# Built by ems_builder
def ems_cplex(inputs):
    # Variable allocation
    # allocated variables for base: 
    allocated_vars_base = dict(
        P_HPP_ts = np.zeros(length_ts),
        P_curtailment_ts = np.zeros(length_ts),
        penalty_ts = np.zeros(length_ts))

    # allocated variables for wind: 

    # allocated variables for solar: 

    # allocated variables for bess: 
    allocated_vars_bess = dict(
        P_charge_discharge_ts = np.zeros(length_ts),
        E_SOC_ts = np.zeros(length_ts + 1)
    )

    # allocated variables for all technologies: 
    allocated_vars_all = {
                    **allocated_vars_base, 
                    **allocated_vars_bess, 
    }

    # split in batches, ussually a week
    wind_ts = inputs['wind_ts']
    batch_size_default = 2*24 # should depend on technology_types
    batch_size = inputs.get('batch_size', batch_size_default)
    batches_all = split_in_batch(list(range(len(wind_ts))), batch_size)
    # Make sure the last batch is not smaller than the others
    # instead append it to the previous last one
    batches = batches_all[:-1]
    batches[-1] = batches_all[-2]+batches_all[-1]
    
    for ib, batch in enumerate(batches):
    
        # inputs_to_ems_cplex_parts
        # inputs_to_ems_cplex_parts for base: 
        inputs_to_ems_cplex_parts_base = dict(
            price_ts=inputs['price_ts'].iloc[batch],
            G_MW=inputs['G_MW'],
            peak_hr_quantile=inputs['peak_hr_quantile'],
            cost_of_battery_P_fluct_in_peak_price_ratio=inputs['cost_of_battery_P_fluct_in_peak_price_ratio'],  # [0, 0.8]. For higher values might cause errors
            n_full_power_hours_expected_per_day_at_peak_price=inputs['n_full_power_hours_expected_per_day_at_peak_price'],
            technology_types=inputs['technology_types'],
        )

        # inputs_to_ems_cplex_parts for wind: 
        inputs_to_ems_cplex_parts_wind =dict(wind_ts=inputs['wind_ts'].iloc[batch])

        # inputs_to_ems_cplex_parts for solar: 
        inputs_to_ems_cplex_parts_solar =dict(solar_ts=inputs['solar_ts'].iloc[batch])

        # inputs_to_ems_cplex_parts for bess: 
        inputs_to_ems_cplex_parts_bess = dict(
            b_P=inputs['b_P'],
            battery_depth_of_discharge=inputs['battery_depth_of_discharge'],
            battery_charge_efficiency=inputs['battery_charge_efficiency'],
            cost_of_battery_P_fluct_in_peak_price_ratio=inputs['cost_of_battery_P_fluct_in_peak_price_ratio'],  # [0, 0.8]. For higher values might cause errors
            b_E=inputs['b_E'].iloc[batch]
        )

        # inputs_to_ems_cplex_parts for all technologies: 
        inputs_to_ems_cplex_parts_all = {
                    **inputs_to_ems_cplex_parts_base, 
                    **inputs_to_ems_cplex_parts_wind, 
                    **inputs_to_ems_cplex_parts_solar, 
                    **inputs_to_ems_cplex_parts_bess, 
        }

        # run ems cplex parts
        outputs_ems_cplex_parts = ems_cplex_parts(inputs_to_ems_cplex_parts_all)

        # outputs_batch
        # outputs_batch for base: 
        allocated_vars['P_HPP_ts'][batch] = outputs_ems_cplex_parts['P_HPP_ts']
        allocated_vars['P_curtailment_ts'][batch] = outputs_ems_cplex_parts['P_curtailment_ts']
        allocated_vars['penalty_ts'][batch] = outputs_ems_cplex_parts['penalty_ts']

        # outputs_batch for wind: 
        

        # outputs_batch for solar: 
        

        # outputs_batch for bess: 
        allocated_vars['P_charge_discharge_ts'][batch] = outputs_ems_cplex_parts['P_charge_discharge_ts']
        allocated_vars['E_SOC_ts'][batch] = outputs_ems_cplex_parts['E_SOC_ts'][:-1]

    # outputs of ems cplex
    # ems_cplex_outputs for base: 
    ems_cplex_outputs_base = {
            'hpp_t': allocated_vars['P_HPP_ts'],
            'hpp_curt_t': allocated_vars['P_curtailment_ts'],
            'penalty_t': allocated_vars['penalty_ts'],
        }

    # ems_cplex_outputs for wind: 

    # ems_cplex_outputs for solar: 

    # ems_cplex_outputs for bess: 
    ems_cplex_outputs_bess = {
            'b_t': allocated_vars['P_charge_discharge_ts'],
            'b_E_SOC_t': allocated_vars['E_SOC_ts']
        }

    # ems_cplex_outputs for all technologies: 
    ems_cplex_outputs_all = {
                    **ems_cplex_outputs_base, 
                    **ems_cplex_outputs_bess, 
    }

    return ems_cplex_outputs_all