from hydesign.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyDesignVariable,
    AssemblyModuleSpec,
)

BATTERY_ASSEMBLY_MODULE = AssemblyModuleSpec(
    name="battery",
    description="Battery technology contribution to the WSB assembly.",
    source_of_truth=(
        "hydesign/assembly/hpp_assembly.py:527-569",
        "hydesign/assembly/hpp_assembly.py:805-810",
    ),
    depends_on=("base",),
    import_lines=(
        "from .battery_degradation import battery_degradation_comp as battery_degradation",
        "from .battery_degradation import battery_loss_in_capacity_due_to_temp_comp as battery_loss_in_capacity_due_to_temp",
        "from .reliability import battery_with_reliability_comp as battery_with_reliability",
    ),
    component_specs=(
        AssemblyComponentSpec(
            subsystem_name="battery_degradation",
            constructor_code=(
                "battery_degradation(\n"
                "    weather_fn=input_ts_fn,\n"
                "    num_batteries=max_num_batteries_allowed,\n"
                "    life_y=life_y,\n"
                "    weeks_per_season_per_year=weeks_per_season_per_year,\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="battery_loss_in_capacity_due_to_temp",
            constructor_code=(
                "battery_loss_in_capacity_due_to_temp(\n"
                "    weather_fn=input_ts_fn,\n"
                "    life_y=life_y,\n"
                "    weeks_per_season_per_year=weeks_per_season_per_year,\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="battery_with_reliability",
            constructor_code=(
                "battery_with_reliability(\n"
                "    life_y=life_y,\n"
                "    reliability_ts_battery=reliability_ts_battery,\n"
                "    reliability_ts_trans=reliability_ts_trans,\n"
                ")"
            ),
        ),
    ),
    design_variables=(
        AssemblyDesignVariable("b_P", "b_P [MW]"),
        AssemblyDesignVariable("b_E_h", "b_E_h [h]"),
        AssemblyDesignVariable(
            "cost_of_battery_P_fluct_in_peak_price_ratio",
            "cost_of_battery_P_fluct_in_peak_price_ratio",
        ),
    ),
    evaluate_setup_lines=("b_E = b_E_h * b_P",),
    init_set_val_lines=('prob.set_val("min_LoH", sim_pars["min_LoH"])',),
    evaluate_set_val_lines=(
        'prob.set_val("b_P", b_P)',
        'prob.set_val("b_E", b_E)',
        "prob.set_val(",
        '    "cost_of_battery_P_fluct_in_peak_price_ratio",',
        "    cost_of_battery_P_fluct_in_peak_price_ratio,",
        ")",
    ),
    output_labels=(
        "Battery Energy [MWh]",
        "Battery Power [MW]",
        "Number of batteries used in lifetime",
    ),
    output_terms=(
        "b_E",
        "b_P",
        "prob.get_val('battery_degradation.n_batteries') * (b_P > 0)",
    ),
)
