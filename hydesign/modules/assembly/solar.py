from hydesign.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyDesignVariable,
    AssemblyModuleSpec,
)

SOLAR_ASSEMBLY_MODULE = AssemblyModuleSpec(
    name="solar",
    description="Solar technology contribution to the WSB assembly.",
    source_of_truth=(
        "hydesign/assembly/hpp_assembly.py:485-526",
        "hydesign/assembly/hpp_assembly.py:800-804",
    ),
    depends_on=("base",),
    import_lines=(
        "from .pv import pvp_comp as pvp",
        "from .pv import pvp_with_degradation_comp as pvp_with_degradation",
        "from .reliability import pvp_with_reliability_comp as pvp_with_reliability",
    ),
    component_specs=(
        AssemblyComponentSpec(
            subsystem_name="pvp",
            constructor_code=(
                "pvp(\n"
                "    weather_fn=input_ts_fn,\n"
                "    N_time=N_time,\n"
                "    latitude=latitude,\n"
                "    longitude=longitude,\n"
                "    altitude=altitude,\n"
                "    tracking=sim_pars['tracking'],\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="pvp_with_degradation",
            constructor_code=(
                "pvp_with_degradation(\n"
                "    life_y=life_y,\n"
                "    pv_deg_yr=sim_pars['pv_deg_yr'],\n"
                "    pv_deg=sim_pars['pv_deg'],\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="pvp_with_reliability",
            constructor_code=(
                "pvp_with_reliability(\n"
                "    life_y=life_y,\n"
                "    reliability_ts_pv=reliability_ts_pv,\n"
                "    reliability_ts_trans=reliability_ts_trans,\n"
                ")"
            ),
            key_map_code="{'solar_t': 'solar_t_ext_deg'}",
        ),
    ),
    design_variables=(
        AssemblyDesignVariable("solar_MW", "solar_MW [MW]"),
        AssemblyDesignVariable("surface_tilt", "surface_tilt [deg]"),
        AssemblyDesignVariable("surface_azimuth", "surface_azimuth [deg]"),
        AssemblyDesignVariable("DC_AC_ratio", "DC_AC_ratio"),
    ),
    evaluate_set_val_lines=(
        'prob.set_val("surface_tilt", surface_tilt)',
        'prob.set_val("surface_azimuth", surface_azimuth)',
        'prob.set_val("DC_AC_ratio", DC_AC_ratio)',
        'prob.set_val("solar_MW", solar_MW)',
    ),
    output_labels=(
        "solar [MW]",
        "Apvp [km2]",
    ),
    output_terms=(
        "solar_MW",
        "prob.get_val('shared_cost.Apvp')",
    ),
)
