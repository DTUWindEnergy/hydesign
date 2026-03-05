from hydesign.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyDesignVariable,
    AssemblyModuleSpec,
)

WIND_ASSEMBLY_MODULE = AssemblyModuleSpec(
    name="wind",
    description="Wind technology contribution to the WSB assembly.",
    source_of_truth=(
        "hydesign/assembly/hpp_assembly.py:463-514",
        "hydesign/assembly/hpp_assembly.py:786-798",
        "hydesign/assembly/hpp_assembly.py:805-814",
    ),
    depends_on=("base",),
    import_lines=(
        "from .wind import genericWake_surrogate_comp as genericWake_surrogate",
        "from .wind import genericWT_surrogate_comp as genericWT_surrogate",
        "from .wind import get_rotor_d",
        "from .wind import wpp_comp as wpp",
        "from .wind import wpp_with_degradation_comp as wpp_with_degradation",
        "from .reliability import wpp_with_reliability_comp as wpp_with_reliability",
    ),
    component_specs=(
        AssemblyComponentSpec(
            subsystem_name="genericWT",
            constructor_code="genericWT_surrogate(genWT_fn=genWT_fn, N_ws=N_ws)",
        ),
        AssemblyComponentSpec(
            subsystem_name="genericWake",
            constructor_code="genericWake_surrogate(genWake_fn=genWake_fn, N_ws=N_ws)",
        ),
        AssemblyComponentSpec(
            subsystem_name="wpp",
            constructor_code=(
                "wpp(\n"
                "    N_time=N_time,\n"
                "    N_ws=N_ws,\n"
                "    wpp_efficiency=wpp_efficiency,\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="wpp_with_degradation",
            constructor_code=(
                "wpp_with_degradation(\n"
                "    N_time=N_time,\n"
                "    N_ws=N_ws,\n"
                "    wpp_efficiency=wpp_efficiency,\n"
                "    life_y=life_y,\n"
                "    wind_deg_yr=wind_deg_yr,\n"
                "    wind_deg=wind_deg,\n"
                "    share_WT_deg_types=share_WT_deg_types,\n"
                "    weeks_per_season_per_year=weeks_per_season_per_year,\n"
                ")"
            ),
        ),
        AssemblyComponentSpec(
            subsystem_name="wpp_with_reliability",
            constructor_code=(
                "wpp_with_reliability(\n"
                "    life_y=life_y,\n"
                "    reliability_ts_wind=reliability_ts_wind,\n"
                "    reliability_ts_trans=reliability_ts_trans,\n"
                ")"
            ),
            key_map_code="{'wind_t': 'wind_t_ext_deg'}",
        ),
    ),
    design_variables=(
        AssemblyDesignVariable("clearance", "clearance [m]"),
        AssemblyDesignVariable("sp", "sp [W/m2]"),
        AssemblyDesignVariable("p_rated", "p_rated [MW]"),
        AssemblyDesignVariable("Nwt", "Nwt"),
        AssemblyDesignVariable("wind_MW_per_km2", "wind_MW_per_km2 [MW/km2]"),
    ),
    evaluate_setup_lines=(
        "d = get_rotor_d(p_rated * 1e6 / sp)",
        "hh = (d / 2) + clearance",
        "wind_MW = Nwt * p_rated",
        "Awpp = wind_MW / wind_MW_per_km2",
    ),
    evaluate_set_val_lines=(
        'prob.set_val("hh", hh)',
        'prob.set_val("d", d)',
        'prob.set_val("p_rated", p_rated)',
        'prob.set_val("Nwt", Nwt)',
        'prob.set_val("Awpp", Awpp)',
    ),
    evaluate_post_run_lines=(
        "if Nwt == 0:",
        "    cf_wind = np.nan",
        "else:",
        "    cf_wind = (",
        "        prob.get_val('wpp_with_degradation.wind_t_ext_deg').mean()",
        "        / p_rated",
        "        / Nwt",
        "    )",
    ),
    output_labels=(
        "wind [MW]",
        "Awpp [km2]",
        "Rotor diam [m]",
        "Hub height [m]",
        "Capacity factor wind [-]",
    ),
    output_terms=(
        "wind_MW",
        "Awpp",
        "d",
        "hh",
        "cf_wind",
    ),
)
