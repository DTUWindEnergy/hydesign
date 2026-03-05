from hydesign.modules.assembly.contracts import (
    AssemblyComponentSpec,
    AssemblyModuleSpec,
)

OPERATIONS_ASSEMBLY_MODULE = AssemblyModuleSpec(
    name="operations",
    description=(
        "Long-term operation and degradation propagation contribution to the WSB assembly."
    ),
    source_of_truth=("hydesign/assembly/hpp_assembly.py:571-580",),
    depends_on=("base", "wind", "solar", "battery"),
    import_lines=(
        "from .ems_long_term import ems_long_term_operation_comp as ems_long_term_operation",
    ),
    component_specs=(
        AssemblyComponentSpec(
            subsystem_name="ems_long_term_operation",
            constructor_code="ems_long_term_operation(N_time=N_time, life_y=life_y)",
            key_map_code=(
                "{'SoH': 'SoH_all', 'wind_t_ext_deg': 'wind_t_rel', "
                "'solar_t_ext_deg': 'solar_t_rel', 'b_t': 'b_t_rel'}"
            ),
        ),
    ),
)
