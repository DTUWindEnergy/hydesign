from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EMSExecutionStage(str, Enum):
    """Ordered extension points in the original CPLEX EMS flow."""

    OPENMDAO_INPUTS = "openmdao_inputs"
    OPENMDAO_OUTPUTS = "openmdao_outputs"
    COMPUTE_PREPARE = "compute_prepare"
    BATCH_ALLOCATE = "batch_allocate"
    BATCH_SLICE = "batch_slice"
    CPLEX_VARIABLES = "cplex_variables"
    CPLEX_POWER_BALANCE = "cplex_power_balance"
    CPLEX_OBJECTIVE = "cplex_objective"
    CPLEX_CONSTRAINTS = "cplex_constraints"
    CPLEX_EXTRACT = "cplex_extract"
    FINALIZE_BATCH_OUTPUTS = "finalize_batch_outputs"
    LIFETIME_OUTPUTS = "lifetime_outputs"


@dataclass(frozen=True)
class EMSModuleSpec:
    """Metadata contract for an EMS module contribution."""

    name: str
    kind: str
    description: str
    source_of_truth: tuple[str, ...]
    depends_on: tuple[str, ...] = ()
    component_inputs: tuple[str, ...] = ()
    component_outputs: tuple[str, ...] = ()
    config_inputs: tuple[str, ...] = ()
    batch_outputs: tuple[str, ...] = ()
    stages: tuple[EMSExecutionStage, ...] = ()
    notes: str = ""
