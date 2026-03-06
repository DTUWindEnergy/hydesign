from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FinanceModuleSpec:
    name: str
    kind: str
    description: str
    source_of_truth: tuple[str, ...]
    symbols: tuple[str, ...] = ()
    component_inputs: tuple[str, ...] = ()
    component_outputs: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    notes: str = ""
