# basic libraries
from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd
import pyomo.environ as pyo
from docplex.mp.constants import ComparisonType

# import xarray as xr
from docplex.mp.model import Model


class EMSCoreABC(ABC):
    """
    Abstract interface for EMS core classes.
    Concrete implementations must provide:
      - inputs
      - inputs_batch
      - outputs_batch
      - final_process
    """

    def __init__(self, **kwargs):
        pass

    def solve(self, **kwargs):
        ems_solver = self.ems_solver
        for ib, batch in enumerate(self.batches_all):
            self.batch()
            if ems_solver == "cplex":
                self.cplex()
            elif ems_solver == "pyomo":
                self.pyomo()
            elif ems_solver == "ortools":
                self.ortools()
            out_ems

    def inputs(self, **kwargs) -> Dict[str, Any]:
        """Prepare and return the initial inputs / allocated arrays and meta values."""
        return {}

    def cplex(self, mdl, **kwargs) -> Dict[str, Any]:
        """Merge batch outputs back into global arrays stored in kwargs and return kwargs."""
        raise NotImplementedError

    def cplex_variables(self, mdl):
        pass

    def cplex_constraints(self, mdl):
        pass

    def cplex_opjective(self, mdl):
        pass

    def cplex_outputs(self, sol):
        pass

    def ortools_variables(self, solver):
        pass

    def ortools_constraints(self, solver):
        pass

    def ortools_objective(self, solver):
        pass

    def ortools_outputs(self):
        pass

    def pyomo_variables(self, mdl):
        pass

    def pyomo_constraints(self, mdl):
        pass

    def pyomo_opjective(self, mdl):
        pass

    def pyomo_outputs(self, mdl):
        pass

    def final_process(self, **kwargs) -> Dict[str, Any]:
        """
        Final processing step.
        """
        return kwargs

    def update_final_outputs(self):
        return {}


def split_in_batch(array, N):
    batch = []
    counter = 0
    while counter * N < len(array):
        if (counter + 1) * N > len(array):
            end = len(array)
        else:
            end = (counter + 1) * N
        batch += [array[counter * N : end]]
        counter += 1
    return batch


# helper to safely read an indexed Pyomo Var into a pandas Series
def series_from_indexed_var(var, idx_map):
    """
    var: Pyomo Var indexed by integers 0..T-1
    idx_map: list-like of labels for the pandas index (e.g. time_batch)
    returns: pd.Series indexed by idx_map with numeric values
    """
    values = []
    for i in range(len(idx_map)):
        try:
            values.append(pyo.value(var[i]))
        except KeyError:
            # missing index in var
            values.append(0.0)
        except Exception:
            # value not available (uninitialized) -> preserve failure behavior or set nan/0
            values.append(0.0)
    return pd.Series(values, index=idx_map)
