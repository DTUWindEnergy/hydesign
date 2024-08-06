from abc import ABC, abstractmethod

class ems_abc(ABC):
    @abstractmethod
    # add_openmdao_inputs
    def add_inputs(self, component):
        """
        describe
        """

    @abstractmethod
    def add_outputs(self, component):
        pass

    @abstractmethod
    # add_ems_args
    def added_ems_WSB_args():
        pass

    @abstractmethod
    # postprocessing_outputs
    def outputs_extension():
        pass

class ems_cplex_abc(ABC):
    @abstractmethod
    def allocate_vars(self):
        pass

    @abstractmethod
    def inputs_to_ems_cplex_parts(self):
        pass

    @abstractmethod
    def outputs_batch(self):
        pass

    @abstractmethod
    def ems_cplex_outputs(self):
        pass


class ems_cplex_parts_abc(ABC):
    @abstractmethod
    def _variables(self):
        pass

    @abstractmethod
    def _constraints(self):
        pass

    @abstractmethod
    def _objective_function(self):
        pass

    @abstractmethod
    def outputs_cplex_parts(self):
        pass

    def get_updated_model(self):
        return self.mdl


class ems_degredation_abc(ABC):
    """Abstract base class for EMS Degredation."""

    @abstractmethod
    def compute_wind_solar_deg(self):
        pass

    @abstractmethod
    def initialize_soc(self):
        pass

    @abstractmethod
    def adjust_battery_power(self, wind_solar_t_deg):
        pass

    @abstractmethod
    def update_soc(self, b_t_less_sol, b_E_SOC_t_sat):
        pass

    @abstractmethod
    def recompute_battery_power(self, b_E_SOC_t_sat):
        pass

    @abstractmethod
    def compute_penalty(self, Hpp_deg):
        pass
