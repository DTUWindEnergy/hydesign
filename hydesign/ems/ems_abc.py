from abc import ABC, abstractmethod

class ems_abc(ABC):
    @abstractmethod
    def add_inputs(self, component):
        pass

    @abstractmethod
    def add_outputs(self, component):
        pass

    @abstractmethod
    def added_ems_WSB_args():
        pass

    @abstractmethod
    def outputs_extension():
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
    def get_updated_model(self):
        pass

    @abstractmethod
    def get_added_variables(self):
        pass

    def get_updated_model(self):
        return self.mdl