from abc import ABC, abstractmethod
import inspect

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
    def add_ems_args():
        pass

    @abstractmethod
    def postprocessing_outputs():
        pass

    def get_code_addinputs(self):
        code = inspect.getsource(self.add_inputs)
        #code = code.replace("component.", "self.")
        # Specify the line you want to remove (exact match or part of the line)
        line_to_remove = "def"

        # Filter out the line from the input string
        lines = [line for line in code.splitlines() if line_to_remove not in line.strip()]

        # Join the remaining lines into a single string with appropriate line breaks
        output_string = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
        return output_string

    def get_code_addoutputs(self):
        code = inspect.getsource(self.add_outputs)
        #code = code.replace("component.", "self.")
        # Specify the line you want to remove (exact match or part of the line)
        line_to_remove = "def"

        # Filter out the line from the input string
        lines = [line for line in code.splitlines() if line_to_remove not in line.strip()]

        # Join the remaining lines into a single string with appropriate line breaks
        output_string = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
        return output_string

    def get_code_add_ems_args(self):
        code = inspect.getsource(self.add_ems_args)
        code = code.replace("component.", "self.")
        # Specify the line you want to remove (exact match or part of the line)
        lines_to_remove = ["def ", "return "]

        # Filter out lines that start with any of the specified keywords
        lines = [line for line in code.splitlines() if not any(line.strip().startswith(keyword) for keyword in lines_to_remove)]

        # Join the remaining lines into a single string with appropriate line breaks
        output_string = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
        return output_string

    def get_code_postprocessing_outputs(self):
        code = inspect.getsource(self.postprocessing_outputs)
        code = code.replace("component.", "self.")
        # Specify the line you want to remove (exact match or part of the line)
        lines_to_remove = ["def ", "return "]

        # Filter out lines that start with any of the specified keywords
        lines = [line for line in code.splitlines() if not any(line.strip().startswith(keyword) for keyword in lines_to_remove)]

        # Join the remaining lines into a single string with appropriate line breaks
        output_string = "\n".join(line[4:] if line.startswith("    ") else line for line in lines)
        return output_string

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
    def __init__(self):
        pass

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
    def _dict_hydesign_equations(self):
        pass

    @abstractmethod
    def outputs_cplex_parts(self):
        pass


    def get_updated_model(self):
        self._variables()
        self._constraints()
        self._objective_function()
        self._dict_hydesign_equations()
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
