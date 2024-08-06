from abc import ABC, abstractmethod

class hpp_model_abc(ABC):
    @abstractmethod
    def __init__(self, excel_data):
        """
        describe
        """

    @abstractmethod
    def add_subsystems(self, component):
        pass

    @abstractmethod
    def connect_subsytems():
        pass

    @abstractmethod
    def set_prob_values():
        pass

    @abstractmethod
    def list_vars_and_outvars():
        pass

    @abstractmethod
    def evalute_set_and_get_prob_values():
        pass

    @abstractmethod
    def evalute_outputs():
        pass

    @abstractmethod
    def evaluation_in_csv_design_df():
        pass
