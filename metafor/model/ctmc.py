from abc import ABC

from utils.plot_parameters import PlotParameters


class CTMC(ABC):

    def average_lengths_analysis(self, file_name: str, plot_params: PlotParameters):
        pass

    def fault_scenario_analysis(self, file_name: str, plot_params: PlotParameters):
        pass

    def latency_analysis(self, file_name: str, plot_params: PlotParameters, job_type: int = -1):
        pass
