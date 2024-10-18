from analysis.multi_server_ctmc_analysis import fault_scenario_analysis as multi_fault_scenario_analysis
from analysis.single_server_ctmc_analysis import average_lengths_analysis as single_average_lengths_analysis, \
    fault_scenario_analysis as single_fault_scenario_analysis, latency_analysis as single_latency_analysis
from model.ctmc import CTMC
from model.multi_server.multi_server_ctmc import MultiServerCTMC
from model.single_server.single_server_ctmc import SingleServerCTMC
from utils.plot_parameters import PlotParameters


class Analyzer:

    def __init__(self, ctmc: CTMC, file_name: str):
        self.ctmc = ctmc
        self.file_name = file_name

    def average_lengths_analysis(self, plot_params: PlotParameters):
        if isinstance(self.ctmc, SingleServerCTMC):
            single_average_lengths_analysis(self.ctmc, self.file_name, plot_params)
        else:
            raise NotImplementedError

    def fault_scenario_analysis(self, plot_params: PlotParameters):
        if isinstance(self.ctmc, SingleServerCTMC):
            single_fault_scenario_analysis(self.ctmc, self.file_name, plot_params)
        elif isinstance(self.ctmc, MultiServerCTMC):
            multi_fault_scenario_analysis(self.ctmc, self.file_name, plot_params)

    def latency_analysis(self, plot_params: PlotParameters, job_type: int = -1):
        if isinstance(self.ctmc, SingleServerCTMC):
            single_latency_analysis(self.ctmc, self.file_name, plot_params, job_type)
        else:
            raise NotImplementedError
