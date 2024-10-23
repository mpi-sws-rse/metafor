import math
from analysis.experiment import Experiment, Parameter, ParameterList
from dsl.dsl import Source, Server, Work, Program

from numpy import linspace
import pandas
from matplotlib import pyplot as plt

from model.single_server.ctmc import SingleServerCTMC


class TestLatency(Experiment):
    def __init__(self, p: Program):
        self.p = p
        self.main_color = "#A9A9A9"
        self.fade_color = "#D3D3D3"

    def plot(self, figure_name: str, x_axis: str, y_axis: str, x_vals, mean_seq, lower_bound_seq, upper_bound_seq):
        plt.rc("font", size=14)
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = True

        plt.figure()  # row 0, col 0
        plt.plot(x_vals, mean_seq, color=self.main_color)
        plt.fill_between(
            x_vals, lower_bound_seq, upper_bound_seq, color=self.fade_color, alpha=0.4
        )   
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.grid("on")
        plt.xlim(min(x_vals), max(x_vals))
        plt.savefig(figure_name)
        plt.close()

    def build(self, param) -> Program:
        return self.update(self.p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_stationary_distribution()
        # results = ctmc.finite_time_analysis(pi, {'main queue size': ctmc.main_queue_size_analysis})
        server = p.get_root_server()

        results = []
        for (i, req) in enumerate(p.get_requests(server.name)):
            latency = ctmc.latency_average(pi, i)
            variance = ctmc.latency_variance(pi, i)
            stddev = math.sqrt(variance)
            results = results + [latency, stddev]
        return [param_setting] + results

    def show(self, results):
        print(results)
        columns = ['parameter']
        server = self.p.get_root_server()
        for req in self.p.get_requests(server.name):
            columns = columns + ([req+"_avg", req+"_std"])
        pd = pandas.DataFrame(results, columns=columns)
        print(pd)
        # ADD PLOT CODE HERE
        #self.plot_results_latency(
        #    input_seq,
        #    mean_latency_seq,
        #    lower_bound_latency_seq,
        #    upper_bound_latency_seq,
        #    x_axis_label,
        #    "Latency",
        #    "latency_" + variable1 + "_varied_" + job_info + file_name,
        #    main_color,
        #   fade_color,
        #)

def program() -> Program:
    apis = {'rd': Work(10, []), 'wr': Work(10, [])}
    s = Server("server", apis, 100, 20, 1)
    rdsrc = Source("reader", "rd", 4.75, 9, 3)
    wrsrc = Source("writer", "wr", 4.75, 9, 3)

    p = Program("single_server")
    p.add_server(s)
    p.add_source(rdsrc)
    p.add_source(wrsrc)
    p.connect("reader", "server")
    p.connect("writer", "server")
    return p

if __name__ == "__main__":
    p = program()
    t = TestLatency(p)
    p1 = Parameter(("server", "server", "qsize"), range(80, 160, 10))
    t.sweep(ParameterList([p1]))

    p = program()
    p2 = Parameter(("source", "reader", "arrival_rate"), linspace(4.0, 6.0, num=4))
    t.sweep(ParameterList([p2]))

    p = program()
    p3 = Parameter(("source", "reader", "timeout"), range(2, 8))
    t.sweep(ParameterList([p3]))