import math
from typing import Any, Callable
import unittest
from analysis.experiment import Experiment, Parameter, ParameterList, extract_name_val
from dsl.dsl import Source, Server, Work, Program

from numpy import linspace
import pandas
from matplotlib import pyplot as plt

from model.single_server.ctmc import SingleServerCTMC


class LatencyExperiment(Experiment):
    def __init__(self, p: Program):
        self.p = p
        self.main_color = "#A9A9A9"
        self.fade_color = "#D3D3D3"

    def plot(
        self,
        figure_name: str,
        x_axis: str,
        y_axis: str,
        x_vals,
        mean_seq,
        lower_bound_seq,
        upper_bound_seq,
    ):
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
        #plt.savefig(figure_name)
        plt.show()
        plt.close()

    def build(self, param) -> Program:
        return self.update(self.p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_stationary_distribution()
        # results = ctmc.finite_time_analysis(pi, {'main queue size': ctmc.main_queue_size_analysis})
        server = p.get_root_server()

        results = []
        for i, req in enumerate(p.get_requests(server.name)):
            latency = ctmc.latency_average(pi, i)
            variance = ctmc.latency_variance(pi, latency)
            stddev = math.sqrt(variance)
            results = results + [latency, stddev]
        return [param_setting] + results

    def show(self, results):
        # print(results)
        PARAMETER = "parameter"
        columns = [PARAMETER]
        server = self.p.get_root_server()
        reqs = self.p.get_requests(server.name)
        for req in reqs:
            columns = columns + ([req + "_avg", req + "_std"])
        pd = pandas.DataFrame(results, columns=columns)
        paramvals = list(map(lambda pmap: extract_name_val(pmap), pd[PARAMETER]))
        parameter_name = ".".join(paramvals[0][0][1:])
        pd["parameter_values"] = list(map(lambda v: v[1], paramvals))
        
        for req in reqs:
            pd[req + "_lowerbd"] = pd[req + "_avg"] - pd[req + "_std"]
            pd[req + "_upperbd"] = pd[req + "_avg"] + pd[req + "_std"]
            self.plot("Average latency for " + req, 
                      x_axis= parameter_name, 
                      y_axis="Latency of " + req,
                      x_vals = pd["parameter_values"],
                      mean_seq=pd[req + "_avg"], 
                      lower_bound_seq=pd[req + "_lowerbd"], 
                      upper_bound_seq=pd[req+"_upperbd"])
        print(pd)


class EquilibriumValuesExperiment(Experiment):
    def __init__(self, p: Program):
        self.p = p
        self.main_color = "#A9A9A9"
        self.fade_color = "#D3D3D3"

    def build(self, param) -> Program:
        return self.update(self.p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_stationary_distribution()
        server = p.get_root_server()

        results = [param_setting]
        mainqavg = ctmc.main_queue_size_average(pi)
        mainqstd = ctmc.main_queue_size_std(pi, mainqavg)
        results.append(mainqavg)
        results.append(mainqstd)
        return results

    def show(self, results):
        # print(results)
        columns = ["parameter", "qsize", "std"]
        pd = pandas.DataFrame(results, columns=columns)
        print(pd)

class FiniteHorizonExperiment(Experiment):
    def __init__(
        self,
        p: Program,
        analyses: dict[str, Callable[[Any], Any]],
        sim_time: int = 100,
        sim_step: int = 20,
    ):
        self.p = p
        self.analyses = analyses
        self.sim_time = sim_time
        self.sim_step = sim_step

        self.main_color = "#A9A9A9"
        self.fade_color = "#D3D3D3"

    def build(self, param) -> Program:
        return self.update(self.p, param)
    
    def plot(
        self,
        filename,
        figure_name: str,
        df: pandas.DataFrame
    ):
        colors = plt.cm.viridis(linspace(0, 1, len(df.columns) - 1))

        plt.rc("font", size=14)
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = True

        plt.figure()  # row 0, col 0
        x_vals = df["time"]
        for i, column in enumerate(df.columns[1:]):
            plt.plot(df['time'], df[column], marker='o', color=colors[i], label=column)

        plt.title(figure_name)
        y_axis_label = "Average values for " + ", ".join(df.columns[1:])    
        plt.xlabel("Time", fontsize=14)
        plt.ylabel(y_axis_label, fontsize=14)
        plt.grid("on")

        plt.xlim(df["time"].min(), df["time"].max())        
        plt.savefig(filename)
        plt.show()
        plt.close()

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_init_state()
        analyses = {k: f(ctmc) for (k, f) in self.analyses.items()}
        results = ctmc.finite_time_analysis(
            pi, analyses, sim_time=self.sim_time, sim_step=self.sim_step
        )
        # print("Results:", results)
        thisresult = []
        for step, values in results.items():
            a = [step]
            # print("Values = ", values)
            for k in self.analyses:
                a.append(values[k])
            thisresult.append(a)
        return (param_setting, thisresult)

    def show(self, results):
        # print(results)
        for param, values in results:
            print("Parameter: ", param)
            columns = ["time"] + list(self.analyses.keys())
            # print(columns)
            pd = pandas.DataFrame(values, columns=columns)
            print(pd)
            self.plot("trends_v_time", "Trends of " + str(param) + " vs time", pd)


class TestExperiments(unittest.TestCase):
    def program(self) -> Program:
        apis = {"rd": Work(10, []), "wr": Work(10, [])}
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

    def test_latency_v_qsize(self):
        p = self.program()
        t = LatencyExperiment(p)
        p1 = Parameter(("server", "server", "qsize"), range(80, 160, 10))
        t.sweep(ParameterList([p1]))

    def test_latency_v_arrival_rate(self):
        p = self.program()
        t = LatencyExperiment(p)
        p2 = Parameter(("source", "reader", "arrival_rate"), linspace(4.0, 6.0, num=4))
        t.sweep(ParameterList([p2]))

    def test_latency_v_timeout(self):
        p = self.program()
        t = LatencyExperiment(p)
        p3 = Parameter(("source", "reader", "timeout"), range(2, 8))
        t.sweep(ParameterList([p3]))

    def test_convergence_v_qsize(self):
        p = self.program()
        t = FiniteHorizonExperiment(
            p, {"main_q_size": lambda ctmc: ctmc.main_queue_size_average}
        )
        p1 = Parameter(("server", "server", "qsize"), range(80, 160, 10))
        t.sweep(ParameterList([p1]))

    def test_convergence_v_arrival_rate(self):
        p = self.program()
        t = FiniteHorizonExperiment(
            p, 
            { "main_q_size": lambda ctmc: ctmc.main_queue_size_average,
              "latency" : lambda ctmc: (lambda pi: ctmc.latency_average(pi, req_type=0)),
            }, 
            sim_time=200, sim_step=10
        )
        p1 = Parameter(("source", "writer", "arrival_rate"), linspace(2.0, 6.0, 8))
        t.sweep(ParameterList([p1]))

    def test_average_qsize_v_arrival_rate(self):
        p = self.program()
        t = EquilibriumValuesExperiment(p)
        p1 = Parameter(("source", "writer", "arrival_rate"), linspace(2.0, 6.0, 8))
        t.sweep(ParameterList([p1]))



class TestExperimentsLarge(unittest.TestCase):
    def storage_server(self) -> Program:
        apis = {
            "get": Work(10, []),
            "put": Work(20, []),
            "list": Work(2, []),
        }
        node = Server("node", apis, qsize=300, orbit_size=5, thread_pool=32)

        putsrc = Source("putsrc", "put", 2, 5, 1)
        getsrc = Source("getsrc", "get", 2, 3, 1)
        listsrc = Source("listsrc", "list", 2, 5, 1)

        p = Program("storage_node")
        p.add_server(node)
        p.add_source(putsrc)
        p.add_source(getsrc)
        p.add_source(listsrc)
        p.connect("putsrc", "node")
        p.connect("getsrc", "node")
        p.connect("listsrc", "node")
        return p

    def test_storage_system(self):
        storage_server_model = self.storage_server()
        qsizes = Parameter(("server", "node", "qsize"), range(500, 1300, 200))
        t = LatencyExperiment(storage_server_model)
        t.sweep(ParameterList([qsizes]))

    def test_finite_horizon(self):
        storage_server_model = self.storage_server()
        qsizes = Parameter(("server", "node", "qsize"), range(500, 1300, 200))
        t = FiniteHorizonExperiment(
            storage_server_model, {"main_q_size": lambda ctmc: ctmc.main_queue_size_average}, sim_time=30, sim_step=5
        )
        t.sweep(ParameterList([qsizes]))


if __name__ == "__main__":
    unittest.main()
