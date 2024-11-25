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


class MixingTimeExperiment(Experiment):
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
        mixing_times,
    ):
        plt.rc("font", size=14)
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = True

        plt.figure()  # row 0, col 0
        plt.plot(x_vals, mixing_times, color=self.main_color)
        
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.grid("on")
        plt.xlim(min(x_vals), max(x_vals))
        plt.show()
        plt.close()

    def build(self, param) -> Program:
        return self.update(self.p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        mixing_time = ctmc.get_mixing_time()
        return [param_setting, mixing_time]

    def show(self, results):
        # print(results)
        PARAMETER = "parameter"
        MIXING_TIME = "Mixing time"
        columns = [PARAMETER, MIXING_TIME]
        
        pd = pandas.DataFrame(results, columns=columns)
    
        print(pd)


class HittingTimeExperiment(Experiment):
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
            hitting_times,
    ):
        plt.rc("font", size=14)
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = True

        plt.figure()  # row 0, col 0
        plt.plot(x_vals, hitting_times, color=self.main_color)

        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.grid("on")
        plt.xlim(min(x_vals), max(x_vals))
        plt.show()
        plt.close()

    def build(self, param) -> Program:
        return self.update(self.p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()

        # average = ctmc.set_construction([[0, int(1 * ctmc.thread_pool)]], [[0, ctmc.retry_queue_size]])
        # average = ctmc.set_construction([[0, 100]], [[0, ctmc.retry_queue_size]])

        if ctmc.thread_pool > 1:
            average = ctmc.set_construction([[0, int(1 * ctmc.thread_pool)]], [[0, ctmc.retry_queue_size]])
        else:
            average = ctmc.set_construction([[0, int(.1 * ctmc.main_queue_size)]], [[0, ctmc.retry_queue_size]])
        full = ctmc.set_construction([[int(.9 * ctmc.main_queue_size), ctmc.main_queue_size]],
                                   [[0, ctmc.retry_queue_size]])
        hitting_time = ctmc.get_hitting_time_average(full, average)
        return [param_setting, hitting_time]

    def show(self, results):
        # print(results)
        PARAMETER = "parameter"
        HITTING_TIME = "Hitting time"
        columns = [PARAMETER, HITTING_TIME]

        pd = pandas.DataFrame(results, columns=columns)

        print(pd)


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


class Test52(unittest.TestCase):
    def setUp(self):
        self.qsizes = Parameter(("server", "52", "qsize"), range(350, 550, 100))
        self.processing_rates = Parameter(("server", "52", "api", "insert", "processing_rate"), linspace(1/0.010, 1/0.020, 4))
    
    def program(self):
        api = { "insert": Work(1/.016, [],) }
        server = Server("52", api, qsize=300, orbit_size=10, thread_pool=100)
        src = Source('client', 'insert', 6200, timeout=3, retries=4)
        p = Program("Service52")
        p.add_server(server)
        p.add_source(src)
        p.connect('client', '52')
        return p

    def program_reduced_threads(self): 
        api = { "insert": Work(1/.016, [],) }
        server = Server("52", api, qsize=300, orbit_size=10, thread_pool=20)
        src = Source('client', 'insert', 6200, timeout=3, retries=4)
        p = Program("Service52")
        p.add_server(server)
        p.add_source(src)
        p.connect('client', '52')
        return p  
    
    def basic_stats(self, p):
        print("Building CTMC")
        ctmc: SingleServerCTMC = p.build()
        print("Computing stationary distribution")
        pi = ctmc.get_stationary_distribution()
        print("Average queue size = ", ctmc.main_queue_size_average(pi))
        print("Mixing time = ", ctmc.get_mixing_time())
        # S1 = ctmc.set_construction([[0, int(.3*ctmc.main_queue_size)]], [[0, ctmc.retry_queue_size]])
        # S2 = ctmc.set_construction([[int(.9*ctmc.main_queue_size), ctmc.main_queue_size]], [[0, ctmc.retry_queue_size]])
        S1 = ctmc.set_construction([[0, 100]], [[0, ctmc.retry_queue_size]])
        S2 = ctmc.set_construction([[ctmc.main_queue_size-100, ctmc.main_queue_size]], [[0, ctmc.retry_queue_size]])
        ht_su = ctmc.get_hitting_time_average(S1, S2)
        ht_us = ctmc.get_hitting_time_average(S2, S1)
        print("Expected hitting time to go from high to low mode is", ht_us)
        print("Expected hitting time to go from low to high mode is", ht_su)

        print("Time to drain queue = ", ctmc.time_to_drain())


    def test_program_basic(self):
        p = self.program()
        self.basic_stats(p)

    def test_program_reduced_basic(self):
        p = self.program_reduced_threads()
        self.basic_stats(p)

    def test_hitting_times(self):
        print("Computing exact recovery times")
        p = self.program()
        ht = HittingTimeExperiment(p)
        ht.sweep(ParameterList([self.qsizes]))

        p = self.program()
        ht = HittingTimeExperiment(p)
        ht.sweep(ParameterList([self.processing_rates]))

    def test_hitting_times_reduced_threads(self):
        print("Computing exact recovery times")
        p = self.program_reduced_threads()
        ht = HittingTimeExperiment(p)
        ht.sweep(ParameterList([self.qsizes]))

        p = self.program_reduced_threads()
        ht = HittingTimeExperiment(p)
        ht.sweep(ParameterList([self.processing_rates]))


    def test_mixing_times(self):
        print("Computing mixing times")
        p = self.program()
        ht = MixingTimeExperiment(p)
        ht.sweep(ParameterList([self.qsizes]))
        p = self.program_reduced_threads()
        ht = MixingTimeExperiment(p)
        ht.sweep(ParameterList([self.processing_rates]))

    def test_mixing_times_reduced_threads(self):
        print("Computing mixing times")
        p = self.program_reduced_threads()
        ht = MixingTimeExperiment(p)
        ht.sweep(ParameterList([self.qsizes]))

        p = self.program_reduced_threads()
        ht = MixingTimeExperiment(p)
        ht.sweep(ParameterList([self.processing_rates]))

    def test_large_queues(self):
        p = self.program()
        large_queues = Parameter(("server", "52", "qsize"), range(18000, 20000, 5000))
        ht = HittingTimeExperiment(p)
        ht.sweep(ParameterList([large_queues]))

if __name__ == "__main__":
    unittest.main()