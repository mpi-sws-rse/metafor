from analysis.experiment import Experiment, Parameter, ParameterList
from dsl.dsl import Source, Server, Work, Program

from numpy import linspace
import pandas

from model.single_server.ctmc import SingleServerCTMC


class TestFidelityWithDiscreteModel(Experiment):
    def __init__(self):
        apis = {'rd': Work(10, [])}
        s = Server("server", apis, 100, 20, 1)
        rdsrc = Source("reader", "rd", 9.5, 9, 3)
        self.p = Program("single_server")
        self.p.add_server(s)
        self.p.add_source(rdsrc)
        self.p.connect("reader", "server")
        

    def build(self, param) -> Program:
        return self.update(self.p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_stationary_distribution()
        avg = ctmc.main_queue_size_average(pi)[0]
        print("avg = ", avg)
        std = ctmc.main_queue_size_std(pi, avg)
        print("std = ", std)
        print([param_setting, avg, std])
        return [param_setting, avg, std]

    def show(self, results):
        print(results)
        pd = pandas.DataFrame(results, columns=["parameter", "average", "std"])
        print(pd)
        # ADD PLOT CODE HERE


if __name__ == "__main__":
    t = TestFidelityWithDiscreteModel()
    p1 = Parameter(("server", "server", "qsize"), range(20, 40, 10))
    p2 = Parameter(("source", "reader", "arrival_rate"), linspace(8.0, 10.0, num=2))
    t.sweep(ParameterList([p1, p2]))
