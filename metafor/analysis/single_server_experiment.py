from analysis.experiment import Experiment, Parameter, ParameterList
from dsl.dsl import Source, Server, Work, Program

from numpy import linspace
import pandas

from model.single_server.ctmc import SingleServerCTMC


class TestFidelityWithDiscreteModel(Experiment):
    def __init__(self):
        pass

    def build(self, param) -> Program:
        apis = {'rd': Work(10, [])}
        s = Server("server", apis, 100, 20, 1)
        rdsrc = Source("reader", "rd", 9.5, 9, 3)
        p = Program("single_server")
        p.add_server(s)
        p.add_source(rdsrc)
        p.connect("reader", "server")
        return self.update(p, param)

    def analyze(self, param_setting, p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_init_state()
        results = ctmc.finite_time_analysis(pi, {'main queue size': ctmc.main_queue_size_analysis})
        return [param_setting, results]

    def show(self, results):
        print(results)
        pd = pandas.DataFrame(results, columns=["parameter", "average", "std"])
        print(pd)
        # ADD PLOT CODE HERE


if __name__ == "__main__":
    t = TestFidelityWithDiscreteModel()
    p1 = Parameter(("server", "server", "qsize"), range(100, 101))
    #p2 = Parameter(("source", "reader", "arrival_rate"), linspace(8.0, 10.0, num=2))
    t.sweep(ParameterList([p1]))
