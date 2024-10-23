from typing import Any, Dict

from analysis.experiment import Experiment
from dsl.dsl import Source, Server, Work, Program

from model.single_server.ctmc import SingleServerCTMC
from utils.plot import plot_results


class TestLatencyFiniteTime(Experiment):

    def build(self, param) -> Program:
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

    def analyze(self, param_setting: Dict[str, Any], p: Program):
        ctmc: SingleServerCTMC = p.build()
        pi = ctmc.get_init_state()
        res = ctmc.finite_time_analysis(pi, {'main queue size': ctmc.main_queue_size_analysis},
                                        sim_time=param_setting['sim_time'], sim_step=param_setting['sim_step'])
        avg_len = [0.0]
        var_len = [0.0]
        std_len = [0.0]
        runtime = [0.0]

        for res_step in res.values():
            if 'main queue size' in res_step.keys():
                size_res = res_step['main queue size']
                if isinstance(size_res, dict):
                    avg_len.append(size_res['avg'])
                    var_len.append(size_res['variance'])
                    std_len.append(size_res['std'])
            if 'wallclock_time' in res_step.keys():
                runtime.append(res_step['wallclock_time'])

        assert len(avg_len) == len(var_len) == len(std_len) == len(runtime)
        results = {'avg_len': avg_len, 'var_len': var_len, 'std_len': std_len, 'runtime': runtime}
        results.update(param_setting)
        return results

    def show(self, results):
        plot_results(results['sim_step'], results['avg_len'], results['var_len'], results['std_len'],
                     results['runtime'], 'latency_finite_time_' + results['file_name'])


if __name__ == "__main__":
    t = TestLatencyFiniteTime()
    program = t.build({})
    t.show(t.analyze({'sim_time': 2000, 'sim_step': 500, 'file_name': program.name + '.png'}, program))
