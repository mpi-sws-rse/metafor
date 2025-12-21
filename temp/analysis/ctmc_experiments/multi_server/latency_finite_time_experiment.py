from typing import Any, Dict

import numpy as np

from analysis.experiment import Experiment
from dsl.dsl import Source, Server, Work, Program, DependentCall, Constants
from model.multi_server.ctmc import MultiServerCTMC

from utils.plot import plot_results


class TestLatencyFiniteTime(Experiment):

    def build(self, param) -> Program:
        rates = {
            "rd": Work(
                10,
                [
                    DependentCall(
                        "server2", "server", "rd", Constants.CLOSED, 10, 3
                    )
                ],
            )
        }
        apis = {"rd": Work(10, [])}
        s = Server("server", rates, 20, 10, 1)
        rd_src = Source("client", "rd", 9.5, 5, 3)

        s2 = Server("server2", apis, 30, 10, 1)
        p = Program("two_servers")
        p.add_server(s)
        p.add_server(s2)
        p.add_source(rd_src)
        p.connect("client", "server")
        return p

    def analyze(self, param_setting: Dict[str, Any], p: Program):
        ctmc: MultiServerCTMC = p.build()
        pi = ctmc.get_init_state()
        res = ctmc.finite_time_analysis(pi, {'main queue size': ctmc.main_queue_size_analysis},
                                        sim_time=param_setting['sim_time'], sim_step=param_setting['sim_step'])
        avg_len = np.array([0.0] * ctmc.server_num)
        var_len = np.array([0.0] * ctmc.server_num)
        std_len = np.array([0.0] * ctmc.server_num)
        runtime = np.array([0.0])

        for res_step in res.values():
            if 'main queue size' in res_step.keys():
                size_res = res_step['main queue size']
                if isinstance(size_res, dict):
                    avg_len = np.append(avg_len, size_res['avg'])
                    var_len = np.append(var_len, size_res['var'])
                    std_len = np.append(std_len, size_res['std'])
            if 'wallclock_time' in res_step.keys():
                runtime = np.append(runtime, res_step['wallclock_time'])

        assert len(avg_len) == len(var_len) == len(std_len) == ctmc.server_num * len(runtime)
        results = {'avg_len': avg_len, 'var_len': var_len, 'std_len': std_len, 'runtime': runtime}
        results.update(param_setting)
        return results

    def show(self, results):
        avg_len = results['avg_len']
        var_len = results['var_len']
        std_len = results['std_len']
        runtime = results['runtime']
        servers = int(len(avg_len) / len(runtime))
        for server_id in range(servers):
            plot_results(results['sim_step'],
                         avg_len[server_id:len(avg_len):servers],
                         var_len[server_id:len(avg_len):servers],
                         std_len[server_id:len(avg_len):servers],
                         runtime,
                         'latency_finite_time_server_' + str(server_id) + '_' + results['file_name'])


if __name__ == "__main__":
    t = TestLatencyFiniteTime()
    program = t.build({})
    t.show(t.analyze({'sim_time': 2000, 'sim_step': 500, 'file_name': program.name + '.png'}, program))
