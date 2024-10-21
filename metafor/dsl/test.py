import time
import unittest

from dsl import Work, Server, Source, Program, DependentCall, Constants
from utils.plot_parameters import PlotParameters
from model.single_server.single_server_ctmc import SingleServerCTMC


def timed_call(f, *args, **kwargs):
    start = time.time()
    v = f(*args, **kwargs)
    end = time.time()
    print("Time: ", end - start, " seconds")
    return v


class TestCTMC(unittest.TestCase):
    def test_single_server_single_request(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 5,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 3, and has no downstream work"""
        apis = {
            'rd': Work(10, [])  # `rd` API call has service rate 3 and no downstream work
        }
        s = Server("server", apis, 100, 20)
        rd_src = Source("reader", "rd", 9.5, 9, 3)

        p = Program("single_server")
        p.add_server(s)
        p.add_source(rd_src)
        p.connect("reader", "server")

        print("Building CTMC")
        ctmc = timed_call(Program.build, p)
        print("Computing stationary distribution")
        pi = timed_call(SingleServerCTMC.get_stationary_distribution, ctmc)
        print(pi)
        print("Average queue size = ", ctmc.main_queue_average_size(pi))
        print("Average retry queue size = ", ctmc.retry_queue_size_average(pi))

    def test_single_server_multiple_requests(self):
        """A single server and two sources processing API call `rd`: the first source sends requests at rate 5 and the second at rate 2,
        each with a timeout of 10 and 3 retries. The server processes `rd` with rate 10, and has no downstream work"""
        apis = {
            'rd': Work(3, [])  # `rd` API call has rate 10 and no downstream work
        }
        s = Server("server", apis, 100, 100)
        rd_src1 = Source("reader1", "rd", 5, 5, 3)
        rd_src2 = Source("reader2", "rd", 2, 10, 3)
        p = Program("single_server_multiple_sources")
        p.add_server(s)
        p.add_source(rd_src1)
        p.add_source(rd_src2)
        p.connect("reader1", "server")
        p.connect("reader2", "server")

        # p.average_lengths_analysis(PlotParameters(step_time=100, sim_time=1000))
        start = time.time()
        # p.latency_analysis(PlotParameters())
        # ctmc = p.build()
        end = time.time()
        print("Latency analysis took ", end - start, " seconds")

    def test_single_server_single_request_multiple_threads(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 9.5,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 10, and has no downstream work"""
        apis = {
            'rd': Work(10, [])  # `rd` API call has rate 10 and no downstream work
        }
        s = Server("server", apis, 100, 100, 3)
        rd_src = Source("reader", "rd", 9.5, 5, 3)

        p = Program("single_server_multiple_threads")
        p.add_server(s)
        p.add_source(rd_src)
        p.connect("reader", "server")
        # p.average_lengths_analysis(PlotParameters())
        start = time.time()
        # p.latency_analysis(PlotParameters())
        end = time.time()
        print("Latency analysis took ", end - start, " seconds")

    def test_single_server_multiple_reqs(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 9.5,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 10, and has no downstream work"""
        apis = {
            'rd': Work(10, []),  # `rd` API call has rate 10 and no downstream work
            'wr': Work(10, []),
            'bigwr': Work(20, [])
        }
        s = Server("server", apis, 10, 10, 1)
        rd_src = Source("client1", "rd", 9.5, 5, 3)
        wr_src = Source("client2", "wr", 4.5, 10, 3)
        jumbo_src = Source("client3", "bigwr", 1, 10, 3)
        p = Program("server_with_mult_req")
        p.add_server(s)
        p.add_source(rd_src)
        p.add_source(wr_src)
        p.add_source(jumbo_src)
        p.connect("client1", "server")
        p.connect("client2", "server")
        p.connect("client3", "server")
        start = time.time()
        # p.latency_analysis(PlotParameters())
        end = time.time()
        print("Latency analysis took ", end - start, " seconds")

    def test_two_servers(self):
        """Two servers in series and a single source processing API call `rd`: the source sends requests at rate 9.5,
        with a timeout of 10 and 3 retries. The first server processes `rd` with rate 10, pushes work to the second
        server"""
        rates = {  # ignore for the moment... work in progress
            'rd': Work(10, [DependentCall("server2", "server", "rd", Constants.CLOSED, 8.5, 10, 3)])
        }
        apis = {
            'rd': Work(10, [])
        }
        s = Server("server", rates, 20, 100, 1)
        rd_src = Source("client", "rd", 9.5, 5, 3)

        s2 = Server("server2", apis, 100, 100, 1)
        p = Program("two_servers")
        p.add_server(s)
        p.add_server(s2)
        p.add_source(rd_src)
        p.connect("client", "server")
        p.fault_scenario_analysis(PlotParameters())


if __name__ == "__main__":
    unittest.main()
