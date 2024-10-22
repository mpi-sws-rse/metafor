import time
import unittest
import numpy
import numpy.typing as npt
import sys

from dsl import Work, Server, Source, StateMachineSource, Program, DependentCall, Constants
from model.single_server.ctmc import SingleServerCTMC


def timed_call(f, *args, **kwargs):
    start = time.time()
    v = f(*args, **kwargs)
    end = time.time()
    print("Total time: ", end - start, " seconds")
    return v


def simple_analysis(p: Program):
    print("Building CTMC")
    ctmc = timed_call(Program.build, p)
    print("Computing stationary distribution")
    pi = timed_call(SingleServerCTMC.get_stationary_distribution, ctmc)
    print(pi)
    print("Average queue size = ", ctmc.main_queue_size_average(pi))
    print("Average retry queue size = ", ctmc.retry_queue_size_average(pi))
    requests = p.get_requests("server")
    print(requests)
    for i, r in enumerate(requests):
        print("Average latency for ", r, " = ", ctmc.latency_average(pi, i))
    ctmc.finite_time_analysis(
        ctmc.get_init_state(),
        analyses={"mainq_avg": ctmc.main_queue_size_average},
        sim_time=10000,
        sim_step=10,
    )


class TestDSL(unittest.TestCase):
    def test_single_server_single_request(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 5,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 3, and has no downstream work
        """
        apis = {
            "rd": Work(
                10, []
            )  # `rd` API call has service rate 3 and no downstream work
        }
        s = Server("server", apis, 100, 20)
        rd_src = Source("reader", "rd", 9.5, 9, 3)

        p = Program("single_server")
        p.add_server(s)
        p.add_source(rd_src)
        p.connect("reader", "server")
        timed_call(lambda: simple_analysis(p))

    def test_single_server_fast(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 2,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 3, and has no downstream work
        """
        apis = {
            "rd": Work(
                1000/2, []
            )  # `rd` API call has service rate 3 and no downstream work
        }
        s = Server("server", apis, 280, 20)
        rd_src = Source("reader", "rd", 2, 5, 3)

        p = Program("single_server")
        p.add_server(s)
        p.add_source(rd_src)
        p.connect("reader", "server")
        timed_call(lambda: simple_analysis(p))

    def test_single_server_multiple_requests(self):
        """A single server and two sources processing API call `rd`: the first source sends requests at rate 5 and the second at rate 2,
        each with a timeout of 10 and 3 retries. The server processes `rd` with rate 10, and has no downstream work
        """
        apis = {"rd": Work(3, [])}  # `rd` API call has rate 10 and no downstream work
        s = Server("server", apis, 100, 100)
        rd_src1 = Source("reader1", "rd", 5, 5, 3)
        rd_src2 = Source("reader2", "rd", 2, 10, 3)
        p = Program("single_server_multiple_sources")
        p.add_server(s)
        p.add_source(rd_src1)
        p.add_source(rd_src2)
        p.connect("reader1", "server")
        p.connect("reader2", "server")

        timed_call(lambda: simple_analysis(p))

    def test_storage_server(self):
        apis = { 'get' : Work(10, []),
                'put' : Work(20, []),
                'list' : Work(2, []),
        }
        node = Server('node', apis, 300, 5, 1)

        putsrc = Source('putsrc', 'put', 2, 5, 1)
        getsrc = Source('getsrc', 'get', 2, 3, 1)
        listsrc = Source('listsrc', 'list', 2, 5, 1)

        p = Program("storage_node")
        p.add_server(node)
        p.add_source(putsrc)
        p.add_source(getsrc)
        p.add_source(listsrc)
        p.connect('putsrc', 'node')
        p.connect('getsrc', 'node')
        p.connect('listsrc', 'node')

        for qsize in range(10, 30, 20):
            numpy.set_printoptions(threshold=sys.maxsize)
            start = time.time()
            node.qsize = qsize
            ctmc = p.build()
            pi = ctmc.get_stationary_distribution()
            print("Average queue size = ", ctmc.main_queue_size_average(pi))
            print(pi)
            for (i, req) in enumerate(p.get_requests('node')):
                print("Qsize = ", qsize, " Request = ", req, end='')
                l = ctmc.latency_average(pi, i)
                print(' has average latency ', l)
                ctmc.latency_percentile(pi, req_type = i, percentile = 50.0)
            print("Wallclock time = ", time.time() - start)

    def test_single_server_single_request_multiple_threads(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 9.5,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 10, and has no downstream work
        """
        apis = {"rd": Work(10, [])}  # `rd` API call has rate 10 and no downstream work
        s = Server("server", apis, 100, 100, 3)
        rd_src = Source("reader", "rd", 9.5, 5, 3)

        p = Program("single_server_multiple_threads")
        p.add_server(s)
        p.add_source(rd_src)
        p.connect("reader", "server")
        timed_call(lambda: simple_analysis(p))

    def test_single_server_multiple_reqs(self):
        """A single server and a single source processing API call `rd`: the source sends requests at rate 9.5,
        with a timeout of 10 and 3 retries. The server processes `rd` with rate 10, and has no downstream work
        """
        apis = {
            "rd": Work(10, []),  # `rd` API call has rate 10 and no downstream work
            "wr": Work(10, []),
            "bigwr": Work(20, []),
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
        timed_call(lambda: simple_analysis(p))

    def test_two_servers(self):
        """Two servers in series and a single source processing API call `rd`: the source sends requests at rate 9.5,
        with a timeout of 10 and 3 retries. The first server processes `rd` with rate 10, pushes work to the second
        server"""
        rates = {
            "rd": Work(
                10,
                [
                    DependentCall(
                        "server2", "server", "rd", Constants.CLOSED, 8.5, 10, 3
                    )
                ],
            )
        }
        apis = {"rd": Work(10, [])}
        s = Server("server", rates, 20, 100, 1)
        rd_src = Source("client", "rd", 9.5, 5, 3)

        s2 = Server("server2", apis, 100, 100, 1)
        p = Program("two_servers")
        p.add_server(s)
        p.add_server(s2)
        p.add_source(rd_src)
        p.connect("client", "server")

class TestBasic(unittest.TestCase):
    def test_sources(self):
        s = Source('foo', 'foo', 1, 0, 0)
        states = numpy.array([(2, 3, 1), (4, 3, 3)])
        transitions = numpy.array([[0, 10], [2, 0]])
        assert(states.shape[0] == transitions.shape[1])
        s = StateMachineSource('bar', 'foo', transitions, states)
        assert(s.num_states() == 2)

if __name__ == "__main__":
    unittest.main()
