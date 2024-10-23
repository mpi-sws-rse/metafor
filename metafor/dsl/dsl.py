import functools
from typing import List, Set, Tuple, Optional
import numpy as np
import numpy.typing as npt
from numpy import float64

from model.ctmc import CTMC
from model.multi_server.ctmc import MultiServerCTMC
from model.single_server.ctmc import SingleServerCTMC


class Constants:
    CLOSED = 1
    OPEN = 2


# a source of requests: an exponential distribution with rate `arrival_rate`
class Source:
    def __init__(
        self,
        name: str,
        api_name: str,
        arrival_rate: float,
        timeout: int,
        retries: int = 0,
    ):
        self.name = name  # unique name for this source, eg `client1`
        self.api_name = api_name  # name of the API call, e.g., `rd`
        assert arrival_rate >= 0.0
        assert timeout >= 0
        self.arrival_rate = arrival_rate  # lambda for the exponential distribution
        self.timeout = timeout
        self.retries = retries if retries >= 0 else 0

    def __to_string__(self):
        return "%s: generates %s: (arr %f, to %d, re %d)" % (self.name, self.api_name, self.arrival_rate, self.timeout,
                                                             self.retries)
    
    def print(self):
        print(self.__to_string__())
    
    def num_states(self):
        return 1


# a source of requests that is a general phase type distribution
# such a source can be implemented as its own CTMC, with an absorbing state.
# when the state of the CTMC is its absorbing state, we generate a request and re-initialize
#
# Q: can we model mixtures of exponentials more simply?
class PhaseTypeSource(Source):
    pass


class MixtureSource(Source):
    pass

# a state machine source is useful to model bursty traffic. 
# In a state machine source, each state has its own arrival rate/timeout/retry and states transition among themselves according
# to exponential distributions given in the transition matrix
# Q: is this a special case of a Phase type distribution?
class StateMachineSource(Source):
    # TBD: HANDLE THESE IN CTMC CREATION
    def __init__(self, name: str, api_name: str, transition_matrix: npt.NDArray[np.float64], lambda_map: npt.NDArray[tuple[float64, int, int]]):
        self.name = name
        self.api_name = api_name
        assert(np.all(np.vectorize(lambda x: x>=0.0)(transition_matrix))), "All entries in the transition map should be nonnegative"
        assert(np.all(np.vectorize(lambda x: x>=0.0)(lambda_map))), "All entries in the lambda map should be nonnegative"
        assert(lambda_map.shape[0] == transition_matrix.shape[0] == transition_matrix.shape[1]), "Transition map and lambda map have different dimensions"
        self.lambda_map = lambda_map
        self.transition_matrix = transition_matrix
        self.nstates = lambda_map.shape[0]

    def __to_string__(self):
        return "%s: generates %s: (arr %f, to %d, re %d)" % (self.name, self.api_name, self.arrival_rate, self.timeout,
                                                             self.retries)
    
    def print(self):
        print(self.__to_string__())

    def num_states(self):
        return self.nstates
        

class DependentCall:
    # callee is the downstream server to which the request is sent
    # caller is the server that forwarded the request (the parent server)
    # api_name is the name of the API call, e.g., `rd`
    def __init__(
        self,
        server_name: str,
        parent_server_name: str,
        api_name: str,
        call_type: Constants,
        arrival_rate: float, # do we need this? this should be set by the processing speed, no?
        timeout: int,
        retry: int,
    ):
        self.callee = server_name
        self.caller = parent_server_name
        self.api_name = api_name
        assert arrival_rate >= 0.0
        self.arrival_rate = arrival_rate  # lambda for the exponential distribution: check if needed
        self.call_type = call_type
        self.timeout = timeout
        self.retry = retry


# Work describes how a server processes a request
# A Work has a processing_rate and further processing of type `DependentCall` to downstream servers
class Work:
    # Question: We can have two semantics: the downstream jobs are simply sent on or the downstream jobs are all
    # finished before a job is taken off
    # I am not sure how to construct the CTMC in the latter case
    def __init__(self, processing_rate: float, downstream: List[DependentCall]):
        self.processing_rate = processing_rate
        self.downstream = downstream


# a server with queues to handle requests
# `thread_pool` is the number of threads processing requests
# `work` is a map from name to `Work`
class Server:
    def __init__(
        self,
        name: str,
        apis: dict[str, Work],
        qsize: int,
        orbit_size: int,
        thread_pool: int = 1,
    ):
        self.name = name
        self.apis = apis
        self.qsize = qsize
        self.orbit_size = orbit_size
        assert thread_pool >= 1
        self.thread_pool = thread_pool

    def __to_string__(self):
        api_strings = ','.join(self.apis.keys())
        return "%s: serves %s [q %d orbit %d threads %d]" % (self.name, api_strings, self.qsize, self.orbit_size,
                                                             self.thread_pool)

    def print(self):
        print(self.__to_string__())

    def num_states(self) -> int:
        return self.qsize * self.orbit_size


class Program:
    def __init__(self, name="<anon>"):
        self.name = name
        self.sources = {}
        self.servers = {}
        self.connections = []

    def add_server(self, server: Server):
        assert not (server.name in self.servers)
        self.servers[server.name] = server

    def add_source(self, source: Source):
        assert not (source.name in self.sources)
        self.sources[source.name] = source

    def connect(self, source_name: str, server_name: str):
        assert source_name in self.sources
        assert server_name in self.servers
        assert self.sources[source_name].api_name in self.servers[server_name].apis
        self.connections.append((source_name, server_name))

    def get_callees(self, server: Server) -> Set[Server]:
        callees = set()
        jobs: List[Work] = [job for job in server.apis.values()]
        for job in jobs:
            for dependent_call in job.downstream:
                callees.add(self.servers.get(dependent_call.callee))
        return callees

    def get_root_server(self) -> Server:
        all_servers = set(self.servers.values())
        callees = set()
        for server in all_servers:
            callees = callees.union(self.get_callees(server))
        root = all_servers.difference(callees)
        assert len(root) == 1
        return list(root)[0]

    def get_server(self, sname) -> Optional[Server]:
        return self.servers.get(sname, None)
    
    def get_source(self, sname) -> Optional[Source]:
        return self.sources.get(sname, None)

    def get_params(
        self, server: Server, connections: List[Tuple[str, str]]
    ) -> Tuple[List[float], List[float], List[int], List[int], List[Work]]:
        sources: List[Source] = [
            self.sources[source_name] for source_name, _ in connections
        ]
        arrival_rates: List[float] = [source.arrival_rate for source in sources]
        jobs: List[Work] = [server.apis[source.api_name] for source in sources]
        processing_rates: List[float] = [job.processing_rate for job in jobs]
        timeouts: List[int] = [source.timeout for source in sources]
        retries: List[int] = [source.retries for source in sources]
        return arrival_rates, processing_rates, timeouts, retries, jobs

    def get_connections(self, server: Server) -> List[Tuple[str, str]]:
        connections = [
            (source_name, server_name)
            for source_name, server_name in self.connections
            if server_name == server.name
        ]
        return connections

    def get_requests(self, s: str) -> List[str]:
        server = self.servers.get(s, None)
        if server is None:
            raise "Unknown server " + s
        return list(server.apis.keys())

    # build takes a program configuration and constructs a CTMC out of it
    def build(self) -> CTMC:
        num_states = functools.reduce(
            lambda a, s: a * s.num_states(), self.servers.values(), 1
        ) * functools.reduce(lambda a, s: a * s.num_states(), self.sources.values(), 1)
        print("Program: ", self.name, ", Number of states = ", num_states)

        if len(self.servers) == 1:  # single server
            _, server_name = self.connections[0]
            server: Server = self.servers[server_name]
            arrival_rates, processing_rates, timeouts, retries, _ = self.get_params(
                server, self.connections
            )
            ctmc = SingleServerCTMC(
                server.qsize,
                server.orbit_size,
                arrival_rates,
                processing_rates,
                timeouts,
                retries,
                server.thread_pool,
            )
        else:  # multiple servers in serial connection
            root_server: Server = self.get_root_server()
            serial_servers: List[Server] = [root_server]
            callees = self.get_callees(root_server)
            while len(callees) == 1:
                serial_servers += callees
                server = list(callees)[0]
                callees = self.get_callees(server)
            main_queue_sizes: List[int] = [server.qsize for server in serial_servers]
            retry_queue_sizes: List[int] = [
                server.orbit_size for server in serial_servers
            ]

            connections = self.get_connections(root_server)
            arrival_rates, processing_rates, timeouts, retries, jobs = self.get_params(
                root_server, connections
            )

            for job in jobs:
                for dependent_call in job.downstream:
                    arrival_rates.append(dependent_call.arrival_rate)
                    processing_rates.append(job.processing_rate)
                    timeouts.append(dependent_call.timeout)
                    retries.append(dependent_call.retry)
            thread_pools: List[int] = [server.thread_pool for server in serial_servers]
            parent_list: List[List[int]] = [[]]
            parent_list += [[i] for i in range(0, len(serial_servers))]
            sub_tree_list: List[List[int]] = [
                [j for j in range(i, len(serial_servers))]
                for i in range(0, len(serial_servers))
            ]
            ctmc = MultiServerCTMC(
                len(serial_servers),
                main_queue_sizes,
                retry_queue_sizes,
                arrival_rates,
                processing_rates,
                timeouts,
                retries,
                thread_pools,
                parent_list,
                sub_tree_list,
            )
        return ctmc

    def print(self):
        print("Program: ", self.name)
        for (s, c) in self.connections:
            print(s, " --> ", c)
        print("Servers: ", end= " ")
        for sname, s in self.servers.items():
            print("\t", end = " "); s.print()
        print("Sources: ", end= " ")
        for sname, s in self.sources.items():
            print("\t", end = " "); s.print()

        
            
"""
    def average_lengths_analysis(self, plot_params: PlotParameters):
        ctmc: CTMC = self.build()
        file_name = self.name + ".png"
        analyzer: Analyzer = Analyzer(ctmc, file_name)
        analyzer.average_lengths_analysis(plot_params)

    def fault_scenario_analysis(self, plot_params: PlotParameters):
        ctmc: CTMC = self.build()
        file_name = self.name + ".png"
        analyzer: Analyzer = Analyzer(ctmc, file_name)
        analyzer.fault_scenario_analysis(plot_params)

    def latency_analysis(self, plot_params: PlotParameters):
        ctmc: CTMC = self.build()
        file_name = self.name + ".png"
        analyzer: Analyzer = Analyzer(ctmc, file_name)

        _, server_name = self.connections[0]
        server: Server = self.servers[server_name]
        sources: List[Source] = [
            self.sources[source_name] for source_name, _ in self.connections
        ]
        jobs: List[Work] = [server.apis[source.api_name] for source in sources]
        job_types = [job_type for job_type in range(0, len(jobs))]

        for job_type in job_types:
            analyzer.latency_analysis(plot_params, job_type)
"""
