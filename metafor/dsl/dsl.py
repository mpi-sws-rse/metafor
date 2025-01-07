import functools
from typing import List, Set, Tuple, Optional, Dict

import numpy as np
import numpy.typing as npt
from numpy import float64

from metafor.utils.graph import Graph
from metafor.model.ctmc import CTMC, CTMCRepresentation
from metafor.model.multi_server.ctmc import MultiServerCTMC
from metafor.model.single_server.ctmc import SingleServerCTMC

from metafor.simulator import Server as DESServer
from metafor.simulator import OpenLoopClientWithTimeout as DESOpenLoopClientWithTimeout
from metafor.simulator import ExponentialDistribution
from metafor.simulator import Simulator

class Constants:
    WAIT_UNTIL_DONE = 1
    FIRE_AND_FORGET = 2


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

    def __str__(self):
        return "%s: generates %s: (arr %f, to %d, re %d)" % (
            self.name,
            self.api_name,
            self.arrival_rate,
            self.timeout,
            self.retries,
        )

    def print(self):
        print(self.__str__())

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
    def __init__(self, name: str, 
                 api_name: str, 
                 transition_matrix: npt.NDArray[np.float64],
                 lambda_map: npt.NDArray[tuple[float64, int, int]]):
        super().__init__(name, api_name, 0.0, 0, retries=0) # we set the arrival rate, timeout, and retries to default values
        assert np.all(
            np.vectorize(lambda x: x >= 0.0)(transition_matrix)
        ), "All entries in the transition map should be nonnegative"
        assert np.all(
            np.vectorize(lambda x: x >= 0.0)(lambda_map)
        ), "All entries in the lambda map should be nonnegative"
        assert (
                lambda_map.shape[0]
                == transition_matrix.shape[0]
                == transition_matrix.shape[1]
        ), "Transition map and lambda map have different dimensions"
        self.lambda_map = lambda_map
        self.transition_matrix = transition_matrix
        self.nstates = lambda_map.shape[0]

    def __str__(self):
        return "%s: generates %s: (arr %f, to %d, re %d)" % (
            self.name,
            self.api_name,
            self.arrival_rate,
            self.timeout,
            self.retries,
        )

    def print(self):
        print(self.__str__())

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
            timeout: int,
            retry: int,
    ):
        self.callee = server_name
        self.caller = parent_server_name
        self.api_name = api_name
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

    def __str__(self):
        return "%f" % self.processing_rate


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

    def __str__(self):
        apis = [apiname + '@' + work.__str__() for (apiname, work) in self.apis.items()]
        api_strings = ",".join(apis)
        return "%s: serves %s [q %d orbit %d threads %d]" % (
            self.name,
            api_strings,
            self.qsize,
            self.orbit_size,
            self.thread_pool,
        )

    def print(self):
        print(self.__str__())

    def num_states(self) -> int:
        return self.qsize * self.orbit_size
    
    # return the work associated with `apiname` 
    def get_work(self, apiname: str) -> Work:
        return self.apis[apiname]


class Program:
    def __init__(self, name="<anon>", retry_when_full: bool = False):
        self.name = name
        self.sources = {}
        self.servers = {}
        self.connections = []
        self.retry_when_full = retry_when_full

    def add_server(self, server: Server):
        assert not (server.name in self.servers)
        self.servers[server.name] = server

    def add_servers(self, servers: List[Server]):
        for s in servers:
            self.add_server(s)

    def add_source(self, source: Source):
        assert not (source.name in self.sources)
        self.sources[source.name] = source

    def add_sources(self, sources: List[Source]):
        for s in sources:
            self.add_source(s)

    def connect(self, source_name: str, server_name: str):
        assert source_name in self.sources, "Unknown source: %s" % source_name
        assert server_name in self.servers, "Unknown server: %s" % server_name
        assert self.sources[source_name].api_name in self.servers[server_name].apis
        self.connections.append((source_name, server_name))

    def get_callees(self, server: Server) -> Set[Server]:
        callees = set()
        jobs: List[Work] = [job for job in server.apis.values()]
        for job in jobs:
            for dependent_call in job.downstream:
                if dependent_call.caller == server.name:
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

    def get_number_of_servers(self) -> int:
        return len(self.servers)
    
    def get_server(self, sname) -> Optional[Server]:
        return self.servers.get(sname, None)

    def get_source(self, sname) -> Optional[Source]:
        return self.sources.get(sname, None)

    def get_jobs_with_sources(self, server: Server) -> Dict[Work, List[Source]]:
        jobs_with_sources: Dict[Work, List[Source]] = dict()
        sources = self.get_sources(server)
        for api_name, job in server.apis.items():
            sources_for_job = [source for source in sources if source.api_name == api_name]
            assert len(sources_for_job) >= 1
            jobs_with_sources[job] = sources_for_job
        return jobs_with_sources

    def get_job(self, jobs: List[Work], dependant_call: DependentCall) -> Work:
        return [job for job in jobs if dependant_call in job.downstream][0]

    def get_job_processing_rate(self, jobs: List[Work], dependant_call: DependentCall):
        job: Work = self.get_job(jobs, dependant_call)
        return job.processing_rate

    def get_sources(self, server: Server) -> List[Source]:
        return [self.sources[source_name] for source_name, _ in self.get_connections(server)]

    def get_params(self, server: Server) -> Tuple[List[float], List[float], List[int], List[int], List[Work]]:
        sources: List[Source] = self.get_sources(server)
        arrival_rates: List[float] = [source.arrival_rate for source in sources]
        processing_rates: List[float] = []
        jobs_with_sources = self.get_jobs_with_sources(server)
        jobs: List[Work] = [job for job in jobs_with_sources.keys()]
        for job, sources_for_job in jobs_with_sources.items():
            for i in range(len(sources_for_job)):
                processing_rates.append(job.processing_rate)
        timeouts: List[int] = [source.timeout for source in sources]
        retries: List[int] = [source.retries for source in sources]
        assert len(arrival_rates) == len(processing_rates) == len(timeouts) == len(retries)
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

    def get_callgraph(self):
        g = Graph()
        for sname, s in self.servers.items():
            g.add_node(sname)
            callees = self.get_callees(s)
            for c in callees:
                g.add_edge(sname, c.name)
        print("Acyclic? ", g.is_acyclic())
        print(g.__str__())

    # build takes a program configuration and constructs a CTMC out of it
    def build(self, representation: CTMCRepresentation = CTMCRepresentation.EXPLICIT) -> CTMC:
        num_states = functools.reduce(
            lambda a, s: a * s.num_states(), self.servers.values(), 1
        ) * functools.reduce(lambda a, s: a * s.num_states(), self.sources.values(), 1)
        print("Program: ", self.name, ", Number of states = ", num_states)

        if len(self.servers) == 1:  # single server
            _, server_name = self.connections[0]
            server: Server = self.servers[server_name]
            arrival_rates, processing_rates, timeouts, retries, _ = self.get_params(server)
            ctmc = SingleServerCTMC(
                server.qsize,
                server.orbit_size,
                arrival_rates,
                processing_rates,
                timeouts,
                retries,
                server.thread_pool,
                representation=representation,
                retry_when_full=self.retry_when_full,
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

            arrival_rates, processing_rates, timeouts, retries, jobs = self.get_params(root_server)
            for job in jobs:
                for dependent_call in job.downstream:
                    arrival_rates.append(self.get_job_processing_rate(jobs, dependent_call))
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
        for s, c in self.connections:
            print(s, "\t -->\t ", c)
        print("Servers: ", end=" ")
        for sname, s in self.servers.items():
            print("\t", end=" ")
            s.print()
        print("Sources: ", end=" ")
        for sname, s in self.sources.items():
            print("\t", end=" ")
            s.print()

class Transpiler:
    def __init__(self, p: Program):
        self.p = p
        self.des_server = None
        self.des_clients = []

    # The next two functions are used to externally set triggers by changing parameters  
    # It's bad global variable programming .... but we did not want to spend too much
    # time designing the simulation interface  
    def get_server(self) -> DESServer:
        assert self.des_server is not None
        return self.des_server
    
    def get_client(self) -> DESOpenLoopClientWithTimeout:
        return self.des_clients
    
    # transpile a program into a simulation model
    def transpile(self):
        # currently works only for a single server
        server = self.p.get_root_server()
        rates = { }
        for apiname, apiwork in server.apis.items():
            rates[apiname] = ExponentialDistribution(apiwork.processing_rate)
            # XXX TODO: Process dependent calls for multi server settings

        self.des_server = DESServer(server.name, 
                                          service_time_distribution=rates, 
                                          queue_size=server.qsize, 
                                          thread_pool=server.thread_pool, 
                                          retry_queue_size=server.orbit_size)
        clients = self.p.get_sources(server)
        self.des_clients = []
        for client in clients:
            des_client = DESOpenLoopClientWithTimeout(client.name, 
                                                              client.api_name, 
                                                              ExponentialDistribution(client.arrival_rate), 
                                                              client.timeout, client.retries)
            des_client.server = self.des_server
            self.des_clients.append(des_client)

    def simulate(self, sim_id: int, sim_time=30):
        assert self.des_server is not None, "Server is None: did you forget to transpile?"
        simulator = Simulator(self.des_server, self.des_clients, "exp.csv")
        simulator.sim(sim_time)
        simulator.analyze()


def run_simulation(p: Program, number_of_runs: int, time: float):
    tp = Transpiler(p)
    tp.transpile()
    for i in range(number_of_runs):
        tp.reset()
        tp.simulate(i, sim_time=time)

        # set trigger


        
