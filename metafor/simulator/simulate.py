import heapq
import math
import multiprocessing
import os
import time
from typing import List

import itertools
import numpy as np
import pandas as pd
import random
from metafor.simulator.server import Context, Server, TokenBucket
from metafor.simulator.server_with_throttling import ServerWithThrottling
from metafor.simulator.server_with_LIFO import ServerWithLIFO
from metafor.simulator.statistics import StatData
from metafor.simulator.client import Client, OpenLoopClient, OpenLoopClientWithTimeout
from metafor.simulator.preprocessing import mean_variance_std_dev, compute_mean_variance_std_deviation
from metafor.utils.plot import plot_results
from metafor.simulator.job import exp_job, bimod_job, wei_job, ExponentialDistribution, WeibullDistribution, NormalDisttribution, LogNormalDistribution

import logging
logger = logging.getLogger(__name__)

import pickle




class Simulator:
    def __init__(
        self, 
        servers: dict[int, 'Server'],
        clients: List['Client'], 
        dag: dict,
        sim_id: int
    ):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.servers = servers

        # self.contexts = [Context(sim_id + 1, i + 1) for i in range(len(servers))]
        # for server, ctx in zip(servers, self.contexts):
        #     server.set_context(ctx)
        
        # for server_id, server in self.servers.items():
        #     server.set_context(Context(sim_id, server_id, dag))
        self.clients = clients
        self.dag = dag
        self.event_counter = itertools.count()

        self.reset()

        for server in self.servers.values():
            server.print()
        for client in self.clients:
            client.print()
        logger.setLevel(logging.INFO)
    
    # to start a simulator from the beginning, reset the state
    def reset(self):
        self.t = 0.0
        for server in self.servers.values():
            server.context.result = []   # clear between runs

        self.q = [
            (self.t, next(self.event_counter), client.generate, None)
            for client in self.clients
        ]
        heapq.heapify(self.q)


    def sim(self, max_t: float = 60.0):
        # This is the core simulation loop. Until we've reached the maximum simulation time, pull the next event off
        #  from a heap of events, fire whichever callback is associated with that event, and add any events it generates
        #  back to the heap.
        while len(self.q) > 0 and self.t < max_t:
            # Get the next event. Because `q` is a heap, we can just pop this off the front of the heap.
            (t, _, call, payload) = heapq.heappop(self.q)
            self.t = t
            # Execute the callback associated with this event
            new_events = call(t, payload)
            # If the callback returned any follow-up events, add them to the event queue `q` and make sure it is still a
            # valid heap
            if new_events:
                # self.q.extend(new_events)
                for ev in new_events:
                    t2, call2, payload2 = ev
                    heapq.heappush(self.q, (t2, next(self.event_counter), call2, payload2)) 

    def step(self):
        """
        Execute exactly one simulation event.
        Returns:
            (t, call, payload) of the executed event,
            or None if no events remain.
        """
        if not self.q:
            return None  # No events left

        # Pop next event
        (t, _, call, payload) = heapq.heappop(self.q)
        self.t = t

        # Execute event callback
        new_events = call(t, payload)
        #print("new events >>>>> ",new_events)

        # Add follow-up events
        if new_events:
            # self.q.extend(new_events)
            # heapq.heapify(self.q)
            for ev in new_events:
                t2, call2, payload2 = ev
                heapq.heappush(self.q, (t2, next(self.event_counter), call2, payload2)) 


        return (t, call, payload)
    
    def analyze(self):
        for server in self.servers.values():
            server.context.analyze()


def run_sims(max_t: float, fn: str, num_runs: int, step_time: int, sim_fn, mean_t: float, rho,
             queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, 
             fault_duration, throttle, ts, ap, queue_type, dist, dag):
    """
    Run a simulation `run_nums` times, outputting the results to `x_fn`, where x in [1, num_runs].
    One simulation is run for each client in `clients`.
    `max_t` is the maximum time to run the simulation (in ms).
    
    """
    file_names: List[str] = []

    #####################################################
    # Initialize multiple servers and aggregate results.
    #
    for i in range(num_runs):
        print("Running simulation " + str(i + 1) + " time(s)")
        current_fn = str(i + 1) + '_' + fn
        file_names.append(current_fn)
        job_type = exp_job(mean_t)
        servers, clients = sim_fn(mean_t, "client", "request", rho, queue_size,
                                  timeout_t, max_retries, rho_fault, rho_reset, fault_start, 
                                  fault_duration, throttle, ts, ap, queue_type, dist, dag, i)
        
        # for client in clients:
        #     client.server.file = current_fn
        #     client.server.start_time = time.time()
        siml = Simulator(servers, clients, dag, i+1)
        siml.reset()
        siml.sim(max_t)
        #print("LIST OF SERVERS ",servers)
        # print(siml.contexts[0].result[0:5])
        # print(siml.contexts[1].result[0:5])
        
        directory = os.path.dirname(f"data/{i + 1}_{fn}")
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        for sid in servers.keys():
            # print(server,"   ",server.id,"      ",server.downstream_server)
            # print(server.context.result[0:5])
            # print(server.downstream_server.context.result[0:5])
            # #exit() 

            df = pd.DataFrame(servers[sid].context.result)
            # reorder columns explicitly if needed
            df = df[['server','timestamp','latency','queue_length','retries',
                    'dropped','runtime','retries_left','service_time',
                    'throughput','request_id','attempt_id',
                    'retry_origin','client_retries_used','server_retries_used',
                    'dropped_queue_full', 'dropped_token_bucket']]
            df.to_csv(f"data/{i+1}_{fn}", header=False, mode='a', index=False)
         
         

    #exit()
    for i in range(1,len(servers)+1):
        latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std = mean_variance_std_dev(file_names, max_t, num_runs, step_time, mean_t,i)
        plot_results(step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, "results/discrete_results_server"+str(i)+".pdf")
        directory = os.path.dirname("data/server"+str(i)+"/sim_data.pkl")
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open("data/server"+str(i)+"/sim_data.pkl", "wb") as f:
            pickle.dump((step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, rho), f)

# Simulation with unimodal exponential service time and timeout
def make_sim_exp(mean_t: float, name: str, apiname: str, rho: float, queue_size: int, timeout_t: float,
                 max_retries: int, rho_fault: float, rho_reset: float, fault_start: float,
                 fault_duration: float, throttle:bool, ts:float, ap:float, queue_type:str,
                 dist: str, dag :dict, sim_id: int) -> List[Client]:
    
    clients = []
    job_name = apiname

    if dist=="exp":
        job_type = exp_job(mean_t)
        distribution = ExponentialDistribution
    elif dist=="wei":
        job_type = wei_job(mean_t)
        distribution = WeibullDistribution
    else:
        raise ValueError(f"Unsupported distribution: {dist}")

    # 1. Create servers 
    servers: dict[str, Server] = {}
    prev_server = []
    

 
 
    # retry_policy = {
    #     1: (20, 3),
    #     2: (15, 3),
    #     3: (10, 2),    
    #     4: (10, 2),
    #     5: (20, 0),     
    # }
    # servers = {
    #         1: ServerConfig(server_id=1, threads=10,  mean_service=0.10,
    #                         timeout=3.0, max_retries=3, net_delay=0.01),
    #         2: ServerConfig(server_id=2, threads=5,  mean_service=0.10,
    #                         timeout=1.5, max_retries=3, net_delay=0.01),
    #         3: ServerConfig(server_id=3, threads=5,  mean_service=0.25,
    #                         timeout=1.2, max_retries=2, net_delay=0.01),
    #         4: ServerConfig(server_id=4, threads=4,  mean_service=0.20,
    #                         timeout=1.0, max_retries=2, net_delay=0.01),
    #         5: ServerConfig(server_id=5, threads=4,  mean_service=0.15,
    #                         timeout=0.5, max_retries=0, net_delay=0.01),
    #     }

    retry_policy = {
        1: (6.0, 3),
        2: (2.7, 3),
        3: (2.5, 2),    
        4: (2.0, 2),
        5: (1.0, 0),     
    }

    # service distributions — pass rate = 1/mean_service_time
    service_dists = {
        1: ExponentialDistribution(1/0.10),    
        2: ExponentialDistribution(1/0.10),
        3: ExponentialDistribution(1/0.25),
        4: ExponentialDistribution(1/0.20),
        5: ExponentialDistribution(1/0.15),
    }

    # one shared latency distribution or per-service ones
    network_dists = {
        1: ExponentialDistribution(1 / 0.01),   # mean 1ms Auth → Gateway
        2: ExponentialDistribution(1 / 0.01),   # mean 2ms Gateway → Rec/Order
        3: ExponentialDistribution(1 / 0.01),
        4: ExponentialDistribution(1 / 0.01),   # mean 2ms Gateway → Rec/Order
        5: ExponentialDistribution(1 / 0.01),
    }

    thread_pool = {
        1: 10,   # Auth
        2: 5,   # Gateway
        3: 5,
        4: 4,   # Gateway
        5: 4,
    }

    bucket_config = {
        1: TokenBucket(capacity=50, refill_rate=12.0),
        2: TokenBucket(capacity=20, refill_rate=9.0),
        3: TokenBucket(capacity=15, refill_rate=10.0),
        4: TokenBucket(capacity=15, refill_rate=9.5),
        5: TokenBucket(capacity=30, refill_rate=20.0),
    }
    
    for i in dag.keys():
        timeout, retries = retry_policy[i]
        bucket = bucket_config.get(i)   # None means no rate limiting

        server_name = f"server_{i}"

        if throttle==False:
            server = Server(i, server_name, queue_size, thread_pool[i], service_dists[i], None, downstream_server=prev_server, timeout=timeout, max_retries=retries,token_bucket=bucket,network_dist=network_dists[i])
        else:    
            server = ServerWithThrottling(i, server_name, queue_size, thread_pool[i], service_dists[i], None, throttle, ts,ap, downstream_server=prev_server, timeout=timeout, max_retries=retries, token_bucket=bucket,network_dist=network_dists[i])
        
        if queue_type=="lifo":
            server = ServerWithLIFO(i, server_name, queue_size, thread_pool[i], service_dists[i], None,  downstream_server=prev_server, timeout=timeout, max_retries=retries, token_bucket=bucket,network_dist=network_dists[i])
        
        
        server.set_context(Context(sim_id,i))  #check
        
        servers[i] = server

        
    # 2. Connect servers
    for src, dsts in dag.items():
        servers[src].downstream_server = [servers[d] for d in dsts]

    # 3. Identify entry servers (roots) 
    all_nodes = set(dag.keys())
    downstream_nodes = {d for dsts in dag.values() for d in dsts}
    entry_nodes = list(all_nodes - downstream_nodes)

    # 4. Attach clients only to entry nodes
    clients = []
    
    client_timeout = 36
    client_retry = 3
    
    for entry in entry_nodes:
        client = OpenLoopClientWithTimeout(
            name, 
            apiname, 
            distribution, 
            rho, 
            job_type, 
            client_timeout, 
            client_retry, 
            rho_fault, 
            rho_reset, 
            fault_start,
            fault_duration
        )

        client.server = servers[entry]
        servers[entry].client = client

        clients.append(client)
        
    
    #print("  ",servers[0].downstream_server.id)
    return servers, clients



def run_discrete_experiment(
        max_t: float, runs: int, mean_t: float, rho: float, queue_size: int,
        timeout_t: float, max_retries: int, total_time: float, step_time: int,
        rho_fault: float, rho_reset: float, fault_start: float, fault_duration: float,
        throttle : bool, ts : float, ap : float, queue_type : str, dist: str, 
        dag : dict
):
   
    results_file_name = "exp_results.csv"
    start_time = time.time()
    process = multiprocessing.Process(target=run_sims, args=(max_t, results_file_name, runs, step_time, make_sim_exp,
                                                             mean_t, rho, queue_size, timeout_t, max_retries, rho_fault, 
                                                             rho_reset, fault_start, fault_duration, throttle, ts, ap, 
                                                             queue_type, dist, dag))
    process.start()
    process.join(total_time)
    if process.is_alive():
        print("Max simulation time reached, stopped the simulation")
        process.kill()
        # check if the mean, variance, and standard deviation have been computed; if not, compute them
        # if not done:
        compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t)
    end_time = time.time()
    runtime = end_time - start_time
    print("Running time: " + str(runtime) + " s")
