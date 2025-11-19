# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List


import numpy as np
import pandas as pd
import random
from metafor.simulator.server_multi import Context, Server
from metafor.simulator.statistics import StatData
from metafor.simulator.client_multi import Client, OpenLoopClient, OpenLoopClientWithTimeout

from metafor.utils.plot import plot_results
from metafor.simulator.job import exp_job, bimod_job, wei_job, ExponentialDistribution, WeibullDistribution

import logging
logger = logging.getLogger(__name__)

import pickle


class Simulator:
    def __init__(self, servers: List['Server'], clients: List['Client'], sim_id: int):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.servers = servers

        self.contexts = [Context(sim_id+1, i+1) for i,s in enumerate(servers)] 
        #[Context(sim_id + i, f"sim_{sim_id}_server_{s.sim_name}.csv") for i, s in enumerate(servers)]
        for i,server in enumerate(servers):
            server.set_context(Context(sim_id+1,i+1))
        self.clients = clients

        self.reset()

        for server in self.servers:
            server.print()
        for client in self.clients:
            client.print()
        logger.setLevel(logging.INFO)
    
    def __del__(self):
        for c in self.contexts:
            c.close()

    # to start a simulator from the beginning, reset the state
    def reset(self):
        self.t = 0.0
        # start the simulator by adding each client
        self.q = [(self.t, client.generate, None) for client in self.clients]
        heapq.heapify(self.q)

    def close(self):
        self.fp.close()

    def sim(self, max_t: float = 60.0):
        # This is the core simulation loop. Until we've reached the maximum simulation time, pull the next event off
        #  from a heap of events, fire whichever callback is associated with that event, and add any events it generates
        #  back to the heap.
        while len(self.q) > 0 and self.t < max_t:
            # Get the next event. Because `q` is a heap, we can just pop this off the front of the heap.
            (t, call, payload) = heapq.heappop(self.q)
            self.t = t
            # Execute the callback associated with this event
            new_events = call(t, payload)
            # If the callback returned any follow-up events, add them to the event queue `q` and make sure it is still a
            # valid heap
            if new_events is not None:
                self.q.extend(new_events)
                #print("new_events ",new_events)
                heapq.heapify(self.q)   

    def analyze(self):
        self.context.analyze()


def run_sims(max_t: float, fn: str, num_runs: int, step_time: int, sim_fn, mean_t: float, rho,
             queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, 
             fault_duration, dist, num_servers):
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
        servers, clients = sim_fn(mean_t, "client", "request", rho, queue_size, retry_queue_size, timeout_t, 
            max_retries, rho_fault, rho_reset, fault_start, fault_duration, dist, num_servers, i)
        
        # for client in clients:
        #     client.server.file = current_fn
        #     client.server.start_time = time.time()
        siml = Simulator(servers, clients, i+1)
        siml.reset()
        siml.sim(max_t)
        print("LIST OF SERVERS ",servers)
        # print(siml.contexts[0].result[0:5])
        # print(siml.contexts[1].result[0:5])
        
        directory = os.path.dirname(f"data/{i + 1}_{fn}")
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        for server in servers:
            # print(server,"   ",server.id,"      ",server.downstream_server)
            # print(server.context.result[0:5])
            # print(server.downstream_server.context.result[0:5])
            # #exit() 
            df = pd.DataFrame(server.context.result, columns=['server', 'timestamp', 'latency', 'queue_length', 'retries', 'dropped', 'runtime', 'retries_left', 'service_time'])
            df.to_csv(f"data/{i + 1}_{fn}",  header=False,  mode='a', index=False)
            # if server.downstream_server:
            #     df = pd.DataFrame(server.downstream_server.context.result, columns=['server', 'timestamp', 'queue_length', 'latency', 'retries', 'dropped', 'runtime', 'retries_left', 'service_time'])
            #     df.to_csv(f"{i + 1}_{fn}", header=False, mode='a', index=False)
            #exit()
            #file_names.append(f"{i + 1}_{server.sim_name}_{fn}")
        # sim_loop(max_t, client, rho_fault, rho_reset, fault_start, fault_duration)
        # print(len(client.server.context.result))
        #f.write(client.server.context.result)
        # df = pd.DataFrame(client.server.context.result)
        # df.to_csv(current_fn, index=False)

    #exit()
    for i in range(1,num_servers+1):
        latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std = mean_variance_std_dev(file_names, max_t, num_runs, step_time, mean_t,i)
        plot_results(step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, "discrete_results_multi_"+str(i)+".pdf")
        directory = os.path.dirname("data/sim_data_multi_"+str(i)+".pkl")
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open("data/sim_data_multi_"+str(i)+".pkl", "wb") as f:
            pickle.dump((step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, rho), f)


def mean_variance_std_dev(file_names: List[str], max_t: float, num_runs: int, step_time: int, mean_t: float, sid:int):
    """
    Print the mean value, the variance, and the standard deviation at each stat point in each second
    """
    num_datapoints = math.ceil(max_t / step_time)
    latency_dateset = np.zeros((num_runs, num_datapoints))
    runtime_dateset = np.zeros((num_runs, num_datapoints))
    qlen_dateset = np.zeros((num_runs, num_datapoints))
    run_ind = 0
    for file_name in file_names:
        step_ind = 0
        with open("data/"+file_name, "r") as f:
            row_num = len(pd.read_csv("data/"+file_name))
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # drop the header
                split_line: List[str] = line.split(',')
                #print(split_line)
                server_id = int(split_line[0])
                if server_id==sid:
                    if float(split_line[1]) > (step_ind + 1) * step_time:
                        #latency = float(split_line[2]) * 1
                        latency = float(split_line[2]) * 1
                        runtime = float(split_line[6])
                        qlen = float(split_line[3])
                        latency_dateset[run_ind, step_ind] = latency
                        runtime_dateset[run_ind, step_ind] = runtime
                        qlen_dateset[run_ind, step_ind] = qlen
                        step_ind += 1
                    elif i == row_num - 1 and step_ind < num_datapoints:
                        # latency = float(split_line[2]) * 1
                        latency = float(split_line[2]) * 1
                        runtime = float(split_line[6])
                        qlen = float(split_line[3])
                        latency_dateset[run_ind, step_ind] = latency
                        runtime_dateset[run_ind, step_ind] = runtime
                        qlen_dateset[run_ind, step_ind] = qlen
        run_ind += 1
    latency_ave = [0]
    latency_var = [0]
    latency_std = [0]
    runtime = [0]
    qlen_ave = [0]
    qlen_var = [0]
    qlen_std = [0]
    for step in range(num_datapoints):
        latency_ave.append(np.mean(latency_dateset[:, step]))
        latency_var.append(np.var(latency_dateset[:, step]))
        latency_std.append(np.std(latency_dateset[:, step]))
        runtime.append(np.sum(runtime_dateset[:, step]))
        qlen_ave.append(np.mean(qlen_dateset[:, step]))
        qlen_var.append(np.var(qlen_dateset[:, step]))
        qlen_std.append(np.std(qlen_dateset[:, step]))
    return latency_ave,  latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std


def write_to_file(fn: str, stats_data: List[StatData], stat_fn, first: bool):
    with open(fn, "a") as f:
        if first:
            f.write(stats_data[0].header() + '\n')
        result: StatData = stat_fn(stats_data)
        f.write(result.as_csv() + '\n')


def compute_mean_variance_std_deviation(fn: str, max_t: float, step_time: int, num_runs: int, mean_t: float):
    current_folder = os.getcwd()
    file_names = [file for file in os.listdir(current_folder) if file.endswith(fn)]
    mean_variance_std_dev(file_names, max_t, step_time, num_runs, mean_t)


# Simulation with unimodal exponential service time and timeout
def make_sim_exp(mean_t: float, name: str, apiname: str, rho: float, queue_size: int, retry_queue_size: int, timeout_t: float,
                 max_retries: int, rho_fault: float, rho_reset: float, fault_start: float,
                 fault_duration: float, dist: str, num_servers :int, sim_id: int) -> List[Client]:
    
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


    servers = []
    prev_server = None
    
    for i in range(num_servers,0,-1):
        server_name = f"server_{i}"

        
        client = OpenLoopClientWithTimeout(name, apiname, distribution, rho, job_type, timeout_t, max_retries, rho_fault, rho_reset, fault_start,
                                    fault_duration)
    
        if dist =="exp":
            service_time_distribution = {"request":distribution(1/job_type.mean())}
        elif dist=="wei":
            service_time_distribution = {"request":distribution(1/job_type.mean())}
        
        #downstream = servers[-1] if servers else None
        server = Server(i,server_name, queue_size, 1, service_time_distribution, retry_queue_size, client, downstream_server=prev_server)
        server.set_context(Context(sim_id,i))  #check
        client.server = server
        #clients.append(client)
        servers.append(server)
        prev_server = server

        if i == 1:  # Only the first server gets a client generating jobs
            clients.append(client)
    
    # Reverse the downstream links (Server1 -> Server2 -> ... -> ServerN)
    servers = servers[::-1]
    # for i in range(len(servers) - 1):
    #     servers[i].downstream_server = servers[i + 1]

    #print("inside make_sim_exp ",servers[0].id,"  ")
    #print("  ",servers[1].id)
    
    #print("  ",servers[0].downstream_server.id)
    return servers, clients


# Simulation with bimodal service time, without and with timeout
def make_sim_bimod(mean_t: float, mean_t_2: float, bimod_p: float, rho: float, queue_size: int, retry_queue_size: int,
                   timeout_t: float, max_retries: int) -> List[Client]:
    clients = []
    job_name = "bimod"
    job_type = bimod_job(mean_t, mean_t_2, bimod_p)

    for name, client in [
        ("%s" % job_name, OpenLoopClient(rho, job_type)),
        ("%s_timeout_%d" % (job_name, int(timeout_t)), OpenLoopClientWithTimeout(rho, rho_fault, rho_reset, job_type,
                                                                                 timeout, max_retries, fault_start,
                                                                                 fault_duration))
    ]:
        server = Server(1, name, client, rho, queue_size, retry_queue_size,client)
        client.server = server
        clients.append(client)
    return clients


def run_discrete_experiment(max_t: float, runs: int, mean_t: float, rho: float, queue_size: int, retry_queue_size: int,
                            timeout_t: float, max_retries: int, total_time: float, step_time: int,
                            rho_fault: float, rho_reset: float, fault_start: float, fault_duration: float, 
                            dist: str, num_servers : int):
   
    results_file_name = "exp_results.csv"
    start_time = time.time()
    process = multiprocessing.Process(target=run_sims, args=(max_t, results_file_name, runs, step_time, make_sim_exp,
                                                             mean_t, rho, queue_size, retry_queue_size,
                                                             timeout_t,
                                                             max_retries, rho_fault, rho_reset, fault_start,
                                                             fault_duration, dist, num_servers))
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
