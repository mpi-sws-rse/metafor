# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List

import numpy as np
import pandas

from metafor.simulator.server import Context, Server
from metafor.simulator.statistics import StatData
from metafor.simulator.client import Client, OpenLoopClient, OpenLoopClientWithTimeout

from utils.plot import plot_results
from metafor.simulator.job import exp_job, bimod_job

import logging
logger = logging.getLogger(__name__)




class Simulator:
    def __init__(self, server: Server, clients: List[Client], sim_id: int):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)

        self.server = server

        self.context = Context(sim_id)
        self.server.set_context(self.context)
        self.clients = clients

        self.reset()

        self.server.print()
        for c in self.clients:
            c.print()
        logger.setLevel(logging.INFO)
    
    def __del__(self):
        self.context.close()

    # to start a simulator from the beginning, reset the state
    def reset(self):
        self.t = 0.0
        # start the simulator by adding each client
        self.q = [(self.t, client.generate, None) for client in self.clients]

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
                heapq.heapify(self.q)   

    def analyze(self):
        self.context.analyze()

# Run a single simulation.
def sim_loop(max_t: float, client: Client, rho_fault, rho_reset, fault_start, fault_duration):
    t = 0.0
    q = [(t, client.generate, None)]
    # This is the core simulation loop. Until we've reached the maximum simulation time, pull the next event off
    #  from a heap of events, fire whichever callback is associated with that event, and add any events it generates
    #  back to the heap.
    while len(q) > 0 and t < max_t:
        # Get the next event. Because `q` is a heap, we can just pop this off the front of the heap.
        (t, call, payload) = heapq.heappop(q)
        # Execute the callback associated with this event
        new_events = call(t, payload)
        # If the callback returned any follow-up events, add them to the event queue `q` and make sure it is still a
        # valid heap
        if new_events is not None:
            q.extend(new_events)
            heapq.heapify(q)


# Run a simulation `run_nums` times, outputting the results to `x_fn`, where x in [1, num_runs].
# One simulation is run for each client in `clients`.
# `max_t` is the maximum time to run the simulation (in ms).
def run_sims(max_t: float, fn: str, num_runs: int, step_time: int, sim_fn, mean_t: float, rho,
             queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration):
    file_names: List[str] = []

    for i in range(num_runs):
        print("Running simulation " + str(i + 1) + " time(s)")
        current_fn = str(i + 1) + '_' + fn
        file_names.append(current_fn)
        clients = sim_fn(mean_t, rho, queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
        with open(current_fn, "w") as f:
            f.write("t,rho,service_time,name,qlen,retries,dropped,runtime\n")
            for client in clients:
                client.server.file = f
                client.server.start_time = time.time()
                sim_loop(max_t, client, rho_fault, rho_reset, fault_start, fault_duration)

    latency_ave, latency_var, latency_std, runtime = mean_variance_std_dev(file_names, max_t, num_runs, step_time, mean_t)
    plot_results(step_time, latency_ave, latency_std, runtime, 'discrete_results.png')


# Print the mean value, the variance, and the standard deviation at each stat point in each second
def mean_variance_std_dev(file_names: List[str], max_t: float, num_runs: int, step_time: int, mean_t: float):
    num_datapoints = math.ceil(max_t / step_time)
    latency_dateset = np.zeros((num_runs, num_datapoints))
    runtime_dateset = np.zeros((num_runs, num_datapoints))
    run_ind = 0
    for file_name in file_names:
        step_ind = 0
        with open(file_name, "r") as f:
            row_num = len(pandas.read_csv(file_name))
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # drop the header
                split_line: List[str] = line.split(',')
                if float(split_line[0]) > (step_ind + 1) * step_time:
                    #latency = float(split_line[2]) * 1
                    latency = float(split_line[4]) * 1
                    runtime = float(split_line[7])
                    latency_dateset[run_ind, step_ind] = latency
                    runtime_dateset[run_ind, step_ind] = runtime
                    step_ind += 1
                elif i == row_num - 1 and step_ind < num_datapoints:
                    # latency = float(split_line[2]) * 1
                    latency = float(split_line[4]) * 1
                    runtime = float(split_line[7])
                    latency_dateset[run_ind, step_ind] = latency
                    runtime_dateset[run_ind, step_ind] = runtime
        run_ind += 1
    latency_ave = [0]
    latency_var = [0]
    latency_std = [0]
    runtime = [0]
    for step in range(num_datapoints):
        latency_ave.append(np.mean(latency_dateset[:, step]))
        latency_var.append(np.var(latency_dateset[:, step]))
        latency_std.append(np.std(latency_dateset[:, step]))
        runtime.append(np.sum(runtime_dateset[:, step]))
    return latency_ave,  latency_var, latency_std, runtime


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
def make_sim_exp(mean_t: float, rho: float, queue_size: int, retry_queue_size: int, timeout_t: float,
                 max_retries: int, rho_fault: float, rho_reset: float, fault_start: float,
                 fault_duration: float) -> List[Client]:
    clients = []
    job_name = "exp"
    job_type = exp_job(mean_t)

    for name, client in [
        ("%s_open_timeout_%d" % (job_name, int(timeout_t)),
         OpenLoopClientWithTimeout(rho, job_type, timeout_t, max_retries, rho_fault, rho_reset, fault_start,
                                   fault_duration))
    ]:
        server = Server(1, name, client, rho, queue_size, retry_queue_size)
        client.server = server
        clients.append(client)
    return clients


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
        server = Server(1, name, client, rho, queue_size, retry_queue_size)
        client.server = server
        clients.append(client)
    return clients


def run_discrete_experiment(max_t: float, runs: int, mean_t: float, rho: float, queue_size: int, retry_queue_size: int,
                            timeout_t: float, max_retries: int, total_time: float, step_time: int,
                            rho_fault: float, rho_reset: float, fault_start: float, fault_duration: float):
    results_file_name = "exp_results.csv"
    start_time = time.time()
    process = multiprocessing.Process(target=run_sims, args=(max_t, results_file_name, runs, step_time, make_sim_exp,
                                                             mean_t, rho, queue_size, retry_queue_size,
                                                             timeout_t,
                                                             max_retries, rho_fault, rho_reset, fault_start,
                                                             fault_duration))
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
