# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List

import numpy as np
import pandas

from simulator.server import Server
from simulator.statistics import StatData
from simulator.client import Client, OpenLoopClient, OpenLoopClientWithTimeout
from simulator.job import exp_job, bimod_job
from utils.plot import plot_results

done: bool = False


# Run a single simulation.
def sim_loop(max_t: float, client: Client):
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
def run_sims(max_t: float, fn: str, num_runs: int, step_time: int, sim_fn, *args):
    file_names: List[str] = []

    for i in range(num_runs):
        print("Running simulation " + str(i + 1) + " time(s)")
        current_fn = str(i + 1) + "_" + fn
        file_names.append(current_fn)
        clients = sim_fn(*args)
        with open(current_fn, "w") as f:
            f.write("t,rho,service_time,name,qlen,retries,dropped,runtime\n")
            for client in clients:
                client.server.file = f
                client.server.start_time = time.time()
                sim_loop(max_t, client)

    queue_ave_len, queue_var_len, queue_std_len, runtime = mean_variance_std_dev(
        file_names, max_t, num_runs, step_time
    )
    plot_results(
        step_time,
        queue_ave_len,
        queue_var_len,
        queue_std_len,
        runtime,
        "discrete_results.png",
    )


# Print the mean value, the variance, and the standard deviation at each stat point in each second
def mean_variance_std_dev(
    file_names: List[str], max_t: float, num_runs: int, step_time: int
):
    global done
    num_datapoints = math.ceil(max_t / step_time)
    qlen_dateset = np.zeros((num_runs, num_datapoints))
    runtime_dateset = np.zeros((num_runs, num_datapoints))
    run_ind = 0
    for file_name in file_names:
        step_ind = 0
        with open(file_name, "r") as f:
            row_num = len(pandas.read_csv(file_name))
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # drop the header
                split_line: List[str] = line.split(",")
                if float(split_line[0]) > (step_ind + 1) * step_time:
                    qlen = int(split_line[4])
                    runtime = float(split_line[7])
                    qlen_dateset[run_ind, step_ind] = qlen
                    runtime_dateset[run_ind, step_ind] = runtime
                    step_ind += 1
                elif i == row_num - 1 and step_ind < num_datapoints:
                    qlen = int(split_line[4])
                    runtime = float(split_line[7])
                    qlen_dateset[run_ind, step_ind] = qlen
                    runtime_dateset[run_ind, step_ind] = runtime
        run_ind += 1
    done = True
    main_queue_ave_len = [0]
    main_queue_var_len = [0]
    main_queue_std_len = [0]
    runtime = [0]
    for step in range(num_datapoints):
        main_queue_ave_len.append(np.mean(qlen_dateset[:, step]))
        main_queue_var_len.append(np.var(qlen_dateset[:, step]))
        main_queue_std_len.append(np.std(qlen_dateset[:, step]))
        runtime.append(np.sum(runtime_dateset[:, step]))
    return main_queue_ave_len, main_queue_var_len, main_queue_std_len, runtime


def write_to_file(fn: str, stats_data: List[StatData], stat_fn, first: bool):
    with open(fn, "a") as f:
        if first:
            f.write(stats_data[0].header() + "\n")
        result: StatData = stat_fn(stats_data)
        f.write(result.as_csv() + "\n")


def compute_mean_variance_std_deviation(
    fn: str, max_t: float, step_time: int, num_runs: int
):
    current_folder = os.getcwd()
    file_names = [file for file in os.listdir(current_folder) if file.endswith(fn)]
    mean_variance_std_dev(file_names, max_t, step_time, num_runs)


# Simulation with unimodal exponential service time and timeout
def make_sim_exp(
    mean_t: float,
    rho: float,
    queue_size: int,
    retry_queue_size: int,
    timeout_t: float,
    max_retries: int,
) -> List[Client]:
    clients = []
    job_name = "exp"
    job_type = exp_job(mean_t)

    for name, client in [
        (
            "%s_open_timeout_%d" % (job_name, int(timeout_t)),
            OpenLoopClientWithTimeout(rho, job_type, timeout_t, max_retries),
        )
    ]:
        server = Server(1, name, client, rho, queue_size, retry_queue_size)
        client.server = server
        clients.append(client)
    return clients


# Simulation with bimodal service time, without and with timeout
def make_sim_bimod(
    mean_t: float,
    mean_t_2: float,
    bimod_p: float,
    rho: float,
    queue_size: int,
    retry_queue_size: int,
    timeout_t: float,
    max_retries: int,
) -> List[Client]:
    clients = []
    job_name = "bimod"
    job_type = bimod_job(mean_t, mean_t_2, bimod_p)

    for name, client in [
        ("%s" % job_name, OpenLoopClient(rho, job_type)),
        (
            "%s_timeout_%d" % (job_name, int(timeout_t)),
            OpenLoopClientWithTimeout(rho, job_type, timeout_t, max_retries),
        ),
    ]:
        server = Server(1, name, client, rho, queue_size, retry_queue_size)
        client.server = server
        clients.append(client)
    return clients


def run_discrete_experiment(
    max_t: float,
    runs: int,
    mean_t: float,
    rho: float,
    queue_size: int,
    retry_queue_size: int,
    timeout_t: float,
    max_retries: int,
    total_time: float,
    step_time: int,
):
    results_file_name = "exp_results.csv"
    start_time = time.time()
    process = multiprocessing.Process(
        target=run_sims,
        args=(
            max_t,
            results_file_name,
            runs,
            step_time,
            make_sim_exp,
            mean_t,
            rho,
            queue_size,
            retry_queue_size,
            timeout_t,
            max_retries,
        ),
    )
    process.start()
    process.join(total_time)
    if process.is_alive():
        print("Max simulation time reached, stopped the simulation")
        process.kill()
        # check if the mean, variance, and standard deviation have been computed; if not, compute them
        if not done:
            compute_mean_variance_std_deviation(
                results_file_name, max_t, runs, step_time
            )
    end_time = time.time()
    runtime = end_time - start_time
    print("Running time: " + str(runtime) + " s")
