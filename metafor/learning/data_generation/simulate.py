# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List

import numpy as np
import pandas

from Server import Server
from Statistics import StatData
from Client import Client, OpenLoopClient, OpenLoopClientWithTimeout
from Job import exp_job, bimod_job

done: bool = False
import pickle

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

    #q_seq, o_seq = mean_variance_std_dev(file_names, max_t, num_runs, step_time, mean_t, rho, rho_fault, fault_start,
                                   #fault_duration, qsize, osize)
    #return err_seq

# Print the mean value, the variance, and the standard deviation at each stat point in each second
def mean_variance_std_dev(file_names: List[str], max_t: float, num_runs: int, step_time: int,
                          mean_t: float, rho, rho_fault, fault_start, fault_duration, qsize, osize):
    global done
    ss_size = qsize * osize
    num_traj = len(fault_start)  # number of continuous trajectories
    num_datapoints = []
    qlen_dataset = []
    olen_dataset = []
    for l in range(num_traj):
        if l == 0:
            traj_start = 0
        else:
            traj_start = fault_start[l-1] + fault_duration
        num_datapoints.append(math.ceil((fault_start[l]-traj_start)/step_time))
        qlen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        olen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
    run_ind = 0
    actual_data_num_seq = [[]*i for i in range(len(file_names))]
    for file_name in file_names:
        step_ind = 0
        k_overall = 1
        traj_idx = 0
        k_traj = 1
        discrete_time_point = 1 * step_time
        last_q_val = 0
        last_o_val = 0
        wait_ind = False # if true, must wait until the end of the fault period
        can_continue = True # will be set to True w
        with open(file_name, "r") as f:
            row_num = len(pandas.read_csv(file_name))
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # drop the header
                split_line: List[str] = line.split(',')
                current_cont_time = float(split_line[0])
                if current_cont_time > fault_start[traj_idx] and current_cont_time < max_t:
                    actual_data_num_seq[run_ind].append(k_overall)
                    k_traj = 0
                    k_overall = 0
                    traj_idx += 1
                    wait_ind = True
                    discrete_time_point = (math.floor((fault_start[traj_idx - 1] + fault_duration)/step_time)*step_time)
                #fault_start_i = fault_start[traj_idx]  # corresponding to the end of current trajectory
                if  wait_ind:
                    can_continue = current_cont_time > (fault_start[traj_idx - 1] + fault_duration)
                    if can_continue:
                        wait_ind = False
                if (current_cont_time < max_t) and (current_cont_time > discrete_time_point) and can_continue:


                    n_mid = math.floor((current_cont_time-discrete_time_point)/step_time)
                    for i in range(n_mid):
                        qlen_dataset[traj_idx][run_ind, k_traj] = last_q_val
                        olen_dataset[traj_idx][run_ind, k_traj] = last_o_val
                        k_traj += 1
                        k_overall += 1
                    last_q_val = float(split_line[4]) * 1
                    last_o_val = float(split_line[5]) * 1
                    qlen_dataset[traj_idx][run_ind, k_traj] = last_q_val
                    olen_dataset[traj_idx][run_ind, k_traj] = last_o_val
                    k_traj += 1
                    k_overall += 1
                    discrete_time_point = (math.floor(current_cont_time/step_time)+1) * step_time
        actual_data_num_seq[run_ind].append(k_overall)
        run_ind += 1

    step_ind = np.min(actual_data_num_seq, axis = 0)
    q_seq = [[]*num_traj for l in range(num_traj)]
    o_seq = [[]*num_traj for l in range(num_traj)]
    pi_seq = [[]*num_traj for l in range(num_traj)]
    q_seq_stochastic = []
    for traj_idx in range(num_traj):
        for step in range(step_ind[traj_idx]):
            q_step = 0
            o_step = 0
            pi_step = np.zeros(ss_size)
            for run_ind in range(num_runs):
                q_step += qlen_dataset[traj_idx][run_ind, step] / num_runs
                o_step += olen_dataset[traj_idx][run_ind, step] / num_runs
                pi_diff = np.zeros(ss_size)
                state = min(ss_size - 1,
                            index_composer(qlen_dataset[traj_idx][run_ind, step], olen_dataset[traj_idx][run_ind, step], qsize, osize))
                pi_diff[int(state)] = 1 / num_runs
                pi_step += pi_diff
            q_seq[traj_idx].append(q_step)
            o_seq[traj_idx].append(o_step)
            pi_seq[traj_idx].append(pi_step)
        for run_ind in range(num_runs):
            q_seq_stochastic.append(qlen_dataset[traj_idx][run_ind, :])




    return q_seq, o_seq, pi_seq, q_seq_stochastic


def write_to_file(fn: str, stats_data: List[StatData], stat_fn, first: bool):
    with open(fn, "a") as f:
        if first:
            f.write(stats_data[0].header() + '\n')
        result: StatData = stat_fn(stats_data)
        f.write(result.as_csv() + '\n')


def compute_mean_variance_std_deviation(fn: str, max_t: float, step_time: int, num_runs: int, mean_t: float, rho, rho_fault, fault_start, fault_duration):
    current_folder = os.getcwd()
    file_names = [file for file in os.listdir(current_folder) if file.endswith(fn)]
    # qsize and osize are fixed...
    q_seq, o_seq, pi_seq, q_seq_stochastic = mean_variance_std_dev(file_names, max_t, step_time, num_runs, mean_t, rho, rho_fault, fault_start, fault_duration, 100, 20)    # fault_duration, qsize, osize
    return q_seq, o_seq, pi_seq, q_seq_stochastic


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
        if not done:
            q_seq, o_seq, pi_seq, q_seq_stochastic = compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t, rho, rho_fault, fault_start, fault_duration)
    end_time = time.time()
    runtime = end_time - start_time
    print("Running time: " + str(runtime) + " s")
    q_seq, o_seq, pi_seq, q_seq_stochastic = compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t, rho,
                                                       rho_fault, fault_start, fault_duration)
    # Save
    with open("q_seq.pkl", "wb") as f:
        pickle.dump(q_seq, f)
    with open("o_seq.pkl", "wb") as f:
        pickle.dump(o_seq, f)
    with open("pi_seq.pkl", "wb") as f:
        pickle.dump(pi_seq, f)
    with open("q_seq_stochastic.pkl", "wb") as f:
        pickle.dump(q_seq_stochastic, f)

def index_composer(n_main_queue, n_retry_queue, qsize, osize):
    """This function converts two given input indices into one universal index in range [0, state_num].
    The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit."""
    main_queue_size = qsize

    total_ind = n_retry_queue * main_queue_size + n_main_queue
    return total_ind


def index_decomposer(total_ind, qsize, osize):
    """This function converts a given index in range [0, state_num]
    into two indices corresponding to (1) number of jobs in orbit and (2) jobs in the queue."""
    main_queue_size = qsize

    n_retry_queue = total_ind // main_queue_size
    n_main_queue = total_ind % main_queue_size
    return [n_main_queue, n_retry_queue]


def dependent_columns(A, tol=1e-6):
    """
    Return the indices of linearly independent columns of matrix A.

    Parameters:
    - A: 2D numpy array (matrix)
    - tol: tolerance level for determining independence

    Returns:
    - indices: List of indices of the independent columns
    """
    # Perform QR decomposition on the transpose of A (so we're working with columns of A)
    Q, R = np.linalg.qr(A)

    # Find the indices of independent columns based on the diagonal of R
    dependent = np.where(np.abs(np.diag(R)) < tol)[0]

    return dependent.tolist()




