# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List

import numpy as np
import pandas

from metafor.simulator.server import  Server
from Statistics import StatData
from metafor.simulator.client import Client, OpenLoopClient, OpenLoopClientWithTimeout
from metafor.simulator.job import exp_job, bimod_job, ExponentialDistribution

done: bool = False
import pickle

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

    return q_seq, o_seq, pi_seq



def compute_mean_variance_std_deviation(fn: str, max_t: float, step_time: int, num_runs: int, mean_t: float, rho, rho_fault, fault_start, fault_duration):
    current_folder = os.getcwd()
    file_names = [file for file in os.listdir(current_folder) if file.endswith(fn)]
    # qsize and osize are fixed...
    q_seq, o_seq, pi_seq = mean_variance_std_dev(file_names, max_t, step_time, num_runs, mean_t, rho, rho_fault, fault_start, fault_duration, 100, 20)    # fault_duration, qsize, osize
    return q_seq, o_seq, pi_seq



def convert_csv_to_pkl(max_t: float, runs: int, mean_t: float, rho: float, step_time: int,
                            rho_fault: float, fault_start: float, fault_duration: float):
    results_file_name = "exp_results.csv"

    q_seq, o_seq, pi_seq = compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t, rho,
                                                       rho_fault, fault_start, fault_duration)
    # Save
    with open("q_seq.pkl", "wb") as f:
        pickle.dump(q_seq, f)
    with open("o_seq.pkl", "wb") as f:
        pickle.dump(o_seq, f)
    with open("pi_seq.pkl", "wb") as f:
        pickle.dump(pi_seq, f)



def index_composer(n_main_queue, n_retry_queue, qsize, osize):
    """This function converts two given input indices into one universal index in range [0, state_num].
    The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit."""
    main_queue_size = qsize

    total_ind = n_retry_queue * main_queue_size + n_main_queue
    return total_ind
