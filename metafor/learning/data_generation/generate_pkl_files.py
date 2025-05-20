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
                          mean_t: float, rho: float, rho_fault: float, fault_start: float, fault_duration: float,
                          qsize: int, osize: int):
    global done
    ss_size = qsize * osize # this is used to create the two-dimensional state space for computing empirical dist pi_seq
    num_traj = len(fault_start)  # number of continuous trajectories (currently there are only two)
    num_datapoints = [] # list containing number of time steps within each continuous trajectory
    qlen_dataset = [] # dataset containing number of jobs in queue per run
    olen_dataset = [] # dataset containing retry numbers per run
    for l in range(num_traj):
        if l == 0:
            traj_start = 0 # corresponding to the trajectory starting from zero
        else:
            traj_start = fault_start[l-1] + fault_duration  # corresponding to the trajectory starting from full
        num_datapoints.append(math.ceil((fault_start[l]-traj_start)/step_time))
        qlen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        olen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
    run_ind = 0
    actual_data_num_seq = [[]*i for i in range(len(file_names))] # list of the "actual" number of datapoints per run
    for file_name in file_names:
        k_overall = 1 # this index is used to track number of datapoints in the run
        traj_idx = 0 # this index keeps track of the trajectory id
        discrete_time_point = 1 * step_time # the exact (discrete) time
        # initially for every run muner of jobs and number of retries are zero
        last_q_val = 0
        last_o_val = 0
        wait_ind = False # while true, must wait until the end of the fault period
        with open(file_name, "r") as f:
            row_num = len(pandas.read_csv(file_name))
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # drop the header
                split_line: List[str] = line.split(',')
                current_cont_time = float(split_line[0])
                # checking if the current traj is over (and simulation isn't over)
                if current_cont_time > fault_start[traj_idx] and current_cont_time < max_t:
                    # since previous traj is over, k_overall contains the actual
                    actual_data_num_seq[run_ind].append(k_overall)
                    k_overall = 0 # reset k_overall for the next traj
                    traj_idx += 1 # update traj index
                    wait_ind = True # activated once the first trajectory was ended
                    # compute the discrete time point corresponding to the current continuous time
                    discrete_time_point = (math.floor((fault_start[traj_idx - 1] + fault_duration)/step_time)*step_time)
                if  wait_ind: # only activated during the fault trigger
                    # check if the fault duration is over and deactivate wait_ind
                    if current_cont_time > (fault_start[traj_idx - 1] + fault_duration)
                        wait_ind = False #
                if (current_cont_time < max_t) and (current_cont_time > discrete_time_point) and (not wait_ind):
                    # compute number of intermediate discrete time points between the until current_cont_time
                    n_mid = math.floor((current_cont_time-discrete_time_point)/step_time)
                    # for the time steps between prev discrete_time_point and current_cont_time copy the last values
                    for i in range(n_mid):
                        qlen_dataset[traj_idx][run_ind, k_overall] = last_q_val
                        olen_dataset[traj_idx][run_ind, k_overall] = last_o_val
                        k_overall += 1 #
                    # use data in csv files to update values
                    last_q_val = float(split_line[4])
                    last_o_val = float(split_line[5])
                    # update the content of datasets
                    qlen_dataset[traj_idx][run_ind, k_overall] = last_q_val
                    olen_dataset[traj_idx][run_ind, k_overall] = last_o_val
                    k_overall += 1
                    discrete_time_point = (math.floor(current_cont_time/step_time)+1) * step_time
        actual_data_num_seq[run_ind].append(k_overall) # store number of actual datapoints in the current run
        run_ind += 1 # update the run number

    common_data_num = np.min(actual_data_num_seq, axis = 0) # minimum number of data points among all runs
    q_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average number of jobs in the queue
    o_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average number of retried jobs
    pi_emp_seq = [[]*num_traj for l in range(num_traj)] # seq of empirical distributions
    for traj_idx in range(num_traj):
        for step in range(common_data_num[traj_idx]):
            q_step = 0 # initialization for average number of jobs in the queue
            o_step = 0 # initialization for average number of retried jobs
            pi_step = np.zeros(ss_size)  # initialization for empirical distribution
            # compute empirical averages
            for run_ind in range(num_runs):
                q_step += qlen_dataset[traj_idx][run_ind, step] / num_runs
                o_step += olen_dataset[traj_idx][run_ind, step] / num_runs
                pi_diff = np.zeros(ss_size)
                state = min(ss_size - 1,
                            index_composer(qlen_dataset[traj_idx][run_ind, step], olen_dataset[traj_idx][run_ind, step], qsize, osize))
                pi_diff[int(state)] = 1 / num_runs
                pi_step += pi_diff
            q_ave_seq[traj_idx].append(q_step)
            o_ave_seq[traj_idx].append(o_step)
            pi_emp_seq[traj_idx].append(pi_step)

    return q_ave_seq, o_ave_seq, pi_emp_seq



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
