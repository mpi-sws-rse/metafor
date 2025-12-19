# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List

import numpy as np
import pandas as pd

from metafor.simulator.server import  Server
from metafor.simulator.statistics import StatData
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
    llen_dataset = [] # dataset containing latency numbers per run
    dlen_dataset = [] # dataset containing dropped numbers per run
    rlen_dataset = [] # dataset containing retries left numbers per run
    slen_dataset = [] # dataset containing service time numbers per run
    for l in range(num_traj):
        if l == 0:
            traj_start = 0 # corresponding to the trajectory starting from zero
        else:
            traj_start = fault_start[l-1] + fault_duration  # corresponding to the trajectory starting from full
        num_datapoints.append(math.ceil((fault_start[l]-traj_start)/step_time))
        qlen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        olen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        llen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        dlen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        rlen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        slen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
    run_ind = 0
    actual_data_num_seq = [[]*i for i in range(len(file_names))] # list of the "actual" number of datapoints per run
    for file_name in file_names:
        k_overall = 1 # this index is used to track number of datapoints in the run
        traj_idx = 0 # this index keeps track of the trajectory id
        discrete_time_point = 1 * step_time # the exact (discrete) time
        # initially for every run muner of jobs and number of retries are zero
        last_q_val = 0
        last_o_val = 0
        last_l_val = 0
        last_d_val = 0
        last_r_val = 0
        last_s_val = 0

        wait_ind = False # while true, must wait until the end of the fault period
        with open("data/"+file_name, "r") as f:
            #print(file_name)
            row_num = len(pd.read_csv("data/"+file_name))
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
                    if current_cont_time > (fault_start[traj_idx - 1] + fault_duration): # check if the fault is over
                        wait_ind = False #
                if (current_cont_time < max_t) and (current_cont_time > discrete_time_point) and (not wait_ind):
                    # compute number of intermediate discrete time points between the until current_cont_time
                    n_mid = math.floor((current_cont_time-discrete_time_point)/step_time)
                    
                    # for the time steps between prev discrete_time_point and current_cont_time copy the last values
                    for i in range(n_mid):
                        qlen_dataset[traj_idx][run_ind, k_overall] = last_q_val
                        olen_dataset[traj_idx][run_ind, k_overall] = last_o_val
                        llen_dataset[traj_idx][run_ind, k_overall] = last_l_val
                        dlen_dataset[traj_idx][run_ind, k_overall] = last_d_val
                        rlen_dataset[traj_idx][run_ind, k_overall] = last_r_val
                        slen_dataset[traj_idx][run_ind, k_overall] = last_s_val
                        k_overall += 1 #
                    # use data in csv files to update values
                    last_q_val = float(split_line[2])
                    last_o_val = float(split_line[3])
                    last_l_val = float(split_line[1])
                    last_d_val = float(split_line[4])
                    last_r_val = float(split_line[6])
                    last_s_val = float(split_line[7])
                    # update the content of datasets
                    qlen_dataset[traj_idx][run_ind, k_overall] = last_q_val
                    olen_dataset[traj_idx][run_ind, k_overall] = last_o_val
                    llen_dataset[traj_idx][run_ind, k_overall] = last_l_val
                    dlen_dataset[traj_idx][run_ind, k_overall] = last_d_val
                    rlen_dataset[traj_idx][run_ind, k_overall] = last_r_val
                    slen_dataset[traj_idx][run_ind, k_overall] = last_s_val
                    k_overall += 1
                    discrete_time_point = (math.floor(current_cont_time/step_time)+1) * step_time
        actual_data_num_seq[run_ind].append(k_overall) # store number of actual datapoints in the current run
        run_ind += 1 # update the run number

  
    common_data_num = np.min(actual_data_num_seq, axis = 0) # minimum number of data points among all runs
    q_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average number of jobs in the queue
    o_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average number of retried jobs
    l_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average latency
    d_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average number of dropped jobs
    r_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average number of retries left
    s_ave_seq = [[]*num_traj for l in range(num_traj)] # seq of average service time
    pi_emp_seq = [[]*num_traj for l in range(num_traj)] # seq of empirical distributions

    for traj_idx in range(num_traj):
        for step in range(common_data_num[traj_idx]):
            q_step = 0 # initialization for average number of jobs in the queue
            o_step = 0 # initialization for average number of retried jobs
            l_step = 0 
            d_step = 0 
            r_step = 0 
            s_step = 0 
            pi_step = np.zeros(ss_size)  # initialization for empirical distribution
            # compute empirical averages
            for run_ind in range(num_runs):
                q_step += qlen_dataset[traj_idx][run_ind, step] / num_runs
                o_step += olen_dataset[traj_idx][run_ind, step] / num_runs
                l_step += llen_dataset[traj_idx][run_ind, step] / num_runs
                d_step += dlen_dataset[traj_idx][run_ind, step] / num_runs
                r_step += rlen_dataset[traj_idx][run_ind, step] / num_runs
                s_step += slen_dataset[traj_idx][run_ind, step] / num_runs
                pi_diff = np.zeros(ss_size)
                state = min(ss_size - 1,
                            index_composer(qlen_dataset[traj_idx][run_ind, step], olen_dataset[traj_idx][run_ind, step], qsize, osize))
                pi_diff[int(state)] = 1 / num_runs
                pi_step += pi_diff
            q_ave_seq[traj_idx].append(q_step)
            o_ave_seq[traj_idx].append(o_step)
            l_ave_seq[traj_idx].append(l_step)
            d_ave_seq[traj_idx].append(d_step)
            r_ave_seq[traj_idx].append(r_step)
            s_ave_seq[traj_idx].append(s_step)
            pi_emp_seq[traj_idx].append(pi_step)

    return q_ave_seq, o_ave_seq, l_ave_seq, d_ave_seq, r_ave_seq, s_ave_seq, pi_emp_seq



def compute_mean_variance_std_deviation(fn: str, max_t: float, step_time: int, num_runs: int, mean_t: float, rho, rho_fault, fault_start, fault_duration, qsize, osize):
    current_folder = os.getcwd()+"/data/"
    file_names = [file for file in os.listdir(current_folder) if file.endswith(fn)]
    # qsize and osize are fixed...
    q_seq, o_seq, l_seq, d_seq, r_seq, s_seq, pi_seq = mean_variance_std_dev(file_names, max_t, step_time, num_runs, mean_t, rho, rho_fault, fault_start, fault_duration, qsize, osize)
    return q_seq, o_seq, l_seq, d_seq, r_seq, s_seq, pi_seq



def convert_csv_to_pkl(max_t: float, runs: int, mean_t: float, rho: float, step_time: int,
                            rho_fault: float, fault_start: float, fault_duration: float, qsize: int=100, osize: int=30):
    results_file_name = "exp_results.csv"

    q_seq, o_seq, l_seq, d_seq, r_seq, s_seq, pi_seq = compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t, rho,
                                                       rho_fault, fault_start, fault_duration, qsize, osize)

    directory = os.path.dirname("data")
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    # Save
    with open("data/q_seq.pkl", "wb") as f:
        pickle.dump(q_seq, f)
    with open("data/o_seq.pkl", "wb") as f:
        pickle.dump(o_seq, f)
    with open("data/l_seq.pkl", "wb") as f:
        pickle.dump(l_seq, f)
    with open("data/d_seq.pkl", "wb") as f:
        pickle.dump(d_seq, f)
    with open("data/r_seq.pkl", "wb") as f:
        pickle.dump(r_seq, f)
    with open("data/s_seq.pkl", "wb") as f:
        pickle.dump(s_seq, f)
    with open("data/pi_seq.pkl", "wb") as f:
        pickle.dump(pi_seq, f)



def index_composer(n_main_queue, n_retry_queue, qsize, osize):
    """This function converts two given input indices into one universal index in range [0, state_num].
    The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit."""
    main_queue_size = qsize
    # note that n_main_queue can become equal to qsize (strictly greater than qsie - 1)
    total_ind = min(osize - 1, n_retry_queue) * main_queue_size + min(n_main_queue, main_queue_size - 1)
    # total_ind = n_retry_queue * main_queue_size + n_main_queue
    return total_ind


if __name__ == '__main__':
    mean_t = 0.1 # mean of the exponential distribution (in ms) related to processing time
    rho = 9.7/10 # server's utilization rate
    runs = 100 # how many times should the simulation be run
    step_time = .5 # sampling time
    sim_time = 10000 # maximum simulation time for an individual simulation
    rho_fault = rho*10 # utilization rate during a fault
    fault_start = [sim_time * .45, sim_time]  # start time for fault (last entry is not an actual fault time)
    fault_duration = sim_time * .01  # fault duration
    qsize = 100  # maximum size of the arrivals queue
    osize = 30  # bound over the orbit size
    convert_csv_to_pkl(sim_time, runs, mean_t, rho, step_time, rho_fault, fault_start, fault_duration, qsize, osize)
