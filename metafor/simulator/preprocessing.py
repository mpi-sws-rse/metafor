# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List


import numpy as np
import pandas as pd
from metafor.simulator.statistics import StatData

import logging
logger = logging.getLogger(__name__)






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

