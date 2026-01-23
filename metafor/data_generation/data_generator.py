import argparse
import os
from typing import List
import argparse
import numpy as np
from metafor.simulator.simulate import run_discrete_experiment
from metafor.data_generation.generate_pkl_files import convert_csv_to_pkl


def delete_files(folder: str, extension: str):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for file in os.listdir(folder):
        if file.endswith(extension):
            os.remove(os.path.join(folder, file))


def delete_results(extensions: List[str]):
    current_folder = os.getcwd()+"/data"

    for extension in extensions:
        delete_files(current_folder, extension)




def data_generation(
        sim_time, 
        runs, 
        mean_t, 
        rho, 
        queue_size,  
        timeout_t, 
        max_retries,
        total_time, 
        step_time, 
        rho_fault, 
        rho_reset, 
        fault_start, 
        fault_duration,  
        throttle=False, 
        ts=0.9, 
        ap=0.5, 
        queue_type="fifo", 
        dist="exp", 
        num_servers=1,
        verbose=True,
        
):
    """ 
    Main function that generates simulation data (.csv files) based on values of different parameters.
    """

    delete_results([".csv", ".png"])


    run_discrete_experiment(sim_time, runs, mean_t, rho, queue_size, timeout_t, max_retries,
                            total_time, step_time, rho_fault, rho_reset, fault_start, fault_duration,  throttle, 
                            ts, ap, queue_type, dist, num_servers)


       
    convert_csv_to_pkl(sim_time, runs, mean_t, rho, step_time, rho_fault, fault_start, fault_duration, num_servers)


