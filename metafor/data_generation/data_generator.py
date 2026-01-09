import argparse
import os
from typing import List
import argparse
import numpy as np
from metafor.simulator.simulate import run_discrete_experiment
from generate_pkl_files import convert_csv_to_pkl

import logging
logging.disable(logging.CRITICAL)

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

    

    # parser = argparse.ArgumentParser()

    # parser.add_argument("--sim_time", help="maximum simulation time for an individual simulation", type=int, default=10000)
    # parser.add_argument("--runs", help="how many times should the simulation be run", default=100, type=int, required=False)
    # parser.add_argument("--qsize", help="maximum size of the arrivals queue", default=100, type=int, required=False)
    # parser.add_argument("--rsize", help="maximum size of the retries queue", default=20, type=int, required=False)
    # parser.add_argument("--genpkl", help="Generate the pkl files from csv", default=False, type=bool, required=False)
    # parser.add_argument("--fault_duration", help="Duration of fault rate injection as a fraction of simulation time", default=0.01*sim_time, type=float, required=False)
    # parser.add_argument("--throttle", help="Apply throttling strategy", default=False, type=bool, required=False)
    # parser.add_argument("--ts", help="Fraction of queue length used to activate throttling", default=1, type=float, required=False)
    # parser.add_argument("--ap", help="Admission probability for throttling", default=1, type=float, required=False)
    # parser.add_argument("--queue_type", help="Select fifo or lifo", default="fifo", type=str, required=False)
    
    
    # parser.add_argument("--verbose", help="Verbosity", default=True, type=bool, required=False)
    # parser.add_argument("--num_servers", help="number of servers", default=2, type=int, required=False)
   
  
    # args = parser.parse_args()

    # sim_time = args.sim_time
    # runs = args.runs
    # main_queue_size = args.qsize
    # retry_queue_size = args.rsize
    # genpkl = args.genpkl
    # fault_duration = args.fault_duration * sim_time
    # fault_start = [sim_time * .45, sim_time]  # start time for fault (last entry is not an actual fault time)
    # throttle = args.throttle
    # ts = args.ts
    # ap = args.ap
    # queue_type = args.queue_type


    # verbose = args.verbose
    # num_servers = args.num_servers

    if verbose==False:
        logging.disable(logging.CRITICAL)

    run_discrete_experiment(sim_time, runs, mean_t, rho, queue_size, 50, timeout_t, max_retries,
                            total_time, step_time, rho_fault, rho_reset, fault_start, fault_duration,  throttle, 
                            ts, ap, queue_type, dist, num_servers)


       
    convert_csv_to_pkl(sim_time, runs, mean_t, rho, step_time, rho_fault, fault_start, fault_duration, num_servers)


