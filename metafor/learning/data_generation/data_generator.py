import argparse
import os
from typing import List
import argparse
import numpy as np
from metafor.simulator.simulate import run_discrete_experiment

def delete_files(folder: str, extension: str):
    for file in os.listdir(folder):
        if file.endswith(extension):
            os.remove(os.path.join(folder, file))


def delete_results(extensions: List[str]):
    current_folder = os.getcwd()
    for extension in extensions:
        delete_files(current_folder, extension)




def main():
    """ 
    Main function that generates simulation data (.csv files) based on values of different parameters.
    """

    delete_results([".csv", ".png"])

    total_time = 1000000 # maximum simulation time (in s) for all the simulations
    main_queue_size = 100 # maximum size of the arrivals queue
    retry_queue_size = 20 # only used when learning in the space of prob distributions is desired.
    mean_t = 0.1 # mean of the exponential distribution (in ms) related to processing time
    rho = 9.7/10 # server's utilization rate
    timeout_t = 9 # timeout after which the client retries, if the job is not done
    max_retries = 3 # how many times should a client retry to send a job if it doesn't receive a response before the timeout
    runs = 10 # how many times should the simulation be run
    step_time = .5 # sampling time
    sim_time = 1000 # maximum simulation time for an individual simulation
    rho_fault = np.random.uniform(rho,rho*200) # utilization rate during a fault
    rho_reset = rho * 5 / 5 # utilization rate after removing the fault
    fault_start = [sim_time * .45, sim_time]  # start time for fault (last entry is not an actual fault time)
    fault_duration = sim_time * .1  # fault duration


    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_time", help="maximum simulation time for an individual simulation", type=int, default=1000)
    parser.add_argument("--runs", help="how many times should the simulation be run", default=100, type=int, required=False)
    parser.add_argument("--qsize", help="maximum size of the arrivals queue", default=100, type=int, required=False)
    parser.add_argument("--rsize", help="maximum size of the retries queue", default=20, type=int, required=False)
    parser.add_argument("--genpkl", help="Generate the pkl files from csv", default=False, type=bool, required=False)
  
    args = parser.parse_args()

    sim_time = args.sim_time
    runs = args.runs
    main_queue_size = args.qsize
    retry_queue_size = args.rsize
    genpkl = args.genpkl
    run_discrete_experiment(sim_time, runs, mean_t, rho, main_queue_size, retry_queue_size, timeout_t, max_retries,
                            total_time, step_time, rho_fault, rho_reset, fault_start, fault_duration)

    if genpkl:
        # When genpkl is true, .pkl files are also generated from the .csv files
        
        from generate_pkl_files import convert_csv_to_pkl
        convert_csv_to_pkl(sim_time, runs, mean_t, rho, step_time, rho_fault, fault_start, fault_duration)

if __name__ == '__main__':
    main()
