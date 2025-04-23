import argparse
import os
from typing import List

from simulate import run_discrete_experiment


def delete_files(folder: str, extension: str):
    for file in os.listdir(folder):
        if file.endswith(extension):
            os.remove(os.path.join(folder, file))


def delete_results(extensions: List[str]):
    current_folder = os.getcwd()
    for extension in extensions:
        delete_files(current_folder, extension)



parser.add_argument('-total_time', type=float, help='maximum simulation time (in s) for all the simulations',
                        default=1000000)
    parser.add_argument('-sim_time', type=float, help='maximum simulation time (in ms) for an individual simulation',
                        default=1000)
    parser.add_argument('-step_time', type=int, help='step time used for plots', default=.5)
    parser.add_argument('-runs', type=int, help='how many times should the simulation be run', default=3)
    parser.add_argument('-main_queue_size', type=int, help='maximum size of the arrivals queue', default=100)
    parser.add_argument('-retry_queue_size', type=int, help='maximum size of the retries queue', default=20)



total_time = 1000000 # maximum simulation time (in s) for all the simulations
main_queue_size = 100 # maximum size of the arrivals queue
retry_queue_size = 20 # only used when learning in the space of prob distributions is desired.
mean_t = 0.1 # mean of the exponential distribution (in ms) related to processing time
rho = 9.7/10 # server's utilization rate
timeout_t = 9 # timeout after which the client retries, if the job is not done
max_retries = 3 # how many times should a client retry to send a job if it doesn't receive a response before the timeout
runs = 3 # how many times should the simulation be run
step_time = .5 # sampling time
sim_time = 1000 # maximum simulation time for an individual simulation
rho_fault = 200 # utilization rate during a fault
rho_reset = rho * 5 / 5 # utilization rate after removing the fault
fault_start = [sim_time * .45, sim_time]  # start time for fault (last entry is not an actual fault time)
fault_duration = sim_time * .1  # fault duration



def main():
    delete_results([".csv", ".png"])
    
    
    run_discrete_experiment(sim_time, runs, mean_t, rho, main_queue_size, retry_queue_size, timeout_t, max_retries,
                            total_time, step_time, rho_fault, rho_reset, fault_start, fault_duration)


if __name__ == '__main__':
    main()