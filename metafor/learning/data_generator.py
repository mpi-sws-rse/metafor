import argparse
import os
from typing import List

from simulator.simulate import run_discrete_experiment


def delete_files(folder: str, extension: str):
    for file in os.listdir(folder):
        if file.endswith(extension):
            os.remove(os.path.join(folder, file))


def delete_results(extensions: List[str]):
    current_folder = os.getcwd()
    for extension in extensions:
        delete_files(current_folder, extension)





total_time = 1000000
main_queue_size = 100
retry_queue_size = 20 # only used when learning in the space of prob distributions is desired.
mean_t = 0.1
rho = 9.7/10
timeout_t = 9
max_retries = 3
runs = 3
step_time = .5 # sampling time
sim_time = 1000
rho_fault = 200
rho_reset = rho * 5 / 5
fault_start = [sim_time * .45, sim_time]  # last entry is not an actual fault time
fault_duration = sim_time * .1  # sim_time * .1



def main():
    delete_results([".csv", ".png"])


    run_discrete_experiment(sim_time, runs, mean_t, rho, main_queue_size, retry_queue_size, timeout_t, max_retries,
                            total_time, step_time, rho_fault, rho_reset, fault_start, fault_duration)


if __name__ == '__main__':
    main()