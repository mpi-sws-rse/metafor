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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-total_time', type=float, help='maximum simulation time (in s) for all the simulations',
                        default=1000000)
    parser.add_argument('-sim_time', type=float, help='maximum simulation time (in ms) for an individual simulation',
                        default=1000)
    parser.add_argument('-step_time', type=int, help='step time used for plots', default=.5)
    parser.add_argument('-runs', type=int, help='how many times should the simulation be run', default=100)
    parser.add_argument('-main_queue_size', type=int, help='maximum size of the arrivals queue', default=100)
    parser.add_argument('-retry_queue_size', type=int, help='maximum size of the retries queue', default=20)
    parser.add_argument('-mean', type=float, help='mean of the exponential distribution (in ms)', default=1/10)
    parser.add_argument('-mean2', type=float, help='second mean of the bimodal distribution (in ms)', default=10.0)
    parser.add_argument('-bimod_p', type=float, help='probability of having the second mean in the bimodal '
                                                     'distribution', default=0.001)
    parser.add_argument('-rho', type=float, help='server\'s utilization rate', default=9.7/10)
    parser.add_argument('-timeout', type=float, help='timeout after which the client retries, if the job is not done '
                                                     '(in ms)', default=9)
    parser.add_argument('-max_retries', type=int, help='how many times should a client retry to send a job if it does '
                                                       'not receive a response before the timeout', default=3)
    parser.add_argument('-lambda_max', type=int, help='job arrival rate determined by the throttling scheme', default=9.5)
    parser.add_argument('-mu0_p', type=int, help='initial service rate', default=10)
    parser.add_argument('-alpha', type=float, help='scaling factor', default=1)

    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--no-discrete', dest='discrete', action='store_false')
    parser.set_defaults(discrete=True)


    args = parser.parse_args()
    total_time = args.total_time
    main_queue_size = args.main_queue_size
    retry_queue_size = args.retry_queue_size
    mean_t = args.mean
    rho = args.rho
    timeout_t = args.timeout
    max_retries = args.max_retries
    runs = args.runs

    lambda_max = args.lambda_max
    mu0_p = args.mu0_p
    step_time = args.step_time
    sim_time = args.sim_time
    alpha = args.alpha
    discrete_simulation = args.discrete

    delete_results([".csv", ".png"])

    rho_fault = 200
    rho_reset = rho * 5/5
    fault_start = [sim_time * .45, sim_time]  # last entry is not an actual fault time
    fault_duration = sim_time * .1 #sim_time * .1
    if discrete_simulation:
        print('\nDiscrete event system')
        err_seq = run_discrete_experiment(sim_time, runs, mean_t, rho, main_queue_size, retry_queue_size, timeout_t, max_retries,
                                total_time, step_time, rho_fault, rho_reset, fault_start, fault_duration, 100, 20)


if __name__ == '__main__':
    main()