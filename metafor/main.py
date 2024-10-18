import argparse
import os
from typing import List

from analysis.analyzer import Analyzer
from utils.plot_parameters import PlotParameters
from simulator.simulate import run_discrete_experiment
from model.single_server.single_server_ctmc import SingleServerCTMC


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
    parser.add_argument('-total_time', type=int, help='maximum simulation time (in s) for all the simulations',
                        default=1000000)
    parser.add_argument('-sim_time', type=int, help='maximum simulation time (in ms) for an individual simulation',
                        default=6000)
    parser.add_argument('-step_time', type=int, help='step time used for plots', default=500)
    parser.add_argument('-main_queue_size', type=int, help='maximum size of the arrivals queue', default=100)
    parser.add_argument('-retry_queue_size', type=int, help='maximum size of the retries queue', default=20)
    parser.add_argument('-mean', type=float, help='mean of the exponential distribution (in ms)', default=0.1)
    parser.add_argument('-mean2', type=float, help='second mean of the bimodal distribution (in ms)', default=10.0)
    parser.add_argument('-bimod_p', type=float, help='probability of having the second mean in the bimodal '
                                                     'distribution', default=0.001)
    parser.add_argument('-rho', type=float, help='server\'s utilization rate', default=0.8)
    parser.add_argument('-timeouts', nargs='+', help='timeout after which the client retries, if the job is not done '
                                                     '(in ms, can be different for different job types)', default=[3])
    parser.add_argument('-retries', nargs='+', help='how many times should a client retry to send a job if it does '
                                                    'not receive a response before the timeout (can be different for '
                                                    'different job types)', default=[3])
    parser.add_argument('-runs', type=int, help='how many times should the simulation be run', default=100)
    parser.add_argument('-lambda0s', nargs='+', help='job arrival rate(s) determined by the throttling scheme '
                                                     '(can be different for different job types)', default=[8])
    parser.add_argument('-mu0_ps', nargs='+', help='service rate(s) (can be different for different job types)',
                        default=[10])
    parser.add_argument('-alpha', type=float, help='scaling factor', default=0.25)
    parser.add_argument('-thread_pool', type=int, help='number of threads processing requests', default=1)

    parser.add_argument('--discrete', action='store_true')
    parser.add_argument('--no-discrete', dest='discrete', action='store_false')
    parser.set_defaults(discrete=True)

    parser.add_argument('--ctmc', action='store_true')
    parser.add_argument('--no-ctmc', dest='ctmc', action='store_false')
    parser.set_defaults(ctmc=True)

    parser.add_argument('-qlen_max', type=int, help='maximum queue length', default=150)
    parser.add_argument('-qlen_step', type=int, help='step size used for effect of queue length plots', default=10)

    parser.add_argument('-olen_max', type=int, help='maximum orbit length', default=60)
    parser.add_argument('-olen_step', type=int, help='step size used for effect of orbit length plots', default=10)

    parser.add_argument('-retry_max', type=int, help='maximum number of retries', default=10)
    parser.add_argument('-retry_step', type=int, help='step size used for effect of retry number plots', default=1)

    parser.add_argument('-lambda_max', type=int, help='maximum arrival rate', default=12)
    parser.add_argument('-lambda_min', type=int, help='minimum arrival rate', default=8)
    parser.add_argument('-lambda_step', type=int, help='step size used for effect of arrival rate plots', default=.5)

    parser.add_argument('-mu_max', type=int, help='maximum processing rate', default=12)
    parser.add_argument('-mu_min', type=int, help='minimum processing rate', default=8)
    parser.add_argument('-mu_step', type=int, help='step size used for effect of arrival rate plots', default=.5)

    parser.add_argument('-reset_lambda_max', type=float, help='maximum arrival rate after reset', default=10)
    parser.add_argument('-reset_lambda_min', type=float, help='minimum arrival rate after reset', default=6)
    parser.add_argument('-reset_lambda_step', type=float, help='step size used for effect of arrival rate after reset',
                        default=.5)
    parser.add_argument('-timeout_max', type=float, help='maximum timeout after reset', default=15)
    parser.add_argument('-timeout_min', type=float, help='minimum timeout after reset', default=5)
    parser.add_argument('-lambda_fault', type=float, help='the arrival rate associated with the fault scenario',
                        default=20)
    parser.add_argument('-start_time_fault', type=float, help='the time point at which the fault starts', default=200)
    parser.add_argument('-duration_fault', type=float, help='the duration of the fault event', default=200)

    args = parser.parse_args()
    total_time = args.total_time
    main_queue_size = args.main_queue_size
    retry_queue_size = args.retry_queue_size
    mean_t = args.mean
    rho = args.rho
    timeouts = args.timeouts
    retries = args.retries
    runs = args.runs

    lambda0s = args.lambda0s
    mu0_ps = args.mu0_ps
    step_time = args.step_time
    sim_time = args.sim_time
    alpha = args.alpha
    thread_pool = args.thread_pool

    discrete_simulation = args.discrete
    ctmc = args.ctmc

    # values used for plots
    qlen_max = args.qlen_max
    qlen_step = args.qlen_step
    olen_max = args.olen_max
    olen_step = args.olen_step
    retry_max = args.retry_max
    retry_step = args.retry_step
    lambda_min = args.lambda_min
    lambda_max = args.lambda_max
    lambda_step = args.lambda_step
    mu_max = args.mu_max
    mu_min = args.mu_min
    mu_step = args.mu_step
    reset_lambda_max = args.reset_lambda_max
    reset_lambda_min = args.reset_lambda_min
    reset_lambda_step = args.reset_lambda_step
    lambda_fault = args.lambda_fault
    start_time_fault = args.start_time_fault
    duration_fault = args.duration_fault
    timeout_max = args.timeout_max
    timeout_min = args.timeout_min

    delete_results([".csv", ".png"])

    if discrete_simulation:
        print('\nDiscrete event system')
        run_discrete_experiment(sim_time, runs, mean_t, rho, main_queue_size, retry_queue_size, timeouts[0],
                                retries[0], total_time, step_time)

    if ctmc:
        print('\nCTMC')
        plot_params = PlotParameters(step_time, sim_time, qlen_max, qlen_step, olen_max, olen_step, retry_max,
                                     retry_step, lambda_max, lambda_min, lambda_step, mu_max, mu_min, mu_step,
                                     reset_lambda_max, reset_lambda_min, reset_lambda_step, [lambda_fault],
                                     start_time_fault, duration_fault, timeout_max, timeout_min)
        ctmc = SingleServerCTMC(main_queue_size, retry_queue_size, lambda0s, mu0_ps, timeouts, retries, thread_pool,
                                alpha)
        file_name = 'single_server_results.png'
        analyzer = Analyzer(ctmc, file_name)
        analyzer.average_lengths_analysis(plot_params)
        analyzer.latency_analysis(plot_params)
        analyzer.fault_scenario_analysis(plot_params)


if __name__ == '__main__':
    main()
