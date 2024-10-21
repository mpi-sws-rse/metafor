import argparse

from analysis.analyzer import Analyzer
from utils.plot_parameters import PlotParameters
from model.multi_server.ctmc import MultiServerCTMC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sim_time",
        type=int,
        help="maximum simulation time (in ms) for an individual simulation",
        default=500,
    )
    parser.add_argument(
        "-step_time", type=int, help="step time used for plots", default=50
    )
    parser.add_argument(
        "-start_time_fault",
        type=float,
        help="The first time point (in ms) at which fault occurs",
        default=150,
    )
    parser.add_argument(
        "-reset_time",
        type=float,
        help="The time duration (in ms) between occurrence and diagnosis of fault",
        default=150,
    )
    parser.add_argument(
        "-lambda0s",
        nargs="+",
        help="job arrival rate(s) determined by the throttling scheme ",
        default=[8, 0.5],
    )
    parser.add_argument(
        "-lambda_fault",
        nargs="+",
        help="job arrival rate during the fault scenario (may be determined by the throttling scheme)",
        default=[8, 4.5],
    )
    parser.add_argument(
        "-lambda_reset",
        nargs="+",
        help="job arrival rate after eliminating fault",
        default=[8, 0.25],
    )
    parser.add_argument(
        "-mu0_p", nargs="+", help="initial service rate", default=[10, 15]
    )
    parser.add_argument(
        "-config_list",
        nargs="+",
        help="the set of configs (currently only arrival rates)",
        default=[[8, 0.5], [8, 4.5], [8, 0.25]],
    )

    parser.add_argument(
        "-main_queue_sizes",
        nargs="+",
        help="maximum sizes of the arrivals queues",
        default=[10, 10],
    )
    parser.add_argument(
        "-retry_queue_sizes",
        nargs="+",
        help="maximum sizes of the retries queues",
        default=[2, 2],
    )
    parser.add_argument(
        "-timeouts",
        nargs="+",
        help="timeouts after which the clients retry, if the jobs are not done "
        "(in ms)",
        default=[10, 10],
    )
    parser.add_argument(
        "-max_retries",
        nargs="+",
        help="how many times should a client retry to send a job if it does "
        "not receive a response before the timeout",
        default=[3, 3],
    )
    parser.add_argument(
        "-thread_pools",
        nargs="+",
        help="number of threads per server processing requests",
        default=[1, 1],
    )
    parser.add_argument(
        "-parent_list",
        nargs="+",
        help="The entry i denotes the list of parent servers for the i-th server",
        default=[[], [0], [1]],
    )
    parser.add_argument(
        "-sub_tree_list",
        nargs="+",
        help="The entry i denotes the list of servers that belong to the sub-tree of the i-th server "
        "(the i-th server is included in its own sub-tree)",
        default=[[0, 1], [1]],
    )
    parser.add_argument("-server_num", type=int, help="number of servers", default=2)
    parser.add_argument(
        "-q_min_list",
        nargs="+",
        help="list of minimum number of requests associated to high mode",
        default=[
            int(parser.parse_args().main_queue_sizes[i] * 0.9)
            for i in range(parser.parse_args().server_num)
        ],
    )
    parser.add_argument(
        "-o_min_list",
        nargs="+",
        help="list of minimum number of retrials associated to high mode",
        default=[
            parser.parse_args().retry_queue_sizes[i] // 2
            for i in range(parser.parse_args().server_num)
        ],
    )
    parser.add_argument(
        "-q_max_list",
        nargs="+",
        help="list of maximum number of requests associated to low mode",
        default=[
            int(parser.parse_args().main_queue_sizes[i] * 0.1)
            for i in range(parser.parse_args().server_num)
        ],
    )
    parser.add_argument(
        "-o_max_list",
        nargs="+",
        help="list of maximum number of retrials associated to low mode",
        default=[2 for _ in range(parser.parse_args().server_num)],
    )

    args = parser.parse_args()
    sim_time = args.sim_time
    step_time = args.step_time
    start_time_fault = args.start_time_fault
    reset_time = args.reset_time
    lambda0s = args.lambda0s
    lambda_fault = args.lambda_fault
    lambda_reset = args.lambda_reset
    config_set = args.config_list
    main_queue_sizes = args.main_queue_sizes
    retry_queue_sizes = args.retry_queue_sizes
    timeouts = args.timeouts
    max_retries = args.max_retries
    thread_pools = args.thread_pools
    mu0_p = args.mu0_p
    sub_tree_list = args.sub_tree_list
    parent_list = args.parent_list
    server_num = args.server_num
    q_min_list = args.q_min_list
    q_max_list = args.q_max_list
    o_min_list = args.o_min_list
    o_max_list = args.o_max_list

    print("\nCTMC")
    plot_params = PlotParameters(
        step_time,
        sim_time,
        lambda_fault=lambda_fault,
        start_time_fault=start_time_fault,
        lambda_reset=lambda_reset,
        reset_time=reset_time,
        config_set=config_set,
    )
    ctmc = MultiServerCTMC(
        server_num,
        main_queue_sizes,
        retry_queue_sizes,
        lambda0s,
        mu0_p,
        timeouts,
        max_retries,
        thread_pools,
        parent_list,
        sub_tree_list,
        q_min_list,
        q_max_list,
        o_min_list,
        o_max_list,
    )
    print(ctmc.get_stationary_distribution())
    file_name = "multi_server_results.png"
    analyzer = Analyzer(ctmc, file_name)
    analyzer.fault_scenario_analysis(plot_params)


if __name__ == "__main__":
    main()
