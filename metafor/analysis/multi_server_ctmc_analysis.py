import copy
import math
import time
import scipy

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs

from model.multi_server.generator_matrix import GeneratorMatrix
from model.multi_server.ctmc import MultiServerCTMC
from utils.plot_parameters import PlotParameters


def hitting_time_approx(ctmc: MultiServerCTMC, Q):
    A = copy.deepcopy(Q)

    def matvec_func(x):
        return A.dot(x)

    def rmatvec_func(x):
        return A.T.dot(x)

    A_op = GeneratorMatrix(
        shape=(ctmc.state_num_prod, ctmc.state_num_prod),
        matvec=matvec_func,
        rmatvec=rmatvec_func,
        dtype=A.dtype,
    )

    eigenvalues_sorted, _ = eigs(A_op, k=2, which="LR")

    mixing_time = abs(math.log2(0.1) / eigenvalues_sorted[1].real)
    return mixing_time


def fault_simulation_data_generator(
    ctmc: MultiServerCTMC,
    pi_q_seq,
    main_queue_ave_len_seq,
    lambda_seq,
    plot_params: PlotParameters,
):
    """To compute the simulation data corresponding to the fixed fault scenario"""
    stable_configs = []
    unstable_configs = []
    metastable_configs = []
    lambda_reset = plot_params.lambda_reset
    fault_time = plot_params.start_time_fault
    reset_time = plot_params.reset_time
    lambda_fault = plot_params.lambda_fault
    step_time = plot_params.step_time

    data_init = []
    row_ind_init = 0
    col_ind_init = 0
    data_fault = []
    row_ind_fault = 0
    col_ind_fault = 0
    data_reset = []
    row_ind_reset = 0
    col_ind_reset = 0

    for lambda_config in plot_params.config_set:
        # Computing the generator matrices and the stationary distributions
        pi_ss, row_ind, col_ind, data, Q_op, Q_op_T = (
            ctmc.compute_stationary_distribution(lambda_config)
        )
        if lambda_config == ctmc.lambdaas:
            row_ind_init, col_ind_init, data_init = row_ind, col_ind, data
            np.save("pi_ss", pi_ss)
        elif lambda_config == lambda_fault:
            row_ind_fault, col_ind_fault, data_fault = row_ind, col_ind, data
        elif lambda_config == lambda_reset:
            row_ind_reset, col_ind_reset, data_reset = row_ind, col_ind, data

        #  approximate hitting time via using spectral gap
        h_su = []
        h_us = []
        for node_id in range(ctmc.server_num):
            start = time.time()
            q_range_u = [ctmc.q_min_list[node_id], ctmc.main_queue_sizes[node_id]]
            o_range_u = [ctmc.o_min_list[node_id], ctmc.retry_queue_sizes[node_id]]
            row_ind_u, col_ind_u, data_ind_u = ctmc.sparse_info_calculator(
                lambda_config, node_id, q_range_u, o_range_u
            )
            Q_u = scipy.sparse.csr_matrix(
                (data_ind_u, (row_ind_u, col_ind_u)),
                shape=(ctmc.state_num_prod, ctmc.state_num_prod),
            )
            h_su.append(hitting_time_approx(ctmc, Q_u))
            print("h_su is", h_su[node_id])
            print("time taken to compute h_su is", time.time() - start)

            start = time.time()
            q_range_s = [0, ctmc.q_max_list[node_id]]
            o_range_s = [0, ctmc.o_max_list[node_id]]
            row_ind_s, col_ind_s, data_ind_s = ctmc.sparse_info_calculator(
                lambda_config, node_id, q_range_s, o_range_s
            )
            Q_s = scipy.sparse.csr_matrix(
                (data_ind_s, (row_ind_s, col_ind_s)),
                shape=(ctmc.state_num_prod, ctmc.state_num_prod),
            )
            h_us.append(hitting_time_approx(ctmc, Q_s))
            print("h_us is", h_us[node_id])
            print("time taken to compute h_us is", time.time() - start)

        h_overall = min(min(h_su), min(h_us))

        if h_overall > 1000:
            clusters = [[], []]
        else:
            clusters = [[]]

        # computing occupancy prob
        cumulative_prob_stable = []
        cumulative_prob_unstable = []
        for node_id in range(ctmc.server_num):
            q_range_s = []
            o_range_s = []
            for server_id in range(ctmc.server_num):
                if server_id == node_id:
                    q_range_s.append([0, ctmc.q_max_list[node_id]])
                    o_range_s.append([0, ctmc.o_max_list[node_id]])
                else:
                    q_range_s.append([0, ctmc.main_queue_sizes[node_id]])
                    o_range_s.append([0, ctmc.retry_queue_sizes[node_id]])
            # compute the probability assigned to stable portion of the state space of the individual servers
            cumulative_prob_stable.append(
                ctmc.cumulative_prob_computer(pi_ss, q_range_s, o_range_s)
            )

            q_range_u = []
            o_range_u = []
            for server_id in range(ctmc.server_num):
                if server_id == node_id:
                    q_range_u.append(
                        [ctmc.q_min_list[server_id], ctmc.main_queue_sizes[server_id]]
                    )
                    o_range_u.append(
                        [ctmc.o_min_list[server_id], ctmc.retry_queue_sizes[server_id]]
                    )
                else:
                    q_range_u.append([0, ctmc.main_queue_sizes[server_id]])
                    o_range_u.append([0, ctmc.retry_queue_sizes[server_id]])
            # compute the probability assigned to stable portion of the state space of the individual servers
            cumulative_prob_unstable.append(
                ctmc.cumulative_prob_computer(pi_ss, q_range_u, o_range_u)
            )
        print("cumulative_prob_stable is", cumulative_prob_stable)
        print("cumulative_prob_unstable is", cumulative_prob_unstable)

        # computing clustering
        start = time.time()
        eigenvalues_Q_sorted, eigenvectors_Q_sorted = eigs(Q_op, k=2, which="LR")
        print(eigenvalues_Q_sorted)
        if lambda_config == plot_params.config_set[0]:
            np.save("eigenvectors_Q_sorted", eigenvectors_Q_sorted)
        print("time taken to compute clustering is", time.time() - start)
        if len(clusters) == 1:
            clusters.append([i for i in range(0, ctmc.state_num_prod)])
        else:
            X = np.real(eigenvectors_Q_sorted[:, 1])
            for i in range(ctmc.state_num_prod):
                if X[i] > 0:
                    clusters[0].append(i)
                else:
                    clusters[1].append(i)

        config_type = ""
        if len(clusters) == 1:
            if (
                min(cumulative_prob_stable) > 0.5
                and max(cumulative_prob_unstable) < 0.05
            ):
                config_type = "stable"
            elif (
                min(cumulative_prob_unstable) > 0.5
                and max(cumulative_prob_stable) < 0.05
            ):
                config_type = "unstable"
        else:
            if (
                min(cumulative_prob_stable) > 0.5
                and max(cumulative_prob_unstable) < 0.05
            ):
                config_type = "stable"
            elif (
                min(cumulative_prob_unstable) > 0.5
                and max(cumulative_prob_stable) < 0.05
            ):
                config_type = "unstable"
            elif (
                min(cumulative_prob_unstable) > 0.33
                and min(cumulative_prob_stable) > 0.33
            ):
                config_type = "metastable"
            else:
                print("the configuration is not classified correctly")
                config_type = "dontcare"

        if config_type == "stable":
            stable_configs.append(lambda_config)
        elif config_type == "unstable":
            unstable_configs.append(lambda_config)
        elif config_type == "metastable":
            metastable_configs.append(lambda_config)
        else:
            print("the configuration is not classified correctly")
            config_type = "dontcare"

        print("config_type is", config_type)
    # SIMULATION
    for t in range(0, plot_params.sim_time, plot_params.step_time):
        if t <= fault_time:
            lambda_seq.append(ctmc.lambdaas[1])
            data = [data_init[i] * step_time for i in range(len(data_init))]
            row_ind = copy.deepcopy(row_ind_init)
            col_ind = copy.deepcopy(col_ind_init)
        elif fault_time <= t <= fault_time + reset_time:
            lambda_seq.append(lambda_fault[1])
            data = [data_fault[i] * step_time for i in range(len(data_fault))]
            row_ind = copy.deepcopy(row_ind_fault)
            col_ind = copy.deepcopy(col_ind_fault)
        else:
            lambda_seq.append(lambda_reset[1])
            data = [data_reset[i] * step_time for i in range(len(data_reset))]
            row_ind = copy.deepcopy(row_ind_reset)
            col_ind = copy.deepcopy(col_ind_reset)
        Q = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(ctmc.state_num_prod, ctmc.state_num_prod)
        )

        def matvec_func(x):
            return Q.T.dot(x)

        def rmatvec_func(x):
            return Q.dot(x)

        Q_op_T = GeneratorMatrix(
            shape=(ctmc.state_num_prod, ctmc.state_num_prod),
            matvec=matvec_func,
            rmatvec=rmatvec_func,
            dtype=Q.dtype,
        )

        start = time.time()
        pi_q_new = scipy.sparse.linalg.expm_multiply(Q_op_T, pi_q_seq[t // step_time])
        print(time.time() - start)
        # collecting new measurements
        pi_q_seq.append(copy.copy(pi_q_new))
        mean_queue_lengths = ctmc.main_queue_size_average(pi_q_new)
        print(mean_queue_lengths)
        main_queue_ave_len_seq.append(np.sum(mean_queue_lengths))
        print(t)
    return [pi_q_seq, main_queue_ave_len_seq, lambda_seq]


def fault_scenario_analysis(
    ctmc: MultiServerCTMC, file_name: str, plot_params: PlotParameters
):
    print("Started the fault scenario analysis")

    pi_q_new = np.zeros(ctmc.state_num_prod)
    pi_q_new[0] = 1  # Initially the queue is empty
    lambda_seq = []
    pi_q_seq = [copy.copy(pi_q_new)]  # Initializing the initial distribution
    main_queue_ave_len_seq = [0]

    fault_simulation_data_generator(
        ctmc, pi_q_seq, main_queue_ave_len_seq, lambda_seq, plot_params
    )

    print("Creating the plots")
    timee = [
        i * plot_params.step_time
        for i in list(range(0, len(main_queue_ave_len_seq) - 1))
    ]
    # Create 4x1 sub plots
    plt.rcParams["figure.figsize"] = [6, 10]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
    ax1.plot(
        timee,
        main_queue_ave_len_seq[0 : len(main_queue_ave_len_seq) - 1],
        color="tab:blue",
    )
    ax1.set_xlabel("Time (ms)", fontsize=16)
    ax1.set_ylabel("Mean queue length", fontsize=16)
    ax1.grid("on")
    ax1.set_xlim(0, max(timee))

    ax3 = plt.subplot(ax[1, 0])  # row 2, col 0
    ax3.plot(timee, lambda_seq, color="tab:purple")
    ax3.set_xlabel("Time (ms)", fontsize=16)
    ax3.set_ylabel("Job arrival rate", fontsize=16)
    ax3.grid("on")
    ax3.set_xlim(0, max(timee))

    plt.savefig("good_policy_output_" + file_name)
    plt.close()
    print("Finished the fault scenario analysis\n")
