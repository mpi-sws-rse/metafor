import copy
import math
import time
import scipy

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs

from model.multi_server.generator_matrix import GeneratorMatrix
from model.multi_server.multi_server_ctmc import MultiServerCTMC
from utils.plot_parameters import PlotParameters


def hitting_time_approx(ctmc: MultiServerCTMC, Q):
    state_num = ctmc.params.state_num_prod
    A = copy.deepcopy(Q)

    def matvec_func(x):
        return A.dot(x)

    def rmatvec_func(x):
        return A.T.dot(x)

    A_op = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func, dtype=A.dtype)

    eigenvalues_sorted, _ = eigs(A_op, k=2, which='LR')

    mixing_time = abs(math.log2(.1) / eigenvalues_sorted[1].real)
    return mixing_time


def sparse_info_calculator(ctmc: MultiServerCTMC, lambda_list, node_selected, q_range, o_range):
    state_num = ctmc.params.state_num_prod
    server_num = ctmc.params.server_num
    parent_list = ctmc.params.parent_list
    mu0_p = ctmc.params.mu0_ps
    timeout = ctmc.params.timeouts
    max_retries = ctmc.params.max_retries
    main_queue_size = ctmc.params.main_queue_sizes
    retry_queue_size = ctmc.params.retry_queue_sizes
    row_ind = []
    col_ind = []
    data = []
    for total_ind in range(state_num):
        q, o = ctmc.index_decomposer(total_ind)
        absorbing_flg = False
        for node_id in range(server_num):
            if node_id == node_selected:
                if q_range[1] >= q[node_id] >= q_range[0] and o_range[1] >= o[node_id] >= o_range[0]:
                    absorbing_flg = True

        if not absorbing_flg:
            val_sum = 0
            tail_prob_list = ctmc.tail_prob_computer(total_ind)
            q_next = [0 * i for i in range(server_num)]
            o_next = [0 * i for i in range(server_num)]
            # compute the non-synchronized transitions' rates of the generator matrix
            for node_id in range(server_num):
                mu_drop_base = 1 / (timeout[node_id] * (max_retries[node_id] + 1))
                mu_retry_base = max_retries[node_id] / (timeout[node_id] * (max_retries[node_id] + 1))
                # Check which arrival source is active for the selected node_id
                lambdaa = lambda_list[node_id]
                if len(parent_list[node_id]) == 0:  # if there exists only a local source of job arrival
                    lambda_summed = lambdaa
                elif q[parent_list[node_id][0]] == 0:
                    lambda_summed = lambdaa
                else:  # if there exists local and non-local sources of job arrival
                    lambda_summed = lambdaa + mu0_p[parent_list[node_id][0]]
                if q[node_id] == 0:  # queue is empty
                    q_next[:] = q
                    o_next[:] = o
                    # Setting the rates related to job arrivals
                    q_next[node_id] = q[node_id] + 1
                    val = lambda_summed
                    col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                    data.append(val)
                    val_sum += val
                    row_ind.append(total_ind)
                    q_next[:] = q
                    o_next[:] = o
                    # Setting the rates related to abandon and retry
                    if o[node_id] > 0:  # if there is any job in the server's orbit
                        o_next[node_id] = o[node_id] - 1
                        val = o[node_id] * mu_drop_base  # drop rate
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[node_id] = q[node_id] + 1
                        o_next[node_id] = o[node_id] - 1
                        val = o[node_id] * mu_retry_base  # retry rate
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o

                elif q[node_id] == main_queue_size[node_id] - 1:  # queue is full
                    q_next[:] = q
                    o_next[:] = o
                    # setting the rates related to job processing
                    q_next[node_id] = q[node_id] - 1
                    val = mu0_p[node_id]
                    col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                    data.append(val)
                    val_sum += val
                    row_ind.append(total_ind)
                    q_next[:] = q
                    o_next[:] = o
                    # setting the rates related to abandon
                    if o[node_id] > 0:  # if there is any job in the server's orbit
                        o_next[node_id] = o[node_id] - 1
                        val = o[node_id] * mu_drop_base
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                    # setting the rates related to moving to the orbit space
                    if o[node_id] < retry_queue_size[node_id] - 1:  # if orbit isn't full
                        o_next[node_id] = o[node_id] + 1
                        val = lambda_summed * mu_retry_base * tail_prob_list[node_id]
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o

                else:  # queue is neither full nor empty
                    q_next[:] = q
                    o_next[:] = o
                    # Setting the rates related to job arrivals
                    q_next[node_id] = q[node_id] + 1
                    val = lambda_summed
                    col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                    data.append(val)
                    val_sum += val
                    row_ind.append(total_ind)
                    q_next[:] = q
                    o_next[:] = o
                    # setting the rates related to job processing
                    q_next[node_id] = q[node_id] - 1
                    val = mu0_p[node_id]
                    col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                    data.append(val)
                    val_sum += val
                    row_ind.append(total_ind)
                    q_next[:] = q
                    o_next[:] = o
                    # Setting the rates related to abandon and retry
                    if o[node_id] > 0:  # if there is any job in the server's orbit
                        o_next[node_id] = o[node_id] - 1
                        val = o[node_id] * mu_drop_base
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[node_id] = q[node_id] + 1
                        o_next[node_id] = o[node_id] - 1
                        val = o[node_id] * mu_retry_base * (1 - tail_prob_list[node_id])
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                    # setting the rates related to moving to the orbit space
                    if o[node_id] < retry_queue_size[node_id] - 1:  # if orbit isn't full
                        q_next[node_id] = q[node_id] + 1
                        o_next[node_id] = o[node_id] + 1
                        val = lambda_summed * mu_retry_base * tail_prob_list[node_id]
                        col_ind.append(ctmc.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
            val = - val_sum
            col_ind.append(total_ind)
            data.append(val)
            row_ind.append(total_ind)
    return [row_ind, col_ind, data]


def fault_simulation_data_generator(ctmc: MultiServerCTMC, pi_q_seq, main_queue_ave_len_seq, lambda_seq,
                                    plot_params: PlotParameters):
    """To compute the simulation data corresponding to the fixed fault scenario"""
    stable_configs = []
    unstable_configs = []
    metastable_configs = []
    lambda_reset = plot_params.lambda_reset
    lambda_init = ctmc.params.lambda_init
    fault_time = plot_params.start_time_fault
    reset_time = plot_params.reset_time
    lambda_fault = plot_params.lambda_fault
    step_time = plot_params.step_time
    main_queue_size = ctmc.params.main_queue_sizes
    retry_queue_size = ctmc.params.retry_queue_sizes
    state_num = ctmc.params.state_num_prod
    server_num = ctmc.params.server_num
    q_min_list = ctmc.params.q_min_list
    q_max_list = ctmc.params.q_max_list
    o_min_list = ctmc.params.o_min_list
    o_max_list = ctmc.params.o_max_list

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
        # Computing the generator matrices
        start1 = time.time()
        row_ind, col_ind, data = sparse_info_calculator(ctmc, lambda_config, -1, [0, 0], [0, 0])
        if lambda_config == lambda_init:
            row_ind_init, col_ind_init, data_init = row_ind, col_ind, data
        elif lambda_config == lambda_fault:
            row_ind_fault, col_ind_fault, data_fault = row_ind, col_ind, data
        elif lambda_config == lambda_reset:
            row_ind_reset, col_ind_reset, data_reset = row_ind, col_ind, data

        Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num, state_num))

        def matvec_func(x):
            return Q.T.dot(x)

        def rmatvec_func(x):
            return Q.dot(x)

        Q_op = GeneratorMatrix(shape=(state_num, state_num), matvec=rmatvec_func, rmatvec=matvec_func,
                               dtype=Q.dtype)
        Q_op_T = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func,
                                 dtype=Q.dtype)

        rt1 = time.time() - start1
        print(rt1)

        #  approximate hitting time via using spectral gap
        h_su = []
        h_us = []
        for node_id in range(server_num):
            start = time.time()
            q_range_u = [q_min_list[node_id], main_queue_size[node_id]]
            o_range_u = [o_min_list[node_id], retry_queue_size[node_id]]
            row_ind_u, col_ind_u, data_ind_u = sparse_info_calculator(ctmc, lambda_config, node_id, q_range_u,
                                                                      o_range_u)
            Q_u = scipy.sparse.csr_matrix((data_ind_u, (row_ind_u, col_ind_u)), shape=(state_num, state_num))
            h_su.append(hitting_time_approx(ctmc, Q_u))
            print("h_su is", h_su[node_id])
            print("time taken to compute h_su is", time.time() - start)

            start = time.time()
            q_range_s = [0, q_max_list[node_id]]
            o_range_s = [0, o_max_list[node_id]]
            row_ind_s, col_ind_s, data_ind_s = sparse_info_calculator(ctmc, lambda_config, node_id, q_range_s,
                                                                      o_range_s)
            Q_s = scipy.sparse.csr_matrix((data_ind_s, (row_ind_s, col_ind_s)), shape=(state_num, state_num))
            h_us.append(hitting_time_approx(ctmc, Q_s))
            print("h_us is", h_us[node_id])
            print("time taken to compute h_us is", time.time() - start)

        h_overall = min(min(h_su), min(h_us))

        if h_overall > 1000:
            clusters = [[], []]
        else:
            clusters = [[]]

        # computing stationary distribution
        start = time.time()
        _, eigenvectors = eigs(Q_op_T, k=1, which='SM')
        pi_ss = np.real(eigenvectors) / np.linalg.norm(np.real(eigenvectors), ord=1)
        if pi_ss[0] < -.00000001:
            pi_ss = -pi_ss
        if lambda_config == plot_params.config_set[0]:
            np.save("pi_ss", pi_ss)
        print("time taken to compute pi_ss is", time.time() - start)

        # computing occupancy prob
        cumulative_prob_stable = []
        cumulative_prob_unstable = []
        for node_id in range(server_num):
            q_range_s = []
            o_range_s = []
            for server_id in range(server_num):
                if server_id == node_id:
                    q_range_s.append([0, q_max_list[node_id]])
                    o_range_s.append([0, o_max_list[node_id]])
                else:
                    q_range_s.append([0, main_queue_size[node_id]])
                    o_range_s.append([0, retry_queue_size[node_id]])
            # compute the probability assigned to stable portion of the state space of the individual servers
            cumulative_prob_stable.append(ctmc.cumulative_prob_computer(pi_ss, q_range_s, o_range_s))

            q_range_u = []
            o_range_u = []
            for server_id in range(server_num):
                if server_id == node_id:
                    q_range_u.append([q_min_list[server_id], main_queue_size[server_id]])
                    o_range_u.append([o_min_list[server_id], retry_queue_size[server_id]])
                else:
                    q_range_u.append([0, main_queue_size[server_id]])
                    o_range_u.append([0, retry_queue_size[server_id]])
            # compute the probability assigned to stable portion of the state space of the individual servers
            cumulative_prob_unstable.append(ctmc.cumulative_prob_computer(pi_ss, q_range_u, o_range_u))
        print("cumulative_prob_stable is", cumulative_prob_stable)
        print("cumulative_prob_unstable is", cumulative_prob_unstable)

        # computing clustering
        start = time.time()
        eigenvalues_Q_sorted, eigenvectors_Q_sorted = eigs(Q_op, k=2, which='LR')
        print(eigenvalues_Q_sorted)
        if lambda_config == plot_params.config_set[0]:
            np.save("eigenvectors_Q_sorted", eigenvectors_Q_sorted)
        print("time taken to compute clustering is", time.time() - start)
        if len(clusters) == 1:
            clusters.append([i for i in range(0, state_num)])
        else:
            X = np.real(eigenvectors_Q_sorted[:, 1])
            for i in range(state_num):
                if X[i] > 0:
                    clusters[0].append(i)
                else:
                    clusters[1].append(i)

        config_type = ""
        if len(clusters) == 1:
            if min(cumulative_prob_stable) > .5 and max(cumulative_prob_unstable) < .05:
                config_type = "stable"
            elif min(cumulative_prob_unstable) > .5 and max(cumulative_prob_stable) < .05:
                config_type = "unstable"
        else:
            if min(cumulative_prob_stable) > .5 and max(cumulative_prob_unstable) < .05:
                config_type = "stable"
            elif min(cumulative_prob_unstable) > .5 and max(cumulative_prob_stable) < .05:
                config_type = "unstable"
            elif min(cumulative_prob_unstable) > .33 and min(cumulative_prob_stable) > .33:
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
            lambda_seq.append(lambda_init[1])
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
        Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num, state_num))

        def matvec_func(x):
            return Q.T.dot(x)

        def rmatvec_func(x):
            return Q.dot(x)

        Q_op_T = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func,
                                 dtype=Q.dtype)

        start = time.time()
        pi_q_new = scipy.sparse.linalg.expm_multiply(Q_op_T, pi_q_seq[t // step_time])
        print(time.time() - start)
        # collecting new measurements
        pi_q_seq.append(copy.copy(pi_q_new))
        mean_queue_lengths = ctmc.main_queue_average_size(pi_q_new)
        print(mean_queue_lengths)
        main_queue_ave_len_seq.append(np.sum(mean_queue_lengths))
        print(t)
    return [pi_q_seq, main_queue_ave_len_seq, lambda_seq]


def fault_scenario_analysis(ctmc: MultiServerCTMC, file_name: str, plot_params: PlotParameters):
    print('Started the fault scenario analysis')

    pi_q_new = np.zeros(ctmc.params.state_num_prod)
    pi_q_new[0] = 1  # Initially the queue is empty
    lambda_seq = []
    pi_q_seq = [copy.copy(pi_q_new)]  # Initializing the initial distribution
    main_queue_ave_len_seq = [0]

    fault_simulation_data_generator(ctmc, pi_q_seq, main_queue_ave_len_seq, lambda_seq, plot_params)

    print('Creating the plots')
    timee = [i * plot_params.step_time for i in list(range(0, len(main_queue_ave_len_seq) - 1))]
    # Create 4x1 sub plots
    plt.rcParams["figure.figsize"] = [6, 10]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
    ax1.plot(timee, main_queue_ave_len_seq[0: len(main_queue_ave_len_seq) - 1], color='tab:blue')
    ax1.set_xlabel('Time (ms)', fontsize=16)
    ax1.set_ylabel('Mean queue length', fontsize=16)
    ax1.grid('on')
    ax1.set_xlim(0, max(timee))

    ax3 = plt.subplot(ax[1, 0])  # row 2, col 0
    ax3.plot(timee, lambda_seq, color='tab:purple')
    ax3.set_xlabel('Time (ms)', fontsize=16)
    ax3.set_ylabel('Job arrival rate', fontsize=16)
    ax3.grid('on')
    ax3.set_xlim(0, max(timee))

    plt.savefig("good_policy_output_" + file_name)
    plt.close()
    print('Finished the fault scenario analysis\n')
