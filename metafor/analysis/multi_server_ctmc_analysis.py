import copy
import math
import time
import scipy

import numpy as np
from matplotlib import pyplot as plt
from scipy.sparse.linalg import eigs

from model.multi_server.generator_matrix import GeneratorMatrix
from model.multi_server.ctmc import MultiServerCTMC

def analysis(ctmc: MultiServerCTMC,
             step_time,
             sim_time,
             lambda_init,
             lambda_fault,
             fault_time,
             lambda_reset,
             reset_time
             ):
    # analyze mixing time variations against queue size changes
    """qsize_seq, mixing_time_q_seq = ctmc.mixing_time_vs_qsize_data_generator(lambda_init,
                                                                          ctmc.mu0_ps, ctmc.timeouts,
                                                                          ctmc.retry_queue_sizes, 2, 10+1, 1)
    ctmc.plot_mixing_time(1, qsize_seq, mixing_time_q_seq, "Queue length", "Mixing time", "mix_time_vs_qsize", 'blue')

    # analyze mixing time variations against orbit size changes
    osize_seq, mixing_time_o_seq = ctmc.mixing_time_vs_osize_data_generator(lambda_init,
                                                                          ctmc.mu0_ps, ctmc.timeouts,
                                                                          ctmc.main_queue_sizes, 1, 5+1, 1)
    ctmc.plot_mixing_time(1, osize_seq, mixing_time_o_seq, "Orbit length", "Mixing time", "mix_time_vs_osize", 'red')

    # analyze mixing time variations against processing rate changes
    mu_seq, mixing_time_mu_seq = ctmc.mixing_time_vs_mu_data_generator(lambda_init, ctmc.timeouts,
                                                                            ctmc.main_queue_sizes, ctmc.retry_queue_sizes,
                                                                          1, 5+1, 2)
    ctmc.plot_mixing_time(2, mu_seq, mixing_time_mu_seq, "Processing rate", "Mixing time", "mix_time_vs_mu", 'green')

    # analyze mixing time variations against timeout changes
    mu_seq, mixing_time_mu_seq = ctmc.mixing_time_vs_to_data_generator(lambda_init, ctmc.mu0_ps,
                                                                       ctmc.main_queue_sizes, ctmc.retry_queue_sizes,
                                                                       1, 9+1, 2)
    ctmc.plot_mixing_time(2, mu_seq, mixing_time_mu_seq, "Timeout", "Mixing time", "mix_time_vs_to", 'brown')"""


    # analyze the fault scenario
    # the following is tailored to the specific AWS example!
    # initialize prob distribution assuming that
    pi_q_new = np.zeros(ctmc.state_num_prod)
    pi_q_new[ctmc._index_composer([0, min(ctmc.thread_pools[0],ctmc.main_queue_sizes[0])], [0, 0],
                                 ctmc.main_queue_sizes, ctmc.retry_queue_sizes)] = 1
    lambda_seq = []
    pi_q_seq = [copy.copy(pi_q_new)]  # Initializing the initial distribution
    main_queue_ave_len_seq = [0]

    ctmc.fault_simulation_data_generator(pi_q_seq, main_queue_ave_len_seq, lambda_seq,
                                         step_time,
                                         sim_time,
                                         lambda_init,
                                         lambda_fault,
                                         fault_time,
                                         lambda_reset,
                                         reset_time)

    timee = [i * step_time for i in list(range(0, len(main_queue_ave_len_seq) - 1))]
    # Create 4x1 sub plots
    plt.rcParams["figure.figsize"] = [6, 10]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(2, 1)
    ax.update(wspace=0.5, hspace=0.5)

    ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
    ax1.plot(timee, main_queue_ave_len_seq[0: len(main_queue_ave_len_seq) - 1], color='tab:blue')
    ax1.set_xlabel('Time', fontsize=16)
    ax1.set_ylabel('Expected sum of queue lengths', fontsize=16)
    ax1.grid('on')
    ax1.set_xlim(0, max(timee))

    ax3 = plt.subplot(ax[1, 0])  # row 2, col 0
    ax3.plot(timee, lambda_seq, color='tab:purple')
    ax3.set_xlabel('Time', fontsize=16)
    ax3.set_ylabel('Job arrival rate', fontsize=16)
    ax3.grid('on')
    ax3.set_xlim(0, max(timee))

    plt.savefig("fault_output")
    plt.close()


analysis(MultiServerCTMC(server_no= 2, main_queue_sizes= [10, 10], retry_queue_sizes= [3, 3], lambdaas= [.5, 0],
                        mu0_ps= [2.8, 2.8], timeouts= [5,5], max_retries= [5, 3], thread_pools= [1, 1],
                        parent_list= [[], [0]], sub_tree_list= [[0, 1], [1]], q_min_list= [90, 90], q_max_list= [10, 10],
                        o_min_list= [0, 0], o_max_list = [3, 3]),
             step_time= 1,
             sim_time= 100,
             lambda_init = [.5, 0],
             lambda_fault= [18, 0],
             fault_time= 0,
             lambda_reset= [.5, 0],
             reset_time= 20)
