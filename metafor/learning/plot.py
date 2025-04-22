import numpy as np
from typing import List

from matplotlib import pyplot as plt


def plot_results(step_time: int, q_seq_data, q_seq_learned_model, qsize: int,
                 osize: int, figure_name: str):
    """This function plots variations of four quantities over different time bounds:
        (1) mean queue length, (2) variance of queue length, (3) standard deviation of queue length, (4) runtime."""
    num_data = len(q_seq_data)
    q_list_data = []
    q_list_model = []

    for i in range(num_data):
        q_list_data.append(q_seq_data[i])
        q_list_model.append(q_seq_learned_model[i])
    time = [i * step_time for i in list(range(0, num_data))]
    # Create 1x1 sub plots
    plt.rc('font', size=14)
    plt.rcParams["figure.figsize"] = [5, 5]
    plt.rcParams["figure.autolayout"] = True

    ax = plt.GridSpec(1, 1)
    ax.update(wspace=0.5, hspace=0.5)
    plt.subplots_adjust(left=0.2)
    
    ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
    ax1.plot(time, q_list_data, color='tab:blue', linewidth=2)
    ax1.plot(time, q_list_model, color='tab:red', linewidth=2)
    #ax1.axvline(x=200, color='red', linestyle='--', linewidth=2)
    #ax1.axvline(x=400, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Time', fontsize=14)
    ax1.set_ylabel('Number of requests in the system', fontsize=14)
    ax1.grid('on')
    ax1.set_xlim(0, max(time))

    """ax2 = plt.subplot(ax[0, 1])  # row 0, col 1
    ax2.plot(time, rho_seq, color='tab:purple')
    ax2.set_xlabel('Time', fontsize=14)
    ax2.set_ylabel('Arrival rate of requests', fontsize=14)
    ax2.grid('on')
    ax2.set_xlim(0, max(time))"""
    """ax3 = plt.subplot(ax[1, 0])  # row 2, col 0
    ax3.plot(time, latency_std_seq, color='tab:purple')
    ax3.set_xlabel('Time bound', fontsize=14)
    ax3.set_ylabel('Standard deviation of latency', fontsize=14)
    ax3.grid('on')
    ax3.set_xlim(0, max(time))

    ax4 = plt.subplot(ax[2, 0])  # row 3, col 0
    ax4.plot(time, runtime_seq, color='tab:green')
    ax4.set_xlabel('Time bound', fontsize=14)
    ax4.set_ylabel('Runtime (sec)', fontsize=14)
    ax4.grid('on')
    ax4.set_xlim(0, max(time))"""

    plt.savefig(figure_name)
    plt.show()
    plt.close()


def main_queue_average_size(pi, qsize, osize) -> float:
    """This function computes the average queue length for a given prob distribution pi"""
    main_queue_size = qsize
    retry_queue_size = osize
    length = 0
    for n_main_queue in range(main_queue_size):
        weight = 0
        for n_retry_queue in range(retry_queue_size):
            weight += pi[index_composer(n_main_queue, n_retry_queue, main_queue_size, retry_queue_size)]
        length += weight * n_main_queue
    return length

def index_composer(n_main_queue, n_retry_queue, qsize, osize):
    """This function converts two given input indices into one universal index in range [0, state_num].
    The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit."""
    main_queue_size = qsize

    total_ind = n_retry_queue * main_queue_size + n_main_queue
    return total_ind