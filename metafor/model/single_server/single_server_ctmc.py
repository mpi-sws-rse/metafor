from typing import List

import numpy as np
import math
import copy
import scipy
import time

from utils.plot_parameters import PlotParameters
from model.ctmc import CTMC
from model.ctmc_parameters import CTMCParameters
from utils.calculate import tail_prob_computer


class SingleServerCTMC(CTMC):

    def __init__(self, main_queue_size: int, retry_queue_size: int, lambdaas: List[float], mu0_ps: List[float],
                 timeouts: List[int], retries: List[int], thread_pool: int, alpha: float = 0.25):
        assert (thread_pool >= 1)
        assert (len(lambdaas) == len(mu0_ps) == len(timeouts) == len(retries))
        self.params = self.init_parameters(main_queue_size, retry_queue_size, lambdaas, mu0_ps, timeouts, retries,
                                           thread_pool, alpha)

    @staticmethod
    def init_parameters(main_queue_size: int, retry_queue_size: int, lambdaas: List[float], mu0_ps: List[float],
                        timeouts: List[int], retries: List[int], thread_pool: int, alpha: float) -> CTMCParameters:
        state_num = main_queue_size * retry_queue_size
        lambdaa = sum(lambdaas)
        mu0_p = 0
        timeout = 0
        max_retries = 0
        for lambdaa_i, mu0_p_i, timeout_i, retries_i in zip(lambdaas, mu0_ps, timeouts, retries):
            mu0_p += mu0_p_i * (lambdaa_i / lambdaa)
            timeout += timeout_i * (lambdaa_i / lambdaa)
            max_retries += retries_i * (lambdaa_i / lambdaa)

        # the rate of retrying jobs in the retry queue
        mu_retry_base = max_retries * lambdaa / ((max_retries + 1) * timeout)
        # the rate of dropping jobs in the retry queue
        mu_drop_base = lambdaa / ((max_retries + 1) * timeout)
        return CTMCParameters(main_queue_size, retry_queue_size, lambdaa, lambdaas, state_num, mu_retry_base,
                              mu_drop_base, mu0_p, mu0_ps, timeout, timeouts, max_retries, retries, thread_pool, alpha)

    def index_decomposer(self, total_ind):
        """This function converts a given index in range [0, state_num]
        into two indices corresponding to (1) number of jobs in orbit and (2) jobs in the queue."""
        main_queue_size = self.params.main_queue_size

        n_retry_queue = total_ind // main_queue_size
        n_main_queue = total_ind % main_queue_size
        assert 0 <= n_retry_queue < self.params.state_num
        assert 0 <= n_main_queue < self.params.state_num
        return [n_retry_queue, n_main_queue]

    def index_composer(self, n_main_queue, n_retry_queue):
        """This function converts two given input indices into one universal index in range [0, state_num].
        The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit."""
        main_queue_size = self.params.main_queue_size

        total_ind = n_retry_queue * main_queue_size + n_main_queue
        assert 0 <= total_ind < self.params.state_num
        return total_ind

    def generator_mat_exact(self, transition_matrix: bool = False):
        alpha = self.params.alpha
        state_num = self.params.state_num
        lambdaa = self.params.lambdaa
        mu_drop_base = self.params.mu_drop_base
        mu_retry_base = self.params.mu_retry_base
        mu0_p = self.params.mu0_p
        main_queue_size = self.params.main_queue_size
        retry_queue_size = self.params.retry_queue_size
        thread_pool = self.params.thread_pool

        Q = np.zeros((state_num, state_num))
        tail_seq = tail_prob_computer(main_queue_size, mu0_p, self.params.timeout)
        for total_ind in range(state_num):
            n_retry_queue, n_main_queue = self.index_decomposer(total_ind)
            tail_main = tail_seq[n_main_queue]
            if n_main_queue == 0:  # queue is empty
                Q[total_ind, self.index_composer(n_main_queue + 1, n_retry_queue)] = lambdaa
                if n_retry_queue > 0:
                    Q[total_ind, self.index_composer(n_main_queue, n_retry_queue - 1)] = \
                        n_retry_queue * mu_drop_base
                    Q[total_ind, self.index_composer(n_main_queue + 1, n_retry_queue - 1)] = \
                        n_retry_queue * mu_retry_base
            elif n_main_queue == main_queue_size - 1:  # queue is full
                Q[total_ind, self.index_composer(n_main_queue - 1, n_retry_queue)] = \
                    min(main_queue_size, thread_pool) * mu0_p
                if n_retry_queue > 0:
                    Q[total_ind, self.index_composer(n_main_queue, n_retry_queue - 1)] = \
                        n_retry_queue * mu_drop_base
                if n_retry_queue < retry_queue_size - 1:
                    Q[total_ind, self.index_composer(n_main_queue, n_retry_queue + 1)] = \
                        alpha * (lambdaa + n_retry_queue * mu_retry_base) * tail_main
            else:  # queue is neither full nor empty
                alpha_tail_prob_sum = alpha * lambdaa * tail_main
                if n_retry_queue < retry_queue_size - 1:
                    Q[total_ind, self.index_composer(n_main_queue + 1, n_retry_queue + 1)] = alpha_tail_prob_sum
                Q[total_ind, self.index_composer(n_main_queue + 1, n_retry_queue)] = \
                    lambdaa + n_retry_queue * mu_retry_base * tail_main
                if n_retry_queue > 0:
                    Q[total_ind, self.index_composer(n_main_queue + 1, n_retry_queue - 1)] = \
                        n_retry_queue * mu_retry_base * (1 - tail_main)
                    Q[total_ind, self.index_composer(n_main_queue, n_retry_queue - 1)] = n_retry_queue * mu_drop_base
                Q[total_ind, self.index_composer(n_main_queue - 1, n_retry_queue)] = \
                    min(main_queue_size, thread_pool) * mu0_p
            if not transition_matrix:
                Q[total_ind, total_ind] = - np.sum(Q[total_ind, :])
        return Q

    def get_stationary_distribution(self):
        Q = self.generator_mat_exact()
        QT = np.transpose(Q)
        ns = scipy.linalg.null_space(QT)
        self.pi = ns / np.linalg.norm(ns, ord=1)
        if sum(self.pi) < - 0.01: # the null space can return the negation of the stationary distribution
                self.pi = - self.pi
        return self.pi
    
    def main_queue_average_size(self, pi) -> float:
        """This function computes the average queue length for a given prob distribution pi"""
        main_queue_size = self.params.main_queue_size
        retry_queue_size = self.params.retry_queue_size

        length = 0
        for n_main_queue in range(main_queue_size):
            weight = 0
            for n_retry_queue in range(retry_queue_size):
                weight += pi[self.index_composer(n_main_queue, n_retry_queue)]
            length += weight * n_main_queue
        return length

    def main_queue_size_var(self, pi, mean_queue_length) -> float:
        """This function computes the variance over queue length for a given prob distribution pi"""
        main_queue_size = self.params.main_queue_size
        retry_queue_size = self.params.retry_queue_size

        var = 0
        for n_main_queue in range(main_queue_size):
            weight = 0
            for n_retry_queue in range(retry_queue_size):
                weight += pi[self.index_composer(n_main_queue, n_retry_queue)]
            var += weight * (n_main_queue - mean_queue_length) ** 2
        return var

    def retry_queue_average_size(self, pi) -> float:
        """This function computes the average queue length for a given prob distribution pi"""
        main_queue_size = self.params.main_queue_size
        retry_queue_size = self.params.retry_queue_size

        length = 0
        for n_retry_queue in range(retry_queue_size):
            weight = 0
            for n_main_queue in range(main_queue_size):
                weight += pi[self.index_composer(n_main_queue, n_retry_queue)]
            length += weight * n_retry_queue
        return length

    @staticmethod
    def main_queue_size_std(main_queue_variance) -> float:
        """This function computes the standard deviation over queue length from its variance"""
        return math.sqrt(main_queue_variance)

    # job_type represents the index of the job
    def latency_average(self, pi, job_type: int = -1) -> float:
        retry_queue_size = self.params.retry_queue_size
        main_queue_size = self.params.main_queue_size
        # use the law of total expectation
        mu0_p = self.params.mu0_ps[job_type]
        val = 1 / mu0_p
        for n_main_queue in range(main_queue_size):
            weight = 0
            for n_retry_queue in range(retry_queue_size):
                weight += pi[self.index_composer(n_main_queue, n_retry_queue)]
            val += weight * (self.params.lambdaas[job_type] / self.params.lambdaa) * n_main_queue * (1 / mu0_p)
        return val[0]

    def latency_var(self, pi, job_type: int) -> float:
        retry_queue_size = self.params.retry_queue_size
        main_queue_size = self.params.main_queue_size
        mu0_p = self.params.mu0_ps[job_type]
        ave = self.latency_average(pi, job_type)
        # use the law of total variance
        var1 = 1 / (mu0_p ** 2)  # var1 := Var(E(Y|X))
        for n_main_queue in range(main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                weight = pi[state]
                var1 += weight * (((self.params.lambdaas[job_type] / self.params.lambdaa) * n_main_queue *
                                   (1 / mu0_p) - ave) ** 2)

        var2 = 0  # var2 := E(Var(Y|X))
        for n_main_queue in range(main_queue_size):
            weight = 0
            for n_retry_queue in range(retry_queue_size):
                weight += pi[self.index_composer(n_main_queue, n_retry_queue)]
            var2 += weight * (self.params.lambdaas[job_type] / self.params.lambdaa) * n_main_queue * \
                             (1 / self.params.mu0_p ** 2)

        var = var1 + var2
        return var[0]

    def hitting_time_average_us(self, Q, pi, qlen_max) -> float:
        state_num = self.params.state_num
        retry_queue_size = self.params.retry_queue_size
        main_queue_size = self.params.main_queue_size
        A = np.copy(Q)
        b = -np.ones(state_num)
        for n_main_queue in range(qlen_max):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                A[state, :] = 0
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)
        hitting_time_mean = 0
        for n_main_queue in range(main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                hitting_time_mean += pi[state] * u[state]
        return hitting_time_mean

    def hitting_time_average_su(self, Q, pi, qlen_min) -> float:
        state_num = self.params.state_num
        retry_queue_size = self.params.retry_queue_size
        main_queue_size = self.params.main_queue_size
        A = np.copy(Q)
        b = -np.ones(state_num)
        for n_main_queue in range(qlen_min, main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                A[state, :] = 0
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)
        hitting_time_mean = 0
        for n_main_queue in range(main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                hitting_time_mean += pi[state] * u[state]
        return hitting_time_mean

    def hitting_time_variance_us(self, Q, pi, qlen_max) -> float:
        state_num = self.params.state_num
        retry_queue_size = self.params.retry_queue_size
        main_queue_size = self.params.main_queue_size
        A = np.copy(Q)
        b = -np.ones(state_num)
        for n_main_queue in range(qlen_max):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                A[state, :] = np.zeros(state_num)
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)

        v = np.linalg.solve(A, -2 * u)

        hitting_time_var = 0
        for n_main_queue in range(main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                hitting_time_var += (pi[state] * (v[state] - (u[state] ** 2)))
        return hitting_time_var

    def hitting_time_variance_su(self, Q, pi, qlen_min) -> float:
        state_num = self.params.state_num
        retry_queue_size = self.params.retry_queue_size
        main_queue_size = self.params.main_queue_size
        A = np.copy(Q)
        b = -np.ones(state_num)
        for n_main_queue in range(qlen_min, main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                A[state, :] = np.zeros(state_num)
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)

        v = np.linalg.solve(A, -2 * u)

        hitting_time_var = 0
        for n_main_queue in range(main_queue_size):
            for n_retry_queue in range(retry_queue_size):
                state = self.index_composer(n_main_queue, n_retry_queue)
                hitting_time_var += (pi[state] * (v[state] - (u[state] ** 2)))
        return hitting_time_var

    def server_population_queue_model(self, Q, pi_q_seq, plot_params: PlotParameters):
        """This function runs CTMC over the given simulation time and
        provides some analysis over the system's performance."""
        step_time = plot_params.step_time
        sim_time = int(plot_params.sim_time)

        main_queue_avg_len_seq = [0]
        main_queue_var_len_seq = [0]
        main_queue_std_len_seq = [0]
        runtime_seq = [0]

        retry_queue_avg_len_seq = [0]

        Q_T = np.transpose(Q)
        print('Computing the average length of the main queue/orbit in the time interval %d-%d' % (step_time, sim_time))
        for t in range(step_time, sim_time + 1, step_time):
            print("Time bound is equal to", t)
            start_time = time.time()
            # Updating the system states
            pi_q_new = np.transpose(np.matmul(scipy.linalg.expm(Q_T * t), pi_q_seq[0]))
            pi_q_seq.append(copy.copy(pi_q_new))
            main_queue_mean_length = self.main_queue_average_size(pi_q_new)
            main_queue_avg_len_seq.append(main_queue_mean_length)
            main_queue_variance_length = self.main_queue_size_var(pi_q_new, main_queue_mean_length)
            main_queue_var_len_seq.append(main_queue_variance_length)
            main_queue_std_len_seq.append(self.main_queue_size_std(main_queue_variance_length))
            retry_queue_avg_len_seq.append(self.retry_queue_average_size(pi_q_new))
            runtime_seq.append(time.time() - start_time)

        return (main_queue_avg_len_seq, main_queue_var_len_seq, main_queue_std_len_seq,
                retry_queue_avg_len_seq, runtime_seq)

    def reach_prob_computation(self, Q, plot_params: PlotParameters) -> List[float]:
        print('Computing the probability to reach the queue full mode')

        state_num = self.params.state_num
        step_time = plot_params.step_time
        sim_time = int(plot_params.sim_time)
        target_set = [state_num - 1]

        probabilities = []

        Q_mod = copy.copy(Q)
        init_vec = np.zeros(state_num)
        init_vec[target_set[0]] = 1
        # making the target states absorbing
        for state in target_set:
            Q_mod[state, :] = np.zeros(state_num)
        for t in range(step_time, sim_time + 1, step_time):
            start_time = time.time()
            reach_prob_vec = np.matmul(scipy.linalg.expm(Q_mod * t), init_vec)
            end_time = time.time()
            runtime = end_time - start_time
            print("P = %f for time bound %d" % (reach_prob_vec[0], t))
            print("Analysis time for time bound %d: %f s" % (t, runtime))
            probabilities.append(reach_prob_vec[0])
        return probabilities

    def reach_prob(self, Q, t: int, plot_params: PlotParameters) -> float:
        assert t % plot_params.step_time == 0
        probabilities = self.reach_prob_computation(Q, plot_params)
        return probabilities[int(plot_params.sim_time / t)]

    def queues_avg_len(self, Q, t: int, plot_params: PlotParameters) -> (float, float):
        assert t % plot_params.step_time == 0

        pi_q_new = np.zeros(self.params.state_num)
        pi_q_new[0] = 1  # Initially the queue is empty
        pi_q_seq = [copy.copy(pi_q_new)]  # Initializing the initial distribution

        main_queue_avg_len, _, _, retry_queue_avg_len, _ = self.server_population_queue_model(Q, pi_q_seq, plot_params)
        index = int(plot_params.sim_time / t)
        return main_queue_avg_len[index], retry_queue_avg_len[index]

    """This function computes the generator matrix of CTMC."""
    def sparse_info_calculator(self):
        row_ind = []
        col_ind = []
        data_point = []
        state_num = self.params.state_num
        lambdaa = self.params.lambdaa
        mu_drop_base = self.params.mu_drop_base
        mu_retry_base = self.params.mu_retry_base
        mu0_p = self.params.mu0_p
        main_queue_size = self.params.main_queue_size
        retry_queue_size = self.params.retry_queue_size
        timeout = self.params.timeout

        tail_seq = tail_prob_computer(main_queue_size, mu0_p, timeout)
        for total_ind in range(state_num):
            data_sum = 0
            n_retry_queue, n_main_queue = self.index_decomposer(total_ind)
            tail_main = tail_seq[n_main_queue]
            if n_main_queue == 0:  # queue is empty
                row_ind.append(total_ind)
                col_ind.append(self.index_composer(n_main_queue + 1, n_retry_queue))
                data_point.append(lambdaa)
                data_sum += lambdaa
                if n_retry_queue > 0:
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue, n_retry_queue - 1))
                    data_point.append(n_retry_queue * mu_drop_base)
                    data_sum += n_retry_queue * mu_drop_base
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue + 1, n_retry_queue - 1))
                    data_point.append(n_retry_queue * mu_retry_base)
                    data_sum += n_retry_queue * mu_retry_base

            elif n_main_queue == main_queue_size - 1:  # queue is full
                row_ind.append(total_ind)
                col_ind.append(self.index_composer(n_main_queue - 1, n_retry_queue))
                data_point.append(mu0_p)
                data_sum += mu0_p
                if n_retry_queue > 0:
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue, n_retry_queue - 1))
                    data_point.append(n_retry_queue * mu_drop_base)
                    data_sum += n_retry_queue * mu_drop_base
                if n_retry_queue < retry_queue_size - 1:
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue, n_retry_queue + 1))
                    data_point.append((lambdaa + n_retry_queue * mu_retry_base) * tail_main)
                    data_sum += (lambdaa + n_retry_queue * mu_retry_base) * tail_main
            else:  # queue is neither full nor empty
                alpha_tail_prob_sum = lambdaa * tail_main
                if n_retry_queue < retry_queue_size - 1:
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue + 1, n_retry_queue + 1))
                    data_point.append(alpha_tail_prob_sum)
                    data_sum += alpha_tail_prob_sum
                row_ind.append(total_ind)
                col_ind.append(self.index_composer(n_main_queue + 1, n_retry_queue))
                data_point.append(lambdaa + n_retry_queue * mu_retry_base * tail_main)
                data_sum += lambdaa + n_retry_queue * mu_retry_base * tail_main
                if n_retry_queue > 0:
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue + 1, n_retry_queue - 1))
                    data_point.append(n_retry_queue * mu_retry_base * (1 - tail_main))
                    data_sum += n_retry_queue * mu_retry_base * (1 - tail_main)
                    row_ind.append(total_ind)
                    col_ind.append(self.index_composer(n_main_queue, n_retry_queue - 1))
                    data_point.append(n_retry_queue * mu_drop_base)
                    data_sum += n_retry_queue * mu_drop_base
                row_ind.append(total_ind)
                col_ind.append(self.index_composer(n_main_queue - 1, n_retry_queue))
                data_point.append(mu0_p)
                data_sum += mu0_p
            row_ind.append(total_ind)
            col_ind.append(total_ind)
            data_point.append(-data_sum)
        return [row_ind, col_ind, data_point]

    def prob_dist_accumulator(self, pi, q1, q2, o1, o2):
        val = 0
        for q in range(q1, q2):
            for o in range(o1, o2):
                state = self.index_composer(q, o)
                val += pi[state]
        return val
