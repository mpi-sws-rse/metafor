from typing import Any, Callable, List, Tuple, Dict

import numpy as np
import numpy.typing as npt

import math
import copy
import scipy
import time

from utils.plot_parameters import PlotParameters
from model.ctmc import CTMC
from utils.calculate import tail_prob_computer


class SingleServerCTMC(CTMC):

    def __init__(
        self,
        main_queue_size: int,
        retry_queue_size: int,
        lambdaas: List[float],
        mu0_ps: List[float],
        timeouts: List[int],
        retries: List[int],
        thread_pool: int,
        alpha: float = 0.25,
    ):
        """alpha is a parameter that allows us to adjust the CTMC with simulation data"""

        super().__init__()
        assert thread_pool >= 1
        assert len(lambdaas) == len(mu0_ps) == len(timeouts) == len(retries)
        self.main_queue_size = main_queue_size
        self.retry_queue_size = retry_queue_size
        self.state_num = self.main_queue_size * self.retry_queue_size
        self.lambdaas = lambdaas  # arrival rates for different types of requests
        self.lambdaa = sum(lambdaas)

        self.mu0_ps = mu0_ps
        self.timeouts = timeouts  # timeouts for different types of requests
        self.retries = retries

        mu0_p = 0
        timeout = 0
        max_retries = 0
        for lambdaa_i, mu0_p_i, timeout_i, retries_i in zip(
            lambdaas, mu0_ps, timeouts, retries
        ):
            mu0_p += mu0_p_i * (lambdaa_i / self.lambdaa)
            timeout += timeout_i * (lambdaa_i / self.lambdaa)
            max_retries += retries_i * (lambdaa_i / self.lambdaa)

        self.mu0_p = mu0_p
        self.timeout = timeout
        self.max_retries = max_retries

        print(
            "main qsize = ",
            self.main_queue_size,
            " retry qsize = ",
            self.retry_queue_size,
        )
        print("lambdaas = ", self.lambdaas)
        print("lambdaa = ", self.lambdaa)
        print("mu = ", self.mu0_ps)

        # the rate of retrying jobs in the retry queue
        self.mu_retry_base = max_retries * self.lambdaa / ((max_retries + 1) * timeout)
        # the rate of dropping jobs in the retry queue
        self.mu_drop_base = self.lambdaa / ((max_retries + 1) * timeout)

        self.thread_pool = thread_pool
        self.alpha = alpha

        self.Q = self.generator_mat_exact(transition_matrix=False)

        # Debug: check that each row sums to 0 (approximately)
        # np.set_printoptions(threshold=sys.maxsize)
        for row in self.Q:
            assert abs(sum(row)) < 0.0001
        # np.set_printoptions(threshold=None)

    def get_init_state(self) -> npt.NDArray[np.float64]:
        pi = np.zeros(self.state_num)
        pi[0] = 1.0  # Initially the queue is empty
        return pi

    # compute the stationary distribution and cache it
    def get_stationary_distribution(self) -> npt.NDArray[np.float64]:
        if self.pi is None:
            # calculate the stationary distribution
            QT = np.transpose(self.Q)
            ns = scipy.linalg.null_space(QT)
            self.pi = ns / np.linalg.norm(ns, ord=1)
            if sum(self.pi) < -0.01:  # the null space may return `-pi`
                self.pi = -self.pi
            return self.pi
        else:
            return self.pi

    def get_eigenvalues(self):
        eigenvalues = np.linalg.eigvals(self.Q)
        sorted_eigenvalues = np.sort(eigenvalues.real)[::-1]
        return sorted_eigenvalues

    def get_mixing_time(self):
        eigenvalues = self.get_eigenvalues()  # RM: we only need the first eigenvalue
        t_mixing = 1 / abs(eigenvalues[1]) * math.log2(100)
        return t_mixing

    def _index_decomposer(self, total_ind):
        """This function converts a given index in range [0, state_num]
        into two indices corresponding to number of (1) jobs in the orbit and (2) jobs in the queue.
        """
        n_retry_queue = total_ind // self.main_queue_size
        n_main_queue = total_ind % self.main_queue_size
        assert 0 <= n_retry_queue < self.state_num
        assert 0 <= n_main_queue < self.state_num
        return [n_retry_queue, n_main_queue]

    def _index_composer(self, n_main_queue, n_retry_queue):
        """This function converts two given input indices into one universal index in range [0, state_num].
        The input indices correspond to number of (1) jobs in the queue and (2) jobs in the orbit.
        """
        total_ind = n_retry_queue * self.main_queue_size + n_main_queue
        assert 0 <= total_ind < self.state_num
        return total_ind

    def generator_mat_exact(self, transition_matrix: bool = False):
        Q = np.zeros((self.state_num, self.state_num))
        tail_seq = tail_prob_computer(self.main_queue_size, self.mu0_p, self.timeout)
        for total_ind in range(self.state_num):
            n_retry_queue, n_main_queue = self._index_decomposer(total_ind)
            tail_main = tail_seq[n_main_queue]
            if n_main_queue == 0:  # queue is empty
                Q[total_ind, self._index_composer(n_main_queue + 1, n_retry_queue)] = (
                    self.lambdaa
                )
                if n_retry_queue > 0:
                    Q[
                        total_ind, self._index_composer(n_main_queue, n_retry_queue - 1)
                    ] = (n_retry_queue * self.mu_drop_base)
                    Q[
                        total_ind,
                        self._index_composer(n_main_queue + 1, n_retry_queue - 1),
                    ] = (
                        n_retry_queue * self.mu_retry_base
                    )
            elif n_main_queue == self.main_queue_size - 1:  # queue is full
                Q[total_ind, self._index_composer(n_main_queue - 1, n_retry_queue)] = (
                    min(self.main_queue_size, self.thread_pool) * self.mu0_p
                )
                if n_retry_queue > 0:
                    Q[
                        total_ind, self._index_composer(n_main_queue, n_retry_queue - 1)
                    ] = (n_retry_queue * self.mu_drop_base)
                if n_retry_queue < self.retry_queue_size - 1:
                    Q[
                        total_ind, self._index_composer(n_main_queue, n_retry_queue + 1)
                    ] = (
                        self.alpha
                        * (self.lambdaa + n_retry_queue * self.mu_retry_base)
                        * tail_main
                    )
            else:  # queue is neither full nor empty
                alpha_tail_prob_sum = self.alpha * self.lambdaa * tail_main
                if n_retry_queue < self.retry_queue_size - 1:
                    Q[
                        total_ind,
                        self._index_composer(n_main_queue + 1, n_retry_queue + 1),
                    ] = alpha_tail_prob_sum
                Q[total_ind, self._index_composer(n_main_queue + 1, n_retry_queue)] = (
                    self.lambdaa + n_retry_queue * self.mu_retry_base * tail_main
                )
                if n_retry_queue > 0:
                    Q[
                        total_ind,
                        self._index_composer(n_main_queue + 1, n_retry_queue - 1),
                    ] = (
                        n_retry_queue * self.mu_retry_base * (1 - tail_main)
                    )
                    Q[
                        total_ind, self._index_composer(n_main_queue, n_retry_queue - 1)
                    ] = (n_retry_queue * self.mu_drop_base)
                Q[total_ind, self._index_composer(n_main_queue - 1, n_retry_queue)] = (
                    min(self.main_queue_size, self.thread_pool) * self.mu0_p
                )
            if not transition_matrix:
                Q[total_ind, total_ind] = -np.sum(Q[total_ind, :])
        return Q

    def finite_time_analysis(
        self,
        pi0,
        analyses: dict[str, Callable[[Any], Any]] = {},
        sim_time=60,
        sim_step=10,
    ):
        initial_result = {n: 0.0 for n in analyses}
        # XXX: we are assuming the initial result for the analyses is 0, which may not be true
        initial_result["pi"] = pi0
        results = {0: initial_result}
        print(
            "Computing finite time statistics for time quantum %d time units with step size %d"
            % (sim_time, sim_step)
        )
        QT = np.transpose(self.Q)
        start = time.time()
        print("Starting matrix exponentiation...", end=" ")
        matexp = scipy.linalg.expm(QT * sim_step)
        print("Matrix exponentiation took %f s" % (time.time() - start))

        piq = pi0
        for t in range(sim_step, sim_time, sim_step):
            result = {}
            start = time.time()
            piq = np.transpose(np.matmul(matexp, piq))
            elapsed_time = time.time() - start
            result["step"] = t
            result["pi"] = piq
            # pi = np.transpose(np.matmul(scipy.linalg.expm(QT * t), pi0))
            # result['inefficientpi'] = pi
            result["wallclock_time"] = elapsed_time
            # now run all the analyses passed to this function with the current distribution
            for analysis_name, analysis_fn in analyses.items():
                v = analysis_fn(piq)
                result[analysis_name] = v

            results[t] = result
            print(result)
        return results

    ########### Various analyses built on top of probability distributions ###############
    def main_queue_size_average(self, pi) -> float:
        """This function computes the average queue length for a given prob distribution pi"""
        length = 0.0
        for n_main_queue in range(self.main_queue_size):
            weight = 0.0
            for n_retry_queue in range(self.retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            length += weight * n_main_queue
        return length

    def main_queue_size_variance(self, pi, mean_queue_length) -> float:
        """This function computes the variance over queue length for a given prob distribution pi"""
        var = 0
        for n_main_queue in range(self.main_queue_size):
            weight = 0
            for n_retry_queue in range(self.retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            var += (
                weight
                * (n_main_queue - mean_queue_length)
                * (n_main_queue - mean_queue_length)
            )
        return var

    def main_queue_size_std(self, pi, mean_queue_length) -> float:
        return math.sqrt(self.main_queue_size_variance(pi, mean_queue_length))

    def main_queue_size_analysis(self, pi) -> Dict[str, float]:
        avg = self.main_queue_size_average(pi)
        variance = self.main_queue_size_variance(pi, avg)
        std = self.main_queue_size_std(pi, avg)
        return {"avg": avg, "variance": variance, "std": std}

    def retry_queue_size_average(self, pi) -> float:
        """This function computes the average queue length for a given prob distribution pi"""
        length = 0
        for n_retry_queue in range(self.retry_queue_size):
            weight = 0
            for n_main_queue in range(self.main_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            length += weight * n_retry_queue
        return length

    def retry_queue_size_variance(self, pi, mean_queue_length) -> float:
        """This function computes the variance over queue length for a given prob distribution pi"""
        var = 0
        for n_retry_queue in range(self.retry_queue_size):
            weight = 0
            for n_main_queue in range(self.main_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            var += (
                weight
                * (n_retry_queue - mean_queue_length)
                * (n_retry_queue - mean_queue_length)
            )
        return var

    def retry_queue_size_std(self, pi, mean_queue_length) -> float:
        return math.sqrt(self.retry_queue_size_variance(pi, mean_queue_length))

    def throughput_average(self, pi) -> float:
        main_queue_length = self.main_queue_size_average(pi)
        return main_queue_length/self.mu0_p

    def failure_rate_average(self, pi, req_type: int = 0) -> float:
        main_queue_length = self.main_queue_size_average(pi)
        total_reqs = self.lambdaas[req_type] * self.latency_average(pi, req_type)
        successful_reqs = main_queue_length * self.lambdaas[req_type] / self.lambdaa
        return abs(total_reqs - successful_reqs)/total_reqs

    # req_type represents the index of the request
    def latency_average(self, pi, req_type: int = 0) -> float:
        # use the law of total expectation
        mu0_p = self.mu0_ps[req_type]
        val = 1 / mu0_p
        for n_main_queue in range(self.main_queue_size):
            weight = 0
            for n_retry_queue in range(self.retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            val += (
                weight
                * (self.lambdaas[req_type] / self.lambdaa)
                * n_main_queue
                * (1 / mu0_p)
            )
        return val[0]

    def latency_variance(self, pi, mean: float, req_type: int = 0) -> float:
        mu0_p = self.mu0_ps[req_type]
        # use the law of total variance
        var1 = 1 / (mu0_p * mu0_p)  # var1 := Var(E(Y|X))
        for n_main_queue in range(self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                weight = pi[state]
                var1 += weight * (
                    (
                        (self.lambdaas[req_type] / self.lambdaa)
                        * n_main_queue
                        * (1 / mu0_p)
                        - mean
                    )
                    ** 2
                )

        var2 = 0  # var2 := E(Var(Y|X))
        for n_main_queue in range(self.main_queue_size):
            weight = 0
            for n_retry_queue in range(self.retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            var2 += (
                weight
                * (self.lambdaas[req_type] / self.lambdaa)
                * n_main_queue
                * (1 / self.mu0_p**2)
            )

        var = var1 + var2
        return var[0]

    def latency_percentile(self, pi, req_type: int = 0, percentile: float = 50.0):
        assert percentile <= 100.0
        retry_queue_size = self.retry_queue_size
        main_queue_size = self.main_queue_size
        distribution = np.zeros((self.state_num, 2))
        mu0_p = self.mu0_ps[req_type]
        val = 1 / mu0_p
        index = 0
        for n_main_queue in range(main_queue_size):
            weight = 0
            for n_retry_queue in range(retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            val += (self.lambdaas[req_type] / self.lambdaa) * n_main_queue * (1 / mu0_p)
            distribution[index][0] = val
            distribution[index][1] = weight
            index = index + 1
        distribution.sort(axis=0)
        cum = 0
        index = 0
        while cum * 100 < percentile:
            cum = cum + distribution[index][1]
            index = index + 1
        print(distribution)
        print(index)
        return distribution[index][0]

    def hitting_time_average_us(self, Q, pi, qlen_max) -> float:
        A = np.copy(Q)
        b = -np.ones(self.state_num)
        for n_main_queue in range(qlen_max):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                A[state, :] = 0
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)
        hitting_time_mean = 0
        for n_main_queue in range(self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                hitting_time_mean += pi[state] * u[state]
        return hitting_time_mean

    def hitting_time_average_su(self, Q, pi, qlen_min) -> float:
        A = np.copy(Q)
        b = -np.ones(self.state_num)
        for n_main_queue in range(qlen_min, self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                A[state, :] = 0
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)
        hitting_time_mean = 0
        for n_main_queue in range(self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                hitting_time_mean += pi[state] * u[state]
        return hitting_time_mean

    def hitting_time_variance_us(self, Q, pi, qlen_max) -> float:
        A = np.copy(Q)
        b = -np.ones(self.state_num)
        for n_main_queue in range(qlen_max):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                A[state, :] = np.zeros(self.state_num)
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)

        v = np.linalg.solve(A, -2 * u)

        hitting_time_var = 0
        for n_main_queue in range(self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                hitting_time_var += pi[state] * (v[state] - (u[state] ** 2))
        return hitting_time_var

    def hitting_time_variance_su(self, Q, pi, qlen_min) -> float:
        A = np.copy(Q)
        b = -np.ones(self.state_num)
        for n_main_queue in range(qlen_min, self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                A[state, :] = np.zeros(self.state_num)
                A[state, state] = 1
                b[state] = 0

        u = np.linalg.solve(A, b)

        v = np.linalg.solve(A, -2 * u)

        hitting_time_var = 0
        for n_main_queue in range(self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                hitting_time_var += pi[state] * (v[state] - (u[state] ** 2))
        return hitting_time_var

    def reach_prob_computation(self, Q, plot_params: PlotParameters) -> List[float]:
        print("Computing the probability to reach the queue full mode")
        step_time = plot_params.step_time
        sim_time = int(plot_params.sim_time)
        target_set = [self.state_num - 1]

        probabilities = []

        Q_mod = copy.copy(Q)
        init_vec = np.zeros(self.state_num)
        init_vec[target_set[0]] = 1
        # making the target states absorbing
        for state in target_set:
            Q_mod[state, :] = np.zeros(self.state_num)
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

    def sparse_info_calculator(self):
        row_ind = []
        col_ind = []
        data_point = []

        tail_seq = tail_prob_computer(self.main_queue_size, self.mu0_p, self.timeout)
        for total_ind in range(self.state_num):
            data_sum = 0
            n_retry_queue, n_main_queue = self._index_decomposer(total_ind)
            tail_main = tail_seq[n_main_queue]
            if n_main_queue == 0:  # queue is empty
                row_ind.append(total_ind)
                col_ind.append(self._index_composer(n_main_queue + 1, n_retry_queue))
                data_point.append(self.lambdaa)
                data_sum += self.lambdaa
                if n_retry_queue > 0:
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue, n_retry_queue - 1)
                    )
                    data_point.append(n_retry_queue * self.mu_drop_base)
                    data_sum += n_retry_queue * self.mu_drop_base
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue + 1, n_retry_queue - 1)
                    )
                    data_point.append(n_retry_queue * self.mu_retry_base)
                    data_sum += n_retry_queue * self.mu_retry_base

            elif n_main_queue == self.main_queue_size - 1:  # queue is full
                row_ind.append(total_ind)
                col_ind.append(self._index_composer(n_main_queue - 1, n_retry_queue))
                data_point.append(self.mu0_p)
                data_sum += self.mu0_p
                if n_retry_queue > 0:
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue, n_retry_queue - 1)
                    )
                    data_point.append(n_retry_queue * self.mu_drop_base)
                    data_sum += n_retry_queue * self.mu_drop_base
                if n_retry_queue < self.retry_queue_size - 1:
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue, n_retry_queue + 1)
                    )
                    data_point.append(
                        (self.lambdaa + n_retry_queue * self.mu_retry_base) * tail_main
                    )
                    data_sum += (
                        self.lambdaa + n_retry_queue * self.mu_retry_base
                    ) * tail_main
            else:  # queue is neither full nor empty
                alpha_tail_prob_sum = self.lambdaa * tail_main
                if n_retry_queue < self.retry_queue_size - 1:
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue + 1, n_retry_queue + 1)
                    )
                    data_point.append(alpha_tail_prob_sum)
                    data_sum += alpha_tail_prob_sum
                row_ind.append(total_ind)
                col_ind.append(self._index_composer(n_main_queue + 1, n_retry_queue))
                data_point.append(
                    self.lambdaa + n_retry_queue * self.mu_retry_base * tail_main
                )
                data_sum += (
                    self.lambdaa + n_retry_queue * self.mu_retry_base * tail_main
                )
                if n_retry_queue > 0:
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue + 1, n_retry_queue - 1)
                    )
                    data_point.append(
                        n_retry_queue * self.mu_retry_base * (1 - tail_main)
                    )
                    data_sum += n_retry_queue * self.mu_retry_base * (1 - tail_main)
                    row_ind.append(total_ind)
                    col_ind.append(
                        self._index_composer(n_main_queue, n_retry_queue - 1)
                    )
                    data_point.append(n_retry_queue * self.mu_drop_base)
                    data_sum += n_retry_queue * self.mu_drop_base
                row_ind.append(total_ind)
                col_ind.append(self._index_composer(n_main_queue - 1, n_retry_queue))
                data_point.append(self.mu0_p)
                data_sum += self.mu0_p
            row_ind.append(total_ind)
            col_ind.append(total_ind)
            data_point.append(-data_sum)
        return [row_ind, col_ind, data_point]

    def prob_dist_accumulator(self, pi, q1, q2, o1, o2):
        val = 0
        for q in range(q1, q2):
            for o in range(o1, o2):
                state = self._index_composer(q, o)
                val += pi[state]
        return val
