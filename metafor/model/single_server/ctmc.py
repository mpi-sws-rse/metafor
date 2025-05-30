from typing import Any, Callable, List, Tuple

import numpy as np
import numpy.typing as npt

import math
import copy
import scipy

import time
import itertools

from model.ctmc import CTMC, CTMCRepresentation, Matrix


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
            representation=CTMCRepresentation.EXPLICIT,
            retry_when_full: bool = False,
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

        # print(
        #    "main qsize = ",
        #    self.main_queue_size,
        #    " retry qsize = ",
        #    self.retry_queue_size,
        # )
        # print("lambdaas = ", self.lambdaas)
        # print("lambdaa = ", self.lambdaa)
        # print("mu = ", self.mu0_ps)

        # the rate of retrying jobs in the retry queue
        # Todo: These rates must be checked through validation data...
        self.mu_retry_base = max_retries / ((max_retries + 1) * timeout)
        # the rate of dropping jobs in the retry queue
        self.mu_drop_base = 1 / ((max_retries + 1) * timeout)

        self.thread_pool = thread_pool

        self.alpha = alpha
        self.representation = representation

        self.retry_when_full = retry_when_full
        self.Q = self.generator_mat_exact(transition_matrix=False, representation=representation,
                                          retry_when_full=retry_when_full)

        # Debug: check that each row sums to 0 (approximately)
        # np.set_printoptions(threshold=sys.maxsize)
        for row in self.Q:
            assert abs(sum(row)) < 0.0001
        # np.set_printoptions(threshold=None)

    def is_irreducible(self) -> bool:
        transition_matrix = np.logical_or(np.eye(self.state_num), self.Q.astype(bool))
        t_reach = np.linalg.matrix_power(transition_matrix, self.state_num - 1)
        return np.all(t_reach)
    
    # check if the CTMC is reversible: pi is the precomputed stationary distribution
    def check_detailed_balance(self, pi):
        n = self.Q.shape[0]
        for i in range(n):
            for j in range(i + 1, n):  # check pairs i, j
                if not np.isclose(pi[i] * Q[i, j], pi[j] * Q[j, i]):
                    return False
        return True

    def get_init_state(self) -> npt.NDArray[np.float64]:
        pi = np.zeros(self.state_num,)
        pi[0] = 1.0  # Initially the queue is empty
        return pi

    # compute the stationary distribution
    def compute_stationary_distribution(self, remove_non_negative: bool = True) -> npt.NDArray[np.float64]:
        # calculate the stationary distribution
        QT = np.transpose(self.Q)
        ns = scipy.linalg.null_space(QT)
        assert ns.shape == (self.state_num, 1)
        # print("ns = ", ns)
        if remove_non_negative:
            ns = np.array([abs(val) for val in ns])
        pi = ns / np.linalg.norm(ns, ord=1)
        if sum(pi) < -0.01:  # the null space may return `-pi`
            pi = -pi
        # due to numerical imprecision, some entries may still be negative, so we force them to be positive
        pi = abs(pi)

        # print(pi)
        # print(pi[:,0])
        # print(QT * pi[:,0])

        return pi

    @staticmethod
    def _tail_prob_computer(qsize: float, service_rate: float, timeout: float, thread_pool: float):
        """Compute the timeout probabilities for the case that service time is distributed exponentially."""

        tail_seq = [0]  # The timeout prob is zero when there is no job in the queue!
        """current_sum = 0
        last = 1
        for job_num in range(
                1, qsize
        ):  # compute the timeout prob for all different queue sizes.
            mu = min(job_num, thread_pool) * service_rate  # to remain close to the math symbol
            mu_x_timeout = mu * timeout
            exp_mu_timeout = math.exp(-mu_x_timeout)
            if exp_mu_timeout == 0:
                return [0] * qsize
            last = last * mu_x_timeout / job_num
            current_sum = current_sum + last
            tail_seq.append(current_sum * exp_mu_timeout)"""
        # exact method is unstable for large values...we overapproximate using chebyshev ineq!
        for job_num in range(1, qsize):  # compute the timeout prob for all different queue sizes.
            service_rate_effective = min(job_num, thread_pool) * service_rate
            ave = job_num / service_rate_effective
            var = job_num / (service_rate_effective**2)
            sigma = math.sqrt(var)
            if timeout - ave > sigma:
                k_inv = sigma / (timeout - ave)
                tail_seq.append(k_inv ** 2)
            else:
                tail_seq.append(1)
        return tail_seq

    def get_eigenvalues(self):
        eigenvalues = np.linalg.eigvals(self.Q)
        print(eigenvalues)
        sorted_eigenvalues = np.sort(eigenvalues.real)[::-1]
        #sorted_eigenvalues = sorted(eigenvalues, key=lambda x: np.real(x), reverse=True)
        return sorted_eigenvalues

    def get_mixing_time(self):     
        eigenvalues = self.get_eigenvalues()  # RM: we only need the first eigenvalue
        print("Sorted eigenvalues (real parts):", eigenvalues)
        t_mixing = 1 / abs(np.real(eigenvalues[1])) * math.log2(100)
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

    def generator_mat_exact(self, transition_matrix: bool = False, representation=CTMCRepresentation.EXPLICIT,
                            retry_when_full: bool = False):
        Q = Matrix(self.state_num, representation=representation)
        tail_seq = self._tail_prob_computer(self.main_queue_size, self.mu0_p, self.timeout, self.thread_pool)

        for total_ind in range(self.state_num):
            n_retry_queue, n_main_queue = self._index_decomposer(total_ind)
            tail_main = tail_seq[n_main_queue]
            if n_main_queue == 0:  # queue is empty
                Q.set(total_ind, self._index_composer(n_main_queue + 1, n_retry_queue), self.lambdaa)
                if n_retry_queue > 0:
                    Q.set(
                        total_ind, self._index_composer(n_main_queue, n_retry_queue - 1),
                        (n_retry_queue * self.mu_drop_base))
                    Q.set(
                        total_ind,
                        self._index_composer(n_main_queue + 1, n_retry_queue - 1),
                        n_retry_queue * self.mu_retry_base
                    )
            elif n_main_queue == self.main_queue_size - 1:  # queue is full
                Q.set(total_ind, self._index_composer(n_main_queue - 1, n_retry_queue),
                      min(self.main_queue_size, self.thread_pool) * self.mu0_p
                      )
                if not retry_when_full: # if retrirs aren't allowed when queue is full
                    if n_retry_queue > 0:
                        Q.set(
                            total_ind, self._index_composer(n_main_queue, n_retry_queue - 1),
                            (n_retry_queue * (self.mu_drop_base+self.mu_retry_base)))
                else: # if retrirs are allowed when queue is full
                    if n_retry_queue > 0:
                        Q.set(
                            total_ind, self._index_composer(n_main_queue, n_retry_queue - 1),
                            (n_retry_queue * self.mu_drop_base))
                    if n_retry_queue < self.retry_queue_size - 1:
                        Q.set(
                            total_ind, self._index_composer(n_main_queue, n_retry_queue + 1),
                            self.alpha
                            * (self.lambdaa + n_retry_queue * self.mu_retry_base))
                    """Q.set(total_ind, self._index_composer(n_main_queue - 1, n_retry_queue),
                          min(self.main_queue_size, self.thread_pool) * self.mu0_p
                          )
                    if n_retry_queue > 0:
                        Q.set(
                            total_ind, self._index_composer(n_main_queue, n_retry_queue - 1),
                            (n_retry_queue * self.mu_drop_base))
                    if n_retry_queue < self.retry_queue_size - 1:
                        Q.set(
                            total_ind, self._index_composer(n_main_queue, n_retry_queue + 1),
                            self.alpha
                            * (self.lambdaa + n_retry_queue * self.mu_retry_base)
                            * tail_main
                        )"""
            else:  # queue is neither full nor empty
                alpha_tail_prob_sum = self.alpha * self.lambdaa * tail_main
                if n_retry_queue < self.retry_queue_size - 1:
                    Q.set(
                        total_ind,
                        self._index_composer(n_main_queue + 1, n_retry_queue + 1),
                        alpha_tail_prob_sum)
                Q.set(total_ind, self._index_composer(n_main_queue + 1, n_retry_queue),
                      self.lambdaa + n_retry_queue * self.mu_retry_base * tail_main
                      )
                if n_retry_queue > 0:
                    Q.set(
                        total_ind,
                        self._index_composer(n_main_queue + 1, n_retry_queue - 1),
                        n_retry_queue * self.mu_retry_base * (1 - tail_main)
                    )
                    Q.set(
                        total_ind, self._index_composer(n_main_queue, n_retry_queue - 1),
                        (n_retry_queue * self.mu_drop_base))
                Q.set(total_ind, self._index_composer(n_main_queue - 1, n_retry_queue),
                      min(n_main_queue, self.thread_pool) * self.mu0_p
                      )
            if not transition_matrix:
                Q.set(total_ind, total_ind, -Q.sum(total_ind))
        return Q.matrix()

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
        for t in range(sim_step, sim_time + 1, sim_step):
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
            # print(result)
        return results

    ########### Various analyses built on top of probability distributions ###############
    def queue_full_probability(self, pi) -> float:
        qfull = self.main_queue_size - 1
        weight = 0.0
        for n_retry_queue in range(self.retry_queue_size):
            weight += pi[self._index_composer(qfull, n_retry_queue)]
        return weight

    def main_queue_size_average(self, pi) -> float:
        """This function computes the average queue length for a given prob distribution pi"""
        length = 0.0
        for n_main_queue in range(self.main_queue_size):
            weight = 0.0
            for n_retry_queue in range(self.retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            length += weight * n_main_queue
        return length

    def main_queue_size_variance(self, pi: npt.NDArray[np.float64], mean_queue_length: float) -> float:
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

    def main_queue_size_std(self, pi, mean_queue_length):
        return math.sqrt(self.main_queue_size_variance(pi, mean_queue_length))

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

    def throughput_average(self, pi, req_type: int = 0) -> float:
        arr_rate = self.lambdaas[req_type]
        proc_rate = self.mu0_ps[req_type]
        # compute the probability with which server is idle
        idle_prob = 0
        for n_retry_queue in range(self.retry_queue_size):
            idle_prob += pi[self._index_composer(0, n_retry_queue)]
        thp = proc_rate * (1 - idle_prob) * (arr_rate / self.lambdaa)
        return thp

    # req_type represents the index of the request
    def failure_rate_average(self, pi, req_type: int = 0) -> float:
        rate = 0.0
        arr_rate = self.lambdaas[req_type]
        for n_retry_queue in range(self.retry_queue_size):
            weight = 0.0
            for n_main_queue in range(self.main_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            rate += weight * self.mu_drop_base * (arr_rate / self.lambdaa) * n_retry_queue
        return rate

    def latency_average(self, pi, req_type: int = 0) -> float:
        # use the law of total expectation
        mu0_p = self.mu0_ps[req_type]
        val = 1 / mu0_p
        arr_rate = self.lambdaas[req_type]

        for n_main_queue in range(self.main_queue_size):
            weight = 0
            for n_retry_queue in range(self.retry_queue_size):
                weight += pi[self._index_composer(n_main_queue, n_retry_queue)]
            val += (
                    weight
                    * (arr_rate / self.lambdaa)
                    * n_main_queue
                    * (1 / mu0_p)
            )
        if isinstance(val, float):
            return val
        return val[0]

    def latency_variance(self, pi, mean: float, req_type: int = 0) -> float:
        mu0_p = self.mu0_ps[req_type]
        # use the law of total variance
        var1 = 1 / (mu0_p * mu0_p)  # var1 := Var(E(Y|X))
        arr_rate = self.lambdaas[req_type]
        for n_main_queue in range(self.main_queue_size):
            for n_retry_queue in range(self.retry_queue_size):
                state = self._index_composer(n_main_queue, n_retry_queue)
                weight = pi[state]
                var1 += weight * (
                        (
                                (arr_rate / self.lambdaa)
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
                    * (arr_rate / self.lambdaa)
                    * n_main_queue
                    * (1 / self.mu0_p ** 2)
            )

        var = var1 + var2
        if isinstance(var, float):
            return var
        return var[0]

    def latency_std(self, pi, mean: float, req_type: int = 0) -> float:
        return math.sqrt(self.latency_variance(pi, mean, req_type))

    def latency_percentile(self, pi, req_type: int = 0, percentile: float = 50.0):
        # BUGGY
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

    def reach_prob_computation(self, Q, sim_time: int, sim_step: int) -> List[float]:
        print("Computing the probability to reach the queue full mode")
        target_set = [self.state_num - 1]

        probabilities = []

        Q_mod = copy.copy(Q)
        init_vec = np.zeros(self.state_num)
        init_vec[target_set[0]] = 1
        # making the target states absorbing
        for state in target_set:
            Q_mod[state, :] = np.zeros(self.state_num)
        for t in range(sim_step, sim_time + 1, sim_step):
            start_time = time.time()
            reach_prob_vec = np.matmul(scipy.linalg.expm(Q_mod * t), init_vec)
            end_time = time.time()
            runtime = end_time - start_time
            print("P = %f for time bound %d" % (reach_prob_vec[0], t))
            print("Analysis time for time bound %d: %f s" % (t, runtime))
            probabilities.append(reach_prob_vec[0])
        return probabilities

    def reach_prob(self, Q, t: int, sim_time: int, sim_step: int) -> float:
        assert t % sim_step == 0
        probabilities = self.reach_prob_computation(Q, sim_time, sim_step)
        return probabilities[int(sim_time / t)]

    def prob_dist_accumulator(self, pi, q1, q2, o1, o2):
        val = 0
        for q in range(q1, q2):
            for o in range(o1, o2):
                state = self._index_composer(q, o)
                val += pi[state]
        return val

    def reach(self, start, target_set) -> float:
        Q = np.copy(self.Q)
        # vector whose target component is 0 and every other component is one
        b = -np.ones(self.state_num)
        # make the target set absorbing
        for target in target_set:
            Q[target, :] = np.zeros(self.state_num)
            Q[target, target] = 1
            b[target] = 0
        
        u = np.linalg.solve(Q, b)
        print("Reach times = ", u)
        return u[start]
    
    def time_to_drain(self) -> float:
        queue_full_state = self.state_num - 1
        queue_empty_state = 0
        return self.reach(queue_full_state, [queue_empty_state])

    def set_construction(self, q_range_list, o_range_list):
        server_no = 1 # self.server_no
        set = []
        q_range = []
        o_range = []
        for node_id in range(server_no):
            q_range.append(list(range(q_range_list[node_id][0], q_range_list[node_id][1])))
        for node_id in range(server_no):
            o_range.append(list(range(o_range_list[node_id][0], o_range_list[node_id][1])))
        q_prod_list = list(itertools.product(*q_range))
        o_prod_list = list(itertools.product(*o_range))
        for q in q_prod_list:
            for o in o_prod_list:
                state = self._index_composer(q[0], o[0])  #
                set.append(state)
        return set

    def get_hitting_time_average(self, S1, S2) -> float:
        non_target_states = list(set(list(range(0, self.state_num))).difference(set(S2)))
        non_target_state_num = len(non_target_states)
        A = np.zeros((non_target_state_num, non_target_state_num))
        b = -np.ones(non_target_state_num)

        # Fill the matrix A
        for idx, i in enumerate(non_target_states):
            A[idx, :] = self.Q[i, non_target_states]  #

        u = np.linalg.solve(A, b)
        print("Maximum error in solving the linear equation is", np.max(np.matmul(A, u) - b))
        hitting_time_min = -10
        for state in S1:
            idx = non_target_states.index(state)
            if hitting_time_min < 0:
                hitting_time_min = u[idx]
            else:
                if u[idx] < hitting_time_min:
                    hitting_time_min = u[idx]
        assert (hitting_time_min != -10), "Hitting time was not updated: is the source set S1 empty?"
        return hitting_time_min


    def get_hitting_time_average_and_deviation(self, S1, S2) -> Tuple[float, float]:
        non_target_states = list(set(list(range(0, self.state_num))).difference(set(S2)))
        non_target_state_num = len(non_target_states)
        A = np.zeros((non_target_state_num, non_target_state_num))
        b = -np.ones(non_target_state_num)

        # Fill the matrix A
        for idx, i in enumerate(non_target_states):
            A[idx, :] = self.Q[i, non_target_states]  #

        u = np.linalg.solve(A, b)
        print("Maximum error in solving the linear equation is", np.max(np.matmul(A, u) - b))

        # Now set up linear equation system for variance
        # a numerically nice way to do it is to use the following equations:
        # var(s) = 0 for all target states
        # otherwise,
        # var(s) = \sum_t P(s,t) [ var(t) + (1 + mu_t - mu_s)^2]
        # But check with Mahmoud why this code works:

        v = np.linalg.solve(A, - 2 * u)
        v = v - u * u

        hitting_time_min = -10
        for state in S1:
            idx = non_target_states.index(state)
            if hitting_time_min < 0:
                hitting_time_min = u[idx]
            else:
                if u[idx] < hitting_time_min:
                    hitting_time_min = u[idx]
        assert (hitting_time_min != -10), "Hitting time not updated: is the set of source states empty?"

        hitting_time_var = 0
        for state in S1:
            idx = non_target_states.index(state)
            if v[idx] > hitting_time_var:
                hitting_time_var = u[idx]

        return hitting_time_min, math.sqrt(hitting_time_var)
