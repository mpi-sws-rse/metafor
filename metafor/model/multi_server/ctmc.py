from typing import List, Tuple, Optional, Callable, Any

import numpy as np
import numpy.typing as npt
import math
import copy

import scipy

from model.ctmc import CTMC
from model.multi_server.generator_matrix import GeneratorMatrix
import time
from scipy.sparse.linalg import gmres, eigs
import itertools


class MultiServerCTMC(CTMC):

    def __init__(
        self,
        server_num: int,
        main_queue_sizes: List[int],
        retry_queue_sizes: List[int],
        lambdaas: List[float],
        mu0_ps: List[float],
        timeouts: List[int],
        max_retries: List[int],
        thread_pools: List[int],
        parent_list: List[List[int]],
        sub_tree_list: List[List[int]],
        q_min_list: Optional[List[int]] = None,
        q_max_list: Optional[List[int]] = None,
        o_min_list: Optional[List[int]] = None,
        o_max_list: Optional[List[int]] = None,
    ):
        super().__init__()
        assert (
            len(main_queue_sizes)
            == len(retry_queue_sizes)
            == len(lambdaas)
            == len(mu0_ps)
            == len(timeouts)
            == len(max_retries)
            == len(thread_pools)
            == server_num
        )

        self.server_num = server_num
        self.main_queue_sizes = main_queue_sizes
        self.retry_queue_sizes = retry_queue_sizes
        self.lambdaas = lambdaas
        self.mu0_ps = mu0_ps
        self.timeouts = timeouts
        self.max_retries = max_retries
        self.thread_pools = thread_pools
        self.sub_tree_list = sub_tree_list
        self.parent_list = parent_list
        self.q_min_list = q_min_list
        self.q_max_list = q_max_list
        self.o_min_list = o_min_list
        self.o_max_list = o_max_list

        self.state_num = []
        self.state_num_prod = 1
        for i in range(len(mu0_ps)):
            self.state_num.append(main_queue_sizes[i] * retry_queue_sizes[i])
            self.state_num_prod *= self.state_num[i]

        if q_min_list is None:
            self.q_min_list = [
                int(main_queue_sizes[i] * 0.9) for i in range(server_num)
            ]
        if o_min_list is None:
            self.o_min_list = [retry_queue_sizes[i] // 2 for i in range(server_num)]
        if q_max_list is None:
            self.q_max_list = [main_queue_sizes[i] * 0.1 for i in range(server_num)]
        if o_max_list is None:
            self.o_max_list = [2 for _ in range(server_num)]

        self.row_ind, self.col_ind, self.data = self.sparse_info_calculator(
            self.lambdaas, -1, [0, 0], [0, 0]
        )
        self.Q = scipy.sparse.csr_matrix(
            (self.data, (self.row_ind, self.col_ind)), shape=(self.state_num_prod, self.state_num_prod)
        )

        def matvec_func(x):
            return self.Q.T.dot(x)

        def rmatvec_func(x):
            return self.Q.dot(x)

        self.Q_op = GeneratorMatrix(
            shape=(self.state_num_prod, self.state_num_prod),
            matvec=rmatvec_func,
            rmatvec=matvec_func,
            dtype=self.Q.dtype,
        )

        self.Q_op_T = GeneratorMatrix(
            shape=(self.state_num_prod, self.state_num_prod),
            matvec=matvec_func,
            rmatvec=rmatvec_func,
            dtype=self.Q.dtype,
        )

    def _index_decomposer(self, total_ind) -> List[List[int]]:
        """This function converts a given index in range [0, state_num]
        into two indices corresponding to (1) number of jobs in orbit and (2) jobs in the queue.
        """

        n_main_queue = []
        n_retry_queue = []
        for node_id in range(self.server_num):
            ancestors = []
            ss_res = 1
            ancestor = self.parent_list[node_id]
            while len(ancestor) != 0:
                ss_res *= self.state_num[ancestor[0]]
                ancestors.append(ancestor[0])
                ancestor = self.parent_list[ancestor[0]]

            s = (total_ind // ss_res) % self.state_num[
                node_id
            ]  # state index within SS of node_id
            n_main_queue.append(s % self.main_queue_sizes[node_id])
            n_retry_queue.append(s // self.main_queue_sizes[node_id])

        for node_id in range(self.server_num):
            assert 0 <= n_main_queue[node_id] < self.state_num[node_id]
            assert 0 <= n_retry_queue[node_id] < self.state_num[node_id]
        return [n_main_queue, n_retry_queue]

    def _index_composer(self, n_main_queue_list, n_retry_queue_list) -> int:
        """This function converts two given input indices into one universal index in range [0, state_num].
        The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit.
        """

        total_ind = (
            n_main_queue_list[0] + n_retry_queue_list[0] * self.main_queue_sizes[0]
        )
        for node_id in range(1, self.server_num):
            ss_size_bias = 1
            for i in range(node_id):
                ss_size_bias *= self.state_num[i]
            total_ind += (
                n_main_queue_list[node_id]
                + n_retry_queue_list[node_id] * self.main_queue_sizes[node_id]
            ) * ss_size_bias
        assert 0 <= total_ind < self.state_num_prod
        return total_ind

    def _tail_prob_computer(self, total_ind):
        """This function computes the timeout probabilities for the case
        that service time is distributed exponentially."""

        [q_list, _] = self._index_decomposer(total_ind)
        tail_prob = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            ave = 0
            var = 0
            sub_tree = self.sub_tree_list[node_id]
            for downstream_node_id in sub_tree:
                # effective_mu = self.effective_mu(node_id, 1, q_list, o_list) # to generalize, code must be modified...
                ave += q_list[downstream_node_id] / self.mu0_ps[downstream_node_id]
                var += (
                    q_list[downstream_node_id]
                    * 1
                    / (self.mu0_ps[downstream_node_id] ** 2)
                )
            sigma = math.sqrt(var)
            if self.timeouts[node_id] - ave > sigma:
                k_inv = sigma / (self.timeouts[node_id] - ave)
                tail_prob[node_id] = k_inv**2
            else:
                tail_prob[node_id] = 1
        for node_id in range(self.server_num):
            assert 0 <= tail_prob[node_id] <= 1
        return tail_prob


    def kmeans_manual(self, vector, k=2, max_iters=100):
        # Step 1: Randomly initialize centroids
        vector = vector.squeeze()
        centroids = np.random.choice(vector, k)

        for _ in range(max_iters):
            # Step 2: Assign points to the nearest centroid
            distances = np.abs(vector[:, np.newaxis] - centroids)
            labels = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.array([vector[labels == i].mean() for i in range(k)])

            # Check for convergence
            if np.all(centroids == new_centroids):
                break

            centroids = new_centroids

        return labels, centroids

    def cumulative_prob_computer(self, pi, q_range, o_range):
        """To compute the probability mass of set of states with queue length between q_min and q_max,
        and orbit length between o_min and o_max"""

        q_local_list = []
        o_local_list = []
        for node_id in range(self.server_num):
            q_local_list.append(list(range(q_range[node_id][0], q_range[node_id][1])))
            o_local_list.append(list(range(o_range[node_id][0], o_range[node_id][1])))
        q_prod_list = list(itertools.product(*q_local_list))
        o_prod_list = list(itertools.product(*o_local_list))
        prod_state_list = []
        for q in q_prod_list:
            for o in o_prod_list:
                prod_state_list.append(self._index_composer(q, o))
        cumulative_prob = 0
        for state in prod_state_list:
            cumulative_prob += pi[state]
        return cumulative_prob

    def finite_time_analysis(self, pi0, analyses: dict[str, Callable[[Any], Any]], sim_time: int, sim_step: int):
        initial_result = {n: 0.0 for n in analyses}
        # XXX: we are assuming the initial result for the analyses is 0, which may not be true
        initial_result["pi"] = pi0
        results = {0: initial_result}
        print(
            "Computing finite time statistics for time quantum %d time units with step size %d"
            % (sim_time, sim_step)
        )
        piq = pi0
        for t in range(sim_step, sim_time + 1, sim_step):
            result = {}
            start = time.time()
            piq = scipy.sparse.linalg.expm_multiply(self.Q_op_T, piq)
            elapsed_time = time.time() - start
            result["step"] = t
            result["pi"] = piq
            result["wallclock_time"] = elapsed_time
            # now run all the analyses passed to this function with the current distribution
            for analysis_name, analysis_fn in analyses.items():
                v = analysis_fn(piq)
                result[analysis_name] = v

            results[t] = result
        return results

    def main_queue_size_average(self, pi: npt.NDArray[np.float64]) -> List[float]:
        q_len = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            q_len_node = 0
            for q_node in range(self.main_queue_sizes[node_id]):
                q_range = []
                o_range = []
                for server_id in range(self.server_num):
                    if server_id != node_id:
                        q_range.append(list(range(self.main_queue_sizes[server_id])))
                    else:
                        q_range.append([q_node])
                for server_id in range(self.server_num):
                    o_range.append(list(range(self.retry_queue_sizes[server_id])))
                q_prod_list = list(itertools.product(*q_range))
                o_prod_list = list(itertools.product(*o_range))
                p = 0
                for q in q_prod_list:
                    for o in o_prod_list:
                        p += pi[self._index_composer(q, o)]
                q_len_node += q_node * p
            q_len[node_id] = q_len_node
        return q_len

    def main_queue_size_variance(self, pi: npt.NDArray[np.float64], mean_queue_length: List[float]) -> List[float]:
        q_var = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            q_var_node = 0
            for q_node in range(self.main_queue_sizes[node_id]):
                q_range = []
                o_range = []
                for server_id in range(self.server_num):
                    if server_id != node_id:
                        q_range.append(list(range(self.main_queue_sizes[server_id])))
                    else:
                        q_range.append([q_node])
                    o_range.append(list(range(self.retry_queue_sizes[server_id])))
                q_prod_list = list(itertools.product(*q_range))
                o_prod_list = list(itertools.product(*o_range))
                p = 0
                for q in q_prod_list:
                    for o in o_prod_list:
                        p += pi[self._index_composer(q, o)]
                q_var_node += p * (q_node - mean_queue_length[node_id]) * (q_node - mean_queue_length[node_id])
            q_var[node_id] = q_var_node
        return q_var

    def main_queue_size_std(self, pi, mean_queue_length):
        variances = self.main_queue_size_variance(pi, mean_queue_length)
        return [math.sqrt(variance) for variance in variances]

    def retry_queue_size_average(self, pi: npt.NDArray[np.float64]) -> List[float]:
        q_len = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            q_len_node = 0
            for q_node in range(self.retry_queue_sizes[node_id]):
                q_range = []
                o_range = []
                for server_id in range(self.server_num):
                    q_range.append(list(range(self.main_queue_sizes[server_id])))
                    if server_id != node_id:
                        o_range.append(list(range(self.retry_queue_sizes[server_id])))
                    else:
                        o_range.append([q_node])
                q_prod_list = list(itertools.product(*q_range))
                o_prod_list = list(itertools.product(*o_range))
                p = 0
                for q in q_prod_list:
                    for o in o_prod_list:
                        p += pi[self._index_composer(q, o)]
                q_len_node += q_node * p
            q_len[node_id] = q_len_node
        return q_len

    def sparse_info_calculator_CTMC(self, lambda_list, node_selected, q_range, o_range):
        state_num = self.state_num_prod
        server_no = self.server_no
        parent_list = self.parent_list
        mu0_p = self.mu0_ps
        timeout = self.timeout
        max_retries = self.max_retries
        main_queue_size = self.main_queue_size
        retry_queue_size = self.retry_queue_size
        num_threads = self.thread_pools
        row_ind = []
        col_ind = []
        data = []
        # val_sum_col = np.zeros(state_num)
        # val_sum_row = np.zeros(state_num)
        for total_ind in range(state_num):
            start = time.time()
            q, o = self._index_decomposer(total_ind)
            absorbing_flg = False
            for node_id in range(server_no):
                if node_id == node_selected:
                    if q[node_id] < q_range[1] and q[node_id] >= q_range[0] and o[node_id] < o_range[1] and o[
                        node_id] >= o_range[0]:
                        absorbing_flg = True

            if absorbing_flg:
                do_nothing = True
                row_ind.append(total_ind)
                col_ind.append(total_ind)
                data.append(1)

            else:
                val_sum_row = 0
                q_next = [0 * i for i in range(server_no)]
                o_next = [0 * i for i in range(server_no)]
                # compute the non-synchronized transitions' rates of the generator matrix
                for node_id in range(server_no):
                    q_next[:] = q
                    o_next[:] = o
                    # Setting the rates related to job arrivals
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            q_next[node_id] = q[node_id] + i
                            o_next[node_id] = o[node_id] + j
                            total_ind_next = self._index_composer(q_next, o_next)
                            skip_flg = False
                            """for node_id in range(server_no):
                                if node_id == node_selected:
                                    if q_next[node_id] <= q_range[1] and q_next[node_id] >= q_range[0] and o_next[node_id] <= o_range[1] and o_next[node_id] >= o_range[0]:
                                        skip_flg = True"""
                            if skip_flg == False and min(q) >= 0 and min(o) >= 0 and min(q_next) >= 0 and min(o_next) >= 0:
                                if ((np.array(q) - np.array(main_queue_size)) < 0).all() and (
                                        (np.array(o) - np.array(retry_queue_size)) < 0).all() and (
                                        (np.array(q_next) - np.array(main_queue_size)) < 0).all() and (
                                        (np.array(o_next) - np.array(retry_queue_size)) < 0).all():
                                    if (i != 0 or j != 0) and (total_ind < state_num) and (total_ind_next < state_num):
                                        break_flg = False
                                        for node in range(server_no):
                                            if node == node_id:
                                                do_nothing = True
                                            else:
                                                if q[node] != q_next[node] or o[node] != o_next[node]:
                                                    break_flg = True
                                        if break_flg == False:
                                            val_forw = self.forward_trans_computer(lambda_list, q, o, q_next, o_next,
                                                                                   node_id)
                                            val_back = val_forw # self.forward_trans_computer(lambda_list, q_next, o_next, q, o,
                                                                             #      node_id)
                                            val = (val_forw + val_back) / 2
                                            row_ind.append(total_ind)
                                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                                            data.append(val)
                                            val_sum_row += val
                            q_next[:] = q
                            o_next[:] = o
                val = - val_sum_row
                row_ind.append(total_ind)
                col_ind.append(total_ind)
                data.append(val)
        return [row_ind, col_ind, data]

    def forward_trans_computer(self, lambda_list, q, o, q_next, o_next, node_id):
        state_num = self.state_num_prod
        server_no = self.server_no
        parent_list = self.parent_list
        mu0_p = self.mu0_ps
        timeout = self.timeouts
        max_retries = self.max_retries
        main_queue_size = self.main_queue_sizes
        retry_queue_size = self.retry_queue_sizes
        num_threads = self.thread_pools
        tail_prob_list = self._tail_prob_computer(self._index_composer(q, o))
        mu_drop_base = 1 / (timeout[node_id] * (max_retries[node_id] + 1))
        mu_retry_base = max_retries[node_id] / (timeout[node_id] * (max_retries[node_id] + 1))
        # Check which arrival source is active for the selected node_id
        """lambdaa = lambda_list[node_id]
        if parent_list[node_id] == []:  # if there exists only a local source of job arrival
            lambda_summed = lambdaa
        else:  # if there exists local and non-local sources of job arrival
            num_jobs_upstream = min(q[parent_list[node_id][0]], num_threads[parent_list[node_id][0]])
            lambda_summed = lambdaa + num_jobs_upstream * mu0_p[parent_list[node_id][0]]"""
        lambda_summed = self.effective_lambda(node_id, 1, lambda_list, q, o)
        if q_next[node_id] == q[node_id] + 1 and o_next[node_id] == o[node_id] + 1:
            rate = lambda_summed * tail_prob_list[node_id]
        elif q_next[node_id] == q[node_id] + 1 and o_next[node_id] == o[node_id]:
            rate = lambda_summed * (1 - tail_prob_list[node_id]) + mu_retry_base * tail_prob_list[node_id] * o[node_id]
        elif q_next[node_id] == q[node_id] + 1 and o_next[node_id] == o[node_id] - 1:
            rate = mu_retry_base * (1 - tail_prob_list[node_id]) * o[node_id]
        elif q_next[node_id] == q[node_id] and o_next[node_id] == o[node_id] - 1:
            rate = mu_drop_base * o[node_id]
        elif q_next[node_id] == q[node_id] - 1 and o_next[node_id] == o[node_id]:
            # assuming that the system is closed
            """if self.sub_tree_list[node_id] == [node_id]:
                rate = mu0_p[node_id] * min(q[node_id], num_threads[node_id])
            else:
                for node in self.sub_tree_list[node_id]:
                    if node == node_id:
                        rate = mu0_p[node_id] * min(q[node_id], num_threads[node_id])
                    else:
                        rate = min(rate, mu0_p[node] * min(q[node], num_threads[node]))"""
            rate = self.effective_mu(node_id, 1, q, o)
        else:
            rate = 0
        return rate

    def effective_mu(self, node_id, closed, q, o):
        if closed == True:  # if the system is closed
            if self.sub_tree_list[node_id] == [node_id]:
                rate = self.mu0_ps[node_id] * min(q[node_id], self.thread_pools[node_id])
            else:
                for node in self.sub_tree_list[node_id]:  #
                    if node == node_id:
                        rate = self.mu0_p[node_id] * min(q[node_id], self.thread_pools[node_id])
                    else:
                        rate = min(rate, self.mu0_p[node] * min(q[node], self.thread_pools[node]))
        else:  # if the system is open
            rate = mu0_p[node_id] * min(q[node_id], self.thread_pools[node_id])
        return rate

    def effective_lambda(self, node_id, closed, lambda_list, q, o):
        rate = lambda_list[node_id]
        # finding the set of ancestors; must be changed when there are multiple branches!
        ancestors = []
        ancestor = self.parent_list[node_id]
        while ancestor != []:
            ancestors.append(ancestor[0])
            ancestor = self.parent_list[ancestor[0]]
        added_lambda = 0
        for node in reversed(ancestors):
            num_jobs_upstream = min(q[node], self.thread_pools[node])
            mu_retry_base_node = self.max_retries[node] / (self.timeout[node] * (self.max_retries[node] + 1))
            effective_arr_rate_node = added_lambda + lambda_list[node] + mu_retry_base_node * o[node]
            effective_proc_rate_node = self.effective_mu(node, closed, q, o)
            added_lambda += min(effective_arr_rate_node, effective_proc_rate_node)
        rate += added_lambda
        return rate

    

    def get_init_state(self) -> npt.NDArray[np.float64]:
        pi = np.zeros(self.state_num_prod)
        pi[0] = 1.0  # Initially the queue is empty
        return pi

    def compute_stationary_distribution(self, remove_non_negative: bool = True) -> npt.NDArray[np.float64]:
        start = time.time()
        _, eigenvectors = eigs(self.Q_op_T, k=1, which="SM")
        if remove_non_negative:
            eigenvectors = np.array([abs(val) for val in eigenvectors])
        pi = np.real(eigenvectors) / np.linalg.norm(np.real(eigenvectors), ord=1)
        if pi[0] < -0.00000001:
            pi = -pi
        print("Computing the stationary distribution took ", time.time() - start, " seconds")
        assert 0.99999999 <= sum(pi) <= 1.00000001
        for prob in pi:
            assert 0 <= prob <= 1
        return pi

    def hitting_time_average(self, Q, S1, S2) -> float:
        A = copy.deepcopy(Q)
        b = -np.ones(self.state_num)
        b[S2] = 0

        def matvec_func(x):
            return A.dot(x)

        def rmatvec_func(x):
            return A.T.dot(x)

        A_op = GeneratorMatrix(
            shape=(self.state_num, self.state_num),
            matvec=matvec_func,
            rmatvec=rmatvec_func,
            dtype=A.dtype,
        )
        start = time.time()
        u, info = gmres(A_op, b, rtol=1e-03)
        print(time.time() - start)
        hitting_time_min = -10
        for state in S1:
            if hitting_time_min < 0:
                hitting_time_min = u[state]
            else:
                if u[state] < hitting_time_min:
                    if u[state] == -1:
                        sss = 1
                    hitting_time_min = u[state]
        return hitting_time_min

    def set_construction(
        self, q_min1, q_min2, q_max1, q_max2, o_min1, o_min2, o_max1, o_max2
    ):
        new_set = []
        for q1 in range(q_min1, q_max1):
            for q2 in range(q_min2, q_max2):
                for o1 in range(o_min1, o_max1):
                    for o2 in range(o_min2, o_max2):
                        state = self._index_composer([q1, q2, 0], [o1, o2, 0])
                        new_set.append(state)
        return new_set

    def latency_average(self, pi: npt.NDArray[np.float64], req_type: int = 0):
        # TODO: implement this
        return 0
