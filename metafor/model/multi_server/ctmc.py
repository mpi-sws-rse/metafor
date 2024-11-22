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

        self.row_ind, self.col_ind, self.data = self.sparse_info_calculator_CTMC(
            self.lambdaas, self.mu0_ps, self.timeouts, self.main_queue_sizes, self.retry_queue_sizes, -1, [0, 0], [0, 0]
        )

        self.Q = scipy.sparse.csr_matrix(
            (self.data, (self.row_ind, self.col_ind)), shape=(self.state_num_prod, self.state_num_prod)
        )

        def matvec_func(x):
            return self.Q.dot(x)

        def rmatvec_func(x):
            return self.Q.T.dot(x)

        self.Q_op = GeneratorMatrix(
            shape=(self.state_num_prod, self.state_num_prod),
            matvec=matvec_func,
            rmatvec=rmatvec_func,
            dtype=self.Q.dtype,
        )

        self.Q_op_T = GeneratorMatrix(
            shape=(self.state_num_prod, self.state_num_prod),
            matvec=matvec_func,
            rmatvec=rmatvec_func,
            dtype=self.Q.dtype,
        )

    def _index_decomposer(self, total_ind, main_queue_sizes, retry_queue_sizes) -> List[List[int]]:
        """This function converts a given index in range [0, state_num]
        into two indices corresponding to (1) number of jobs in orbit and (2) jobs in the queue.
        """
        server_no = self.server_no
        state_num = []
        state_num_prod = 1
        for i in range(server_no):
            state_num.append(main_queue_sizes[i] * retry_queue_sizes[i])
            state_num_prod *= state_num[i]
        n_main_queue = []
        n_retry_queue = []
        for node_id in range(self.server_num):
            ancestors = []
            ss_res = 1
            ancestor = self.parent_list[node_id]
            while len(ancestor) != 0:
                ss_res *= state_num[ancestor[0]]
                ancestors.append(ancestor[0])
                ancestor = self.parent_list[ancestor[0]]

            s = (total_ind // ss_res) % state_num[
                node_id
            ]  # state index within SS of node_id
            n_main_queue.append(s % main_queue_sizes[node_id])
            n_retry_queue.append(s // main_queue_sizes[node_id])

        for node_id in range(self.server_num):
            assert 0 <= n_main_queue[node_id] < state_num[node_id]
            assert 0 <= n_retry_queue[node_id] < state_num[node_id]
        return [n_main_queue, n_retry_queue]

    def _index_composer(self, n_main_queue_list, n_retry_queue_list, main_queue_sizes, retry_queue_sizes) -> int:
        """This function converts two given input indices into one universal index in range [0, state_num].
        The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit.
        """
        state_num = []
        state_num_prod = 1
        for i in range(self.server_no):
            state_num.append(main_queue_sizes[i] * retry_queue_sizes[i])
            state_num_prod *= state_num[i]
        total_ind = (
            n_main_queue_list[0] + n_retry_queue_list[0] * main_queue_sizes[0]
        )
        for node_id in range(1, self.server_num):
            ss_size_bias = 1
            for i in range(node_id):
                ss_size_bias *= state_num[i]
            total_ind += (
                n_main_queue_list[node_id]
                + n_retry_queue_list[node_id] * self.main_queue_sizes[node_id]
            ) * ss_size_bias
        assert 0 <= total_ind < state_num_prod
        return total_ind

    def _tail_prob_computer(self, total_ind, mu0_ps, timeouts, main_queue_sizes, retry_queue_sizes):
        """This function computes the timeout probabilities for the case
        that service time is distributed exponentially."""

        [q_list, o_list] = self._index_decomposer(total_ind, main_queue_sizes, retry_queue_sizes)
        tail_prob = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            ave = 0
            var = 0
            sub_tree = self.sub_tree_list[node_id]
            for downstream_node_id in sub_tree:
                # effective_mu = self.effective_mu(node_id, 1, q_list, o_list) # to generalize, code must be modified...
                ave += q_list[downstream_node_id] / mu0_ps[downstream_node_id]
                var += (
                    q_list[downstream_node_id]
                    * 1
                    / (self.mu0_ps[downstream_node_id] ** 2)
                )
            sigma = math.sqrt(var)
            if timeouts[node_id] - ave > sigma:
                k_inv = sigma / (timeouts[node_id] - ave)
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

    def cumulative_prob_computer(self, pi_vec, q_range, o_range, main_queue_size, retry_queue_size):
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
                prod_state_list.append(self._index_composer(q, o, main_queue_size, retry_queue_size))
        cumulative_prob = 0
        for state in prod_state_list:
            cumulative_prob += pi[state]
        return cumulative_prob

    # The following requires Q_op_T
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

    def main_queue_average_size(self, pi: npt.NDArray[np.float64], main_queue_size, retry_queue_size) -> List[float]:
        q_len = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            q_len_node = 0
            for q_node in range(main_queue_sizes[node_id]):
                q_range = []
                o_range = []
                for server_id in range(self.server_num):
                    if server_id != node_id:
                        q_range.append(list(range(main_queue_sizes[server_id])))
                    else:
                        q_range.append([q_node])
                for server_id in range(self.server_num):
                    o_range.append(list(range(retry_queue_sizes[server_id])))
                q_prod_list = list(itertools.product(*q_range))
                o_prod_list = list(itertools.product(*o_range))
                p = 0
                for q in q_prod_list:
                    for o in o_prod_list:
                        p += pi[self._index_composer(q, o)]
                q_len_node += q_node * p
            q_len[node_id] = q_len_node
        return q_len

    def orbit_average_size(self, pi_vec, main_queue_sizes, retry_queue_sizes):
        server_no = self.server_no
        o_len = [0 for i in range(server_no)]
        for node_id in range(server_no):
            o_len_node = 0
            for o_node in range(main_queue_sizes[node_id]):
                q_range = []
                o_range = []
                for id in range(server_no):
                    if id != node_id:
                        o_range.append(list(range(retry_queue_sizes[id])))
                    else:
                        o_range.append([o_node])
                for id in range(server_no):
                    q_range.append(list(range(main_queue_sizes[id])))
                q_prod_list = list(itertools.product(*q_range))
                o_prod_list = list(itertools.product(*o_range))
                p = 0
                for q in q_prod_list:
                    for o in o_prod_list:
                        p += pi_vec[self.index_composer(q, o, main_queue_sizes, retry_queue_sizes)]
                o_len_node += o_node * p
            o_len[node_id] = o_len_node
        return o_len

    # must be modified
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

    # must be modified
    def main_queue_size_std(self, pi, mean_queue_length):
        variances = self.main_queue_size_variance(pi, mean_queue_length)
        return [math.sqrt(variance) for variance in variances]


    def sparse_info_calculator_CTMC(self, lambda_list, mu0_p, timeout, main_queue_size,
                                    retry_queue_size, node_selected, q_range, o_range):
        server_no = self.server_no
        state_num = []
        state_num_prod = 1
        for i in range(server_no):
            state_num.append(main_queue_size[i] * retry_queue_size[i])
            state_num_prod *= state_num[i]
        parent_list = self.parent_list
        max_retries = self.max_retries
        num_threads = self.thread_pool
        row_ind = []
        col_ind = []
        data = []
        for total_ind in range(state_num_prod):
            start = time.time()
            q, o = self._index_decomposer(total_ind, main_queue_size, retry_queue_size)
            # check if total_ind corresponds to a target state for coputing hitting time
            # absorbing_flg is only used for the case we want to compute the CTMC with target states being absorbing.
            absorbing_flg = False
            for node_id in range(server_no):
                if node_id == node_selected:
                    if q[node_id] < q_range[1] and q[node_id] >= q_range[0] and o[node_id] < o_range[1] and o[
                        node_id] >= o_range[0]:
                        absorbing_flg = True
            if absorbing_flg:
                row_ind.append(total_ind)
                col_ind.append(total_ind)
                data.append(1)

            else: # if total_ind doesn't correspond to a target state
                val_sum_row = 0
                q_next = [0 * i for i in range(server_no)]
                o_next = [0 * i for i in range(server_no)]
                # compute the non-synchronized transitions' rates of the generator matrix
                for node_id in range(server_no):
                    q_next[:] = q
                    o_next[:] = o
                    # only transition to the neighboring states is possible --> i,j \in [-1;1]
                    for i in range(-1, 2):
                        for j in range(-1, 2):
                            q_next[node_id] = q[node_id] + i
                            o_next[node_id] = o[node_id] + j
                            # check for the invalid pairs of (q,o) & (q_next, o_next)
                            if min(q) >= 0 and min(o) >= 0 and min(q_next) >= 0 and min(o_next) >= 0:
                                if ((np.array(q) - np.array(main_queue_size)) < 0).all() and (
                                        (np.array(o) - np.array(retry_queue_size)) < 0).all() and (
                                        # let q_next & o_next go beyond the limits by one!
                                        (np.array(q_next) - np.array(main_queue_size)) <= 0).all() and (
                                        (np.array(o_next) - np.array(retry_queue_size)) <= 0).all():
                                    # check if at most one node's state gets updated
                                    if (i != 0 or j != 0): # and (total_ind < state_num) and (total_ind_next < state_num):
                                        break_flg = False
                                        for node in range(server_no):
                                            if node == node_id:
                                                do_nothing = True
                                            else:
                                                if q[node] != q_next[node] or o[node] != o_next[node]:
                                                    break_flg = True
                                                    break
                                    # exclude non-existing transitions
                                    if [i, j] in [[0, 0], [-1, -1], [-1, 1], [0, 1]]:
                                        break_flg = True
                                    if break_flg == False:
                                        val_forw, total_ind_next = self.forward_trans_computer(lambda_list, mu0_p,
                                                                                               timeout, main_queue_size,
                                                                                               retry_queue_size, q, o,
                                                                                               q_next, o_next, node_id)
                                        if total_ind_next != []:
                                            val = val_forw
                                            row_ind.append(total_ind)
                                            col_ind.append(total_ind_next)
                                            data.append(val)
                                            val_sum_row += val
                            q_next[:] = q
                            o_next[:] = o
                val = - val_sum_row
                row_ind.append(total_ind)
                col_ind.append(total_ind)
                data.append(val)
        return [row_ind, col_ind, data]

    def forward_trans_computer(self, lambda_list, mu0_p, timeout, main_queue_size, retry_queue_size,
                               q, o, q_next, o_next, node_id):
        server_no = self.server_no
        state_num = []
        state_num_prod = 1
        for i in range(server_no):
            state_num.append(main_queue_size[i] * retry_queue_size[i])
            state_num_prod *= state_num[i]
        parent_list = self.parent_list
        max_retries = self.max_retries
        main_queue_size = self.main_queue_sizes
        retry_queue_size = self.retry_queue_sizes
        num_threads = self.thread_pool
        tail_prob_list = self._tail_prob_computer(self._index_composer(q, o))
        mu_drop_base = 1 / (timeout[node_id] * (max_retries[node_id] + 1))
        mu_retry_base = max_retries[node_id] / (timeout[node_id] * (max_retries[node_id] + 1))
        lambda_summed = self.effective_lambda(node_id, timeout, mu0_p, 1, lambda_list, q, o)
        q_next_mod = copy.deepcopy(q_next)  # will be used to analyze entries over the borders
        o_next_mod = copy.deepcopy(o_next)  # will be used to analyze entries over the borders
        if q_next[node_id] == q[node_id] + 1 and o_next[node_id] == o[node_id] + 1:
            if q[node_id] == main_queue_size[node_id] - 1:  # if queue was full
                if o[node_id] == retry_queue_size[node_id] - 1:  # if orbit is full as well
                    rate = 0  # will not be considered
                    total_ind_next = []  # will not be considered
                else:
                    q_next_mod[node_id] = q[node_id]  # since there won't be any possibility to add jobs to the queue
                    rate = lambda_summed * 1  # multiplier is changed to one
                    total_ind_next = self.index_composer(q_next_mod, o_next, main_queue_size, retry_queue_size)
            elif o[node_id] == retry_queue_size[node_id] - 1:  # queue isn't full, whereas orbit is full
                o_next_mod[node_id] = o[node_id]  # since there won't be any possibility to add jobs to the queue
                rate = lambda_summed * 1  # multiplier is changed to one
                total_ind_next = self.index_composer(q_next, o_next_mod, main_queue_size, retry_queue_size)
            else:
                rate = lambda_summed * tail_prob_list[node_id]
                total_ind_next = self.index_composer(q_next, o_next, main_queue_size, retry_queue_size)
        elif q_next[node_id] == q[node_id] + 1 and o_next[node_id] == o[node_id]:
            if q[node_id] == main_queue_size[node_id] - 1:  # if queue was full
                rate = 0  # will not be considered-->the effect was considered above
                total_ind_next = []  # will not be considered-->the effect was considered above
            else:  # if queue isn't full
                rate = lambda_summed * (1 - tail_prob_list[node_id]) + mu_retry_base * tail_prob_list[node_id] * o[
                    node_id]
                total_ind_next = self.index_composer(q_next, o_next, main_queue_size, retry_queue_size)
        elif q_next[node_id] == q[node_id] + 1 and o_next[node_id] == o[node_id] - 1:
            if q[node_id] != main_queue_size[node_id] - 1:  # if queue isn't full
                rate = mu_retry_base * (1 - tail_prob_list[node_id]) * o[node_id]
                total_ind_next = self.index_composer(q_next, o_next, main_queue_size, retry_queue_size)
            else:
                rate = 0  # will not be considered
                total_ind_next = []  # will not be considered
        elif q_next[node_id] == q[node_id] and o_next[node_id] == o[node_id] - 1:
            rate = mu_drop_base * o[node_id]
            total_ind_next = self.index_composer(q_next, o_next, main_queue_size, retry_queue_size)
        elif q_next[node_id] == q[node_id] - 1 and o_next[node_id] == o[node_id]:
            # assuming that the system is closed
            rate = self.effective_mu(node_id, mu0_p, 1, q, o)
            total_ind_next = self.index_composer(q_next, o_next, main_queue_size, retry_queue_size)
        else:
            print("why did I end up here?!")
            assert 0 <= total_ind_next < state_num_prod
        return rate, total_ind_next

    def effective_mu(self, node_id, mu0_p, closed, q, o):
        if closed == True:  # if the system is closed
            if self.sub_tree_list[node_id] == [node_id]:
                rate = mu0_p[node_id] * min(q[node_id], self.num_threads[node_id])
            else:
                for node in self.sub_tree_list[node_id]:  #
                    if node == node_id:
                        rate = mu0_p[node_id] * min(q[node_id], self.num_threads[node_id])
                    else:
                        # +1 is considered to avoid being stuck for all-ampty initialization
                        rate = min(rate, mu0_p[node] * min(q[node] + 1, self.num_threads[node]))
        else:  # if the system is open
            rate = mu0_p[node_id] * min(q[node_id], self.num_threads[node_id])
        return rate

    def effective_lambda(self, node_id, timeout, mu0_p, closed, lambda_list, q, o):
        rate = lambda_list[node_id]
        # finding the set of ancestors; must be changed when there are multiple branches!
        ancestors = []
        ancestor = self.parent_list[node_id]
        while ancestor != []:
            ancestors.append(ancestor[0])
            ancestor = self.parent_list[ancestor[0]]
        added_lambda = 0
        for node in reversed(ancestors):
            effective_arr_rate_node = added_lambda + lambda_list[node]  # + mu_retry_base_node * o[node]
            effective_proc_rate_node = self.effective_mu(node, mu0_p, closed, q, o)
            added_lambda = min(effective_arr_rate_node, effective_proc_rate_node)
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
        # only correct for the default setting used for fault scenario simulation!
        state_num = self.state_num_prod
        A = copy.deepcopy(Q)
        b = -np.ones(state_num)
        b[S2] = 0

        def matvec_func(x):
            return A.dot(x)

        def rmatvec_func(x):
            return A.T.dot(x)

        A_op = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func,
                               dtype=A.dtype)
        u, info = gmres(A_op, b)
        hitting_time_min = -10
        for state in S1:
            if hitting_time_min < 0:
                hitting_time_min = u[state]
            else:
                if u[state] < hitting_time_min:
                    hitting_time_min = u[state]
        return hitting_time_min

    def set_construction(self, q_range_list, o_range_list, main_queue_size, retry_queue_size):
        server_no = self.server_no
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
                state = self.index_composer(q, o, main_queue_size, retry_queue_size)
                set.append(state)
        return set

    def latency_average(self, pi: npt.NDArray[np.float64], req_type: int = 0):
        # TODO: implement this
        return 0

    def plot_mixing_time(self, step_size: float, input_seq: List[float], output_seq,
                         x_axis: str, y_axis: str, figure_name: str, color: str):
        plt.rc('font', size=14)
        plt.rcParams["figure.figsize"] = [5, 5]
        plt.rcParams["figure.autolayout"] = True

        plt.figure()  # row 0, col 0

        plt.plot(input_seq, output_seq, color=color)
        plt.xlabel(x_axis, fontsize=14)
        plt.ylabel(y_axis, fontsize=14)
        plt.grid('on')
        plt.xlim(min(input_seq), max(input_seq))

        plt.savefig(figure_name)
        plt.close()



    def mixing_time_vs_qsize_data_generator(self, lambda_config, mu0_p, timeout, retry_queue_size, lb, hb, step):
        server_no = self.server_no
        qsize_seq = []
        mixing_time_seq = []
        for size in range(lb, hb, step):
            qsize_seq.append(size)
            qsize = [size for i in range(server_no)]
            state_num = []
            state_num_prod = 1
            for i in range(server_no):
                state_num.append(qsize[i] * retry_queue_size[i])
                state_num_prod *= state_num[i]
            # Computing the generator matrix
            print("computing the generator matrix for queue size", size)
            start = time.time()
            row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, mu0_p, timeout, qsize,
                                                                      retry_queue_size, -1, [0, 0], [0, 0])
            Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num_prod, state_num_prod))

            def matvec_func(x):
                return Q.dot(x)

            def rmatvec_func(x):
                return Q.T.dot(x)

            Q_op = GeneratorMatrix(shape=(state_num_prod, state_num_prod), matvec=matvec_func, rmatvec=rmatvec_func,
                                   dtype=Q.dtype)
            Q_op_t = Q_op.T
            # compute eigen space for the generator matrix
            print("Compute eigenspace")
            start = time.time()
            eigvals, eigvecs = eigs(Q_op_t, k=2, which='SM')
            print("eigenspace is computed in", time.time() - start)
            print("second largest eigenvalue is", np.min(eigvals))
            mixing_time = abs(math.log2(.1) / np.min(np.real(eigvals)))
            mixing_time_seq.append(mixing_time)
        return qsize_seq, mixing_time_seq

    def mixing_time_vs_osize_data_generator(self, lambda_config, mu0_p, timeout, main_queue_size, lb, hb, step):
        server_no = self.server_no
        osize_seq = []
        mixing_time_seq = []
        for size in range(lb, hb, step):
            osize_seq.append(size)
            osize = [size for i in range(server_no)]
            state_num = []
            state_num_prod = 1
            for i in range(server_no):
                state_num.append(main_queue_size[i] * osize[i])
                state_num_prod *= state_num[i]
            # Computing the generator matrix
            print("computing the generator matrix for orbit size", size)
            start = time.time()
            row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, mu0_p, timeout, main_queue_size,
                                                                      osize, -1, [0, 0], [0, 0])
            Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num_prod, state_num_prod))

            def matvec_func(x):
                return Q.dot(x)

            def rmatvec_func(x):
                return Q.T.dot(x)

            Q_op = GeneratorMatrix(shape=(state_num_prod, state_num_prod), matvec=matvec_func, rmatvec=rmatvec_func,
                                   dtype=Q.dtype)
            Q_op_t = Q_op.T
            # compute eigen space for the generator matrix
            print("Compute eigenspace")
            start = time.time()
            eigvals, eigvecs = eigs(Q_op_t, k=2, which='SM')
            print("eigenspace is computed in", time.time() - start)
            print("second largest eigenvalue is", np.min(eigvals))
            mixing_time = abs(math.log2(.1) / np.min(np.real(eigvals)))
            mixing_time_seq.append(mixing_time)
        return osize_seq, mixing_time_seq

    def mixing_time_vs_mu_data_generator(self, lambda_config, timeout, main_queue_size, retry_queue_size, lb, hb, step):
        server_no = self.server_no
        mu_seq = []
        mixing_time_seq = []
        for mu in range(lb, hb, step):
            mu_seq.append(mu)
            mus = [mu for i in range(server_no)]
            state_num_prod = self.state_num_prod
            # Computing the generator matrix
            print("computing the generator matrix for mu", mu)
            start = time.time()
            row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, mus, timeout, main_queue_size,
                                                                      retry_queue_size, -1, [0, 0], [0, 0])
            Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num_prod, state_num_prod))

            def matvec_func(x):
                return Q.dot(x)

            def rmatvec_func(x):
                return Q.T.dot(x)

            Q_op = GeneratorMatrix(shape=(state_num_prod, state_num_prod), matvec=matvec_func, rmatvec=rmatvec_func,
                                   dtype=Q.dtype)
            Q_op_t = Q_op.T
            # compute eigen space for the generator matrix
            print("Compute eigenspace")
            start = time.time()
            eigvals, eigvecs = eigs(Q_op_t, k=2, which='SM')
            print("eigenspace is computed in", time.time() - start)
            print("second largest eigenvalue is", np.min(eigvals))
            mixing_time = abs(math.log2(.1) / np.min(np.real(eigvals)))
            mixing_time_seq.append(mixing_time)
        return mu_seq, mixing_time_seq



    def mixing_time_vs_to_data_generator(self, lambda_config, mu0_p, main_queue_size, retry_queue_size, lb, hb, step):
        server_no = self.server_no
        to_seq = []
        mixing_time_seq = []
        for to in range(lb, hb, step):
            to_seq.append(to)
            tos = [to for i in range(server_no)]
            state_num_prod = self.state_num_prod
            # Computing the generator matrix
            print("computing the generator matrix for timeout", to)
            start = time.time()
            row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, mu0_p, main_queue_size,
                                                                      retry_queue_size, tos, -1, [0, 0], [0, 0])
            Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num_prod, state_num_prod))

            def matvec_func(x):
                return Q.dot(x)

            def rmatvec_func(x):
                return Q.T.dot(x)

            Q_op = GeneratorMatrix(shape=(state_num_prod, state_num_prod), matvec=matvec_func, rmatvec=rmatvec_func,
                                   dtype=Q.dtype)
            Q_op_t = Q_op.T
            # compute eigen space for the generator matrix
            print("Compute eigenspace")
            start = time.time()
            eigvals, eigvecs = eigs(Q_op_t, k=2, which='SM')
            print("eigenspace is computed in", time.time() - start)
            print("second largest eigenvalue is", np.min(eigvals))
            mixing_time = abs(math.log2(.1) / np.min(np.real(eigvals)))
            mixing_time_seq.append(mixing_time)
        return to_seq, mixing_time_seq

    def fault_simulation_data_generator(self, pi_q_seq, main_queue_ave_len_seq, lambda_seq):
        """To compute the simulation data corresponding to the fixed fault scenario"""
        lambda_reset = self.lambda_reset
        lambda_init = self.lambda_init
        fault_time = self.fault_time
        reset_time = self.reset_time
        lambda_fault = self.lambda_fault
        step_time = self.step_time
        state_num = self.state_num_prod
        q_min_list = self.q_min_list
        o_min_list = self.o_min_list
        q_max_list = self.q_max_list
        o_max_list = self.o_max_list
        main_queue_size = self.main_queue_size
        retry_queue_size = self.retry_queue_size
        server_no = self.server_no
        mu0_p = self.mu0_p
        timeout = self.timeout
        # constructing the config set
        config_set = []
        config_set.append(lambda_init)
        config_set.append(lambda_fault)
        config_set.append(lambda_reset)
        row_ind_set = []
        col_ind_set = []
        data_set = []
        for lambda_config in config_set:
            # Computing the generator matrix
            print("computing the generator matrix")
            start = time.time()
            row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, mu0_p, main_queue_size,
                                                                      retry_queue_size, timeout, -1, [0, 0], [0, 0])
            print("time taken to compute the generator matrix", time.time() - start)
            row_ind_set.append(row_ind)
            col_ind_set.append(col_ind)
            data_set.append(data)

            Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num, state_num))
            def matvec_func(x):
                return Q.dot(x)

            def rmatvec_func(x):
                return Q.T.dot(x)
            Q_op = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func,
                                   dtype=Q.dtype)

            # compute stationary distribution for the ctmc
            """print("Compute stationary distribution for ctmc")
            Q_op_t = Q_op.T
            start = time.time()
            eigvals, eigvecs = eigs(Q_op_t, k=1, which='SM')
            eigvals
            stationary_vec = np.real(eigvecs[:, 0])
            if np.sum(stationary_vec) < 0:
                stationary_vec = - stationary_vec
            stationary_vec = stationary_vec / np.linalg.norm(stationary_vec, ord=1)
            if lambda_config == lambda_init:
                np.save("pi_ss_init", stationary_vec)
            elif lambda_config == lambda_fault:
                np.save("pi_ss_fault", stationary_vec)
            print("stationary dist is computed in", time.time() - start)"""

            # compute eigen space for the generator matrix
            """print("Compute eigenspace")
            start = time.time()
            eigvals, eigvecs = eigs(Q_op_t, k=2, which='SM')
            print("eigenspace is computed in", time.time() - start)
            print("second largest eigenvalue is", np.min(eigvals))

            # compute spectral gap and the clustering for the CTMC
            # compute SVD
            print("Compute SVD")
            start = time.time()
            U, Sigma, Vt = svds(Q_op, k=2, which='LM')
            print("time taken to compute SVD is", time.time()-start)"""

            # select the singular/eigen vectors corresponding to small singular/eigen values
            """second_singular_ind = np.argmax(Sigma)
            dominant_singular_vectors = Vt[second_singular_ind:second_singular_ind+1, :]  # Take the 2nd singular vector
            second_eigen_ind = np.argmax(eigvals)
            dominant_eigen_vectors = np.real(eigvecs[:, second_eigen_ind:second_eigen_ind+1])  # Take the 2nd eigenvector"""

            # cluster the states using k-means on the dominant singular/eigen vectors
            """state_coordinates_svd = dominant_singular_vectors.T  # Each state is a row
            state_coordinates_eig = dominant_eigen_vectors  # Each state is a row"""

            # perform clustering using k-means
            """start = time.time()
            if np.real(eigvals[1]) > -.01:  # spectral gap is large
                num_clusters = 1
            else:
                num_clusters = 2  # spectral gap is small --> check for metastability
            #kmeans_svd = KMeans(n_clusters=2)
            #kmeans_svd.fit(state_coordinates_svd)
            #labels_svd = kmeans_svd.labels_  # This gives the metastable cluster assignment for each state
            labels_svd, _ = self.kmeans_manual(state_coordinates_svd, k=num_clusters, max_iters=100)
            if lambda_config == lambda_init:
                np.save("labels_svd_init", labels_svd)
            elif lambda_config == lambda_fault:
                np.save("labels_svd_fault", labels_svd)
            #kmeans_eig = KMeans(n_clusters=2)
            #kmeans_eig.fit(state_coordinates_eig)
            #labels_eig = kmeans_eig.labels_  # This gives the metastable cluster assignment for each state
            labels_eig, _ = self.kmeans_manual(state_coordinates_eig, k=num_clusters, max_iters=100)
            print("time taken to compute clustering is", time.time() - start)
            if lambda_config == lambda_init:
                np.save("labels_eig_init", labels_eig)
            elif lambda_config == lambda_fault:
                np.save("labels_eig_fault", labels_eig)

            # analyze and interpret the clusters
            print("Cluster labels (metastable states):", labels_eig)"""

            # computing the accumulated probs over low and high regimes
            """stable_prob_list = []
            unstable_prob_list = []
            for node_id in range(server_no):
                # identify the low-regime set of states in the target node
                q_range_list = []
                o_range_list = []
                for node in range(server_no):
                    if node == node_id:
                        q_range_list.append([0, q_max_list[node]])
                        o_range_list.append([0, retry_queue_size[node]])
                    else:
                        q_range_list.append([0, main_queue_size[node]])
                        o_range_list.append([0, retry_queue_size[node]])
                # compute the prob dist assigned to the low-regime set of states for the target node
                stable_prob_list.append(self.cumulative_prob_computer(stationary_vec, q_range_list, o_range_list))
                print("prob dedicated to low-regime of node", node_id, "is", stable_prob_list[node_id])
                # identify the high-regime set of states in the target node
                q_range_list = []
                o_range_list = []
                for node in range(server_no):
                    if node == node_id:
                        q_range_list.append([q_min_list[node], main_queue_size[node]])
                        o_range_list.append([0, retry_queue_size[node]])
                    else:
                        q_range_list.append([0, main_queue_size[node]])
                        o_range_list.append([0, retry_queue_size[node]])
                # compute the prob dist assigned to the high-regime set of states for the target node
                unstable_prob_list.append(self.cumulative_prob_computer(stationary_vec, q_range_list, o_range_list,
                                                                        main_queue_size, retry_queue_size))
                print("prob dedicated to high-regime of node", node_id, "is", unstable_prob_list[node_id])

                if num_clusters == 1: # spectral gap is large
                    if stable_prob_list[node_id] > .5:
                        print("config is stable for node", node_id)
                    elif unstable_prob_list[node_id] > .5:
                        print("config is unstable for node", node_id)
                    else:
                        print("config type is not determined for node", node_id)
                else: # spectral gap is small --> check for metastability
                    if stable_prob_list[node_id] > .33 and unstable_prob_list[node_id] > .33:
                        print("config is stable for node", node_id)
                    elif stable_prob_list[node_id] > .5 and unstable_prob_list[node_id] < .1:
                        print("config is stable for node", node_id)
                    elif unstable_prob_list[node_id] > .5 and stable_prob_list[node_id] < .1:
                        print("config is unstable for node", node_id)
                    else:
                        print("config type is not determined for node", node_id)"""
            # compute hitting times
            """print("computing hitting times...")
            for node_id in range(server_no):
                start = time.time()
                # identify the high-regime set of states in the target node
                q_range_high_list = []
                o_range_high_list = []
                for node in range(server_no):
                    if node == node_id:
                        q_range_high_list.append([q_min_list[node], main_queue_size[node]])
                        o_range_high_list.append([0, retry_queue_size[node]])
                    else:
                        q_range_high_list.append([0, main_queue_size[node]])
                        o_range_high_list.append([0, retry_queue_size[node]])
                # identify the low-regime set of states in the target node
                q_range_low_list = []
                o_range_low_list = []
                for node in range(server_no):
                    if node == node_id:
                        q_range_low_list.append([0, q_max_list[node]])
                        o_range_low_list.append([0, retry_queue_size[node]])
                    else:
                        q_range_low_list.append([0, main_queue_size[node]])
                        o_range_low_list.append([0, retry_queue_size[node]])
                # computing low to high hitting time
                # compute the sparse gen matrix for the CTMC in which high-regime states are made absorbing
                row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, node_id,
                                                                               q_range_high_list[node_id],
                                                                               o_range_high_list[node_id])
                Q_su = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num, state_num))

                def matvec_func(x):
                    return Q_su.dot(x)

                def rmatvec_func(x):
                    return Q_su.T.dot(x)

                Q_su_op = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func,
                                          dtype=Q_su.dtype)


                S1 = self.set_construction(q_range_low_list, o_range_low_list, main_queue_size, retry_queue_size)

                S2 = self.set_construction(q_range_high_list, o_range_high_list, main_queue_size, retry_queue_size)
                ht_su = self.hitting_time_average(Q_su, S1, S2)
                print("ht_su is", ht_su)
                print("time required for computing ht_su for node", node_id, "is", time.time() - start)

                # computing high to low hitting time
                # compute the sparse gen matrix for the CTMC in which high-regime states are made absorbing
                row_ind, col_ind, data = self.sparse_info_calculator_CTMC(lambda_config, node_id,
                                                                               q_range_low_list[node_id],
                                                                               o_range_low_list[node_id])
                Q_us = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num, state_num))

                def matvec_func(x):
                    return Q_us.dot(x)

                def rmatvec_func(x):
                    return Q_us.T.dot(x)

                Q_us_op = GeneratorMatrix(shape=(state_num, state_num), matvec=matvec_func, rmatvec=rmatvec_func,
                                          dtype=Q_us.dtype)

                S1 = self.set_construction(q_range_high_list, o_range_high_list, main_queue_size, retry_queue_size)
                S2 = self.set_construction(q_range_low_list, o_range_low_list, main_queue_size, retry_queue_size)


                ht_us = self.hitting_time_average(Q_us, S1, S2)
                print("ht_us is", ht_us)
                print("time required for computing ht_us for node", node_id, "is", time.time() - start)"""





        # SIMULATION
        for t in range(0, self.sim_time, self.step_time):
            start = time.time()
            if t <= fault_time:
                lambda_seq.append(lambda_init[0])
                data = [data_set[0][i]*step_time for i in range(len(data_set[0]))]
                row_ind = copy.deepcopy(row_ind_set[0])
                col_ind = copy.deepcopy(col_ind_set[0])
            elif t >= fault_time and t <= fault_time + reset_time:
                lambda_seq.append(lambda_fault[0])
                data = [data_set[1][i] * step_time for i in range(len(data_set[1]))]
                row_ind = copy.deepcopy(row_ind_set[1])
                col_ind = copy.deepcopy(col_ind_set[1])
            else:
                lambda_seq.append(lambda_reset[0])
                data = [data_set[2][i] * step_time for i in range(len(data_set[2]))]
                row_ind = copy.deepcopy(row_ind_set[2])
                col_ind = copy.deepcopy(col_ind_set[2])
            Q = scipy.sparse.csr_matrix((data, (row_ind, col_ind)), shape=(state_num, state_num))

            def matvec_func(x):
                return Q.dot(x)

            def rmatvec_func(x):
                return Q.T.dot(x)

            Q_op_T = GeneratorMatrix(shape=(state_num, state_num), matvec=rmatvec_func, rmatvec=matvec_func,
                                     dtype=Q.dtype)

            pi_q_new = scipy.sparse.linalg.expm_multiply(Q_op_T, pi_q_seq[t // step_time])


            # collecting new measurements
            pi_q_seq.append(copy.copy(pi_q_new))
            mean_queue_lengths = self.main_queue_average_size(pi_q_new, main_queue_size, retry_queue_size)
            print("expected queue length is", mean_queue_lengths)
            mean_orbit_lengths = self.orbit_average_size(pi_q_new, main_queue_size, retry_queue_size)
            print("expected orbit length is", mean_orbit_lengths)
            main_queue_ave_len_seq.append(np.sum(mean_queue_lengths))
            print("current time step is", t)
            print("time taken for one computation of exp matrix is", time.time() - start)
        return [pi_q_seq, main_queue_ave_len_seq, lambda_seq]
