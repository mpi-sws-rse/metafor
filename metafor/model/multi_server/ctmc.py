from typing import List, Tuple, Optional

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

    def _index_decomposer(self, total_ind):
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
        return [n_main_queue, n_retry_queue]

    def _index_composer(self, n_main_queue_list, n_retry_queue_list):
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
        return total_ind

    def tail_prob_computer(self, total_ind):
        """This function computes the timeout probabilities for the case
        that service time is distributed exponentially."""

        [q_list, _] = self._index_decomposer(total_ind)
        tail_prob = [0 for _ in range(self.server_num)]
        for node_id in range(self.server_num):
            ave = 0
            var = 0
            sub_tree = self.sub_tree_list[node_id]
            for downstream_node_id in sub_tree:
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
        return tail_prob

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
                for server_id in range(self.server_num):
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

    def sparse_info_calculator(
        self, lambda_list, node_selected, q_range, o_range
    ) -> Tuple[int, int, List]:
        row_ind = []
        col_ind = []
        data = []
        for total_ind in range(self.state_num_prod):
            q, o = self._index_decomposer(total_ind)
            absorbing_flg = False
            for node_id in range(self.server_num):
                if node_id == node_selected:
                    if (
                        q_range[1] >= q[node_id] >= q_range[0]
                        and o_range[1] >= o[node_id] >= o_range[0]
                    ):
                        absorbing_flg = True

            if not absorbing_flg:
                val_sum = 0
                tail_prob_list = self.tail_prob_computer(total_ind)
                q_next = [0 * i for i in range(self.server_num)]
                o_next = [0 * i for i in range(self.server_num)]
                # compute the non-synchronized transitions' rates of the generator matrix
                for node_id in range(self.server_num):
                    mu_drop_base = 1 / (
                        self.timeouts[node_id] * (self.max_retries[node_id] + 1)
                    )
                    mu_retry_base = self.max_retries[node_id] / (
                        self.timeouts[node_id] * (self.max_retries[node_id] + 1)
                    )
                    # Check which arrival source is active for the selected node_id
                    lambdaa = lambda_list[node_id]
                    if (
                        len(self.parent_list[node_id]) == 0
                    ):  # if there exists only a local source of job arrival
                        lambda_summed = lambdaa
                    elif q[self.parent_list[node_id][0]] == 0:
                        lambda_summed = lambdaa
                    else:  # if there exists local and non-local sources of job arrival
                        lambda_summed = (
                            lambdaa + self.mu0_ps[self.parent_list[node_id][0]]
                        )
                    if q[node_id] == 0:  # queue is empty
                        q_next[:] = q
                        o_next[:] = o
                        # Setting the rates related to job arrivals
                        q_next[node_id] = q[node_id] + 1
                        val = lambda_summed
                        col_ind.append(self._index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # Setting the rates related to abandon and retry
                        if o[node_id] > 0:  # if there is any job in the server's orbit
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_drop_base  # drop rate
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[node_id] = q[node_id] + 1
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_retry_base  # retry rate
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[:] = q
                            o_next[:] = o

                    elif (
                        q[node_id] == self.main_queue_sizes[node_id] - 1
                    ):  # queue is full
                        q_next[:] = q
                        o_next[:] = o
                        # setting the rates related to job processing
                        q_next[node_id] = q[node_id] - 1
                        val = self.mu0_ps[node_id]
                        col_ind.append(self._index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # setting the rates related to abandon
                        if o[node_id] > 0:  # if there is any job in the server's orbit
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_drop_base
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[:] = q
                            o_next[:] = o
                        # setting the rates related to moving to the orbit space
                        if (
                            o[node_id] < self.retry_queue_sizes[node_id] - 1
                        ):  # if orbit isn't full
                            o_next[node_id] = o[node_id] + 1
                            val = (
                                lambda_summed * mu_retry_base * tail_prob_list[node_id]
                            )
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
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
                        col_ind.append(self._index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # setting the rates related to job processing
                        q_next[node_id] = q[node_id] - 1
                        val = self.mu0_ps[node_id]
                        col_ind.append(self._index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # Setting the rates related to abandon and retry
                        if o[node_id] > 0:  # if there is any job in the server's orbit
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_drop_base
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[node_id] = q[node_id] + 1
                            o_next[node_id] = o[node_id] - 1
                            val = (
                                o[node_id]
                                * mu_retry_base
                                * (1 - tail_prob_list[node_id])
                            )
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[:] = q
                            o_next[:] = o
                        # setting the rates related to moving to the orbit space
                        if (
                            o[node_id] < self.retry_queue_sizes[node_id] - 1
                        ):  # if orbit isn't full
                            q_next[node_id] = q[node_id] + 1
                            o_next[node_id] = o[node_id] + 1
                            val = (
                                lambda_summed * mu_retry_base * tail_prob_list[node_id]
                            )
                            col_ind.append(self._index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[:] = q
                            o_next[:] = o
                val = -val_sum
                col_ind.append(total_ind)
                data.append(val)
                row_ind.append(total_ind)
        return [row_ind, col_ind, data]

    def compute_stationary_distribution(
        self, lambda_config
    ) -> Tuple[
        npt.NDArray[np.float64], int, int, List, GeneratorMatrix
    ]:
        row_ind, col_ind, data = self.sparse_info_calculator(
            lambda_config, -1, [0, 0], [0, 0]
        )
        Q = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(self.state_num_prod, self.state_num_prod)
        )

        def matvec_func(x):
            return Q.T.dot(x)

        def rmatvec_func(x):
            return Q.dot(x)

        Q_op = GeneratorMatrix(
            shape=(self.state_num_prod, self.state_num_prod),
            matvec=rmatvec_func,
            rmatvec=matvec_func,
            dtype=Q.dtype,
        )
        Q_op_T = GeneratorMatrix(
            shape=(self.state_num_prod, self.state_num_prod),
            matvec=matvec_func,
            rmatvec=rmatvec_func,
            dtype=Q.dtype,
        )
        start = time.time()
        _, eigenvectors = eigs(Q_op_T, k=1, which="SM")
        pi_ss = np.real(eigenvectors) / np.linalg.norm(np.real(eigenvectors), ord=1)
        if pi_ss[0] < -0.00000001:
            pi_ss = -pi_ss
        print("Computing the stationary distribution took ", time.time() - start)
        return pi_ss, row_ind, col_ind, data, Q_op

    def get_stationary_distribution(self) -> npt.NDArray[np.float64]:
        if self.pi is None:
            self.pi, _, _, _, _ = self.compute_stationary_distribution(self.lambdaas)
        return self.pi

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
