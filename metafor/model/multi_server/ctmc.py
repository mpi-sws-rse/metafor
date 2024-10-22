from typing import List, Tuple

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

from model.multi_server.ctmc_parameters import MultiServerCTMCParameters


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
        q_min_list: List[int] = None,
        q_max_list: List[int] = None,
        o_min_list: List[int] = None,
        o_max_list: List[int] = None,
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
        state_num = []
        state_num_prod = 1
        for i in range(len(mu0_ps)):
            state_num.append(main_queue_sizes[i] * retry_queue_sizes[i])
            state_num_prod *= state_num[i]

        if q_min_list is None:
            q_min_list = [int(main_queue_sizes[i] * 0.9) for i in range(server_num)]
        if o_min_list is None:
            o_min_list = [retry_queue_sizes[i] // 2 for i in range(server_num)]
        if q_max_list is None:
            q_max_list = [main_queue_sizes[i] * 0.1 for i in range(server_num)]
        if o_max_list is None:
            o_max_list = [2 for _ in range(server_num)]

        self.params = MultiServerCTMCParameters(
            server_num,
            main_queue_sizes,
            retry_queue_sizes,
            lambdaas,
            mu0_ps,
            timeouts,
            max_retries,
            thread_pools,
            sub_tree_list,
            parent_list,
            q_min_list,
            q_max_list,
            o_min_list,
            o_max_list,
            state_num,
            state_num_prod,
        )

    def index_decomposer(self, total_ind):
        """This function converts a given index in range [0, state_num]
        into two indices corresponding to (1) number of jobs in orbit and (2) jobs in the queue.
        """
        server_num = self.params.server_num
        main_queue_size = self.params.main_queue_sizes
        parent_list = self.params.parent_list
        state_num = self.params.state_num

        n_main_queue = []
        n_retry_queue = []
        for node_id in range(server_num):
            ancestors = []
            ss_res = 1
            ancestor = parent_list[node_id]
            while len(ancestor) != 0:
                ss_res *= state_num[ancestor[0]]
                ancestors.append(ancestor[0])
                ancestor = parent_list[ancestor[0]]

            s = (total_ind // ss_res) % state_num[
                node_id
            ]  # state index within SS of node_id
            n_main_queue.append(s % main_queue_size[node_id])
            n_retry_queue.append(s // main_queue_size[node_id])
        return [n_main_queue, n_retry_queue]

    def index_composer(self, n_main_queue_list, n_retry_queue_list):
        """This function converts two given input indices into one universal index in range [0, state_num].
        The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit.
        """
        server_num = self.params.server_num
        main_queue_size = self.params.main_queue_sizes
        state_num = self.params.state_num

        total_ind = n_main_queue_list[0] + n_retry_queue_list[0] * main_queue_size[0]
        for node_id in range(1, server_num):
            ss_size_bias = 1
            for i in range(node_id):
                ss_size_bias *= state_num[i]
            total_ind += (
                n_main_queue_list[node_id]
                + n_retry_queue_list[node_id] * main_queue_size[node_id]
            ) * ss_size_bias
        return total_ind

    def tail_prob_computer(self, total_ind):
        """This function computes the timeout probabilities for the case
        that service time is distributed exponentially."""
        mu0_p = self.params.mu0_ps
        timeout = self.params.timeouts
        server_num = self.params.server_num
        sub_tree_list = self.params.sub_tree_list

        [q_list, _] = self.index_decomposer(total_ind)
        tail_prob = [0 for _ in range(server_num)]
        for node_id in range(server_num):
            ave = 0
            var = 0
            sub_tree = sub_tree_list[node_id]
            for downstream_node_id in sub_tree:
                ave += q_list[downstream_node_id] / mu0_p[downstream_node_id]
                var += q_list[downstream_node_id] * 1 / (mu0_p[downstream_node_id] ** 2)
            sigma = math.sqrt(var)
            if timeout[node_id] - ave > sigma:
                k_inv = sigma / (timeout[node_id] - ave)
                tail_prob[node_id] = k_inv**2
            else:
                tail_prob[node_id] = 1
        return tail_prob

    def cumulative_prob_computer(self, pi, q_range, o_range):
        """To compute the probability mass of set of states with queue length between q_min and q_max,
        and orbit length between o_min and o_max"""
        # index_composer = self.index_composer()
        server_num = self.params.server_num

        q_local_list = []
        o_local_list = []
        for node_id in range(server_num):
            q_local_list.append(list(range(q_range[node_id][0], q_range[node_id][1])))
            o_local_list.append(list(range(o_range[node_id][0], o_range[node_id][1])))
        q_prod_list = list(itertools.product(*q_local_list))
        o_prod_list = list(itertools.product(*o_local_list))
        prod_state_list = []
        for q in q_prod_list:
            for o in o_prod_list:
                prod_state_list.append(self.index_composer(q, o))
        cumulative_prob = 0
        for state in prod_state_list:
            cumulative_prob += pi[state]
        return cumulative_prob

    def main_queue_size_average(self, pi: npt.NDArray[np.float64]) -> float:
        server_num = self.params.server_num
        main_queue_size = self.params.main_queue_sizes
        retry_queue_size = self.params.retry_queue_sizes
        q_len = [0 for _ in range(server_num)]
        for node_id in range(server_num):
            q_len_node = 0
            for q_node in range(main_queue_size[node_id]):
                q_range = []
                o_range = []
                for server_id in range(server_num):
                    if server_id != node_id:
                        q_range.append(list(range(main_queue_size[server_id])))
                    else:
                        q_range.append([q_node])
                for server_id in range(server_num):
                    o_range.append(list(range(retry_queue_size[server_id])))
                q_prod_list = list(itertools.product(*q_range))
                o_prod_list = list(itertools.product(*o_range))
                p = 0
                for q in q_prod_list:
                    for o in o_prod_list:
                        p += pi[self.index_composer(q, o)]
                q_len_node += q_node * p
            q_len[node_id] = q_len_node
        return q_len

    def sparse_info_calculator(
        self, lambda_list, node_selected, q_range, o_range
    ) -> Tuple[int, int, List]:
        state_num = self.params.state_num_prod
        server_num = self.params.server_num
        parent_list = self.params.parent_list
        mu0_p = self.params.mu0_ps
        timeout = self.params.timeouts
        max_retries = self.params.max_retries
        main_queue_size = self.params.main_queue_sizes
        retry_queue_size = self.params.retry_queue_sizes
        row_ind = []
        col_ind = []
        data = []
        for total_ind in range(state_num):
            q, o = self.index_decomposer(total_ind)
            absorbing_flg = False
            for node_id in range(server_num):
                if node_id == node_selected:
                    if (
                        q_range[1] >= q[node_id] >= q_range[0]
                        and o_range[1] >= o[node_id] >= o_range[0]
                    ):
                        absorbing_flg = True

            if not absorbing_flg:
                val_sum = 0
                tail_prob_list = self.tail_prob_computer(total_ind)
                q_next = [0 * i for i in range(server_num)]
                o_next = [0 * i for i in range(server_num)]
                # compute the non-synchronized transitions' rates of the generator matrix
                for node_id in range(server_num):
                    mu_drop_base = 1 / (timeout[node_id] * (max_retries[node_id] + 1))
                    mu_retry_base = max_retries[node_id] / (
                        timeout[node_id] * (max_retries[node_id] + 1)
                    )
                    # Check which arrival source is active for the selected node_id
                    lambdaa = lambda_list[node_id]
                    if (
                        len(parent_list[node_id]) == 0
                    ):  # if there exists only a local source of job arrival
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
                        col_ind.append(self.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # Setting the rates related to abandon and retry
                        if o[node_id] > 0:  # if there is any job in the server's orbit
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_drop_base  # drop rate
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[node_id] = q[node_id] + 1
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_retry_base  # retry rate
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
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
                        col_ind.append(self.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # setting the rates related to abandon
                        if o[node_id] > 0:  # if there is any job in the server's orbit
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_drop_base
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[:] = q
                            o_next[:] = o
                        # setting the rates related to moving to the orbit space
                        if (
                            o[node_id] < retry_queue_size[node_id] - 1
                        ):  # if orbit isn't full
                            o_next[node_id] = o[node_id] + 1
                            val = (
                                lambda_summed * mu_retry_base * tail_prob_list[node_id]
                            )
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
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
                        col_ind.append(self.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # setting the rates related to job processing
                        q_next[node_id] = q[node_id] - 1
                        val = mu0_p[node_id]
                        col_ind.append(self.index_composer(q_next[:], o_next[:]))
                        data.append(val)
                        val_sum += val
                        row_ind.append(total_ind)
                        q_next[:] = q
                        o_next[:] = o
                        # Setting the rates related to abandon and retry
                        if o[node_id] > 0:  # if there is any job in the server's orbit
                            o_next[node_id] = o[node_id] - 1
                            val = o[node_id] * mu_drop_base
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
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
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
                            data.append(val)
                            val_sum += val
                            row_ind.append(total_ind)
                            q_next[:] = q
                            o_next[:] = o
                        # setting the rates related to moving to the orbit space
                        if (
                            o[node_id] < retry_queue_size[node_id] - 1
                        ):  # if orbit isn't full
                            q_next[node_id] = q[node_id] + 1
                            o_next[node_id] = o[node_id] + 1
                            val = (
                                lambda_summed * mu_retry_base * tail_prob_list[node_id]
                            )
                            col_ind.append(self.index_composer(q_next[:], o_next[:]))
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
    ) -> Tuple[npt.NDArray[np.float64], int, int, List, GeneratorMatrix, GeneratorMatrix]:
        state_num = self.params.state_num_prod
        row_ind, col_ind, data = self.sparse_info_calculator(
            lambda_config, -1, [0, 0], [0, 0]
        )
        Q = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), shape=(state_num, state_num)
        )

        def matvec_func(x):
            return Q.T.dot(x)

        def rmatvec_func(x):
            return Q.dot(x)

        Q_op = GeneratorMatrix(
            shape=(state_num, state_num),
            matvec=rmatvec_func,
            rmatvec=matvec_func,
            dtype=Q.dtype,
        )
        Q_op_T = GeneratorMatrix(
            shape=(state_num, state_num),
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
        return pi_ss, row_ind, col_ind, data, Q_op, Q_op_T

    def get_stationary_distribution(self) -> npt.NDArray[np.float64]:
        if self.pi is None:
            self.pi, _, _, _, _, _ = self.compute_stationary_distribution(
                self.params.lambda_init
            )
        return self.pi

    def hitting_time_average(self, Q, S1, S2) -> float:
        state_num = self.params.state_num_prod
        A = copy.deepcopy(Q)
        b = -np.ones(state_num)
        b[S2] = 0

        def matvec_func(x):
            return A.dot(x)

        def rmatvec_func(x):
            return A.T.dot(x)

        A_op = GeneratorMatrix(
            shape=(state_num, state_num),
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
                        state = self.index_composer([q1, q2, 0], [o1, o2, 0])
                        new_set.append(state)
        return new_set
