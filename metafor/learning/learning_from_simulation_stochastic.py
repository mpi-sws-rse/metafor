import math
import os
import time
from typing import List

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pickle
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

from collections import defaultdict, Counter
from itertools import product
from typing import List, Tuple, Dict
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import LinearOperator
import scipy.sparse as sp

from scipy.optimize import least_squares
import cvxpy as cp
import osqp

os.makedirs("results", exist_ok=True)


def learn_dtmc_transition_matrix(
        trajectories: List[List[int]],
        q_max: int,
        history_length: int
) -> Tuple[np.ndarray, Dict[Tuple[int, ...], int]]:
    """
    Learn a DTMC transition probability matrix explicitly

    Args:
        trajectories (List[List[int]]): List of integer-valued trajectories.
        q_max (int): Maximum value a state can take (non-negative integer).
        history_length (int): Length of history for each state (d in the DTMC).

    Returns:
        transition_matrix (np.ndarray): A square matrix of shape (num_states, num_states).
        state_to_index (dict): Mapping from state tuple to row/column index.
    """

    # Enumerate all possible states (tuples of length `history_length`)
    all_states = list(product(range(q_max), repeat=history_length))
    state_to_index = {state: idx for idx, state in enumerate(all_states)}

    num_states = len(all_states)
    transition_counts = np.zeros((num_states, num_states), dtype=np.float64)

    # Count empirical transitions: (q_{t-d}, ..., q_{t-1}) → (q_{t-d+1}, ..., q_t)
    for traj in trajectories:
        if len(traj) <= history_length:
            continue
        for i in range(len(traj) - history_length):
            from_state = tuple(traj[i: i + history_length])
            to_state = tuple(traj[i + 1: i + history_length + 1])
            if from_state in state_to_index and to_state in state_to_index:
                from_idx = state_to_index[from_state]
                to_idx = state_to_index[to_state]
                transition_counts[from_idx, to_idx] += 1

    # Normalize each row to convert counts to probabilities
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        transition_matrix = np.divide(transition_counts, row_sums, where=row_sums != 0)

    return transition_matrix, state_to_index


def prepare_training_data(pi_seq, depth):
    """ Preparing input-output training datasets"""
    X = []
    Y = []

    for pi_traj in pi_seq:
        T = len(pi_traj)
        for t in range(depth, T):
            x = pi_traj[t - depth:t]  # concatenate d history from pi_traj
            y = pi_traj[t: t + 1]
            X.append(x)
            Y.append(y)

    X = np.array(X).squeeze()
    Y = np.array(Y).squeeze()
    return X, Y


def index_composer(n_main_queue, n_retry_queue, qsize, osize):
    """This function converts two given input indices into one universal index in range [0, state_num].
    The input indices correspond to number of (1) jobs in queue and (2) jobs in the orbit."""
    main_queue_size = qsize

    total_ind = n_retry_queue * main_queue_size + n_main_queue
    return total_ind


def index_decomposer(total_ind, qsize, osize):
    """This function converts a given index in range [0, state_num]
    into two indices corresponding to (1) number of jobs in orbit and (2) jobs in the queue."""
    main_queue_size = qsize

    n_retry_queue = total_ind // main_queue_size
    n_main_queue = total_ind % main_queue_size
    return [n_main_queue, n_retry_queue]


def dependent_columns(A, tol=1e-6):
    """
    Return the indices of linearly independent columns of matrix A.

    Parameters:
    - A: 2D numpy array (matrix)
    - tol: tolerance level for determining independence

    Returns:
    - indices: List of indices of the independent columns
    """
    # Perform QR decomposition on the transpose of A (so we're working with columns of A)
    Q, R = np.linalg.qr(A)

    # Find the indices of independent columns based on the diagonal of R
    dependent = np.where(np.abs(np.diag(R)) < tol)[0]

    return dependent.tolist()


def qlen_average(pi, qsize, osize) -> float:
    # use the law of total expectation
    val = 0

    for n_main_queue in range(qsize):
        weight = 0
        for n_retry_queue in range(osize):
            weight += pi[index_composer(n_main_queue, n_retry_queue, qsize, osize)]
        val += (
                weight
                * n_main_queue
        )
    if isinstance(val, float):
        return val
    return val[0]

def sparsity_measure(P):
    ss_size = np.shape(P)[0]
    #P /= np.max(np.abs(P))
    sparsity_measure_list = []
    for s in range(ss_size):

        q, o = index_decomposer(s, qsize, osize)
        D_s = 0
        for s_prine in range(ss_size):
            q_prine, o_prine = index_decomposer(s_prine, qsize, osize)
            D_s += (abs(q-q_prine) + abs(o-o_prine)) * P[s, s_prine]
        sparsity_measure_list.append(D_s)
        #plt.plot(sparsity_measure_list)
        #plt.show()
    return np.array(sparsity_measure_list)

def high_probability_states(X: np.ndarray, eps: float) -> np.ndarray:
    max_probs = np.max(X, axis=0)              # max over rows for each column (state)
    high_prob_indices = np.where(max_probs > eps)[0]  # indices where max prob > eps
    return high_prob_indices


def augment_matrix(P: np.ndarray, important_indices: np.ndarray, n: int) -> np.ndarray:
    P_augmented = np.zeros((n, n))
    for i, idx_i in enumerate(important_indices):
        for j, idx_j in enumerate(important_indices):
            P_augmented[idx_i, idx_j] = P[i, j]
    return P_augmented


class linear_model():
    def train_linear_least_squares(X, Y):
        """
        Fit a linear model using least squares
        Returns: theta (weights)
        """
        theta, _, _, _ = lstsq(X, Y, rcond=None)
        return theta  # shape: (input_dim,)

    def train_linear_least_squares_with_structure(pi_seq, qsize, osize):
        """Only works for depth equal to one!"""
        """
        Fit a linear model using least squares with zeros enforced corresponding to transitions with rate 0.
        Returns: theta (weights)
        """

        n = len(pi_seq[0][0])  # dimension of the transition prob matrix
        traj_num = len(pi_seq)
        theta = []
        P_mat = np.zeros((n, n))
        err_seq = []
        for s in range(n):
            q, o = index_decomposer(s, qsize, osize)
            if q == 0:
                if o == 0:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                                         index_composer(q, o + 1, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q + 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q, o + 1, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
                elif o == osize - 1:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q + 1, o, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
                else:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                                         index_composer(q, o + 1, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q + 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q, o + 1, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
            elif q == qsize - 1:
                if o == 0:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                                         index_composer(q - 1, o + 1, qsize, osize),
                                         index_composer(q, o + 1, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q, o + 1, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
                elif o == osize - 1:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                                         index_composer(q - 1, o - 1, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o - 1, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
                else:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o + 1, qsize, osize),
                                         index_composer(q - 1, o + 1, qsize, osize),
                                         index_composer(q - 1, o, qsize, osize),
                                         index_composer(q - 1, o - 1, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o - 1, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
            else:
                if o == 0:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                                         index_composer(q - 1, o + 1, qsize, osize),
                                         index_composer(q, o + 1, qsize, osize),
                                         index_composer(q + 1, o, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q + 1, o, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
                elif o == osize - 1:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                                         index_composer(q - 1, o - 1, qsize, osize),
                                         index_composer(q + 1, o, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o - 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q + 1, o, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
                else:
                    X = []
                    Y = []
                    for traj_idx in range(traj_num):
                        pi_traj = pi_seq[traj_idx]
                        T = len(pi_traj)
                        full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o + 1, qsize, osize),
                                         index_composer(q - 1, o + 1, qsize, osize),
                                         index_composer(q - 1, o, qsize, osize),
                                         index_composer(q - 1, o - 1, qsize, osize),
                                         index_composer(q + 1, o, qsize, osize)]
                        for t in range(depth, T):
                            X.append(np.array([pi_traj[t - depth][index_composer(q, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o + 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q - 1, o - 1, qsize, osize)],
                                               pi_traj[t - depth][index_composer(q + 1, o, qsize, osize)]]))
                            Y.append(pi_traj[t][index_composer(q, o, qsize, osize)])
                    X = np.array(X)
                    Y = np.array(Y)
                    dependent_cols = dependent_columns(X)
                    reduced_col_list = []
                    for j in range(len(full_col_list)):
                        if j not in dependent_cols:
                            reduced_col_list.append(full_col_list[j])
                    X_mod = np.delete(X, dependent_cols, axis=1)
                    theta_i = linear_model.train_linear_least_squares(X_mod, Y)
                    theta.append(theta_i)
                    for j in range(len(reduced_col_list)):
                        s_prin = reduced_col_list[j]
                        P_mat[s_prin, s] = theta_i[j]
            err_avg = np.average(np.matmul(X_mod, theta[s]) - Y)
            if err_avg > 1:
                print("LS estimations is poor for state", s)
            err_seq.append(err_avg)
            print("computation for state", s)
        return P_mat

    def sgd_with_reset(X, Y):
        n = np.shape(X)[1] # number of states
        m = np.shape(X)[0] # number of data points
        batch_size = 100
        learning_rate = .01
        epochs = 1000
        gamma = 0.99 # decay factor

        # Initialize transition matrix P randomly
        P = np.random.rand(n, n)
        P /= P.sum(axis=1, keepdims=True)  # Ensure P is stochastic (rows sum to 1)

        # SGD optimization loop
        for epoch in range(epochs):
            learning_rate_epoch = learning_rate * gamma ** epoch
            for i in range(0, m, batch_size):
                # Mini-batch of data
                X_batch = X[i:i + batch_size]
                Y_batch = Y[i:i + batch_size]

                # Compute the gradient of the loss function w.r.t. P
                gradient = -2 * X_batch.T @ (Y_batch - X_batch @ P)

                # Update P using the SGD update rule
                P -=  learning_rate_epoch * gradient

                # Enforce constraints
                P = np.maximum(P, 0)  # Ensure non-negativity
                P /= P.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1 (stochastic)

            # Optional: Print loss at each epoch
            loss = np.linalg.norm(Y - X @ P, 'fro') ** 2
            if epoch % 10 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')
        return P

    def convex_op(X, Y):
        n = np.shape(X)[1]
        # Variables
        P = cp.Variable((n, n))  # The unknown transition matrix P (n x n)

        # Objective function
        # objective = cp.Minimize(cp.norm(Y - X @ P, 'fro') ** 2)
        objective = cp.Minimize(cp.sum_squares(Y - X @ P))

        # Constraints
        constraints = [P >= 0, cp.sum(P, axis=1) == 1]

        # Form the problem
        problem = cp.Problem(objective, constraints)

        # Solve the problem
        problem.solve(solver=cp.OSQP)

        # Solution
        P_opt = P.value
        #print("Optimal transition matrix P:\n", P_opt)
        return P_opt

    def convex_op_sparse(X, Y):
        n = X.shape[1]  # Number of states
        m = X.shape[0]  # Number of data points

        # We vectorize P as a length-n^2 variable: p = vec(P.T) (column-major order)
        # Objective: minimize ||Y - X @ P||_F^2
        # Equivalent to: minimize 1/2 * p^T Q p + c^T p

        # Build Q and c for the vectorized P
        Q_blocks = []
        c_blocks = []
        for i in range(n):
            Xi = X
            yi = Y[:, i]
            Q_i = 2 * Xi.T @ Xi  # (n x n)
            c_i = -2 * Xi.T @ yi  # (n,)
            Q_blocks.append(sp.csc_matrix(Q_i))
            c_blocks.append(c_i)

        Q = sp.block_diag(Q_blocks, format='csc')  # (n^2 x n^2)
        c = np.hstack(c_blocks)  # (n^2,)

        # === Constraints ===
        # 1. Non-negativity: p >= 0
        # 2. Row stochastic: sum_j P[i, j] = 1 for each i

        # 1. Identity matrix for p >= 0
        A_ineq = sp.eye(n * n, format='csc')
        l_ineq = np.zeros(n * n)
        u_ineq = np.inf * np.ones(n * n)

        # 2. Row sum == 1 constraints
        # For each row i: sum over j of P[i, j] == 1
        A_eq_rows = []
        for i in range(n):
            row = np.zeros(n * n)
            for j in range(n):
                idx = j * n + i  # because vec(P.T)
                row[idx] = 1
            A_eq_rows.append(row)

        A_eq = sp.csc_matrix(np.vstack(A_eq_rows))
        l_eq = np.ones(n)
        u_eq = np.ones(n)

        # Combine constraints
        A_combined = sp.vstack([A_ineq, A_eq]).tocsc()
        l_combined = np.hstack([l_ineq, l_eq])
        u_combined = np.hstack([u_ineq, u_eq])
        """A_combined = sp.vstack([A_ineq]).tocsc()
        l_combined = np.hstack([l_ineq])
        u_combined = np.hstack([u_ineq])"""

        # Solve with OSQP
        prob = osqp.OSQP()
        prob.setup(P=Q, q=c, A=A_combined, l=l_combined, u=u_combined, max_iter=4000,
                   eps_abs=1e-4,
                   eps_rel=1e-4,
                   verbose=True)
        res = prob.solve()

        # Reshape solution into matrix form
        p_opt = res.x
        P_opt = p_opt.reshape((n, n)).T  # Because we flattened P.T
        return P_opt


    # List of functions used for LS estimation:
    def simulate_linear_model(theta, pi_seq, depth, qsize, osize):
        """
        Autoregressive rollout using a linear model that predicts both [q_t, o_t]

        Args:
            theta: weight matrix of shape (2*depth, 2)
            pi_seq: list of real-valued sequences
            depth: number of historical steps used as input

        Returns:
            model_preds: list of predicted q sequences
        """
        model_preds = []
        true_vals = []

        for pi_traj in zip(pi_seq):
            pi_traj = np.array(pi_traj).squeeze()
            T = len(pi_traj)
            pi_pred = list(pi_traj[:depth])  # True pi values for initialization
            q_pred = []
            q_true = []
            for t in range(depth):
                q_pred.append(qlen_average(pi_traj[t], qsize, osize))
                q_true.append(qlen_average(pi_traj[t], qsize, osize))
            for t in range(depth, T):
                pi_input = pi_pred[-depth:]
                x = np.array(pi_input).squeeze()
                y_pred = x @ theta  #

                """print("min value is", np.min(y_pred))
                print("max value is", np.max(y_pred))
                print("summation is", np.sum(y_pred))"""
                pi_pred.append(y_pred)
                q_pred.append(qlen_average(y_pred, qsize, osize))
                q_true.append(qlen_average(pi_traj[t], qsize, osize))

            model_preds.append(q_pred)
            true_vals.append(q_true)

        return model_preds, true_vals

    def plot_predictions_vs_true(q_seq, model_preds, save_prefix="results/linear_model_traj"):
        for i, (true_q, pred_q) in enumerate(zip(q_seq, model_preds)):
            plt.figure(figsize=(10, 4))
            plt.plot(true_q, label="True q", marker='o')
            plt.plot(pred_q, label="Linear Model q", marker='x')
            plt.title(f"Trajectory {i}")
            plt.xlabel("Time Step")
            plt.ylabel("q value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_{i}.png")
            plt.close()



# Loading the trajectories...
with open("data_generation/pi_seq.pkl", "rb") as f:
    pi_seq = pickle.load(f)
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("data_generation/q_seq_stochastic.pkl", "rb") as f:
    q_seq_stochastic = pickle.load(f)
traj_num = len(pi_seq)  # Number of trajectories within the dataset
depth = 1  # History length, also known as depth in system identification

X, Y = prepare_training_data(pi_seq, depth)
important_indices = high_probability_states(X, .009)
X_mod = X[np.ix_(np.arange(np.shape(X)[0]), important_indices)]
Y_mod = Y[np.ix_(np.arange(np.shape(Y)[0]), important_indices)]
# Evaluating the performance of least-squares optimizer

# Compute the LS gain
qsize = 100
osize = 30

# theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
# theta = linear_model.train_linear_least_squares(X, Y) # without structure
# theta = linear_model.train_linear_least_squares_with_structure(pi_seq, qsize, osize)  # enforcing structure
# theta = linear_model.sgd_with_reset(X, Y)

theta_red = linear_model.train_linear_least_squares(X_mod, Y_mod) # without structure
# theta_red = linear_model.convex_op(X_mod, Y_mod)
# theta_red = linear_model.convex_op_sparse(X_mod, Y_mod)

theta = augment_matrix(theta_red, important_indices, np.shape(X)[1])


print("the learning error is", np.linalg.norm(Y_mod - X_mod @ theta_red, 'fro') ** 2)
distance_array = sparsity_measure(theta)

# Compute the model predictions
model_preds, true_vals = linear_model.simulate_linear_model(theta, pi_seq, depth, qsize, osize)

# Plot and compare the output of model and true trajectories
linear_model.plot_predictions_vs_true(true_vals, model_preds)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)

# learn_dtmc_transition_matrix(q_seq_stochastic, qsize, 2)