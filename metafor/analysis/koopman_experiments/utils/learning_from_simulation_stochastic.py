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

from scipy.ndimage import gaussian_filter
from scipy.linalg import expm, eigvals, eig

import sys
import random
import cma
from functools import partial

os.makedirs("results", exist_ok=True)


from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize

def residuals(params, k, y):
    c, lam = params
    return y - c * (1 - lam ** k)

def fit_least_squares(k, y, c0=50.0, lam0=0.98):
    result = least_squares(residuals, x0=[c0, lam0], args=(k, y), bounds=([0, 0], [np.inf, 1.0]))
    return result.x  # returns [c, lambda]

def model_y(k, c, lam):
    return c * (1 - lam ** k)

def model_dy(k, c, lam):
    return c * (lam ** (k-1))

def fit_curve_fit(k, y, model, c0=50.0, lam0=0.98):
    popt, _ = curve_fit(model, k, y, p0=[c0, lam0], bounds=([0, 0], [100, 1.0]))
    return popt  # returns [c, lambda]


# --- Loss function: only validate on extrapolation region ---
def generalization_loss(params, k_train, y_train, k_val, y_val):
    c, lam = params
    """if not (0 <= lam <= 1):
        return np.inf  # penalize invalid lambda"""
    y_pred_val = model_y(k_val, c, lam)
    return np.mean((y_val - y_pred_val) ** 2)

# --- Main fitting routine ---
def generalization_guided_fit(y, N1, N2):
    k = np.arange(len(y))
    k_train, y_train = k[:N1], y[:N1]
    k_val, y_val = k[N1:N1 + N2], y[N1:N1+ N2]

    result = minimize(
        generalization_loss,
        x0=[50.0, 0.98],  # initial guess
        args=(k_train, y_train, k_val, y_val),
        bounds=[(0, 100), (0, 1)],
        method='Powell',  # or method='Powell'
    )
    c_opt, lam_opt = result.x
    return c_opt, lam_opt

# ---- Extrapolation-Guided Fit Function ----
def extrapolation_guided_fit(y, N1, N2):
    """
    y: observed sequence (1D array)
    N1: number of initial samples used to fit the dynamics
    N2: number of future points to extrapolate
    """
    # Step 1: Fit difference sequence y_{i+1} - y_i
    dy = np.diff(y[:N1])                    # length N1 - 1
    k_diff = np.arange(N1 - 1)
    c1, lam1 = fit_curve_fit(k_diff, dy, model_dy)        # First-stage fit

    # Step 2: Generate N2 future values
    y_aug = list(y[:N1])
    for i in range(N1, N1 + N2):
        delta = c1 * lam1 ** (i - 1)
        y_aug.append(y_aug[-1] + delta)

    y_aug = np.array(y_aug)
    k_aug = np.arange(len(y_aug))

    # Step 3: Final fit to full sequence
    c_final, lam_final = fit_curve_fit(k_aug, y_aug, model_y)

    return {
        "c1": c1, "lam1": lam1,
        "c_final": c_final, "lam_final": lam_final,
        "k_aug": k_aug, "y_aug": y_aug
    }


def fit_c_lambda(y, c0=50, lambda0=0.98, lr=1e-6, steps=1000):
    N = len(y)
    c = c0
    lam = lambda0

    for step in range(steps):
        r = y - c * (1 - lam ** np.arange(N))
        grad_c = -np.sum(r * (1 - lam ** np.arange(N)))
        grad_lam = np.sum(r * c * np.arange(N) * lam ** (np.arange(N) - 1))

        c -= lr * grad_c
        lam -= lr * grad_lam

        # Optional: project lambda to [0, 1]
        # lam = np.clip(lam, 1e-6, 1.0)

        # Optional: print loss
        if step % 100 == 0:
            loss = 0.5 * np.sum(r ** 2)
            print(f"Step {step}: loss = {loss:.6f}, c = {c:.4f}, lambda = {lam:.4f}")

    return c, lam

# === Compute stationary distribution & second-largest real eigenvalue of Q ===
def compute_stationary_distribution_lambda2(Q):
    """
        Computes the stationary distribution of a CTMC generator matrix Q using eigen decomposition.

        Args:
            Q (ndarray): Generator matrix (n x n)

        Returns:
            pi (ndarray): Stationary distribution (1D array of length n)
            lambda2: second-largest real part
        """
    # Transpose Q to find left eigenvectors
    w, v = eig(Q.T)
    idx = np.argmin(np.abs(w))  # eigenvalue closest to 0
    pi = np.real(v[:, idx])
    pi = pi / np.sum(pi)  # normalize to ensure it's a probability vector
    lambdas = np.real(w)
    lambdas_sorted = np.sort(lambdas)[::-1]
    lambda2 = lambdas_sorted[1]
    return pi, lambda2

# === Fitness function: squared error on qlen and settling time ===
def evaluate_fitness_traj(x, mu, max_retries, qsize, osize):
    lambdaa = x[0]
    timeout_t = x[1]
    print("computing the fitness value")
    Q = get_analytic_ctmc(lambdaa, mu, timeout_t, max_retries, qsize, osize)
    try:
        print("arr_rate and Timeout are", lambdaa, timeout_t)
        error = 0
        num_steps = 50
        T_s = 0.5
        discrete_transition_mat = expm(Q * T_s * num_steps)
        for traj_id in range(len(q_seq)):
            pi_t = pi_seq[traj_id][0] # initial distribution taken from the collected data
            for t in range(0, len(q_seq[traj_id]) - num_steps, num_steps):
                pi_t = np.matmul(pi_t, discrete_transition_mat)
                q_ave_model = qlen_average(pi_t, qsize, osize)
                q_ave_true = q_seq[traj_id][t+1]
                error += (q_ave_true - q_ave_model)**2
        print("error is", error)
        return error
    except Exception:
        return np.inf  # penalize invalid matrices

# === Fitness function: squared error on qlen and settling time ===
def evaluate_fitness(x, mu, max_retries, qsize, osize, q_ave_target, T_s_target):
    lambdaa = x[0]
    timeout_t = x[1]
    print("computing the fitness value")
    try:
        # print("Timeout and max_retries are", timeout_t, max_retries)
        print("arr_rate and Timeout are", lambdaa, timeout_t)
        Q = get_analytic_ctmc(lambdaa, mu, timeout_t, max_retries, qsize, osize)
        pi_ss, lambda2 = compute_stationary_distribution_lambda2(Q)
        q_ave = qlen_average(pi_ss, qsize, osize)
        T_s = 1 / abs(lambda2) if lambda2 != 0 else np.inf
        print("q_ave, T_s are taking values", q_ave, T_s)
        error = (q_ave - q_ave_target)**2 + .01 * (T_s - T_s_target)**2
        return error
    except Exception:
        return np.inf  # penalize invalid matrices

# --- CMA-ES wrapper ---
def run_cmaes_optimization(q_ave_target, T_s_target, lambdaa, mu, timeout_t, max_retries, qsize, osize,
                           sigma, max_iter):
    x0 = [lambdaa, timeout_t]  # Initial guess: [arr_rate, timeout]
    bounds = [[lambdaa * .9, timeout_t * .8], [lambdaa * 1.1, timeout_t * 1.2]]  # bounds for params
    es = cma.CMAEvolutionStrategy(x0, sigma, {'bounds': bounds, 'maxiter': max_iter})

    """def f(x, mu, max_retries_nom, qsize, osize, q_ave_target, T_s_target):
        return evaluate_fitness(x, mu, max_retries_nom, qsize, osize, q_ave_target, T_s_target)"""

    """f = partial(evaluate_fitness, mu=mu, max_retries=max_retries, qsize=qsize, osize=osize,
                q_ave_target=q_ave_target, T_s_target=T_s_target)"""

    f = partial(evaluate_fitness_traj, mu=mu, max_retries=max_retries, qsize=qsize, osize=osize)

    es.optimize(f)

    best_params = np.abs(es.result.xbest)
    best_params[3] = int(np.clip(round(best_params[3]), 1, 10))  # Ensure integer retry

    return best_params, es.result.fbest

# === Evolutionary optimizer ===
def evolutionary_optimization(q_ave_target, T_s_target, lambdaa_nom, mu, timeout_nom, max_retries_nom, qsize, osize,
                              pop_size, generations):
    """population = [(
        np.random.uniform(0.1, 5.0),  # arr_rate
        np.random.uniform(0.1, 5.0),  # mu
        np.random.uniform(0.1, 5.0),  # timeout
        random.randint(1, 10)         # max_retry
    ) for _ in range(pop_size)]"""

    population = [(
        np.random.uniform(0.9 * lambdaa_nom, 1.1 * lambdaa_nom),  # arr_rate
        np.random.uniform(timeout_nom * .8, timeout_nom * 1.2),  # timeout
        # random.randint(max(1, max_retries_nom // 3), int(max_retries_nom * 3))  # max_retry
    ) for _ in range(pop_size)]

    for gen in range(generations):
        print("computing the generation", gen)
        #fitnesses = [evaluate_fitness(lambdaa, mu, population[ind][0], population[ind][1], qsize, osize,
         #                             q_ave_target, T_s_target) for ind in range(pop_size)]
        fitnesses = [evaluate_fitness(population[ind][0], mu, population[ind][1], max_retries_nom, qsize, osize,
                                      q_ave_target, T_s_target) for ind in range(pop_size)]
        sorted_pop = [x for _, x in sorted(zip(fitnesses, population))]
        elites = sorted_pop[:int(0.2 * pop_size)]

        new_pop = elites[:]
        while len(new_pop) < pop_size:
            p1, p2 = random.sample(elites, 2)
            child = tuple(
                max(0.01, np.random.normal((x + y) / 2, abs(x - y) * 0.3)) if i < 3
                else random.randint(1, 10)
                for i, (x, y) in enumerate(zip(p1, p2))
            )
            new_pop.append(child)

        population = new_pop

    best = min(population, key=lambda x: evaluate_fitness(lambdaa, mu, x[0], x[1], qsize, osize, q_ave_target,
                                                          T_s_target))

    return best


def stationary_distribution_eig(P):
    """
    Computes the stationary distribution via eigen decomposition.

    Args:
        P: (n x n) transition matrix

    Returns:
        pi: stationary distribution (1D array)
    """
    eigvals, eigvecs = np.linalg.eig(P.T)
    # Find the eigenvector associated with eigenvalue 1
    idx = np.argmin(np.abs(eigvals - 1))
    pi = np.real(eigvecs[:, idx])
    pi = pi / np.sum(pi)
    return pi

def is_irreducible(P):
    """
    Checks irreducibility of a transition matrix using graph connectivity.

    Args:
        P: (n x n) numpy array, transition probability matrix

    Returns:
        True if irreducible, False otherwise
    """
    n = P.shape[0]
    G = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0:
                G.add_edge(i, j)

    return nx.is_strongly_connected(G)

def get_analytic_ctmc(lambdaa, mu, timeout_t, max_retries, qsize, osize):
    # Define server processing rate
    api = {"insert": Work(mu, [])}

    # Configure server parameters: queue size, orbit size, threads
    server = Server("52", api, qsize=qsize, orbit_size=osize, thread_pool=1)

    # Define client request behavior
    src = Source("client", "insert", lambdaa, timeout=timeout_t, retries=max_retries)

    # Build the request-response system
    p = Program("Service52")
    p.add_server(server)
    p.add_source(src)
    p.connect("client", "52")

    Q = p.build().Q

    np.save("results/Q_mat.npy", Q)
    return Q


def compute_prior_counts(Q, step_time, beta):
    """
    Converts CTMC generator Q into prior pseudocounts for Dirichlet prior.

    Args:
        Q: CTMC generator matrix (n x n)
        tau: sampling time
        beta: total prior count mass per row

    Returns:
        B: prior count matrix (n x n)
    """
    P0 = expm(Q * step_time)
    B = beta * P0
    return B

def simulate_prior_counts(Q, step_time, N, T, rng=None):
    """
        Simulate DTMC derived from CTMC generator Q to compute empirical transition counts.

        Args:
            Q (ndarray): CTMC generator matrix (n x n)
            step_time (float): Time step to discretize the CTMC
            T (float): Total simulation time per trajectory
            N (int): Number of trajectories to simulate
            rng (np.random.Generator or None): Optional random number generator

        Returns:
            B (ndarray): Transition count matrix (n x n)
        """
    if rng is None:
        rng = np.random.default_rng()

    n = Q.shape[0]
    num_steps = int(T / step_time)
    P = expm(Q * step_time)  # DTMC transition matrix
    B = np.zeros((n, n), dtype=float)

    for _ in range(N):
        state = rng.integers(n)  # Uniform initial state
        for _ in range(num_steps):
            next_state = rng.choice(n, p=P[state])
            B[state, next_state] += 1
            state = next_state

    return B


def sample_transition_matrix_on_the_fly(counts, alpha, num_samples, seed=None):
    """
    Samples transition matrices from posterior and computes mean/std on-the-fly.

    Args:
        counts: (n, n) transition count matrix.
        alpha: (n, n) Dirichlet prior (default = ones).
        num_samples: number of posterior samples.
        seed: random seed (optional).

    Returns:
        mean_estimate: (n, n) posterior mean estimate.
        std_estimate: (n, n) posterior standard deviation estimate.
    """
    if seed is not None:
        np.random.seed(seed)

    n = counts.shape[0]
    if alpha is None:
        alpha = np.zeros_like(counts)

    mean_est = np.zeros((n, n))
    M2 = np.zeros((n, n))  # Sum of squares of differences from the mean

    """for s in range(1, num_samples + 1):
        print("computations for sample", s)
        for i in range(n):
            sample_row = np.random.dirichlet(np.maximum(counts[i] + alpha[i], 1))
            delta = sample_row - mean_est[i]
            mean_est[i] += delta / s
            M2[i] += delta * (sample_row - mean_est[i])

    std_est = np.sqrt(M2 / (num_samples - 1)) if num_samples > 1 else np.zeros_like(M2)
    return mean_est, std_est"""
    row_sums = (counts + alpha).sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # prevent division by zero
    mean_est = (counts + alpha) / row_sums

    return mean_est, 0

def matrix_modifier(A):
    """Given a square matrix, this function computes a modified matrix,
     which is suitable for visualization."""
    n = np.shape(A)[0] # dimension of the linear mapping
    A_mod = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A_mod[i, j] = max(0, A[i, j]) - min(0, A[j, i])
    return A_mod

def flow_computer(A, q: int, o: int, qsize: int, osize: int):
    """Given a square matrix which is entry-wise non-negative, this function computes compute a flow vector,
    that determines the dominant flow strength along x and y directions.
    """
    n = np.shape(A)[0]
    s = index_composer(q, o, qsize, osize)
    q_sum = 0 # summation of probabilities along queue direction
    o_sum = 0 # summation of probabilities along orbit/retry direction
    for s_prin in range(n):
        q_prin, o_prin = index_decomposer(s_prin, qsize, osize)
        q_sum += (q_prin - q) * A[s, s_prin]
        o_sum += (o_prin - o) * A[s, s_prin]
        #q_sum += max(-1, min( 1, (q_prin - q))) * A[s, s_prin]
        #o_sum += max(-1, min( 1, (o_prin - o))) * A[s, s_prin]
    return q_sum, o_sum

def viz_linear_mapping(A, qsize: int, osize: int,  num_x, num_y, show_equilibrium=True):
    """visualize the dynamics of a linear mapping"""
    if qsize > osize:
        x_to_y_range = int(qsize / osize)
    else:
        x_to_y_range = 1
        assert False, "For visualization, set queue size > orbit size (revisit this assumption)"

    # Downsample the i and j ranges for better visibility
    i_values = np.linspace(0, qsize / x_to_y_range, num_x, endpoint=False)  #
    j_values = np.linspace(0, osize, num_y, endpoint=False)  #

    # Create meshgrid for i and j values
    I, J = np.meshgrid(i_values, j_values)

    # Create arrays for the horizontal (U) and vertical (V) components
    U = np.zeros(I.shape)  # Horizontal component
    V = np.zeros(I.shape)  # Vertical component


    A_mod = matrix_modifier(A)

    # Compute magnitudes and angles for each (i, j)
    for idx_i, i in enumerate(i_values):
        for idx_j, j in enumerate(j_values):
            u, v = flow_computer(A, i * x_to_y_range, j, qsize, osize)
            U[idx_j, idx_i] = u
            V[idx_j, idx_i] = v
    U = gaussian_filter(U, sigma=1)
    V = gaussian_filter(V, sigma=1)
    # Compute magnitude (for color) and angle (for arrow direction)
    magnitude = np.sqrt(U ** 2 + V ** 2)  # Magnitude of the vector
    angle = np.arctan2(V, U)  # Angle of the vector (atan2 handles f_x=0 correctly)

    # Find the maximum absolute values
    max_mag = np.max(magnitude)

    # Normalize the horizontal (U) and vertical (V) components by the maximum values
    # magnitude_normalized = (magnitude / max_mag)

    # Define a fixed maximum arrow length for visibility
    fixed_max_length = qsize / (x_to_y_range * max(num_x, num_y))

    # Flatten the arrays for plotting
    I_flat = I.flatten()
    J_flat = J.flatten()
    U_flat = np.cos(angle).flatten() * fixed_max_length  # Normalize the direction to length fixed_max_length
    V_flat = np.sin(angle).flatten() * fixed_max_length  # Normalize the direction to length fixed_max_length
    # magnitude_flat = magnitude_normalized.flatten()
    magnitude_flat = magnitude.flatten()

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    # Create a colormap for the arrow colors based on the magnitude
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.min(magnitude_flat), vmax=np.max(magnitude_flat))
    colors = cmap(norm(magnitude_flat))

    # Plot the arrows using the fixed length and color by magnitude
    _ = ax.quiver(I_flat, J_flat, U_flat, V_flat, color=colors,
                  angles='xy', scale_units='xy', scale=1, width=0.003)

    # Add a colorbar based on the magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(magnitude_flat)  # Link the data to the ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)  # Attach the colorbar to the current axis

    if show_equilibrium:
        # Create a circle at the (almost) equilibrium point
        res, obj_val = self.equilibrium_computer(qsize, osize, arrival_rate, service_rate, mu_retry_base, mu_drop_base,
                                                 thread_pool,
                                                 tail_prob)
        if abs(obj_val) < .01:
            print("found an almost equilibrium point")
            circle = plt.Circle((res.x[0] / x_to_y_range, res.x[1]), .01 * qsize / x_to_y_range, color='red', fill=True)
            ax.add_artist(circle)

    # Get current tick positions on the x-axis
    xticks = ax.get_xticks()

    # Re-scale the tick labels to the correct numbers
    scaled_xticks = xticks * x_to_y_range
    scaled_xticks.astype(int)

    # Set the new scaled tick labels
    ax.set_xticklabels(scaled_xticks)

    # Set labels for the axes
    ax.set_xlabel('Queue length')
    ax.set_ylabel('Orbit length')

    # Display and save the plot
    # plt.show()
    plt.savefig("results/2D_flow.png")



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
    return int(total_ind)


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

def olen_average(pi, qsize, osize) -> float:
    """This function computes the average orbit length for a given prob distribution pi"""
    length = 0
    for n_retry_queue in range(osize):
        weight = 0
        for n_main_queue in range(qsize):
            weight += pi[index_composer(n_main_queue, n_retry_queue, qsize, osize)]
        length += weight * n_retry_queue
    return length

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

    def soft_constrained_with_gd(X):
        pi_seq = torch.tensor(X, dtype=torch.float32)  # Shape: (N+1, n)
        N, n = pi_seq.shape[0] - 1, pi_seq.shape[1]

        # Parameter: transition matrix
        P = nn.Parameter(torch.randn(n, n))

        optimizer = torch.optim.Adam([P], lr=1e-2)
        lambda1 = 10.0  # non-negativity penalty
        lambda2 = 10.0  # row sum penalty

        for epoch in range(1000):
            pred_pis = pi_seq[:-1] @ P
            data_loss = torch.sum((pi_seq[1:] - pred_pis) ** 2)

            nonneg_penalty = torch.sum(torch.relu(-P) ** 2)
            rowsum_penalty = torch.sum((P.sum(dim=1) - 1) ** 2)

            loss = data_loss + lambda1 * nonneg_penalty + lambda2 * rowsum_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        return P


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
        model_preds_q = []
        true_vals_q = []
        model_preds_o = []
        true_vals_o = []

        for pi_traj in zip(pi_seq):
            pi_traj = np.array(pi_traj).squeeze()
            T = len(pi_traj)
            pi_pred = list(pi_traj[:depth])  # True pi values for initialization
            q_pred = []
            q_true = []
            o_pred = []
            o_true = []
            for t in range(depth):
                q_pred.append(qlen_average(pi_traj[t], qsize, osize))
                q_true.append(qlen_average(pi_traj[t], qsize, osize))
                o_pred.append(olen_average(pi_traj[t], qsize, osize))
                o_true.append(olen_average(pi_traj[t], qsize, osize))
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
                o_pred.append(olen_average(y_pred, qsize, osize))
                o_true.append(olen_average(pi_traj[t], qsize, osize))

            model_preds_q.append(q_pred)
            true_vals_q.append(q_true)
            model_preds_o.append(o_pred)
            true_vals_o.append(o_true)

        return model_preds_q, true_vals_q, model_preds_o, true_vals_o

    def plot_predictions_vs_true(q_seq, model_preds_q, o_seq, model_preds_o,
                                 save_prefix="results/linear_model_traj"):
        for i, (true_q, pred_q) in enumerate(zip(q_seq, model_preds_q)):
            plt.figure(figsize=(10, 4))
            plt.plot(true_q, label="True q", marker='o')
            plt.plot(pred_q, label="Linear Model q", marker='x')
            plt.title(f"Trajectory {i}")
            plt.xlabel("Time Step")
            plt.ylabel("q value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_q_{i}.png")
            plt.close()
        for i, (true_o, pred_o) in enumerate(zip(o_seq, model_preds_o)):
            plt.figure(figsize=(10, 4))
            plt.plot(true_o, label="True o", marker='o')
            plt.plot(pred_o, label="Linear Model o", marker='x')
            plt.title(f"Trajectory {i}")
            plt.xlabel("Time Step")
            plt.ylabel("o value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_o_{i}_.png")
            plt.close()


# Loading the trajectories...
with open("data_generation/pi_seq.pkl", "rb") as f:
    pi_seq = pickle.load(f)
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("data_generation/o_seq.pkl", "rb") as f:
    o_seq = pickle.load(f)
with open("data_generation/q_seq_stochastic.pkl", "rb") as f:
    q_seq_stochastic = pickle.load(f)
with open("data_generation/o_seq_stochastic.pkl", "rb") as f:
    o_seq_stochastic = pickle.load(f)
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


sys.path.append("/Users/mahmoud/Documents/GITHUB")
sys.path.append("/Users/mahmoud/Documents/GITHUB/metafor")
sys.path.append("/Users/mahmoud/Documents/GITHUB/metafor/metafor")

from metafor.dsl.dsl import Server, Work, Source, Program
"""c, lam = generalization_guided_fit(q_seq[0][0:800], 100, 600)
c1, lam1, c, lam, _, _ = extrapolation_guided_fit(q_seq[0][0:800], 700, 1000)
q_ave_target, T_s_target = fit_least_squares(range(0,400), q_seq[0][0:400])
q_ave_target, T_s_target = c_fit, lam_fit = fit_curve_fit(range(0,800), q_seq[0][0:800])
q_ave_target, T_s_target = fit_c_lambda(q_seq[0][0:100])"""


#best_params = run_cmaes_optimization(10000, 1111, 9.7, 10, 9, 3, 100, 30, 0.5, 200)
"""q_ave_target = 45
T_s_target = 900 / 3
# best_params = evolutionary_optimization(q_ave_target, T_s_target, 9.7, 10, 9, 3, 100, 30, 10, 100)
best_params = run_cmaes_optimization(q_ave_target, T_s_target, 9.7, 10, 9, 3, 100, 30, 0.5, 200)"""


# Check if analytical ctmc is already computed
if not os.path.exists("results/Q_mat.npy"):
    sys.path.append("/Users/mahmoud/Documents/GITHUB/metafor")
    from metafor.dsl.dsl import Server, Work, Source, Program
    get_analytic_ctmc(lambdaa = 9.7, mu = 10, timeout_t = 9, max_retries = 3, qsize = 100, osize = 30)
else:
    print("Q_mat already exists. Skipping generation.")
Q = np.load("results/Q_mat.npy")

n_states = qsize * osize
counts = np.zeros((n_states, n_states), dtype=int)
traj_num = len(q_seq_stochastic)
for traj_idx in range(traj_num):
    traj_len = len(q_seq_stochastic[traj_idx])
    for i in range(traj_len - 1):
        q_t = min(qsize-1, q_seq_stochastic[traj_idx][i])
        q_next = min(qsize-1, q_seq_stochastic[traj_idx][i + 1])
        o_t = o_seq_stochastic[traj_idx][i]
        o_next = o_seq_stochastic[traj_idx][i + 1]
        s_t = index_composer(q_t, o_t, qsize, osize)
        s_next = index_composer(q_next, o_next, qsize, osize)
        if s_t >= n_states or s_next >= n_states:
            aaa = 1
        counts[s_t, s_next] += 1


# Simulate transitions to get count matrix
np.random.seed(42)

#B = compute_prior_counts(Q, step_time=0.5, beta=10000.0)
B = simulate_prior_counts(Q, step_time=.5, N=n_states, T=10)
# B = np.zeros((n_states, n_states))
# Sample from Bayesian posterior & compute mean estimate and uncertainty
posterior_mean, posterior_std = sample_transition_matrix_on_the_fly(counts, alpha = B, num_samples=100)

print("Posterior mean transition matrix:")
print(posterior_mean)

print("\nPosterior standard deviation:")
print(posterior_std)

theta = posterior_mean


# Q = get_analytic_ctmc(lambdaa = 9.671253344100766, mu = 10, timeout_t = 10.799982905445487, max_retries = 3,
                      # qsize = 100, osize = 30)
# theta = expm(Q * .5)

# theta = linear_model.soft_constrained_with_gd(X)

# theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
# theta = linear_model.train_linear_least_squares(X, Y) # without structure
# theta = linear_model.train_linear_least_squares_with_structure(pi_seq, qsize, osize)  # enforcing structure
# theta = linear_model.sgd_with_reset(X, Y)

# theta_red = linear_model.train_linear_least_squares(X_mod, Y_mod) # without structure
# theta_red = linear_model.convex_op(X_mod, Y_mod)
# theta_red = linear_model.convex_op_sparse(X_mod, Y_mod)

# theta = augment_matrix(theta_red, important_indices, np.shape(X)[1])

# viz_linear_mapping(theta, qsize, osize,  30, 30, False)

# print("the learning error is", np.linalg.norm(Y_mod - X_mod @ theta_red, 'fro') ** 2)
# distance_array = sparsity_measure(theta)


# Compute the model predictions
model_preds_q, true_vals_q, model_preds_o, true_vals_o = linear_model.simulate_linear_model(theta, pi_seq, depth, qsize, osize)

# Plot and compare the output of model and true trajectories
linear_model.plot_predictions_vs_true(true_vals_q, model_preds_q, model_preds_o, true_vals_o)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)

finish = 1
# learn_dtmc_transition_matrix(q_seq_stochastic, qsize, 2)