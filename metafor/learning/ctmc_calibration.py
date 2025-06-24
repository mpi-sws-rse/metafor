import os
import time

import numpy as np

import pickle
import matplotlib.pyplot as plt

import scipy.sparse as sp


from scipy.linalg import expm, eigvals, eig

import sys
import random
import cma
from functools import partial

os.makedirs("results", exist_ok=True)


from scipy.optimize import curve_fit
from scipy.optimize import least_squares
from scipy.optimize import minimize

sys.path.append("/Users/mahmoud/Documents/GITHUB")
sys.path.append("/Users/mahmoud/Documents/GITHUB/metafor")
sys.path.append("/Users/mahmoud/Documents/GITHUB/metafor/metafor")

from metafor.dsl.dsl import Server, Work, Source, Program

# List of functions used for LS estimation:
def simulate_linear_model(theta, theta_nom, pi_seq, depth, qsize, osize):
    """
    Autoregressive rollout using a linear model that predicts both [q_t, o_t]

    Args:
        theta: weight matrix of shape (2*depth, 2)
        pi_seq: list of real-valued sequences
        depth: number of historical steps used as input

    Returns:
        model_preds: list of predicted q sequences using the calibrated CTMC
        model_preds_nom: list of predicted q sequences using the nominal CTMC
    """
    model_preds = []
    model_preds_nom = []
    true_vals = []

    for pi_traj in zip(pi_seq):
        pi_traj = np.array(pi_traj).squeeze()
        T = len(pi_traj)
        pi_pred = list(pi_traj[:depth])  # True pi values for initialization
        pi_pred_nom = list(pi_traj[:depth])
        q_pred = []
        q_pred_nom = []
        q_true = []
        for t in range(depth):
            q_pred.append(qlen_average(pi_traj[t], qsize, osize))
            q_pred_nom.append(qlen_average(pi_traj[t], qsize, osize))
            q_true.append(qlen_average(pi_traj[t], qsize, osize))
        for t in range(depth, T):
            pi_input = pi_pred[-depth:]
            pi_input_nom = pi_pred_nom[-depth:]
            x = np.array(pi_input).squeeze()
            x_nom = np.array(pi_input_nom).squeeze()
            y_pred = x @ theta  #
            y_pred_nom = x_nom @ theta_nom  #

            """print("min value is", np.min(y_pred))
            print("max value is", np.max(y_pred))
            print("summation is", np.sum(y_pred))"""
            pi_pred.append(y_pred)
            pi_pred_nom.append(y_pred_nom)
            q_pred.append(qlen_average(y_pred, qsize, osize))
            q_pred_nom.append(qlen_average(y_pred_nom, qsize, osize))
            q_true.append(qlen_average(pi_traj[t], qsize, osize))

        model_preds.append(q_pred)
        model_preds_nom.append(q_pred_nom)
        true_vals.append(q_true)

    return model_preds, model_preds_nom, true_vals

def plot_predictions_vs_true(q_seq, model_preds, model_preds_nom, sampling_time, save_prefix="results/linear_model_traj"):
    for i, (true_q, pred_q, pred_q_nom) in enumerate(zip(q_seq, model_preds, model_preds_nom)):
        time_seq = [x * sampling_time for x in range(0, len(true_q))]
        plt.figure(figsize=(10, 5))
        plt.rc("font", size=14)
        plt.rcParams["figure.figsize"] = [10, 5]
        plt.rcParams["figure.autolayout"] = True
        plt.plot(time_seq, true_q, label="DES output", marker='o')
        plt.plot(time_seq, pred_q, label="Calibrated CTMC output", marker='x')
        plt.plot(time_seq, pred_q_nom, label="Nominal CTMC output", marker='x')
        # plt.title(f"Trajectory {i}")
        plt.xlabel("Time", fontsize=14)
        plt.ylabel("Average number of jobs", fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_{i}.png")
        plt.close()



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

    return c1, lam1, c_final, lam_final, k_aug, y_aug

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
    # best_params[3] = int(np.clip(round(best_params[3]), 1, 10))  # Ensure integer retry

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








# Loading the trajectories...
with open("data_generation/pi_seq.pkl", "rb") as f:
    pi_seq = pickle.load(f)
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("data_generation/r_seq.pkl", "rb") as f:
    o_seq = pickle.load(f)
with open("data_generation/q_seq_stochastic.pkl", "rb") as f:
    q_seq_stochastic = pickle.load(f)
with open("data_generation/o_seq_stochastic.pkl", "rb") as f:
    o_seq_stochastic = pickle.load(f)
traj_num = len(pi_seq)  # Number of trajectories within the dataset
depth = 1  # History length (should be replaced by one; not being considered for calibration experiments)

# server spec
qsize = 100 # queue bound
osize = 30 # orbit bound
lambdaa = 9.5
mu = 10
timeout = 9
max_retries = 3
sampling_time = 0.5
# CMAES parameters:
sigma = .5
max_iter = 30

# Uncomment the suitable approach below if CMAES takes reference settling time & pi_ss as references
# N1 = 100 # Number of datapoints used for training
# N2 = 600 # Number of datapoints used for validation
# N = 900 # Number of datapoints used for training+validation
# c, lam = generalization_guided_fit(q_seq[0][0:800], N1, N2) #
# c1, lam1, c, lam, _, _ = extrapolation_guided_fit(q_seq[0][0:800], N1, N2)
# c, lam = fit_least_squares(range(0,N), q_seq[0][0:N])
# c, lam = c_fit, lam_fit = fit_curve_fit(range(0,N), q_seq[0][0:N])
# c, lam = fit_c_lambda(q_seq[0][0:100])
# q_ave_target = c # sum(q_seq[0][-10:]) / 10
# T_s_target = 1 / lam # len(q_seq[0]) / 3
# best_params = evolutionary_optimization(q_ave_target, T_s_target, lambdaa, mu, timeout, max_retries, qsize, osize, sigma, max_iter)


# Uncomment if CMAES takes sequences of number of jobs and retries as references
dont_care_val = 1e8 # not important
best_params = run_cmaes_optimization(dont_care_val, dont_care_val, lambdaa, mu, timeout, max_retries,
                                     qsize, osize, sigma, max_iter)




lambda_mod = 9.433 # best_params[0][0] # 9.671253344100766
timeout_mod = 10.54 # best_params[0][1] # 10.799982905445487


Q = get_analytic_ctmc(lambdaa = lambda_mod, mu = mu, timeout_t = timeout_mod, max_retries = max_retries,
                      qsize = qsize, osize = osize)
Q_nom = get_analytic_ctmc(lambdaa = lambdaa, mu = mu, timeout_t = timeout, max_retries = max_retries,
                      qsize = qsize, osize = osize)
theta = expm(Q * sampling_time)
theta_nom = expm(Q_nom * sampling_time)

# Compute the model predictions
model_preds, model_preds_nom, true_vals = simulate_linear_model(theta, theta_nom, pi_seq, depth, qsize, osize)

# Plot and compare the output of the calibrated model and true trajectories
plot_predictions_vs_true(true_vals, model_preds, model_preds_nom, sampling_time)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)

