# Adapted from: https://github.com/mbrooker/simulator_example/blob/main/omission/omission.py

import heapq
import math
import multiprocessing
import os
import time
from typing import List

import numpy as np
import pandas

from Server import Server
from Statistics import StatData
from Client import Client, OpenLoopClient, OpenLoopClientWithTimeout
from plot import plot_results
from Job import exp_job, bimod_job
import cvxpy as cp

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import scipy
done: bool = False

import tensorflow as tf
import pickle
# Run a single simulation.
def sim_loop(max_t: float, client: Client, rho_fault, rho_reset, fault_start, fault_duration):
    t = 0.0
    q = [(t, client.generate, None)]
    # This is the core simulation loop. Until we've reached the maximum simulation time, pull the next event off
    #  from a heap of events, fire whichever callback is associated with that event, and add any events it generates
    #  back to the heap.
    while len(q) > 0 and t < max_t:
        # Get the next event. Because `q` is a heap, we can just pop this off the front of the heap.
        (t, call, payload) = heapq.heappop(q)
        # Execute the callback associated with this event
        new_events = call(t, payload)
        # If the callback returned any follow-up events, add them to the event queue `q` and make sure it is still a
        # valid heap
        if new_events is not None:
            q.extend(new_events)
            heapq.heapify(q)


# Run a simulation `run_nums` times, outputting the results to `x_fn`, where x in [1, num_runs].
# One simulation is run for each client in `clients`.
# `max_t` is the maximum time to run the simulation (in ms).
def run_sims(max_t: float, fn: str, num_runs: int, step_time: int, sim_fn, mean_t: float, rho,
             queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration,
             qsize, osize):
    file_names: List[str] = []

    for i in range(num_runs):
        print("Running simulation " + str(i + 1) + " time(s)")
        current_fn = str(i + 1) + '_' + fn
        file_names.append(current_fn)
        clients = sim_fn(mean_t, rho, queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
        with open(current_fn, "w") as f:
            f.write("t,rho,service_time,name,qlen,retries,dropped,runtime\n")
            for client in clients:
                client.server.file = f
                client.server.start_time = time.time()
                sim_loop(max_t, client, rho_fault, rho_reset, fault_start, fault_duration)

    #q_seq, o_seq = mean_variance_std_dev(file_names, max_t, num_runs, step_time, mean_t, rho, rho_fault, fault_start,
                                   #fault_duration, qsize, osize)
    #return err_seq

# Print the mean value, the variance, and the standard deviation at each stat point in each second
def mean_variance_std_dev(file_names: List[str], max_t: float, num_runs: int, step_time: int,
                          mean_t: float, rho, rho_fault, fault_start, fault_duration, qsize, osize):
    global done
    ss_size = qsize * osize
    num_traj = len(fault_start)  # number of continuous trajectories
    num_datapoints = []
    qlen_dataset = []
    olen_dataset = []
    for l in range(num_traj):
        if l == 0:
            traj_start = 0
        else:
            traj_start = fault_start[l-1] + fault_duration
        num_datapoints.append(math.ceil((fault_start[l]-traj_start)/step_time))
        qlen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
        olen_dataset.append(np.zeros((num_runs, num_datapoints[l])))
    run_ind = 0
    actual_data_num_seq = [[]*i for i in range(len(file_names))]
    for file_name in file_names:
        step_ind = 0
        k_overall = 1
        traj_idx = 0
        k_traj = 1
        discrete_time_point = 1 * step_time
        last_q_val = 0
        last_o_val = 0
        wait_ind = False # if true, must wait until the end of the fault period
        can_continue = True # will be set to True w
        with open(file_name, "r") as f:
            row_num = len(pandas.read_csv(file_name))
            for i, line in enumerate(f.readlines()):
                if i == 0:
                    continue  # drop the header
                split_line: List[str] = line.split(',')
                current_cont_time = float(split_line[0])
                if current_cont_time > fault_start[traj_idx] and current_cont_time < max_t:
                    actual_data_num_seq[run_ind].append(k_overall)
                    k_traj = 0
                    k_overall = 0
                    traj_idx += 1
                    wait_ind = True
                    discrete_time_point = (math.floor((fault_start[traj_idx - 1] + fault_duration)/step_time)*step_time)
                #fault_start_i = fault_start[traj_idx]  # corresponding to the end of current trajectory
                if  wait_ind:
                    can_continue = current_cont_time > (fault_start[traj_idx - 1] + fault_duration)
                    if can_continue:
                        wait_ind = False
                if (current_cont_time < max_t) and (current_cont_time > discrete_time_point) and can_continue:


                    n_mid = math.floor((current_cont_time-discrete_time_point)/step_time)
                    for i in range(n_mid):
                        qlen_dataset[traj_idx][run_ind, k_traj] = last_q_val
                        olen_dataset[traj_idx][run_ind, k_traj] = last_o_val
                        k_traj += 1
                        k_overall += 1
                    last_q_val = float(split_line[4]) * 1
                    last_o_val = float(split_line[5]) * 1
                    if k_traj == 90:
                        ffff  = 1
                    qlen_dataset[traj_idx][run_ind, k_traj] = last_q_val
                    olen_dataset[traj_idx][run_ind, k_traj] = last_o_val
                    k_traj += 1
                    k_overall += 1
                    discrete_time_point = (math.floor(current_cont_time/step_time)+1) * step_time
        actual_data_num_seq[run_ind].append(k_overall)
        run_ind += 1

    step_ind = np.min(actual_data_num_seq, axis = 0)
    q_seq = [[]*num_traj for l in range(num_traj)]
    o_seq = [[]*num_traj for l in range(num_traj)]
    for traj_idx in range(num_traj):
        for step in range(step_ind[traj_idx]):
            q_step = 0
            o_step = 0
            for run_ind in range(num_runs):
                q_step += qlen_dataset[traj_idx][run_ind, step] / num_runs
                o_step += olen_dataset[traj_idx][run_ind, step] / num_runs
            q_seq[traj_idx].append(q_step)
            o_seq[traj_idx].append(o_step)
    return q_seq, o_seq


def write_to_file(fn: str, stats_data: List[StatData], stat_fn, first: bool):
    with open(fn, "a") as f:
        if first:
            f.write(stats_data[0].header() + '\n')
        result: StatData = stat_fn(stats_data)
        f.write(result.as_csv() + '\n')


def compute_mean_variance_std_deviation(fn: str, max_t: float, step_time: int, num_runs: int, mean_t: float, rho, rho_fault, fault_start, fault_duration):
    current_folder = os.getcwd()
    file_names = [file for file in os.listdir(current_folder) if file.endswith(fn)]
    q_seq, o_seq = mean_variance_std_dev(file_names, max_t, step_time, num_runs, mean_t, rho, rho_fault, fault_start, fault_duration, 100, 20)    # fault_duration, qsize, osize
    return q_seq, o_seq


# Simulation with unimodal exponential service time and timeout
def make_sim_exp(mean_t: float, rho: float, queue_size: int, retry_queue_size: int, timeout_t: float,
                 max_retries: int, rho_fault: float, rho_reset: float, fault_start: float,
                 fault_duration: float) -> List[Client]:
    clients = []
    job_name = "exp"
    job_type = exp_job(mean_t)

    for name, client in [
        ("%s_open_timeout_%d" % (job_name, int(timeout_t)),
         OpenLoopClientWithTimeout(rho, job_type, timeout_t, max_retries, rho_fault, rho_reset, fault_start,
                                   fault_duration))
    ]:
        server = Server(1, name, client, rho, queue_size, retry_queue_size)
        client.server = server
        clients.append(client)
    return clients


# Simulation with bimodal service time, without and with timeout
def make_sim_bimod(mean_t: float, mean_t_2: float, bimod_p: float, rho: float, queue_size: int, retry_queue_size: int,
                   timeout_t: float, max_retries: int) -> List[Client]:
    clients = []
    job_name = "bimod"
    job_type = bimod_job(mean_t, mean_t_2, bimod_p)

    for name, client in [
        ("%s" % job_name, OpenLoopClient(rho, job_type)),
        ("%s_timeout_%d" % (job_name, int(timeout_t)), OpenLoopClientWithTimeout(rho, rho_fault, rho_reset, job_type,
                                                                                 timeout, max_retries, fault_start,
                                                                                 fault_duration))
    ]:
        server = Server(1, name, client, rho, queue_size, retry_queue_size)
        client.server = server
        clients.append(client)
    return clients


def run_discrete_experiment(max_t: float, runs: int, mean_t: float, rho: float, queue_size: int, retry_queue_size: int,
                            timeout_t: float, max_retries: int, total_time: float, step_time: int,
                            rho_fault: float, rho_reset: float, fault_start: float, fault_duration: float, qsize: int,
                            osize: int):
    results_file_name = "exp_results.csv"
    start_time = time.time()
    process = multiprocessing.Process(target=run_sims, args=(max_t, results_file_name, runs, step_time, make_sim_exp,
                                                             mean_t, rho, queue_size, retry_queue_size,
                                                             timeout_t,
                                                             max_retries, rho_fault, rho_reset, fault_start,
                                                             fault_duration, qsize, osize))
    process.start()
    process.join(total_time)
    if process.is_alive():
        print("Max simulation time reached, stopped the simulation")
        process.kill()
        # check if the mean, variance, and standard deviation have been computed; if not, compute them
        if not done:
            q_seq, o_seq = compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t, rho, rho_fault, fault_start, fault_duration)
    end_time = time.time()
    runtime = end_time - start_time
    print("Running time: " + str(runtime) + " s")
    q_seq, o_seq = compute_mean_variance_std_deviation(results_file_name, max_t, runs, step_time, mean_t, rho,
                                                       rho_fault, fault_start, fault_duration)
    # Save
    with open("q_seq.pkl", "wb") as f:
        pickle.dump(q_seq, f)
    with open("o_seq.pkl", "wb") as f:
        pickle.dump(o_seq, f)

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




# Define the Encoder network
class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 2000)
        self.fc2 = nn.Linear(2000, latent_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        y = self.fc2(x)
        return y

# Define the Linear map K
class LinearMap(nn.Module):
    def __init__(self, latent_dim):
        super(LinearMap, self).__init__()
        self.K = nn.Linear(latent_dim, latent_dim, bias=False)  # Linear map K (trainable)
    def forward(self, y):
        return self.K(y)

# Define the Decoder network
class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, 2000)
        self.fc2 = nn.Linear(2000, output_dim)

    def forward(self, y_prime):
        y_prime = torch.relu(self.fc1(y_prime))
        x_prime = self.fc2(y_prime)
        return x_prime

# Define the full Autoencoder structure
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.linear_map = LinearMap(latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def forward(self, x):
        y = self.encoder(x)
        y_prime = self.linear_map(y)
        x_prime = self.decoder(y_prime)
        return x_prime




# used for V2
class AutoEncoderModel(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoderModel, self).__init__()
        # Encoder: maps x to latent space y.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, latent_dim)
        )
        # Decoder: maps latent representation y back to x-hat.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, input_dim)
        )
        # Trainable square matrix K of shape (latent_dim, latent_dim).
        self.K = nn.Parameter(torch.eye(latent_dim))

    def forward(self, x0, steps):
        """
        x0: tensor of shape [batch_size, input_dim] (initial state)
        steps: list of integers representing future time steps (e.g., [1, 2, ..., N])
        Returns: tensor of predictions of shape [len(steps), batch_size, input_dim]
        """
        y0 = self.encoder(x0)  # Compute initial latent representation.
        predictions = []
        for i in steps:
            # Compute K^i using PyTorch's built-in matrix_power.
            K_power = torch.matrix_power(self.K, i)
            # Propagate the latent state: y_i = K^i * y0.
            y_i = torch.matmul(y0, K_power.t())
            # Decode the latent state to get x-hat.
            xhat_i = self.decoder(y_i)
            predictions.append(xhat_i)
        # Stack predictions along a new dimension.
        return torch.stack(predictions, dim=0)




# Used for V3
class AutoEncoderModelStochastic(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(AutoEncoderModelStochastic, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Encoder: maps x to a latent vector which we turn into a probability distribution.
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)
            # No softmax here; we'll apply it explicitly in forward.
        )

        # Decoder: maps the latent distribution back to x-hat.
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

        # Learnable parameter for K, a square matrix (latent_dim x latent_dim).
        # We'll convert this to a row-stochastic matrix via softmax.
        self.K_logits = nn.Parameter(torch.randn(latent_dim, latent_dim))

    def forward(self, x0, steps):
        """
        x0: tensor of shape [batch_size, input_dim] (initial state)
        steps: list of integers (time steps for which to predict)
        Returns: tensor of shape [num_steps, batch_size, input_dim]
        """
        # Compute latent representation for x0 and convert to a probability distribution.
        latent_logits = self.encoder(x0)  # shape: [batch_size, latent_dim]
        y0 = F.softmax(latent_logits, dim=1)  # y0 is a probability distribution over latent states

        # Convert raw K parameters to a row-stochastic matrix.
        K_stochastic = F.softmax(self.K_logits, dim=1)  # Each row sums to 1

        predictions = []
        for i in steps:
            # Compute K^i (using the row-stochastic matrix).
            K_power = torch.matrix_power(K_stochastic, i)
            # Propagate the latent state: y_i = y0 * (K^i)^T.
            y_i = torch.matmul(y0, K_power.t())
            # Decode the latent distribution.
            xhat_i = self.decoder(y_i)
            predictions.append(xhat_i)
        return torch.stack(predictions, dim=0)



# For v4 implementation
def prepare_training_batches(trajectories, history_len, steps):
    x_histories, x_futures = [], []
    for traj in trajectories:
        if len(traj) < history_len + steps:
            continue
        for i in range(len(traj) - history_len - steps + 1):
            x_histories.append(traj[i:i + history_len])
            x_futures.append(traj[i + history_len:i + history_len + steps])
    return np.stack(x_histories), np.stack(x_futures)

# -------------------------------
# Model components
# -------------------------------

class TemporalEncoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.lstm = layers.LSTM(latent_dim, return_state=True)

    def call(self, x):
        _, h, _ = self.lstm(x)
        return h  # (batch, latent_dim)

class LinearDynamics(tf.keras.layers.Layer):
    def __init__(self, latent_dim):
        super().__init__()
        self.A = self.add_weight(
            shape=(latent_dim, latent_dim),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, z0, steps):
        zs = []
        current = z0
        for _ in range(steps):
            current = tf.matmul(current, self.A)
            zs.append(current)
        return tf.stack(zs, axis=1)  # (batch, steps, latent_dim)

class Decoder_v4(tf.keras.Model):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(output_dim)

    def call(self, z_seq):
        batch_size, steps, dim = tf.shape(z_seq)[0], z_seq.shape[1], z_seq.shape[2]
        z_flat = tf.reshape(z_seq, [-1, dim])
        x_hat = self.fc2(self.fc1(z_flat))
        return tf.reshape(x_hat, [batch_size, steps, -1])

class LongTermDynamicsModel(tf.keras.Model):
    def __init__(self, input_dim, latent_dim, history_len):
        super().__init__()
        self.encoder = TemporalEncoder(input_dim, latent_dim)
        self.dynamics = LinearDynamics(latent_dim)
        self.decoder = Decoder_v4(latent_dim, input_dim)
        self.history_len = history_len

    def call(self, x_traj, steps):
        # Use last `history_len` steps for encoding
        z0 = self.encoder(x_traj[:, -self.history_len:, :])
        z_seq = self.dynamics(z0, steps)
        return self.decoder(z_seq)