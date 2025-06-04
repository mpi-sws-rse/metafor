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


# List of functions and classes used for V2 autoencoder formulation:
class autoencoder():
    def get_trajectories(traj_num, X, Y, q_seq):
        """Getting a list of trajectories within the input dataset, hence taking the history length into account"""
        trajectory_list = []
        trajectory_length_list = []
        total_idx = 0  # idx with respect to the accumulated data made by all trajectories
        for traj_idx in range(traj_num):
            num_steps = len(q_seq[traj_idx]) - depth - 1  # Number of future steps to predict

            X_traj = X[total_idx: total_idx + num_steps + 1]
            Y_traj = Y[total_idx: total_idx + num_steps + 1]
            #
            trajectory_length_list.append(num_steps + 1)
            #
            trajectory_list.append(
                [torch.from_numpy(X_traj).float().unsqueeze(1), torch.from_numpy(Y_traj).float().unsqueeze(1)])
            total_idx += num_steps + 1
        return trajectory_list, trajectory_length_list

    def autoencoder_training(input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list):
        """Training the AE model"""
        # Create an instance of the model
        model = AutoEncoderModel(input_dim, latent_dim, output_dim)
        # Optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            loss = 0
            for traj_idx in range(traj_num):
                trajectory = trajectory_list[traj_idx][0]
                steps = list(range(0, trajectory_length_list[traj_idx]))
                # Use the first state x_0 as the input
                x0 = trajectory[0]  #
                # Target states
                target = trajectory_list[traj_idx][1]  #

                # Compute the predictions
                output = model(x0, steps)  #

                # Compute the loss (mean squared error)
                loss += loss_fn(output, target)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")
        return model

    def simulate_and_plot_from_initial_state(model, trajectory_list, true_q_seq, save_dir="./results/", prefix="traj"):
        """
        Args:
            model: a callable model such that model(x0, [i]) → prediction at time i
            trajectory_list: list of lists, each inner list holds the initial state x0 for a trajectory
            true_q_seq: list of true q trajectories (same length as trajectory_list)
            save_dir: where to save plots
            prefix: filename prefix for saved plots
        """
        traj_num = len(trajectory_list)
        q_seq_learned_model = [[] for _ in range(traj_num)]

        for traj_idx in range(traj_num):
            x0 = trajectory_list[traj_idx][0][0]  # initial state or sequence
            q_seq_learned_model[traj_idx].append(x0[0][0].numpy())

            for i in range(1, len(true_q_seq[traj_idx])):
                y = model(x0, [i]).detach().numpy()[0][0][0]
                q_seq_learned_model[traj_idx].append(y)

            # Plot true vs predicted
            plt.figure(figsize=(10, 4))
            plt.plot(np.array(true_q_seq[traj_idx]), label="True q", marker='o')
            plt.plot(np.array(q_seq_learned_model[traj_idx]), label="Model q", marker='x')
            plt.title(f"Trajectory {traj_idx}")
            plt.xlabel("Time step")
            plt.ylabel("q value")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{prefix}_{traj_idx}.png")
            plt.close()

        return q_seq_learned_model


class AutoEncoderModel(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super(AutoEncoderModel, self).__init__()
        # Encoder: maps x to latent space y
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, latent_dim)
        )
        # Decoder: maps latent representation y back to x-hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, output_dim)
        )
        # Trainable square matrix K of shape (latent_dim, latent_dim)
        self.K = nn.Parameter(torch.eye(latent_dim))

    def forward(self, x0, steps):
        """
        x0: tensor of shape [1, input_dim] (initial state)
        steps: list of integers representing future time steps (e.g., [1, 2, ..., N])
        Returns: tensor of predictions of shape [len(steps), batch_size, input_dim]
        """
        y0 = self.encoder(x0)  # Compute initial latent representation
        predictions = []
        for i in steps:
            # Compute K^i
            K_power = torch.matrix_power(self.K, i)
            # Propagate the latent state: y_i = K^i * y0
            y_i = torch.matmul(y0, K_power.t())
            # Decode the latent state to get x-hat
            xhat_i = self.decoder(y_i)
            predictions.append(xhat_i)
        # Stack predictions along a new dimension
        return torch.stack(predictions, dim=0)


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

# Evaluating the performance of least-squares optimizer

# Compute the LS gain
qsize = 100
osize = 30

input_dim = 2 * depth  # Input space dimension
output_dim = 2
latent_dim = 10  # Latent space dimension
num_epochs = 250

# Get trajectories within X
trajectory_list, trajectory_length_list = autoencoder.get_trajectories(traj_num, X, Y, q_seq)

model = autoencoder.autoencoder_training(
    input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list)

autoencoder.simulate_and_plot_from_initial_state(
    model=model,
    trajectory_list=trajectory_list,
    true_q_seq=q_seq,
    save_dir="./results/",
    prefix="q_model_vs_true"
)

# Analyzing the linear mapping
K_matrix = model.K.detach().cpu().numpy()
eigvals = np.linalg.eigvals(K_matrix)
# Printing sorted eigenvalues
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues for K_matrix:", eigvals_sorted)
