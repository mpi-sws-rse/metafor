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
import matplotlib.pyplot as plt
from numpy.linalg import lstsq


# used for V1 autoencoder

def simulate_and_plot_sum_model(model, q_seq, o_seq, d):
    for i, (q, o) in enumerate(zip(q_seq, o_seq)):
        T = len(q)
        q_pred = list(q[:d])
        o_pred = list(o[:d])

        for t in range(d, T):
            q_input = q_pred[t - d:t]
            o_input = o_pred[t - d:t]
            x_input = np.array(q_input + o_input).reshape(1, -1)

            # Predict q[t] , o[t]
            xx = torch.from_numpy(np.array(x_input)).float()
            q_next, o_next = model(xx).detach().numpy()[0]

            #
            q_pred.append(q_next)

            #
            o_pred.append(o_next)

        # Plot
        plt.figure(figsize=(10, 4))
        plt.plot(q, label="True q", marker='o')
        plt.plot(q_pred, label="Predicted q (model)", marker='x')
        plt.title(f"Trajectory {i}")
        plt.xlabel("Time Step")
        plt.ylabel("q value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        # plt.show()
        plt.savefig(f"Trajectory {i}.png")


def simulate_and_plot_sum_model_linear_evolved(model, q_seq, o_seq, d, steps=None):
    """
    model: assumed to have model.encoder(input), model.linear(z, t), model.decoder(z)
    """
    for i, (q, o) in enumerate(zip(q_seq, o_seq)):
        T = len(q)
        steps = T - d if steps is None else steps

        # 1. Prepare initial input (concatenated q+o history)
        q_init = q[:d]
        o_init = o[:d]
        x_input = np.array(q_init + o_init).reshape(1, -1)
        x_tensor = torch.from_numpy(x_input).float()

        # 2. Encode to latent space
        z0 = model.encoder(x_tensor)

        # 3. Generate future predictions using linear latent evolution
        q_pred = list(q_init)
        o_pred = list(o_init)

        for j in range(steps):
            z_t = model.linear_evolve(z0, j)       # z_t = A^j z0
            x_hat = model.decoder(z_t).detach().numpy().flatten()
            q_next, o_next = x_hat[0], x_hat[1]
            q_pred.append(q_next)
            o_pred.append(o_next)

        # 4. Plot true vs predicted q
        plt.figure(figsize=(10, 4))
        plt.plot(q, label="True q", marker='o')
        plt.plot(q_pred, label="Predicted q (model)", marker='x')
        plt.title(f"Trajectory {i}")
        plt.xlabel("Time Step")
        plt.ylabel("q value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"Trajectory_{i}.png")


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

    def linear_evolve(self, z0, t):
        z = z0
        for _ in range(t):
            z = self.linear_map(z)
        return z

    def forward(self, x):
        y = self.encoder(x)
        y_prime = self.linear_map(y)
        x_prime = self.decoder(y_prime)
        return x_prime













def prepare_training_data(q_seq, o_seq, depth):
    """ Preparing input-output training datasets"""
    X = []
    Y = []

    for q, o in zip(q_seq, o_seq):
        T = len(q)
        for t in range(depth, T):
            x = q[t - depth:t] + o[t - depth:t]  # concatenate d history from q and o
            y = q[t: t + 1] + o[t: t + 1]
            X.append(x)
            Y.append(y)

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def train_linear_least_squares(X, Y):
    """
    Fit a linear model using least squares
    Returns: theta (weights)
    """
    theta, _, _, _ = lstsq(X, Y, rcond=None)
    return theta  # shape: (input_dim,)



# List of functions used for LS estimation:
def simulate_linear_model(theta, q_seq, o_seq, depth):
    """
    Autoregressive rollout using a linear model that predicts both [q_t, o_t]

    Args:
        theta: weight matrix of shape (2*depth, 2)
        q_seq, o_seq: list of real-valued sequences
        depth: number of historical steps used as input

    Returns:
        model_preds: list of predicted q sequences
    """
    model_preds = []

    for q, o in zip(q_seq, o_seq):
        T = len(q)
        q_pred = list(q[:depth])  # True q values for initialization
        o_hist = list(o[:depth])  # True o values for initialization

        for t in range(depth, T):
            q_input = q_pred[-depth:]
            o_input = o_hist[-depth:]
            x = np.array(q_input + o_input)
            y_pred = x @ theta  #

            q_pred.append(y_pred[0])
            o_hist.append(y_pred[1])

        model_preds.append(q_pred)

    return model_preds

def plot_predictions_vs_true(q_seq, model_preds, save_prefix="linear_model_traj"):
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

def build_effective_transition_matrix(theta, depth):
    """
    Using (approximately) controllable canonical realization for theta;
    Builds the full autoregressive state-transition matrix A_theta such that:
    x_{t+1} = A_theta @ x_t

    Args:
        theta: (2*depth, 2) weight matrix from least squares
        depth: history length used in model

    Returns:
        A_theta: (2*depth, 2*depth) autoregressive system matrix
    """
    d = depth
    A_theta = np.zeros((2*d, 2*d))

    # First row:
    A_theta[0, :] = theta[:, 0]  # next q

    # Second row
    A_theta[1, :] = theta[:, 1]  # next o

    # Shift previous q and o values down by 1
    A_theta[2:, :-2] = np.eye(2*d - 2)

    return A_theta


# List of functions and classes used for V2 autoencoder formulation:
def get_trajectories(traj_num, X, q_seq):
    """Getting a list of trajectories within the input dataset, hence taking the history length into account"""
    trajectory_list = []
    trajectory_length_list = []
    total_idx = 0  # idx with respect to the accumulated data made by all trajectories
    for traj_idx in range(traj_num):
        num_steps = len(q_seq[traj_idx]) - depth - 1  # Number of future steps to predict

        X_traj = X[total_idx: total_idx + num_steps + 1]
        #
        trajectory_length_list.append(num_steps + 1)
        #
        trajectory_list.append(torch.from_numpy(X_traj).float().unsqueeze(1))
        total_idx += num_steps + 1
    return trajectory_list, trajectory_length_list

def autoencoder_training(input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list):
    """Training the """
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
            trajectory = trajectory_list[traj_idx]
            steps = list(range(1, trajectory_length_list[traj_idx]))
            # Use the first state x_0 as the input
            x0 = trajectory[0]  #
            # Target states
            target = trajectory[1:]  #

            # Compute the predictions
            output = model(x0, steps)  #

            # Compute the loss (mean squared error)
            loss += loss_fn(output, target)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")
    return model


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


def simulate_and_plot_from_initial_state(model, trajectory_list, true_q_seq, save_dir=".", prefix="traj"):
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
        x0 = trajectory_list[traj_idx][0]  # initial state or sequence
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







# Loading the trajectories...
with open("q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("o_seq.pkl", "rb") as f:
    o_seq = pickle.load(f)

traj_num = len(q_seq) # Number of trajectories within the dataset
depth = 10 # History length, also known as depth in system identification

X, Y = prepare_training_data(q_seq, o_seq, depth)

# Evaluating the performance of least-squares optimizer

# Compute the LS gain
#theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
theta = train_linear_least_squares(X, Y)

# Compute the model predictions
model_preds = simulate_linear_model(theta, q_seq, o_seq, depth=depth)

# Plot and compare the output of model and true trajectories
plot_predictions_vs_true(q_seq, model_preds)

# Construct the linear dynamics associated with theta
A_theta = build_effective_transition_matrix(theta, depth=depth)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(A_theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)



input_dim = 2 * depth  # Input space dimension
output_dim = 2
latent_dim = 10  # Latent space dimension
num_epochs = 250

# Get trajectories within X
trajectory_list, trajectory_length_list = get_trajectories(traj_num, X, q_seq)


model = autoencoder_training(input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list)


simulate_and_plot_from_initial_state(
    model=model,
    trajectory_list=trajectory_list,
    true_q_seq=q_seq,
    save_dir=".",
    prefix="q_model_vs_true"
)

# Analyzing the linear mapping
K_matrix = model.K.detach().cpu().numpy()
eigvals = np.linalg.eigvals(K_matrix)
# Printing sorted eigenvalues
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues for K_matrix:", eigvals_sorted)




q_seq_learned_model = [[] for i in range(traj_num)]
for traj_idx in range(traj_num):

    x0 = trajectory_list[traj_idx][0]
    q_seq_learned_model[traj_idx].append(x0)
    for i in range(1, len(q_seq[traj_idx])):
        y = model(x0, [i]).detach().numpy()[0][0][0]
        q_seq_learned_model[traj_idx].append(y)
# run V1 autoencoder-based formulation
depth = 1
dim = 2
n = dim

input_dim = 2  # Input space dimension
latent_dim = 100  # Latent space dimension
# V1 implementation
# Initialize the model, optimizer, and loss function
model = Autoencoder(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
# datasets
x = torch.from_numpy(X).float()
y = torch.from_numpy(Y).float()
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    x_prime = model(x)  # This is the predicted output x'_i

    # Compute loss: the difference between the predicted output x_prime and the target x_target
    loss = criterion(x_prime, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Reset gradients from previous step
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
#simulate_and_plot_sum_model(model, q_seq, o_seq, depth)
simulate_and_plot_sum_model_linear_evolved(model, q_seq, o_seq, depth)
print("V1 implementation of autoencoder is over")

# the block below can be run to compute LS estimates with no structural assumptions --> not stable
"""depth = 200
dim = 2
n = depth * dim
m = len(q_seq) - depth
X = np.zeros((m, n))
Y = np.zeros((m, dim))
for i in range(m):
    for j in range(dim):
        for k in range(depth):
            if j == 0: #
                X[i, j * depth + k] = q_seq[i + k]
                Y[i, j] = q_seq[i + depth]
            else:
                X[i, j * depth + k] = o_seq[i + k]
                Y[i, j] = o_seq[i + depth]
theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))

err_seq = []
for i in range(m):
    err = np.linalg.norm(np.matmul(X[i, :], theta) - Y[i, :])
    err_seq.append(err)
q_seq_learned_model = [q_seq[0], o_seq[0]]
x = [q_seq[0], o_seq[0]]
for i in range(1, m + 1):
    x =
    q_seq_learned_model.append(np.matmul(pi_seq[i - 1], P_mat))"""


# the block below can be run to compute LS estimates with structural assumptions; dependent col del is enableled.
"""theta = []
err_seq = []
P_mat = np.zeros((n,n))
for s in range(n):
    q, o = index_decomposer(s, qsize, osize)
    if q == 0:
        if o == 0:
            X = np.zeros((m, 3))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q+1, o, qsize, osize),
                             index_composer(q, o+1, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q+1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q, o+1, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
        elif o == osize - 1:
            X = np.zeros((m, 2))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q+1, o, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q+1, o, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
        else:
            X = np.zeros((m, 3))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                                   index_composer(q, o + 1, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q + 1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q, o + 1, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
    elif q == qsize - 1:
        if o == 0:
            X = np.zeros((m, 4))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q-1, o, qsize, osize),
                                   index_composer(q-1, o+1, qsize, osize), index_composer(q, o+1, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q, o+1, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
        elif o == osize - 1:
            X = np.zeros((m, 3))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q-1, o, qsize, osize),
                                   index_composer(q-1, o-1, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o-1, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
        else:
            X = np.zeros((m, 5))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o+1, qsize, osize),
                                   index_composer(q-1, o+1, qsize, osize), index_composer(q-1, o, qsize, osize),
                                   index_composer(q-1, o-1, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o-1, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
    else:
        if o == 0:
            X = np.zeros((m, 5))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q-1, o, qsize, osize),
                                   index_composer(q-1, o+1, qsize, osize), index_composer(q, o+1, qsize, osize),
                                   index_composer(q+1, o, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q+1, o, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
        elif o == osize - 1:
            X = np.zeros((m, 4))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q-1, o, qsize, osize),
                                   index_composer(q-1, o-1, qsize, osize), index_composer(q+1, o, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o-1, qsize, osize)],
                                   pi_seq[i][index_composer(q+1, o, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
        else:
            X = np.zeros((m, 6))
            Y = np.zeros((m, 1))
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o+1, qsize, osize), index_composer(q-1, o+1, qsize, osize), index_composer(q-1, o, qsize, osize), index_composer(q-1, o-1, qsize, osize), index_composer(q+1, o, qsize, osize)]
            for i in range(m):
                X[i, :] = np.array([pi_seq[i][index_composer(q, o, qsize, osize)],
                                   pi_seq[i][index_composer(q, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o+1, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o, qsize, osize)],
                                   pi_seq[i][index_composer(q-1, o-1, qsize, osize)],
                                   pi_seq[i][index_composer(q+1, o, qsize, osize)]])
                Y[i, :] = pi_seq[i + 1][index_composer(q, o, qsize, osize)]
            dependent_cols = dependent_columns(X)
            reduced_col_list = []
            for j in range(len(full_col_list)):
                if j not in dependent_cols:
                    reduced_col_list.append(full_col_list[j])
            X_mod = np.delete(X, dependent_cols, axis=1)
            theta_i = np.matmul(np.linalg.inv(np.matmul(X_mod.T, X_mod)), np.matmul(X_mod.T, Y))
            theta.append(theta_i)
            for j in range(len(reduced_col_list)):
                s_prin = reduced_col_list[j]
                P_mat[s_prin, s] = theta_i[j]
    err_avg = np.average(np.matmul(X_mod, theta[s]) - Y)
    if err_avg > 1:
        print("LS estimations is poor for state", s)
    err_seq.append(err_avg)
pi_seq_learned_model = [pi_seq[0]]
for i in range(1, m + 1):
    pi_seq_learned_model.append(np.matmul(pi_seq[i - 1], P_mat))"""

# The following tries to solve convex optimization with constraints to make the matrix stochastic -->
"""for i in range(m):
    X[i, :] = pi_seq[i][:]
    Y[i, :] = pi_seq[i + 1][:]

# Variables
P = cp.Variable((n, n))  # The unknown transition matrix P (n x n)

# Objective function
objective = cp.Minimize(cp.norm(Y - X @ P, 'fro') ** 2)

# Constraints
constraints = [P >= 0, cp.sum(P, axis=1) == 1]

# Form the problem
problem = cp.Problem(objective, constraints)

# Solve the problem
problem.solve(solver='SCS')

# Solution
P_opt = P.value
print("Optimal transition matrix P:\n", P_opt)"""

# convex constrained optimization with structure enforcement
"""for i in range(m):
    X[i, :] = pi_seq[i][:]
    Y[i, :] = pi_seq[i + 1][:]
theta = []
err_seq = []
P_mask = np.zeros((n, n))
for s in range(n):
    q, o = index_decomposer(s, qsize, osize)
    if q == 0:
        if o == 0:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                             index_composer(q, o + 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
        elif o == osize - 1:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
        else:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                             index_composer(q, o + 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
    elif q == qsize - 1:
        if o == 0:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q, o + 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
        elif o == osize - 1:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
        else:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o + 1, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
    else:
        if o == 0:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q, o + 1, qsize, osize),
                             index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
        elif o == osize - 1:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize), index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
        else:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o + 1, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize), index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                P_mask[s_prin, s] = 1
# Number of variables in P that are non-zero
num_vars = int(np.sum(P_mask))

# Define a reduced optimization variable for the non-zero elements of P
P_reduced = cp.Variable(num_vars)

# Create an index list to map the reduced variable back to the full matrix P
P_full = np.zeros((n, n), dtype=object)  # Full matrix P (with fixed zero entries)
var_idx = 0
for i in range(n):
    for j in range(n):
        if P_mask[i, j] == 1:
            P_full[i, j] = P_reduced[var_idx]
            var_idx += 1
        else:
            P_full[i, j] = 0  # Known zero elements

# Convert the full P matrix to a cvxpy expression
P_full_expr = cp.bmat(P_full)

# Objective function: minimize the Frobenius norm squared
objective = cp.Minimize(cp.norm(Y - X @ P_full_expr, 'fro') ** 2)

# Define the constraints
constraints = [P_full_expr >= 0, cp.sum(P_full_expr, axis=1) == 1]  # Row-wise sum constraint

# Form and solve the problem
prob = cp.Problem(objective, constraints)
prob.solve()

# Retrieve the optimized P matrix (with zeros where specified)
P_optimized = np.zeros((n, n))
var_idx = 0
for i in range(n):
    for j in range(n):
        if P_mask[i, j] == 1:
            P_optimized[i, j] = P_reduced.value[var_idx]
            var_idx += 1

print("Optimized P matrix:")
print(P_optimized)

pi_seq_learned_model = [pi_seq[0]]
for i in range(1, m + 1):
    pi_seq_learned_model.append(np.matmul(pi_seq[i - 1], P_mat))"""

# the following block runs SGD-based solution for the convex optimization problem, enabling resets.
# SGD parameters
"""learning_rate = 0.005
epochs = 1000
batch_size = 32

# Data: X (input distribution at time t), Y (output distribution at time t+1)
for i in range(m):
    X[i, :] = pi_seq[i][:]
    Y[i, :] = pi_seq[i + 1][:]

# Initialize transition matrix P randomly
P = np.random.rand(n, n)
P /= P.sum(axis=1, keepdims=True)  # Ensure P is stochastic (rows sum to 1)

# SGD optimization loop
for epoch in range(epochs):
    for i in range(0, m, batch_size):
        # Mini-batch of data
        X_batch = X[i:i + batch_size]
        Y_batch = Y[i:i + batch_size]

        # Compute the gradient of the loss function w.r.t. P
        gradient = -2 * X_batch.T @ (Y_batch - X_batch @ P)


        # Update P using the SGD update rule
        P -= learning_rate * gradient

        # Enforce constraints
        P = np.maximum(P, 0)  # Ensure non-negativity
        P /= P.sum(axis=1, keepdims=True)  # Normalize rows to sum to 1 (stochastic)

    # Optional: Print loss at each epoch
    loss = np.linalg.norm(Y - X @ P, 'fro') ** 2
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')
# Final optimized transition matrix
print("Optimized transition matrix P:\n", P)
pi_seq_learned_model = [pi_seq[0]]
for i in range(1, m+1):
    pi_seq_learned_model.append(np.matmul(pi_seq[i-1], P))"""

# SGD-based constrained optimization with structural enforcement
"""M = np.zeros((n, n))
for s in range(n):
    q, o = index_decomposer(s, qsize, osize)
    if q == 0:
        if o == 0:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                             index_composer(q, o + 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
        elif o == osize - 1:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
        else:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q + 1, o, qsize, osize),
                             index_composer(q, o + 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
    elif q == qsize - 1:
        if o == 0:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q, o + 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
        elif o == osize - 1:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
        else:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o + 1, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
    else:
        if o == 0:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q, o + 1, qsize, osize),
                             index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
        elif o == osize - 1:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize), index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
        else:
            full_col_list = [index_composer(q, o, qsize, osize), index_composer(q, o + 1, qsize, osize),
                             index_composer(q - 1, o + 1, qsize, osize), index_composer(q - 1, o, qsize, osize),
                             index_composer(q - 1, o - 1, qsize, osize), index_composer(q + 1, o, qsize, osize)]
            for j in range(len(full_col_list)):
                s_prin = full_col_list[j]
                M[s_prin, s] = 1
# Get the indices of the non-zero elements in the mask
non_zero_indices = np.argwhere(M == 1)

# Initialize the free variables randomly
P_free = np.random.rand(len(non_zero_indices))

# Data: X (input distribution at time t), Y (output distribution at time t+1)
for i in range(m):
    X[i, :] = pi_seq[i][:]
    Y[i, :] = pi_seq[i + 1][:]

# Hyperparameters for SGD
learning_rate = 0.01
num_iterations = 1000

# Helper function to reconstruct full matrix P from the free variables
def construct_P(P_free):
    P = np.zeros((n, n))
    for idx, (i, j) in enumerate(non_zero_indices):
        P[i, j] = P_free[idx]
    return P

# Objective function: Frobenius norm squared
def objective(P):
    return np.linalg.norm(Y - X @ P, 'fro') ** 2

# Gradient of the objective with respect to the free variables
def gradient(P_free):
    P = construct_P(P_free)
    gradient_full = -2 * X.T @ (Y - X @ P)
    grad_free = np.zeros_like(P_free)
    for idx, (i, j) in enumerate(non_zero_indices):
        grad_free[idx] = gradient_full[i, j]
    return grad_free

# SGD optimization loop
for iteration in range(num_iterations):
    # Compute the gradient with respect to the free variables
    grad = gradient(P_free)

    # Update the free variables using SGD
    P_free -= learning_rate * grad

    # Enforce constraints: P_free >= 0 (non-negativity)
    P_free = np.maximum(P_free, 0)

    # Reconstruct the full matrix P and enforce the row-sum constraint
    P = construct_P(P_free)
    row_sums = P.sum(axis=1, keepdims=True)
    P = P / np.maximum(row_sums, 1e-8)  # Normalize rows to sum to 1
    P_free = np.array([P[i, j] for (i, j) in non_zero_indices])  # Update free variables

    # Optional: print objective value every 100 iterations
    if iteration % 10 == 0:
        print(f"Iteration {iteration}, Objective: {objective(P)}")

# Final optimized matrix P
P_opt = construct_P(P_free)
np.save('P_opt.npy', P_opt)


#P_opt = np.load('P_opt.npy')
# Output the solution
print("Optimal P matrix:")
print(P_opt)
pi_seq_learned_model = [pi_seq[0]]
for i in range(1, m + 1):
    pi_seq_learned_model.append(np.matmul(pi_seq_learned_model[i - 1], P_opt))"""

# run autoencoder-based formulation
"""depth = 1
dim = 2
n = dim
m = len(q_seq) - depth
X = np.zeros((m, n))
Y = np.zeros((m, dim))
for i in range(m):
    for j in range(dim):
        for k in range(depth):
            if j == 0:  #
                X[i, j * depth + k] = q_seq[i + k]
                Y[i, j] = q_seq[i + depth]
            else:
                X[i, j * depth + k] = o_seq[i + k]
                Y[i, j] = o_seq[i + depth]
input_dim = n  # Input space dimension
latent_dim = 100  # Latent space dimension
# V1 implementation
# Initialize the model, optimizer, and loss function
model = Autoencoder(input_dim, latent_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()
# datasets
x = torch.from_numpy(X).float()
y = torch.from_numpy(Y).float()
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    x_prime = model(x)  # This is the predicted output x'_i

    # Compute loss: the difference between the predicted output x_prime and the target x_target
    loss = criterion(x_prime, y)

    # Backward pass and optimization
    optimizer.zero_grad()  # Reset gradients from previous step
    loss.backward()  # Backpropagation
    optimizer.step()  # Update model parameters

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
q_seq_learned_model = [q_seq[0]]
# Compute absolute differences between consecutive elements
diffs = np.abs(np.diff(q_seq))

# Find the first index of maximum difference
max_index = np.argmax(diffs)
x = torch.from_numpy(np.array([q_seq[0], o_seq[0]])).float()
for i in range(1, max_index):
    x = model(x)
    q_seq_learned_model.append(x.detach().numpy()[0])
x = torch.from_numpy(np.array([q_seq[max_index], o_seq[max_index]])).float()
for i in range(max_index, m + 1):
    x = model(x)
    q_seq_learned_model.append(x.detach().numpy()[0])
print("V1 implementation of autoencoder is over")"""

# V2 implementation (with the cost function that envolves summing up over future predictions)
"""num_steps = m - 1  # Number of future steps to predict.
batch_size = 1

# Create an instance of the model.
model = AutoEncoderModel(input_dim, latent_dim)

# Create dummy trajectory data.
# Here, we simulate a trajectory: x_0, x_1, ..., x_N.
trajectory_length = num_steps + 1
# Trajectory shape: [trajectory_length, batch_size, input_dim]
trajectory = torch.from_numpy(X).float().unsqueeze(1)

# Optimizer and loss function.
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop (demonstration for a few epochs).
num_epochs = 100
steps = list(range(1, trajectory_length))
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Use the first state x_0 as the input.
    x0 = trajectories[0]  # Shape: [batch_size, input_dim]
    # Target states: x_1, x_2, ..., x_N.
    target = trajectory[1:]  # Shape: [num_steps, batch_size, input_dim]

    # Compute the predictions: for each i, x̂_i = decoder(K^i * encoder(x_0))
    output = model(x0, steps)  # Output shape: [num_steps, batch_size, input_dim]

    # Compute the loss (mean squared error).
    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")
q_seq_learned_model = [q_seq[0]]
# Compute absolute differences between consecutive elements
diffs = np.abs(np.diff(q_seq))

# Find the first index of maximum difference
max_index = np.argmax(diffs)
x0 = trajectory[0]
for i in range(1, max_index):
    y = model(x0, [i]).detach().numpy()[0][0][0]
    q_seq_learned_model.append(y)
x0 = trajectory[max_index]
for i in range(max_index, m + 1):
    y = model(x0, [i-max_index]).detach().numpy()[0][0][0]
    q_seq_learned_model.append(y)"""

# V3 implementation (K is stochastic)
"""
# Create the model.
model = AutoEncoderModelStochastic(input_dim, latent_dim)

# Convert X to a torch tensor and add a batch dimension: shape becomes (N, 1, input_dim)
trajectory = torch.from_numpy(X).float().unsqueeze(1)

# Define optimizer and loss function.
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop (example for a few epochs)
num_epochs = 100
steps = list(range(1, m))
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Use the first state x0 (shape: [batch_size, input_dim]).
    x0 = trajectory[0]  # shape: (1, input_dim)
    # Target states: x_1, x_2, ..., x_N (shape: [num_steps, batch_size, input_dim]).
    target = trajectory[1:]  # shape: (num_steps, 1, input_dim)

    # Forward pass: predict future states.
    output = model(x0, steps)  # shape: (num_steps, batch_size, input_dim)

    loss = loss_fn(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item()}")

q_seq_learned_model = [q_seq[0]]
x0 = trajectory[0]
for i in range(1, m + 1):
    y = model(x0, [i]).detach().numpy()[0][0][0]
    q_seq_learned_model.append(y)"""


# plot the output qsizes
plot_results(step_time, q_seq, q_seq_learned_model, qsize, osize, 'discrete_results.png')























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