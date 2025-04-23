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

os.makedirs("results", exist_ok=True)

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

class linear_model():
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
            trajectory_list.append([torch.from_numpy(X_traj).float().unsqueeze(1), torch.from_numpy(Y_traj).float().unsqueeze(1)])
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
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("data_generation/o_seq.pkl", "rb") as f:
    o_seq = pickle.load(f)

traj_num = len(q_seq) # Number of trajectories within the dataset
depth = 10 # History length, also known as depth in system identification

X, Y = prepare_training_data(q_seq, o_seq, depth)

# Evaluating the performance of least-squares optimizer

# Compute the LS gain
#theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
theta = linear_model.train_linear_least_squares(X, Y)

# Compute the model predictions
model_preds = linear_model.simulate_linear_model(theta, q_seq, o_seq, depth=depth)

# Plot and compare the output of model and true trajectories
linear_model.plot_predictions_vs_true(q_seq, model_preds)

# Construct the linear dynamics associated with theta
A_theta = linear_model.build_effective_transition_matrix(theta, depth=depth)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(A_theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)



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
