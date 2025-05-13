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

def prepare_training_data(q_seq, o_seq, r_seq, depth):
    """ Preparing input-output training datasets"""
    X = []
    Y = []

    for q, o, r in zip(q_seq, o_seq, r_seq):
        T = len(q)
        for t in range(depth, T):
            x = q[t - depth:t] + o[t - depth:t] + r[t - depth:t]  # concatenate d history from q, o and r
            y = q[t: t + 1] + o[t: t + 1] + r[t: t + 1]
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
    def simulate_linear_model(theta, q_seq, o_seq, r_seq, depth):
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

        for q, o, r in zip(q_seq, o_seq, r_seq):
            T = len(q)
            q_pred = list(q[:depth])  # True q values for initialization
            o_hist = list(o[:depth])  # True o values for initialization
            r_hist = list(r[:depth])  # True r values for initialization

            for t in range(depth, T):
                q_input = q_pred[-depth:]
                o_input = o_hist[-depth:]
                r_input = r_hist[-depth:]
                
                x = np.array(q_input + o_input + r_input)
                y_pred = x @ theta  #

                q_pred.append(y_pred[0])
                o_hist.append(y_pred[1])
                r_hist.append(y_pred[2])

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
        A_theta = np.zeros((3*d, 3*d))

        # First row:
        A_theta[0, :] = theta[:, 0]  # next q

        # Second row
        A_theta[1, :] = theta[:, 1]  # next o

        # Third row
        A_theta[2, :] = theta[:, 2]  # next r

        # Shift previous q, o, r values down by 1
        A_theta[3:, :-3] = np.eye(3*d - 3)

        return A_theta





# Loading the trajectories...
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("data_generation/o_seq.pkl", "rb") as f:
    o_seq = pickle.load(f)
with open("data_generation/r_seq.pkl", "rb") as f:
    r_seq = pickle.load(f)

traj_num = len(q_seq) # Number of trajectories within the dataset
depth = 10 # History length, also known as depth in system identification

X, Y = prepare_training_data(q_seq, o_seq, r_seq, depth)

# Evaluating the performance of least-squares optimizer

# Compute the LS gain
#theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
theta = linear_model.train_linear_least_squares(X, Y)

# Compute the model predictions
model_preds = linear_model.simulate_linear_model(theta, q_seq, o_seq, r_seq, depth=depth)

# Plot and compare the output of model and true trajectories
linear_model.plot_predictions_vs_true(q_seq, model_preds)

# Construct the linear dynamics associated with theta
A_theta = linear_model.build_effective_transition_matrix(theta, depth=depth)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(A_theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)

