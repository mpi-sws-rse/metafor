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


class linear_model():
    def train_linear_least_squares(X, Y):
        """
        Fit a linear model using least squares
        Returns: theta (weights)
        """
        theta, _, _, _ = lstsq(X, Y, rcond=None)
        return theta  # shape: (input_dim,)


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

        for pi_traj in zip(pi_seq):
            pi_traj = np.array(pi_traj).squeeze()
            T = len(pi_traj)
            pi_pred = list(pi_traj[:depth])  # True pi values for initialization
            q_pred = []
            for t in range(depth):
                q_pred.append(qlen_average(pi_traj[t], qsize, osize))
            for t in range(depth, T):
                pi_input = pi_pred[-depth:]
                x = np.array(pi_input).squeeze()
                y_pred = x @ theta  #

                pi_pred.append(y_pred)
                q_pred.append(qlen_average(y_pred, qsize, osize))

                

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








# Loading the trajectories...
with open("data_generation/pi_seq.pkl", "rb") as f:
    pi_seq = pickle.load(f)
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
traj_num = len(pi_seq) # Number of trajectories within the dataset
depth = 1 # History length, also known as depth in system identification

X, Y = prepare_training_data(pi_seq, depth)

# Evaluating the performance of least-squares optimizer

# Compute the LS gain
qsize = 100
osize = 20
#theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
theta = linear_model.train_linear_least_squares(X, Y)

# Compute the model predictions
model_preds = linear_model.simulate_linear_model(theta, pi_seq, depth, qsize, osize)

# Plot and compare the output of model and true trajectories
linear_model.plot_predictions_vs_true(q_seq, model_preds)



# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)



