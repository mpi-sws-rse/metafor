import math
import os
import time
from typing import List

import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, copy

import pickle
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

os.makedirs("results", exist_ok=True)


def set_seed(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)   # hash-based ops
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_data(q_seq, o_seq): #, l_seq, d_seq, r_seq, s_seq, depth):
    """ Preparing input-output training datasets"""
    X = []
    Y = []
    # print(q_seq)
    # exit()
    #for q, o, l, d, r, s  in zip(q_seq, o_seq, l_seq, d_seq, r_seq, s_seq):
    for q, o in zip(q_seq, o_seq):
        T = len(q)
        for t in range(depth, T):
            x = q[t - depth:t] + o[t - depth:t]  # concatenate d history of features
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
    def simulate_linear_model(theta, q_seq, o_seq, l_seq, d_seq, r_seq, s_seq, depth):
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

        for q, o, l, d, r, s  in zip(q_seq, o_seq, l_seq, d_seq, r_seq, s_seq):
            T = len(q)
            q_pred = list(q[:depth])  # True q values for initialization
            o_hist = list(o[:depth])  # True o values for initialization
            l_hist = list(l[:depth])  # True l values for initialization
            d_hist = list(d[:depth])  # True d values for initialization
            r_hist = list(r[:depth])  # True r values for initialization
            s_hist = list(s[:depth])  # True s values for initialization
            for t in range(depth, T):
                q_input = q_pred[-depth:]
                o_input = o_hist[-depth:]
                l_input = l_hist[-depth:]
                d_input = d_hist[-depth:]
                #r_input = r_hist[-depth:]
                s_input = s_hist[-depth:]
                x = np.array(q_input + l_input)
                y_pred = x @ theta  #

                q_pred.append(y_pred[0])
                #o_hist.append(y_pred[1])
                l_hist.append(y_pred[1])
                #d_hist.append(y_pred[2])
                #r_hist.append(y_pred[1])
                #s_hist.append(y_pred[2])

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

        # Third row
        #A_theta[2, :] = theta[:, 2]  # next l

        # Fourth row
        #A_theta[3, :] = theta[:, 3]  # next d

        # Fifth row
        #A_theta[4, :] = theta[:, 4]  # next r

        # Sixth row
        #A_theta[5, :] = theta[:, 5]  # next s
        # Shift previous values down by 1
        A_theta[2:, :-2] = np.eye(2*d - 2)
        return A_theta



def pretrain_autoencoder_recon(model, trajectory_list, trajectory_length_list, pretrain_epochs=50, lr=1e-3, device='cpu'):
    """Quick reconstruction pretrain: minimize decoder(encoder(x)) ~ x (one-step)"""
    model.to(device)
    opt = torch.optim.Adam(list(model.encoder.parameters()) + list(model.decoder.parameters()), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(pretrain_epochs):
        opt.zero_grad()
        loss = 0.0
        count = 0
        for traj_idx in range(len(trajectory_list)):
            X_traj = trajectory_list[traj_idx][0].to(device)   # shape (T, 1, input_dim)
            # For each time in trajectory, reconstruct the input window
            # note: X_traj[t] has shape (1, input_dim)
            for t in range(X_traj.shape[0]):
                x = X_traj[t].squeeze(0)   # shape (input_dim)
                x = x.unsqueeze(0)         # shape (1, input_dim)
                z = model.encoder(x)
                xhat = model.decoder(z)
                loss = loss + loss_fn(xhat, x)
                count += 1
        loss = loss / (count + 1e-12)
        loss.backward()
        opt.step()
        if (ep % 10) == 0:
            print(f"[AE pretrain] Epoch {ep} recon loss: {loss.item():.6f}")
    print("[AE pretrain] done")
    return model

def build_latent_snapshots(model, trajectory_list, device='cpu'):
    """Construct Z and Zp by encoding consecutive inputs (windows) from trajectories.
       Returns Z (latent_dim x N), Zp (latent_dim x N) as numpy arrays (column snapshots).
    """
    model.to(device)
    model.eval()
    Z_list = []
    Zp_list = []
    with torch.no_grad():
        for traj_idx in range(len(trajectory_list)):
            X_traj = trajectory_list[traj_idx][0].to(device)  # shape (T, 1, input_dim)
            T = X_traj.shape[0]
            if T < 2:
                continue
            # encode each window to latent
            Z_traj = []
            for t in range(T):
                x = X_traj[t].squeeze(0).unsqueeze(0)  # shape (1, input_dim)
                z = model.encoder(x)                   # shape (1, latent_dim)
                Z_traj.append(z.cpu().numpy().reshape(-1))  # flatten to (latent_dim,)
            Z_traj = np.stack(Z_traj, axis=1)  # shape (latent_dim, T)
            # snapshots pairs (columns): z_0 -> z_1, z_1 -> z_2, ..., z_{T-2}->z_{T-1}
            Z_list.append(Z_traj[:, :-1])
            Zp_list.append(Z_traj[:, 1:])
    if len(Z_list) == 0:
        raise RuntimeError("No snapshot data found. Check trajectory_list")
    Z = np.concatenate(Z_list, axis=1)   # (latent_dim, N_total)
    Zp = np.concatenate(Zp_list, axis=1) # (latent_dim, N_total)
    return Z, Zp

def dmd_regression(Z, Zp, reg=1e-6):
    """Compute K such that Zp ≈ K Z using Tikhonov regularization:
       K = Zp @ Z^T @ (Z @ Z^T + reg*I)^{-1}
       Returns K as numpy array with shape (latent_dim, latent_dim).
    """
    # Z, Zp shape: (m, N)
    m, N = Z.shape
    # compute covariance-like matrices
    ZZt = Z @ Z.T   # (m,m)
    ZptZt = Zp @ Z.T  # (m,m)
    # regularize and invert
    reg_eye = reg * np.eye(m)
    inv = np.linalg.inv(ZZt + reg_eye)
    K = ZptZt @ inv
    return K


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

    def autoencoder_training(input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list,
                             seed = None):
        """Training the AE model"""
        # hyperparams
        rolling_len = 1  # number of future steps to include in each window
        step_time = 1  # stride for moving the start t, and spacing between horizons
        # Create an instance of the model
        model = AutoEncoderModel(input_dim, latent_dim, output_dim)
        # Set and record a reproducibility seed (keeps API unchanged)
        # --- minimal seeding right here ---
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        
        ##########################################################################

        device="cpu"
        model = pretrain_autoencoder_recon(model, trajectory_list, trajectory_length_list, pretrain_epochs=50, lr=1e-3, device=device)

        # Build latent snapshots Z, Zp
        Z, Zp = build_latent_snapshots(model, trajectory_list, device=device)
        print("Built latent snapshots shapes:", Z.shape, Zp.shape)  # (latent_dim, N)

        # Compute DMD (regularized least squares)
        reg = 1e-6
        K_init = dmd_regression(Z, Zp, reg=reg)  # shape (latent_dim, latent_dim)

        # Assign into model.K (careful: model.forward uses y_row @ K.T,
        # and we solved z_next_col = K * z_col, which is consistent)
        with torch.no_grad():
            model.K.data.copy_(torch.from_numpy(K_init).float().to(model.K.device))
            # model.b can be set to the empirical latent offset if you want (optional)
            # compute empirical bias: b ≈ mean(z_{t+1} - K z_t)
            bias = np.mean(Zp - (K_init @ Z), axis=1)  # shape (latent_dim,)
            model.b.data.copy_(torch.from_numpy(bias).float().to(model.b.device))

        # enforce spectral clipping / stability if you want
        model.spectral_clip_()

        ###########################################################################

        # Track the best model
        best_loss = float('inf')
        best_state = None
        best_epoch = -1

        # Optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=1*1e-4)
        loss_fn = nn.MSELoss()

        for epoch in range(num_epochs):
            model.train()
            optimizer.zero_grad()

            loss = 0
            for traj_idx in range(traj_num):
                T = trajectory_length_list[traj_idx]
                trajectory = trajectory_list[traj_idx][0]
                steps = list(np.arange(0, T, 1))
                # Use the first state x_0 as the input
                x0 = trajectory[0]
                # Target states
                target = trajectory_list[traj_idx][1][0:trajectory_length_list[traj_idx]]#[::1] #
                # Compute the predictions
                output = model(x0, steps) #
                #print("x0 ",x0,"  output ",output[0][0],"  target ",target[0][0],"  steps ",steps)
                # Compute the loss (mean squared error)
                #print(target[0:T,:,:].shape)
                loss += loss_fn(output, target[0:T,:,:])
                #print(" traj id ",traj_idx,"  loss ",loss)
                """for t in np.arange(0, T-1, step_time):
                    horizon = min(rolling_len, T - 1 - t)
                    steps = list(range(0, horizon+1))
                    #steps = list(np.arange(0, trajectory_length_list[traj_idx], 1))
                    #steps = list(range(0, trajectory_length_list[traj_idx]//10))
                    # Use the first state x_0 as the input
                    x0 = trajectory[t]  #
                    # Target states
                    #target = trajectory_list[traj_idx][1][0:trajectory_length_list[traj_idx]//10]#[::1]  #
                    target = trajectory_list[traj_idx][1][t:t+horizon + 1]  # [::1]
                    # Compute the predictions
                    output = model(x0, steps)  #

                    # Compute the loss (mean squared error)
                    loss += loss_fn(output, target)"""
            loss.backward()
            optimizer.step()
            # keep A stable; affine part guarantees one eigenvalue = 1 in augmented system
            model.spectral_clip_()

            # Update best snapshot
            curr = loss.item()
            if curr < best_loss:
                best_loss = curr
                best_state = copy.deepcopy(model.state_dict())
                best_epoch = epoch

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item()}")

        # Restore the best weights and attach metadata
        if best_state is not None:
            model.load_state_dict(best_state)
        model.training_seed = seed  # retrieve later to reproduce
        model.best_epoch = best_epoch
        model.best_loss = best_loss

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
            T = len(true_q_seq[traj_idx])
            steps = list(np.arange(0, T, 1))
            # Use the first state x_0 as the input
            x0 = trajectory_list[traj_idx][0][0]
            # Target states
            target = trajectory_list[traj_idx][1][0:trajectory_length_list[traj_idx]]#[::1] #
            # Compute the predictions
            model_output = model(x0, steps) #
            q_seq_learned_model[traj_idx] = model_output[:, 0, 0].detach().cpu().tolist()
            """x0 = trajectory_list[traj_idx][0][0]  # initial state or sequence
            q_seq_learned_model[traj_idx].append(x0[0][0].numpy())

            for i in range(1, len(true_q_seq[traj_idx])):
                y = model(x0, [i]).detach().numpy()[-1][0][0]
                q_seq_learned_model[traj_idx].append(y)"""

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
            nn.Linear(input_dim, 100 *1),
            nn.ReLU(),
            nn.Linear(100*1, 500*1),
            nn.ReLU(),
            nn.Linear(500*1, latent_dim)
        )
        # Decoder: maps latent representation y back to x-hat
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 500*1),
            nn.ReLU(),
            nn.Linear(500*1, 100),
            nn.ReLU(),
            nn.Linear(100*1, output_dim)
        )
        """self.encoder = nn.Sequential(nn.Linear(input_dim, latent_dim))#,
            #nn.Tanh(),nn.Linear(5*1, latent_dim))
        # Decoder: maps latent representation y back to x-hat
        self.decoder = nn.Sequential(nn.Linear(latent_dim, output_dim))#,
                                     #nn.Linear(5*1, output_dim))"""
        # Trainable square matrix K of shape (latent_dim, latent_dim)
        K0 = 0.8 * torch.eye(latent_dim)  # already stable
        #K0 = torch.rand(latent_dim,latent_dim) #gives nan
        #K0 = torch.rand(latent_dim,1)* torch.eye(latent_dim)
        self.K = nn.Parameter(K0)
        self.b = nn.Parameter(torch.zeros(latent_dim))

    def forward(self, x0, steps):
        """
        x0: tensor of shape [1, input_dim] (initial state)
        steps: list of integers representing future time steps (e.g., [1, 2, ..., N])
        Returns: tensor of predictions of shape [len(steps), batch_size, input_dim]
        """
        if len(steps) > 1: # call during training
            horizon = len(steps)
        else:
            horizon = steps[0]
        y_i = self.encoder(x0)  # Compute initial latent representation
        predictions = []
        for i in range(horizon):
            if i > 0:
                # Compute K^i
                #K_power = torch.matrix_power(self.K, i)
                # Propagate the latent state: y_i = K * y_i-1 + b
                y_i = torch.matmul(y_i, self.K.t()) + self.b
            # Decode the latent state to get x-hat
            xhat_i = self.decoder(y_i)
            predictions.append(xhat_i)
        # Stack predictions along a new dimension
        return torch.stack(predictions, dim=0)

    @torch.no_grad()
    def spectral_clip_(self, n_power_iter=2, eps=1e-12):
        """Project A so its spectral norm <= alpha (cheap power iteration)."""
        W = self.K
        # flatten to 2D just in case
        Wm = W.reshape(W.shape[0], -1)
        # power iteration
        u = torch.randn(Wm.size(0), device=W.device)
        u = u / (u.norm() + eps)
        for _ in range(n_power_iter):
            v = Wm.t().mv(u);
            v = v / (v.norm() + eps)
            u = Wm.mv(v);
            u = u / (u.norm() + eps)
        sigma = (u @ (Wm @ v)).item()
        if sigma > .99:
            W.mul_(.99 / (sigma + eps))

 
# Loading the trajectories...
with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
with open("data_generation/o_seq.pkl", "rb") as f:
    o_seq = pickle.load(f)
with open("data_generation/l_seq.pkl", "rb") as f:
    l_seq = pickle.load(f)
with open("data_generation/d_seq.pkl", "rb") as f:
    d_seq = pickle.load(f)
with open("data_generation/r_seq.pkl", "rb") as f:
    r_seq = pickle.load(f)
with open("data_generation/s_seq.pkl", "rb") as f:
    s_seq = pickle.load(f)

traj_num = len(q_seq) # Number of trajectories within the dataset
depth = 1 # History length, also known as depth in system identification

X, Y = prepare_training_data(q_seq, l_seq) #, l_seq, d_seq, r_seq, s_seq, depth)

# Evaluating the performance of least-squares optimizer

# Compute the LS gain
#theta = np.matmul(np.linalg.inv(np.matmul(X.T, X)), np.matmul(X.T, Y))
"""theta = linear_model.train_linear_least_squares(X, Y)

# Compute the model predictions
model_preds = linear_model.simulate_linear_model(theta, q_seq, o_seq, l_seq, d_seq, r_seq, s_seq, depth=depth)

# Plot and compare the output of model and true trajectories
linear_model.plot_predictions_vs_true(q_seq, model_preds)

# Construct the linear dynamics associated with theta
A_theta = linear_model.build_effective_transition_matrix(theta, depth=depth)

# Printing sorted eigenvalues
eigvals = np.linalg.eigvals(A_theta)
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues of the system:", eigvals_sorted)"""

set_seed(858257303)
input_dim = 2 * depth  # Input space dimension
output_dim = 2
latent_dim = 20  # Latent space dimension
num_epochs = 1000

# Get trajectories within X
trajectory_list, trajectory_length_list = autoencoder.get_trajectories(traj_num, X, Y, q_seq)


model = autoencoder.autoencoder_training(
    input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list, seed = 858257303)


autoencoder.simulate_and_plot_from_initial_state(
    model=model,
    trajectory_list=trajectory_list,
    true_q_seq=q_seq,
    save_dir="./results/",
    prefix="q_model_vs_true"
)

# Analyzing the linear mapping
K_matrix = model.K.detach().cpu().numpy()


with open("model.pkl", "wb") as f:
        pickle.dump(model,f)


with open("K_matrix.pkl", "wb") as f:
        pickle.dump((K_matrix,X,Y,trajectory_list,trajectory_length_list),f)

print("K_matrix ",K_matrix)

eigvals = np.linalg.eigvals(K_matrix)
# Printing sortd eigenvalues
eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
print("Eigenvalues for K_matrix:", eigvals_sorted)
