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
import argparse
import pickle
import matplotlib.pyplot as plt
from numpy.linalg import lstsq

os.makedirs("results", exist_ok=True)

depth = 1 # History length, also known as depth in system identification
def set_seed(seed: int):

    os.environ["PYTHONHASHSEED"] = str(seed)   # hash-based ops
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_training_data(q_seq, o_seq, depth): #, l_seq, d_seq, r_seq, s_seq, depth):
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

    
    if len(X[-1])==1:
        X = X[:-1]
    if len(Y[-1])==1:
        Y = Y[:-1]
    
            
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

    def autoencoder_training(input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list, traj_num,
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
  
                #loss += loss_fn(output, target[0:T,:,:])
                loss += loss_fn(output, target)
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

    def simulate_and_plot_from_initial_state(model, trajectory_list, trajectory_length_list, true_q_seq, save_dir="./results/", prefix="traj"):
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
            plt.title(f"Trajectory {traj_idx}",fontsize=16)
            plt.xlabel("Timestep",fontsize=16)
            plt.ylabel("q value",fontsize=16)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{prefix}_{traj_idx}.pdf")
            plt.close()

        return q_seq_learned_model
    

    @staticmethod
    def simulate_latent_trajectories(model, trajectory_list, traj_num, save_dir="./results/", prefix="latent"):
        """
        Simulate trajectories in the latent space using the learned Koopman operator K.

        Args:
            model: trained AutoEncoderModel with attributes encoder, K, b
            trajectory_list: list of (X_traj, Y_traj) pairs
            traj_num: number of trajectories
            save_dir: where to save latent trajectory plots
            prefix: filename prefix for saved plots
        """
        K = model.K.detach().cpu()
        b = model.b.detach().cpu()

        latent_trajectories = []

        for traj_idx in range(traj_num):
            # get initial input
            x0 = trajectory_list[traj_idx][0][0]  # initial state [1, input_dim]
            y0 = model.encoder(x0).detach().cpu()  # encode to latent space

            T = len(trajectory_list[traj_idx][0])  # trajectory length
            y_seq = [y0.squeeze().numpy()]

            # propagate in latent space
            y = y0
            for t in range(1, T):
                y = y @ K.T + b
                y_seq.append(y.squeeze().numpy())

            latent_trajectories.append(np.array(y_seq))

            # optional: plot first 2 dims
            plt.figure(figsize=(6, 5))
            y_seq = np.array(y_seq)
            plt.plot(y_seq[:, 0], y_seq[:, 1], '-o', markersize=2)
            plt.title(f"Latent trajectory {traj_idx}",fontsize=16)
            plt.xlabel("Latent dim 1",fontsize=16)
            plt.ylabel("Latent dim 2",fontsize=16)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{prefix}_{traj_idx}.png")
            plt.close()

        return latent_trajectories

    def save_autoencoder(model, path, input_dim, latent_dim, output_dim, optimizer=None, extra=None):
        ckpt = {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "latent_dim": latent_dim,
            "output_dim": output_dim,
            "best_epoch": getattr(model, "best_epoch", None),
            "best_loss": getattr(model, "best_loss", None),
            "training_seed": getattr(model, "training_seed", None),
        }
        if optimizer is not None:
            ckpt["optimizer_state_dict"] = optimizer.state_dict()
        if extra is not None:
            ckpt["extra"] = extra

        torch.save(ckpt, path)


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



def main():
    """ 
    Main function 
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="files", type=str, default="data")
    parser.add_argument("--epochs", help="number of epochs", type=int, default=1000)

    args = parser.parse_args()
    data_dir =  args.data_dir
    epochs = args.epochs

    
    # Loading the trajectories...
    with open(data_dir+"/q_seq.pkl", "rb") as f:
        q_seq = pickle.load(f)

    # with open("data_generation/o_seq.pkl", "rb") as f:
    #     o_seq = pickle.load(f)
    with open(data_dir+"/l_seq.pkl", "rb") as f:
        l_seq = pickle.load(f)

    #scaling factor  - sampling data every 10 steps

    if len(q_seq[0])>5000:
        q_seq[0] = q_seq[0][0::10]
        q_seq[1] = q_seq[1][0::10]
        l_seq[0] = l_seq[0][0::10]
        l_seq[1] = l_seq[1][0::10]

    # with open("data_generation/d_seq.pkl", "rb") as f:
    #     d_seq = pickle.load(f)
    # with open("data_generation/r_seq.pkl", "rb") as f:
    #     r_seq = pickle.load(f)
    # with open("data_generation/s_seq.pkl", "rb") as f:
    #     s_seq = pickle.load(f)

    traj_num = len(q_seq) # Number of trajectories within the dataset

    depth=1
    X, Y = prepare_training_data(q_seq, l_seq, depth) #, l_seq, d_seq, r_seq, s_seq, depth)

    # q_seq[0] = q_seq[0][0:100]
    # #q_seq[1] = q_seq[1][0:100] 
    # q_seq[1] = q_seq[0][0:100] #this basically enforces only the intital traj 
    #print(len(q_seq),"  ",len(q_seq[0]))

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
    num_epochs = epochs




    # Get trajectories within X
    trajectory_list, trajectory_length_list = autoencoder.get_trajectories(traj_num, X, Y, q_seq)


    model = autoencoder.autoencoder_training(
        input_dim, latent_dim, output_dim, num_epochs, trajectory_list, trajectory_length_list, traj_num, seed = 858257303)


    autoencoder.simulate_and_plot_from_initial_state(
        model=model,
        trajectory_list=trajectory_list,
        trajectory_length_list=trajectory_length_list,
        true_q_seq=q_seq,
        save_dir="./results/",
        prefix="q_model_vs_true"
    )

    latent_trajs = autoencoder.simulate_latent_trajectories(
        model=model,
        trajectory_list=trajectory_list,
        traj_num=traj_num,
        save_dir="./results/",
        prefix="latent_traj"
    )

    # Analyzing the linear mapping
    K_matrix = model.K.detach().cpu().numpy()



    with open("models/learned_model.pkl", "wb") as f:
            pickle.dump((model,K_matrix,X,Y,trajectory_list,trajectory_length_list,latent_trajs),f)

    print("K_matrix ",K_matrix)

    eigvals = np.linalg.eigvals(K_matrix)
    # Printing sortd eigenvalues
    eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
    print("Eigenvalues for K_matrix:", eigvals_sorted)



if __name__ == '__main__':
    main()
