import pickle
import numpy as np
import torch
# from learning_from_simulation_data import AutoEncoderModel

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random, copy
import argparse
import os

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



def settling_time(t, y, epsilon=0.02):
    """
    Finds the time at which the trajectory enters an epsilon band around 
    the final value and stays within it thereafter.

    Parameters
    ----------
    t : array_like
        Time vector (same length as y).
    y : array_like
        Trajectory or response.
    epsilon : float
        Band half-width (absolute).

    Returns
    -------
    ts : float or None
        Settling time (None if it never settles).
    """
    t = np.asarray(t)
    y = np.asarray(y)
    final_value = 60 #y[-1]

    lower = final_value-epsilon
    upper = final_value+epsilon

    for i in range(len(y)-200):
        # If from this point onward the signal stays within the band
        if np.all((y[i] >= lower) & (y[i] <= upper)):
            return t[i]
    return np.inf  # never settles




def koopman_settling_time(A, L_e, L_e_prime, D_S, delta, eps=1e-12):
    """
    Compute the Koopman-based delta-settling time lower bound.

    Parameters
    ----------

    """
    # Compute eigenvalues
    eigvals = np.linalg.eigvals(A)
    eigvals = np.sort(np.abs(eigvals))[::-1]  # sort descending by magnitude

    lambda_2 = eigvals[0]
    if lambda_2 >= 1 - eps:
        # no decay: infinite settling time
        return np.inf

    # Compute the bound
    numerator = np.log((L_e * D_S) / (L_e_prime * delta))
    denominator = -np.log(lambda_2)
    T_delta = numerator / denominator

    return T_delta




def main():
    """ 
    Main function 
    """

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", help="model files", type=str, default="models/learned_model.pkl")
    parser.add_argument("--data_path", help="data files", type=str, default="data/sim_data.pkl")
    args = parser.parse_args()
    file = args.model_path
    dfile = args.data_path

    with open(file, "rb") as f:
            model,K_matrix,X,Y,trajectory_list,trajectory_length_list,Z_trajs = pickle.load(f)

    traj_num = len(trajectory_list)

    # Example Koopman matrix (diagonalizable, spectral radius ~1)
    A = K_matrix

    # Encoder Lipschitz constants
    L_uB = 1.66      # upper Lipschitz
    L_lB = 0.82 # co-Lipschitz (lower bound)

    # Diameter of state space
    D_S = 103.05

   
    data_list = []
    

    with open(dfile, "rb") as f:
        data = pickle.load(f)
        data_list.append(data)
        print(f"Loaded {file} with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

    data = data_list[0]
    print(len(data))
    step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, rho = data
    print(len(qlen_ave))
    time1 = [i * step_time for i in list(range(0, len(qlen_ave)))]

    t = np.arange(len(qlen_ave))
    y = qlen_ave


    epsilons = np.arange(1,30,1)
    data1 = []
    for ep in epsilons:
        ts = settling_time(t, y, epsilon=ep)
        print(f"Settling time = {ts:.3f} s")

        data1.append(ts)

    data1 = np.array(data1)

    plt.plot(epsilons, data1, color="tab:blue",label='Simulation')

    deltas = np.arange(1,30,1)
    data = []
    for delta in deltas:
        ts = koopman_settling_time(A, L_uB, L_lB, D_S, delta=delta)
        print(f"Settling time = ",ts*10)

        data.append(ts*10)

    data = np.array(data)

    plt.plot(deltas, data, color="tab:green",label='Theoretical upper bound')

    #plt.title("Settling time in simulations")
    plt.xlabel("$\delta$",fontsize=22)
    plt.ylabel("Time",fontsize=22)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("results/Mixing_times_simulation.pdf")
    plt.close()



if __name__ == '__main__':
    main()
