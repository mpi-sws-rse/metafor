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
        x= 0 
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
            x = x + 5*np.arange(len(np.array(true_q_seq[traj_idx])))
            plt.plot(x,np.array(true_q_seq[traj_idx]), label="True q", marker='o',linewidth=1)
            plt.plot(x,np.array(q_seq_learned_model[traj_idx]), label="Model q", marker='x',linewidth=1)
            #plt.title(f"Trajectory {traj_idx+1}",fontsize=16)
            plt.xlabel("Time",fontsize=22)
            plt.ylabel("Average Queue Length",fontsize=22)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=22)
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{prefix}_{traj_idx}.pdf")
            plt.close()
            x = x[-1] + 100 #fault injection duration
        return q_seq_learned_model

file =  'K_matrix.pkl'

with open(file, "rb") as f:
        model,K_matrix,X,Y,trajectory_list,trajectory_length_list,Z_trajs = pickle.load(f)

# eigvals = np.linalg.eigvals(K_matrix)
# # Printing sortd eigenvalues
# eigvals_sorted = eigvals[np.argsort(-eigvals.real)]
# print( "   ",eigvals_sorted[0])
# exit()

with open("data_generation/q_seq.pkl", "rb") as f:
    q_seq = pickle.load(f)
q_seq[0] = q_seq[0][0::10]
q_seq[1] = q_seq[1][0::10]

simulate_and_plot_from_initial_state(
    model=model,
    trajectory_list=trajectory_list,
    true_q_seq=q_seq,
    save_dir="./results/",
    prefix="q_model_vs_true"
)


