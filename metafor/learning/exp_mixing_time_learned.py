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




file =  'K_matrix.pkl'

with open(file, "rb") as f:
        model,K_matrix,X,Y,trajectory_list,trajectory_length_list,Z_trajs = pickle.load(f)

traj_num = len(trajectory_list)

# file =  'model.pkl'

# with open(file, "rb") as f:
#         model = pickle.load(f)

# print(len(Z_trajs))
# print(Z_trajs[0].shape)
# print(Z_trajs[1].shape)
# exit()
X_traj = []
for i in range(2):
    t1 = []
    for x in trajectory_list[i]:
        l = [l[0] for l in x]
        t1.append(l)
    X_traj.append(np.array(t1))


# print(len(X_traj),"  ",X_traj[0].shape)


# print(len(trajectory_list),"  ",len(trajectory_list[0][0]),"  ",len(trajectory_list[0][0][0]))




def estimate_P(K, kmax=500):
    return np.linalg.matrix_power(K, kmax)

def empirical_T_delta(K, P, Z0_list, delta, Tmax=2000):
    times = []
    #print(len(Z0_list),"  ",len(Z0_list[0]))
    #Z0_list = Z0_list[0:10]
    for z0 in Z0_list:
        z = z0.copy()
        for t in range(Tmax):
            if np.max(np.abs(z - P @ z0)) < delta:
                times.append(t); break
            z = K @ z
        else:
            times.append(Tmax)
    return np.array(times)



deltas = [0.05,0.1,0.2,0.3,0.5,0.8,1]
data = []
data1 = []
for d in deltas:
    P = estimate_P(K_matrix, kmax=100000)
    times = empirical_T_delta(K_matrix, P, [Z for Z in Z_trajs[0]], delta=d, Tmax=50000)
    print("empirical mean T_delta:", np.mean(times),"  ",np.std(times))
    data.append(times.mean())
    data1.append(times.std())

data = np.array(data)
data1 = np.array(data1)

plt.plot(deltas, data, color="tab:green",label='Mean')

# Plot shaded standard deviation
plt.fill_between(deltas, data - data1, data + data1, color='blue', alpha=0.2, label='±1 std. dev.')
# Label and show
plt.title("$\delta$-settling times")
plt.xlabel("$\delta$",fontsize=16)
plt.ylabel("Timesteps",fontsize=16)
plt.legend()
plt.savefig("Mixing_times.pdf")
plt.close()
