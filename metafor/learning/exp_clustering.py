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
import matplotlib.patches as mpatches

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

print(Z_trajs[0].shape)

X_traj = []
for i in range(2):
    t1 = []
    for x in trajectory_list[i]:
        l = [l[0] for l in x]
        t1.append(l)
    X_traj.append(np.array(t1))


print(len(X_traj),"  ",X_traj[0].shape)


print(len(trajectory_list),"  ",len(trajectory_list[0][0]),"  ",len(trajectory_list[0][0][0]))


def latent_one_step_residuals(Z_trajs, K, b=None):
    res_list = []
    for Z in Z_trajs:
        Z = np.asarray(Z)  # (T, m)
        pred = (Z[:-1] @ K.T) + (b.reshape(1,-1) if b is not None else 0)
        r = Z[1:] - pred
        res_list.append(r)
    R = np.vstack(res_list)
    return R

R = latent_one_step_residuals(Z_trajs, K_matrix, b=model.b.detach().cpu().numpy())
print("Residual mean norm:", np.mean(np.linalg.norm(R, axis=1)))
print("Residual max norm:", np.max(np.linalg.norm(R, axis=1)))


def estimate_P(K, kmax=500):
    return np.linalg.matrix_power(K, kmax)

def empirical_T_delta(K, P, Z0_list, delta, Tmax=2000):
    times = []
    for z0 in Z0_list:
        z = z0.copy()
        for t in range(Tmax):
            if np.max(np.abs(z - P @ z0)) < delta:
                times.append(t); break
            z = K @ z
        else:
            times.append(Tmax)
    return np.array(times)

P = estimate_P(K_matrix, kmax=2000)
times = empirical_T_delta(K_matrix, P, [Z[0] for Z in Z_trajs], delta=0.01, Tmax=2000)
print("empirical mean T_delta:", times.mean())

from sklearn.cluster import KMeans

Z_all = np.vstack(Z_trajs)
kmeans = KMeans(n_clusters=3).fit(Z_all)
labels = [kmeans.predict(Z) for Z in Z_trajs]

# build transitions
Kc = 5
Pmat = np.zeros((Kc,Kc))
for lab in labels:
    for a,b in zip(lab[:-1], lab[1:]):
        Pmat[a,b]+=1
Pmat = (Pmat.T / np.maximum(Pmat.sum(axis=1),1)).T  # row-stochastic
eigvals = np.linalg.eigvals(Pmat)
print("Markov eigenvalues:", np.sort(eigvals)[::-1])



from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

def cluster_latent_metastability(Z_trajs, n_clusters=3, plot=True, save_path="./results/metastability_analysis.pdf"):
    """
    Cluster latent trajectories and compute transition matrix between metastable sets.

    Args:
        Z_trajs: list of arrays [T_i x latent_dim]
        n_clusters: number of metastable clusters
        plot: whether to visualize cluster assignments and eigenvalues
        save_path: where to save plots

    Returns:
        P: transition probability matrix (n_clusters x n_clusters)
        eigvals: eigenvalues of P (sorted)
        labels: cluster labels per trajectory
    """
    # Stack all latent states
    Z_all = np.vstack(Z_trajs)
    
    # Fit k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels_all = kmeans.fit_predict(Z_all)
    
    # Assign back per trajectory
    labels = []
    idx = 0
    for Z in Z_trajs:
        labels.append(labels_all[idx:idx+len(Z)])
        idx += len(Z)

    # Build transition matrix
    P = np.zeros((n_clusters, n_clusters))
    for lab in labels:
        for a, b in zip(lab[:-1], lab[1:]):
            P[a, b] += 1
    P = (P.T / np.maximum(P.sum(axis=1), 1)).T  # row normalize

    # Eigen-decomposition
    eigvals = np.linalg.eigvals(P)
    eigvals = np.sort(eigvals)[::-1]

    print(f"Transition matrix (row-stochastic):\n{P}")
    print(f"Eigenvalues of P: {np.round(eigvals,4)}")
    #print(f"Spectral gap = 1 - |lambda_2| = {1 - np.abs(eigvals[1]):.4f}")

    if plot:
        plt.figure(figsize=(10,4))
        plt.subplot(1,2,1)
        plt.scatter(Z_all[:,0], Z_all[:,1], c=labels_all, cmap='viridis', s=5)
        plt.title(f"Latent clustering (k={n_clusters})")
        plt.xlabel("z₁"); plt.ylabel("z₂")
        plt.subplot(1,2,2)
        plt.scatter(eigvals.real, eigvals.imag, color='r')
        circle = plt.Circle((0,0),1,fill=False,linestyle='--',color='gray')
        plt.gca().add_artist(circle)
        plt.title("Transition matrix eigenvalues")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    return P, eigvals, labels

# ----- Generate latent trajectories -----
Z_trajs = []
for traj_idx, (X_traj, _) in enumerate(trajectory_list):
    with torch.no_grad():
        Z_trajs.append(model.encoder(X_traj.squeeze(1)).cpu().numpy())

# ----- Run metastability experiment -----
P, eigvals, labels = cluster_latent_metastability(Z_trajs, n_clusters=5)



import networkx as nx
from collections import Counter

def cluster_latent_metastability_full(
    Z_trajs,
    n_clusters=3,
    plot=True,
    save_prefix="./results/metastability"
):
    """
    Perform metastability analysis in latent space:
    - Cluster latent states
    - Build transition matrix
    - Plot metastability graph
    - Visualize latent trajectories colored by cluster ID
    - Compute empirical sojourn times

    Args:
        Z_trajs: list of arrays [T_i x latent_dim]
        n_clusters: number of metastable clusters
        plot: whether to visualize results
        save_prefix: prefix for saving figures
    """
    # Stack all latent states
    Z_all = np.vstack(Z_trajs)

    # Cluster latent states
    kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=42)
    labels_all = kmeans.fit_predict(Z_all)

    # Split labels per trajectory
    labels = []
    idx = 0
    for Z in Z_trajs:
        labels.append(labels_all[idx:idx+len(Z)])
        idx += len(Z)

    # Build empirical transition matrix
    P = np.zeros((n_clusters, n_clusters))
    for lab in labels:
        for a, b in zip(lab[:-1], lab[1:]):
            P[a, b] += 1
    P = (P.T / np.maximum(P.sum(axis=1), 1)).T  # row normalize

    # Compute eigenvalues
    eigvals = np.linalg.eigvals(P)
    eigvals = np.sort(eigvals)[::-1]
    spectral_gap = 1 - np.abs(eigvals[1])
    print(f"\n[Metastability analysis]")
    print(f"Eigenvalues: {np.round(eigvals,4)}")
    print(f"Spectral gap = {spectral_gap:.4f}")
    print(f"Transition matrix P:\n{np.round(P,3)}")

    #  Compute sojourn times per cluster
    sojourns = []
    for lab in labels:
        count = 1
        for i in range(1, len(lab)):
            if lab[i] == lab[i-1]:
                count += 1
            else:
                sojourns.append((lab[i-1], count))
                count = 1
        sojourns.append((lab[-1], count))
    sojourn_times = {c: [] for c in range(n_clusters)}
    for c, dur in sojourns:
        sojourn_times[c].append(dur)
    print("\n[Empirical sojourn times]")
    for c, durations in sojourn_times.items():
        if durations:
            print(f"Cluster {c}: mean={np.mean(durations):.2f}, std={np.std(durations):.2f}, n={len(durations)}")
        else:
            print(f"Cluster {c}: no visits")

    # Plot latent space clustering
    if plot:
        #plt.figure(figsize=(10,5))
        plt.scatter(Z_all[:,0], Z_all[:,1], c=labels_all, cmap='viridis', s=6)
        plt.title(f"Latent clustering (k={n_clusters})")
        plt.xlabel("Latent dim 1"); plt.ylabel("Latent dim 2")
        plt.grid(True)

        # Create a legend mapping cluster ID → color
        cmap = plt.get_cmap('viridis', n_clusters)
        handles = [
            mpatches.Patch(color=cmap(i), label=f"Cluster {i}")
            for i in range(n_clusters)
        ]
        plt.legend(handles=handles, title="Clusters", loc='best', fontsize=9)

        plt.tight_layout()
        plt.savefig(f"{save_prefix}_latent_clusters.pdf")
        plt.close()

        # Plot transition graph
        G = nx.DiGraph()
        for i in range(n_clusters):
            for j in range(n_clusters):
                if P[i,j] > 1e-3:
                    G.add_edge(i, j, weight=P[i,j])
        pos = nx.spring_layout(G, seed=42)
        #plt.figure(figsize=(6,6))
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=800)
        nx.draw_networkx_labels(G, pos)
        edges = nx.draw_networkx_edges(
            G, pos,
            arrowstyle='-|>',
            arrowsize=20,
            connectionstyle='arc3,rad=0.1',
            width=[3*w['weight'] for _,_,w in G.edges(data=True)]
        )
        nx.draw_networkx_edge_labels(G, pos,
            edge_labels={(i,j): f"{w['weight']:.2f}" for i,j,w in G.edges(data=True)}
        )
        plt.title("Metastable transition graph")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_transition_graph.pdf")
        plt.close()

        #  Plot latent trajectories colored by cluster ID
        for traj_idx, (Z, lab) in enumerate(zip(Z_trajs, labels)):
            plt.figure(figsize=(6,5))
            plt.scatter(Z[:,0], Z[:,1], c=lab, cmap='viridis', s=8)
            plt.title(f"Latent trajectory {traj_idx} (colored by cluster)")
            plt.xlabel("z₁"); plt.ylabel("z₂")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_traj_{traj_idx}.pdf")
            plt.close()

        #  Plot cluster ID time-series
        for traj_idx, lab in enumerate(labels[:3]):  # first few trajectories
            plt.figure(figsize=(8,2))
            plt.plot(lab, marker='o')
            plt.title(f"Cluster transitions (trajectory {traj_idx})")
            plt.xlabel("Time step"); plt.ylabel("Cluster ID")
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_time_traj_{traj_idx}.pdf")
            plt.close()

    return P, eigvals, labels, sojourn_times


P, eigvals, labels, sojourn_times = cluster_latent_metastability_full(
    Z_trajs=Z_trajs,
    n_clusters=5,
    plot=True,
    save_prefix="./results/metastability"
)






