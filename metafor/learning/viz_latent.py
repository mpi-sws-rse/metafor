from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx

def extract_latent_and_original_trajectories(model, trajectory_list):
    """Extract corresponding latent and original (input) trajectories."""
    all_latent, all_original = [], []

    model.eval()
    with torch.no_grad():
        for traj_idx in range(len(trajectory_list)):
            X_traj = trajectory_list[traj_idx][0]  # shape (T, 1, input_dim)
            z_traj = model.encoder(X_traj.squeeze(1))  # (T, latent_dim)
            all_latent.append(z_traj.cpu().numpy())
            all_original.append(X_traj.squeeze(1).cpu().numpy())
    return all_original, all_latent


def perform_clustering(all_original, all_latent, n_clusters=3):
    """Perform clustering in both spaces."""
    X_orig = np.concatenate(all_original, axis=0)
    Z_lat = np.concatenate(all_latent, axis=0)

    kmeans_orig = KMeans(n_clusters=n_clusters, random_state=0).fit(X_orig)
    kmeans_lat = KMeans(n_clusters=n_clusters, random_state=0).fit(Z_lat)

    labels_orig = kmeans_orig.predict(X_orig)
    labels_lat = kmeans_lat.predict(Z_lat)
    return labels_orig, labels_lat, kmeans_orig, kmeans_lat


def plot_cluster_trajectories(all_data, labels, title, save_path, pca=False):
    """Plot cluster-colored trajectories."""
    plt.figure(figsize=(8, 6))
    if pca and all_data[0].shape[1] > 2:
        pca_model = PCA(n_components=2)
        all_concat = np.concatenate(all_data, axis=0)
        reduced = pca_model.fit_transform(all_concat)
        offset = 0
        for traj in all_data:
            T = len(traj)
            z_reduced = reduced[offset:offset + T]
            traj_labels = labels[offset:offset + T]
            plt.scatter(z_reduced[:, 0], z_reduced[:, 1], c=traj_labels, cmap="viridis", s=10)
            offset += T
        plt.xlabel("PC1"); plt.ylabel("PC2")
    else:
        offset = 0
        for traj in all_data:
            T = len(traj)
            traj_labels = labels[offset:offset + T]
            plt.scatter(traj[:, 0], traj[:, 1], c=traj_labels, cmap="viridis", s=10)
            offset += T
    plt.title(title)
    plt.colorbar(label="Cluster ID")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def build_transition_matrix(labels, n_clusters):
    """Build empirical transition probability matrix."""
    P = np.zeros((n_clusters, n_clusters))
    for i in range(len(labels) - 1):
        P[labels[i], labels[i + 1]] += 1
    P = P / (P.sum(axis=1, keepdims=True) + 1e-12)
    return P


def plot_transition_graph(P, save_path):
    """Plot directed weighted graph from transition matrix."""
    G = nx.DiGraph()
    n = P.shape[0]
    for i in range(n):
        for j in range(n):
            if P[i, j] > 0.01:
                G.add_edge(i, j, weight=P[i, j])
    pos = nx.circular_layout(G)
    plt.figure(figsize=(5, 5))
    nx.draw(G, pos, with_labels=True, node_color="skyblue", node_size=1200, font_size=12)
    labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels={k: f"{v:.2f}" for k, v in labels.items()})
    plt.title("Metastable Transition Graph")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def compute_sojourn_times(labels):
    """Compute how long trajectories stay in each cluster before switching."""
    sojourns = []
    current_cluster = labels[0]
    duration = 1
    for i in range(1, len(labels)):
        if labels[i] == current_cluster:
            duration += 1
        else:
            sojourns.append(duration)
            duration = 1
            current_cluster = labels[i]
    sojourns.append(duration)
    return np.array(sojourns)


# === Run comparison experiment ===
all_original, all_latent = extract_latent_and_original_trajectories(model, trajectory_list)
labels_orig, labels_lat, km_orig, km_lat = perform_clustering(all_original, all_latent, n_clusters=3)

plot_cluster_trajectories(all_original, labels_orig,
                          title="Clusters in Original Nonlinear Space",
                          save_path="results/original_clusters.png")

plot_cluster_trajectories(all_latent, labels_lat,
                          title="Clusters in Latent (Koopman) Space",
                          save_path="results/latent_clusters.png", pca=True)

# Compute transition matrices
P_orig = build_transition_matrix(labels_orig, n_clusters=3)
P_lat = build_transition_matrix(labels_lat, n_clusters=3)

plot_transition_graph(P_lat, "results/latent_transition_graph.png")

# Compute sojourn times
sojourn_lat = compute_sojourn_times(labels_lat)
plt.hist(sojourn_lat, bins=20, color="teal", alpha=0.7)
plt.xlabel("Sojourn duration (timesteps)")
plt.ylabel("Frequency")
plt.title("Sojourn times in latent space clusters")
plt.tight_layout()
plt.savefig("results/latent_sojourn_times.png")
plt.close()

print("Transition matrix (latent):\n", np.round(P_lat, 3))
print("Mean sojourn time (latent):", np.mean(sojourn_lat))

