import pickle
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("results", exist_ok=True)
os.makedirs("models", exist_ok=True)

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

def predict_next_state(q,r):
    # Loading the trained autoencoder model
    with open("../../koopman_AE_model/models/learned_model.pkl", "rb") as f:
        (model, K_matrix, X, Y,
         trajectory_list, trajectory_length_list, latent_trajs) = pickle.load(f)
    # Constructing the input
    x = np.array([q, r])
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    x0 = x.unsqueeze(0)  # [1, input_dim]

    steps = [1]  # predict 1 steps into the future

    with torch.no_grad():
        prediction = model(x0, steps)
    return [float(prediction[0][0][0].item() - x[0]), float(prediction[0][0][1].item() - x[1])]


def viz_2d(qsize, osize, num_points_x, num_points_y):
    if qsize > osize:
        x_to_y_range = int(qsize / osize)  # ensure that qsize is a multiple of osize
    else:
        assert False, "For visualization, set queue size > orbit size (revisit this assumption)"

    i_max = qsize / x_to_y_range  # used to make the plot x&y coordinates of arrow sizes reasonable
    j_max = osize

    # Downsample the i and j ranges for better visibility
    i_values = np.linspace(0, i_max, num_points_x, endpoint=False)  #
    j_values = np.linspace(0, j_max, num_points_y, endpoint=False)  #

    # Create meshgrid for i and j values
    I, J = np.meshgrid(i_values, j_values)

    # Create arrays for the horizontal (U) and vertical (V) components
    U = np.zeros(I.shape)  # Horizontal component
    V = np.zeros(I.shape)  # Vertical component

    # Compute magnitudes and angles for each (i, j)
    for idx_i, i in enumerate(i_values):
        for idx_j, j in enumerate(j_values):
            u, v = predict_next_state(int(i * x_to_y_range), int(j))
            U[idx_j, idx_i] = u
            V[idx_j, idx_i] = v

    # Compute magnitude (for color) and angle (for arrow direction)
    magnitude = np.sqrt(U ** 2 + V ** 2)  # Magnitude of the vector
    angle = np.arctan2(V, U)  # Angle of the vector (atan2 handles f_x=0 correctly)

    # Find the maximum absolute values
    max_mag = np.max(magnitude)

    # Normalize the horizontal (U) and vertical (V) components by the maximum values
    # magnitude_normalized = (magnitude / max_mag)

    # Define a fixed maximum arrow length for visibility
    fixed_max_length = i_max / max(num_points_x, num_points_y)

    # Flatten the arrays for plotting
    I_flat = I.flatten()
    J_flat = J.flatten()
    U_flat = np.cos(angle).flatten() * fixed_max_length  # Normalize the direction to length fixed_max_length
    V_flat = np.sin(angle).flatten() * fixed_max_length  # Normalize the direction to length fixed_max_length
    # magnitude_flat = magnitude_normalized.flatten()
    magnitude_flat = magnitude.flatten()

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 6))

    # Create a colormap for the arrow colors based on the magnitude
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=np.min(magnitude_flat), vmax=np.max(magnitude_flat))
    colors = cmap(norm(magnitude_flat))

    # Plot the arrows using the fixed length and color by magnitude
    _ = ax.quiver(I_flat, J_flat, U_flat, V_flat, color=colors,
                  angles='xy', scale_units='xy', scale=1, width=0.003)

    # Add a colorbar based on the magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(magnitude_flat)  # Link the data to the ScalarMappable
    cbar = plt.colorbar(sm, ax=ax)  # Attach the colorbar to the current axis

    # Get current tick positions on the x-axis
    xticks = ax.get_xticks()

    # Re-scale the tick labels to the correct numbers
    scaled_xticks = xticks * x_to_y_range
    scaled_xticks.astype(int)

    # Set the new scaled tick labels
    ax.set_xticklabels(scaled_xticks)

    # Set labels for the axes
    ax.set_xlabel('Queue length')
    ax.set_ylabel('Orbit length')

    # Display/save the plot
    # plt.show()
    plt.savefig("results/2D.png")


# Run the visualization
viz_2d(100, 50, 20, 20)