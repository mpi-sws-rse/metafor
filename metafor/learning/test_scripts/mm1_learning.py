import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt

def create_mm1_generator_matrix(queue_length, lambdaa, mu):
    """Construct Generator Matrix"""
    N = queue_length + 1
    Q = np.zeros((N, N))
    for i in range(N):
        if i < queue_length:
            Q[i, i + 1] = lambdaa
        if i > 0:
            Q[i, i - 1] = mu
        Q[i, i] = -np.sum(Q[i])
    return Q


def simulate_ctmc(Q, initial_state, tau, max_time):
    """Monte Carlo Simulation"""
    N = Q.shape[0]
    state = initial_state
    t = 0.0
    next_sample_time = 0.0
    trajectory = []

    while next_sample_time < max_time:
        # Wait for next reaction
        total_rate = -Q[state, state]
        if total_rate == 0:
            break

        dt = np.random.exponential(1 / total_rate)
        t += dt

        # Advance and sample states at fixed tau
        while next_sample_time <= t and next_sample_time < max_time:
            trajectory.append(state)
            next_sample_time += tau

        # Choose next state
        rates = Q[state].copy()
        rates[state] = 0
        probs = rates / total_rate
        state = np.random.choice(N, p=probs)

    return trajectory


def estimate_empirical_distribution(trajectories, N):
    """
    Empirical Distribution Estimation...
    trajectories: list of lists (each list is a trajectory of states)
    N: total number of states (max state + 1)

    Returns:
        probs: list of empirical distributions at each time step
               shape: (T, N) where T = number of time steps
    """
    T = len(trajectories[0])  # assume all trajectories are same length
    counts = np.zeros((T, N))

    for traj in trajectories:
        for t, s in enumerate(traj):
            counts[t, s] += 1

    return counts / len(trajectories)  # shape (T, N)

def estimate_transition_matrix_from_two_trajectories(emp1, emp2):
    """
    Least Squares Estimation of Embedded DTMC...
    emp1, emp2: empirical distribution trajectories of shape (T, N)
    Returns: least-squares estimate P_hat of shape (N, N)
    """
    X = np.vstack([emp1[:-1], emp2[:-1]])   # shape (2*T-2, N)
    Y = np.vstack([emp1[1:], emp2[1:]])     # shape (2*T-2, N)
    P_hat, _, _, _ = lstsq(X, Y, rcond=None)
    return P_hat

def compare_avg_queue_length(P_hat, pi_0, emp_distributions, label):
    """Compare model-based trajectory against empirical distribution"""
    T = emp_distributions.shape[0]
    state_values = np.arange(emp_distributions.shape[1])  # [0, 1, ..., N]

    # Model-based expected values: pi_0 @ P^k @ s
    model_avg = []
    pi = pi_0.copy()
    for _ in range(T):
        avg = np.dot(pi, state_values)
        model_avg.append(avg)
        pi = pi @ P_hat

    # Empirical expected values
    empirical_avg = emp_distributions @ state_values

    # Plot comparison
    plt.figure(figsize=(10, 5))
    plt.plot(empirical_avg, label="Empirical", linestyle="--")
    plt.plot(model_avg, label="Model")
    plt.title(f"Average Queue Length Over Time ({label})")
    plt.xlabel("Time Step")
    plt.ylabel("Expected Queue Length")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# Parameters
queue_length = 100
lambdaa = 8
mu = 10
tau = 0.05
M = 100  # number of simulations per init
max_time = 100.0

# Construct generator matrix
Q = create_mm1_generator_matrix(queue_length, lambdaa, mu)

# Simulate from empty and full queue
traj_empty = [simulate_ctmc(Q, 0, tau, max_time) for _ in range(M)]
traj_full = [simulate_ctmc(Q, queue_length, tau, max_time) for _ in range(M)]

# Estimate distributions and DTMCs
emp_dist_empty = estimate_empirical_distribution(traj_empty, queue_length + 1)
emp_dist_full = estimate_empirical_distribution(traj_full, queue_length + 1)
N = queue_length + 1

P_hat = estimate_transition_matrix_from_two_trajectories(emp_dist_empty, emp_dist_full)

pi_0_empty = np.zeros(N)
pi_0_empty[0] = 1

pi_0_full = np.zeros(N)
pi_0_full[-1] = 1

compare_avg_queue_length(P_hat, pi_0_empty, emp_dist_empty, "Init: Empty Queue")
compare_avg_queue_length(P_hat, pi_0_full, emp_dist_full, "Init: Full Queue")
