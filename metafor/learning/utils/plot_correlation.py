import numpy as np 
from pandas import DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import pickle


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

print(len(q_seq))
a = np.array([q_seq[0],o_seq[0]])


def plot_correlation_heatmap(q_seq, o_seq, l_seq, d_seq, r_seq, s_seq):
    """Plots a correlation heatmap for two arrays."""
    # Create a DataFrame (easier for correlation calculation)
    df = pd.DataFrame({'q_seq': q_seq, 'o_seq': o_seq, 'l_seq': l_seq, 'd_seq': d_seq, 'r_seq': r_seq, 's_seq': s_seq})
    #df = pd.DataFrame({'q_seq': q_seq, 'o_seq': o_seq, 'r_seq': r_seq, 's_seq': s_seq})

    # Calculate the correlation matrix
    corr_matrix = df.corr()

    # Plot the heatmap
    plt.figure(figsize=(5, 4))  # Adjust figure size as needed
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
    plt.title('Correlation Heatmap')
    plt.show()

# Example Usage
array1 = np.random.rand(100)  # Example array 1
array2 = np.random.rand(100)  # Example array 2
plot_correlation_heatmap(q_seq[0], o_seq[0], l_seq[0], d_seq[0], r_seq[0], s_seq[0])
plt.savefig("corr_full1.pdf")
plt.close()
plot_correlation_heatmap(q_seq[1], o_seq[1], l_seq[1], d_seq[1], r_seq[1], s_seq[1])
plt.savefig("corr_full2.pdf")
