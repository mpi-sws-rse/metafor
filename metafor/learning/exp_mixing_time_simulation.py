import numpy as np
import matplotlib.pyplot as plt
import pickle
import pickle
import matplotlib.pyplot as plt
import os
from metafor.utils.plot import plot_results
import numpy as np
import pandas as pd
import csv 


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
    final_value = y[-1]

    lower = final_value - epsilon
    upper = final_value + epsilon

    for i in range(len(y)):
        # If from this point onward the signal stays within the band
        if np.all((y[i:] >= lower) & (y[i:] <= upper)):
            return t[i]
    return None  # never settles



# Example usage
if __name__ == "__main__":
    pickle_files = ["discrete_results_LONG.pkl"]  # can be one or multiple files


    data_list = []
    for file in pickle_files:
    
        with open(file, "rb") as f:
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


    epsilons = np.arange(20)
    data = []
    data1 = []
    for ep in epsilons:
        ts = settling_time(t, y, epsilon=ep)
        print(f"Settling time = {ts:.3f} s")

        data.append(ts)

    data = np.array(data)

    plt.plot(epsilons, data, color="tab:green",label='Mean')

    plt.title("Settling time in simulations")
    plt.xlabel("Error band",fontsize=16)
    plt.ylabel("Timesteps",fontsize=16)
    plt.legend()
    plt.savefig("Mixing_times_simulation.pdf")
    plt.close()




    ts = settling_time(t, y, epsilon=10)
    print(f"Settling time = {ts:.3f} s")

    plt.plot(t, y, label="Response")
    # plt.axhline(y[-1]*(1+0.02), color='r', linestyle='--', label="±2% band")
    # plt.axhline(y[-1]*(1-0.02), color='r', linestyle='--')
    plt.axvline(ts, color='g', linestyle='--', label=f"Settling time = {ts:.2f}s")
    plt.legend()
    plt.show()



