
"""
This files is for data visualization only. We can modify the feature that we want to visualize.
By default, we use "queue_length" for visualization. 
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


runs=11 #select the number of runs. Ideally this should be equal to the "-runs" value used for data generation.

suffix = "exp_results.csv"
feature = "retries" # options={timestamp,latency,queue_length,retries,dropped,runtime,retries_left,service_time,last_status}
# reading csv file 
for i in range(1,runs):
    results_file_name = str(i)+"_"+suffix
    #df = pd.read_csv("people.csv")
    df = pd.read_csv(results_file_name, usecols=["timestamp","latency","queue_length","retries","dropped","runtime","retries_left","service_time","last_status","context"])
    #print(df["qlen"][0:100])
    plt.plot(df[feature], label="run_"+str(i))
    

plt.xlabel("Timestep")
plt.ylabel(feature)
plt.legend()
plt.savefig("viz_data.pdf")
