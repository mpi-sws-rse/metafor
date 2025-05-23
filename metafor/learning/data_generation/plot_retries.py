import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


runs=11
suffix = "exp_results.csv"
# reading csv file 
for i in range(1,runs):
    results_file_name = str(i)+"_"+suffix
    #df = pd.read_csv("people.csv")
    df = pd.read_csv(results_file_name, usecols=["timestamp","latency","queue_length","retries","dropped","runtime"])
    #print(df["qlen"][0:100])
    plt.plot(df["retries"], label="run_"+str(i))
    

plt.xlabel("Timestep")
plt.ylabel("ret")
plt.legend()
plt.savefig("viz_ret.pdf")