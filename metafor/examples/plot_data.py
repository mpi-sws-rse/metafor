import pickle
import matplotlib.pyplot as plt
import os

import pandas as pd
import csv 
pickle_files = ["data/server1/q_seq.pkl","data/server2/q_seq.pkl","data/server3/q_seq.pkl","data/server4/q_seq.pkl","data/server5/q_seq.pkl"]  # can be one or multiple files
#pickle_files = ["data/server1/q_seq.pkl"]  # can be one or multiple files


data_list = []
for file in pickle_files:
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
            data_list.append(data)
            print(f"Loaded {file} with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    else:
        print(f"File not found: {file}")

plt.figure()
for i, data in enumerate(data_list):
    if isinstance(data, (list, tuple)):
        
        # print(len(data))
        # print(data[1])
        plt.plot(data[0],label=i+1)
plt.title(f"Data ")
plt.xlabel("Index")
plt.ylabel("q_seq")
plt.grid(True)
plt.legend()
plt.show()
#plt.savefig("q_seq_all.pdf")
