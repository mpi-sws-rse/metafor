import pickle
import matplotlib.pyplot as plt
import os

import pandas as pd
import csv 
pickle_files = ["q_seq.pkl"]  # can be one or multiple files


data_list = []
for file in pickle_files:
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
            data_list.append(data)
            print(f"Loaded {file} with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    else:
        print(f"File not found: {file}")

pickle_files1 = ["l_seq.pkl"]  # can be one or multiple files


data_list1 = []
for file in pickle_files1:
    if os.path.exists(file):
        with open(file, "rb") as f:
            data = pickle.load(f)
            data_list1.append(data)
            print(f"Loaded {file} with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")
    else:
        print(f"File not found: {file}")


data = data_list[0]
data1 = data_list1[0]
# Saving data to CSV
myfile = "Metafor_train"+str(0)+"_x.csv"

import numpy as np
rows = np.array((data[0],data1[0])).T

np.savetxt(myfile, rows, delimiter=",", fmt='%d')


myfile = "Metafor_train"+str(1)+"_x.csv"
rows = np.array((data[1],data1[1])).T
np.savetxt(myfile, rows, delimiter=",", fmt='%d')




# for i, data in enumerate(data_list):
#     if isinstance(data, (list, tuple)):
#         plt.figure()
#         print(len(data))
#         plt.plot(data[0])
#         plt.title(f"Data from {pickle_files[i]}")
#         plt.xlabel("Index")
#         plt.ylabel("Value")
#         plt.grid(True)
#         plt.show()
#         #plt.savefig("qseq_"+str(i)+".pdf")
#         df = pd.DataFrame(data[0])
#         df.to_csv("Metafor_train"+str(0)+"_x.csv", index=False)
#         df = pd.DataFrame(data[1])
#         df.to_csv("Metafor_train"+str(1)+"_x.csv", index=False)

