import pickle
import matplotlib.pyplot as plt
import os
from metafor.utils.plot import plot_results
import numpy as np
import pandas as pd
import csv 
pickle_files = ["discrete_results_10.pkl","discrete_results_20.pkl","discrete_results_50.pkl","discrete_results_75.pkl","discrete_results_90.pkl","discrete_results_97.pkl"]  # can be one or multiple files


data_list = []
for file in pickle_files:
   
    with open(file, "rb") as f:
        data = pickle.load(f)
        data_list.append(data)
        print(f"Loaded {file} with keys: {list(data.keys()) if isinstance(data, dict) else type(data)}")

data = data_list[0]
print(len(data))
step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, rho = data


figure_name = 'plot_latency.pdf'
time = [i * step_time for i in list(range(0, len(qlen_ave)))]
# Create 4x1 sub plots
#plt.rcParams["figure.figsize"] = [6, 10]
plt.rcParams["figure.autolayout"] = True

ax = plt.GridSpec(1, 1)
ax.update(wspace=0.5, hspace=0.5)
cmap = plt.cm.tab10
color="tab:blue"
ax1 = plt.subplot(ax[0, 0])  # row 0, col 0
data_list = []
k=0
for file in pickle_files:
    color = cmap(k) 
    with open(file, "rb") as f:
        data = pickle.load(f)
        step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, rho = data
        ax1.plot(time, latency_ave, color=color,label=np.round(0.1*int(file[-6:-4]),2))
        k=k+1
ax1.set_xlabel("Time", fontsize=16)
ax1.set_ylabel("Average Latency", fontsize=16, color=color)
ax1.grid("on")
ax1.set_xlim(0, max(time))
ax1.tick_params(axis='y', labelcolor=color)
ax1.legend()

ax11 = ax1.twinx()
color="tab:red"
ax11.set_ylabel("Average Arrival Rate",fontsize=16, color=color)
avg_arr_rate = np.ones(len(time))
avg_arr_rate[np.arange(int(0.45*len(time)),int(0.46*len(time)))] = 10
ax11.plot(time, avg_arr_rate, color=color)
ax11.tick_params(axis='y',labelcolor=color)

plt.savefig(figure_name)
plt.close()

# ax2 = plt.subplot(ax[1, 0])  # row 3, col 0
# ax2.plot(time, runtime_seq, color="tab:green")
# ax2.set_xlabel("Time bound (ms)", fontsize=8)
# ax2.set_ylabel("Runtime (sec)", fontsize=8)
# ax2.grid("on")
# ax2.set_xlim(0, max(time))
figure_name = 'plot_qlen.pdf'
time = [i * step_time for i in list(range(0, len(qlen_ave)))]
# Create 4x1 sub plots
#plt.rcParams["figure.figsize"] = [6, 10]
plt.rcParams["figure.autolayout"] = True


ax = plt.GridSpec(1, 1)
ax.update(wspace=0.5, hspace=0.5)
cmap = plt.cm.tab10
color="tab:blue"

ax3 = plt.subplot(ax[0, 0])  # row 4, col 0
color="tab:blue"
data_list = []
k=0
cmap = plt.cm.tab10
for file in pickle_files:
    color = cmap(k) 
    with open(file, "rb") as f:
        data = pickle.load(f)
        step_time, latency_ave, latency_var, latency_std, runtime, qlen_ave,  qlen_var, qlen_std, rho = data
        
        ax3.plot(time, qlen_ave, color=color,label=np.round(0.1*int(file[-6:-4]),2))
        k=k+1
ax3.set_xlabel("Time", fontsize=16)
ax3.set_ylabel("Average Queue Length", fontsize=16,color=color)
ax3.grid("on")
ax3.set_xlim(0, max(time))
ax3.tick_params(axis='y', labelcolor=color)
ax3.legend()

ax31 = ax3.twinx()
color="tab:red"
ax31.set_ylabel("Average Arrival Rate", fontsize=16,color=color)
avg_arr_rate = np.ones(len(time))
avg_arr_rate[np.arange(int(0.45*len(time)),int(0.46*len(time)))] = 10
ax31.plot(time, avg_arr_rate, color=color)
ax31.tick_params(axis='y',labelcolor=color)
#plt.show()
plt.savefig(figure_name)
plt.close()

exit()


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

