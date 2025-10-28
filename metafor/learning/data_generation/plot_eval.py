import pickle
import matplotlib.pyplot as plt
import os

import pandas as pd
import csv 

x = [0.1, 1,    2,   5,  7.5,  9,  9.7]
y = [0.81,0.85,0.88,0.95,0.9913,0.9981,0.999482]

plt.figure()

plt.plot(x,y,'ro-')
plt.title(f"Largest Eigenvalue for different arrival rates")
plt.xlabel("Arr Rate")
plt.ylabel("Largest Eval")
plt.grid(True)
plt.savefig("eval.pdf")


