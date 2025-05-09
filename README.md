# 🔍 Metafor: Analyzing Metastability in Server Systems

**Metafor** is a Python-based tool designed to analyze metastability in server systems by modeling them as continuous-time Markov chains (CTMCs). It provides analytical and visual tools to explore long transient behaviors, steady states, and mixing properties.

## ⚙️ Prerequisites

Ensure you have **Python 3.12 or later** installed on your machine.

## 📦 Installing Metafor

To install the required Python packages and set up your environment, run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install numpy matplotlib scipy pandas
export PYTHONPATH=/path/to/metafor:/path/to/metafor/metafor
```

Replace `/path/to/metafor` with the actual path to your local clone of the Metafor repository.

## 🚀 Example: Single-Threaded Single Server System

Below is an example where a single-threaded server handles requests at an average rate of \( \mu = 10 \) requests per second (RPS). A client sends requests at \( \lambda = 9.5 \) RPS, each with a timeout of 3 seconds and a maximum of 4 retries.

```python
import math
import numpy as np
from numpy import linspace
import pandas
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors

from metafor.dsl.dsl import Server, Work, Source, Program
from metafor.model.single_server.ctmc import SingleServerCTMC
from metafor.analysis.experiment import Parameter, ParameterList
from metafor.analysis.visualize import Visualizer

# Define server processing rate
api = {"insert": Work(10, [])}

# Configure server parameters: queue size, orbit size, threads
server = Server("52", api, qsize=100, orbit_size=20, thread_pool=1)

# Define client request behavior
src = Source("client", "insert", 9.5, timeout=3, retries=4)

# Build the request-response system
p = Program("Service52")
p.add_server(server)
p.add_source(src)
p.connect("client", "52")
```

## 📊 Running Basic Analysis

To compute key system metrics such as the average steady-state queue size, sorted eigenvalues of the CTMC, mixing time, and expected hitting time from full to empty queue:

```python
basic_stats(p)
```

## 🧭 Visualizing the CTMC

To visualize the underlying CTMC and identify potential metastable behavior:

```python
v = Visualizer(program())
v.visualize(show_equilibrium=True)
```

---

Feel free to modify this example or plug in more complex workloads and topologies using Metafor's flexible DSL and analysis tools.

## ✉️ Contact

For questions, feedback, or collaboration inquiries, feel free to reach out to **msalamati@mpi-sws.org**.








