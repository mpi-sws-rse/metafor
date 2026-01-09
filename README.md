# 🔍 Metafor: Analyzing Metastability in Server Systems

**Metafor** is a Python-based tool designed to analyze metastability in server systems by modeling them as continuous-time Markov chains (CTMCs). It provides analytical and visual tools to explore long transient behaviors, steady states, and mixing properties.

## ⚙️ Prerequisites

Ensure you have **Python 3.10 or later** installed on your machine.

## 📦 Installing Metafor

To install the required Python packages and set up your environment, run the following commands:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install .
export PYTHONPATH=/path/to/metafor:/path/to/metafor/metafor
```

Replace `/path/to/metafor` with the actual path to your local clone of the Metafor repository.

# 🚀 Example 1 : Modelling Using CTMCs

Below is an example where a single-threaded server handles requests at an average rate of \( \mu = 10 \) requests per second (RPS). A client sends requests at \( \lambda = 9.5 \) RPS, each with a timeout of 3 seconds and a maximum of 4 retries.

```python
from metafor.dsl.dsl import Server, Work, Source, Program
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

### 📊 Running Basic Analysis

To compute key system metrics such as the average steady-state queue size, sorted eigenvalues of the CTMC, mixing time, and expected hitting time from full to empty queue:

```python
basic_stats(p)
```

### 🧭 Visualizing the CTMC

To visualize the underlying CTMC and identify potential metastable behavior:

```python
v = Visualizer(program())
v.visualize(show_equilibrium=True)
```



# 🚀 Example 2 : Modelling Using Koopman Autoencoders
The below example shows how to generate data from simulator, use the data to    
train a koopman autoencoder model and then perform analysis using the model.   


```python
from metafor.data_generation.data_generator import data_generation
from metafor.koopman_AE_model.train import training
from metafor.analysis.koopman_experiments.exp_mixing_time_simulation import mixing_time_simulation
from metafor.analysis.koopman_experiments.exp_mixing_time_learned import mixing_time_learned
from metafor.analysis.koopman_experiments.exp_mixing_time_learned_all import mixing_time_learned_all



"""
Part 1 : Data generation

"""

# Configuration
total_time = 1000000 # maximum simulation time (in s) for all the simulations
queue_size = 100 # maximum size of the arrivals queue
mean_t = 0.1 # mean of the exponential distribution (in ms) related to processing time
rho = 9.7/10 # server's utilization rate
timeout_t = 9 # timeout after which the client retries, if the job is not done
max_retries = 3 # how many times should a client retry to send a job if it doesn't receive a response before the timeout
runs = 10 # how many times should the simulation be run
step_time = 0.5 # sampling time
sim_time = 10000 # maximum simulation time for an individual simulation
#rho_fault = np.random.uniform(rho,rho*10) # utilization rate during a fault
rho_fault = rho*10 # utilization rate during a fault
fault_start = [sim_time * .45, sim_time]  # start time for fault (last entry is not an actual fault time)
rho_reset = rho * 5 / 5 # utilization rate after removing the fault
fault_duration = sim_time * .01  # fault duration
dist = "exp"
throttle=False
ts = 0.9
ap = 0.5
queue_type="fifo"
verbose = True
num_servers = 1

data_generation(
    sim_time, runs, mean_t, rho, queue_size, timeout_t, 
    max_retries, total_time, step_time, rho_fault, 
    rho_reset, fault_start, fault_duration, throttle, 
    ts, ap, queue_type, dist, num_servers, verbose       
)


"""
Part 2 : Training
"""

data_dir="data/" #directory containing the pkl files
epochs=1000 # number of epochs 

training(data_dir, epochs)


"""
Part 3. Analysis
"""

# 3.1 : Empirical Mixing time using Simulation 

model_path="models/learned_model_server1.pkl"
data_path="data/server1/sim_data.pkl"  
mixing_time_simulation(model_path, data_path)


# 3.2 : Estimating mixing time using learned model for one server

model_path="models/learned_model_server2.pkl"
mixing_time_learned(model_path)

#3.3 Estimating mixing time for all servers

model_dir="models/"
mixing_time_learned_all(model_dir)

```

---

Feel free to modify this example or plug in more complex workloads and topologies using Metafor toolbox.

## ✉️ Contact

For questions, feedback, or collaboration inquiries, feel free to reach out to **msalamati@mpi-sws.org**.








