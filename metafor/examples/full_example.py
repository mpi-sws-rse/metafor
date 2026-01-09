
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
