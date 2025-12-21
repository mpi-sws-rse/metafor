import unittest
from unittest.mock import patch, MagicMock
import time
import requests
from requests.exceptions import Timeout, RequestException
import random
import numpy as np
# from metafor.simulator.server import  Server
# from metafor.simulator.statistics import StatData
# from metafor.simulator.client import Client, OpenLoopClient, OpenLoopClientWithTimeout
from metafor.simulator.job import exp_job
from metafor.simulator.simulate import Simulator, make_sim_exp
import matplotlib.pyplot as plt


def test_sim(mean_t : float,max_t : float,rho : float, queue_size : int, retry_queue_size : int, timeout_t : float, 
                 max_retries : int, rho_fault : float, rho_reset : float, fault_start : float, fault_duration : float):

    print("Running Test simulation ")
    job_type = exp_job(mean_t)
    clients = make_sim_exp(mean_t, "client", "request", rho, queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
    for client in clients:
       
        client.server.file = "test1"
        client.server.start_time = time.time()
        siml = Simulator(client.server, clients, 1)
        siml.reset()
        siml.sim(max_t)
        result = client.server.context.result

    return result

def get_result(result):
    timestamp = []
    latency = []
    qlen = []
    olen = []
    drop = []
    rleft = []
    stime = []
    for i in range(len(result)):
        timestamp.append(result[i]["timestamp"])
        latency.append(result[i]["latency"])
        qlen.append(result[i]["queue_length"])
        olen.append(result[i]["retries"])
        drop.append(result[i]["dropped"])
        rleft.append(result[i]["retries_left"])
        stime.append(result[i]["service_time"])
    return timestamp, latency, qlen, olen, drop, rleft, stime

np.random.seed(0)
random.seed(0)
main_queue_size = 100 # maximum size of the arrivals queue
retry_queue_size = 20 # only used when learning in the space of prob distributions is desired.
timeout_t = 9 # timeout after which the client retries, if the job is not done
max_retries = 3 # how many times should a client retry to send a job if it doesn't receive a response before the timeout
step_time = 1 # sampling time
sim_time = 1 # maximum simulation time for an individual simulation


# mean_t = 1/10 # mean of the exponential distribution (in ms) related to processing time
# rho = 0.97 # server's utilization rate

############### Test 1 : Arrival rate less than processing rate #################
mean_t = 1/10 # mean of the exponential distribution (in ms) related to processing time
rho = 0.9 # Arrival rate
rho_fault = rho # utilization rate during a fault
rho_reset = rho * 5 / 5 # utilization rate after removing the fault
fault_start = [sim_time * 1, sim_time]  # start time for fault (last entry is not an actual fault time)
fault_duration = sim_time * 0  # fault duration

result = test_sim(mean_t,sim_time,rho,main_queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
timestamp, latency, qlen, olen, drop, rleft, stime = get_result(result)
print("Stats : ",np.max(qlen),"  ",np.max(olen),"  ",np.max(drop),"  ",np.max(latency))
assert(np.max(qlen)==1)
assert (np.max(drop)==0)
assert (np.max(olen)==0)
assert (np.max(rleft)==3 and np.min(rleft)==3)


############### Test 2 : Arrival rate more than processing rate #################
mean_t = 1/10 # mean of the exponential distribution (in ms) related to processing time
rho = 1.2 # Arrival rate
rho_fault = rho # utilization rate during a fault
rho_reset = rho * 5 / 5 # utilization rate after removing the fault
fault_start = [sim_time * 1, sim_time]  # start time for fault (last entry is not an actual fault time)
fault_duration = sim_time * 0  # fault duration

result = test_sim(mean_t,sim_time,rho,main_queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
timestamp, latency, qlen, olen, drop, rleft, stime = get_result(result)
print("Stats : ",np.max(qlen),"  ",np.max(olen),"  ",np.max(drop),"  ",np.max(latency))
assert(np.max(qlen)==6)
assert (np.max(drop)==0)
assert (np.max(olen)==0)
assert (np.max(rleft)==3 and np.min(rleft)==3)


############### Test 3 : Test Fault + No retries #################
mean_t = 1/10 # mean of the exponential distribution (in ms) related to processing time
rho = 0.9 # Arrival rate
rho_fault = 100 # utilization rate during a fault
rho_reset = rho * 5 / 5 # utilization rate after removing the fault
fault_start = [0, sim_time]  # start time for fault (last entry is not an actual fault time)
fault_duration = sim_time * 1  # fault duration

result = test_sim(mean_t,sim_time,rho,main_queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
timestamp, latency, qlen, olen, drop, rleft, stime = get_result(result)
print("Stats : ",np.max(qlen),"  ",np.max(olen),"  ",np.max(drop),"  ",np.max(latency))
assert(np.max(qlen)==100)
assert (np.max(drop)==846)
assert (np.max(olen)==0)
assert (np.min(rleft)==max_retries)



############### Test 3 : Test Fault + retries #################
mean_t = 1/10 # mean of the exponential distribution (in ms) related to processing time
rho = 1 # Arrival rate
rho_fault = 1000 # utilization rate during a fault
rho_reset = rho * 5 / 5 # utilization rate after removing the fault
fault_start = [0.49*sim_time, sim_time]  # start time for fault (last entry is not an actual fault time)
fault_duration = sim_time * 0.01  # fault duration
timeout_t = 1
max_retries = 3 
sim_time = 5

result = test_sim(mean_t,sim_time,rho,main_queue_size, retry_queue_size, timeout_t, max_retries, rho_fault, rho_reset, fault_start, fault_duration)
timestamp, latency, qlen, olen, drop, rleft, stime = get_result(result)
print("Stats : ",np.max(qlen),"  ",np.max(olen),"  ",np.max(drop),"  ",np.max(latency))
assert(np.max(qlen)==6)
assert (np.max(drop)==0)
assert (np.max(olen)==3)
assert (np.min(rleft)<max_retries)