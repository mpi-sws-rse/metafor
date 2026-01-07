
## 📦 Data Generation    

To generate data:     

```bash 
python data_generation/data_generator.py --runs=100 --sim_time=10000 --fault_duration=0.01  --genpkl=True --num_server=2
```

This generates data files inside the "data/" folder. Additionally, it also    
generates a file "discrete_results.pdf" which shows the average queue length    
and latency for the simulations. Note that this command generates individual     
files for each server inside the data folder. Files for server $k$ can be    
found in "data/serverk/" folder.   



## Throttling strategy   
To generate data from simulator while using throttling strategy:     

```bash
python data_generation/data_generator.py --throttle=True --ts=0.9 --ap=0.5

```
This specifies that the throttling strategy comes info effect when current queue length is 90% of the maximum size (ts=0.9) 
with an admission probability of 0.5 (ap=0.5), i.e., we drop 50%  of the jobs entering the system.     


## LIFO based servers   
To generate data from simulator wherein servers use stack (LIFO) instead of queues (FIFO):     

```bash
python data_generation/data_generator.py --queue_type="lifo" 

```



