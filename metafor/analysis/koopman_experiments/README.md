
## 🚀 Analysing the model       

To compute the settling time from the simulation and the theoretical upper bound for server1, run:      

```bash 
python analysis/koopman_experiments/exp_mixing_time_simulation.py --model_path="models/learned_model_server1.pkl" --data_path="data/server1/sim_data.pkl"   
```

To compute the settling time from the learned Koopman matrix for server2, run:      

```bash 
python analysis/koopman_experiments/exp_mixing_time_learned.py --model_path="models/learned_model_server2.pkl"
```

To compute the settling time from the learned Koopman matrix across all the servers:      

```bash 
 python analysis/koopman_experiments/exp_mixing_time_learned_all.py --model_dir="models/"   
```

