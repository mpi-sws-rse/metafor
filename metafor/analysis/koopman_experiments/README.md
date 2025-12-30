
## 🚀 Analysing the model       

To compute the settling time from the simulation and the theoretical upper bound:      

```bash 
python utils/exp_mixing_time_simulation.py --model_path="models/learned_model.pkl" --data_path="data/sim_data.pkl"   
```

To compute the settling time from the learned Koopman matrix:      

```bash 
python utils/exp_mixing_time_learned.py --model_path="models/learned_model.pkl"
```

To compute the settling time from the learned Koopman matrix in the multiserver setting:      

```bash 
python utils/exp_mixing_time_learned_multi.py --model1_path="models/learned_model_multi_1.pkl"   
       --model2_path="models/learned_model_multi_2.pkl"
```

To compare multiple simulations runs:      

```bash 
python utils/compare_simulations.py --data_dir="data/"
```

