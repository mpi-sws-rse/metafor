# Learning from Simulation Data   



## 📦 Data Generation    

To generate data:    

```bash
python data_generation/data_generator.py --runs=100 --sim_time=10000 --fault_duration=0.01  --genpkl=True

```
This generates data files inside the "data/" folder. Additionally, it also    
generates a file "discrete_results.pdf" which shows the average queue length    
and latency for the simulations. 

To generate data in multiserver setting:    

```bash 
python data_generation/data_generator_multi.py --runs=100 --sim_time=10000 --fault_duration=0.01  --genpkl=True
```
Note that this command generates individual files for each server inside the data folder.
Files for server $k$ can be found in "data/serverk/" folder.


## 🚀 Learning the model       

To learn the Koopman based autoencoder model, run      

```bash 
 python train.py --data_dir="data/" --epochs=100    
```
This generates a file "models/learned_model.pkl" which stores the all the variables associated
with the learned model.

In the multi-server setting with two servers, we learn two models, each using the   
data for the corresponding server. We do this by passing the directory for the    
corresponding server:   

```bash 
 python train.py --data_dir="data/server1"  --epochs=100    
```

For analysis, we follow the convention of renaming files to distinguish files for
different servers. Hence, in the above run, we rename the file "models/learned_model.pkl"   
to ""models/learned_model_multi_1.pkl" to mark this file as the learned model for server 1   
in multiserver setting.    

   

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


