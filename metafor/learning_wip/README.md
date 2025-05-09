# Learning from Simulation Data   



## 📦 Data Generation    

To generate data:    

```bash
cd data_generation/   
python data_generator.py --sim_time=1000 --runs=100 --qsize=100   
```

## 🚀 Example: Learning the Dynamics    

To learn a linear model of the dynamics:    

```bash 
python learning_linear_model.py    
```

To learn via Autoencoders:      

```bash 
python learning_autoencoder_model.py    
```

The plots are saved in the result folder.     

