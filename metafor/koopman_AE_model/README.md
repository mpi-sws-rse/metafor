## 🚀 Learning the model       

To learn the Koopman based autoencoder model, run      

```bash 
 python koopman_AE_model/train.py --data_dir="data/" --epochs=1000  
```

This generates a file "learned_model_serverk.pkl" insider "models/serverk/" for 
each server k=1..n . The pkl file stores the all the variables associated
with the learned model.

For instance, for k=2, the training is perfomed two times resulting into two models,   
each using the data for the corresponding server. The two files can be found at    
"models/learned_model_server1.pkl" and ""models/learned_model_server2.pkl".        

   
