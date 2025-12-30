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

   
