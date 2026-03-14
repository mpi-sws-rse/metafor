rm nohup.out 
python full_example.py
cp data/server1/sim_data.pkl discrete_results_multi_1.pkl
cp data/server2/sim_data.pkl discrete_results_multi_2.pkl
cp data/server3/sim_data.pkl discrete_results_multi_3.pkl
cp data/server4/sim_data.pkl discrete_results_multi_4.pkl
cp data/server5/sim_data.pkl discrete_results_multi_5.pkl
python plot_pkl.py 
