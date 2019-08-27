# sumolights
SUMO adaptive traffic signal control - DQN, DDPG, Webster's, Max-pressure, Self-Organizing Traffic Lights

## Comparing adaptive traffic signal controllers
First train reinforcement learning controllers:
```
./train_dqn.sh
./train_ddpg.sh
```

Then execute simulations to generate performande results for all controllers:
```
./gen_results.sh
```

Visualize results with:
```
python graph_results.py
```

## Optimizing hyperparameters
Search for optimal hyperparameters for each controller:
```
./hp_optimization
```
Warning, search for reinforcement learning can require significant compute time!

Visualize hyperparameters with:
```
python graph_results.py -type hp
```
