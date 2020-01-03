

# Simulation

## The data we get for each traffic light 

```python
self.conn.junction.subscribeContext(tsc_id, traci.constants.CMD_GET_VEHICLE_VARIABLE, 150,
                                        [traci.constants.VAR_LANEPOSITION,
                                        traci.constants.VAR_SPEED,
                                        traci.constants.VAR_LANE_ID])
```

Because of the `subscribeContext`, we could identify the vehicle info within 150m. The vehicle info includes its lanepositions, speed and land id. If you print it out, it shows as 
```python
{
'gneE11311': # vehicle id
  {
    86: 85.5756588880405,  # lane position
    64: 6.263429403624231, # vehicle speed
    81: '-gneE10_0'       # lane id
  }
}
```

To get more info, you want to check out the traci constants [LINK](https://sumo.dlr.de/pydoc/traci.constants.html).



# sumolights
SUMO adaptive traffic signal control - DQN, DDPG, Webster's, Max-pressure, Self-Organizing Traffic Lights

Technical details available at [An Open-Source Framework for Adaptive Traffic Signal Control](https://arxiv.org/abs/1909.00395)

## Setup

Install SUMO traffic microsimulator by [following instructions here](https://sumo.dlr.de/wiki/Installing) (v1.2).

Using Python 3, create a virtual environment and then install dependancies with:
```
pip install -r requirements.txt
```

## Comparing adaptive traffic signal controllers
First train reinforcement learning controllers:
```
./train_dqn.sh
./train_ddpg.sh
```

Then execute simulations to generate performance results for all controllers:
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

