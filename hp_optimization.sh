#!/usr/bin/env bash
python hp_optimization.py -sim double -tsc maxpressure 
python hp_optimization.py -sim double -tsc sotl 
python hp_optimization.py -sim double -tsc websters
python hp_optimization.py -sim double -tsc uniform 
python hp_optimization.py -sim double -tsc dqn -load
python hp_optimization.py -sim double -tsc dpg -load
