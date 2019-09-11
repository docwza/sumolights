#!/usr/bin/env bash
python run.py -sim double -tsc dqn -lr 0.00005 -lre 0.0000001 -nreplay 15000 -nsteps 2 -target_freq 128 -updates 15000 -batch 32 -save -nogui -n_hidden 3 -mode train -gmin 5
