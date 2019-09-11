#!/usr/bin/env bash
python run.py -sim double -tsc ddpg -lr 0.0001 -lrc 0.0005 -lre 0.0001 -save -nogui -nreplay 15000 -updates 15000 -nsteps 1 -save_t 30 -batch 32 -tau 0.01 -target_freq 64
