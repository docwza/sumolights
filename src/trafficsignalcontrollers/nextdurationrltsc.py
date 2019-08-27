import numpy as np
from itertools import cycle
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController

class NextDurationRLTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, gmin, gmax, rlagent):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.cycle = cycle(self.green_phases)
        self.phase_deque = deque()
        self.data = None
        self.rlagent = rlagent
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases+[self.all_red])
        #help convert tanh rl action to constrained
        #next phase duration
        self.mid = ((gmax - gmin)/2.0) + gmin
        self.interval = gmax - self.mid
        #experience dict
        self.acting = False
        self.s = None
        self.a = None

    def next_phase(self):
        if len(self.phase_deque) == 0:
            next_phase = self.get_next_phase()
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases+[next_phase])
        return self.phase_deque.popleft()

    def get_next_phase(self):
        #find the next green phase
        #with vehicles in approaching lanes
        i = 0
        while i <= len(self.green_phases):
            phase = next(self.cycle)
            if not self.phase_lanes_empty(phase):
                if self.acting:
                    state = np.concatenate( [self.get_state(), self.phase_to_one_hot[phase]] )
                    terminal = False
                    self.store_experience(state, terminal)
                if not self.acting:
                    state = np.concatenate( [self.get_state(), self.phase_to_one_hot[phase]] ) 
                self.s = state                                                                         
                action = self.rlagent.get_action(state)                                                       
                self.a = action                                                                        
                self.acting = True
                return phase
            i += 1
        ##if no vehicles approaching intersection
        #default to all red
        phase = self.all_red
        if self.acting:
            #print('-------TERMINAL---------')
            state = np.concatenate( [self.get_state(), self.phase_to_one_hot[phase]] )
            terminal = True
            self.store_experience(state, terminal)
            self.acting = False
        return phase 

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            t = self.convert_action(self.a)
            #print(' phase time '+str(t))
            return t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def store_experience(self, next_state, terminal):
        self.rlagent.store_experience(self.s, self.a, next_state, self.get_reward(), terminal)
        #put experience in buffer
        #update rl stats

    def convert_action(self, action):
        #convert rl action to green
        #phase time, respecting constraints
        return int((action*self.interval)+self.mid)

    def update(self, data):
        self.data = data

    def phase_lanes_empty(self, phase):
        for l in self.phase_lanes[phase]:
            if len(self.data[l]) > 0:
                return False
        return True
