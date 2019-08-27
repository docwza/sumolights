import random
import numpy as np
from itertools import cycle
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController

class NextPhaseRLTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t, rlagent):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.green_t = green_t
        self.t = 0
        #for keeping track of vehicle counts for websters calc
        #print(tsc_id)
        self.phase_deque = deque()
        self.data = None
        self.delay_green = False
        self.phase_to_one_hot = self.input_to_one_hot(self.green_phases+[self.all_red])
        self.int_to_phase = self.int_to_input(self.green_phases)
        self.rlagent = rlagent
        #experience dict
        self.acting = False
        #store how many green movements each phase has
        #for breaking ties in max pressure
        self.s = None
        self.a = None

    def next_phase(self):
        ###need to do deque here
        if len(self.phase_deque) == 0:
            #max_pressure_phase = self.max_pressure()
            next_phase = self.get_next_phase()
            phases = self.get_intermediate_phases(self.phase, next_phase)
            self.phase_deque.extend(phases+[next_phase])
        return self.phase_deque.popleft()

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def get_next_phase(self):
        #check which action lane groups 
        #have vehicles
        #check if any vehicles at intersection, if yes
        if self.empty_intersection():
            #go to all red phase
            if self.acting:
                #state = np.concatenate( [self.get_state(), self.phase_to_one_hot[self.phase]] )[np.newaxis,...]
                state = np.concatenate( [self.get_state(), self.phase_to_one_hot[self.phase]] )
                terminal = True
                self.store_experience(state, terminal)
            self.acting = False
            return self.all_red
        else:
            #check if last phase was red, might need to
            #delay action selection a little
            #to allow more vehicles to approach
            #intersection
            if self.phase == self.all_red and not self.delay_green:
                self.delay_green = True
                self.acting = False
                return self.all_red
            self.delay_green = False
            #state is a concatenation of the normalized density
            #and one hot hot vector encoding the previous phase
            #state = np.concatenate( [self.get_state(), self.phase_to_one_hot[self.phase]] )[np.newaxis,...]
            state = np.concatenate( [self.get_state(), self.phase_to_one_hot[self.phase]] )
            if self.acting:
                terminal = False
                self.store_experience(state, terminal)
            action_idx = self.rlagent.get_action(state)
            next_phase = self.int_to_phase[action_idx]
            self.s = state
            self.a = action_idx
            self.acting = True
            return next_phase
            #return random.choice(self.green_phases)

    def store_experience(self, next_state, terminal):
        self.rlagent.store_experience(self.s, self.a, next_state, self.get_reward(), terminal)
        
    def update(self, data):
        self.data = data 
