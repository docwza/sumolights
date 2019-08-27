import random, os, sys
from itertools import cycle
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class SOTLTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, g_min, theta, omega, mu):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.g_min = g_min
        self.theta = theta
        self.omega = omega
        self.mu = mu
        self.kappa = 0 
        self.data = None
        self.phase_idx = 0
        self.time_in_phase = 0
        self.phase_red_lanes = self.get_phase_red_lanes()
        self.phase_deque = deque([self.green_phases[self.phase_idx]])

    def next_phase(self):
        #stay in green phase for 
        #minimum amount of time
        if len(self.phase_deque) == 0:
            if self.time_in_phase >= self.g_min:                                       
                n = self.approaching_vehicles()
                #if too many vehicles approaching green or no vehicles, go straight to kappa check
                if n > self.mu or n == 0:
                    if self.kappa > self.theta:
                        self.phase_idx += 1
                        self.kappa = 0
                ###need to do deque here
                '''
                if self.kappa > self.theta:
                    n = self.approaching_vehicles()
                    #increment only if not vehicles are approaching
                    #or if too many vehicles, given that kappa
                    #has been satisifed. This allows platoons
                    #to stay together.
                    #self.phase_idx += 1
                    #self.kappa = 0
                    if n > self.mu or n == 0:
                        self.phase_idx += 1
                        self.kappa = 0

                    #else:
                    #    print('holding on ' +str(self.id))
                '''                                                                   
            next_green = self.green_phases[self.phase_idx%len(self.green_phases)] 
            phases = self.get_intermediate_phases(self.phase, next_green)            
            self.phase_deque.extend(phases+[next_green])                             

        next_phase = self.phase_deque.popleft()
        if next_phase is not self.phase:
            self.time_in_phase = 0
        return next_phase 

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return 1
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data):
        self.data = data 
        
        self.time_in_phase += 1
        g = self.green_phases[self.phase_idx%len(self.green_phases)]
        #vehicle time integral, used to control
        #incrementing phase
        self.kappa += sum([ len(data[l])  for l in self.phase_red_lanes[g]])

    def get_phase_red_lanes(self):
        all_incoming_lanes = []
        for g in self.green_phases:
            all_incoming_lanes.extend(self.phase_lanes[g])
        all_incoming_lanes = set(all_incoming_lanes)

        #store all lanes that are red
        #under any given green phase
        phase_red_lanes = {}
        for g in self.green_phases:
            phase_red_lanes[g] = all_incoming_lanes - set(self.phase_lanes[g])
        return phase_red_lanes

    def approaching_vehicles(self):
        #count the number of vehicles
        #approaching (within omega distance)
        #the intersection in green lanes
        for l in self.phase_lanes[self.phase]:
            approaching_v = 0
            for v in self.data[l]:
                if self.netdata['lane'][l]['length'] - self.data[l][v][traci.constants.VAR_LANEPOSITION] < self.omega:
                    approaching_v += 1
                if approaching_v > self.mu:
                    break
        return approaching_v

        
