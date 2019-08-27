from itertools import cycle
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController

class WebstersTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, g_min, c_min, c_max, sat_flow=0.38, update_freq=None):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.cycle = self.get_phase_cycle()
        self.g_min = g_min
        self.c_min = c_min
        self.c_max = c_max 
        self.update_freq = update_freq
        self.t = 0
        self.sat_flow = sat_flow
        self.green_phase_duration = { g:g_min for g in self.green_phases}
        #for keeping track of vehicle counts for websters calc
        self.phase_lane_counts = self.get_empty_phase_lane_counts()
        self.prev_data = None

    def get_phase_cycle(self):
        phase_cycle = []
        greens = self.green_phases
        next_greens = self.green_phases[1:] + [self.green_phases[0]]
        for g, next_g in zip(greens, next_greens):
            phases = self.get_intermediate_phases(g, next_g)
            phase_cycle.append(g)
            phase_cycle.extend(phases)
        return cycle(phase_cycle)

    def next_phase(self):
        return next(self.cycle)

    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_phase_duration[self.phase]
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data):
        #update vehicle counts
        if self.phase in self.green_phases:
            self.update_phase_lane_counts(data)
        ###need to keep track of lane counts using data
        if self.t % self.update_freq == 0:
            self.websters()
            self.phase_lane_counts = self.get_empty_phase_lane_counts()
        self.prev_data = data
        self.t += 1

    def update_phase_lane_counts(self, data):
        """keep track of vehicles that have traversed
        each lane at the intersection, this is hacky
        but it works, SUMO should really implement
        this natively FFS
        """
        if self.prev_data:
            incoming_vehicles = set([]) 
            for l in self.phase_lanes[self.phase]:
                for v in data[l]:
                    incoming_vehicles.add(v)

            for l in self.phase_lanes[self.phase]:
                for v in self.prev_data[l]:
                    if v not in incoming_vehicles:
                        self.phase_lane_counts[self.phase][l] += 1

    def get_empty_phase_lane_counts(self):
        phase_lane_counts = {}
        for p in self.phase_lanes:
            phase_lane_counts[p] = {}
            for l in self.phase_lanes[p]:
                phase_lane_counts[p][l] = 0
        return phase_lane_counts

    def websters(self):
        """update green phase times using lane
        vehicle counts and Websters' traffic signal
        control method
        """
        ##compute flow ratios for all lanes in all green phases
        ##find critical 
        y_crit = []
        for g in self.green_phases:
            sat_flows = [(self.phase_lane_counts[g][l]/self.update_freq)/(self.sat_flow) for l in self.phase_lanes[g]]
            y_crit.append(max(sat_flows))

        #compute intersection critical lane flow rattios
        Y = sum(y_crit)
        if Y > 0.85:
            Y = 0.85
        elif Y == 0.0:
            Y = 0.01

        #limit in case too saturated
        #compute lost time
        L = len(self.green_phases)*(self.red_t + self.yellow_t)

        #compute cycle time
        C = int(((1.5*L) + 5)/(1.0-Y))
        #constrain if necessary
        if C > self.c_max:
            C = self.c_max
        elif C < self.c_min:
            C = self.c_min

        G = C - L
        #compute green times for each movement
        #based on total green times
        for g, y in zip(self.green_phases, y_crit):
            g_t = int((y/Y)*G)
            #constrain green phase time if necessary
            if g_t < self.g_min:
                g_t = self.g_min
            self.green_phase_duration[g] = g_t
