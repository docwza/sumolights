import random
from itertools import cycle
from collections import deque

from src.trafficsignalcontroller import TrafficSignalController

class MaxPressureTSC(TrafficSignalController):
    def __init__(self, conn, tsc_id, mode, netdata, red_t, yellow_t, green_t):
        super().__init__(conn, tsc_id, mode, netdata, red_t, yellow_t)
        self.green_t = green_t
        self.t = 0
        #for keeping track of vehicle counts for websters calc
        #print(tsc_id)
        self.phase_deque = deque()
        self.max_pressure_lanes = self.max_pressure_lanes()
        self.data = None
        #store how many green movements each phase has
        #for breaking ties in max pressure
        self.phase_g_count = {}
        for p in self.green_phases:
            self.phase_g_count[p] = sum([1 for m in p if m == 'g' or m == 'G'])

    def next_phase(self):
        ###need to do deque here
        if len(self.phase_deque) == 0:
            max_pressure_phase = self.max_pressure()
            phases = self.get_intermediate_phases(self.phase, max_pressure_phase)
            self.phase_deque.extend(phases+[max_pressure_phase])
        return self.phase_deque.popleft()

    def max_pressure_lanes(self):
        """for each green phase, get all incoming
        and outgoing lanes for that phase, store
        in dict for max pressure calculation
        """
        max_pressure_lanes = {}
        for g in self.green_phases:
            inc_lanes = set()
            out_lanes = set()
            for l in self.phase_lanes[g]:
                inc_lanes.add(l)
                for ol in self.netdata['lane'][l]['outgoing']:
                    out_lanes.add(ol)

            max_pressure_lanes[g] = {'inc':inc_lanes, 'out':out_lanes}
        return max_pressure_lanes

    def max_pressure(self):
        phase_pressure = {}
        no_vehicle_phases = []
        #compute pressure for all green movements
        for g in self.green_phases:
            inc_lanes = self.max_pressure_lanes[g]['inc']
            out_lanes = self.max_pressure_lanes[g]['out']
            #pressure is defined as the number of vehicles in a lane
            inc_pressure = sum([ len(self.data[l]) if l in self.data else 0 for l in inc_lanes])
            out_pressure = sum([ len(self.data[l]) if l in self.data else 0 for l in out_lanes])
            phase_pressure[g] = inc_pressure - out_pressure
            if inc_pressure == 0 and out_pressure == 0:
                no_vehicle_phases.append(g)

        ###if no vehicles randomly select a phase
        if len(no_vehicle_phases) == len(self.green_phases):
            return random.choice(self.green_phases)
        else:
            #choose phase with max pressure
            #if two phases have equivalent pressure
            #select one with more green movements
            #return max(phase_pressure, key=lambda p:phase_pressure[p])
            phase_pressure = [ (p, phase_pressure[p]) for p in phase_pressure]
            phase_pressure = sorted(phase_pressure, key=lambda p:p[1], reverse=True)
            phase_pressure = [ p for p in phase_pressure if p[1] == phase_pressure[0][1] ]
            return random.choice(phase_pressure)[0]

            '''
            if len(phase_pressure) == 1:
                return phase_pressure[0][0]
            else:
                #if two phases have same pressure and same number of green movements
                green_count = [ (p[0], self.phase_g_count[p[0]]) for p in phase_pressure ]
                green_count = sorted(green_count, key=lambda p:p[1], reverse=True)
                green_count = [ p for p in green_count if p[1] == green_count[0][1] ]
                if len(green_count) == 1:
                    return green_count[0][0]
                else:
                    return random.choice(green_count)[0]
            '''


    def next_phase_duration(self):
        if self.phase in self.green_phases:
            return self.green_t
        elif 'y' in self.phase:
            return self.yellow_t
        else:
            return self.red_t

    def update(self, data):
        self.data = data 
