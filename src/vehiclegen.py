import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

import numpy as np


class VehicleGen:
    def __init__(self, netdata, sim_len, demand, scale, mode, conn):
        np.random.seed()
        self.conn = conn
        self.v_data = None
        self.vehicles_created = 0
        self.netdata = netdata
        ###for generating vehicles
        self.origins = self.netdata['origin']
        self.destinations = self.netdata['destination'] 
        self.add_origin_routes()
        self.scale = scale
        self.sim_len = sim_len
        self.t = 0

        ###determine what function we run every step to 
        ###generate vehicles into sim
        if demand == 'single':
            self.gen_vehicles = self.gen_single
        elif demand == 'dynamic':
            self.v_schedule = self.gen_dynamic_demand(mode)
            self.gen_vehicles = self.gen_dynamic

    def run(self):
        self.gen_vehicles()
        self.t += 1

    def gen_dynamic(self):
        ###get next set of edges from v schedule, use them to add new vehicles
        ###this is batch vehicle generation
        try:
            new_veh_edges = next(self.v_schedule)
            self.gen_veh( new_veh_edges  )
        except StopIteration:
            print('no vehicles left')

    def gen_dynamic_demand(self, mode):
        ###use sine wave as rate parameter for dynamic traffic demand
        t = np.linspace(1*np.pi, 2*np.pi, self.sim_len)                                          
        sine = np.sin(t)+1.55
        ###create schedule for number of vehicles to be generated each second in sim
        v_schedule = []
        second = 1.0
        for t in range(int(self.sim_len)):
            n_veh = 0.0
            while second > 0.0:
                headway = np.random.exponential( sine[t], size=1)
                second -= headway
                if second > 0.0:
                    n_veh += 1
            second += 1.0
            v_schedule.append(int(n_veh))
                                                                                            
        ###randomly shift traffic pattern as a form of data augmentation
        v_schedule = np.array(v_schedule)
        if mode == 'test':
            random_shift = 0
        else:
            random_shift = np.random.randint(0, self.sim_len)
        v_schedule = np.concatenate((v_schedule[random_shift:], v_schedule[:random_shift]))
        ###zero out the last minute for better comparisons because of random shift
        v_schedule[-60:] = 0
        ###randomly select from origins, these are where vehicles are generated
        v_schedule = [ np.random.choice(self.origins, size=int(self.scale*n_veh), replace = True) 
                       if n_veh > 0 else [] for n_veh in v_schedule  ]
        ###fancy iterator, just so we can call next for sequential access
        return v_schedule.__iter__() 

    def add_origin_routes(self):
        for origin in self.origins:
            self.conn.route.add(origin, [origin] )

    def gen_single(self):
        if self.conn.vehicle.getIDCount() == 0:
            ###if no vehicles in sim, spawn 1 on random link
            veh_spawn_edge = np.random.choice(self.origins)
            self.gen_veh( [veh_spawn_edge] )

    def gen_veh( self, veh_edges ):
        for e in veh_edges:
            vid = e+str(self.vehicles_created)
            self.conn.vehicle.addFull( vid, e, departLane="best" )
            self.set_veh_route(vid)
            self.vehicles_created += 1

    def set_veh_route(self, veh):
        current_edge = self.conn.vehicle.getRoute(veh)[0]
        route = [current_edge]
        while current_edge not in self.destinations:
            next_edge = np.random.choice(self.netdata['edge'][current_edge]['outgoing'])
            route.append(next_edge)
            current_edge = next_edge
        self.conn.vehicle.setRoute( veh, route )    
