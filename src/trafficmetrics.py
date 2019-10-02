import os, sys

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    from sumolib import checkBinary
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci

class TrafficMetrics:
    def __init__(self, _id, incoming_lanes, netdata, metric_args, mode):
        self.metrics = {}
        if 'delay' in metric_args:
            lane_lengths = {lane:netdata['lane'][lane]['length'] for lane in incoming_lanes}
            lane_speeds = {lane:netdata['lane'][lane]['speed'] for lane in incoming_lanes}
            self.metrics['delay'] = DelayMetric(_id, incoming_lanes, mode, lane_lengths, lane_speeds )

        if 'queue' in metric_args:
            self.metrics['queue'] = QueueMetric(_id, incoming_lanes, mode)

    def update(self, v_data):
        for m in self.metrics:
            self.metrics[m].update(v_data)

    def get_metric(self, metric):
        return self.metrics[metric].get_metric()

    def get_history(self, metric):
        return self.metrics[metric].get_history()

class TrafficMetric:
    def __init__(self, _id, incoming_lanes, mode):
        self.id = _id
        self.incoming_lanes = incoming_lanes
        self.history = []
        self.mode = mode

    def get_metric(self):
        pass

    def update(self):
        pass

    def get_history(self):
        return self.history

class DelayMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode, lane_lengths, lane_speeds):
        super().__init__( _id, incoming_lanes, mode)
        self.lane_travel_times = {lane:lane_lengths[lane]/float(lane_speeds[lane]) for lane in incoming_lanes}
        self.old_v = set()
        self.v_info = {}
        self.t = 0

    def get_v_delay(self, v):
        return ( self.t - self.v_info[v]['t'] ) - self.lane_travel_times[self.v_info[v]['lane']]

    def get_metric(self):
        #calculate delay of vehicles on incoming lanes
        delay = 0
        for v in self.old_v:
            #calculate individual vehicle delay
            v_delay = self.get_v_delay(v)
            if v_delay > 0:
                delay += v_delay

        return delay

    def update(self, v_data):
        new_v = set()

        #record start time and lane of new_vehicles
        for lane in self.incoming_lanes:
            for v in v_data[lane]:
                if v not in self.old_v:
                    self.v_info[v] = {}
                    self.v_info[v]['t'] = self.t
                    self.v_info[v]['lane'] = lane
            new_v.update( set(v_data[lane].keys()) )

        if self.mode == 'test':
            self.history.append(self.get_metric())

        #remove vehicles that have left incoming lanes
        remove_vehicles = self.old_v - new_v
        delay = 0
        for v in remove_vehicles:
            del self.v_info[v]
        
        self.old_v = new_v
        self.t += 1

class QueueMetric(TrafficMetric):
    def __init__(self, _id, incoming_lanes, mode):
        super().__init__( _id, incoming_lanes, mode)
        self.stop_speed = 0.3
        self.lane_queues = {lane:0 for lane in self.incoming_lanes}

    def get_metric(self):
        return sum([self.lane_queues[lane] for lane in self.lane_queues])

    def update(self, v_data):
        lane_queues = {}
        for lane in self.incoming_lanes:
            lane_queues[lane] = 0
            for v in v_data[lane]:
                if v_data[lane][v][traci.constants.VAR_SPEED] < self.stop_speed:
                    lane_queues[lane] += 1

        self.lane_queues = lane_queues
        if self.mode == 'test':
            self.history.append(self.get_metric())
