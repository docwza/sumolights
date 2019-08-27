import sys, subprocess, os
import inspect

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
'''
# we need to import python modules from the $SUMO_HOME/tools directory
try:
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', "tools")) # tutorial in tests
    sys.path.append(os.path.join(os.environ.get("SUMO_HOME", os.path.join(os.path.dirname(__file__), "..", "..", "..")), "tools")) # tutorial in docs
    #from sumolib import checkBinary
    import sumolib
except ImportError:
    sys.exit("please declare environment variable 'SUMO_HOME' as the root directory of your sumo installation (it should contain folders 'bin', 'tools' and 'docs')")
'''

import numpy as np

class NetworkData:
    def __init__(self, net_fp):
        print(net_fp) 
        self.net = sumolib.net.readNet(net_fp)
        ###get edge data
        self.edge_data = self.get_edge_data(self.net)
        self.lane_data = self.get_lane_data(self.net)
        self.node_data, self.intersection_data = self.get_node_data(self.net)
        print("SUCCESSFULLY GENERATED NET DATA")

    def get_net_data(self):
        return {'lane':self.lane_data, 'edge':self.edge_data, 'origin':self.find_origin_edges(), 'destination':self.find_destination_edges(), 'node':self.node_data, 'inter':self.intersection_data}

    def find_destination_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['incoming']:
                next_edges[next_e] += 1
                                                                 
        destinations = [ e for e in next_edges if next_edges[e] == 0]
        return destinations

    def find_origin_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['outgoing']:
                next_edges[next_e] += 1

        origins = [ e for e in next_edges if next_edges[e] == 0]
        return origins

    def get_edge_data(self, net):
        edges = net.getEdges()
        edge_data = {str(edge.getID()):{} for edge in edges} 

        for edge in edges:
            edge_ID = str(edge.getID())
            edge_data[edge_ID]['lanes'] = [str(lane.getID()) for lane in edge.getLanes()]
            edge_data[edge_ID]['length'] = float(edge.getLength())
            edge_data[edge_ID]['outgoing'] = [str(out.getID()) for out in edge.getOutgoing()]
            edge_data[edge_ID]['noutgoing'] = len(edge_data[edge_ID]['outgoing'])
            edge_data[edge_ID]['nlanes'] = len(edge_data[edge_ID]['lanes'])
            edge_data[edge_ID]['incoming'] = [str(inc.getID()) for inc in edge.getIncoming()]
            edge_data[edge_ID]['outnode'] = str(edge.getFromNode().getID())
            edge_data[edge_ID]['incnode'] = str(edge.getToNode().getID())
            edge_data[edge_ID]['speed'] = float(edge.getSpeed())
          
            ###coords for each edge
            incnode_coord = edge.getFromNode().getCoord()
            outnode_coord = edge.getToNode().getCoord()
            edge_data[edge_ID]['coord'] = np.array([incnode_coord[0], incnode_coord[1], outnode_coord[0], outnode_coord[1]]).reshape(2,2)
            #print edge_data[edge_ID]['coord']
        return edge_data 

    def get_lane_data(self, net):
        #create lane objects from lane_ids
        lane_ids = []
        for edge in self.edge_data:
            lane_ids.extend(self.edge_data[edge]['lanes'])

        lanes = [net.getLane(lane) for lane in lane_ids]
        #lane data dict
        lane_data = {lane:{} for lane in lane_ids}

        for lane in lanes:
            lane_id = lane.getID()
            lane_data[ lane_id ]['length'] = lane.getLength()
            lane_data[ lane_id ]['speed'] = lane.getSpeed()
            lane_data[ lane_id ]['edge'] = str(lane.getEdge().getID())
            #lane_data[ lane_id ]['outgoing'] = []
            lane_data[ lane_id ]['outgoing'] = {}
            ###.getOutgoing() returns a Connection type, which we use to determine the next lane
            moveid = []
            for conn in lane.getOutgoing():
                out_id = str(conn.getToLane().getID())
                lane_data[ lane_id ]['outgoing'][out_id] = {'dir':str(conn.getDirection()), 'index':conn.getTLLinkIndex()}
                moveid.append(str(conn.getDirection()))
            lane_data[ lane_id ]['movement'] = ''.join(sorted(moveid))
            #create empty list for incoming lanes 
            lane_data[ lane_id ]['incoming'] = []
               
        #determine incoming lanes using outgoing lanes data
        for lane in lane_data:
            for inc in lane_data:
                if lane == inc:
                    continue
                else:
                    if inc in lane_data[lane]['outgoing']:
                        lane_data[inc]['incoming'].append(lane)

        return lane_data


    def get_node_data(self, net):
        #read network from .netfile
        nodes = net.getNodes()
        node_data = {node.getID():{} for node in nodes}

        for node in nodes:
            node_id = node.getID()
            #get incoming/outgoing edge
            node_data[node_id]['incoming'] = set(str(edge.getID()) for edge in node.getIncoming())
            node_data[node_id]['outgoing'] = set(str(edge.getID()) for edge in node.getOutgoing())
            node_data[node_id]['tlsindex'] = { conn.getTLLinkIndex():str(conn.getFromLane().getID()) for conn in node.getConnections()}
            node_data[node_id]['tlsindexdir'] = { conn.getTLLinkIndex():str(conn.getDirection()) for conn in node.getConnections()}

            if node_id == '-13968':
                missing = []
                negative = []
                for i in range(len(node_data[node_id]['tlsindex'])):
                    if i not in node_data[node_id]['tlsindex']:
                        missing.append(i)

                for k in node_data[node_id]['tlsindex']:
                    if k < 0  :
                        negative.append(k)
              
                for m,n in zip(missing, negative):
                    node_data[node_id]['tlsindex'][m] = node_data[node_id]['tlsindex'][n]
                    del node_data[node_id]['tlsindex'][n]
                    #for index dir
                    node_data[node_id]['tlsindexdir'][m] = node_data[node_id]['tlsindexdir'][n]
                    del node_data[node_id]['tlsindexdir'][n]
            
            #get XY coords
            pos = node.getCoord()
            node_data[node_id]['x'] = pos[0]
            node_data[node_id]['y'] = pos[1]

        intersection_data = {str(node):node_data[node] for node in node_data if "traffic_light" in net.getNode(node).getType()} 

        return node_data, intersection_data
