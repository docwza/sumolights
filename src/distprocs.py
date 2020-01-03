import sys, os, subprocess, time
from multiprocessing import *

from tensorflow.python.framework.ops import disable_eager_execution

from src.simproc import SimProc
from src.learnerproc import LearnerProc
from src.networkdata import NetworkData
from src.sumosim import SumoSim
import numpy as np


def get_sim(sim_str):
    if sim_str == 'lust':                                        
        cfg_fp = 'networks/lust/scenario/dua.actuated.sumocfg'     
        net_fp = 'networks/lust/scenario/lust.net.xml'               
    elif sim_str == 'single':                                       
        cfg_fp = 'networks/single.sumocfg'                         
        net_fp = 'networks/single.net.xml'                           
    elif sim_str == 'double':                                       
        cfg_fp = 'networks/double.sumocfg'                         
        net_fp = 'networks/double.net.xml'                           
    return cfg_fp, net_fp                                           


class DistProcs:
    def __init__(self, args, tsc, mode):
        self.args = args
        rl_tsc = ['ddpg', 'dqn']
        traditional_tsc = ['websters', 'maxpressure', 'sotl', 'uniform']

        # depending on tsc alg, different hyper param checks
        if tsc in rl_tsc:
            # disable_eager_execution()
            # need actors and at least one learner
            if mode == 'train':
                # ensure we have at least one learner
                if args.l < 1:
                    args.l = 1
            elif mode == 'test':
                # no learners necessary for testing
                if args.l > 0:
                    args.l = 0
        elif tsc in traditional_tsc:
            # traditional tsc doesn't require learners
            if args.l > 0:
                args.l = 0
        else:
            print('Input argument tsc '+str(tsc)+' not found, please provide valid tsc.')
            return

        # ensure at least one sim, otherwise no point in running program
        if args.n < 0:
            args.n = 1

        # if sim arg provided, use to get cfg and netfp
        # otherwise, continue with args default
        if args.sim:
            args.cfg_fp, args.net_fp = get_sim(args.sim)

        args.nreplay = int(args.nreplay/args.nsteps)

        barrier = Barrier(args.n+args.l)

        nd = NetworkData(args.net_fp)
        netdata = nd.get_net_data()

        # create a dummy sim to get tsc data for creating nn
        # print('creating dummy sim for netdata...')
        sim = SumoSim(args.cfg_fp, args.sim_len, args.tsc, True, netdata, args, -1)
        sim.gen_sim()
        netdata = sim.update_netdata()
        sim.close()
        # print('...finished with dummy sim')

        tsc_ids = netdata['inter'].keys()

        # create mp dict for sharing
        # reinforcement learning stats
        rl_stats = self.create_mp_stats_dict(tsc_ids)
        exp_replays = self.create_mp_exp_replay(tsc_ids)

        eps_rates = self.get_exploration_rates(args.eps, args.n, args.mode, args.sim)
        print(eps_rates)
        offsets = self.get_start_offsets(args.mode, args.sim_len, args.offset, args.n)
        print(offsets)

        # create sumo sim procs to generate experiences
        sim_procs = [SimProc(i, args, barrier, netdata, rl_stats, exp_replays, eps_rates[i], offsets[i]) for i in range(args.n)]

        # create learner procs which are assigned tsc/rl agents
        # to compute neural net updates for
        if args.l > 0:
            learner_agents = self.assign_learner_agents( tsc_ids, args.l)
            print('===========LEARNER AGENTS')
            for l in learner_agents:
                print('============== '+str(l))
            learner_procs = [ LearnerProc(i, args, barrier, netdata, learner_agents[i], rl_stats, exp_replays) for i in range(args.l)]
        else:
            learner_procs = []

        # learner_procs = []

        self.procs = sim_procs + learner_procs
        print("self procs:\t", self.procs)

    def run(self):
        print('Starting up all processes...')
        # start everything
        for p in self.procs:
            p.start()
                              
        # join when finished
        for p in self.procs:
            p.join()

        print('...finishing all processes')

    def create_mp_stats_dict(self, tsc_ids):
        # use this mp shared dict for data between procs
        manager = Manager()
        rl_stats = manager.dict({})
        for i in tsc_ids:
            rl_stats[i] = manager.dict({})
            rl_stats[i]['n_exp'] = 0
            rl_stats[i]['updates'] = 0
            rl_stats[i]['max_r'] = 1.0
            rl_stats[i]['online'] = None
            rl_stats[i]['target'] = None
            rl_stats['n_sims'] = 0
            rl_stats['total_sims'] = 104
            rl_stats['delay'] = manager.list()
            rl_stats['queue'] = manager.list()
            rl_stats['throughput'] = manager.list()

        return rl_stats

    def create_mp_exp_replay(self, tsc_ids):
        # create shared memory for experience replay
        # (governs agents appending and learners accessing and deleting)
        manager = Manager()
        return manager.dict({ tsc: manager.list() for tsc in tsc_ids })

    def assign_learner_agents(self, agents, n_learners):
        learner_agents = [ [] for _ in range(n_learners)]
        for agent, i in zip(agents, range(len(agents))):
            learner_agents[i%n_learners].append(agent)
        # list of lists, each sublit is the agents a learner is responsible for
        return learner_agents

    def get_exploration_rates(self, eps, n_actors, mode, net):
        if mode == 'test':
            return [eps for _ in range(n_actors)]
        elif mode == 'train':
            if net == 'lust':
                # for lust we restrict the exploration rates
                e = [1.0, 0.5, eps]
                erates = []
                for i in range(n_actors):
                    erates.append(e[i%len(e)])
                return erates
            else:
                return np.linspace(1.0, eps, num = n_actors) 

    def get_start_offsets(self, mode, simlen, offset, n_actors):
        if mode == 'test':
            return [0]*n_actors
        elif mode == 'train':
            return np.linspace(0, simlen*offset, num = n_actors) 
