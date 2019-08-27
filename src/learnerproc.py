import time, os
from multiprocessing import *
import tensorflow as tf

import numpy as np

from tensorflow.python.framework.ops import disable_eager_execution

from src.nn_factory import gen_neural_networks
from src.rl_factory import rl_factory
from src.helper_funcs import write_line_to_file, check_and_make_dir, get_time_now, write_to_log
from src.picklefuncs import save_data, load_data

class LearnerProc(Process):
    def __init__(self, idx, args, barrier, netdata, agent_ids, rl_stats, exp_replay):
        Process.__init__(self)
        self.idx = idx
        self.args = args
        self.barrier = barrier
        self.netdata = netdata
        self.agent_ids = agent_ids
        self.rl_stats = rl_stats
        self.exp_replay = exp_replay
        self.save_t = 0
        self.replay_fp =  self.args.save_replay+'/'+self.args.tsc+'/'
        #for saving agent progress
        if self.idx == 0:
            path = 'tmp/'                                                                    
            check_and_make_dir(path)
            now = get_time_now()
            self.updates_path = path + str(self.args.tsc)+'_'+str(now)+'_agent_updates.csv' 
            self.replay_path = path + str(self.args.tsc)+'_'+str(now)+'_agent_replay.csv' 
            self.n_exp_path = path + str(self.args.tsc)+'_'+str(now)+'_agent_nexp.csv' 
            self.tsc_ids = list(sorted(list(self.netdata['inter'].keys())))
            #write header line with tsc names
            write_line_to_file( self.updates_path, 'a+', ','.join([now]+self.tsc_ids) )
            write_line_to_file( self.replay_path, 'a+', ','.join([now]+self.tsc_ids) )
            write_line_to_file( self.n_exp_path, 'a+', ','.join([now]+self.tsc_ids) )

    def run(self):
        #gen neural networks
        learner = True
        
        neural_networks = gen_neural_networks(self.args, 
                                              self.netdata, 
                                              self.args.tsc, 
                                              self.agent_ids,
                                              learner,
                                              self.args.load,
                                              self.args.n_hidden)

        print('learner proc trying to send weights------------')
        write_to_log(' LEARNER #'+str(self.idx)+' SENDING WEIGHTS...')

        neural_networks = self.distribute_weights(neural_networks) 
        #wait for all procs to sync weights
        print('learner waiting at barrier ------------')
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED SENDING WEIGHTS, WAITING AT BARRIER...')
        self.barrier.wait()
        write_to_log(' LEARNER #'+str(self.idx)+' GENERATING AGENTS...')

        if self.args.load_replay:
            self.load_replays()

        #create agents
        agents = self.gen_agents(neural_networks)

        print('learner proc '+str(self.idx)+' waiting at offset barrier------------')
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED GEN AGENTS, WAITING AT OFFSET BARRIER...')
        self.barrier.wait()
        write_to_log(' LEARNER #'+str(self.idx)+' BROKEN OFFSET BARRIER...')
        print('learner proc '+str(self.idx)+' broken offset barrier ------------')

        self.save_t = time.time()
        othert = time.time()
        #keep looping until all agents have
        #achieved sufficient batch updates
        while not self.finished_learning(self.agent_ids):
            for tsc in self.agent_ids:
                #wait until exp replay buffer full
                if len(self.exp_replay[tsc]) >= self.args.nreplay:
                    #reset the number of experiences once when the 
                    #exp replay is filled for the first time
                    if self.rl_stats[tsc]['updates'] == 0:
                        if self.args.save:
                            self.save_replays()
                        print(tsc+' exp replay full, beginning batch updates********')
                        #write_to_log(' LEARNER #'+str(self.idx)+' START LEARNING '+str(tsc))
                        self.rl_stats[tsc]['n_exp'] = len(self.exp_replay[tsc])
                    if self.rl_stats[tsc]['updates'] < self.args.updates and self.rl_stats[tsc]['n_exp'] > 0: 
                        for i in range(min(self.rl_stats[tsc]['n_exp'], 4)):
                           agents[tsc].train_batch(self.args.target_freq)
                        agents[tsc].clip_exp_replay()

            t = time.time()
            if t - othert > 90:
                othert = t
                n_replay = [str(len(self.exp_replay[i])) for i in self.agent_ids]
                updates = [str(self.rl_stats[i]['updates']) for i in self.agent_ids]
                nexp = [str(self.rl_stats[i]['n_exp']) for i in self.agent_ids]
                write_to_log(' LEARNER #'+str(self.idx)+'\n'+str(self.agent_ids)+'\n'+str(nexp)+'\n'+str(n_replay)+'\n'+str(updates))                           


            #save weights periodically
            if self.args.save:
                if self.time_to_save():
                    self.save_weights(neural_networks)

                    #write agent training progress
                    #only on one learner
                    if self.idx == 0:
                        self.write_progress()
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED TRAINING LOOP ===========')

        if self.idx == 0:
            #if other agents arent finished learning
            #keep updating progress
            while not self.finished_learning(self.tsc_ids):
                if self.time_to_save():
                    self.write_progress()


        if self.args.save:
            self.save_weights(neural_networks)
        print('finished learning for all agents on learner proc '+str(self.idx))
        n_replay = [str(len(self.exp_replay[i])) for i in self.agent_ids]
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED REPLAY '+str(n_replay))
        updates = [str(self.rl_stats[i]['updates']) for i in self.agent_ids]
        write_to_log(' LEARNER #'+str(self.idx)+' FINISHED UPDATES'+str(updates))

    def time_to_save(self):
        t = time.time()                        
        if t - self.save_t > self.args.save_t:
            self.save_t = t
            return True
        return False

    def write_progress(self):
        self.write_training_progress()
        self.write_replay_progress()
        self.write_n_exp_progress()

    def finished_learning(self, agent_ids):
        for agent in agent_ids:
            if self.rl_stats[agent]['updates'] < self.args.updates:
                return False
        return True

    def gen_agents(self, neural_networks):
        agents = {}
        for agent in self.agent_ids:
            n_actions = 1 if self.args.tsc == 'ddpg' else len(self.netdata['inter'][agent]['green_phases'])
            agents[agent] = rl_factory(self.args.tsc, 
                                       self.args, 
                                       neural_networks[agent], 
                                       self.exp_replay[agent], 
                                       self.rl_stats[agent], 
                                       n_actions,
                                       self.args.eps)
        return agents
        
    def distribute_weights(self, neural_networks):
        for nn in neural_networks:
            if self.args.tsc == 'ddpg':
                #sync actor weights
                weights = neural_networks[nn]['actor'].get_weights('online')
                critic_weights = neural_networks[nn]['critic'].get_weights('online')
                #synchronize target/online weights
                neural_networks[nn]['actor'].set_weights(weights, 'target')
                neural_networks[nn]['critic'].set_weights(critic_weights, 'target')
            elif self.args.tsc == 'dqn':
                weights = neural_networks[nn].get_weights('online')
                #synchronize target/online weights
                neural_networks[nn].set_weights(weights, 'target')
            else:
                #raise not found exceptions
                assert 0, 'Supplied RL traffic signal controller '+str(self.args.tsc)+' does not exist.'
            #send weights to sim processes
            self.rl_stats[nn]['online'] = weights
        return neural_networks

    def save_weights(self, neural_networks):
        path_dirs = [self.args.save_path, self.args.tsc]
        for nn in neural_networks:
            if self.args.tsc == 'ddpg':
                path = '/'.join(path_dirs+['critic'])+'/'
                neural_networks[nn]['critic'].save_weights('online', path, nn)
                path = '/'.join(path_dirs+['actor'])+'/'
                neural_networks[nn]['actor'].save_weights('online', path, nn)
            elif self.args.tsc == 'dqn':
                path = '/'.join(path_dirs)+'/'
                neural_networks[nn].save_weights('online', path, nn)
            else:
                #raise not found exceptions
                assert 0, 'Supplied RL traffic signal controller '+str(self.args.tsc)+' does not exist, cannot save.'

    def write_training_progress(self):
        updates = [str(self.rl_stats[i]['updates']) for i in self.tsc_ids]
        write_line_to_file( self.updates_path, 'a+', ','.join([get_time_now()]+updates) )

    def write_replay_progress(self):
        n_replay = [str(len(self.exp_replay[i])) for i in self.tsc_ids]
        write_line_to_file( self.replay_path, 'a+', ','.join([get_time_now()]+n_replay) )

    def write_n_exp_progress(self):
        n_replay = [str(self.rl_stats[i]['n_exp']) for i in self.tsc_ids]
        write_line_to_file( self.n_exp_path, 'a+', ','.join([get_time_now()]+n_replay) )

    def save_replays(self):
        check_and_make_dir(self.replay_fp)
        for _id in self.agent_ids:                                     
            save_data(self.replay_fp+_id+'.p', [ _ for _ in self.exp_replay[_id]])
            print('FINISHED SAVING REPLAY FOR '+str(_id))

    def load_replays(self):
        for _id in self.agent_ids:
            replay_fp = self.replay_fp+_id+'.p' 
            if os.path.isfile(replay_fp):
                data = load_data(replay_fp)
                rewards = []
                for traj in data:
                    for exp in traj:
                        rewards.append(abs(exp['r']))
                    self.exp_replay[_id].append(traj) 
                #find largest reward to reward normalization
                print('mean '+str(np.mean(rewards))+' std '+str(np.std(rewards))+' median '+str(np.median(rewards)))
                self.rl_stats[_id]['r_max'] = max(rewards)
                print(str(self.idx)+' LARGEST REWARD '+str(self.rl_stats[_id]['r_max']))
                print('SUCCESSFULLY LOADED REPLAY FOR '+str(_id))
            else:
                print('WARNING, tried to load experience replay at '+str(replay_fp)+' but it does not exist, continuing without loading...')
