import numpy as np
import random

from src.rlagent import RLAgent

class DQNAgent(RLAgent):
    def __init__(self, networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates):
        super().__init__(networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates) 
        
    def get_action(self, state):
        #get newest weights before acting
        #get q values of current state
        if np.random.uniform(0.0, 1.0) < self.epsilon:
            ###act randomly
            action = np.random.randint(self.n_actions)
        else:
            if self.mode == 'train':
                self.retrieve_weights('online')
            ###act greedily
            q_state = self.networks.forward(state[np.newaxis,...], 'online')
            action = np.argmax(q_state)
        ###return action integer
        return action

    def train_batch(self, update_freq):
        ###sample from replay
        sample_batch = self.sample_replay()
        ###process nstep, generate n step returns
        batch_inputs, batch_targets = self.process_batch(sample_batch)
        self.networks.backward(batch_inputs, batch_targets)
        self.rl_stats['updates'] += 1
        self.rl_stats['n_exp'] -= 1
        #send online weights for actor processes
        self.send_weights('online')

        if self.rl_stats['updates'] % update_freq == 0:
            self.networks.transfer_weights()

    def process_batch(self, sample_batch):
        ###each list in the sample batch is an experience trajectory
        ###use experiences in trajectory to generate targets
        max_r = self.rl_stats['max_r']

        processed_exps = []

        states, next_states, terminals = [], [], []
        for trajectory in sample_batch:
            next_states.append(trajectory[-1]['next_s'])
            terminals.append(trajectory[-1]['terminal'])
            for exp in trajectory:
                states.append(exp['s'])
                ###normalize reward by comparison to maximum reward 
                ###agent has experienced across all actors
                
        #batch forward q_s estimate
        Q_S = self.networks.forward(np.stack(states), 'target')
        #batch forward bootstrap
        R = self.next_state_bootstrap(np.stack(next_states), terminals)

        i = 0
        processed_exps = []
        for trajectory, r in zip(sample_batch, R):
            states, actions, rewards = [], [], []
            for exp in trajectory:
                states.append(exp['s'])
                actions.append(exp['a'])
                ###normalize reward by comparison to maximum reward 
                ###agent has experienced across all actors
                rewards.append(exp['r']/max_r)
            #get q values 
            q_s = Q_S[i:i+len(trajectory)]
            i += len(trajectory)
            p_exps = self.process_trajectory( states, actions, 
                                              rewards, r, q_s )
            ###add processed experiences from trajectory to batch
            processed_exps.extend(p_exps)
        ###account for n step returns, randomly select batch for training
        if self.n_steps > 1:
            processed_exps = random.sample(processed_exps, self.n_batch)
        batch_inputs = np.squeeze(np.stack([ e['s'] for e in processed_exps]))
        batch_targets =  np.stack([ e['target'] for e in processed_exps])
        return batch_inputs, batch_targets

    def next_state_bootstrap(self, next_states, terminals):
        
        q_next_s = self.networks.forward(next_states, 'target')
        R = np.amax(q_next_s, axis=-1) 

        return [ 0.0 if t is True else r for t, r in zip(terminals, R)]
                                                                                         
    def process_trajectory(self, states, actions, rewards, R, q_s):
                                                                                         
        targets = self.compute_targets(rewards, R)
                                                                                         
        exps = []
        for i in range(len(actions)):
            q_s[i, actions[i]] = targets[i]
            exps.append({'target':q_s[i], 's':states[i]})
        return exps
    
    def set_params(self, nettype, weights):
        self.networks.set_weights(weights, nettype)

    def get_params(self, nettype):
        return self.networks.get_weights(nettype)

    def send_weights(self, nettype):
        self.rl_stats[nettype] = self.networks.get_weights(nettype)

    def retrieve_weights(self, nettype):
        self.networks.set_weights(self.rl_stats[nettype], nettype)

