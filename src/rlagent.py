import numpy as np

class RLAgent:
    def __init__(self, networks, epsilon, exp_replay, n_actions, n_steps, n_batch, n_exp_replay, gamma, rl_stats, mode, updates):
        ###this is a dict, keys = 'online', 'target'
        self.networks = networks
        self.epsilon = epsilon
        self.exp_replay = exp_replay
        self.n_actions = n_actions
        self.n_steps = n_steps
        self.n_batch = n_batch
        self.n_exp_replay = n_exp_replay
        self.gamma = gamma
        self.experience_trajectory = []
        self.rl_stats = rl_stats
        self.exp_replay = exp_replay
        self.mode = mode
        self.updates = updates

    def get_action(self, state):
       pass 

    def store_experience(self, state, action, next_state, reward, terminal):
        ### here we append to a temporary experience sequence/trajectory buffer, 
        #and when terminal or steps length, at to experience replay
        if self.rl_stats['updates'] < self.updates:
            experience = {'s':state, 'a':action,                                     
                          'next_s':next_state, 'r':reward, 'terminal':terminal}
                                                                                     
            #append experience to trajectory
            self.experience_trajectory.append(experience)
                                                                                    
            ###check if need to add trajectory to exp replay
            if len(self.experience_trajectory) == self.n_steps or terminal == True:
                self.exp_replay.append(self.experience_trajectory)
                #rl stats bookkeeping
                self.rl_stats['n_exp'] += 1
                self.experience_trajectory = []

            #update maximum reward
            abs_reward = np.abs(reward)
            if abs_reward > self.rl_stats['max_r']:
                self.rl_stats['max_r'] = abs_reward

    def train_batch(self, update_freq):
        pass

    def process_batch(self, sample_batch):
        pass 

    def process_trajectory(self):
        pass

    def compute_targets(self, rewards, R):
        ###compute targets using discounted rewards
        target_batch = []

        for i in reversed(range(len(rewards))):
            R = rewards[i] + (self.gamma * R)
            target_batch.append(R)

        target_batch.reverse()
        return target_batch

    def sample_replay(self):
        ###randomly sampled trajectories from shared experience replay
        idx = np.random.randint(0, self.n_exp_replay, size = self.n_batch)
        return [ self.exp_replay[i] for i in idx ]

    def clip_exp_replay(self):
        diff = len(self.exp_replay) - self.n_exp_replay
        if diff > 0:
            del self.exp_replay[:diff]

    def send_weights(self):
        pass

    def retrieve_weights(self):
        pass

