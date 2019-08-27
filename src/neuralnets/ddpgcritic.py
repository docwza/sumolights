#https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404
import os

import numpy as np
import tensorflow as tf
from tensorflow.initializers import he_uniform

from src.picklefuncs import save_data, load_data
from src.neuralnet import NeuralNet
from src.neuralnets.varstate import VariableState
from src.helper_funcs import check_and_make_dir

class DDPGCriticNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, name, sess):
        with tf.compat.v1.variable_scope(name, reuse=tf.compat.v1.AUTO_REUSE):
            #build critic network
            self.input = tf.compat.v1.placeholder(tf.float32,
                                        shape=[None, input_d],
                                        name='inputs')
                                                                                                                  
            self.actions = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, output_d],
                                          name='actions')
                                                                                                                  
            self.q_target = tf.compat.v1.placeholder(tf.float32,
                                           shape=[None,1],
                                           name='targets')
                                                                                                                  
            dense1 = tf.compat.v1.layers.dense(self.input, units=hidden_d[0],
                                     kernel_initializer=he_uniform())
            batch1 = tf.compat.v1.layers.batch_normalization(dense1)
            layer1_activation = tf.nn.elu(batch1)
                                                                                                                  
            dense2 = tf.compat.v1.layers.dense(layer1_activation, units=hidden_d[1],
                                     kernel_initializer=he_uniform())
            batch2 = tf.compat.v1.layers.batch_normalization(dense2)
            layer2_activation = tf.nn.elu(batch2)
                                                                                                                  
            '''
            action_in = tf.compat.v1.layers.dense(self.actions, units=hidden_d[1],
                                        activation='elu')
            state_actions = tf.add(batch2, action_in)
            state_actions = tf.nn.elu(state_actions)
            self.q = tf.compat.v1.layers.dense(state_actions, units=1,
                               kernel_initializer=he_uniform(),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))

            '''
            state_actions = tf.concat( [layer2_activation, self.actions], axis=-1 )

            dense3 = tf.compat.v1.layers.dense(state_actions, units=hidden_d[1],
                                     kernel_initializer=he_uniform())
            batch3 = tf.compat.v1.layers.batch_normalization(dense3)
            layer3_activation = tf.nn.elu(batch3)
                                                                                                                  
            self.q = tf.compat.v1.layers.dense(layer3_activation, units=1,
                               kernel_initializer=he_uniform(),
                               kernel_regularizer=tf.keras.regularizers.l2(0.01))
                                                                                                                  
            self.loss = tf.compat.v1.losses.mean_squared_error(self.q_target, self.q)

            self.params = tf.compat.v1.trainable_variables(scope=name)

            #optimizer
            self.optimize = tf.compat.v1.train.AdamOptimizer(learning_rate = lr, epsilon=lre).minimize(self.loss)
            self.action_gradients = tf.gradients(self.q, self.actions)

            #saving stuff
            self.varstate = VariableState(sess, self.params)

class DDPGCritic(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, tau, learner=False, name='', sess=None):
        self.sess=sess
        self.name = name
        self.lr = lr
        self.lre = lre
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner)
        self.tau = tau
        
        self.new_w = []
        self.varstate = {}
            
        if learner:
            self.update_critic = [self.models['target'].params[i].assign(
                                  tf.multiply(self.models['online'].params[i], self.tau)
                                  + tf.multiply(self.models['target'].params[i], 1. - self.tau))
                                  for i in range(len(self.models['target'].params))]

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act):
        return DDPGCriticNet(input_d, hidden_d, hidden_act, output_d, output_act, self.lr, self.lre, self.name, self.sess)

    def gradients(self, states, actions):
        return self.sess.run(self.models['online'].action_gradients,
                             feed_dict={self.models['online'].input: states,
                                        self.models['online'].actions: actions})

    def forward(self, x, a, nettype):
        return self.sess.run(self.models[nettype].q,
                             feed_dict={self.models[nettype].input: x,
                                        self.models[nettype].actions: a})

    def backward(self, states, actions, critic_target):
        return self.sess.run(self.models['online'].optimize,
                      feed_dict={self.models['online'].input: states,
                                 self.models['online'].actions: actions,
                                 self.models['online'].q_target: critic_target})

    def transfer_weights(self):
        """ Transfer model weights to target model with a factor of Tau
        """
        self.sess.run(self.update_critic)

    def get_weights(self, nettype):
        #return self.sess.run(self.models[nettype].params)
        return self.models[nettype].varstate.export_variables()

    def set_weights(self, weights, nettype):
        self.models[nettype].varstate.import_variables(weights)

    def save_weights(self, nettype, path, fname):
        check_and_make_dir(path)
        weights = self.get_weights('online')
        save_data(path+fname+'.p', weights)

    def load_weights(self, path):
        path += '.p' 
        if os.path.exists(path):
            weights = load_data(path)
            self.set_weights(weights, 'online')
        else:
            #raise not found exceptions
            assert 0, 'Failed to load weights, supplied weight file path '+str(path)+' does not exist.'

if __name__ == '__main__':
    input_d = [40]
    hidden_d = [40, 40]
    output_d = 1
    lr = 0.0001
    lre = 0.00001
    tau = 0.005
    learner = True
    name = 'tsc0'
    ddpg = DDPGCritic(input_d, hidden_d, 'relu', output_d, 'tanh', lr, lre, tau, learner=learner, name=name, sess=None)
    weights = ddpg.get_weights('online')
    print(type(weights))
    print(weights)
    print(weights[0])

