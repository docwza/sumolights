import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten
from tensorflow.keras.optimizers import Adam

from src.neuralnet import NeuralNet
from src.helper_funcs import check_and_make_dir

class DQN(NeuralNet):
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, lr, lre, learner=False):
        super().__init__(input_d, hidden_d, hidden_act, output_d, output_act, learner=learner)
        for model in self.models:
            #self.models[model].compile(Adam(learning_rate=lr, epsilon=lre), loss='mse')
            self.models[model].compile(Adam(lr=lr, epsilon=lre), loss='mse')

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act):
        model_in = Input((input_d,))
        for i in range(len(hidden_d)):
            if i == 0:
                model = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model_in)
            else:
                model = Dense(hidden_d[i], activation=hidden_act, kernel_initializer='he_uniform')(model)

        model_out = Dense(output_d, activation=output_act, kernel_initializer='he_uniform')(model)
        return Model(model_in, model_out)

    def forward(self, _input, nettype):
        return self.models[nettype].predict(_input)
  
    def backward(self, _input, _target):
        self.models['online'].fit(_input, _target, batch_size = 1, epochs = 1,  verbose=0 )

    def transfer_weights(self):
        """ Transfer online weights to target model.
        """
        self.set_weights(self.get_weights('online'), 'target')

    def get_weights(self, nettype):
        return self.models[nettype].get_weights()
                                                          
    def set_weights(self, weights, nettype):
        return self.models[nettype].set_weights(weights)

    def save_weights(self, nettype, path, fname):
        check_and_make_dir(path)
        self.models[nettype].save_weights(path+fname+'.h5', save_format='h5', overwrite='True')
       
    def load_weights(self, path):
        path += '.h5'
        if os.path.exists(path):
            self.models['online'].load_weights(path)
        else:
            #raise not found exceptions
            assert 0, 'Failed to load weights, supplied weight file path '+str(path)+' does not exist.'

if __name__ == '__main__':
    input_d = 10
    dqn = DQN( input_d, [20, 20], 'relu', 4, 'linear', 0.0001, 0.0000001)

    x = np.random.uniform(0.0, 1.0, size=(1,input_d))

    ouptut = dqn.forward(x, 'online')
    print(output)
