import os
import numpy as np
#import tensorflow as tf
#import tensorflow.keras.backend as K

#from tensorflow.keras.models import Model
#from tensorflow.keras.layers import Input, Dense, Reshape, Flatten

class NeuralNet:
    def __init__(self, input_d, hidden_d, hidden_act, output_d, output_act, learner=False):
        self.models = {}
        self.models['online'] = self.create_model(input_d, hidden_d, hidden_act, output_d, output_act)
        if learner:
            self.models['target'] = self.create_model(input_d, hidden_d, hidden_act, output_d, output_act)

    def create_model(self, input_d, hidden_d, hidden_act, output_d, output_act):
        pass

    def forward(self, _input, nettype):
        return self.models[nettype].predict(_input)

    def backward(self):
        pass

    def transfer_weights(self):
        pass

    def get_weights(self, nettype):
        pass

    def set_weights(self, weights, nettype):
        pass

    def save_weights(self, nettype, path, fname):
        pass

    def load_weights(self, path):
        pass
