import os
import pickle
import random

import numpy as np
from agent_code.savage_agent import utils
from agent_code.rule_based_agent.callbacks import act as rule_based_act
from agent_code.rule_based_agent.callbacks import setup as rule_based_setup
import tensorflow as tf
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    if utils.IMITATE:
        imitate_setup(self)
    if self.train:  # on the training mode
        self.model = utils.create_model()
        if utils.CONTINUE_CKPT:  # load model from checkpoint
            if os.path.exists("./checkpoints/savage-RNN-1616611589/rnnckpt" + '.index'):
                print('-------------load the model-----------------')
                self.model.load_weights("./checkpoints/savage-RNN-1616611589/rnnckpt")
                print("*******************model loaded***************************")
    else:
        self.model = tf.keras.models.load_model('./RNNModel')
        print(self.model.summary())
        print("load model on non-training mode")

# def act(self, game_state: dict) -> str:
#     # print("act called")
#     # print("epsilon:", self.epsilon)
#     state_matrix = utils.get_state_matrix(game_state)
#     possible_actions = utils.get_possible_actions(state_matrix, game_state['self'][3], game_state['self'][2])
#     if np.random.random() <= self.epsilon:
#         # chose actions randomly
#         action = np.random.choice(possible_actions)
#     else:
#         for idx in np.sort(self.model.predict(state_matrix.flatten()/7))[::-1]:
#             if utils.index2action[idx] in possible_actions:
#                 action = utils.index2action[idx]
#                 break
#     return action


def act(self, game_state: dict) -> str:
    # print("act called")
    # print(utils.get_state_matrix(game_state).reshape((5, 17, 17))[1])
    # print(utils.get_state_matrix(game_state).reshape((5, 17, 17))[4])
    if utils.IMITATE:
        return imitate_act(self, game_state)
    if self.train:
        if np.random.random() <= self.epsilon:
            # chose actions randomly
            # return np.random.choice(utils.ACTION_SPACE)
            return np.random.choice(['UP', 'DOWN', 'LEFT', 'RIGHT', 'WAIT', 'BOMB'], p=[.245, .245, .245, .245, .01, .01])
    # chose actions based on q values .09, .03
    state_matrix = utils.get_state_matrix(game_state)
    # print(game_state)
    # print(self.model.predict(np.array([state_matrix.flatten()/7])))
    act_idx = np.argmax(self.model.predict(np.array([state_matrix])))
    # print(act_idx)
    action = utils.ACTION_SPACE[act_idx]
    # print(action)
    return action


def imitate_setup(self):
    rule_based_setup(self)


def imitate_act(self, game_state):
    return rule_based_act(self, game_state)
