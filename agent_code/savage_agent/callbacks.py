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
            if os.path.exists(utils.check_point_save_path + '.index'):
                print('-------------load the model-----------------')
                self.model.load_weights(utils.check_point_save_path)
    else:
        # todo: load model from files
        self.model = tf.keras.models.load_model('./RNNModel')
        pass


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
    if utils.IMITATE:
        return imitate_act(self, game_state)
    if np.random.random() <= self.epsilon:
        # chose actions randomly
        action = np.random.choice(utils.ACTION_SPACE)
    else:
        state_matrix = utils.get_state_matrix(game_state)
        np.sort(self.model.predict(state_matrix.flatten() / 7))
        act_idx = np.argmax(self.model.predict(state_matrix.flatten()/7))
        action = utils.index2action(act_idx)
    return action


def imitate_setup(self):
    rule_based_setup(self)

def imitate_act(self, game_state):
    return rule_based_act(self, game_state)