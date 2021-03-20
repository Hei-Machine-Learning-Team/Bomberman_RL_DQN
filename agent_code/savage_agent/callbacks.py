import os
import pickle
import random

import numpy as np
from agent_code.savage_agent import utils
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    pass


def act(self, game_state: dict) -> str:
    # print("act called")
    # print("epsilon:", self.epsilon)
    if np.random.random() <= self.epsilon:
        act_idx = np.random.randint(0, len(utils.ACTION_SPACE))
    else:
        state_matrix = utils.get_state_matrix(game_state)
        act_idx = np.argmax(self.model.predict(state_matrix))
    action = utils.index2action[act_idx]
    # print("action:", action)
    return action
