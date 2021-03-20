import os
import pickle
import random

import numpy as np
import utils
from collections import deque


ACTIONS = ['UP', 'RIGHT', 'DOWN', 'LEFT', 'WAIT', 'BOMB']


def setup(self):
    pass


def act(self, game_state: dict) -> str:
    state_matrix = utils.get_state_matrix(game_state)
    act_idx = np.argmax(self.model.predict(state_matrix))
    return utils.index2action[act_idx]
