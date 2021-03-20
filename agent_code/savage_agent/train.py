import utils
from collections import deque
import time
import random
import numpy as np


def setup_training(self):
    self.model = utils.create_model()  # 非training 模式下需要变通
    self.target_model = utils.create_model()
    self.transitions = deque(maxlen=utils.TRANSITION_MAX_LEN)
    self.tensorboard = utils.ModifiedTensorBoard(log_dir=f"logs/{utils.MODEL_NAME}-{int(time.time())}")
    self.round_num = 0


def game_events_occurred(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    self.update_transitions(old_game_state, self_action, new_game_state, events)
    self.train(is_final=False)


def end_of_round(self, last_game_state: dict, last_action: str, events: list[str]):
    self.train(is_final=False)


def update_transitions(self, old_game_state: dict, self_action: str, new_game_state: dict, events: list[str]):
    reward = utils.reward_from_events(events)
    old_state_matrix = utils.get_state_matrix(old_game_state)
    new_state_matrix = utils.get_state_matrix(new_game_state)
    self.transitions.append((old_state_matrix, self_action, new_state_matrix, reward))


def train(self, is_final):
    if len(self.transitions) < utils.MIN_TRAINING_SIZE:
        return
    training_batch = random.sample(self.transitions, utils.TRAINING_BATCH_SIZE)
    old_state_matrices = np.array([transition[0] for transition in training_batch])
    old_qs_list = self.model.predict(old_state_matrices)

    new_state_matrices = np.array([transition[3] for transition in training_batch])
    new_qs_list = self.target_model.predict(new_state_matrices)

    x_train = []
    y_train = []

    for index, (old_state_matrix, action, new_state_matrix, reward, done) in enumerate(training_batch):
        max_new_q = np.max(new_qs_list[index])
        new_q = reward + utils.DISCOUNT * max_new_q

        old_qs = old_qs_list[index]
        act_idx = utils.action2index[action]
        old_qs[act_idx] = new_q

        x_train.append(old_state_matrix)
        y_train.append(old_qs)

    if is_final:
        self.model.fit(np.array(x_train), np.array(y_train), batch_size=utils.TRAINING_BATCH_SIZE,
                       verbose=0, shuffle=False, callbacks=[self.tensorboard])
    else:
        self.model.fit(np.array(x_train), np.array(y_train), batch_size=utils.TRAINING_BATCH_SIZE,
                       verbose=0, shuffle=False)

    if is_final:
        self.round_num += 1

        if self.round_num > utils.UPDATE_ROUNDS_NUM:
            self.target_model.set_weights(self.model.get_weights())
            self.round_num = 0

