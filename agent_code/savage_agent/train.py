from agent_code.savage_agent import utils
import events as e
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
    self.epsilon = 1


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    update_transitions(self, old_game_state, self_action, new_game_state, events, done=False)
    train(self, is_final=False)


def end_of_round(self, last_game_state, last_action, events):
    update_transitions(self, last_game_state, last_action, None, events, done=True)
    train(self, is_final=True)
    # if e.KILLED_SELF in events or e.GOT_KILLED in events:
    #     print("*************************************************")
    #     print(events)
    print("round:", last_game_state['round'], "step:", last_game_state['step'], "score:", last_game_state['self'][1], "e:", self.epsilon)


def update_transitions(self, old_game_state, self_action, new_game_state, events, done):
    if old_game_state is None or self_action is None:
        return
    reward = utils.reward_from_events(events)
    old_state_matrix = utils.get_state_matrix(old_game_state)
    new_state_matrix = utils.get_state_matrix(new_game_state)
    self.transitions.append((old_state_matrix, self_action, new_state_matrix, reward, done))


def train(self, is_final):
    if len(self.transitions) < utils.MIN_TRAINING_SIZE:
        return
    training_batch = random.sample(self.transitions, utils.TRAINING_BATCH_SIZE)
    old_state_matrices = np.array([transition[0].flatten() for transition in training_batch])
    old_qs_list = self.model.predict(old_state_matrices/7)

    x_train = []
    y_train = []

    for index, (old_state_matrix, action, new_state_matrix, reward, done) in enumerate(training_batch):
        if done:   # if this transition is from end_of_round
            new_q = reward
        else:
            max_new_q = np.max(self.target_model.predict(np.asarray([new_state_matrix.flatten()])))
            new_q = reward + utils.DISCOUNT * max_new_q

        old_qs = old_qs_list[index]

        act_idx = utils.action2index[action]
        old_qs[act_idx] = new_q

        x_train.append(old_state_matrix.flatten())
        y_train.append(old_qs)

    # if is_final:
    #     self.model.fit(np.array(x_train), np.array(y_train), batch_size=utils.TRAINING_BATCH_SIZE,
    #                    verbose=0, shuffle=False, callbacks=[self.tensorboard])
    # else:
    #     self.model.fit(np.array(x_train), np.array(y_train), batch_size=utils.TRAINING_BATCH_SIZE,
    #                    verbose=0, shuffle=False)

    self.model.fit(np.array(x_train)/7, np.array(y_train), batch_size=utils.TRAINING_BATCH_SIZE,
                   verbose=0, shuffle=False)

    if is_final:
        self.round_num += 1

        if self.round_num > utils.UPDATE_ROUNDS_NUM:
            self.target_model.set_weights(self.model.get_weights())
            self.round_num = 0

        if self.epsilon > utils.MIN_EPSILON:
            self.epsilon *= utils.EPSILON_DECAY
            self.epsilon = max(utils.MIN_EPSILON, self.epsilon)

