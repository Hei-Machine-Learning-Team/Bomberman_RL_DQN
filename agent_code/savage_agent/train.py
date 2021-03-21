from agent_code.savage_agent import utils
import events as e
from collections import deque
import time
import random
import numpy as np
from settings import COLS, ROWS


def setup_training(self):
    self.target_model = utils.create_model()
    self.target_model.set_weights(self.model.get_weights())
    self.transitions = deque(maxlen=utils.TRANSITION_MAX_LEN)
    self.tensorboard = utils.ModifiedTensorBoard(log_dir=f"logs/{utils.MODEL_NAME}-{int(time.time())}")
    self.ep_rewards = []
    self.round_reward = 0
    self.round_num = 0
    self.epsilon = 1


def game_events_occurred(self, old_game_state, self_action, new_game_state, events):
    update_transitions(self, old_game_state, self_action, new_game_state, events, done=False)
    train(self, is_final=False)


def end_of_round(self, last_game_state, last_action, events):
    update_transitions(self, last_game_state, last_action, None, events, done=True)
    self.ep_rewards.append(self.round_reward)
    self.round_reward = 0
    if self.round_num % utils.AGGREGATE_STATS_EVERY == 0 and self.round_num != 0:
        avg_reward = sum(self.ep_rewards[-utils.AGGREGATE_STATS_EVERY:])/len(self.ep_rewards[-utils.AGGREGATE_STATS_EVERY:])
        min_reward = min(self.ep_rewards[-utils.AGGREGATE_STATS_EVERY:])
        max_reward = max(self.ep_rewards[-utils.AGGREGATE_STATS_EVERY:])
        self.tensorboard.update_stats(reward_avg=avg_reward, reward_min=min_reward, max_reward=max_reward, epsilon=self.epsilon)
    train(self, is_final=True)
    # if e.KILLED_SELF in events or e.GOT_KILLED in eveNnts:
    #     print("*************************************************")
    #     print(events)
    print("round:", last_game_state['round'], "step:", last_game_state['step'], "score:", last_game_state['self'][1], "e:", self.epsilon)


def update_transitions(self, old_game_state, self_action, new_game_state, events, done):
    if old_game_state is None or self_action is None:
        return
    reward = utils.reward_from_events(events)
    self.round_reward += reward
    # print("reward:", reward)
    # if reward == -1:
    #     print(events)
    old_state_matrix = utils.get_state_matrix(old_game_state)
    # detect if the agent has performed invalid action
    if utils.detect_invalid_action(old_state_matrix, old_game_state['self'][3], old_game_state['self'][2], self_action):
        events.append(utils.INVALID_ACTION)
    # if this transition is from the end of a round
    if done or new_game_state is None:
        new_state_matrix = np.zeros((COLS, ROWS))
    else:
        new_state_matrix = utils.get_state_matrix(new_game_state)
    self.transitions.append((old_state_matrix, self_action, new_state_matrix, reward, done))


def train(self, is_final):
    if len(self.transitions) < utils.MIN_TRAINING_SIZE:
        return
    training_batch = random.sample(self.transitions, utils.TRAINING_BATCH_SIZE)
    old_state_matrices = np.array([transition[0].flatten() for transition in training_batch])
    old_qs_list = self.model.predict(old_state_matrices/7)

    new_state_matrices = np.array([transition[2].flatten() for transition in training_batch])
    new_qs_list = self.target_model.predict(new_state_matrices / 7)

    x_train = []
    y_train = []

    for index, (old_state_matrix, action, new_state_matrix, reward, done) in enumerate(training_batch):
        if done:   # if this transition is from end_of_round
            new_q = reward
        else:
            max_new_q = np.max(new_qs_list[index])
            new_q = reward + utils.DISCOUNT * max_new_q

        old_qs = old_qs_list[index]

        act_idx = utils.action2index[action]
        old_qs[act_idx] = new_q

        x_train.append(old_state_matrix.flatten())
        y_train.append(old_qs)

    callbacks = []
    if is_final:
        callbacks.append(self.tensorboard)
    if self.round_num % utils.CHECKPOINT_ROUNDS_NUM == 0:
        callbacks.append(utils.cp_callbacks)
        if self.round_num != 0:
            self.model.save(f'./models/RNN-{int(time.time())}')

    # print("train**********")
    # print(np.array(x_train)/7, np.array(y_train))
    self.model.fit(np.array(x_train)/7, np.array(y_train), batch_size=utils.TRAINING_BATCH_SIZE,
                   verbose=0, shuffle=False, callbacks=callbacks)

    if is_final:
        self.round_num += 1

        if self.round_num % utils.UPDATE_ROUNDS_NUM == 0:
            self.target_model.set_weights(self.model.get_weights())

        if self.epsilon > utils.MIN_EPSILON:
            self.epsilon *= utils.EPSILON_DECAY
            self.epsilon = max(utils.MIN_EPSILON, self.epsilon)

