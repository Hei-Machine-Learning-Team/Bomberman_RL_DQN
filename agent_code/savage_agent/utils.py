import tensorflow as tf
import os
import events
from settings import ROWS, COLS, BOMB_TIMER, BOMB_POWER, EXPLOSION_TIMER
import time
import numpy as np

ACTION_NUM = 6
TRANSITION_MAX_LEN = 1000
MIN_TRAINING_SIZE = 800
TRAINING_BATCH_SIZE = 64
UPDATE_ROUNDS_NUM = 20
CHECKPOINT_ROUNDS_NUM = 130
AGGREGATE_STATS_EVERY = 10
DISCOUNT = 0.99
EPSILON_DECAY = 0.99976
MIN_EPSILON = 0.0001


MODEL_NAME = "savage-RNN"

IMITATE = False
CONTINUE_CKPT = False
check_point_save_path = "./checkpoints/rnn.ckpt"
cp_callbacks = tf.keras.callbacks.ModelCheckpoint(filepath=f"./checkpoints/{MODEL_NAME}-{int(time.time())}/rnnckpt",
                                                  save_weights_only=True)

ACTION_SPACE = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'BOMB', 'WAIT']
action2index = {
    'UP': 0,
    'DOWN': 1,
    'LEFT': 2,
    'RIGHT': 3,
    'BOMB': 4,
    'WAIT': 5
}
index2action = {
    0: 'UP',
    1: 'DOWN',
    2: 'LEFT',
    3: 'RIGHT',
    4: 'BOMB',
    5: 'WAIT'
}

INVALID_ACTION = "INVALID_ACTION"

game_rewards_table = {
        events.COIN_COLLECTED: 200,
        events.KILLED_OPPONENT: 500,
        events.GOT_KILLED: -400,
        events.KILLED_SELF: -200,
        events.CRATE_DESTROYED: 30,
        # events.SURVIVED_ROUND: 1,
        events.OPPONENT_ELIMINATED: 5,
        # events.MOVED_UP: -1,
        # events.MOVED_DOWN: -1,
        # events.MOVED_LEFT: -1,
        # events.MOVED_RIGHT: -1,
        # events.WAITED: -1,
        INVALID_ACTION: -3
    }


def reward_from_events(event_list):
    reward = 0
    for event in event_list:
        if event in game_rewards_table:
            reward += game_rewards_table[event]
    return reward


def get_possible_actions(state_matrix, player_position, bomb_left):
    possible_actions = ['WAIT']
    if bomb_left:
        possible_actions.append("BOMB")
    x, y = player_position
    if state_matrix[(x, y-1)] == 4:
        possible_actions.append("UP")
    if state_matrix[(x, y+1)] == 4:
        possible_actions.append("DOWN")
    if state_matrix[(x-1, y)] == 4:
        possible_actions.append("LEFT")
    if state_matrix[(x+1, y)] == 4:
        possible_actions.append("RIGHT")
    return possible_actions


# def detect_invalid_action(state_matrix, player_position, bomb_left, action):
#     possible_actions = get_possible_actions(state_matrix, player_position, bomb_left)
#     if action not in possible_actions:
#         return True

def detect_invalid_action(old_state, new_state, action):
    if new_state is None:
        return False
    if action == "BOMB":
        bomb_left = old_state['self'][2]
        return not bomb_left  # bomb_left true -> valid;  bomb_left false -> invalid
    if action in ["UP", "DOWN", "LEFT", "RIGHT"]:
        old_position = old_state['self'][3]
        new_position = new_state['self'][3]
        return new_position == old_position  # if the agent didn't move, it's a invalid action


# def get_state_matrix(state):
#     """
#     Represent the state using a single matrix.
#     In this matrix,
#     0 -> player,  1 -> enemies,  2 -> crates,  3 -> walls
#     4 -> tiles,   5 -> bombs,    6 -> coins,   7 -> explosion
#     """
#     if state is None:
#         return None
#     player_position = state['self'][3]
#     enemy_positions = [player_state[3] for player_state in state['others']]
#     field = np.copy(state['field'])
#     bomb_positions = [bomb_state[0] for bomb_state in state['bombs']]
#     coin_positions = [coin_pos for coin_pos in state['coins']]
#     explosion = state['explosion_map']
#     for i in range(field.shape[0]):
#         for j in range(field.shape[1]):
#             if field[i][j] == -1:  # walls
#                 field[i][j] = 3
#                 continue
#             elif field[i][j] == 0:  # tiles
#                 field[i][j] = 4
#             elif field[i][j] == 1:  # crates
#                 field[i][j] = 2
#             if explosion[i][j] > 0:  # explosion
#                 field[i][j] = 7
#     field[player_position] = 0
#     for pos in enemy_positions:
#         field[pos] = 1
#     for pos in bomb_positions:
#         field[pos] = 5
#     for pos in coin_positions:
#         field[pos] = 6
#     return field


def get_blast_coords(bomb_pos, power, arena):
    x, y = bomb_pos
    blast_coords = [(x, y)]
    for i in range(1, power + 1):
        if arena[x + i, y] == -1:
            break
        blast_coords.append((x + i, y))
    for i in range(1, power + 1):
        if arena[x - i, y] == -1:
            break
        blast_coords.append((x - i, y))
    for i in range(1, power + 1):
        if arena[x, y + i] == -1:
            break
        blast_coords.append((x, y + i))
    for i in range(1, power + 1):
        if arena[x, y - i] == -1:
            break
        blast_coords.append((x, y - i))
    return blast_coords


def get_state_matrix(state):
    if state is None:
        return None
    maps = []
    # field matrix
    field = np.copy(state['field']).astype(np.float32)
    maps.append(field)
    # player map
    player_position = state['self'][3]
    player_map = np.zeros((17, 17), dtype='float32')
    player_map[player_position] = 1
    # enemy
    enemy_positions = [player_state[3] for player_state in state['others']]
    enemy_map = np.zeros((17, 17), dtype='float32')
    for pos in enemy_positions:
        enemy_map[pos] = 1
    # coin
    coin_positions = [coin_pos for coin_pos in state['coins']]
    coin_map = np.zeros((17, 17), dtype='float32')
    for pos in coin_positions:
        coin_map[pos] = 1
    # danger
    danger_map = np.copy(np.array(state['explosion_map'] > 0, dtype='float32'))
    # sort based on timer, bigger timer means less dangerous
    bomb_states = sorted([bomb_state for bomb_state in state['bombs']], key=lambda bomb: bomb[1], reverse=True)
    for bomb_pos, timer in bomb_states:
        danger = (BOMB_TIMER - timer) / BOMB_TIMER
        blast_coords = get_blast_coords(bomb_pos, BOMB_POWER, field)
        for pos in blast_coords:
            if danger_map[pos] != 1:
                danger_map[pos] = danger
    return np.array([field, player_map, enemy_map, coin_map, danger_map]).flatten().astype(np.float32)



class ModifiedTensorBoard(tf.keras.callbacks.TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()


# def create_model():
#     model = tf.keras.models.Sequential([
#         tf.keras.Input(shape=ROWS*COLS),
#         tf.keras.layers.Dense(128, activation='relu'),
#         tf.keras.layers.Reshape((1, 128)),
#         tf.keras.layers.LSTM(80),
#         tf.keras.layers.Dropout(0.2),
#         tf.keras.layers.Dense(ACTION_NUM, activation='linear')
#     ])
#     model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
#     return model

def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.Input(shape=ROWS*COLS*5),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    return model

