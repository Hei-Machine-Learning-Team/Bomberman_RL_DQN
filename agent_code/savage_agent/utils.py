import tensorflow as tf
import os
import events

ACTION_NUM = 6
TRANSITION_MAX_LEN = 5_000
MIN_TRAINING_SIZE = 500
TRAINING_BATCH_SIZE = 32
MODEL_NAME = "savage-RNN"

game_rewards_table = {
        events.COIN_COLLECTED: 10,
        events.KILLED_OPPONENT: 50,
        events.GOT_KILLED: -100,
        events.KILLED_SELF: -100,
        events.CRATE_DESTROYED: 1,
        events.SURVIVED_ROUND: 10
    }


def reward_from_events(event_list):
    reward = 0
    for event in event_list:
        if event in game_rewards_table:
            reward += game_rewards_table[event]
    return reward


def get_state_matrix(state):
    """
    Represent the state using a single matrix.
    In this matrix,
    0 -> player,  1 -> enemies,  2 -> crates,  3 -> walls
    4 -> tiles,   5 -> bombs,    6 -> coins,   7 -> explosion
    """
    player_position = state['self'][3]
    enemy_positions = [player_state[3] for player_state in state['others']]
    field = state['field']
    bomb_positions = [bomb_state[0] for bomb_state in state['bombs']]
    coin_positions = [coin_pos for coin_pos in state['coins']]
    explosion = state['explosion_map']

    field[player_position] = 0
    for pos in enemy_positions:
        field[pos] = 1
    for pos in bomb_positions:
        field[pos] = 5
    for pos in coin_positions:
        field[pos] = 6
    for i in field.shape[0]:
        for j in field.shape[1]:
            if field[i][j] == -1:  # walls
                field[i][j] = 3
                continue
            elif field[i][j] == 0:  # tiles
                field[i][j] = 4
            elif field[i][j] == 1:  # crates
                field[i][j] = 2
            if explosion[i][j] > 0:  # explosion
                field[i][j] = 7
    return field


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


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.SimpleRNN(120, return_sequences=True),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.SimpleRNN(120, return_sequences=False),
        tf.keras.layers.Dropout(0.2),

        tf.keras.layers.Dense(64),
        tf.keras.layers.Dense(ACTION_NUM, activation='linear')
    ])
    model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=0.001), metrics=['accuracy'])
    return model


