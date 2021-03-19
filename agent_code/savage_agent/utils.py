
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


