import numpy as np
from .utils import parse_grid_input

board_size = (10, 9)
max_value = 25
min_max = True
rotational_symetry = False
vertical_symetry = True
horizontal_symetry = True
terminal_state = True
reward_span = 10
max_stalemate_count = 30


def ValueModel():
    from keras.models import Model
    from keras import regularizers
    from keras.layers import Conv2D, Dense, Flatten, Input, ReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import Add

    input_dimensions = (8, *board_size)
    output_dimension = 2 * max_value + 1
    filter_size = (3, 3)
    n_filters = 128
    n_res_blocks = 20
    batch_norm_momentum = 0.999
    l2_reg = 1e-4

    inputs = Input(shape=input_dimensions)
    x = Conv2D(
        n_filters,
        filter_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        data_format="channels_first")(inputs)
    # residual blocks
    for i in range(n_res_blocks):
        x_in = x
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6.)(x)
        x = Conv2D(
            n_filters,
            filter_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            data_format="channels_first")(x)
        x = Add()([x, x_in])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6.)(x)
    x = Conv2D(
        1, (1, 1),
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        data_format="channels_first")(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6.)(x)
    x = Flatten()(x)
    outputs = Dense(
        output_dimension,
        activation='softmax',
        kernel_regularizer=regularizers.l2(l2_reg))(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# keras model, based on alphazero and mobilenetv2
def ValueModelMNV2():
    from keras.models import Model
    from keras import regularizers
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Flatten, Input, ReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import Add

    input_dimensions = (8, *board_size)
    output_dimension = 51
    n_filters = 128
    expansion_factor = 5
    n_res_blocks = 10
    batch_norm_momentum = 0.999
    l2_reg = 1e-5

    inputs = Input(shape=input_dimensions)
    x = Conv2D(
        n_filters, (3, 3),
        padding='same',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    for i in range(n_res_blocks):
        x_in = x
        x = Conv2D(
            n_filters * expansion_factor, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D(
            (3, 3),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(
            n_filters, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = Add()([x, x_in])
    for stride in (2, 2):
        x = Conv2D(
            n_filters * expansion_factor, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D(
            (3, 3),
            padding='same',
            strides=stride,
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(
            n_filters, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(
        n_filters * expansion_factor, (1, 1),
        padding='same',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = DepthwiseConv2D(
        (3, 3),
        padding='valid',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = Flatten()(x)
    outputs = Dense(
        output_dimension,
        kernel_regularizer=regularizers.l2(l2_reg),
        activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def get_player(piece):
    return np.sign(piece)


def is_within_bounds(location):
    return location[0] >= 0 and location[0] < board_size[0] and location[1] >= 0 and location[1] < board_size[1]


def is_across_river(player, location):
    return ((location[0] - (board_size[0] - 1) * 0.5) * player) > 0


def is_in_palace(player, location):
    row, col = location
    return ((row -
             (board_size[0] - 1) * 0.5) * player) < -2 and col > 2 and col < 6


def is_valid_move(board, player, move):
    if not is_within_bounds(move):
        return False
    if get_player(board[move]) == player:
        return False
    return True


def get_banned_move(action_history):
    if len(action_history) < 4:
        return None
    opponent_location, opponent_move = action_history[-1]
    if (opponent_move, opponent_location) == action_history[-3]:
        previous_location, previous_move = action_history[-2]
        if (action_history[-4][1] == previous_location):
            return (previous_move, previous_location)


def move_piece(board, location, move):
    board = np.copy(board)
    reward = rewards_lookup[board[move]]
    board[move] = board[location]
    board[location] = EMPTY
    return board, reward


def pawn_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = [(row + player, col)]
    if is_across_river(player, location):
        moves.append((row, col + 1))
        moves.append((row, col - 1))
    moves = [move for move in moves if is_valid_move(board, player, move)]
    return moves


def cannon_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = []
    for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        move = (row + d_row, col + d_col)
        while is_within_bounds(move) and board[move] == EMPTY:
            moves.append(move)
            move = (move[0] + d_row, move[1] + d_col)
        move = (move[0] + d_row, move[1] + d_col)
        while is_within_bounds(move):
            if board[move] != EMPTY:
                if get_player(board[move]) == -player:
                    moves.append(move)
                break
            move = (move[0] + d_row, move[1] + d_col)
    return moves


def rook_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = []
    for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        move = (row + d_row, col + d_col)
        while is_valid_move(board, player, move):
            moves.append(move)
            if board[move] != EMPTY:
                break
            move = (move[0] + d_row, move[1] + d_col)
    return moves


def knight_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = []
    for d_row, d_col in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        if (is_within_bounds((row + d_row, col + d_col))
                and board[row + d_row, col + d_col] == EMPTY):
            moves.extend([
                move for move in (
                    (row + 2 * d_row + d_col, col + 2 * d_col + d_row),
                    (row + 2 * d_row - d_col, col + 2 * d_col - d_row),
                ) if is_valid_move(board, player, move)
            ])
    return moves


def elephant_moves(board, location):
    row, col = location
    player = get_player(board[location])
    return [
        move for move in ((row + 2, col + 2), (row + 2, col - 2),
                          (row - 2, col + 2), (row - 2, col - 2))
        if is_valid_move(board, player, move)
        and not is_across_river(player, move) and board[(move[0] + row) // 2, (
            move[1] + col) // 2] == EMPTY
    ]


def guard_moves(board, location):
    row, col = location
    player = get_player(board[location])
    return [
        move for move in ((row + 1, col + 1), (row + 1, col - 1),
                          (row - 1, col + 1), (row - 1, col - 1))
        if is_valid_move(board, player, move) and is_in_palace(player, move)
    ]


def king_moves(board, location):
    row, col = location
    player = get_player(board[location])
    moves = [
        move for move in ((row, col + 1), (row, col - 1), (row + 1, col),
                          (row - 1, col))
        if is_valid_move(board, player, move) and is_in_palace(player, move)
    ]
    col += player
    while is_within_bounds((row, col)):
        if board[row, col] != EMPTY:
            if enemy(board[(row, col)]) == board[location]:
                moves.append((row, col))
            break
        col += player
    return moves


def enemy(piece):
    if piece > 0:
        piece -= 8
    elif piece < 0:
        piece += 8
    return piece


EMPTY = 0
PAWN = 1
CANNON = 2
ROOK = 3
KNIGHT = 4
ELEPHANT = 5
GUARD = 6
KING = 7

moves_lookup = (
    None,
    pawn_moves,
    cannon_moves,
    rook_moves,
    knight_moves,
    elephant_moves,
    guard_moves,
    king_moves,
)

rewards_lookup = (0, 1, 2, 4, 2, 1, 1, 25)

symbols_lookup = (
    '·',
    'P',
    'C',
    'R',
    'N',
    'E',
    'G',
    'K',
    'p',
    'c',
    'r',
    'n',
    'e',
    'g',
    'k',
)

red_spawn = ((PAWN, (3, 0)), (PAWN, (3, 2)), (PAWN, (3, 4)), (PAWN, (3, 6)),
             (PAWN, (3, 8)), (CANNON, (2, 1)), (CANNON, (2, 7)), (ROOK, (0,
                                                                         0)),
             (ROOK, (0, 8)), (KNIGHT, (0, 1)), (KNIGHT, (0, 7)), (ELEPHANT,
                                                                  (0, 2)),
             (ELEPHANT, (0, 6)), (GUARD, (0, 3)), (GUARD, (0, 5)), (KING, (0,
                                                                           4)))

black_spawn = (
    (PAWN, (6, 0)),
    (PAWN, (6, 2)),
    (PAWN, (6, 4)),
    (PAWN, (6, 6)),
    (PAWN, (6, 8)),
    (CANNON, (7, 1)),
    (CANNON, (7, 7)),
    (ROOK, (9, 0)),
    (ROOK, (9, 8)),
    (KNIGHT, (9, 1)),
    (KNIGHT, (9, 7)),
    (ELEPHANT, (9, 2)),
    (ELEPHANT, (9, 6)),
    (GUARD, (9, 3)),
    (GUARD, (9, 5)),
    (KING, (9, 4)),
)


class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player = 1
        self.board = np.zeros(board_size, dtype=np.int8)
        self.action_history = []
        self.stalemate_count = 0
        for piece, location in red_spawn:
            self.board[location] = piece
        for piece, location in black_spawn:
            self.board[location] = enemy(piece)
        return State(self.board, -self.player, [], 0, 0), 0, 0

    def do_action(self, action):
        location, move = action
        if (not is_within_bounds(location)
                or get_player(self.board[location]) != self.player
                or move not in moves_lookup[self.board[location]](self.board,
                                                                  location)
                or (self.stalemate_count > 2
                    and action == get_banned_move(self.action_history))):
            return State(self.board, self.player, self.action_history[-4:],
                         len(self.action_history),
                         self.stalemate_count), -max_value, 1
        board, reward = move_piece(self.board, location, move)
        self.action_history.append(action)
        self.player *= -1
        if reward == 0:
            self.stalemate_count += 1
        else:
            self.stalemate_count = 0
        if reward >= max_value or self.stalemate_count >= max_stalemate_count:
            reset = len(self.action_history)
            self.reset()
        else:
            self.board = board
            reset = 0
        return State(board, -self.player, self.action_history[-4:],
                     len(self.action_history),
                     self.stalemate_count), reward, reset


#------------ The below is required game interface for betazero


class State:
    def __init__(self,
                 board,
                 player,
                 action_history=[],
                 n_turns=0,
                 stalemate_count=0):
        self.board = board
        self.player = player
        self.action_history = action_history
        self.n_turns = n_turns
        self.stalemate_count = stalemate_count

    def flip(self):
        return State(self.board, -self.player, self.action_history,
                     self.n_turns, self.stalemate_count)

    def array(self):
        board_array = np.zeros((8, *board_size), dtype=np.int8)
        for location, piece in np.ndenumerate(self.board):
            if piece != EMPTY:
                piece_player = get_player(piece)
                board_array[piece, location[0], location[
                    1]] = piece_player * self.player
                for move in moves_lookup[piece](self.board, location):
                    if self.board[move] != EMPTY:
                        board_array[0, move[0], move[1]] = rewards_lookup[
                            self.board[move]] * piece_player * self.player
        return board_array[np.newaxis]

    def key(self):
        return self.board.tobytes()

    def __str__(self):
        ascii_board = [' '.join([''] + [str(i) for i in range(board_size[1])])]
        for i, row in enumerate(self.board):
            ascii_board.append(
                ' '.join([str(i)] + [symbols_lookup[piece] for piece in row]))
        return '\n'.join(ascii_board)


def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    banned_move = get_banned_move(
        state.action_history) if state.stalemate_count > 2 else None
    return [(location, move)
            for location, piece in np.ndenumerate(state.board)
            if get_player(piece) == state.player
            for move in moves_lookup[piece](state.board, location)
            if (location, move) != banned_move]


def predict_action(state, action):
    board, reward = move_piece(state.board, *action)
    stalemate_count = state.stalemate_count + 1 if reward == 0 else 0
    reset = state.n_turns + 1 if (
        reward >= max_value or stalemate_count >= max_stalemate_count) else 0
    action_history = state.action_history[-2:].append(action)
    return State(board, state.player, action_history, state.n_turns + 1,
                 stalemate_count), reward, reset


def get_human_action():
    return parse_grid_input(board_size), parse_grid_input(board_size)
