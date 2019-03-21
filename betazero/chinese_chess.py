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


# setup C lib xiangqimoves
import ctypes, os


class Location(ctypes.Structure):
    _fields_ = [('row', ctypes.c_byte), ('col', ctypes.c_byte)]

    @property
    def tup(self):
        return self.row, self.col


Board = np.ctypeslib.ndpointer(
    dtype=np.int8, ndim=2, shape=board_size, flags=['C_CONTIGUOUS', 'ALIGNED'])

locations_buf = (Location * 256)()
moves_buf = (Location * 256)()

lxqm = np.ctypeslib.load_library("libxiangqimoves", os.path.dirname(__file__))
lxqm.lookup_moves.restype = ctypes.c_byte
lxqm.lookup_moves.argtypes = [ctypes.POINTER(Location), Board, Location]
lxqm.lookup_actions.restype = ctypes.c_ubyte
lxqm.lookup_actions.argtypes = [
    ctypes.POINTER(Location),
    ctypes.POINTER(Location), Board, ctypes.c_byte
]

get_player = np.sign


def is_within_bounds(location):
    return location[0] >= 0 and location[0] < board_size[0] and location[1] >= 0 and location[1] < board_size[1]


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


EMPTY = 0
PAWN = 1
CANNON = 2
ROOK = 3
KNIGHT = 4
ELEPHANT = 5
GUARD = 6
KING = 7
N_PIECES = 8

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
            self.board[location] = piece - N_PIECES
        return State(self.board, -self.player, [], 0, 0), 0, 0

    def do_action(self, action):
        location, move = action
        if (not location or not move or not is_within_bounds(location)
                or get_player(self.board[location]) != self.player
                or move not in
            (moves_buf[i].tup for i in range(
                lxqm.lookup_moves(moves_buf, self.board, Location(*location))))
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
        self.internal_array = None

    def flip(self):
        return State(self.board, -self.player, self.action_history,
                     self.n_turns, self.stalemate_count)

    def array(self):
        if (self.internal_array is not None):
            return self.internal_array
        board_array = np.zeros((8, *board_size), dtype=np.int8)
        for location, piece in np.ndenumerate(self.board):
            if piece != EMPTY:
                board_array[piece, location[0], location[1]] = get_player(
                    piece) * self.player
        for i in range(
                lxqm.lookup_actions(locations_buf, moves_buf, self.board, 0)):
            threatened_piece = self.board[moves_buf[i].tup]
            if (threatened_piece != EMPTY):
                board_array[0, moves_buf[i].row, moves_buf[
                    i].col] = rewards_lookup[threatened_piece] * -get_player(
                        threatened_piece) * self.player
        self.internal_array = board_array[np.newaxis]
        return self.internal_array

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
    n_actions = lxqm.lookup_actions(locations_buf, moves_buf, state.board,
                                    state.player)
    actions = [(locations_buf[i].tup, moves_buf[i].tup)
               for i in range(n_actions)
               if (locations_buf[i].tup, moves_buf[i].tup) != banned_move]
    return actions


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
