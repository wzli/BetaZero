import numpy as np
from keras.models import Model
from keras import regularizers
from keras.layers import Conv2D, DepthwiseConv2D, Dense, Flatten, Input, ReLU
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Add

# keras model
board_size = (10, 9)
input_dimensions = (*board_size, 7)
output_dimension = 51
max_value = 25
min_max = True
rotational_symetry = False
vertical_symetry = True
horizontal_symetry = True
# end states of the game may corresponds to terminal nodes in the state tree
terminal_state = True
reward_span = 6

max_stalemate_count = 30

# keras model, based on alphazero and mobilenetv2
def ValueModel():
    n_filters = 256
    expansion_factor = 5
    n_res_blocks = 20
    batch_norm_momentum = 0.999
    l2_reg = 1e-4

    inputs = Input(shape=input_dimensions)
    x = Conv2D(n_filters, (3, 3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    for i in range(n_res_blocks):
        x_in = x
        x = Conv2D(n_filters * expansion_factor, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D((3,3), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = Add()([x, x_in])
    for stride in (2, 2):
        x = Conv2D(n_filters * expansion_factor, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D((3,3), padding='same', strides=stride, use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(n_filters, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(n_filters * expansion_factor, (1, 1), padding='same', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = DepthwiseConv2D((3,3), padding='valid', use_bias=False, kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = Flatten()(x)
    outputs = Dense(output_dimension, kernel_regularizer=regularizers.l2(l2_reg), activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def is_within_bounds(move):
    return move[0] >= 0 and move[0] < board_size[0] and move[1] >= 0 and move[1] < board_size[1]

class Piece:
    def __init__(self, board, pieces, player, spawn_location):
        self.board = board
        self.pieces = pieces
        self.player = player
        self.location = spawn_location
        board[spawn_location] = self
        pieces[player].add(self)

    def is_across_river(self, move=None):
        row = move[0] if move else self.location[0]
        return ((row - 4.5) * self.player) > 0

    def is_in_palace(self, move=None):
        row, col = move if move else self.location
        return ((row - 4.5) * self.player) < -2 and col > 2 and col < 6

    def is_valid(self, move):
        if not is_within_bounds(move):
            return False
        if self.board[move] and self.board[move].player == self.player:
            return False
        return True

    def move(self, move, mutable = True):
        if mutable:
            board = self.board
            if board[move]:
                self.pieces[-self.player].remove(board[move])
        else:
            board = np.copy(self.board)
        reward = board[move].reward if board[move] else 0
        board[move] = self
        board[self.location] = None
        if mutable:
            self.location = move
            board = np.copy(board)
        return board, reward


class Pawn(Piece):
    reward = 1

    def get_moves(self):
        row, col = self.location
        moves = [(row + self.player, col)]
        if self.is_across_river():
            moves.append((row, col + 1))
            moves.append((row, col - 1))
        moves = [move for move in moves if self.is_valid(move)]
        return moves

    def __str__(self):
        return 'P' if self.player > 0 else 'p'


class Guard(Piece):
    reward = 1

    def get_moves(self):
        row, col = self.location
        return [
            move for move in ((row + 1, col + 1), (row + 1, col - 1),
                              (row - 1, col + 1), (row - 1, col - 1))
            if self.is_valid(move) and self.is_in_palace(move)
        ]

    def __str__(self):
        return 'G' if self.player > 0 else 'g'


class King(Piece):
    reward = 25

    def get_moves(self):
        row, col = self.location
        moves = [
            move for move in ((row, col + 1), (row, col - 1),
                              (row + 1, col), (row - 1, col))
            if self.is_valid(move) and self.is_in_palace(move)
        ]
        col += self.player
        while is_within_bounds((row,col)):
            if self.board[row,col]:
                if isinstance(self.board[(row, col)], King):
                    moves.append((row,col))
                break
            col += self.player
        return moves

    def __str__(self):
        return 'K' if self.player > 0 else 'k'


class Elephant(Piece):
    reward = 1

    def get_moves(self):
        row, col = self.location
        return [
            move for move in ((row + 2, col + 2), (row + 2, col - 2),
                              (row - 2, col + 2), (row - 2, col - 2))
            if self.is_valid(move) and not self.is_across_river(move)
        ]

    def __str__(self):
        return 'E' if self.player > 0 else 'e'


class Knight(Piece):
    reward = 2

    def get_moves(self):
        row, col = self.location
        moves = []
        for d_row, d_col in ((0 ,1), (0, -1), (1, 0), (-1, 0)):
            if is_within_bounds((row + d_row, col + d_col)) and not self.board[row + d_row, col + d_col]:
                moves.extend([move for move in (
                    (row + 2 * d_row + d_col, col + 2 * d_col + d_row),
                    (row + 2 * d_row - d_col, col + 2 * d_col - d_row),
                    ) if self.is_valid(move)])
        return moves

    def __str__(self):
        return 'N' if self.player > 0 else 'n'


class Rook(Piece):
    reward = 4

    def get_moves(self):
        row, col = self.location
        moves = []
        for d_row, d_col in ((0 ,1), (0, -1), (1, 0), (-1, 0)):
            move = (row + d_row, col + d_col)
            while self.is_valid(move):
                moves.append(move)
                if self.board[move]:
                    break
                move = (move[0] + d_row, move[1] + d_col)
        return moves

    def __str__(self):
        return 'R' if self.player > 0 else 'r'


class Cannon(Piece):
    reward = 2

    def get_moves(self):
        row, col = self.location
        moves = []
        for d_row, d_col in ((0 ,1), (0, -1), (1, 0), (-1, 0)):
            move = (row + d_row, col + d_col)
            while is_within_bounds(move) and not self.board[move]:
                moves.append(move)
                move = (move[0] + d_row, move[1] + d_col)
            move = (move[0] + d_row, move[1] + d_col)
            while is_within_bounds(move):
                if self.board[move]:
                    if self.board[move].player == -self.player:
                        moves.append(move)
                    break
                move = (move[0] + d_row, move[1] + d_col)
        return moves

    def __str__(self):
        return 'C' if self.player > 0 else 'c'


default_spawn = (None, (
    (Pawn, (3, 0)),
    (Pawn, (3, 2)),
    (Pawn, (3, 4)),
    (Pawn, (3, 6)),
    (Pawn, (3, 8)),
    (Rook, (0, 0)),
    (Rook, (0, 8)),
    (Cannon, (2, 1)),
    (Cannon, (2, 7)),
    (Knight, (0, 1)),
    (Knight, (0, 7)),
    (Elephant, (0, 2)),
    (Elephant, (0, 6)),
    (Guard, (0, 3)),
    (Guard, (0, 5)),
    (King, (0, 4)),
), (
    (Pawn, (6, 0)),
    (Pawn, (6, 2)),
    (Pawn, (6, 4)),
    (Pawn, (6, 6)),
    (Pawn, (6, 8)),
    (Rook, (9, 0)),
    (Rook, (9, 8)),
    (Cannon, (7, 1)),
    (Cannon, (7, 7)),
    (Knight, (9, 1)),
    (Knight, (9, 7)),
    (Elephant, (9, 2)),
    (Elephant, (9, 6)),
    (Guard, (9, 3)),
    (Guard, (9, 5)),
    (King, (9, 4)),
))


class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.player = 1
        self.board = np.empty(board_size, dtype=object)
        self.pieces = (None, set(), set())
        self.move_history = []
        self.stalemate_count = 0
        for player in (1, -1):
            for piece, location in default_spawn[player]:
                piece(self.board, self.pieces, player, location)
        return State(self, np.copy(self.board), -self.player), 0, 0

    def get_banned_move(self):
        if len(self.move_history) < 3:
            return None
        opponent_location, opponent_move = self.move_history[-1]
        if (opponent_move, opponent_location) == self.move_history[-3]:
            move, location = self.move_history[-2]
            return (location, move)

    def do_action(self, action):
        if any((not is_within_bounds(index) for index in action)):
            return State(self, np.copy(self.board), self.player), -max_value, 1
        location, move = action
        piece = self.board[location]
        if (not piece
                or piece.player != self.player
                or move not in piece.get_moves()
                or action == self.get_banned_move
                ):
            return State(self, np.copy(self.board), self.player), -max_value, 1
        board, reward = self.board[location].move(move)
        self.move_history.append(action)
        self.player *= -1
        if reward == 0:
            self.stalemate_count += 1
        else:
            self.stalemate_count = 0
        if reward >= max_value or self.stalemate_count >= max_stalemate_count:
            reset = len(self.move_history)
            self.reset()
        else:
            reset = 0
        return State(self, board, piece.player), reward, reset


#------------ The below is required game interface for betazero

class State:
    def __init__(self, session, board, player):
        self.session = session
        self.board = board
        self.player = player

    def flip(self):
        return State(self.session, self.board, -self.player)

    def array(self):
        return np.array(
            tuple(
                np.vectorize(
                    lambda x: x.player * self.player if isinstance(x, piece) else 0
                )(self.board) for piece in (Pawn, Rook, Cannon, Knight, Elephant, Guard, King)))[np.newaxis]

    def key(self):
        return self.board.tobytes()


    def __str__(self):
        ascii_board = [' '.join([''] + [str(i) for i in range(board_size[1])])]
        for i, row in enumerate(self.board):
            ascii_board.append(' '.join([str(i)] + [str(piece) if piece else '·'
                                    for piece in row]))
        return '\n'.join(ascii_board)


def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    banned_move = state.session.get_banned_move()
    actions = [(piece.location, move) for piece in state.session.pieces[state.player]
            for move in piece.get_moves() if (piece.location, move) != banned_move]
    return actions


def predict_action(state, action):
    location, move = action
    board, reward = state.session.board[location].move(move, mutable=False)
    if reward >= max_value or state.session.stalemate_count + 1 >= max_stalemate_count:
        reset = len(state.session.move_history) + 1
    else:
        reset = 0
    return State(state.session, board, state.player), reward, reset
