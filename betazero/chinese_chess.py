import numpy as np

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

    def move(self, move, check_invalid = True):
        reward = 0
        if check_invalid and move not in self.get_moves():
            return self.board, -max_value
        if self.board[move]:
            reward = self.board[move].reward
            self.peices[-self.player].remove(self.board[move])
        self.board[move] = self
        self.board[self.location] = None
        self.location = move
        return self.board, reward

    def test(self, move, check_invalid = False):
        board = np.copy(self.board)
        if check_invalid and move not in self.get_moves():
            return board, -max_value
        reward = board[move].reward if board[move] else 0
        board[move] = self
        board[self.location] = None
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
        self.n_turns = 0
        self.player = 1
        self.board = np.empty(board_size, dtype=object)
        self.pieces = (None, set(), set())
        for player in (1, -1):
            for piece, location in default_spawn[player]:
                piece(self.board, self.pieces, player, location)
        return State(self, np.copy(self.board), self.player), 0, 0

    def do_action(self, check_invalid = True):
        if not self.board[location] or self.board[location].player != self.player:
            return State(self, np.copy(self.board), self.player), -max_value, 1
        board, reward = self.board[location].move(move)
        if reward < 0:
            reset = 1
        else:
            reset = 0
            self.n_turns += 1
        if reward >= max_value:
            reset = self.n_turns
        return State(self, np.copy(self.board), self.player), reward, reset


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
        return '\n'.join((' '.join((str(piece) if piece else '·'
                                    for piece in row)) for row in self.board))


def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    return [(piece.location, move) for piece in state.session.pieces[state.player]
            for move in piece.get_moves()]


def predict_action(state, action):
    location, move = action
    board, reward = state.board[location].test(move)
    reset = state.session.n_turns + 1 if reward >= max_value else 0
    return State(state.session, board, state.player), reward, reset

#session = Session()
#state = State(session, session.board, 1)
#
#for action in get_actions(state):
#    state_transition, reward, reset = predict_action(state, action)
#    print(state_transition, action)
#print(state)
