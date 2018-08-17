import numpy as np

board_size = (10, 9)


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
        col = move[0] if move else self.location[0]
        return ((col - 4.5) * self.player) > 0

    def is_valid(self, move):
        return is_within_bounds(move) and self.board[move] != self.player

    def move(self, move, mutable=True):
        if self.board[move]:
            self.peices[-self.player].remove(self.board[move])
        self.board[move] = self
        self.board[self.location] = None
        self.location = move
        return self.board

    def test(self, move):
        board = np.copy(self.board)
        board[move] = self
        board[self.location] = None
        return board


class Pawn(Piece):
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


class Bishop(Piece):
    def get_moves(self):
        row, col = self.location
        return [
            move for move in ((row + 2, col + 2), (row + 2, col - 2),
                              (row - 2, col + 2), (row - 2, col - 2))
            if self.is_valid(move) and not self.is_across_river(move)
        ]

    def __str__(self):
        return 'B' if self.player > 0 else 'b'


default_spawn = (None, (
    (Pawn, (3, 0)),
    (Pawn, (3, 2)),
    (Pawn, (3, 4)),
    (Pawn, (3, 6)),
    (Pawn, (3, 8)),
    (Bishop, (0, 2)),
    (Bishop, (0, 6)),
), (
    (Pawn, (6, 0)),
    (Pawn, (6, 2)),
    (Pawn, (6, 4)),
    (Pawn, (6, 6)),
    (Pawn, (6, 8)),
    (Bishop, (9, 2)),
    (Bishop, (9, 6)),
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
                )(self.board) for piece in (Pawn, Bishop)))[np.newaxis]

    def key(self):
        return self.board.tobytes()

    def __str__(self):
        return '\n'.join((' '.join((str(piece) if piece else '·'
                                    for piece in row)) for row in self.board))


def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    return [(piece, move) for piece in state.session.pieces[state.player]
            for move in piece.get_moves()]


def predict_action(state, action):
    piece, move = action
    return State(state.session, piece.test(move), state.player)


session = Session()

for row in session.board:
    print(*(piece if piece else '·' for piece in row))

for piece in session.pieces[1]:
    print(piece.get_moves(), piece)

state = State(session, session.board, 1)

for action in get_actions(state):
    print(action)

print(state)
print(state.array())
