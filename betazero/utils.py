import numpy as np
from math import floor

import signal
from contextlib import contextmanager


@contextmanager
def timeout(time):
    # Register a function to raise a TimeoutError on the signal
    def raise_timeout(signum, frame):
        raise TimeoutError(str(time) + "s timeout exceeded")

    signal.signal(signal.SIGALRM, raise_timeout)
    # Schedule the signal to be sent after time
    signal.alarm(time)
    try:
        # continue running code block
        yield
    finally:
        # Unregister the signal
        signal.signal(signal.SIGALRM, signal.SIG_IGN)


def clip(value, upper, lower=0):
    return max(lower, min(upper, value))


def one_hot_pdf(value, length):
    pdf = np.zeros(length)
    pdf[clip(floor(length * 0.5 * (1 + value)), length - 1)] = 1
    return pdf


def shift_pdf(pdf, value):
    shift_index = clip(
        floor(pdf.shape[0] * 0.5 * value), pdf.shape[0] - 1, -pdf.shape[0] + 1)
    if shift_index == 0:
        return pdf
    else:
        shifted_pdf = np.zeros(pdf.shape)
        if shift_index > 0:
            shifted_pdf[shift_index:] = pdf[:-shift_index]
            shifted_pdf[-1] += np.sum(pdf[-shift_index:])
        elif shift_index < 0:
            shifted_pdf[:shift_index] = pdf[-shift_index:]
            shifted_pdf[0] += np.sum(pdf[:-shift_index])
        return shifted_pdf


def max_pdf(pdfs):
    pdfs = np.asarray(pdfs)
    cdfs = np.zeros((pdfs.shape[0], pdfs.shape[1] + 1))
    np.cumsum(pdfs, axis=1, out=cdfs[:, 1:])
    max_cdf = np.prod(cdfs, axis=0)
    max_pdf = np.diff(max_cdf)
    max_pdf = max_pdf / np.sum(max_pdf)
    return max_pdf


def expected_value(pdf, support, return_variance=False):
    expected = np.average(support, weights=pdf)
    if not return_variance:
        return expected
    variance = np.average(support**2, weights=pdf) - expected**2
    return expected, variance


def symetric_arrays(array, rotational_symetry, vertical_symetry,
                    horizontal_symetry):
    """generate list of symetrically equivalent arrays"""
    symetric_arrays = [array]
    if rotational_symetry:
        symetric_arrays.append(
            np.rot90(array, axes=(array.ndim - 2, array.ndim - 1)))
    if vertical_symetry:
        symetric_arrays.extend([
            np.flip(symetric_array, array.ndim - 1)
            for symetric_array in symetric_arrays
        ])
    if horizontal_symetry:
        symetric_arrays.extend([
            np.flip(symetric_array, array.ndim - 2)
            for symetric_array in symetric_arrays
        ])
    return symetric_arrays


def ascii_board(board):
    ascii_board = [' '.join([''] + [str(i) for i in range(board.shape[1])])]
    for i, row in enumerate(board):
        ascii_board.append(' '.join(
            [str(i)] +
            ['·' if cell == 0 else '○' if cell > 0 else '●' for cell in row]))
    return '\n'.join(ascii_board)


def parse_grid_input(board_size):
    while True:
        input_str = input('enter "row col ...": ')
        if input_str == "":
            return None
        try:
            move_index = tuple(int(token) for token in input_str.split(' '))
        except ValueError:
            print("INTEGER PARSING ERROR")
            continue
        if len(move_index) != len(board_size):
            print("INVALID INDEX DIMENSION")
            continue
        if any((i < 0 or i >= axis_len
                for i, axis_len in zip(move_index, board_size))):
            print("INVALID INDEX RANGE")
            continue
        return move_index


class Arena:
    def __init__(self, game, player1, player2):
        self.game = game
        self.session = self.game.Session()
        self.players = (None, player1, player2)
        self.unique_players = {player1, player2}
        self.stats = [0, 0, 0]
        self.score = 0
        updates = self.session.reset()
        for unique_player in self.unique_players:
            unique_player.update_session(*updates)

    def play_match(self,
                   n_games=-1,
                   first_turn=1,
                   verbose=True,
                   turn_timeout=1000):
        self.player_index = first_turn
        # play until n_games limit reached
        while n_games != 0:
            # play turns with a max timeout
            with timeout(turn_timeout):
                reward, reset = self.play_turn(self.player_index, verbose)
                if reset > 1:
                    # game ended
                    n_games -= 1
                    # record stats if not self play
                    if len(self.unique_players) > 1:
                        if reward == 0:
                            self.stats[0] += 1
                        elif reward < 0:
                            self.stats[-self.player_index] += 1
                        elif reward > 0:
                            self.stats[self.player_index] += 1
                        self.score += reward * self.player_index
                        if verbose:
                            print(self.players[1].name, self.stats[1],
                                  self.players[-1].name, self.stats[-1],
                                  "ties", self.stats[0], "score/game",
                                  self.score / sum(self.stats))
                    # the other player's turn to go first
                    first_turn *= -1
                    self.player_index = first_turn
                else:
                    # game continues with next player's turn
                    self.player_index *= -1

    def play_turn(self, player_index, verbose):
        player = self.players[player_index]
        action = player.generate_action(verbose=verbose)
        state, reward, reset = self.session.do_action(action)
        while reset == 1:
            print(player.name, ": INVALID ACTION", action)
            action = player.generate_action(verbose=verbose)
            state, reward, reset = self.session.do_action(action)
        for unique_player in self.unique_players:
            unique_player.update_session(state, reward, reset)
        return reward, reset
