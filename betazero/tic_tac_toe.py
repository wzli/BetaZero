#usr/bin/python3
import numpy as np
from .utils import ascii_board, parse_grid_input

board_size = (3, 3)
max_value = 1
min_max = True
rotational_symetry = True
vertical_symetry = True
horizontal_symetry = True
terminal_state = True
reward_span = 0


#simple fully connected network
def ValueModel():
    from keras.models import Sequential
    from keras.layers import Conv2D, Dense, Flatten

    input_dimensions = (*board_size, 1)
    output_dimension = 3

    model = Sequential()
    model.add(Flatten(input_shape=input_dimensions))
    model.add(Dense(32, activation='selu'))
    model.add(Dense(16, activation='selu'))
    model.add(Dense(output_dimension, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# game rules
win_masks = np.array(
    [[
        [1, 1, 1],
        [0, 0, 0],
        [0, 0, 0],
    ], [
        [0, 0, 0],
        [1, 1, 1],
        [0, 0, 0],
    ], [
        [0, 0, 0],
        [0, 0, 0],
        [1, 1, 1],
    ], [
        [1, 0, 0],
        [1, 0, 0],
        [1, 0, 0],
    ], [
        [0, 1, 0],
        [0, 1, 0],
        [0, 1, 0],
    ], [
        [0, 0, 1],
        [0, 0, 1],
        [0, 0, 1],
    ], [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ], [
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
    ]],
    dtype=np.int8).reshape(8, 3 * 3)

#------------ The below is required game interface for betazero


def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    return [
        action for action, spot in np.ndenumerate(state.board) if spot == 0
    ]


def predict_action(state, action):
    """Returns a tuple consisting of (state_transition, reward, reset_count)."""
    # make buffer
    state_transition = np.copy(state.board)
    action = tuple(action)
    # invalid action
    if state_transition[action] != 0:
        return State(state_transition), -1, 1
    # apply action
    state_transition[action] = 1
    total_actions = np.count_nonzero(state_transition)
    # win condition
    if np.max(win_masks.dot(state_transition.flat)) == 3:
        return State(state_transition), 1, total_actions
    # tie condition
    elif total_actions == 9:
        return State(state_transition), 0, 9
    # not an end condition
    return State(state_transition), 0, 0


class State:
    def __init__(self, board):
        self.board = board

    def flip(self):
        return State(-self.board)

    def array(self):
        return self.board[np.newaxis, np.newaxis]

    def key(self):
        return self.board.tobytes()

    def __str__(self):
        return str(ascii_board(self.board))


class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = State(np.zeros(board_size, dtype=np.int8))
        return self.state, 0, 0

    def do_action(self, action):
        (state, reward, reset_count) = predict_action(self.state, action)
        if reset_count == 0:
            self.state = state.flip()
        elif reset_count == 1:
            self.state = state
        else:
            self.reset()
        return State(state.board), reward, reset_count


def get_human_action():
    return parse_grid_input(board_size)
