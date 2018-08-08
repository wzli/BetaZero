#usr/bin/python3
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten

input_dimensions = (2, 3, 3)
output_dimension = 3
max_value = 1
min_max = True

def ValueModel():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), activation='selu', input_shape=input_dimensions, data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(128, activation='selu'))
    model.add(Dense(output_dimension, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

win_masks = np.array([
[
    [1, 1, 1],
    [0, 0, 0],
    [0, 0, 0],
],[
    [0, 0, 0],
    [1, 1, 1],
    [0, 0, 0],
],[
    [0, 0, 0],
    [0, 0, 0],
    [1, 1, 1],
],[
    [1, 0, 0],
    [1, 0, 0],
    [1, 0, 0],
],[
    [0, 1, 0],
    [0, 1, 0],
    [0, 1, 0],
],[
    [0, 0, 1],
    [0, 0, 1],
    [0, 0, 1],
],[
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
],[
    [0, 0, 1],
    [0, 1, 0],
    [1, 0, 0],
]], dtype=np.int8
).reshape(8, 3*3)

def action_index(i, j):
    """Returns the action encoding given an game board index."""
    action = np.zeros((3, 3), dtype=np.int8)
    action[i][j] = 1
    return action

def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    return (action_index(i,j) for (i, j), spot in np.ndenumerate(state) if spot == 0)

def predict_action(state, action):
    """Returns a tuple consisting of (state_transition, reward, reset_count)."""
    # invalid action
    if np.dot(np.abs(action.flat), np.abs(state.flat)) != 0 or np.sum(np.abs(action)) != 1:
        return (state, -1, 1)
    state_transition = state + action
    total_actions = np.count_nonzero(state_transition)
    # win condition
    if np.max(win_masks.dot(state_transition.flat)) == 3:
        return (state_transition, 1, total_actions)
    # tie condition
    elif total_actions == 9:
        return (state_transition, 0, 9)
    # not an end condition
    return (state_transition, 0, 0)

def reduce_symetry(state):
    """Map symetrically equivalent states to a unique state."""
    symetric_states = ((symetric_state, symetric_state.tobytes()) for symetric_state in
        (
            state,
            np.flip(state, 0),
            np.flip(state, 1),
            np.rot90(state),
            np.rot90(np.flip(state, 0)),
            np.rot90(np.rot90(state))
        ))
    return max(symetric_states, key = lambda x: x[1])

def critical_action_filter(state):
    """Generate a map of critical spots same dimension as input state"""
    critical = np.zeros(9, dtype=np.int8)
    for i, n_inline in np.ndenumerate(win_masks.dot(state.flat)):
        if abs(n_inline) == 2:
            critical += win_masks[i] - np.abs(state.flat * win_masks[i])
        elif abs(n_inline) == 3:
            critical += win_masks[i]
    return critical.reshape((3,3))

def input_transform(state, reduce_symetry_enable = True):
    """Transform an input state to an input format the model requires"""
    if reduce_symetry_enable:
        reduced_state, _ = reduce_symetry(state)
    else:
        reduced_state = state
    return np.array((reduced_state, critical_action_filter(reduced_state)))[np.newaxis]

def generate_action_choices(state):
    """Generate an iterator of tuples consisting of (action, state_transition, state_bytes, reward, reset_count)
    for every valid (symetry reduced) action from a given state
    """
    actions = {}
    for action in get_actions(state):
        state_transition, reward, reset_count = predict_action(state, action)
        reduced_state, reduced_bytes = reduce_symetry(state_transition)
        if reduced_bytes not in actions:
            actions[reduced_bytes] = [action, reduced_state, reduced_bytes, reward, reset_count]
    return actions.values()

class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        return (self.state, 0, 0)

    def do_action(self, action):
        (state, reward, reset_count) = predict_action(self.state, action)
        if reset_count > 1:
            self.reset()
        elif reset_count == 1:
            self.state = state
        else:
            self.state = -state
        return (-state, reward, reset_count)

    def do_action_index(self, i, j):
        return self.do_action(action_index(i, j))
