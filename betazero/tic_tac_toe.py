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
    model.add(Conv2D(16, (3, 3), activation='selu', input_shape=input_dimensions, data_format="channels_first"))
    model.add(Flatten())
    model.add(Dense(8, activation='selu'))
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


def get_actions(state):
    """Returns the list of all valid actions given a game state."""
    return (action for action, spot in np.ndenumerate(state) if spot == 0)


def predict_action(state, action):
    """Returns a tuple consisting of (state_transition, reward, reset_count)."""
    # make buffer
    state_transition = np.copy(state)
    # invalid action
    if state_transition[action[0], action[1]] != 0:
        return state_transition, -1, 1
    # apply action
    state_transition[action[0], action[1]] = 1
    total_actions = np.count_nonzero(state_transition)
    # win condition
    if np.max(win_masks.dot(state_transition.flat)) == 3:
        return state_transition, 1, total_actions
    # tie condition
    elif total_actions == 9:
        return state_transition, 0, 9
    # not an end condition
    return state_transition, 0, 0


def critical_action_filter(state):
    """Generate a map of critical spots same dimension as input state"""
    critical = np.zeros(9, dtype=np.int8)
    for i, n_inline in np.ndenumerate(win_masks.dot(state.flat)):
        if abs(n_inline) == 2:
            critical += win_masks[i] - np.abs(state.flat * win_masks[i])
        elif abs(n_inline) == 3:
            critical += win_masks[i]
    return critical.reshape((3,3))


def reduce_symetry(state):
    """Map symetrically equivalent states to a unique state."""
    symetric_states = [state, np.rot90(state)]
    symetric_states.extend([np.flip(symetric_state, 0) for symetric_state in symetric_states])
    symetric_states.extend([np.flip(symetric_state, 1) for symetric_state in symetric_states])
    byte_representations = (symetric_state.tobytes() for symetric_state in symetric_states)
    return max(zip(symetric_states, byte_representations), key = lambda x: x[1])


def input_transform(state, reduce_symetry_enable = True):
    """Transform an input state to an input format the model requires"""
    reduced_state = reduce_symetry(state)[0] if reduce_symetry_enable else state
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
            actions[reduced_bytes] = [action, reduced_state, reward, reset_count]
    return actions


class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((3, 3), dtype=np.int8)
        return self.state, 0, 0

    def do_action(self, action):
        (state, reward, reset_count) = predict_action(self.state, action)
        if reset_count == 0:
            self.state = -state
        elif reset_count == 1:
            self.state = state
        else:
            self.reset()
        return state, reward, reset_count
