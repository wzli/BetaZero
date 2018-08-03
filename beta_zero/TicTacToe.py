#usr/bin/python3
import numpy as np

input_dimentions = (3, 3, 2)

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
    action = np.zeros((3, 3), dtype=np.int8)
    action[i][j] = 1
    return action

def get_actions(state):
    return [action_index(i,j) for (i, j), value in np.ndenumerate(state) if value == 0]

def predict_action(state, action):
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
    # not an end condition, unknown value
    return (state_transition, None, 0)

def reduce_symetry(state):
    symetric_states = [state, np.flip(state, 0), np.flip(state, 1), np.rot90(state), np.rot90(np.flip(state, 0)), np.rot90(np.rot90(state))]
    byte_representations = [symetric_state.tobytes() for symetric_state in symetric_states]
    max_index = byte_representations.index(max(byte_representations))
    return (symetric_states[max_index], byte_representations[max_index])

def critical_action_filter(state):
    critical = np.zeros(9, dtype=np.int8)
    for i, value in np.ndenumerate(win_masks.dot(state.flat)):
        if abs(value) == 2:
            critical += win_masks[i] - np.abs(state.flat * win_masks[i])
        elif abs(value) == 3:
            critical += win_masks[i]
    return critical.reshape((3,3))

def input_transform(state, reduce_symetry_enable = True):
    if reduce_symetry_enable:
        reduced_state, _ = reduce_symetry(state)
    else:
        reduced_state = state
    return np.array([reduced_state, critical_action_filter(reduced_state)])

def generate_action_inputs(state):
    predictions = [(action, predict_action(state, action)) for action in get_actions(state)]
    reduced_set = {}
    for action, (state_transition, value, reset_count) in predictions:
        reduced_state, reduced_bytes = reduce_symetry(state_transition)
        reduced_set[reduced_bytes] = (action, reduced_state, value, reset_count)
    return tuple(reduced_set.values())

class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((3, 3), dtype=np.int8)

    def do_action(self, action):
        (state, value, reset_count) = predict_action(self.state, action)
        if reset_count > 1:
            self.reset()
        else:
            self.state = -state
        return (state, value, reset_count)

    def do_action_index(self, i, j):
        return self.do_action(action_index(i, j))

class Model:
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def predict(self, x):
        return np.array([0,1])
