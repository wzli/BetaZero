#usr/bin/python3

import numpy as np

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
    return np.array([action_index(i,j) for (i, j), value in np.ndenumerate(state) if value == 0], dtype=np.int8)

def predict_action(state, action):
    # invalid action
    if np.dot(np.abs(action.flat), np.abs(state.flat)) != 0 or np.sum(np.abs(action)) != 1:
        return (state, -1, 1)
    state += action
    total_actions = np.count_nonzero(state)
    # win condition
    if np.max(win_masks.dot(state.flat)) == 3:
        return (state, 1, total_actions)
    # tie condition
    elif total_actions == 9:
        return (state, 0, 9)
    return (state, 0, 0)

def reduce_symetry(state):
    symetric_states = [state, np.flip(state, 0), np.flip(state, 1), np.rot90(state), np.rot90(np.rot90(state)), np.rot90(np.rot90(np.rot90(state)))]
    return symetric_states[np.argmax(np.array([symetric_state.tobytes() for symetric_state in symetric_states]))]

class Game:
    def __init(self):
        self.reset()

    def reset(self):
        self.state = np.zeros((3, 3), dtype=np.int8)

    def do_action(self, action):
        (state, reward, reset_count) = predict_action(self.state, action)
        if reset_count > 1:
            self.reset()
        else:
            self.state = -state
        return (state, reward, reset_count)

    def do_action_index(self, i, j):
        return self.do_action(action_index(i, j))
