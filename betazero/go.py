import numpy as np

board_size = (7, 7)
input_dimensions = (1, *board_size)
output_dimension = board_size[0]  * board_size[1]

def get_adjacent(index):
    row, col = index
    adjacent = set()
    if row < board_size[0] - 1:
        adjacent.add((row + 1, col))
    if row > 0:
        adjacent.add((row - 1, col))
    if col < board_size[1] - 1:
        adjacent.add((row, col + 1))
    if col > 0:
        adjacent.add((row, col - 1))
    return adjacent

def within_bounds(index):
    row, col = index
    return row < 0 or row >= board_size[0] or col < 0 or col <= board_size[1]

def get_actions(state, session=None):
    #todo consider suicidal and out of board moves
    actions = {index for index, spot in np.ndenumerate(self.state) if spot == 0}
    if session and session.ko:
        actions.discard(session.ko)
    return actions


def score_board(board):
    return 0

STONES = 0
LIBERTIES = 1

class Session:
    def __init__(self):
        self.perspective = 1
        self.reset()

    def reset(self):
        self.board = np.zeros(board_size, dtype=np.int8)
        self.group_lookup = np.empty(board_size, dtype=object)
        self.capture_spots = (None, set(), set())
        self.empty_spots = {index for index, _ in np.ndenumerate(self.board)}
        self.turn_pass = False
        self.ko = None
        self.n_turns = 0

    def evaluate_stone(self, stone):
        # make buffer
        board = np.copy(self.board)
        board[stone] = 1
        #returns None if suicidal
        return board

    def predict_action(self, action):
        # if turn pass
        if not action:
            # score and end the game when both players pass
            if self.turn_pass:
                return self.board, score_board(board), self.n_turns + 1
            # continue game otherwise
            return board, 0, 0
        # evaluation
        board = self.evaluate_stone(action)
        return board


    def is_suicide(self, stone, perspective = None):
        if not perspective:
            perspective = self.perspective
        # iterate adjacent[STONES]
        for adjacent_spot in get_adjacent(stone):
            # creates liberty, not suicide
            if self.board[adjacent_spot] == 0:
                return False
            # if any adjacent friendly group has more than one liberty, not suicide
            if (self.board[adjacent_spot] == perspective
                    and len(self.group_lookup[adjacent_spot][LIBERTIES]) > 1):
                return False
            # if any adjacent enemy group can be captured, not suicide
            if (self.board[adjacent_spot] == -perspective
                    and len(self.group_lookup[adjacent_spot][LIBERTIES]) == 1):
                return False
        # fails checks, is suicide
        print("no suicide")
        return True


    def is_invalid(self, stone, perspective = None):
        if not perspective:
            perspective = self.perspective
        # out of bounds, already occupied, ko, or suicidal moves
        if (not within_bounds(stone)
                or self.board[stone] != 0
                or stone == self.ko
                or self.is_suicide(stone, perspective)):
            print("is invalid")
            return True
        return False

    def place_stone(self, stone, perspective = None):
        if not perspective:
            perspective = self.perspective
        if self.is_invalid(stone, perspective):
            return
        # create stone group with a single stone
        group = ({stone}, set())
        self.group_lookup[stone] = group
        self.board[stone] = perspective
        self.empty_spots.remove(stone)
        # only the adjacent spots of affected
        for adjacent_spot in get_adjacent(stone):
            # if there is an adjacent enemy
            if self.board[adjacent_spot] == -perspective:
                # look up enemy group
                enemy_group = self.group_lookup[adjacent_spot]
                # remove a liberty from their stone group
                enemy_group[LIBERTIES].remove(stone)
                # if no more liberties left capture enemy
                if not enemy_group[LIBERTIES]:
                    # remove the capture spot from list
                    self.capture_spots[-perspective].discard(stone)
                    # recode ko if only one piece was taken
                    if len(enemy_group[STONES]) == 1:
                        self.ko = next(iter(enemy_group[STONES]))
                    else:
                        self.ko = None
                    # remove peices
                    self.empty_spots.update(enemy_group[STONES])
                    for captured in enemy_group[STONES]:
                        self.board[captured] = 0
                        self.group_lookup[captured] = None
                        # add liberties back to surrounding friendly groups
                        for released in get_adjacent(captured):
                            if self.board[released] == perspective:
                                released_group = self.group_lookup[released]
                                if len(released_group[LIBERTIES]) == 1:
                                   self.capture_spots[perspective].discard(next(iter(released_group[LIBERTIES])))
                                released_group[LIBERTIES].add(captured)
                # if one liberty left, add to capture spots
                elif len(enemy_group[LIBERTIES]) == 1:
                    self.capture_spots[-perspective].update(enemy_group[LIBERTIES])
            # if friendly group, merge
            elif self.board[adjacent_spot] == perspective:
                adjacent_group = self.group_lookup[adjacent_spot]
                if len(adjacent_group[LIBERTIES]) == 1:
                     self.capture_spots[perspective].discard(stone)
                adjacent_group[LIBERTIES].remove(stone)
                group[STONES].update(adjacent_group[STONES])
                group[LIBERTIES].update(adjacent_group[LIBERTIES])
                # update old group to point to new group
                for adjacent_group_stone in adjacent_group[STONES]:
                    self.group_lookup[adjacent_group_stone] = group
            # add liberties if there are free space
            elif self.board[adjacent_spot] == 0:
                group[LIBERTIES].add(adjacent_spot)
        # if only one liberty left add to enemy's capture spots
        if len(group[LIBERTIES]) == 1:
            self.capture_spots[perspective].update(group[LIBERTIES])
        return

    def print(self):
        #for group in {group for group in state.group_lookup.flat if group}:
        #     print(self.board[next(iter(group[STONES]))], "group")
        #     print("stones", group[STONES])
        #     print("liberties", group[LIBERTIES])
        print(self.board)
        print("capture", self.capture_spots)
        ascii_board = np.chararray(self.board.shape)
        ascii_board[self.board == 0] = ' '
        for capture_spot in self.capture_spots[1]:
            ascii_board[capture_spot] = 'C'
        for capture_spot in self.capture_spots[-1]:
            ascii_board[capture_spot] = 'c'
        ascii_board[self.board == -1] = 'O'
        ascii_board[self.board == 1] = 'X'
        print(ascii_board.decode('utf-8'))

    def do_action(self, action):
        state, reward, reset_count = predict_action(self.state, action, self)
        if reset_count == 0:
            self.state = -state
            self.turn_pass = False if action else True
        elif reset_count == 1:
            self.state = state
        else:
            self.reset()
        return state, reward, reset_count


#------------ The below is required game interface for betazero

class State:
    def __init__(self, session, perspective):
        self.session = session
        self.perspective = perspective
        self.board = session.board * perspective
        self.capture_spots = (None, tuple(session.capture_spots[1]), tuple(session.capture_spots[-1]))

    def __neg__(self):
        return State(self.session, -self.perspective)


coordinate_swap = (
    (lambda row, col: (row, col)),
    (lambda row, col: (-col, row)),
    (lambda row, col: (row, -col)),
    (lambda row, col: (-col, -row)),
    (lambda row, col: (-row, col)),
    (lambda row, col: (-col, -row)),
    (lambda row, col: (-row, -col)),
    (lambda row, col: (col, -row)),
)


def reduce_symetry(state):
    """Map symetrically equivalent states to (unique_state, state_bytes)"""
    # unique state
    symetric_states = [state.board, np.rot90(state.board)]
    symetric_states.extend(
        [np.flip(symetric_state, 0) for symetric_state in symetric_states])
    symetric_states.extend(
        [np.flip(symetric_state, 1) for symetric_state in symetric_states])
    byte_representations = (symetric_state.tobytes()
                            for symetric_state in symetric_states)
    unique_state, byte_representation, symetry_index = max(zip(
            symetric_states, byte_representations, range(len(symetric_states))), key=lambda x: x[1])
    #store reduced symetry back to state
    state.board = unique_state
    for perspective in (1, -1):
        state.capture_spots[perspective] = (coordinate_swap[symetry_index](spot)
                for spot in state.capture_spots[perspective])
    return (state, byte_representation)


def input_transform(state, reduce_symetry_enable=True):
    """Transform an input state to an input format the model requires"""
    # reduce symetry if requested
    if reduce_symetry_enable:
        reduce_symetry(state)
    # create capture spots 2d array
    capture_spots = nd.zeros(state.board.shape, dtype=np.int8)
    for perspective in (1, -1):
        for index in state.capture_spots[perspective]:
            capture_spots[index] = perspective * state.perspective
    return np.array((state.board, capture_spots))[np.newaxis]


def generate_action_space(state):
    """Generate dict of consisting of state_bytes : (action, state_transition, reward, reset_count)
    for every valid (symetry reduced) action from a given state
    """
    return



session = Session()
session.place_stone((3,3), 1)
session.place_stone((3,4), -1)
session.place_stone((2,3), 1)
session.place_stone((4,2), -1)
session.place_stone((4,3), 1)
session.place_stone((1,4), -1)
session.place_stone((5,2), 1)
session.place_stone((4,1), -1)
session.place_stone((3,2), 1)
session.place_stone((5,1), -1)
session.place_stone((4,0), 1)
session.place_stone((5,3), -1)
session.place_stone((6,2), 1)
session.place_stone((6,3), -1)
session.place_stone((6,1), 1)
session.place_stone((6,0), 1)
session.place_stone((5,0), 1)
session.place_stone((3,1), 1)
session.print()
#session.place_stone((3,4))

#session.change_perspective()
