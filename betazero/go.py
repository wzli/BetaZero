import numpy as np

board_size = (7, 7)
input_dimensions = (1, *board_size)
output_dimension = board_size[0] * board_size[1]
max_value = output_dimension
min_max = True

# go helpers
STONES = 0
LIBERTIES = 1


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


def find_group(board, spot):
    color = board[spot]
    group = {spot}
    border = set()
    frontier = [spot]
    while frontier:
        current_spot = frontier.pop()
        group.add(current_spot)
        for adjacent_spot in get_adjacent(current_spot):
            if board[adjacent_spot] == color and adjacent_spot not in group:
                frontier.append(adjacent_spot)
            elif board[adjacent_spot] != color:
                border.add(adjacent_spot)
    return group, border


# core go engine
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

    def get_score(self, perspective=None):
        if not perspective:
            perspective = self.perspective
        # assume all dead stones are already captured, otherwise
        # there's no easy way to tell life or death
        # every peice counts as a point
        unique, counts = np.unique(self.board, return_counts=True)
        score = np.sum(unique * counts)
        # settle the territory
        unclaimed_spots = set(self.empty_spots)
        while unclaimed_spots:
            territory, border = find_group(self.board, unclaimed_spots.pop())
            unclaimed_spots.difference_update(territory)
            # decide owner of territory based on border
            owner = 0
            for stone in border:
                if owner == 0 and self.board[stone] == 1:
                    owner = 1
                elif owner == 0 and self.board[stone] == -1:
                    owner = -1
                elif owner != self.board[stone]:
                    owner = 0
                    break
            score += owner * len(territory)
        return score * perspective

    # this function enforces swaping perspectives, otherwise min max tree won't work
    def do_action(self, action):
        # end of game when both players pass
        if not action:
            #state doesn't change
            state = State(self, self.perspective)
            self.n_turns += 1
            if self.turn_pass:
                reward = self.get_score(self.perspective)
                reset = self.n_turns
                self.reset()
            else:
                self.turn_pass = True
                reward = 0
                reset = 0
                self.perspective *= -1
        else:
            # place stone
            if self.place_stone(action, self.perspective):
                state = State(self, self.perspective)
                self.n_turns += 1
                reward = 0
                reset = 0
                self.perspective *= -1
            # invalid stone
            else:
                state = State(self, self.perspective)
                reward = -max_value
                reset = 1
        return state, reward, reset_count

    def is_suicide(self, stone, perspective=None):
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
        return True

    def is_invalid(self, stone, perspective=None):
        if not perspective:
            perspective = self.perspective
        # out of bounds, already occupied, ko, or suicidal moves
        if (not within_bounds(stone) or self.board[stone] != 0
                or stone == self.ko or self.is_suicide(stone, perspective)):
            return True
        return False

    def place_stone(self, stone, perspective=None):
        if not perspective:
            perspective = self.perspective
        if self.is_invalid(stone, perspective):
            return False
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
                                    self.capture_spots[perspective].discard(
                                        next(iter(released_group[LIBERTIES])))
                                released_group[LIBERTIES].add(captured)
                # if one liberty left, add to capture spots
                elif len(enemy_group[LIBERTIES]) == 1:
                    self.capture_spots[-perspective].update(
                        enemy_group[LIBERTIES])
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
        return True

    def print_status(self):
        groups = []
        for group in self.group_lookup.flat:
            if group and group not in groups:
                groups.append(group)
        for group in groups:
            if self.board[next(iter(group[STONES]))] == 1:
                print("X Group")
            else:
                print("O Group")
            print("  Stones", group[STONES])
            print("  Liberties", group[LIBERTIES])
        print("Capture Spots", self.capture_spots)
        ascii_board = np.chararray(self.board.shape)
        ascii_board[self.board == 0] = ' '
        for capture_spot in self.capture_spots[1]:
            ascii_board[capture_spot] = '^'
        for capture_spot in self.capture_spots[-1]:
            ascii_board[capture_spot] = '~'
        ascii_board[self.board == 1] = 'X'
        ascii_board[self.board == -1] = 'O'
        print(ascii_board.decode('utf-8'))


#------------ The below is required game interface for betazero


def get_actions(state):
    # take all empty spots minus the suicidal ones
    action_set = state.session.empty_spots.difference({
        spot
        for spot in state.session.capture_spots[state.perspective]
        if (state.session.is_suicide(spot, state.perspective))
    })
    # action None is to pass
    action_set.add(None)
    # can't take back a ko
    if state.session.ko:
        action_set.remove(state.session.ko)
    return action_set


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


class State:
    def __init__(self, parent, perspective):
        self.perspective = perspective
        if parent is Session:
            self.session = parent
            #make frozen copy from session
            self.board = np.copy(parent.board)
            self.capture_spots = (None, tuple(session.capture_spots[1]),
                                  tuple(session.capture_spots[-1]))
        elif parent is State:
            #make frozen copy from state
            self.session = parent.session
            self.board = np.copy(parent.board)
            self.capture_spots = parent.capture_spots

    def __neg__(self):
        return State(self.session, -self.perspective)

    def tobytes():
        return self.board.tobytes()


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


def input_transform(state):
    """Transform an input state to an input format the model requires"""
    # create capture spots 2d array
    capture_spots = nd.zeros(state.board.shape, dtype=np.int8)
    for perspective in (1, -1):
        for index in state.capture_spots[perspective]:
            capture_spots[index] = perspective * state.perspective
    return np.array((state.board, capture_spots))[np.newaxis]


session = Session()
session.place_stone((3, 3), 1)
session.place_stone((3, 4), -1)
session.place_stone((2, 3), 1)
session.place_stone((4, 2), -1)
session.place_stone((4, 3), 1)
session.place_stone((1, 4), -1)
session.place_stone((5, 2), 1)
session.place_stone((4, 1), -1)
session.place_stone((3, 2), 1)
session.place_stone((5, 1), -1)
session.place_stone((4, 0), 1)
session.place_stone((5, 3), -1)
session.place_stone((6, 2), 1)
session.place_stone((6, 3), -1)
session.place_stone((6, 1), 1)
session.place_stone((6, 0), 1)
session.place_stone((5, 0), 1)
session.place_stone((3, 1), 1)
session.print_status()
print(session.get_score(-1))

#session.change_perspective()
