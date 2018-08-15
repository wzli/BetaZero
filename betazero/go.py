import numpy as np

board_size = (7, 7)
input_dimensions = (1, *board_size)
output_dimension = board_size[0] * board_size[1]
max_value = output_dimension
min_max = True


# go helpers
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


def liberty_map(board, group_lookup):
    return board * np.vectorize(lambda x: len(x.liberties) if x else 0)(
        group_lookup)


def territory_map(board, group_lookup):
    territory_map = np.copy(board)
    frontier = set()
    # start from the borders of each stone group
    for group in {group for group in group_lookup.flat if group}:
        frontier.update(group.liberties)
    # itteratively convert empty spots to their adjacent majority
    conversions = []
    while frontier:
        next_frontier = set()
        for spot in frontier:
            adjacent_spots = get_adjacent(spot)
            majority = np.sign(
                sum((territory_map[adjacent_spot]
                     for adjacent_spot in adjacent_spots)))
            if majority != 0:
                conversions.append((spot, majority))
                next_frontier.update({
                    adjacent_spot
                    for adjacent_spot in adjacent_spots
                    if territory_map[adjacent_spot] == 0
                    and adjacent_spot not in frontier
                })
        for spot, majority in conversions:
            territory_map[spot] = majority
        conversions.clear()
        frontier = next_frontier
    return territory_map


class StoneGroup:
    def __init__(self, stones=None, liberties=None):
        self.stones = stones if stones else set()
        self.liberties = liberties if liberties else set()

    def copy(self, group_lookup=None, modified_groups=None):
        group = StoneGroup(self.stones.copy(), self.liberties.copy())
        if group_lookup is not None:
            for lookup_entry in self.stones:
                group_lookup[lookup_entry] = group
        if modified_groups:
            modified_groups.add(group)
        return group

    def merge(self, group, link_stone=None, group_lookup=None):
        self.stones.update(group.stones)
        self.liberties.update(group.liberties)
        if link_stone:
            self.liberties.remove(link_stone)
        else:
            self.liberties.difference_update(group.stones)
        # update old group to point to new group
        if group_lookup is not None:
            for stone in group.stones:
                group_lookup[stone] = self


def ascii_board(board):
    ascii_board = np.chararray(board.shape)
    ascii_board[board == 0] = ' '
    ascii_board[board == 1] = 'X'
    ascii_board[board == -1] = 'O'
    return ascii_board.decode('utf-8')


def print_groups(group_lookup):
    for i, group in enumerate({group for group in group_lookup.flat if group}):
        print("Group", i)
        print(len(group.stones), "stones:", group.stones)
        print(len(group.liberties), "liberties:", group.liberties)


# core go engine
def place_stone(stone, perspective, board, group_lookup, session=None):
    # create stone group with a single stone
    group = StoneGroup({stone})
    if session:
        session.empty_spots.remove(stone)
    else:
        modified_groups = {group}
        board = np.copy(board)
        group_lookup = np.copy(group_lookup)
    group_lookup[stone] = group
    board[stone] = perspective
    # only the adjacent spots of affected
    for adjacent_spot in get_adjacent(stone):
        # if there is an adjacent enemy
        if board[adjacent_spot] == -perspective:
            # look up enemy group
            enemy_group = group_lookup[adjacent_spot]
            if not session and enemy_group not in modified_groups:
                enemy_group = enemy_group.copy(group_lookup, modified_groups)
            # remove a liberty from their stone group
            enemy_group.liberties.remove(stone)
            # if no more liberties left capture enemy
            if not enemy_group.liberties:
                # remove the capture spot from list
                if session:
                    session.capture_spots[-perspective].discard(stone)
                    # recode ko if only one piece was taken
                    if len(enemy_group.stones) == 1:
                        session.ko = next(iter(enemy_group.stones))
                    else:
                        session.ko = None
                    # remove peices
                    session.empty_spots.update(enemy_group.stones)
                for captured in enemy_group.stones:
                    board[captured] = 0
                    group_lookup[captured] = None
                    # add liberties back to surrounding friendly groups
                    for released in get_adjacent(captured):
                        if board[released] == perspective:
                            released_group = group_lookup[released]
                            if not session and released_group not in modified_groups:
                                released_group = released_group.copy(
                                    group_lookup, modified_groups)
                            if session and len(released_group.liberties) == 1:
                                session.capture_spots[perspective].discard(
                                    next(iter(released_group.liberties)))
                            released_group.liberties.add(captured)
            # if one liberty left, add to capture spots
            elif session and len(enemy_group.liberties) == 1:
                session.capture_spots[-perspective].update(
                    enemy_group.liberties)
        # if friendly group, merge
        elif board[adjacent_spot] == perspective:
            adjacent_group = group_lookup[adjacent_spot]
            if session and len(adjacent_group.liberties) == 1:
                session.capture_spots[perspective].discard(stone)
            group.merge(adjacent_group, stone, group_lookup)
        # add liberties if there are free space
        elif board[adjacent_spot] == 0:
            group.liberties.add(adjacent_spot)
    # if only one liberty left add to enemy's capture spots
    if session and len(group.liberties) == 1:
        session.capture_spots[perspective].update(group.liberties)
    return board, group_lookup


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
            state = State(self, self.board, self.group_lookup,
                          self.perspective)
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
            if self.is_invalid(action, self.perspective):
                place_stone(action, self.perspective, self.board,
                            self.group_lookup, self.capture_spots,
                            self.empty_spots)
                state = State(self, self.board, self.group_lookup,
                              self.perspective)
                self.n_turns += 1
                reward = 0
                reset = 0
                self.perspective *= -1
            # invalid stone
            else:
                state = State(self, self.board, self.group_lookup,
                              self.perspective)
                reward = -max_value
                reset = 1
        return state, reward, reset_count

    def is_suicide(self, stone, perspective=None):
        if not perspective:
            perspective = self.perspective
        # iterate adjacent.stones
        for adjacent_spot in get_adjacent(stone):
            # creates liberty, not suicide
            if self.board[adjacent_spot] == 0:
                return False
            # if any adjacent friendly group has more than one liberty, not suicide
            if (self.board[adjacent_spot] == perspective
                    and len(self.group_lookup[adjacent_spot].liberties) > 1):
                return False
            # if any adjacent enemy group can be captured, not suicide
            if (self.board[adjacent_spot] == -perspective
                    and len(self.group_lookup[adjacent_spot].liberties) == 1):
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

    def print_status(self):
        print_groups(self.group_lookup)
        print("Capture Spots", self.capture_spots)
        board = ascii_board(self.board)
        for capture_spot in self.capture_spots[1]:
            board[capture_spot] = '^'
        for capture_spot in self.capture_spots[-1]:
            board[capture_spot] = '~'
        print(board)


#------------ The below is required game interface for betazero


#warning, this only works with latest state, due to optomizations
#that required current session state
def get_actions(state, include_pass=False):
    # take all empty spots minus the suicidal ones
    action_set = state.session.empty_spots.difference({
        spot
        for spot in state.session.capture_spots[state.perspective]
        if (state.session.is_suicide(spot, state.perspective))
    })
    # can't take back a ko
    if state.session.ko:
        action_set.remove(state.session.ko)
    # action None is to pass
    if include_pass or not action_set:
        action_set.add(None)
    return action_set


def predict_action(self, state, action):
    # if turn pass
    if not action:
        # score and end the game when both players pass
        if self.turn_pass:
            return state, score_board(board), self.n_turns + 1
        # continue game otherwise
        return board, 0, 0
    # evaluation
    board = self.evaluate_stone(action)
    return board


class State:
    def __init__(self, session, board, group_lookup, perspective):
        self.session = session
        self.board = board
        self.group_lookup = group_lookup
        self.perspective = perspective

    def flip(self):
        return State(self.session, self.board, self.group_lookup,
                     -self.perspective)

    def array(self):
        return np.array([self.board, self.board])[np.newaxis]

    def key(self):
        return self.board.tobytes()


session = Session()

place_stone((3, 3), 1, session.board, session.group_lookup, session)
place_stone((3, 4), -1, session.board, session.group_lookup, session)
place_stone((2, 3), 1, session.board, session.group_lookup, session)
place_stone((4, 2), -1, session.board, session.group_lookup, session)
place_stone((4, 3), 1, session.board, session.group_lookup, session)
place_stone((1, 4), -1, session.board, session.group_lookup, session)
place_stone((5, 2), 1, session.board, session.group_lookup, session)
place_stone((4, 1), -1, session.board, session.group_lookup, session)
place_stone((3, 2), 1, session.board, session.group_lookup, session)
place_stone((5, 1), -1, session.board, session.group_lookup, session)
place_stone((4, 0), 1, session.board, session.group_lookup, session)
place_stone((5, 3), -1, session.board, session.group_lookup, session)
place_stone((6, 2), 1, session.board, session.group_lookup, session)
place_stone((6, 3), -1, session.board, session.group_lookup, session)
place_stone((6, 1), 1, session.board, session.group_lookup, session)
place_stone((6, 0), 1, session.board, session.group_lookup, session)
place_stone((5, 0), 1, session.board, session.group_lookup, session)
board, group_lookup = place_stone((3, 1), 1, session.board,
                                  session.group_lookup)

print(session.get_score(1))
state = State(session, session.board, session.group_lookup, 1)
print(get_actions(state))
print(liberty_map(session.board, session.group_lookup))
print(territory_map(session.board, session.group_lookup))
print(ascii_board(board))
print_groups(group_lookup)
session.print_status()
