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


def get_empty(state):
    return {index for index, spot in np.ndenumerate(self.state) if spot == 0}


class StoneGroup:
    def __init__(self, stone):
        self.stones = {stone}
        self.liberties = set()

def score_board(board):
    return 0

class State:
    def __init__(self):
        self.perspective = 1
        self.board = np.zeros(board_size, dtype=np.int8)
        self.group_lookup = np.empty(board_size, dtype=object)
        self.capture_spots = (None, set(), set())
        self.suicide_spots = (None, set(), set())
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


    def is_suicide(self, stone, perspective):
        for adjacent_spot in get_adjacent(stone):
            if (self.board[adjacent_spot] == 0
                or (self.board[adjacent_spot] == perspective
                    and len(self.group_lookup[adjacent_spot].liberties) > 1)
                or (self.board[adjacent_spot] == -perspective)
                    and len(self.group_lookup[adjacent_spot].liberties) == 1):
                return False
        return True


    def is_invalid(self, stone, perspective):
        # out of bounds, already occupied, ko, or suicidal moves
        if (not within_bounds(stone)
                or self.board[stone] != 0
                or stone == self.ko
                or stone in self.suicide_spots[perspective]):
            print("Invalid Move")
            return True
        return False

    def place_stone(self, stone, perspective):
        if self.is_invalid(stone, perspective):
            return
        # create stone group with a single stone
        group = StoneGroup(stone)
        self.group_lookup[stone] = group
        self.board[stone] = perspective
        for adjacent_spot in get_adjacent(stone):
            # if there is an adjacent enemy
            if self.board[adjacent_spot] == -perspective:
                # look up enemy group
                enemy_group = self.group_lookup[adjacent_spot]
                # remove a liberty from their stone group
                enemy_group.liberties.remove(stone)
                # if no more liberties left, capture
                if not enemy_group.liberties:
                    for captured in enemy_group.stones:
                        self.board[captured] = 0
                        self.group_lookup[captured] = None
                        # add liberties back to friendly groups from the released captured spots
                        for released in get_adjacent(captured):
                            if self.board[released] == perspective:
                                self.group_lookup[released].liberties.add(captured)
                    # the only way to remove suicide spots is to capture enemy
                    for suicide_spot in self.suicide_spots[perspective]:
                        if not is_suicide(suicide_spot):
                            self.suicide_spots[perspective].remove(suicide_spot)
                    for capture_spot in self.capture_spots[perspective]:
                        for adjacent_spot in get_adjacent(capture_spot):
                            if len(self.group_lookup[adjacent_spot].liberties) > 1:
                                self.capture_spots[perspective].remove(suicide_spot)
                    # already captured, remove the capture spot
                    self.capture_spots[-perspective].discard(stone)
                    self.suicide_spots[-perspective].discard(stone)
                    if len(enemy_group.stones) == 1:
                        self.ko = next(iter(enemy_group.stones))
                    else:
                        self.ko = None
                # if one liberty left, add to capture spots
                elif len(enemy_group.liberties) == 1:
                    self.capture_spots[-perspective].update(enemy_group.liberties)
                    for liberty in enemy_group.liberties:
                        if self.is_suicide(liberty, -perspective):
                            self.suicide_spots[-perspective].add(liberty)
            # if friendly group, merge
            elif self.board[adjacent_spot] == perspective:
                adjacent_group = self.group_lookup[adjacent_spot]
                if len(adjacent_group.liberties) == 1:
                     self.capture_spots[perspective].discard(stone)
                adjacent_group.liberties.remove(stone)
                group.stones.update(adjacent_group.stones)
                group.liberties.update(adjacent_group.liberties)
                # update old group to point to new group
                for adjacent_group_stone in adjacent_group.stones:
                    self.group_lookup[adjacent_group_stone] = group
            # add liberties if there are free space
            elif self.board[adjacent_spot] == 0:
                group.liberties.add(adjacent_spot)
        # if only one liberty left add to enemy's capture spots
        if len(group.liberties) == 1:
            self.capture_spots[perspective].update(group.liberties)
            liberty = next(iter(group.libterties))
            if self.is_suicide(liberty, perspective):
                self.suicide_spots[perspective].add(liberty)
        # check for enemys suicide spots
        for adjacent_spot in get_adjacent(stone):
            if self.board[adjacent_spot] == 0 and self.is_suicide(adjacent_spot, -perspective):
                    self.suicide_spots[-perspective].add(adjacent_spot)

        return

    def print(self):
        for index, group in np.ndenumerate(state.group_lookup):
            if group:
                print(index, group)
                print("stones", group.stones)
                print("liberties", group.liberties)
        print(self.board)
        print("capture", self.capture_spots)
        print("suicide", self.suicide_spots)
        ascii_board = np.chararray(self.board.shape)
        ascii_board[self.board == 0] = ' '
        for capture_spot in self.capture_spots[1]:
            ascii_board[capture_spot] = 'C'
        for capture_spot in self.capture_spots[-1]:
            ascii_board[capture_spot] = 'c'
        for suicide_spot in self.suicide_spots[1]:
            ascii_board[suicide_spot] = 'S'
        for suicide_spot in self.suicide_spots[-1]:
            ascii_board[suicide_spot] = 's'
        ascii_board[self.board == -1] = 'O'
        ascii_board[self.board == 1] = 'X'
        print(ascii_board.decode('utf-8'))






class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        return self.state, 0, 0

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




state = State()
state.place_stone((3,3), 1)
state.place_stone((3,4), -1)
state.place_stone((2,3), 1)
state.place_stone((4,2), -1)
state.place_stone((4,3), 1)
state.place_stone((1,4), -1)
state.place_stone((5,2), 1)
state.place_stone((4,1), -1)
state.place_stone((3,2), 1)
state.place_stone((5,1), -1)
state.place_stone((4,0), 1)
state.place_stone((5,3), -1)
state.place_stone((6,2), 1)
state.place_stone((6,3), -1)
state.print()
#state.place_stone((3,4))

#state.change_perspective()