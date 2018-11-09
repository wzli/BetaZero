import numpy as np
from .utils import ascii_board, parse_grid_input

board_size = (5, 5)
max_value = 6
min_max = True
rotational_symetry = True
vertical_symetry = True
horizontal_symetry = True
terminal_state = False
reward_span = 4


# keras model, based on alphazero and mobilenetv2
def ValueModel():
    from keras.models import Model
    from keras import regularizers
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Flatten, Input, ReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import Add

    input_dimensions = (2, *board_size)
    output_dimension = 13
    n_filters = 128
    expansion_factor = 5
    n_res_blocks = 10
    batch_norm_momentum = 0.999
    l2_reg = 1e-4

    inputs = Input(shape=input_dimensions)
    x = Conv2D(
        n_filters, (3, 3),
        padding='same',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    for i in range(n_res_blocks):
        x_in = x
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(
            n_filters * expansion_factor, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D(
            (3, 3),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(
            n_filters, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = Add()([x, x_in])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = Conv2D(
        1, (1, 1),
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        data_format="channels_first")(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6.)(x)
    x = Flatten()(x)
    outputs = Dense(
        output_dimension,
        kernel_regularizer=regularizers.l2(l2_reg),
        activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


def ValueModel_myRes():
    from keras.models import Model
    from keras import regularizers
    from keras.layers import Conv2D, Dense, Flatten, Input, ReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import Add

    input_dimensions = (2, *board_size)
    output_dimension = 13
    filter_size = (3, 3)
    n_filters = 128
    n_res_blocks = 10
    batch_norm_momentum = 0.999
    l2_reg = 1e-4

    inputs = Input(shape=input_dimensions)
    x = Conv2D(
        n_filters,
        filter_size,
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        data_format="channels_first")(inputs)
    # residual blocks
    for i in range(n_res_blocks):
        x_in = x
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6.)(x)
        x = Conv2D(
            n_filters,
            filter_size,
            padding='same',
            use_bias=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            data_format="channels_first")(x)
        x = Add()([x, x_in])
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6.)(x)
    x = Conv2D(
        1, (1, 1),
        padding='same',
        use_bias=False,
        kernel_regularizer=regularizers.l2(l2_reg),
        data_format="channels_first")(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6.)(x)
    x = Flatten()(x)
    outputs = Dense(
        output_dimension,
        activation='softmax',
        kernel_regularizer=regularizers.l2(l2_reg))(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# keras model, based on alphazero and mobilenetv2
def ValueModel_MNV2():
    from keras.models import Model
    from keras import regularizers
    from keras.layers import Conv2D, DepthwiseConv2D, Dense, Flatten, Input, ReLU
    from keras.layers.normalization import BatchNormalization
    from keras.layers.merge import Add

    input_dimensions = (2, *board_size)
    output_dimension = 41
    n_filters = 128
    expansion_factor = 5
    n_res_blocks = 20
    batch_norm_momentum = 0.999
    l2_reg = 1e-4

    inputs = Input(shape=input_dimensions)
    x = Conv2D(
        n_filters, (3, 3),
        padding='same',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(inputs)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    for i in range(n_res_blocks):
        x_in = x
        x = Conv2D(
            n_filters * expansion_factor, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D(
            (3, 3),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(
            n_filters, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = Add()([x, x_in])
    for stride in (2, 2):
        x = Conv2D(
            n_filters * expansion_factor, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = DepthwiseConv2D(
            (3, 3),
            padding='same',
            strides=stride,
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
        x = ReLU(6)(x)
        x = Conv2D(
            n_filters, (1, 1),
            padding='same',
            use_bias=False,
            data_format='channels_first',
            kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = Conv2D(
        n_filters * expansion_factor, (1, 1),
        padding='same',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = DepthwiseConv2D(
        (3, 3),
        padding='valid',
        use_bias=False,
        data_format='channels_first',
        kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = BatchNormalization(momentum=batch_norm_momentum)(x)
    x = ReLU(6)(x)
    x = Flatten()(x)
    outputs = Dense(
        output_dimension,
        kernel_regularizer=regularizers.l2(l2_reg),
        activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model


# go helpers
def get_adjacent(index):
    row, col = index
    adjacent = []
    if row < board_size[0] - 1:
        adjacent.append((row + 1, col))
    if row > 0:
        adjacent.append((row - 1, col))
    if col < board_size[1] - 1:
        adjacent.append((row, col + 1))
    if col > 0:
        adjacent.append((row, col - 1))
    return adjacent


def generate_liberty_map(board, group_lookup):
    return board * np.vectorize(lambda x: len(x.liberties) if x else 0)(
        group_lookup)


def generate_territory_map(board,
                           itterations=max(board_size) // 2 + 1,
                           threshold=3):
    for i in range(itterations):
        buf = board * threshold
        buf[1:] += board[:-1]  # add North
        buf[:-1] += board[1:]  # add South
        buf[:, 1:] += board[:, :-1]  # add West
        buf[:, :-1] += board[:, 1:]  # add East
        board = np.sign(buf)
    return board


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
        if group == self:
            return
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


# core go engine
def place_stone(stone, perspective, board, group_lookup, session=None):
    # create stone group with a single stone
    group = StoneGroup({stone})
    if session:
        session.ko = None
        session.empty_spots.remove(stone)
    else:
        modified_groups = {group}
        board = np.copy(board)
        group_lookup = np.copy(group_lookup)
    group_lookup[stone] = group
    board[stone] = perspective
    captured_group = None
    # only the adjacent spots of affected
    for adjacent_spot in get_adjacent(stone):
        # if there is an adjacent enemy
        if board[adjacent_spot] == -perspective:
            # look up enemy group
            enemy_group = group_lookup[adjacent_spot]
            if not session and enemy_group not in modified_groups:
                enemy_group = enemy_group.copy(group_lookup, modified_groups)
            # remove a liberty from their stone group
            enemy_group.liberties.discard(stone)
            # if no more liberties left capture enemy
            if not enemy_group.liberties:
                captured_group = enemy_group
                # remove the capture spot from list
                if session:
                    # recode ko if only one piece was taken
                    if len(captured_group.stones) == 1:
                        session.ko = adjacent_spot
                    session.empty_spots.update(captured_group.stones)
                # remove peices
                for captured in captured_group.stones:
                    board[captured] = 0
                    group_lookup[captured] = None
                    # add liberties back to surrounding friendly groups
                    for released in get_adjacent(captured):
                        if board[released] == perspective:
                            released_group = group_lookup[released]
                            if not session and released_group not in modified_groups:
                                released_group = released_group.copy(
                                    group_lookup, modified_groups)
                            released_group.liberties.add(captured)
        # if friendly group, merge
        elif board[adjacent_spot] == perspective:
            adjacent_group = group_lookup[adjacent_spot]
            group.merge(adjacent_group, stone, group_lookup)
        # add liberties if there are free space
        elif board[adjacent_spot] == 0:
            group.liberties.add(adjacent_spot)
    return board, group_lookup, captured_group


def print_groups(group_lookup):
    for i, group in enumerate({group for group in group_lookup.flat if group}):
        print("Group", i)
        print(len(group.stones), "stones:", group.stones)
        print(len(group.liberties), "liberties:", group.liberties)


class Session:
    def __init__(self):
        self.reset()

    def reset(self):
        self.perspective = 1
        self.board = np.zeros(board_size, dtype=np.int8)
        self.group_lookup = np.empty(board_size, dtype=object)
        self.empty_spots = {index for index, _ in np.ndenumerate(self.board)}
        self.turn_pass = False
        self.ko = None
        self.n_turns = 0
        self.state = State(self, -self.perspective, np.zeros(board_size),
                           np.zeros(board_size))
        return self.state, 0, 0

    # this function enforces swaping perspectives each turn
    # otherwise min max tree won't work
    def do_action(self, action, mutable=True):
        # returns in the perspective of the action taker
        perspective = self.perspective
        # default values if nothing happens
        board = self.board
        group_lookup = self.group_lookup
        captured_group = None
        reward = 0
        reset = 0
        # if invalid stone, reject, reset 1, and negative reward
        if self.is_invalid(action, perspective):
            reward = -max_value
            reset = 1
        elif mutable:
            self.n_turns += 1
            self.perspective *= -1
            # end of game when both players pass
            if action:
                self.turn_pass = False
                board, group_lookup, captured_group = place_stone(
                    action, perspective, board, group_lookup, self)
            elif self.turn_pass:
                reset = self.n_turns
                self.reset()
            else:
                self.turn_pass = True
        else:
            if action:
                board, group_lookup, captured_group = place_stone(
                    action, perspective, board, group_lookup)
            elif self.turn_pass:
                reset = self.n_turns + 1
        liberty_map = generate_liberty_map(board, group_lookup)
        territory_map = generate_territory_map(board)
        state = State(self, perspective, liberty_map, territory_map)
        if mutable:
            self.state = State(self, self.perspective, liberty_map,
                               territory_map)
        if reset > 1:
            reward = np.sum(territory_map) * perspective
        elif reset == 0 and captured_group:
            reward = len(captured_group.stones)
        return state, reward, reset

    def is_suicide(self, stone, perspective):
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

    def is_invalid(self, stone, perspective):
        # passing is valid
        if not stone:
            return False
        # out of bounds, already occupied, ko, or suicidal moves
        if (stone[0] < 0 or stone[0] > board_size[0] or stone[1] < 0
                or stone[1] > board_size[1] or self.board[stone] != 0
                or stone == self.ko or self.is_suicide(stone, perspective)):
            return True
        else:
            return False

    def print_status(self):
        print_groups(self.group_lookup)
        print(ascii_board(self.board))
        print("turn", self.n_turns)
        if self.perspective > 0:
            print("X move")
        else:
            print("O move")
        if self.turn_pass:
            print("previous player passed")


#------------ The below is required game interface for betazero


#warning, this only works with latest state, due to optomizations
#that required current session state
def get_actions(state, include_pass=True):
    session = state.session
    # take all empty spots minus the suicidal ones
    # can't take back a ko
    actions = [
        spot for spot in session.empty_spots if
        not session.is_suicide(spot, state.perspective) and spot != session.ko
    ]
    # action None is to pass
    if include_pass or not actions:
        actions.append(None)
    return actions


def predict_action(state, action):
    return state.session.do_action(action, mutable=False)


class State:
    def __init__(self, session, perspective, liberty_map, territory_map):
        self.session = session
        self.perspective = perspective
        self.liberty_map = liberty_map
        self.territory_map = territory_map

    def flip(self):
        return State(self.session, -self.perspective, self.liberty_map,
                     self.perspective)

    def array(self):
        return np.array([
            self.liberty_map * self.perspective,
            self.territory_map * self.perspective
        ])[np.newaxis]

    def key(self):
        return self.liberty_map.tobytes()

    def __str__(self):
        return str(ascii_board(self.liberty_map))


def get_human_action():
    return parse_grid_input(board_size)
